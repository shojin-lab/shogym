"""Evaluate Claude Code on a served AutomationBench episode through hgym (RFC 008 §7, issue #42).

The **external-harness** path: hgym does not drive Claude Code — Claude Code spawns
``hgym serve automationbench`` as its MCP server (per the generated ``.mcp.json``), plays the
episode by discovering endpoints with ``api_search`` and mutating the simulated workspace with
``api_fetch``, and writes the JSONL trace. When it finishes, we read the terminal feedback with
:func:`hgym.result_from_trace`.

Completion (from the task instructions the harness reads via ``describe``): the agent carries out
the requested cross-application workflow with the ``api`` toolset, then calls ``done``. ``done`` is
the ``score`` terminal — calling it seals the episode and scores the final state in one step (no
separate ``terminate``). hgym reruns AutomationBench's own rubric (``partial_credit`` /
``task_completed_correctly``) over the sealed end-state.

Run it::

    # Fully offline sim (no OpenAI/Zapier key — the ~47-app world runs in-process; the Claude
    # harness itself still makes model calls / needs Claude credentials):
    uv run python examples/automationbench/claude_code/run.py --task 0
    uv run python examples/automationbench/claude_code/run.py --task 0 --transcript

The first run provisions the pinned upstream source into ``~/.cache/hgym`` (a one-time fetch);
after that it is fully offline. Nothing about hgym is Claude-specific — swap
``build_claude_command`` for a Codex / pi / Hermes invocation pointed at the same ``.mcp.json``
and the trace/score path is identical.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import hgym

ENV_NAME = "automationbench"

# Latest Sonnet (the alias ``sonnet`` also resolves to it); override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# Reasoning effort: low by default to keep the example cheap/fast; override with --effort.
DEFAULT_EFFORT = "low"

# The key under `mcpServers` in the generated .mcp.json. Claude Code namespaces this server's
# tools as ``mcp__<SERVER_KEY>__<tool>``.
SERVER_KEY = "automationbench"

# Allow every tool the served env exposes (api_search / api_fetch / base64_encode / done +
# describe + terminate) without enumerating them. Claude Code's allow-rule grammar requires a
# glob-free ``mcp__<server>__`` prefix, so the "all tools from this server" form is the ``__*``
# suffix. We deliberately do NOT pass ``--tools ""`` — in current Claude Code that strips the MCP
# tools too, leaving the agent with no tools.
ALLOWED_TOOLS = f"mcp__{SERVER_KEY}__*"

# Deny the built-in tools so the agent can't take untraced side-channel actions (shell out, read
# local files, fetch the web) that never appear in the hgym trace — keeping the score
# attributable to the MCP tool surface alone.
DISALLOWED_TOOLS = ",".join(
    [
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "WebFetch",
        "WebSearch",
        "Task",
        "TodoWrite",
        "NotebookEdit",
        "BashOutput",
        "KillShell",
    ]
)

PROMPT = (
    "You are a workflow automation agent, played through the "
    f"`{SERVER_KEY}` MCP tools. First call `describe` to read the task and your tools. Then "
    "carry out the requested workflow over the simulated SaaS workspace: use `api_search` to "
    "find the right endpoint, then `api_fetch` to read and change state (use `base64_encode` "
    "for Gmail bodies). When the workflow is complete, call `done` — that seals the episode and "
    "scores the final state in one step (there is no second submission and no need to call "
    "`terminate` afterward)."
)


def _assert_server_can_import_automationbench() -> None:
    """The MCP server runs under this same interpreter (``write_mcp_config`` uses
    ``sys.executable``). It needs ``datasets`` (the domain task loaders) importable; if it isn't,
    ``hgym serve automationbench`` crashes at startup and Claude Code connects to a toolless
    server and gives up — an opaque failure. Fail loud here with the fix instead."""
    try:
        importlib.import_module("datasets")
    except ImportError as exc:
        raise SystemExit(
            f"[automationbench example] the server interpreter cannot import datasets:\n"
            f"    {sys.executable}\n"
            f"    ({exc})\n"
            f"Sync the dev environment (this repo pins Python 3.12 and installs the "
            f"automationbench extra there):\n"
            f"    uv sync\n"
            f"then run this via `uv run python …`."
        )


def _assert_env_constructible() -> Any:
    """Provision the pinned upstream source (a one-time fetch on a cold cache) and confirm the env
    actually builds — the same path the served process takes — turning any residual failure into
    an actionable message rather than a served-server crash. Returns the constructed env."""
    try:
        from hgym.envs.automationbench import adapter

        adapter.ensure_source()
        return hgym.make(ENV_NAME, config={"domain": "simple"})
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"[automationbench example] building {ENV_NAME!r} failed:\n"
            f"    {type(exc).__name__}: {exc}\n"
            f"The first run fetches the pinned upstream source into ~/.cache/hgym — check your "
            f"network, or set AUTOMATIONBENCH_SRC to a local checkout. Try `uv sync` too."
        )


def preflight() -> None:
    """Verify the interpreter can serve the env before spawning Claude (fail loud, with a fix).
    On a cold cache this performs the one-time upstream-source fetch."""
    _assert_server_can_import_automationbench()
    _assert_env_constructible()


def write_mcp_config(config_path: Path, task: str, trace_path: Path) -> Path:
    """Write a ``.mcp.json`` that makes Claude Code spawn ``hgym serve automationbench``.

    Invoke the CLI through the *current* interpreter (``python -m hgym.cli``) rather than a bare
    ``hgym`` on ``PATH`` — so the example works whether or not the venv that has hgym (with the
    ``automationbench`` extra) installed is the one on the spawned subprocess's ``PATH``."""
    config = {
        "mcpServers": {
            SERVER_KEY: {
                "command": sys.executable,
                "args": [
                    "-m",
                    "hgym.cli",
                    "serve",
                    ENV_NAME,
                    "--task",
                    task,
                    "--trace",
                    str(trace_path),
                ],
            }
        }
    }
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


def build_claude_command(
    mcp_config: Path,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    *,
    transcript: bool = False,
) -> List[str]:
    """The ``claude`` invocation: print mode, the model, reasoning effort, our MCP config
    (``--strict-mcp-config`` keeps inherited user/project servers out, so the trace is
    attributable solely to this ``hgym serve`` process), the env's tools pre-allowed, the
    built-ins denied, and ``--permission-mode dontAsk`` so tool calls run non-interactively."""
    cmd = [
        "claude",
        "-p",
        PROMPT,
        "--model",
        model,
        "--effort",
        effort,
        "--mcp-config",
        str(mcp_config),
        "--strict-mcp-config",
        "--allowedTools",
        ALLOWED_TOOLS,
        "--disallowedTools",
        DISALLOWED_TOOLS,
        "--permission-mode",
        "dontAsk",
    ]
    if transcript:
        cmd += ["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
    return cmd


def _tool_result_text(block: Dict[str, Any]) -> str:
    content = block.get("content")
    if isinstance(content, list):
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict)).strip()
    return str(content)


def _print_event(event: Dict[str, Any]) -> None:
    """Render one stream-json event as a readable transcript line."""
    etype = event.get("type")
    if etype == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text", "").strip():
                print(f"\n🧠 {block['text'].strip()}")
            elif block.get("type") == "tool_use":
                print(f"🔧 {block.get('name')}({json.dumps(block.get('input', {}))})")
    elif etype == "user":
        for block in event.get("message", {}).get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                print(f"   ↳ {_tool_result_text(block)[:500]}")
    elif etype == "result" and event.get("result"):
        print(f"\n✅ {event['result']}")


def _run_with_transcript(cmd: List[str]) -> None:
    """Run ``claude`` in stream-json mode, printing a readable transcript as it goes."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            _print_event(json.loads(line))
        except json.JSONDecodeError:
            continue
    if proc.wait() != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def evaluate_claude(
    task: str,
    *,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    transcript: bool = False,
    workdir: Optional[Path] = None,
) -> hgym.EvalResult:
    """Run Claude Code on one automationbench task and read the terminal feedback off the trace."""
    # Fail loud (with a fix) if the server interpreter can't serve the env — otherwise Claude
    # Code connects to a crashed server, sees no tools, and gives up opaquely.
    preflight()
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hgym-claude-ab-"))
    trace_path = workdir / f"{ENV_NAME}.jsonl"
    mcp_config = write_mcp_config(workdir / ".mcp.json", task, trace_path)

    cmd = build_claude_command(mcp_config, model, effort, transcript=transcript)
    if transcript:
        _run_with_transcript(cmd)
    else:
        subprocess.run(cmd, check=True)

    if not trace_path.exists():
        # No tool call was served (e.g. Claude answered with text only or declined after MCP
        # startup), so no trace file exists: report an unterminated run rather than letting
        # load_traces raise FileNotFoundError.
        return hgym.EvalResult(
            env=ENV_NAME, task=task, terminated=False, trace_path=str(trace_path)
        )
    return hgym.result_from_trace(trace_path, env=ENV_NAME, task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="0", help="task index into the default `public` domain set"
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Claude Code model id/alias (default: {DEFAULT_MODEL}; e.g. sonnet, opus)",
    )
    parser.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        help=f"reasoning effort (default: {DEFAULT_EFFORT}; low|medium|high|xhigh|max)",
    )
    parser.add_argument(
        "--transcript",
        action="store_true",
        help="print Claude Code's turn-by-turn tool calls and reasoning",
    )
    args = parser.parse_args()

    result = evaluate_claude(
        args.task,
        model=args.model,
        effort=args.effort,
        transcript=args.transcript,
    )
    print(
        f"env={result.env} task={result.task} terminated={result.terminated} "
        f"reward={result.value('reward')} partial_credit={result.value('partial_credit')} "
        f"success={result.value('success')}"
    )
    print(f"feedback: {result.feedback}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    main()
