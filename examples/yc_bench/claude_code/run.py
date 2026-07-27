"""Evaluate Claude Code on a served YC-Bench episode through hgym (RFC 008 §7, issue #32).

The **external-harness** path: hgym does not drive Claude Code — Claude Code spawns
``hgym serve yc_bench`` as its MCP server (per the generated ``.mcp.json``), plays the episode
by issuing YC-Bench CLI commands through the ``run_command`` tool, and writes the JSONL trace.
When it finishes, we read the terminal feedback with :func:`hgym.result_from_trace`.

Completion (from the task instructions the harness reads via ``describe``): the agent runs the
one-year simulation with ``run_command`` (accept/assign/dispatch tasks, ``sim resume`` to
advance the clock), then calls ``submit`` — the env's ``score`` terminal, which seals the
episode, reads the final funds / survival / task outcomes off the sim, scores it, and ends the
run in one step (``terminate`` remains available only as the no-score abort).

Run it::

    # Fully offline sim (no OpenAI/YC-Bench API key — the deterministic sim runs in-process;
    # the Claude harness itself still makes model calls / needs Claude credentials):
    uv run python examples/yc_bench/claude_code/run.py --task 0
    uv run python examples/yc_bench/claude_code/run.py --task 0 --transcript

Nothing about hgym is Claude-specific — swap ``build_claude_command`` for a Codex / pi /
Hermes invocation pointed at the same ``.mcp.json`` and the trace/score path is identical.
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

ENV_NAME = "yc_bench"

# Latest Sonnet (the alias ``sonnet`` also resolves to it); override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# Reasoning effort: low by default to keep the example cheap/fast; override with --effort.
DEFAULT_EFFORT = "low"

# The key under `mcpServers` in the generated .mcp.json. Claude Code namespaces this server's
# tools as ``mcp__<SERVER_KEY>__<tool>``.
SERVER_KEY = "yc_bench"

# Allow every tool the served yc_bench env exposes (run_command + submit + describe +
# terminate) without enumerating them. Claude Code's allow-rule grammar requires a glob-free
# ``mcp__<server>__`` prefix, so the correct "all tools from this server" form is the ``__*``
# suffix (a bare ``mcp__yc_bench`` is rejected). We deliberately do NOT pass ``--tools ""`` —
# in current Claude Code that strips the MCP tools too, leaving the agent with no tools.
ALLOWED_TOOLS = f"mcp__{SERVER_KEY}__*"

# Deny the built-in tools so the agent can't take untraced side-channel actions (shell out to
# run `yc-bench` directly, read the sim DB, fetch the web) that never appear in the hgym trace,
# keeping the score attributable to the MCP tool surface alone. `--permission-mode dontAsk`
# already denies everything not in `--allowedTools`, but its read-only exemption would still
# permit Read/Grep; denying them explicitly (and WebFetch/WebSearch) closes that.
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
    "You are the CEO in a YC-Bench startup simulation, played through the "
    f"`{SERVER_KEY}` MCP tools. First call `describe` to read the rules and your run. Then "
    "operate the company with `run_command` (pass full `yc-bench …` command strings): browse "
    "the market, accept/assign/dispatch tasks, and call `yc-bench sim resume` to advance the "
    "clock — repeat until the run ends (bankruptcy or the one-year horizon). When the run is "
    "over, call `submit` — it ends the episode and records your final result in one step."
)


def _assert_server_can_import_yc_bench() -> None:
    """The MCP server runs under this same interpreter (``write_mcp_config`` uses
    ``sys.executable``). If it can't ``import yc_bench``, ``hgym serve yc_bench`` will crash at
    startup and Claude Code will connect to a server with **no tools** and give up — an opaque
    failure. Fail loud here with the fix instead."""
    try:
        importlib.import_module("yc_bench")
    except ImportError as exc:
        raise SystemExit(
            f"[yc_bench example] the server interpreter cannot import yc_bench:\n"
            f"    {sys.executable}\n"
            f"    ({exc})\n"
            f"Sync the dev environment (this repo pins Python 3.12 and installs the yc_bench "
            f"extra there):\n"
            f"    uv sync\n"
            f"then run this via `uv run python …`."
        )


def _assert_env_constructible() -> Any:
    """yc_bench imports — confirm the env actually builds (the same probe the served process
    does), turning any residual failure into an actionable message rather than a served-server
    crash. Returns the constructed env."""
    try:
        return hgym.make(ENV_NAME)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"[yc_bench example] yc_bench is installed but building {ENV_NAME!r} failed:\n"
            f"    {type(exc).__name__}: {exc}\n"
            f"Try `uv sync` to (re)install the yc_bench extra."
        )


def preflight() -> None:
    """Verify the interpreter can serve the env before spawning Claude (fail loud, with a
    fix). YC-Bench needs no external data — its world is generated deterministically from the
    seed — so there is nothing to download."""
    _assert_server_can_import_yc_bench()
    _assert_env_constructible()


def write_mcp_config(config_path: Path, task: str, trace_path: Path) -> Path:
    """Write a ``.mcp.json`` that makes Claude Code spawn ``hgym serve yc_bench``.

    Invoke the CLI through the *current* interpreter (``python -m hgym.cli``) rather than a
    bare ``hgym`` on ``PATH`` — so the example works whether or not the venv that has hgym
    (with the ``yc_bench`` extra) installed is the one on the spawned subprocess's ``PATH``."""
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
    attributable solely to this ``hgym serve`` process), the yc_bench tools pre-allowed, the
    built-ins denied, and ``--permission-mode dontAsk`` so tool calls run non-interactively.

    Note: we deliberately do **not** pass ``--tools ""`` — in current Claude Code that removes
    the MCP tools too, leaving the agent with an empty toolset. ``transcript=True`` emits the
    turn-by-turn stream."""
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
    """Run Claude Code on one yc_bench task and read the terminal feedback off the trace."""
    # Fail loud (with a fix) if the server interpreter can't serve the env — otherwise Claude
    # Code connects to a crashed server, sees no tools, and gives up opaquely.
    preflight()
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hgym-claude-yc-"))
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
    parser.add_argument("--task", default="0", help="task index (selects the world seed)")
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
        f"reward={result.value('reward')} survived={result.value('survived')} "
        f"success={result.value('success')}"
    )
    print(f"feedback: {result.feedback}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    main()
