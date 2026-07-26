"""Evaluate Claude Code on a served Humanity's Last Exam (HLE) env through hgym (issue #33).

The **external-harness** path: hgym does not drive Claude Code — Claude Code spawns
``hgym serve hle`` as its MCP server (per the generated ``.mcp.json``), reads the question
via ``describe`` and calls ``submit_answer`` (the score terminal: it seals the episode, grades
it **server-side** with the LLM judge, and ends the episode in one step — no separate
``terminate``). When it finishes, we read the terminal feedback (``correct`` +
``calibration_error``) with :func:`hgym.result_from_trace`.

Run it::

    # Needs the `hle` extra + an OpenAI key for the judge, and Hugging Face access to the
    # gated cais/hle dataset (accept its terms + `huggingface-cli login`). The preflight
    # fails fast with the exact fix if any piece is missing.
    export OPENAI_API_KEY=sk-...
    uv run python examples/hle/claude_code/run.py --task 0
    uv run python examples/hle/claude_code/run.py --task 0 --transcript

Nothing about hgym is Claude-specific — swap ``build_claude_command`` for a Codex / pi /
Hermes invocation pointed at the same ``.mcp.json`` and the trace/score path is identical.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import hgym

ENV_NAME = "hle"

# Latest Sonnet (the alias ``sonnet`` also resolves to it); override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# HLE questions are frontier-difficulty; give the solver real room to reason by default.
DEFAULT_EFFORT = "high"

# The key under `mcpServers` in the generated .mcp.json. Claude Code namespaces this server's
# tools as ``mcp__<SERVER_KEY>__<tool>``.
SERVER_KEY = "hle"

# Allow every tool the served hle env exposes (submit_answer + describe + terminate) without
# enumerating them. Claude Code's allow-rule grammar requires a glob-free ``mcp__<server>__``
# prefix, so the "all tools from this server" form is the ``__*`` suffix (a bare ``mcp__hle``
# is rejected). We do NOT use ``--tools ""`` — in current Claude Code that strips the *MCP*
# tools too, leaving the agent with no submit_answer tool at all.
ALLOWED_TOOLS = f"mcp__{SERVER_KEY}__*"

# Deny the built-ins so the agent can't take untraced side-channel actions — above all
# WebFetch / WebSearch, since HLE must measure the model's own reasoning, not its ability to
# look the answer up. `--permission-mode dontAsk` denies anything not pre-allowed, but its
# read-only exemption would still permit Read/Grep/WebFetch; denying them explicitly closes that.
DISALLOWED_TOOLS = ",".join(
    [
        "WebFetch",
        "WebSearch",
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Task",
        "TodoWrite",
        "NotebookEdit",
        "BashOutput",
        "KillShell",
    ]
)

PROMPT = (
    "You are answering one question from Humanity's Last Exam, played through the "
    f"`{SERVER_KEY}` MCP tools. First call `describe` to read the question. Reason "
    "carefully from your own knowledge — do not look anything up. Then call "
    "`submit_answer` exactly once with your final `answer` and a `confidence` from 0 to "
    "100. That call grades your answer and ends the episode — do not call `terminate` "
    "afterward."
)


def _assert_server_can_import_extra() -> None:
    """The MCP server runs under this same interpreter (``write_mcp_config`` uses
    ``sys.executable``). If it can't import the ``hle`` extra, ``hgym serve hle`` will crash
    at startup and Claude Code will connect to a toolless server and give up — an opaque
    failure. Fail fast here with the fix instead."""
    for module in ("datasets", "openai"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise SystemExit(
                f"[hle example] the server interpreter cannot import {module!r}:\n"
                f"    {sys.executable}\n"
                f"    ({exc})\n"
                f"Install the hle extra and run via `uv run python …`:\n"
                f"    uv sync   # includes hgym[hle] in the dev group\n"
            )


def _assert_judge_key() -> None:
    """The env grades ``submit_answer`` server-side with an OpenAI LLM judge. Without a key the
    run would grade every answer as incorrect (the judge errors), so check up front."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "[hle example] the HLE judge needs OPENAI_API_KEY (the model-graded verifier "
            "calls an OpenAI model).\n    export OPENAI_API_KEY=sk-..."
        )


def _assert_env_constructible() -> None:
    """Confirm the env actually builds (the same probe the served process does) — which loads
    the gated cais/hle dataset — turning any residual failure into an actionable message
    rather than a served-server crash."""
    try:
        hgym.make(ENV_NAME)
    except SystemExit:
        raise
    except Exception as exc:
        from hgym.envs.hle.data import is_gating_error

        msg = (
            f"[hle example] building {ENV_NAME!r} failed:\n"
            f"    {type(exc).__name__}: {exc}"
        )
        # Only add the gating hint when the failure is *actually* a gating/auth one — otherwise
        # (a missing dep, a network error, a corrupt cache) surface the real error unadorned.
        if is_gating_error(exc):
            msg += (
                "\ncais/hle is gated on the Hugging Face Hub — accept its terms at "
                "https://huggingface.co/datasets/cais/hle and authenticate "
                "(`huggingface-cli login`, or set HF_TOKEN)."
            )
        raise SystemExit(msg)


def preflight() -> None:
    """Verify the interpreter can serve hle (extra installed, judge key present, env builds)
    before spawning Claude — so a missing piece is a clear message, not a crashed server."""
    _assert_server_can_import_extra()
    _assert_judge_key()
    _assert_env_constructible()


def write_mcp_config(config_path: Path, task: str, trace_path: Path) -> Path:
    """Write a ``.mcp.json`` that makes Claude Code spawn ``hgym serve hle``.

    Invoke the CLI through the *current* interpreter (``python -m hgym.cli``) rather than a
    bare ``hgym`` on ``PATH`` — so the example works whether or not the venv that has hgym
    (with the ``hle`` extra) installed is the one on the spawned subprocess's ``PATH``."""
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
    (``--strict-mcp-config`` keeps inherited servers out), the hle tools pre-allowed, the
    built-ins (crucially WebFetch/WebSearch) denied, and ``--permission-mode dontAsk`` so tool
    calls run non-interactively. We deliberately do **not** pass ``--tools ""`` — in current
    Claude Code that removes the MCP tools too. ``transcript=True`` emits the turn-by-turn stream."""
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
                print(f"   ↳ {_tool_result_text(block)}")
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
    """Run Claude Code on one HLE task and read the terminal feedback off the trace."""
    # Fail fast (with a fix) if the server interpreter can't serve hle — otherwise Claude Code
    # connects to a crashed server, sees no tools, and gives up opaquely.
    preflight()
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hgym-claude-hle-"))
    trace_path = workdir / f"{ENV_NAME}.jsonl"
    mcp_config = write_mcp_config(workdir / ".mcp.json", task, trace_path)

    cmd = build_claude_command(mcp_config, model, effort, transcript=transcript)
    if transcript:
        _run_with_transcript(cmd)
    else:
        subprocess.run(cmd, check=True)

    if not trace_path.exists():
        # No tool call was served (e.g. Claude answered with text only or declined after
        # MCP startup), so no trace file exists: report an unterminated run rather than
        # letting load_traces raise FileNotFoundError.
        return hgym.EvalResult(
            env=ENV_NAME, task=task, terminated=False, trace_path=str(trace_path)
        )
    return hgym.result_from_trace(trace_path, env=ENV_NAME, task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="0", help="hle task index within the split")
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
        args.task, model=args.model, effort=args.effort, transcript=args.transcript
    )
    print(
        f"env={result.env} task={result.task} terminated={result.terminated} "
        f"correct={result.value('correct')} "
        f"calibration_error={result.value('calibration_error')}"
    )
    print(f"feedback: {result.feedback}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    main()
