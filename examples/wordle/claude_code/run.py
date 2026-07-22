"""Evaluate Claude Code on Wordle through a served hgym env (RFC 008 §7).

The **external-harness** path: hgym does not drive Claude Code — Claude Code spawns
``hgym serve`` as its MCP server (per the generated ``.mcp.json``), plays the episode using
the ``guess``/``terminate`` tools, and writes the JSONL trace. When it finishes, we read
the terminal feedback with :func:`hgym.result_from_trace`.

Run it (needs the ``claude`` CLI on PATH and credentials configured for it)::

    python examples/wordle/claude_code/run.py --task 0
    python examples/wordle/claude_code/run.py --task 0 --model opus --effort high --transcript

Nothing about hgym is Claude-specific — swap ``build_claude_command`` for a Codex / pi /
Hermes invocation pointed at the same ``.mcp.json`` and the trace/score path is identical.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import hgym

# Latest Sonnet (the alias ``sonnet`` also resolves to it); override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# Reasoning effort: lowest by default — Wordle needs little deliberation, and it keeps the
# example cheap/fast. Override with --effort (low|medium|high|xhigh|max).
DEFAULT_EFFORT = "low"

PROMPT = (
    "You are playing Wordle through the `wordle` MCP tools. First call `describe` to "
    "read the rules, then call `guess` with 5-letter words, using each result's "
    "feedback (G/Y/X) to inform the next guess. Call `terminate` as soon as you solve "
    "it or run out of guesses."
)

# Claude Code namespaces MCP tools as mcp__<server>__<tool>.
ALLOWED_TOOLS = "mcp__wordle__describe,mcp__wordle__guess,mcp__wordle__terminate"


def write_mcp_config(config_path: Path, task: str, trace_path: Path) -> Path:
    """Write a ``.mcp.json`` that makes Claude Code spawn ``hgym serve`` for this task.

    Invoke the CLI through the *current* interpreter (``python -m hgym.cli``) rather than a
    bare ``hgym`` on ``PATH`` — so the example works whether or not the venv that has hgym
    installed is the one on the spawned subprocess's ``PATH``."""
    config = {
        "mcpServers": {
            "wordle": {
                "command": sys.executable,
                "args": [
                    "-m",
                    "hgym.cli",
                    "serve",
                    "wordle_v1",
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
    (``--strict-mcp-config`` keeps out inherited servers), built-in tools disabled
    (``--tools ""``) so only our three MCP tools are reachable and the run stays fully
    trace-attributable, those three pre-allowed. ``transcript=True`` emits the
    turn-by-turn event stream."""
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
        "--tools",
        "",
        "--allowedTools",
        ALLOWED_TOOLS,
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
    """Run Claude Code on one Wordle task and read the terminal feedback off the trace."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hgym-claude-"))
    trace_path = workdir / "wordle.jsonl"
    mcp_config = write_mcp_config(workdir / ".mcp.json", task, trace_path)

    cmd = build_claude_command(mcp_config, model, effort, transcript=transcript)
    if transcript:
        _run_with_transcript(cmd)
    else:
        subprocess.run(cmd, check=True)

    if not trace_path.exists():
        # No tool call was served (e.g. Claude answered with text only or declined
        # after MCP startup), so no trace file exists: report an unterminated run
        # rather than letting load_traces raise FileNotFoundError.
        return hgym.EvalResult(env="wordle_v1", task=task, terminated=False, trace_path=str(trace_path))
    return hgym.result_from_trace(trace_path, env="wordle_v1", task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="0", help="wordle task index")
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
    solved = result.value("check_answer")
    print(f"env={result.env} task={result.task} terminated={result.terminated} solved={solved}")
    print(f"feedback: {result.feedback}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    main()
