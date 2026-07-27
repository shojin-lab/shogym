"""Evaluate Claude Code on a served Frontier-Bench episode through hgym (RFC 008 §7, issue #44).

The **external-harness** path: hgym does not drive Claude Code — Claude Code spawns
``hgym serve frontier_bench`` as its MCP server (per the generated ``.mcp.json``), operates the
task's Docker container through the ``exec`` / ``read_file`` / ``write_file`` tools, and calls
``done`` — the ``score`` terminal — which seals the episode and runs the task's own verifier over
the container end-state, writing the JSONL trace. When it finishes, we read the terminal feedback
with :func:`hgym.result_from_trace`.

Completion (from the task instructions the harness reads via ``describe``): the agent reads the
inputs under ``/app/inputs/`` and writes the required outputs under ``/app`` with ``exec`` /
``write_file``, then calls ``done`` — this seals **and** ends the episode (the verifier runs over
the container's final state and its 0/1 reward is the terminal result). ``done`` is one-shot; no
further tool calls run after it.

Run it::

    # Needs a working Docker daemon (the env builds+runs the task's containers) and Claude
    # credentials (the harness itself makes model calls):
    uv run python examples/frontier_bench/claude_code/run.py --task 0
    uv run python examples/frontier_bench/claude_code/run.py --task 0 --transcript

Nothing about hgym is Claude-specific — swap ``build_claude_command`` for a Codex / pi / Hermes
invocation pointed at the same ``.mcp.json`` and the trace/score path is identical.
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

ENV_NAME = "frontier_bench"

# Latest Sonnet (the alias ``sonnet`` also resolves to it); override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# Reasoning effort: medium by default — the tasks are hard, but keep the example affordable.
DEFAULT_EFFORT = "medium"

# The key under `mcpServers` in the generated .mcp.json. Claude Code namespaces this server's
# tools as ``mcp__<SERVER_KEY>__<tool>``.
SERVER_KEY = "frontier_bench"

# Allow every tool the served frontier_bench env exposes (exec/read_file/write_file/done +
# describe + terminate) without enumerating them. Claude Code's allow-rule grammar requires a
# glob-free ``mcp__<server>__`` prefix, so the correct "all tools from this server" form is the
# ``__*`` suffix. We deliberately do NOT pass ``--tools ""`` — in current Claude Code that
# strips the MCP tools too, leaving the agent with no tools.
ALLOWED_TOOLS = f"mcp__{SERVER_KEY}__*"

# Deny the built-in tools so the agent can't take untraced side-channel actions (shell out on
# the host, read/write host files, fetch the web) that never appear in the hgym trace, keeping
# the score attributable to the MCP tool surface alone. The agent's *only* shell is the served
# `exec` tool, which runs inside the task container.
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
    "You are solving a Frontier-Bench task inside its container, through the "
    f"`{SERVER_KEY}` MCP tools. First call `describe` to read the full task instruction (the "
    "inputs under /app/inputs/, the exact outputs to produce, and their format). Then operate "
    "the container with `exec` (run shell commands), `read_file`, and `write_file` — read the "
    "inputs and write the required outputs under /app. When the outputs are ready, call `done` "
    "once — it seals and ends the episode, runs the task's verifier over the container, and "
    "returns the 0/1 reward. `done` is one-shot; make sure the outputs are complete before "
    "calling it, and do not call any tool after it."
)


def _assert_docker_available() -> None:
    """The served env builds+runs Docker containers; a missing daemon makes ``hgym serve
    frontier_bench`` crash on the first tool call (Claude then sees a broken server). Fail loud
    here with the fix instead."""
    from hgym.envs.frontier_bench import docker_backend as dk

    if not dk.docker_available():
        raise SystemExit(
            "[frontier_bench example] Docker daemon not reachable (`docker info` failed).\n"
            "Frontier-Bench tasks are Docker-backed — start Docker Desktop / the daemon and "
            "retry."
        )


def _assert_env_constructible() -> Any:
    """Confirm the env constructs + its MCP tool schemas probe (the same construction the served
    process does), turning any residual failure into an actionable message."""
    try:
        return hgym.make(ENV_NAME)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"[frontier_bench example] constructing {ENV_NAME!r} failed:\n"
            f"    {type(exc).__name__}: {exc}\n"
        )


def _assert_task_image_builds(task: str) -> None:
    """Build the selected task's environment image up front — the actual ``docker build`` the
    served ``begin_session`` runs. Constructing the env only probes tool schemas; a Dockerfile /
    base-image-pull / platform failure would otherwise not surface until the served server
    crashes on the first tool call. Building here fails loud with the fix, and serving reuses
    the content-addressed image from cache."""
    from hgym.envs.frontier_bench import manifest, mcp_server

    try:
        mcp_server.build_task_image(manifest.resolve_name(task))
    except Exception as exc:
        raise SystemExit(
            f"[frontier_bench example] building the task's environment image failed:\n"
            f"    {type(exc).__name__}: {exc}\n"
            "Check that Docker can build the task's environment/Dockerfile (base-image pull, "
            "platform)."
        )


def preflight(task: str) -> None:
    """Verify Docker is up, the env constructs, and the task image builds before spawning Claude
    (fail loud, with a fix) — so Claude never connects to a server that dies at startup."""
    _assert_docker_available()
    _assert_env_constructible()
    _assert_task_image_builds(task)


def write_mcp_config(config_path: Path, task: str, trace_path: Path) -> Path:
    """Write a ``.mcp.json`` that makes Claude Code spawn ``hgym serve frontier_bench``.

    Invoke the CLI through the *current* interpreter (``python -m hgym.cli``) rather than a bare
    ``hgym`` on ``PATH`` — so the example works whether or not the venv that has hgym installed
    is the one on the spawned subprocess's ``PATH``."""
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
    attributable solely to this ``hgym serve`` process), the frontier_bench tools pre-allowed,
    the built-ins denied, and ``--permission-mode dontAsk`` so tool calls run
    non-interactively. ``transcript=True`` emits the turn-by-turn stream."""
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
                print(f"🔧 {block.get('name')}({json.dumps(block.get('input', {}))[:300]})")
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
    """Run Claude Code on one frontier_bench task and read the terminal feedback off the trace."""
    # Accept a task name OR an index here, but resolve to the canonical integer index before
    # spawning `hgym serve`: the shared serve path (ServedEpisode) casts `--task` with `int(...)`,
    # so a bare name would raise there. Resolving here lets the example take names while the
    # served subprocess always receives a valid index.
    from hgym.envs.frontier_bench import manifest

    task = str(manifest.load_task(task).index)
    # Fail loud (with a fix) if Docker is down or the interpreter can't serve the env —
    # otherwise Claude Code connects to a crashed server, sees no tools, and gives up opaquely.
    preflight(task)
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hgym-claude-frontier-"))
    trace_path = workdir / f"{ENV_NAME}.jsonl"
    mcp_config = write_mcp_config(workdir / ".mcp.json", task, trace_path)

    cmd = build_claude_command(mcp_config, model, effort, transcript=transcript)
    if transcript:
        _run_with_transcript(cmd)
    else:
        subprocess.run(cmd, check=True)

    if not trace_path.exists():
        # No tool call was served (e.g. Claude answered with text only or declined after MCP
        # startup), so no trace file exists: report an unterminated run rather than raising.
        return hgym.EvalResult(
            env=ENV_NAME, task=task, terminated=False, trace_path=str(trace_path)
        )
    return hgym.result_from_trace(trace_path, env=ENV_NAME, task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="0", help="task index (0..N-1) or name (default 0 = fin-saccr-rwa)"
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
        f"reward={result.value('reward')} success={result.value('success')} "
        f"verified={result.value('verified')}"
    )
    print(f"feedback: {result.feedback}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    main()
