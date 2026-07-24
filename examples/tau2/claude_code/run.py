"""Evaluate Claude Code on a served tau2-bench env through hgym (RFC 008 §7, issue #31).

The **external-harness** path: hgym does not drive Claude Code — Claude Code spawns
``hgym serve tau2_<domain>`` as its MCP server (per the generated ``.mcp.json``), plays the
episode using the tau2 domain tools, and writes the JSONL trace. When it finishes, we read
the terminal feedback with :func:`hgym.result_from_trace`.

Two-step completion (from the task instructions the harness reads via ``describe``): the
agent signals task completion with ``done`` (its result reports tau2's evaluator verdict),
then ends the episode with ``terminate``.

Run it::

    # The mock domain is solo (tau2 DummyUser) — no OpenAI key and no tau2 user-simulator
    # cost (the Claude harness itself still makes model calls / needs Claude credentials):
    uv run python examples/tau2/claude_code/run.py --task 0
    uv run python examples/tau2/claude_code/run.py --task 0 --transcript

    # A real domain — additionally needs OPENAI_API_KEY for tau2's user simulator (real cost):
    uv run python examples/tau2/claude_code/run.py --domain telecom --task 0

Nothing about hgym is Claude-specific — swap ``build_claude_command`` for a Codex / pi /
Hermes invocation pointed at the same ``.mcp.json`` and the trace/score path is identical.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import hgym

# tau2-bench does not ship its `data/` in the install, so it must be provisioned at runtime.
# We lazy-clone the pinned revision to a user cache (once) unless TAU2_DATA_DIR is already set.
TAU2_REPO = "https://github.com/sierra-research/tau2-bench"
TAU2_SHA = "1d244f5dca42944b67a379b44bfeb9f5748f189d"

# Latest Sonnet (the alias ``sonnet`` also resolves to it); override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# Reasoning effort: low by default to keep the example cheap/fast; override with --effort.
DEFAULT_EFFORT = "low"

# Default to the mock domain: solo (tau2 DummyUser), so the tau2 side needs no OpenAI key and
# incurs no user-simulator cost — genuinely runnable out of the box. (The Claude harness still
# makes model calls and needs Claude credentials, as any `claude -p` run does.)
DEFAULT_DOMAIN = "mock"

# The key under `mcpServers` in the generated .mcp.json. Claude Code namespaces this server's
# tools as ``mcp__<SERVER_KEY>__<tool>``.
SERVER_KEY = "tau2"

# Allow every tool the served tau2 env exposes (domain tools + describe + done + terminate,
# plus send_message on non-solo domains) without enumerating per domain. Claude Code's
# allow-rule grammar requires a glob-free ``mcp__<server>__`` prefix, so the correct
# "all tools from this server" form is the ``__*`` suffix (a bare ``mcp__tau2`` is rejected).
ALLOWED_TOOLS = f"mcp__{SERVER_KEY}__*"

# Deny the built-in tools so the agent can't take untraced side-channel actions (read the
# tau2 data files, shell out, fetch the web) that never appear in the hgym trace — keeping the
# score attributable to the MCP tool surface alone. `--permission-mode dontAsk` already denies
# everything not in `--allowedTools`, but its read-only exemption would still permit Read/Grep;
# denying them explicitly closes that. We do NOT use `--tools ""` for this: in current Claude
# Code that strips the *MCP* tools too, leaving the agent with no tau2 tools at all.
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
    "You are the agent for a tau2-bench customer-service task, played through the "
    f"`{SERVER_KEY}` MCP tools. First call `describe` to read the domain policy and your "
    "task. Then use the domain tools to complete it (on non-solo domains, call "
    "`send_message` to talk to the user — its result is their reply). When the task is "
    "complete, call `done` (its result reports the score), then call `terminate` to end "
    "the episode."
)


def env_name(domain: str) -> str:
    return f"tau2_{domain}"


def _tau2_data_cache() -> Path:
    return Path(os.path.expanduser("~/.cache/hgym/tau2-bench"))


def ensure_tau2_data() -> None:
    """Make tau2's benchmark ``data/`` available (it isn't shipped in the install).

    Honors ``TAU2_DATA_DIR`` if already set. Otherwise lazy-clones the pinned tau2-bench to a
    user cache (once) and points ``TAU2_DATA_DIR`` at it. Must run *before* tau2 is imported
    (tau2 resolves its data path at import) and *before* spawning Claude Code (so the served
    subprocess inherits the variable). A no-op on repeat runs once the cache exists."""
    if os.environ.get("TAU2_DATA_DIR"):
        return
    cache = _tau2_data_cache()
    data = cache / "data"
    marker = data / "tau2" / "domains" / "mock" / "tasks.json"
    if not marker.exists():
        if shutil.which("git") is None:
            return  # no git — let _assert_env_constructible report the missing data
        print(f"[tau2 example] downloading tau2 data to {cache} (one-time)…", file=sys.stderr)
        cache.mkdir(parents=True, exist_ok=True)
        if not (cache / ".git").exists():
            subprocess.run(["git", "init", "-q", str(cache)], check=True)
            subprocess.run(
                ["git", "-C", str(cache), "remote", "add", "origin", TAU2_REPO], check=True
            )
        # Fetch just the pinned commit (GitHub allows fetch-by-SHA), then check out its tree.
        subprocess.run(
            ["git", "-C", str(cache), "fetch", "-q", "--depth", "1", "origin", TAU2_SHA],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(cache), "checkout", "-q", "FETCH_HEAD"], check=True
        )
    if marker.exists():
        os.environ["TAU2_DATA_DIR"] = str(data)


def _assert_server_can_import_tau2() -> None:
    """The MCP server runs under this same interpreter (``write_mcp_config`` uses
    ``sys.executable``). If it can't ``import tau2``, ``hgym serve tau2_*`` will crash at
    startup and Claude Code will connect to a server with **no tools** and give up — an
    opaque failure. Fail fast here with the fix instead."""
    try:
        importlib.import_module("tau2")
    except ImportError as exc:
        raise SystemExit(
            f"[tau2 example] the server interpreter cannot import tau2:\n"
            f"    {sys.executable}\n"
            f"    ({exc})\n"
            f"Sync the dev environment (this repo pins Python 3.12 and installs tau2 there):\n"
            f"    uv sync\n"
            f"then run this via `uv run python …`. tau2 requires Python <3.13."
        )


def _assert_env_constructible(domain: str) -> Any:
    """tau2 imports and data is provisioned — confirm the domain actually builds (the same
    probe the served process does), turning any residual failure into an actionable message
    rather than a served-server crash. Returns the constructed env."""
    try:
        return hgym.make(env_name(domain))
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"[tau2 example] tau2 is installed but building {env_name(domain)!r} failed:\n"
            f"    {type(exc).__name__}: {exc}\n"
            f"If this is missing data, set TAU2_DATA_DIR to a tau2 data checkout (or ensure "
            f"`git` is available so the example can download it to {_tau2_data_cache()})."
        )


def _assert_user_sim_key(domain: str, env: Any) -> None:
    """Non-solo domains drive tau2's user simulator (an OpenAI LLM) during the episode. Without
    a key the run would fail mid-episode at the first user turn, so check up front."""
    if not getattr(env, "solo_mode", True) and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"[tau2 example] {env_name(domain)!r} is non-solo: tau2's user simulator needs "
            f"OPENAI_API_KEY (real cost).\n"
            f"Set OPENAI_API_KEY to run it, or use the default `--domain mock` for the "
            f"offline, no-key demo."
        )


def preflight(domain: str) -> None:
    """Provision tau2 data + verify the interpreter can serve this domain (and, for non-solo
    domains, that the user-simulator key is present), before spawning Claude."""
    ensure_tau2_data()
    _assert_server_can_import_tau2()
    env = _assert_env_constructible(domain)
    _assert_user_sim_key(domain, env)


def write_mcp_config(config_path: Path, domain: str, task: str, trace_path: Path) -> Path:
    """Write a ``.mcp.json`` that makes Claude Code spawn ``hgym serve tau2_<domain>``.

    Invoke the CLI through the *current* interpreter (``python -m hgym.cli``) rather than a
    bare ``hgym`` on ``PATH`` — so the example works whether or not the venv that has hgym
    (with the ``tau2`` extra) installed is the one on the spawned subprocess's ``PATH``."""
    config = {
        "mcpServers": {
            SERVER_KEY: {
                "command": sys.executable,
                "args": [
                    "-m",
                    "hgym.cli",
                    "serve",
                    env_name(domain),
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
    attributable solely to this ``hgym serve`` process), the tau2 tools pre-allowed, the
    built-ins denied (so the run stays fully trace-attributable), and ``--permission-mode
    dontAsk`` so tool calls run non-interactively (it denies anything not pre-allowed).

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
    domain: str = DEFAULT_DOMAIN,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    transcript: bool = False,
    workdir: Optional[Path] = None,
) -> hgym.EvalResult:
    """Run Claude Code on one tau2 task and read the terminal feedback off the trace."""
    # Fail fast (with a fix) if the server interpreter can't serve this domain — otherwise
    # Claude Code connects to a crashed server, sees no tools, and gives up opaquely.
    preflight(domain)
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hgym-claude-tau2-"))
    env = env_name(domain)
    trace_path = workdir / f"{env}.jsonl"
    mcp_config = write_mcp_config(workdir / ".mcp.json", domain, task, trace_path)

    cmd = build_claude_command(mcp_config, model, effort, transcript=transcript)
    if transcript:
        _run_with_transcript(cmd)
    else:
        subprocess.run(cmd, check=True)

    if not trace_path.exists():
        # No tool call was served (e.g. Claude answered with text only or declined after
        # MCP startup), so no trace file exists: report an unterminated run rather than
        # letting load_traces raise FileNotFoundError.
        return hgym.EvalResult(env=env, task=task, terminated=False, trace_path=str(trace_path))
    return hgym.result_from_trace(trace_path, env=env, task=task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domain",
        default=DEFAULT_DOMAIN,
        help=(
            f"tau2 domain (default: {DEFAULT_DOMAIN} — solo, so no OpenAI key / user-sim "
            "cost; the Claude harness still runs). Real domains (airline, retail, telecom, "
            "banking_knowledge) need OPENAI_API_KEY for tau2's user simulator (real cost)."
        ),
    )
    parser.add_argument("--task", default="0", help="tau2 task index within the split")
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
        domain=args.domain,
        model=args.model,
        effort=args.effort,
        transcript=args.transcript,
    )
    print(
        f"env={result.env} task={result.task} terminated={result.terminated} "
        f"reward={result.value('reward')} success={result.value('success')}"
    )
    print(f"feedback: {result.feedback}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    main()
