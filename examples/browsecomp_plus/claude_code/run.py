"""Evaluate Claude Code on a served BrowseComp-Plus env through hgym (issue #43).

The **external-harness** path: hgym does not drive Claude Code — Claude Code spawns
``hgym serve browsecomp_plus`` as its MCP server (per the generated ``.mcp.json``), reads the
query via ``describe``, gathers evidence with ``search`` / ``get_document``, then calls
``submit_answer`` — the **score terminal**: it seals the episode and the env's ``finalize`` hook
grades the sealed submission with the LLM judge (seal-before-verdict), so that one call ends the
episode (no separate ``terminate``). When it finishes, we read the terminal feedback (``correct``
+ ``retrieval_recall`` + ``citation_recall``) with :func:`hgym.result_from_trace`.

Run it::

    # Needs the `browsecomp_plus` extra + an OpenAI key for the judge, Hugging Face access to
    # `Tevatron/browsecomp-plus`, a **Java 21** runtime (pyserini/Lucene), and the `claude` CLI.
    # The prebuilt BM25 index auto-downloads once to ~/.cache/hgym on first use (no manual step);
    # set HGYM_BROWSECOMP_PLUS_BM25_INDEX only to reuse a pre-provisioned index. The preflight
    # fails fast with the exact fix if any piece is missing.
    export OPENAI_API_KEY=sk-...
    uv run python examples/browsecomp_plus/claude_code/run.py --task 0
    uv run python examples/browsecomp_plus/claude_code/run.py --task 0 --transcript

Nothing about hgym is Claude-specific — swap ``build_claude_command`` for a Codex / pi / Hermes
invocation pointed at the same ``.mcp.json`` and the trace/score path is identical.
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

ENV_NAME = "browsecomp_plus"

# Latest Sonnet (the alias ``sonnet`` also resolves to it); override with --model.
DEFAULT_MODEL = "claude-sonnet-5"

# Deep-Research queries reward extended search + reasoning; give the solver real room by default.
DEFAULT_EFFORT = "high"

# The key under `mcpServers` in the generated .mcp.json. Claude Code namespaces this server's
# tools as ``mcp__<SERVER_KEY>__<tool>``.
SERVER_KEY = "browsecomp_plus"

# Allow every tool the served env exposes (search + get_document + submit_answer + describe +
# terminate) without enumerating them. Claude Code's allow-rule grammar requires the glob-free
# ``mcp__<server>__*`` form (a bare ``mcp__browsecomp_plus`` is rejected).
ALLOWED_TOOLS = f"mcp__{SERVER_KEY}__*"

# Deny the built-ins so the agent can't take untraced side-channel actions — above all
# WebFetch / WebSearch, since the whole point of BrowseComp-Plus is a *fixed* corpus: retrieval
# must go through the served `search` tool, not the live web.
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
    "You are answering one BrowseComp-Plus research question, played through the "
    f"`{SERVER_KEY}` MCP tools. First call `describe` to read the question. Use `search` to "
    "find relevant documents in the fixed corpus and `get_document` to read full texts — you "
    "have no web access. Reason over the evidence, then call `submit_answer` exactly once with "
    "your final `answer` (cite supporting docids as [docid]) and a `confidence` from 0 to 100. "
    "Submitting seals and grades your answer and ends the episode — there is no second submission "
    "and no further step (do not call `terminate` afterward), so commit to your best answer."
)


def _assert_server_can_import_extra() -> None:
    """The MCP server runs under this same interpreter (``write_mcp_config`` uses
    ``sys.executable``). If it can't import the ``browsecomp_plus`` extra, ``hgym serve`` crashes
    at startup and Claude Code connects to a toolless server and gives up — an opaque failure.
    Fail fast here with the fix instead."""
    for module in ("datasets", "openai", "pyserini"):
        try:
            importlib.import_module(module)
        except ImportError as exc:
            raise SystemExit(
                f"[browsecomp_plus example] the server interpreter cannot import {module!r}:\n"
                f"    {sys.executable}\n"
                f"    ({exc})\n"
                f"Install the browsecomp_plus extra and run via `uv run python …`:\n"
                f"    uv sync   # includes hgym[browsecomp_plus] in the dev group\n"
                f"(pyserini's BM25 backend also needs a Java 21 runtime.)"
            )


def _assert_judge_key() -> None:
    """The env grades the sealed ``submit_answer`` in its ``finalize`` hook with an OpenAI LLM
    judge. Without a key the run would grade every answer incorrect, so check up front."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "[browsecomp_plus example] the judge needs OPENAI_API_KEY (the model-graded verifier "
            "calls an OpenAI model).\n    export OPENAI_API_KEY=sk-...\n"
            "Or point the judge at a keyless vLLM Qwen3-32B via judge_base_url (see the README)."
        )


def _assert_env_constructible() -> None:
    """Confirm the env actually builds (the same probe the served process does) — which decrypts
    the queries and joins the qrels — turning any residual failure into an actionable message
    rather than a served-server crash.

    NOTE: building the env downloads the ~2.78 GB ``Tevatron/browsecomp-plus`` dataset on a cold
    cache, so ``preflight`` runs the cheap Java check (inside ``_assert_retriever_ready``) FIRST —
    a missing Java 21 runtime should fail fast without paying for that download or the index's."""
    try:
        hgym.make(ENV_NAME)
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"[browsecomp_plus example] building {ENV_NAME!r} failed:\n"
            f"    {type(exc).__name__}: {exc}\n"
            "Common causes: no Hugging Face access to Tevatron/browsecomp-plus, or the qrel "
            "files could not be downloaded."
        )


def _assert_retriever_ready() -> None:
    """Open the prebuilt BM25 index up front — the env defers ``BM25Searcher`` to episode start,
    so ``hgym.make`` alone would *not* catch a missing Java runtime. ``bm25_index_path`` checks
    Java **first**, then auto-downloads the prebuilt index once (to ``~/.cache/hgym``) if it isn't
    already cached; constructing the searcher here opens the Lucene index — surfacing a missing
    Java 21 / HF access before Claude connects, instead of as an opaque crash inside the spawned
    MCP server's first episode. The Java check runs before the multi-GB download."""
    try:
        from hgym.envs.browsecomp_plus.data import bm25_index_path
        from hgym.envs.browsecomp_plus.searcher import BM25Searcher

        BM25Searcher(bm25_index_path())
    except SystemExit:
        raise
    except Exception as exc:
        raise SystemExit(
            f"[browsecomp_plus example] the BM25 retriever is not ready:\n"
            f"    {type(exc).__name__}: {exc}\n"
            "The prebuilt BM25 index auto-downloads once to ~/.cache/hgym on first use; the "
            "failure you'll actually hit is a missing Java 21 runtime (needed by pyserini/Lucene) "
            "or no Hugging Face access. Install a JDK 21 (`java -version` reports 21). You can "
            "also point HGYM_BROWSECOMP_PLUS_BM25_INDEX at a pre-provisioned index."
        )


def preflight() -> None:
    """Verify the interpreter can serve the env (extra installed, judge key present, env builds,
    retriever opens) before spawning Claude — so a missing piece is a clear message, not a
    crashed server the harness silently gives up on."""
    _assert_server_can_import_extra()
    _assert_judge_key()
    # Cheap Java check BEFORE _assert_env_constructible (which downloads the ~2.78 GB dataset) and
    # before the index download it triggers: a missing Java 21 runtime fails fast without paying
    # for either download. _assert_retriever_ready checks Java first, then auto-provisions the
    # prebuilt BM25 index (once) and opens it.
    _assert_retriever_ready()
    _assert_env_constructible()


def write_mcp_config(config_path: Path, task: str, trace_path: Path) -> Path:
    """Write a ``.mcp.json`` that makes Claude Code spawn ``hgym serve browsecomp_plus``.

    Invoke the CLI through the *current* interpreter (``python -m hgym.cli``) rather than a bare
    ``hgym`` on ``PATH`` — so the example works whether or not the venv that has hgym (with the
    extra) installed is the one on the spawned subprocess's ``PATH``."""
    # Forward the JVM env to the spawned MCP server. Claude Code inherits PATH and OPENAI_API_KEY
    # into MCP subprocesses but NOT JAVA_HOME — and pyserini/pyjnius needs JAVA_HOME (not just
    # `java` on PATH) to locate libjvm, so without this the served retriever crashes at startup
    # ("Unable to find libjvm") and the server never connects. Only non-secret vars go here (the
    # judge key rides in via Claude's own inheritance); nothing is written that isn't already in
    # this process's environment.
    server: Dict[str, Any] = {
        "command": sys.executable,
        "args": ["-m", "hgym.cli", "serve", ENV_NAME, "--task", task, "--trace", str(trace_path)],
    }
    mcp_env = {v: os.environ[v] for v in ("JAVA_HOME", "JVM_PATH", "PATH") if os.environ.get(v)}
    if mcp_env:
        server["env"] = mcp_env
    config = {"mcpServers": {SERVER_KEY: server}}
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
    (``--strict-mcp-config`` keeps inherited servers out), the env's tools pre-allowed, the
    built-ins (crucially WebFetch/WebSearch) denied, and ``--permission-mode dontAsk`` so tool
    calls run non-interactively. ``transcript=True`` emits the turn-by-turn stream."""
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
    """Run Claude Code on one BrowseComp-Plus task and read the terminal feedback off the trace."""
    preflight()
    workdir = workdir or Path(tempfile.mkdtemp(prefix="hgym-claude-bcp-"))
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
    parser.add_argument("--task", default="0", help="browsecomp_plus task index within the split")
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
        f"retrieval_recall={result.value('retrieval_recall')} "
        f"citation_recall={result.value('citation_recall')}"
    )
    print(f"feedback: {result.feedback}")
    print(f"trace: {result.trace_path}")


if __name__ == "__main__":
    main()
