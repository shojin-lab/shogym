"""In-process MCP server for the ``browsecomp_plus`` env — the served retrieval + answer tools.

BrowseComp-Plus already exposes its retriever over MCP (``searcher/mcp_server.py``:
``search`` + optional ``get_document``); shogym reuses that surface and adds a ``submit_answer``
tool to demarcate the final answer. Three tools:

  - **``search(query)``** — rank the corpus and return the top-k hits (``docid``, ``score``,
    ``snippet``), mirroring upstream's ``search`` tool. The docids returned across a
    run are what the pure verifier reads back as ``retrieved_docids`` for retrieval recall.
  - **``get_document(docid)``** — fetch a document's full text (upstream's optional tool).
  - **``submit_answer(answer, confidence)``** — the env's ``score`` **terminal**. The serve
    layer validates its args, atomically **seals** the episode, then runs the env's ``finalize``
    hook (the LLM judge) on the frozen submission — so this handler body is *never dispatched
    inward* for a sealed episode. It stays here only to advertise the tool + its argument schema
    on the manifest; grading lives in
    :meth:`shogym.envs.browsecomp_plus.env_v1.BrowseCompPlusEnv.finalize` (seal-before-verdict),
    never in a tool handler the agent can still race.

State is keyed by ``_session_id`` (shogym injects the real id), so concurrent episodes are
isolated. The searcher is pushed in via ``begin_session`` (an ``InMemorySearcher`` for offline
tests; a pyserini ``BM25Searcher`` for the real env); the judge + gold answer are held by the
env for ``finalize``, not here. Importing this module pulls in only ``fastmcp``.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from shogym.envs.browsecomp_plus.searcher import Searcher

SEARCH_TOOL_NAME = "search"
GET_DOCUMENT_TOOL_NAME = "get_document"
SUBMIT_TOOL_NAME = "submit_answer"

# Confidence is a 0–100 integer (HLE convention). The seal layer validates the raw arg; the env's
# finalize/verify clamp it (see env_v1._clamp_confidence / _confidence_fraction).
DEFAULT_CONFIDENCE = 100

# Default retrieval / snippet knobs (upstream defaults: k=5, snippet_max_tokens=512).
DEFAULT_K = 5
DEFAULT_SNIPPET_MAX_TOKENS = 512

server: FastMCP = FastMCP(name="browsecomp_plus")

# session_id -> session state dict
_sessions: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()


def begin_session(
    session_id: str,
    *,
    searcher: Searcher,
    k: int = DEFAULT_K,
    snippet_max_tokens: int = DEFAULT_SNIPPET_MAX_TOKENS,
) -> None:
    """Register a fresh episode's searcher + retrieval knobs.

    Called by the env (in-process) when an episode starts, before any tool is called. The judge
    and gold answer are NOT registered here — grading runs in the env's ``finalize`` hook after
    the seal, not in a served handler. Idempotent within a ``session_id``."""
    with _lock:
        _sessions[session_id] = {
            "searcher": searcher,
            "k": int(k),
            "snippet_max_tokens": int(snippet_max_tokens),
        }


def end_session(session_id: str) -> None:
    """Drop a finished episode's per-session state. Idempotent."""
    with _lock:
        _sessions.pop(session_id, None)


def reset_state() -> None:
    """Drop all sessions (test hygiene)."""
    with _lock:
        _sessions.clear()


def _state_for(session_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if session_id is None:
        return None
    with _lock:
        return _sessions.get(session_id)


def _snippet(text: str, max_tokens: int) -> str:
    """Truncate ``text`` to ``max_tokens`` whitespace tokens (``-1`` disables truncation).

    Upstream snippets by the Qwen3-0.6B subword tokenizer; shogym approximates with whitespace
    tokens to stay dependency-light (transformers is not a required dep). Snippet length is
    cosmetic for the agent — it never affects scoring (recall uses returned docids)."""
    if max_tokens is None or max_tokens < 0:
        return text
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens])


# ----- MCP tools -----


@server.tool(
    name="search",
    description=(
        "Search the fixed BrowseComp-Plus corpus and return the top-k hits. Each hit has a "
        "`docid`, a relevance `score`, and a `snippet` of the document's contents. Cite the "
        "docids you rely on in your final answer as [docid]."
    ),
)
def search(query: str, _session_id: str) -> str:
    """Retrieve the top-k documents for ``query`` (returns a JSON list of ``{docid, score, snippet}``)."""
    state = _state_for(_session_id)
    if state is None:
        return json.dumps({"error": "session not initialized; env did not call begin_session"})
    searcher: Searcher = state["searcher"]
    k = state["k"]
    snippet_max_tokens = state["snippet_max_tokens"]
    try:
        candidates = searcher.search(query, k)
    except Exception as exc:  # a searcher failure must not crash the episode
        return json.dumps({"error": f"search failed: {exc}"})

    results: List[Dict[str, Any]] = []
    for cand in candidates:
        entry: Dict[str, Any] = {"docid": str(cand["docid"])}
        if cand.get("score") is not None:
            entry["score"] = cand["score"]
        entry["snippet"] = _snippet(str(cand.get("text", "")), snippet_max_tokens)
        results.append(entry)
    return json.dumps(results)


@server.tool(
    name="get_document",
    description="Retrieve the full text of a document from the corpus by its `docid`.",
)
def get_document(docid: str, _session_id: str) -> str:
    """Return the full document ``{docid, text}`` for ``docid`` (JSON), or a not-found notice."""
    state = _state_for(_session_id)
    if state is None:
        return json.dumps({"error": "session not initialized; env did not call begin_session"})
    searcher: Searcher = state["searcher"]
    try:
        doc = searcher.get_document(str(docid))
    except Exception as exc:
        return json.dumps({"error": f"get_document failed: {exc}"})
    if doc is None:
        return json.dumps({"docid": str(docid), "text": None, "error": "document not found"})
    return json.dumps({"docid": str(doc["docid"]), "text": str(doc.get("text", ""))})


@server.tool(
    name="submit_answer",
    description=(
        "Submit your final answer for grading. Provide `answer` (cite supporting docids as "
        "[docid]) and a `confidence` from 0 to 100. This ends the episode: your answer is sealed "
        "and graded against the gold answer, and there is no second submission or further step."
    ),
)
def submit_answer(
    answer: str, _session_id: str, confidence: int = DEFAULT_CONFIDENCE
) -> Dict[str, Any]:
    """The ``score`` terminal. Its args are validated then the seal transaction runs the env's
    ``finalize`` (the LLM judge) on the sealed submission, so this body is never dispatched for a
    sealed episode — the grade never comes from a tool result the agent can inspect or replace."""
    return {"submitted": True}


__all__ = [
    "server",
    "begin_session",
    "end_session",
    "reset_state",
    "SEARCH_TOOL_NAME",
    "GET_DOCUMENT_TOOL_NAME",
    "SUBMIT_TOOL_NAME",
]
