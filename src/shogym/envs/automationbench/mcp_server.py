# AutomationBench's source is provisioned at runtime into a cache (see adapter.ensure_source);
# it is intentionally absent from the base type-check / offline environment, so its imports are
# expected to be unresolved there.
# pyright: reportMissingImports=false
"""In-process MCP server for the ``automationbench`` env — the pinned ``api`` toolset, served.

AutomationBench offers several tool "variants"; this port pins the **``api``** toolset (the
simplest, and the one whose difficulty this port reproduces): three generic tools over a
per-session :class:`WorldState` —

  - **``api_search(query, top_k=5)``** — BM25 search over the upstream endpoint schemas
    (top-5 by default) to discover the URL + params for an endpoint.
  - **``api_fetch(method, url, params, body)``** — route a REST-style call into *this session's*
    ``WorldState``, mimicking the real SaaS API response. All state mutation happens here, in the
    upstream router; service gating (``allowed_services``) is enforced exactly as upstream.
  - **``base64_encode(text)``** — a local helper (the Gmail body encoding); no API call.

plus an shogym ``score`` terminal tool:

  - **``done()``** — the env's ``score`` terminal. The serve layer intercepts a ``done`` call as a
    *validate -> seal -> finalize* transaction, so this tool is **never dispatched inward** for a
    served episode: the score is computed by the env's ``finalize`` hook (which calls
    :func:`score_session` on the *already-sealed* world), not by this handler. The handler body is
    a harmless notice for the (unused) direct-dispatch path; it never scores and leaks nothing.

The session's ``WorldState`` is scored by :func:`score_session` — server-side, with the reused
rubric — returning only the score numbers (never the assertions / targets / world), so no oracle
is exposed through the tool surface. State is keyed by ``_session_id`` (shogym injects the real id),
so concurrent episodes are isolated. All ``automationbench`` imports are funnelled through
:mod:`shogym.envs.automationbench.adapter`, so importing this module requires the provisioned
upstream source — but it is only imported when an ``automationbench`` env is constructed or served.
"""

from __future__ import annotations

import json
import threading
from hashlib import sha256
from typing import Any, Dict, Optional, Tuple

from fastmcp import FastMCP

from shogym.envs.automationbench import adapter

DONE_TOOL_NAME = "done"

server: FastMCP = FastMCP(name="automationbench")

# session_id -> _Session
_sessions: Dict[str, "_Session"] = {}
_lock = threading.RLock()


class _Session:
    """One task's private ``WorldState`` plus the assertions/initial-state it is scored against."""

    def __init__(self, info: Dict[str, Any]) -> None:
        world, initial_state, assertions = adapter.build_world(info)
        self.world = world
        self.initial_state = initial_state
        self.assertions = assertions
        # The world as it was handed over, named by its own bytes. What it answers is whether the
        # world a seal reads is still the one this session was seeded with, which is the one thing
        # a task with no assertion satisfied cannot say for itself: a run that changed nothing and
        # a run that changed the wrong things both score zero and are not the same filing.
        self.seeded_digest = _world_digest(world)


# ----- session lifecycle (called in-process by the env) -----


def begin_session(session_id: str, *, info: Dict[str, Any]) -> None:
    """Create + seed a fresh per-episode ``WorldState`` for this task. Idempotent per id."""
    session = _Session(info)
    with _lock:
        _sessions[session_id] = session


def end_session(session_id: str) -> None:
    """Drop a finished episode's world. Idempotent."""
    with _lock:
        _sessions.pop(session_id, None)


def reset_state() -> None:
    """Drop all sessions (test hygiene)."""
    with _lock:
        _sessions.clear()


def _session_for(session_id: Optional[str]) -> Optional["_Session"]:
    if session_id is None:
        return None
    with _lock:
        return _sessions.get(session_id)


def score_session(session_id: Optional[str]) -> Tuple[float, float]:
    """Score a live session's *current* ``WorldState`` server-side with the reused rubric.

    Returns ``(partial_credit, success)`` as floats — only the score numbers, never the assertions,
    target values, or world dump, so nothing an agent could act on is exposed. This is the sole
    scoring entry point: the env's ``finalize`` hook calls it on the already-sealed world, so a
    graded verdict only ever exists for a sealed, un-continuable episode (no read-score-then-fix
    exploit). A missing session (never began, or already torn down) scores a clean zero.

    Scored against this session's private world, so it can't be forged through the tool surface.
    ``partial_credit`` runs first inside :func:`adapter.score_state` (it caches its score for the
    pass-rate metric); the negative-assertion "must not shotgun" gate is enforced there verbatim.

    The **live** world object is handed to the rubric, never a serialized copy of it: a
    tool-mutated world does not always survive re-validation, and part of what the rubric reads
    is recorded outside the model's declared fields (see :func:`adapter.score_state`).
    """
    session = _session_for(session_id)
    if session is None:
        return 0.0, 0.0
    return adapter.score_state(session.world, session.initial_state, session.assertions)


def _world_digest(world: Any) -> str:
    """Name one world by the bytes it serializes to.

    The dump is not what the rubric reads and is not offered as one: part of what a score is taken
    from is recorded outside the model's declared fields (see :func:`adapter.score_state`). What a
    digest over it is good for is saying which end state this is, which is what a submission has
    to name and what an unchanged world has to be distinguishable by.
    """
    return sha256(world.model_dump_json().encode("utf-8")).hexdigest()


def sealed_state(session_id: str) -> Optional[Dict[str, Any]]:
    """This session's end state and what the reused rubric makes of it, or nothing at all.

    :func:`score_session` answers a session that is not here with zeros, which is not faithful
    under a durable stream: a seal can arrive after the world has been let go, and a zero
    published there is a grade for a world nobody read. So this says which of the two happened
    and lets the caller refuse.

    The world itself does not come back, because a capture is kept and the world is large. What
    comes back is its digest beside the numbers the rubric produced from it, which is everything a
    grade and a submission are taken from.
    """
    session = _session_for(session_id)
    if session is None:
        return None
    partial_credit, success = adapter.score_state(
        session.world, session.initial_state, session.assertions
    )
    digest = _world_digest(session.world)
    return {
        "world_sha256": digest,
        "untouched": digest == session.seeded_digest,
        "partial_credit": float(partial_credit),
        "success": float(success),
    }


def _no_session_error() -> str:
    return json.dumps(
        {
            "error": {
                "code": 500,
                "message": "session not initialized; env did not call begin_session",
            }
        }
    )


# ----- MCP tools (the `api` toolset) -----


@server.tool
def api_search(query: str, _session_id: str, top_k: int = 5) -> str:
    """Search available API endpoints by keyword (BM25 over endpoint descriptions).

    Use this to discover which endpoint to call before using ``api_fetch``. Returns full endpoint
    details — id, method, url, description, parameters, request body, and response format.

    Args:
        query: Space-separated keywords (API-native terms: "messages" not "emails", "trash" not
               "delete").
        top_k: Maximum number of results to return (default 5).
    """
    return adapter.api_search(query, top_k=top_k)


@server.tool
def api_fetch(
    method: str,
    url: str,
    _session_id: str,
    params: Optional[str] = None,
    body: Optional[str] = None,
) -> str:
    """Call an API endpoint by its full URL, routing to the appropriate world-state mutation.

    Use ``api_search`` first to discover the correct URL and parameters for an endpoint.

    Args:
        method: HTTP method (GET, POST, PUT, PATCH, DELETE).
        url: Full API URL from api_search results.
        params: Query parameters as a JSON string (e.g. '{"labelIds": "INBOX"}').
        body: Request body as a JSON string (e.g. '{"subject": "Hi", "body": "Hello"}').

    Returns:
        JSON string mimicking the real API response.
    """
    session = _session_for(_session_id)
    if session is None:
        return _no_session_error()
    return adapter.api_fetch(session.world, method, url, params, body)


@server.tool
def base64_encode(text: str, _session_id: str) -> str:
    """Encode text to base64url — the format Gmail API body fields require.

    A local helper; it does not call any API endpoint. Produce the encoding, then pass the result
    to ``api_fetch``.

    Args:
        text: Plaintext to encode (an email body, or a full RFC 2822 message).
    """
    return adapter.base64_encode(text)


@server.tool
def done() -> str:
    """Finish the task: end the episode and score the final workspace state.

    Call this once your workflow is complete. ``done`` is the terminal action — submitting it ends
    the episode and scores the current state of the workspace (there is no second submission and no
    ``done``-then-fix loop). You do not need to call ``terminate`` after ``done``.

    Scoring is performed by the harness on the sealed final state (AutomationBench's own rubric);
    this tool takes no arguments and reveals nothing about the grade.
    """
    # The serve layer intercepts `done` as the env's `score` terminal (validate -> seal ->
    # finalize), so this body is never dispatched for a served episode — scoring happens in
    # `env_v1.AutomationBenchEnv.finalize` via `score_session`. This notice covers only a direct
    # in-process call that bypasses the seal path; it never scores and leaks nothing.
    return json.dumps({"note": "`done` ends the episode; the harness scores the sealed state"})


__all__ = [
    "server",
    "begin_session",
    "end_session",
    "reset_state",
    "score_session",
    "sealed_state",
    "DONE_TOOL_NAME",
]
