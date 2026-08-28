"""In-process MCP server for the score-terminal fixture env (tests only).

Two tools: ``submit`` (the env's ``score`` terminal — its args are validated then the seal
transaction runs the env's ``finalize``, so this handler is never dispatched inward for a
sealed episode) and ``noop`` (an ordinary tool used to exercise mid-episode steps and the
horizon path). Session state (the gold answer) is keyed by ``_session_id`` so the offline,
deterministic ``finalize`` can grade without any network or key.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from fastmcp import FastMCP

server: FastMCP = FastMCP(name="fixture_score")

_sessions: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()


def begin_session(session_id: str, *, answer: str) -> None:
    with _lock:
        _sessions[session_id] = {"answer": answer}


def end_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def gold(session_id: str) -> str:
    with _lock:
        state = _sessions.get(session_id)
        return "" if state is None else str(state.get("answer", ""))


def reset_state() -> None:
    with _lock:
        _sessions.clear()


@server.tool
def submit(answer: str, _session_id: str, confidence: int = 100) -> Dict[str, Any]:
    """Submit a final answer. This is the score terminal: the serve layer validates these args
    and seals before the env's ``finalize`` grades them, so this body is not the trust source
    (and, for a sealed episode, is never dispatched)."""
    return {"submitted": True}


@server.tool
def noop(_session_id: str) -> Dict[str, Any]:
    """An ordinary mid-episode tool that changes nothing."""
    return {"ok": True}


#: Released by a test to let `block` return. A real blocking worker rather than a cancellable
#: async substitute: FastMCP runs a synchronous tool in a worker thread, which is the shape of
#: AppWorld's own `execute` (a blocking HTTP call into another process). Cancelling the coroutine
#: that awaits it abandons the await and not the operation.
released = threading.Event()
#: Set by `block` once it has actually returned, so a test can tell "still running" from "landed".
landed = threading.Event()
#: What `block` did to the world before it returned, which stands in for a world mutation an
#: abandoned call makes after its caller has gone.
mutations: list = []


@server.tool
def block(_session_id: str) -> Dict[str, Any]:
    """An ordinary tool that does not return until a test lets it, and then mutates."""
    released.wait(timeout=30)
    mutations.append(_session_id)
    landed.set()
    return {"ok": True, "mutations": len(mutations)}


def reset_block() -> None:
    released.clear()
    landed.clear()
    mutations.clear()
