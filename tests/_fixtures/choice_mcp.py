"""In-process MCP server for the second score-terminal fixture env (tests only).

It exists to collide with :mod:`tests._fixtures.score_mcp`: the same tool *names* (``submit``,
``noop``, and the always-present reserved ``terminate``) with a different schema behind
``submit`` — an ``int`` choice rather than a string answer. That is the real multi-env hazard,
since a server registers one schema per tool name.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

from fastmcp import FastMCP

server: FastMCP = FastMCP(name="fixture_choice")

_sessions: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()


def begin_session(session_id: str, *, choice: int) -> None:
    with _lock:
        _sessions[session_id] = {"choice": choice}


def end_session(session_id: str) -> None:
    with _lock:
        _sessions.pop(session_id, None)


def gold(session_id: str) -> int:
    with _lock:
        state = _sessions.get(session_id)
        return -1 if state is None else int(state.get("choice", -1))


@server.tool
def submit(choice: int, _session_id: str) -> Dict[str, Any]:
    """Submit a final choice. The score terminal: the serve layer validates these args and
    seals before the env's ``finalize`` grades them."""
    return {"submitted": True}


@server.tool
def noop(_session_id: str) -> Dict[str, Any]:
    """An ordinary mid-episode tool that changes nothing."""
    return {"ok": True}
