"""FastMCP server for the appworld env.

Two tools. ``execute`` runs one block of Python against the episode's world and answers with what
it printed; ``submit`` is the score terminal, which the serve layer turns into a validate -> seal
-> finalize transaction. State is keyed by ``_session_id`` so one env instance can back several
concurrent episodes, each with a world in a process of its own.

**Nothing here can grade anything.** The tool surface is the world's own Python API and an end
call, and the end call answers with the word that ended it. What the episode scored is produced by
the env's ``finalize`` on the already-sealed episode, out of this process's reach, and reaches an
agent only if the run's feedback policy decides it does.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

server: FastMCP = FastMCP(name="appworld")

sessions: Dict[str, "Session"] = {}

_state_lock = threading.RLock()


@dataclass
class Session:
    """One open world, and the little the tools need to know about it."""

    worker: Any
    task_id: str
    supervisor_email: str
    calls: int = 0


def begin_session(session_id: str, session: "Session") -> None:
    """Register an opened world as this session's. Idempotent within a session id."""
    with _state_lock:
        sessions[session_id] = session


def end_session(session_id: str) -> Optional["Session"]:
    """Drop a finished episode's state and hand the session back so the env can close it."""
    with _state_lock:
        return sessions.pop(session_id, None)


def get_session(session_id: str) -> Optional["Session"]:
    with _state_lock:
        return sessions.get(session_id)


@server.tool
def execute(code: str, _session_id: str) -> Dict[str, Any]:
    """Run one block of Python in the world and return what it printed.

    The block runs in a shell that persists across calls, so a name bound in one call is still
    bound in the next. ``apis`` is already there; ``print`` anything you want to read, because
    only printed output comes back.

    Returns a dict with:
      - ``output``: everything the block printed, or the traceback if it raised
      - ``calls``: how many blocks this episode has run
    """
    session = get_session(_session_id)
    if session is None:
        return {
            "output": "<error: session not initialized; env did not call begin_session>",
            "calls": 0,
        }
    with _state_lock:
        session.calls += 1
        answered = session.worker.call("execute", code=str(code))
    return {"output": str(answered.get("output", "")), "calls": session.calls}


@server.tool
def submit() -> str:
    """Finish the task: end the episode and record what it did.

    Call it once you have finished the instruction and updated the filing log. ``submit`` is the
    terminal action: there is no second submission, no submit-then-fix loop, and no need to call
    ``terminate`` after it. It takes no arguments and reveals nothing about the grade.
    """
    # The serve layer intercepts `submit` as the env's `score` terminal (validate -> seal ->
    # finalize), so this body is never dispatched for a served episode. Scoring happens in
    # `env_v1.AppWorldEnv.finalize` against a key this process has never held. It answers at all
    # only for a direct, unsealed call against this server on its own.
    return "task ended"


__all__: List[str] = [
    "Session",
    "begin_session",
    "end_session",
    "execute",
    "get_session",
    "server",
    "submit",
]
