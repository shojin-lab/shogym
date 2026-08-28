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
    experiment: str
    #: This episode's own served corpus on the host, removed when the session ends (see
    #: `world.derive_view`). The world never sees this path: inside the container it is mounted
    #: at a fixed name.
    view: str = ""
    #: This episode's own output tree on the host, mounted alone into the world's container and
    #: again into the grading one. Two containers, one directory, and no other episode's.
    outputs: Any = None
    #: How many blocks of code this episode may run. The step budget the serve layer enforces has
    #: one slot more than this, so that `submit` always has somewhere to go; without a separate
    #: count the spare slot is just another `execute`, and a task's world can be changed after the
    #: budget it was supposed to be scored under has run out.
    budget: int = 0
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
    with _state_lock:
        session = sessions.get(_session_id)
        if session is None:
            return {
                "output": "<error: session not initialized; env did not call begin_session>",
                "calls": 0,
            }
        if session.budget and session.calls >= session.budget:
            return {
                "output": (
                    "<no code budget left: this task allows "
                    f"{session.budget} blocks and has run {session.calls}. "
                    "Call `submit` to end the task.>"
                ),
                "calls": session.calls,
            }
        session.calls += 1
        calls = session.calls
    # Outside the lock. The lock is over the session table, which is shared by every episode this
    # env instance backs; the call is over one world, which is this episode's alone. Holding the
    # table's lock across a call that runs an agent's code would make two concurrent episodes take
    # turns, and one slow block would stall the other.
    # The block number goes with the request. It is the host's own count, and the world records
    # it beside the save it makes, so a save that never finished is one the host can see because
    # the number it gets back is the block before.
    answered = session.worker.call("execute", code=str(code), block=calls)
    return {"output": str(answered.get("output", "")), "calls": calls}


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
