"""In-process MCP server for ``orca_bench``: the SRE agent's surface on a task's stack.

ORCA-bench's agent works inside the task container: it queries the replayed telemetry through
Grafana's HTTP API (the URL and credentials are in the instruction), reads the demo application's
source under ``/app/opentelemetry-demo``, and writes its incident report to ``/app/report.md``.
So the served surface is a shell plus file I/O, and one score terminal:

  - **``exec(command)``** runs a shell command in the task container.
  - **``read_file(path)`` / ``write_file(path, content)``** do file I/O into the container.
  - **``submit_report()``** is the ``score`` terminal. Calling it seals the episode; the env's
    ``finalize`` hook then runs the task's own verifier over ``/app/report.md`` and returns the
    judge's verdict as core-owned evidence.

The tool *schemas* list without a backend, so ``describe`` and the manifest probe stay offline,
which is what lets phase 1 register, describe and score this env with no stack at all.
Starting a session needs the backend, and until phase 2 lands
:func:`shogym.envs.orca_bench.backend.create_backend` raises.
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from shogym.envs.orca_bench.backend import OrcaBackend, create_backend

EXEC_TOOL_NAME = "exec"
READ_FILE_TOOL_NAME = "read_file"
WRITE_FILE_TOOL_NAME = "write_file"
SUBMIT_TOOL_NAME = "submit_report"

DEFAULT_COMMAND_TIMEOUT_SECONDS = 600.0

server: FastMCP = FastMCP(name="orca_bench")

# session_id -> backend
_sessions: Dict[str, OrcaBackend] = {}
_lock = threading.RLock()


def begin_session(session_id: str, *, task_dir: Any, **kwargs: Any) -> None:
    """Bring up this episode's stack. Raises ``BackendUnavailableError`` until phase 2 lands."""
    backend = create_backend(task_dir, **kwargs)
    with _lock:
        old = _sessions.pop(session_id, None)
        _sessions[session_id] = backend
    if old is not None:
        old.teardown()


def finalize_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Grade the sealed episode; return the raw judge payloads (``None`` if no session).

    Called by the env's ``finalize`` hook after the serve layer has sealed the episode and before
    teardown, so the report this grades is the agent's true final one.

    Two steps, in this order and never merged: **capture** the report once, then verify those
    bytes. Capture is what makes "the agent's true final report" a fact rather than a hope, since
    the seal does not stop processes already running in the agent's container (see
    :meth:`~shogym.envs.orca_bench.backend.OrcaBackend.capture_report`). A capture that found the
    submission ungradeable never reaches the judge: it is a graded zero, and running the verifier
    on it would only produce the judge-error shape this port excludes."""
    backend = _session_for(session_id)
    if backend is None:
        return None
    captured = backend.capture_report()
    if captured.problem:
        return {"reward": None, "details": None, "submission_error": captured.problem}
    return backend.run_verifier(captured)


def end_session(session_id: str) -> None:
    """Tear down a finished episode's stack. Idempotent."""
    with _lock:
        backend = _sessions.pop(session_id, None)
    if backend is not None:
        backend.teardown()


def reset_state() -> None:
    """Drop all sessions (test hygiene)."""
    with _lock:
        ids = list(_sessions)
    for session_id in ids:
        end_session(session_id)


def _session_for(session_id: Optional[str]) -> Optional[OrcaBackend]:
    if session_id is None:
        return None
    with _lock:
        return _sessions.get(session_id)


def _no_session_error(tool: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": f"session not initialized; env did not call begin_session ({tool})",
        }
    )


# ----- MCP tools -----


@server.tool
def exec(command: str, _session_id: str) -> str:
    """Run one shell command inside the task container and return its result.

    ``command`` is a full shell command line. The telemetry stack answers on the Grafana HTTP API
    named in the task instruction (``$GRAFANA_URL``), and the demo application's source is under
    ``/app/opentelemetry-demo``. Returns JSON with ``ok``, ``exit_code``, ``stdout``, ``stderr``.
    """
    backend = _session_for(_session_id)
    if backend is None:
        return _no_session_error("exec")
    return json.dumps(backend.exec(command, timeout=DEFAULT_COMMAND_TIMEOUT_SECONDS))


@server.tool
def read_file(path: str, _session_id: str) -> str:
    """Read a file from the task container. Returns JSON with ``ok`` and ``content``."""
    backend = _session_for(_session_id)
    if backend is None:
        return _no_session_error("read_file")
    return json.dumps(backend.read_file(path))


@server.tool
def write_file(path: str, content: str, _session_id: str) -> str:
    """Write ``content`` to ``path`` inside the task container (parent dirs created).

    The graded artifact is ``/app/report.md``: write the incident RCA report there in the four
    sections the instruction specifies, or write it empty if no incident occurred.
    """
    backend = _session_for(_session_id)
    if backend is None:
        return _no_session_error("write_file")
    return json.dumps(backend.write_file(path, content))


@server.tool
def submit_report(_session_id: str) -> str:
    """Finish the task: seal the episode and grade the report at ``/app/report.md``.

    This is the env's ``score`` terminal. Calling it ends the episode: the serve layer seals the
    run and the task's own LLM judge scores the report against the ground-truth rubrics. The
    judge's reasoning and the rubrics are never returned, and ``submit_report`` is structurally
    one-shot (a failing verdict cannot be inspected and re-graded), so write the full report
    before calling it.

    (The serve layer intercepts this as the score terminal and does not dispatch this handler on
    the sealed path; the body is a defensive inert fallback for any non-sealed invocation.)
    """
    return json.dumps(
        {"note": "submit_report is the score terminal; the serve layer seals and finalizes it"}
    )


__all__ = [
    "EXEC_TOOL_NAME",
    "READ_FILE_TOOL_NAME",
    "SUBMIT_TOOL_NAME",
    "WRITE_FILE_TOOL_NAME",
    "begin_session",
    "end_session",
    "finalize_session",
    "reset_state",
    "server",
]
