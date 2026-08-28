# yc-bench is an optional extra (the `yc_bench` install group); it is intentionally absent
# from the base type-check / offline environment, so its imports are expected to be
# unresolved there.
# pyright: reportMissingImports=false
"""In-process MCP server for the ``yc_bench`` env — yc-bench's command surface, wrapped.

yc-bench ships its own LLM agent loop (``agent/loop.py``), but its sim is built to take an
external driver: something has to issue CLI commands against the deterministic sim, feed the
JSON results back, and collect the next commands. shogym *is* that driver, in place of that
loop. This server exposes the command surface as two MCP tools:

  - **``run_command(command)``** — mirrors upstream's
    ``run_command("yc-bench <cmd>")``: it runs one yc-bench CLI command against *this
    session's* SQLite sim and returns the CLI's JSON. Every observe/act/sim/memory command is
    reached through this one tool (one tool for the whole surface, per issue #32).
  - **``submit()``** — the env's ``score`` terminal. Calling it ends the episode: the serve
    layer seals the episode, then the env's ``finalize`` hook reads the authoritative final
    metrics (survival, final funds, task outcomes) off the *live* sim DB and returns the
    core-owned verdict the pure ``verify`` scores from. Because the metrics are read from the
    sealed, server-side state, the terminal score can't be forged through the command surface.

Each episode gets its own throwaway SQLite database (one company per DB, matching yc-bench's
single-simulation-per-DB model), seeded from the task's seed on ``begin_session`` (same
business attributes every run; upstream mints fresh ``uuid4`` row ids, which reach the agent in
``sim resume`` wake events and break ties between simultaneous events) and torn down on
``end_session``. State is keyed by ``_session_id`` (shogym
injects it), so concurrent episodes are isolated. All ``yc_bench`` imports are funnelled
through :mod:`shogym.envs.yc_bench.adapter`, so importing this module requires the ``yc_bench``
extra — but it is only imported when a ``yc_bench`` env is constructed or served.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from shogym.envs.yc_bench import adapter

RUN_COMMAND_TOOL_NAME = "run_command"
SUBMIT_TOOL_NAME = "submit"

# Default per-command wall-clock budget for a yc-bench CLI subprocess. The deterministic sim
# commands are fast; this only guards against a pathological hang.
DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0

server: FastMCP = FastMCP(name="yc_bench")

# session_id -> _YcSession
_sessions: Dict[str, "_YcSession"] = {}
# Guards the `_sessions` map and serializes teardown. Command execution itself is a subprocess
# with an explicit env, so it needs no global lock.
_lock = threading.RLock()


class _YcSession:
    """One yc-bench simulation: a private SQLite DB seeded for a single company/world."""

    def __init__(
        self,
        *,
        seed: int,
        config_name: str,
        start_date: str,
        horizon_years: Optional[int],
        company_name: str,
        command_timeout_seconds: float,
    ) -> None:
        self.config_name = config_name
        self.command_timeout_seconds = command_timeout_seconds
        self._tmpdir = Path(tempfile.mkdtemp(prefix="shogym-ycbench-"))
        # yc-bench auto-creates parent dirs for relative sqlite:/// paths; use an absolute
        # sqlite://// URL so the DB lands in this session's private temp dir regardless of cwd.
        self.db_url = f"sqlite:////{self._tmpdir.as_posix().lstrip('/')}/yc_bench.db"
        self._engine, self._factory = adapter.build_db(self.db_url)
        self.company_id = adapter.seed_session(
            self._factory,
            seed=seed,
            config_name=config_name,
            start_date=start_date,
            horizon_years=horizon_years,
            company_name=company_name,
        )

    def run_command(self, command: str) -> Dict[str, Any]:
        # The per-command budget is the env-configured maximum, never caller-controllable — a
        # model can't request a huge timeout to bypass the hang safeguard.
        return adapter.run_cli(
            command,
            db_url=self.db_url,
            config_name=self.config_name,
            timeout_seconds=float(self.command_timeout_seconds),
            base_env=dict(os.environ),
        )

    def verdict(self) -> Dict[str, Any]:
        return adapter.read_final_state(self._factory)

    def teardown(self) -> None:
        try:
            self._engine.dispose()
        except Exception:
            pass
        shutil.rmtree(self._tmpdir, ignore_errors=True)


# ----- session lifecycle (called in-process by the env) -----


def begin_session(
    session_id: str,
    *,
    seed: int,
    config_name: str,
    start_date: str,
    horizon_years: Optional[int],
    company_name: str,
    command_timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> None:
    """Create + seed a fresh yc-bench simulation for this episode. Idempotent per id."""
    session = _YcSession(
        seed=seed,
        config_name=config_name,
        start_date=start_date,
        horizon_years=horizon_years,
        company_name=company_name,
        command_timeout_seconds=command_timeout_seconds,
    )
    with _lock:
        old = _sessions.pop(session_id, None)
        _sessions[session_id] = session
    if old is not None:
        old.teardown()


def end_session(session_id: str) -> None:
    """Tear down a finished episode's simulation (drop the DB). Idempotent."""
    with _lock:
        session = _sessions.pop(session_id, None)
    if session is not None:
        session.teardown()


def reset_state() -> None:
    """Drop all sessions (test hygiene)."""
    with _lock:
        ids = list(_sessions)
    for session_id in ids:
        end_session(session_id)


def _session_for(session_id: Optional[str]) -> Optional["_YcSession"]:
    if session_id is None:
        return None
    with _lock:
        return _sessions.get(session_id)


# The unseeded fallback verdict — a session that was never begun (or already torn down) has no
# sim state to read, so it reports the safe, non-terminal all-zero end-state.
_UNSEEDED_VERDICT: Dict[str, Any] = {
    "seeded": False,
    "survived": False,
    "final_funds_cents": 0,
    "tasks_succeeded": 0,
    "tasks_failed": 0,
    "horizon_reached": False,
    "terminal_reason": None,
    "sim_time": None,
    "horizon_end": None,
}


def read_verdict(session_id: str) -> Dict[str, Any]:
    """Read the authoritative terminal metrics off the **live** sim DB for ``session_id``.

    This is the scoring read the env's ``finalize`` hook performs on the already-sealed
    episode. It must run while the session — and its SQLite engine — is still live; the serve
    layer disposes the session (``end_session`` → ``engine.dispose()``) only *after* ``finalize``
    returns, so the read here always sees an open DB. A missing session (never begun, or already
    torn down) yields the unseeded, non-terminal fallback rather than raising."""
    session = _session_for(session_id)
    if session is None:
        return dict(_UNSEEDED_VERDICT)
    return session.verdict()


# ----- MCP tools -----


@server.tool
def run_command(command: str, _session_id: str) -> str:
    """Execute one yc-bench CLI command and return its JSON result.

    ``command`` must be a full ``yc-bench <subcommand …>`` string (e.g.
    ``"yc-bench company status"``, ``"yc-bench sim resume"``). All observe / task / sim /
    memory commands are issued through this one tool. The returned JSON has ``ok``,
    ``exit_code``, ``stdout`` (the command's JSON output), ``stderr``, and ``command``.
    """
    session = _session_for(_session_id)
    if session is None:
        return json.dumps(
            {
                "ok": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "session not initialized; env did not call begin_session",
                "command": command,
            }
        )
    return json.dumps(session.run_command(command))


@server.tool
def submit(_session_id: str) -> str:
    """End the episode and record your final result.

    Calling ``submit`` finishes the run — there is no separate stop step. shogym seals the
    episode, reads the authoritative final metrics off the sim DB (``survived`` (funds ≥ 0),
    ``final_funds_cents``, ``tasks_succeeded`` / ``tasks_failed``, ``horizon_reached``,
    ``terminal_reason``), scores it, and ends the episode. Call it once the run is over
    (``sim resume`` reported bankruptcy or horizon end, or you are stopping). A solvent
    submission *before* the sim has actually ended scores zero.
    """
    # The serve layer intercepts a `submit` call as the `score` terminal (validate → seal →
    # finalize) and never dispatches to this handler; the scoring read lives in the env's
    # `finalize` hook via `read_verdict`. This body is retained only so a direct, non-served
    # invocation still returns the honest end-state.
    return json.dumps(read_verdict(_session_id))


__all__ = [
    "server",
    "begin_session",
    "end_session",
    "read_verdict",
    "reset_state",
    "RUN_COMMAND_TOOL_NAME",
    "SUBMIT_TOOL_NAME",
]
