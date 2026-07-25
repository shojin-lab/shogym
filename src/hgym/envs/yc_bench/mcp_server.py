# yc-bench is an optional extra (the `yc_bench` install group); it is intentionally absent
# from the base type-check / offline environment, so its imports are expected to be
# unresolved there.
# pyright: reportMissingImports=false
"""In-process MCP server for the ``yc_bench`` env — yc-bench's command surface, wrapped.

yc-bench has **no built-in agent loop**: it expects an external driver to issue CLI commands
against its deterministic sim, feed the JSON results back, and collect the next commands. hgym
*is* that driver. This server exposes that command surface as two MCP tools:

  - **``run_command(command)``** — the faithful mirror of upstream's
    ``run_command("yc-bench <cmd>")``: it runs one yc-bench CLI command against *this
    session's* SQLite sim and returns the CLI's JSON. Every observe/act/sim/memory command is
    reached through this one tool (the most faithful surface per issue #32).
  - **``submit()``** — an hgym terminal tool that reads the authoritative final metrics
    (survival, final funds, task outcomes) off the sim DB and returns a *marked* verdict. The
    env's pure ``verify`` trusts only this step, so the terminal score can't be forged through
    the command surface.

Each episode gets its own throwaway SQLite database (one company per DB, matching yc-bench's
single-simulation-per-DB model), seeded deterministically from the task's seed on
``begin_session`` and torn down on ``end_session``. State is keyed by ``_session_id`` (hgym
injects it), so concurrent episodes are isolated. All ``yc_bench`` imports are funnelled
through :mod:`hgym.envs.yc_bench.adapter`, so importing this module requires the ``yc_bench``
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

from hgym.envs.yc_bench import adapter

# Marker key stamped on the `submit` verdict so the pure verifier can find it in the recorded
# trajectory without trusting arbitrary command output.
VERDICT_MARKER = "yc_bench_verdict"
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
        self._tmpdir = Path(tempfile.mkdtemp(prefix="hgym-ycbench-"))
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
        state = adapter.read_final_state(self._factory)
        return {VERDICT_MARKER: True, **state}

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
    """Finish the episode: read the sim's final metrics and return the terminal verdict.

    Reports ``survived`` (funds ≥ 0), ``final_funds_cents``, ``tasks_succeeded`` /
    ``tasks_failed``, ``horizon_reached``, and ``terminal_reason`` — read straight off the sim
    DB. Call this once the run is over (``sim resume`` reported bankruptcy or horizon end, or
    you are stopping), then call ``terminate`` to end the hgym episode.
    """
    session = _session_for(_session_id)
    if session is None:
        return json.dumps(
            {
                VERDICT_MARKER: True,
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
        )
    return json.dumps(session.verdict())


__all__ = [
    "server",
    "begin_session",
    "end_session",
    "reset_state",
    "VERDICT_MARKER",
    "RUN_COMMAND_TOOL_NAME",
    "SUBMIT_TOOL_NAME",
]
