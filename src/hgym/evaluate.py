"""Harness-agnostic evaluation (RFC 008 §7): hold ``(env, task)`` fixed, drive it with a
harness, read the terminal feedback off the trace.

:func:`evaluate` runs an **in-process** harness (an async callable that receives a FastMCP
``Client`` connected to the served env) — the path the example harness and the tests use.
An **external** harness (Claude Code, ...) instead spawns ``hgym serve`` itself and writes
the same JSONL trace; :func:`result_from_trace` turns that file into the same
:class:`EvalResult`, so both paths converge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

from fastmcp import Client

from hgym.serve.episode import ServedEpisode
from hgym.serve.server import build_server
from hgym.trace import load_traces

Harness = Callable[[Client], Awaitable[None]]


@dataclass
class EvalResult:
    """The outcome of one evaluated episode: the terminal feedback (wire form) plus the
    identity needed to attribute it."""

    env: str
    task: Optional[str]
    terminated: bool
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    trace_path: Optional[str] = None

    def value(self, name: str) -> Optional[Any]:
        """The terminal feedback value for ``name`` (e.g. ``"check_answer"``), or None."""
        for item in self.feedback:
            if item.get("name") == name:
                return item.get("value")
        return None


async def evaluate(
    env: str,
    *,
    harness: Harness,
    task: Optional[Union[int, str]] = None,
    trace_path: Optional[Union[str, Path]] = None,
    env_config: Optional[Dict[str, Any]] = None,
) -> EvalResult:
    """Serve ``env`` at ``task``, drive it with ``harness`` (in-process), return the
    terminal feedback read back off the trace."""
    tmp_trace = Path(trace_path) if trace_path is not None else None
    episode = await ServedEpisode.start(env, task=task, trace_path=tmp_trace, env_config=env_config)
    try:
        # Inside the guard: build_server can raise (e.g. a reserved-name collision), and
        # the episode already holds open sessions + pushed state that must be released.
        server = build_server(episode)
        async with Client(server) as client:
            await harness(client)
    finally:
        await episode.close()

    task_id = episode.describe().task_id
    if tmp_trace is not None and tmp_trace.exists():
        # Scope to this episode: the trace store is append-only, so a reused path may
        # hold prior runs whose terminal row would otherwise supply this result.
        return result_from_trace(
            tmp_trace, env=env, task=task_id, session_id=episode.session_id
        )
    # No trace file: report the terminal feedback the episode retained in memory, so the
    # default public API (README quickstart calls `evaluate(...)` without `trace_path`)
    # still surfaces the terminal score rather than an empty feedback list.
    return EvalResult(
        env=env,
        task=task_id,
        terminated=episode.terminated,
        feedback=list(episode.terminal_feedback),
    )


def result_from_trace(
    trace_path: Union[str, Path],
    *,
    env: Optional[str] = None,
    task: Optional[str] = None,
    session_id: Optional[str] = None,
) -> EvalResult:
    """Build an :class:`EvalResult` from a JSONL trace — the path an external harness (it
    spawned ``hgym serve`` and wrote the trace) shares with the in-process path.

    ``env`` / ``task`` / ``session_id`` are **filters**, not just labels: each, when given,
    restricts the rows considered before the terminal row is chosen, so a shared,
    append-only trace can't let another run supply a stale result. ``session_id`` is the
    only *unique* per-run scope (the in-process path passes it). The external path has no
    session id, so it scopes by ``env``/``task`` and then, since those aren't unique across
    repeat runs of the same task, defaults to the **most recent** matching episode — right
    for a reused trace, but for a guaranteed 1:1 mapping give each run its own trace file.
    """
    rows = load_traces(trace_path)
    if env is not None:
        rows = [r for r in rows if r.get("env_name") == env]
    if task is not None:
        rows = [r for r in rows if r.get("task_id") == task]
    if session_id is None and rows:
        # No explicit session: scope to the last episode in the (env/task-filtered)
        # trace so a reused file reports the latest run, not an older terminal row.
        session_id = rows[-1].get("session_id")
    if session_id is not None:
        rows = [r for r in rows if r.get("session_id") == session_id]
    terminal = next((r for r in reversed(rows) if r.get("terminated")), None)
    last = terminal or (rows[-1] if rows else None)
    return EvalResult(
        env=env or (last["env_name"] if last else ""),
        task=task if task is not None else (last.get("task_id") if last else None),
        terminated=bool(terminal),
        feedback=list(terminal["feedback"]) if terminal else [],
        trace_path=str(trace_path),
    )
