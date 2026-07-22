"""Harness-agnostic evaluation (RFC 008 §7): hold ``(env, task)`` fixed, drive it with a
harness, read the terminal feedback off the trace.

:func:`evaluate` runs an **in-process** harness (an async callable that receives a FastMCP
``Client`` connected to the served env) — the path the example harness and the tests use.
An **external** harness (Claude Code, Codex, ...) instead spawns ``hgym serve`` itself and
writes the same JSONL trace; :func:`result_from_trace` turns that file into the same
:class:`EvalResult`, so both paths converge on one result type.
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
    identity needed to attribute it. ``feedback`` is the terminal row's feedback items."""

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
    episode = await ServedEpisode.start(
        env, task=task, trace_path=tmp_trace, env_config=env_config
    )
    server = build_server(episode)
    try:
        async with Client(server) as client:
            await harness(client)
    finally:
        await episode.close()

    if tmp_trace is not None and tmp_trace.exists():
        return result_from_trace(tmp_trace, env=env, task=episode.describe().task_id)
    # No trace file: report only what the engine knows (terminated or not).
    return EvalResult(
        env=env, task=episode.describe().task_id, terminated=episode.terminated
    )


def result_from_trace(
    trace_path: Union[str, Path],
    *,
    env: Optional[str] = None,
    task: Optional[str] = None,
) -> EvalResult:
    """Build an :class:`EvalResult` from a JSONL trace — the path an external harness (it
    spawned ``hgym serve`` and wrote the trace) shares with the in-process path."""
    rows = load_traces(trace_path)
    terminal = next((r for r in reversed(rows) if r.get("terminated")), None)
    last = terminal or (rows[-1] if rows else None)
    return EvalResult(
        env=env or (last["env_name"] if last else ""),
        task=task if task is not None else (last.get("task_id") if last else None),
        terminated=bool(terminal),
        feedback=list(terminal["feedback"]) if terminal else [],
        trace_path=str(trace_path),
    )
