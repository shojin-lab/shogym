"""Append-only JSONL trace store (RFC 008 §4/§8).

One :class:`TraceRecord` per step. The feedback is stored in the same wire form it takes
on the MCP ``_meta`` sidecar (:mod:`hgym.feedback`), so the trace and the in-band signal
never diverge. No surface hashing — hgym does not own the harness's surfaces, so there is
nothing per-surface to hash.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from hgym.feedback.wire import FeedbackItem, dump_item


@dataclass
class TraceRecord:
    """One step's row: which session, on which env/task, what tool was called, and the
    feedback produced (serialized feedback items). ``terminated`` marks the row that
    closed the episode (terminate tool or horizon)."""

    session_id: str
    env_name: str
    task_id: Optional[str]
    step: int
    tool: Optional[str] = None
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    terminated: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


def step_record(
    *,
    session_id: str,
    env_name: str,
    task_id: Optional[str],
    step: int,
    tool: Optional[str] = None,
    feedback: Sequence[FeedbackItem] = (),
    terminated: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> TraceRecord:
    """Build a :class:`TraceRecord`, serializing feedback items to their wire form."""
    return TraceRecord(
        session_id=session_id,
        env_name=env_name,
        task_id=task_id,
        step=step,
        tool=tool,
        feedback=[dump_item(item) for item in feedback],
        terminated=terminated,
        extra=dict(extra or {}),
    )


def append_trace(path: Union[str, Path], record: TraceRecord) -> None:
    """Append one record to the JSONL store at ``path`` (parents created if absent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def load_traces(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read the JSONL store into a list of dicts (blank lines skipped)."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
