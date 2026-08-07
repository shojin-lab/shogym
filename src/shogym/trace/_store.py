"""Append-only JSONL trace store (RFC 008 §4/§8).

One :class:`TraceRecord` per step. Feedback is stored in the same wire form it takes on the
MCP ``_meta`` sidecar (:mod:`shogym.feedback`), so the trace and the in-band signal never
diverge. No surface hashing — shogym does not own the harness's surfaces.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from shogym.feedback.wire import FeedbackItem, _load_item, dump_item
from shogym.types import InferenceFeedback


def _require_str(name: str, value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str):
        suffix = " or None" if optional else ""
        raise ValueError(f"TraceRecord.{name} must be a string{suffix}, got {value!r}")


@dataclass
class TraceRecord:
    """One step's row: which session, on which env/task, what tool was called, and the
    feedback produced. ``terminated`` marks the row that closed the episode."""

    session_id: str
    env_name: str
    task_id: Optional[str]
    step: int
    tool: Optional[str] = None
    feedback: List[Dict[str, Any]] = field(default_factory=list)
    terminated: bool = False

    def __post_init__(self) -> None:
        # Fail fast at construction. Re-checked at append time too: this is a
        # mutable public dataclass, so a valid record can be mutated afterwards.
        _validate_trace_record(self)


def _validate_trace_record(record: "TraceRecord") -> None:
    """Enforce the JSONL wire schema so an invalid row can never be persisted,
    regardless of who built it or whether it was mutated after construction."""
    _require_str("session_id", record.session_id)
    _require_str("env_name", record.env_name)
    _require_str("task_id", record.task_id, optional=True)
    _require_str("tool", record.tool, optional=True)
    # bool is an int subclass; the field is a real integer step, not True/False.
    if isinstance(record.step, bool) or not isinstance(record.step, int):
        raise ValueError(f"TraceRecord.step must be an int, got {record.step!r}")
    if not isinstance(record.terminated, bool):
        raise ValueError(f"TraceRecord.terminated must be a boolean, got {record.terminated!r}")
    if not isinstance(record.feedback, list):
        raise ValueError(f"TraceRecord.feedback must be a list, got {record.feedback!r}")
    for item in record.feedback:  # each item must be valid feedback wire
        loaded = _load_item(item)
        # One row per tool call: inference feedback is the signal *for that step*, so its
        # step must match the row's — a mismatch would silently misattribute the signal.
        if isinstance(loaded, InferenceFeedback) and loaded.step != record.step:
            raise ValueError(
                f"TraceRecord.feedback: inference step {loaded.step} does not match "
                f"row step {record.step}"
            )


def step_record(
    *,
    session_id: str,
    env_name: str,
    task_id: Optional[str],
    step: int,
    tool: Optional[str] = None,
    feedback: Sequence[FeedbackItem] = (),
    terminated: bool = False,
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
    )


@dataclass
class TerminalEvent:
    """The versioned, core-owned ``terminal`` event, appended as a **distinct** row
    (``kind="terminal"``) after the final ``step`` row.

    A step row carries no ``kind`` (readers default an absent ``kind`` to ``"step"``), so this
    additive event never changes the serialization of existing rows: a reader that doesn't know
    about ``terminal`` events ignores unknown kinds. Confidential diagnostics/provenance never
    appear here — they live only in the private finalization store, keyed by
    ``(session_id, finalization_id)``. This row carries the ``args_digest`` (a hash, never the
    raw args) and the public-safe, core-stamped ``verdict``."""

    session_id: str
    env_name: str
    task_id: Optional[str]
    step: int
    source: str
    status: str
    verdict: Dict[str, Any]
    finalization_id: str
    args_digest: Optional[str] = None
    schema_version: int = 1
    kind: str = "terminal"

    def __post_init__(self) -> None:
        _require_str("session_id", self.session_id)
        _require_str("env_name", self.env_name)
        _require_str("task_id", self.task_id, optional=True)
        _require_str("source", self.source)
        _require_str("status", self.status)
        _require_str("finalization_id", self.finalization_id)
        _require_str("args_digest", self.args_digest, optional=True)
        if isinstance(self.step, bool) or not isinstance(self.step, int):
            raise ValueError(f"TerminalEvent.step must be an int, got {self.step!r}")
        if not isinstance(self.verdict, dict):
            raise ValueError(f"TerminalEvent.verdict must be a dict, got {self.verdict!r}")


def terminal_event_record(
    *,
    session_id: str,
    env_name: str,
    task_id: Optional[str],
    step: int,
    source: str,
    status: str,
    verdict: Dict[str, Any],
    finalization_id: str,
    args_digest: Optional[str] = None,
) -> TerminalEvent:
    return TerminalEvent(
        session_id=session_id,
        env_name=env_name,
        task_id=task_id,
        step=step,
        source=source,
        status=status,
        verdict=dict(verdict),
        finalization_id=finalization_id,
        args_digest=args_digest,
    )


def append_terminal_event(path: Union[str, Path], event: TerminalEvent) -> None:
    """Append one ``terminal`` event to the JSONL store (same file as the step rows)."""
    event.__post_init__()  # re-validate at the write boundary (mutable public dataclass)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(event), sort_keys=True, allow_nan=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_trace(path: Union[str, Path], record: TraceRecord) -> None:
    """Append one record to the JSONL store at ``path`` (parents created if absent)."""
    # Re-validate at the write boundary: the record may have been mutated between
    # construction and now (it is a mutable public dataclass with a mutable
    # `feedback` list), and only a schema-valid row may reach the file.
    _validate_trace_record(record)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Every field is a validated primitive and feedback finiteness is already
    # enforced above; allow_nan=False is a cheap final guard that the row stays
    # valid JSON (no NaN/Infinity tokens).
    line = json.dumps(asdict(record), sort_keys=True, allow_nan=False)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_traces(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read the JSONL store into a list of dicts (blank lines skipped)."""
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
