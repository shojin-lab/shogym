"""The trace store (RFC 008 §4/§8): append-only JSONL, one row per step, tagged with the
env, task, and the feedback produced — the out-of-band record the experimenter reads.

Coarser than a harness-owned trace by design: shogym does not own the harness's internal
surfaces, so the row records what shogym *can* attribute — ``(session_id, env, task_id,
step, tool, feedback)`` — and leaves the harness's internals to the harness.
"""

from shogym.trace._store import (
    TerminalEvent,
    TraceRecord,
    append_terminal_event,
    append_trace,
    load_traces,
    step_record,
    terminal_event_record,
)

__all__ = [
    "TerminalEvent",
    "TraceRecord",
    "append_terminal_event",
    "append_trace",
    "load_traces",
    "step_record",
    "terminal_event_record",
]
