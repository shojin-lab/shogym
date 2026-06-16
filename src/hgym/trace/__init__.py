"""The trace store (RFC 007): append-only JSONL rollouts tagged with harness hashes.

- :class:`TraceRecord` / :func:`record_for` — a row stamped with the harness hashes.
- :func:`append_trace` / :func:`load_traces` — write and read the JSONL store.
- :func:`load_rows` / :func:`flatten_record` — dataframe-ready flattened rows.
"""

from hgym.trace._store import (
    TraceRecord,
    append_trace,
    flatten_record,
    load_rows,
    load_traces,
    record_for,
)

__all__ = [
    "TraceRecord",
    "append_trace",
    "flatten_record",
    "load_rows",
    "load_traces",
    "record_for",
]
