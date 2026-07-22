"""The trace store (RFC 008 §4/§8): append-only JSONL, one row per step, tagged with the
env, task, and the feedback produced — the out-of-band record the experimenter reads.

Deliberately coarser than a harness-owned trace: hgym does not own the harness's internal
surfaces (its context strategy, its prompt, its retries), so the row records what hgym
*can* attribute — ``(session_id, env, task_id, step, tool, feedback)`` — and leaves the
harness's internals to the harness. Zero infra: one JSON object per line, append-safe.
"""

from hgym.trace._store import TraceRecord, append_trace, load_traces, step_record

__all__ = ["TraceRecord", "append_trace", "load_traces", "step_record"]
