"""Feedback-on-the-wire (RFC 008 §4): how an env returns feedback to a harness it does
not own, and the visibility rule that keeps evals honest.

Feedback rides a reserved ``_meta`` namespace on MCP tool results — distinct from the
tool's functional ``content`` — so a harness that ignores ``_meta`` still runs and is
still fully scored (the env also records every item to the trace, see :mod:`hgym.trace`).
The value type is the existing ``float | bool | text`` from :mod:`hgym.types`.
"""

from hgym.feedback.wire import (
    FEEDBACK_META_KEY,
    TERMINATE_META_KEY,
    build_meta,
    dump_item,
    parse_meta,
    select_inband,
)

__all__ = [
    "FEEDBACK_META_KEY",
    "TERMINATE_META_KEY",
    "build_meta",
    "dump_item",
    "parse_meta",
    "select_inband",
]
