"""Serving an environment to an external harness (RFC 008 §3.2, §5).

:class:`ServedEpisode` is the transport-independent engine: it opens the env's essential
MCP sessions, turns one incoming tool call into one recorded step, and returns the tool's
result plus the feedback ``_meta`` sidecar. :func:`~hgym.serve.server.run_stdio` wraps it in
a FastMCP stdio server (`hgym serve`); the engine itself is exercised in-process, no
subprocess required.
"""

from hgym.serve.episode import CallResult, ServedEpisode
from hgym.serve.lifecycle import (
    FinalizationRecord,
    FinalizationStore,
    FinalizeRequest,
    LifecycleState,
    TerminalEvidence,
)

__all__ = [
    "CallResult",
    "FinalizationRecord",
    "FinalizationStore",
    "FinalizeRequest",
    "LifecycleState",
    "ServedEpisode",
    "TerminalEvidence",
]
