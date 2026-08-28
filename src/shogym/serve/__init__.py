"""Serving an environment to an external harness (RFC 008 §3.2, §5).

:class:`ServedEpisode` is the transport-independent engine: it opens the env's essential
MCP sessions, turns one incoming tool call into one recorded step, and returns the tool's
result plus the feedback ``_meta`` sidecar. :func:`~shogym.serve.server.run_stdio` wraps it in
a FastMCP stdio server (`shogym serve`); the engine itself is exercised in-process, no
subprocess required.
"""

from shogym.serve.episode import CallResult, ServedEpisode
from shogym.serve.lifecycle import (
    FinalizationRecord,
    FinalizationStore,
    FinalizeRequest,
    LifecycleState,
    TerminalEvidence,
)
from shogym.serve.stream import (
    Closure,
    CompletedTask,
    DispensedTask,
    EvalStream,
    FeedbackPolicy,
    Immediate,
    Information,
    Never,
    Placebo,
    Provenance,
    ProvenanceError,
    ProvenanceSpan,
    QueueInfo,
    ResultRow,
    RunIdentity,
    Score,
    TaskRef,
    TaskStream,
    build_stream_server,
    read_adoptions,
    read_dispenses,
    read_exposures,
    read_results,
    reconcile,
)

__all__ = [
    "CallResult",
    "Closure",
    "CompletedTask",
    "DispensedTask",
    "EvalStream",
    "FeedbackPolicy",
    "FinalizationRecord",
    "FinalizationStore",
    "FinalizeRequest",
    "Immediate",
    "Information",
    "LifecycleState",
    "Never",
    "Placebo",
    "Provenance",
    "ProvenanceError",
    "ProvenanceSpan",
    "QueueInfo",
    "ResultRow",
    "RunIdentity",
    "Score",
    "ServedEpisode",
    "TaskRef",
    "TaskStream",
    "TerminalEvidence",
    "build_stream_server",
    "read_adoptions",
    "read_dispenses",
    "read_exposures",
    "read_results",
    "reconcile",
]
