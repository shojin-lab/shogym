"""Serving an environment to an external harness.

:class:`ServedEpisode` is the transport-independent engine: it opens the env's essential
MCP sessions, turns one incoming tool call into one recorded step, and returns the tool's
result plus the feedback ``_meta`` sidecar. :func:`shogym.serve.protocol_v2.gateway.run_stdio_v2`
wraps it in the durable stream that ``shogym serve`` runs; the engine itself is exercised
in-process, no subprocess required.

:mod:`shogym.serve.v1_runs` reads the run directories the retired version one serving path
wrote. It reads them and does nothing else: there is no writer, and no directory it reads can
be served or resumed.
"""

from shogym.serve.episode import CallResult, ServedEpisode
from shogym.serve.lifecycle import (
    FinalizationRecord,
    FinalizationStore,
    FinalizeRequest,
    LifecycleState,
    TerminalEvidence,
)
from shogym.serve.v1_runs import (
    Closure,
    ResultRow,
    Score,
    read_dispenses,
    read_results,
    reconcile,
)

__all__ = [
    "CallResult",
    "Closure",
    "FinalizationRecord",
    "FinalizationStore",
    "FinalizeRequest",
    "LifecycleState",
    "ResultRow",
    "Score",
    "ServedEpisode",
    "TerminalEvidence",
    "read_dispenses",
    "read_results",
    "reconcile",
]
