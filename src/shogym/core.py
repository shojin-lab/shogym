"""The environment interface (RFC 008): describe, serve, verify.

An shogym environment is not a gym — it has no ``reset``/``step`` loop and no agent. It
**describes** a task, **serves** its essential tools as MCP servers, and **verifies** a
recorded trajectory. A harness (external) drives the tools; the env only publishes the
contract and scores what happened.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from shogym.mcp.types import MCPServerSpec
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import FeedbackCollection

if TYPE_CHECKING:
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence


class Env(ABC):
    """Abstract environment: describe + serve (MCP specs) + verify."""

    # The optional, typed terminal-transaction hook. Default ``None`` means the env has no
    # scoring finalizer (a pure-verify / abort-only env): the serve layer never engages the
    # seal transaction for it and it behaves exactly as before. An env opts in by declaring a
    # ``score`` terminal tool *and* overriding this with
    # ``async def finalize(self, req: FinalizeRequest) -> TerminalEvidence`` — the serve layer
    # runs it on the already-sealed episode to produce the trusted terminal evidence.
    finalize: Optional[
        Callable[["FinalizeRequest"], Awaitable["TerminalEvidence"]]
    ] = None

    @abstractmethod
    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        """Publish the task contract a harness reads to configure itself."""

    @abstractmethod
    def essential_specs(self) -> List[MCPServerSpec]:
        """The MCP servers to serve: the reserved ``terminate`` tool + env-mandatory tools."""

    @abstractmethod
    def load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        """Select a task instance (deterministic when ``task_idx`` is given)."""

    def begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        """Push per-episode state into the (in-process) tool servers. Default: no-op."""

    def end_session(self, session_id: str) -> None:
        """Drop the per-episode state ``begin_session`` created. Default: no-op.

        Symmetric with :meth:`begin_session`. ``close()`` invokes it for any
        session still open, so a stateful in-process tool server does not leak an
        entry per episode.
        """

    @abstractmethod
    def verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: "Optional[TerminalEvidence]" = None,
    ) -> FeedbackCollection:
        """Score the recorded trajectory. Pure — no side effects, no env state read.

        A ``score``-terminal env additionally receives core-owned, immutable ``evidence``
        (the trusted verdict from ``finalize``) and scores from it instead of scanning the
        trajectory for marker JSON. Non-score envs are always called with ``evidence=None``
        and behave exactly as before."""

    async def close(self) -> None:
        """Release any resources. Default: no-op."""

    @property
    def name(self) -> str:
        return getattr(self, "_registered_name", type(self).__name__)

    @property
    def horizon(self) -> Optional[int]:
        return getattr(self, "_horizon", None)
