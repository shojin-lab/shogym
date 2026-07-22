"""The environment interface (RFC 008): describe, serve, verify.

An hgym environment is not a gym — it has no ``reset``/``step`` loop and no agent. It
**describes** a task, **serves** its essential tools as MCP servers, and **verifies** a
recorded trajectory. A harness (external) drives the tools; the env only publishes the
contract and scores what happened.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from hgym.mcp.types import MCPServerSpec
from hgym.task import TaskSpec
from hgym.trajectory import Trajectory
from hgym.types import FeedbackCollection


class Env(ABC):
    """Abstract environment: describe + serve (MCP specs) + verify."""

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
        self, trajectory: Trajectory, task: Dict[str, Any], *, terminated: bool
    ) -> FeedbackCollection:
        """Score the recorded trajectory. Pure — no side effects, no env state read."""

    async def close(self) -> None:
        """Release any resources. Default: no-op."""

    @property
    def name(self) -> str:
        return getattr(self, "_registered_name", type(self).__name__)

    @property
    def horizon(self) -> Optional[int]:
        return getattr(self, "_horizon", None)
