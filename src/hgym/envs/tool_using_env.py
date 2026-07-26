"""``ToolUsingEnv`` — the concrete base for envs whose tools are MCP servers (RFC 008).

A subclass declares its MCP servers, its advisory instruction templates, and its horizon,
and implements three hooks: ``_load_task`` (pick an instance), ``_begin_session`` (push
per-episode state into the in-process tool servers), and ``_verify`` (score the recorded
:class:`~hgym.trajectory.Trajectory`). The base probes the tool manifest at construction so
``describe()`` can publish it; it owns no loop — a harness drives the tools via ``serve``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from abc import abstractmethod
from typing import Any, Awaitable, Dict, List, Optional, Sequence, TypeVar

from hgym.core import Env
from hgym.mcp import MCPServerSpec
from hgym.mcp.toolset import _open_session_for_spec
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from hgym.task import ReferenceTemplate, TaskSpec, TerminalKind, ToolManifest
from hgym.trajectory import Trajectory
from hgym.types import FeedbackCollection, FunctionConfig, ToolConfig
from hgym.utils import seeding

_TERMINATE_SPEC = MCPServerSpec(
    name="__terminate__",
    transport="in_process",
    module="hgym.shared.terminate_mcp",
)

_T = TypeVar("_T")


class ToolUsingEnv(Env):
    # Subclasses override these.
    mcp_servers: Sequence[MCPServerSpec] = ()
    function: FunctionConfig = FunctionConfig()
    function_name: str = "agent"
    # RFC 009: the single tool (if any) this env marks as the `score` terminal. When set,
    # ``describe()`` advertises it with ``terminal_kind="score"`` and the serve layer runs
    # its call as a validate -> seal -> evaluate transaction. Left ``None`` by every env but
    # HLE, so all other envs advertise only `none`/`abort` tools and behave exactly as
    # before (the seal transaction never engages).
    score_terminal_tool: Optional[str] = None

    def __init__(self, *, horizon: int, num_tasks: Optional[int] = None) -> None:
        self._horizon = horizon
        self._num_tasks = num_tasks
        self._registered_name = type(self).__name__
        self._np_random = None
        self._open_session_ids: set[str] = set()
        # Probe the essential servers once for the tool manifest describe() publishes.
        self._tool_configs: Dict[str, ToolConfig] = self._probe_manifest()

    # ----- describe / serve -----

    def essential_specs(self) -> List[MCPServerSpec]:
        """The reserved terminate server plus the env-mandatory servers."""
        return [_TERMINATE_SPEC, *self.mcp_servers]

    def _probe_manifest(self) -> Dict[str, ToolConfig]:
        tools: Dict[str, ToolConfig] = {}
        for spec in self.essential_specs():
            for tc in _sync_run_async(_probe_tool_configs(spec)):
                if tc.name in tools:
                    raise ValueError(f"duplicate tool name {tc.name!r} across mcp_servers")
                tools[tc.name] = tc
        # RFC 009: exactly zero or one `score` terminal per env, and it must be a real
        # advertised tool (never the reserved `terminate`). Fail fast at construction so a
        # typo can't silently leave the env with no scoring terminal.
        score = self.score_terminal_tool
        if score is not None:
            if score not in tools:
                raise ValueError(
                    f"score_terminal_tool {score!r} is not an advertised tool "
                    f"({sorted(tools)})"
                )
            if score == TERMINATE_TOOL_NAME:
                raise ValueError(
                    f"score_terminal_tool may not be the reserved {TERMINATE_TOOL_NAME!r}"
                )
        return tools

    def _terminal_kind(self, tool_name: str) -> TerminalKind:
        """The RFC-009 terminal role of ``tool_name``: the reserved ``terminate`` is
        ``abort``, this env's ``score_terminal_tool`` (if any) is ``score``, everything
        else is ``none``."""
        if tool_name == TERMINATE_TOOL_NAME:
            return "abort"
        if tool_name == self.score_terminal_tool:
            return "score"
        return "none"

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        """Publish the task contract (RFC 008 §3.1). Read-only; opens no session."""
        fc = self.function
        reference_templates: List[ReferenceTemplate] = []
        if fc.example_system_template is not None:
            reference_templates.append(
                ReferenceTemplate(
                    role="system",
                    template=fc.example_system_template,
                    variables_schema=_schema_json(fc.system_schema),
                )
            )
        if fc.example_user_template is not None:
            reference_templates.append(
                ReferenceTemplate(
                    role="user",
                    template=fc.example_user_template,
                    variables_schema=_schema_json(fc.user_schema),
                )
            )

        tools = [
            ToolManifest(
                name=name,
                description=tc.description,
                input_schema=tc.parameters.model_dump(),
                provenance="reserved" if name == TERMINATE_TOOL_NAME else "env-mandatory",
                terminal_kind=self._terminal_kind(name),
            )
            for name, tc in self._tool_configs.items()
        ]

        return TaskSpec(
            env_name=self.name,
            task_id=task_id,
            instructions=_render_static(fc.example_system_template),
            tools=tools,
            reference_templates=reference_templates,
            horizon=self._horizon,
        )

    # ----- task lifecycle -----

    def load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is not None:
            self._np_random, _ = seeding.np_random(task_idx)
        return self._load_task(task_idx)

    def begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        self._open_session_ids.add(session_id)
        self._begin_session(session_id, task)

    def end_session(self, session_id: str) -> None:
        self._end_session(session_id)
        self._open_session_ids.discard(session_id)

    def verify(
        self, trajectory: Trajectory, task: Dict[str, Any], *, terminated: bool
    ) -> FeedbackCollection:
        return self._verify(trajectory, task, terminated=terminated)

    async def close(self) -> None:
        # Tear down every session still open (an env may serve several concurrent
        # episodes) so a stateful in-process tool server doesn't leak per-episode
        # entries `begin_session` created. Snapshot first — end_session mutates the set.
        for session_id in list(self._open_session_ids):
            self.end_session(session_id)
        await self._close()

    @property
    def np_random(self):
        if self._np_random is None:
            self._np_random, _ = seeding.np_random()
        return self._np_random

    @property
    def num_tasks(self) -> Optional[int]:
        return self._num_tasks

    # ----- subclass hooks -----

    @abstractmethod
    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        """Select a task instance (use ``self.np_random`` when ``task_idx`` is None)."""

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        """Push per-episode state into the in-process tool servers. Default: no-op."""

    def _end_session(self, session_id: str) -> None:
        """Drop the per-episode state ``_begin_session`` pushed. Default: no-op."""

    @abstractmethod
    def _verify(
        self, trajectory: Trajectory, task: Dict[str, Any], *, terminated: bool
    ) -> FeedbackCollection:
        """Score the recorded trajectory (pure)."""

    async def _close(self) -> None:
        """Release resources. Default: no-op."""


# ----- module-level helpers -----


def _schema_json(schema: Optional[type]) -> Optional[Dict[str, Any]]:
    return None if schema is None else schema.model_json_schema()


def _render_static(template: Optional[str]) -> str:
    """Render a template with no variables — the durable, instance-independent framing."""
    if not template:
        return ""
    from minijinja import Environment

    try:
        return Environment().render_str(template)
    except Exception:
        return template


async def _probe_tool_configs(spec: MCPServerSpec) -> List[ToolConfig]:
    """Open a one-shot session against ``spec`` and return its ``ToolConfig``s."""
    session = await _open_session_for_spec(spec, session_id="__probe__")
    try:
        return await session.list_tools()
    finally:
        await session.close()


def _sync_run_async(coro: Awaitable[_T]) -> _T:
    """Run a coroutine to completion from sync code (worker thread if a loop is running)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
