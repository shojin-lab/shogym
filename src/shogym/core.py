"""The environment interface (RFC 008): describe, serve, verify.

An shogym environment is not a gym — it has no ``reset``/``step`` loop and no agent. It
**describes** a task, **serves** its essential tools as MCP servers, and **verifies** a
recorded trajectory. A harness (external) drives the tools; the env only publishes the
contract and scores what happened.

:class:`Env` is both that contract and the machinery behind it. A subclass declares its MCP
servers, its advisory instruction templates, and its horizon, and implements two hooks:
``_load_task`` (pick an instance) and ``_verify`` (score the recorded
:class:`~shogym.trajectory.Trajectory`); ``_begin_session`` / ``_end_session`` / ``_close``
are optional. The base probes the tool manifest at construction so ``describe()`` can publish
it, keys per-episode state by session id, and tracks the sessions it opens so ``close()``
tears them all down. It owns no loop — a harness drives the tools via ``serve``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    TypeVar,
    cast,
)

from shogym.mcp.toolset import _open_session_for_spec
from shogym.mcp.types import MCPServerSpec
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.task import ReferenceTemplate, TaskSpec, TerminalKind, ToolManifest
from shogym.trajectory import Trajectory
from shogym.types import FeedbackCollection, FunctionConfig, ToolConfig
from shogym.utils import seeding

if TYPE_CHECKING:
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence

_TERMINATE_SPEC = MCPServerSpec(
    name="__terminate__",
    transport="in_process",
    module="shogym.shared.terminate_mcp",
)

_T = TypeVar("_T")


class Env(ABC):
    """An environment: describe + serve (MCP specs) + verify."""

    # Subclasses override these.
    mcp_servers: Sequence[MCPServerSpec] = ()
    function: FunctionConfig = FunctionConfig()
    function_name: str = "agent"
    # The single tool (if any) this env marks as the `score` terminal. When set, ``describe()``
    # advertises it with ``terminal_kind="score"`` and the serve layer runs its call as a
    # validate -> seal -> evaluate transaction (only if the env also overrides ``finalize``).
    # Left ``None`` by an env with no scoring terminal, so it advertises only `none`/`abort`
    # tools and the seal transaction never engages.
    score_terminal_tool: Optional[str] = None

    # Whether this env's episode feedback may ride out on the terminal result's
    # ``_meta["shogym/feedback"]`` sidecar. True everywhere by default, which is the
    # existing behaviour: a harness reads the score off the terminal result.
    #
    # An env sets it False when the score is what an EXPERIMENT delivers rather than
    # what the episode hands back. The feedback is still computed, still written to the
    # trace, and still returned by ``evaluate()``; it is withheld from the one channel
    # that crosses into the agent's own process. There is no per-call override, because
    # a terminal that returns the score sometimes is a terminal that returns the score.
    inband_terminal_feedback: bool = True

    # The optional, typed terminal-transaction hook. Default ``None`` means the env has no
    # scoring finalizer (a pure-verify / abort-only env): the serve layer never engages the
    # seal transaction for it and it behaves exactly as before. An env opts in by declaring a
    # ``score`` terminal tool *and* overriding this with
    # ``async def finalize(self, req: FinalizeRequest) -> TerminalEvidence`` — the serve layer
    # runs it on the already-sealed episode to produce the trusted terminal evidence.
    finalize: Optional[
        Callable[["FinalizeRequest"], Awaitable["TerminalEvidence"]]
    ] = None

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
        """The MCP servers to serve: the reserved ``terminate`` server + env-mandatory ones."""
        return [_TERMINATE_SPEC, *self.mcp_servers]

    def _probe_manifest(self) -> Dict[str, ToolConfig]:
        tools: Dict[str, ToolConfig] = {}
        for spec in self.essential_specs():
            for tc in _sync_run_async(_probe_tool_configs(spec)):
                if tc.name in tools:
                    raise ValueError(f"duplicate tool name {tc.name!r} across mcp_servers")
                tools[tc.name] = tc
        # Enforce exactly zero-or-one `score` terminal per env, and that it is a real
        # advertised tool (never the reserved `terminate`). Fail fast at construction so a typo
        # can't silently leave the env with no scoring terminal, and describe() can never
        # advertise two `score` terminals.
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
            # A declared score terminal MUST have a *callable* finalize hook, else a v2 contract
            # would advertise seal-before-verdict semantics that never engage. This is an early,
            # clearer error for an env that declares `score_terminal_tool`; the AUTHORITATIVE
            # enforcement lives at the serve boundary (`ServedEpisode.__init__`), which every
            # env — including one that hand-builds its manifest in ``describe()`` and so never
            # trips this check — must pass through. Check `callable`, not just `is None`, so a
            # non-callable finalize (`False`, a stray attribute) also fails fast rather than
            # silently disabling the transaction.
            if not callable(getattr(self, "finalize", None)):
                raise ValueError(
                    f"score_terminal_tool {score!r} is declared but this env has no callable "
                    "`finalize` hook; a score terminal requires an async finalize()"
                )
        return tools

    def _terminal_kind(self, tool_name: str) -> TerminalKind:
        """The terminal role of ``tool_name``: the reserved ``terminate`` is ``abort``, this
        env's ``score_terminal_tool`` (if any) is ``score``, everything else is ``none``."""
        if tool_name == TERMINATE_TOOL_NAME:
            return "abort"
        if tool_name == self.score_terminal_tool:
            return "score"
        return "none"

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        """Publish the task contract a harness reads to configure itself (RFC 008 §3.1).

        Read-only; opens no session."""
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
        """Select a task instance (deterministic when ``task_idx`` is given)."""
        if task_idx is not None:
            self._np_random, _ = seeding.np_random(task_idx)
        return self._load_task(task_idx)

    def begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        """Push per-episode state into the (in-process) tool servers, via ``_begin_session``."""
        self._open_session_ids.add(session_id)
        self._begin_session(session_id, task)

    def claim_session(self, session_id: str) -> Optional[Callable[[], None]]:
        """Take sole responsibility for releasing one session, and get the release to run.

        Returns a zero-argument callable that runs ``_end_session`` for ``session_id``, or
        ``None`` if some other caller has already claimed it. Exactly one caller is ever handed
        the callable, because the claim is a ``set.remove`` and only one remover can win.

        **Claiming and releasing are separate so a caller can claim on its event loop and run the
        hook off it.** ``_end_session`` is an env hook and some envs make it do real work (an env
        whose episode is a world in another process signals and reaps it there), so the serve
        layer runs it in a thread and bounds its wait. Between the two, something else may try to
        release the same session: the completion callback of an abandoned setup, or the
        ``env.close()`` at the end of an episode whose teardown gave up waiting. A session that
        has been claimed is no longer open, so those callers get ``None`` here and
        :meth:`close` finds nothing of it left. One hook per session, whatever the timing, rather
        than two of them on one episode's resources, which is the failure an env's
        ``_end_session`` is under no obligation to survive.

        A hook that raises does not put the session back: it was entered once, and a second entry
        is the thing this exists to prevent."""
        try:
            self._open_session_ids.remove(session_id)
        except KeyError:
            return None
        return lambda: self._end_session(session_id)

    def end_session(self, session_id: str) -> None:
        """Drop the per-episode state ``begin_session`` created, via ``_end_session``.

        Symmetric with :meth:`begin_session`, and idempotent: the claim and the hook together
        (see :meth:`claim_session`). ``close()`` invokes it for any session still open, so a
        stateful in-process tool server does not leak an entry per episode.
        """
        release = self.claim_session(session_id)
        if release is not None:
            release()

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
        # Only forward ``evidence`` to ``_verify`` when the serve layer actually produced it (a
        # score-terminal env's seal path). A non-score env is called with ``evidence=None`` and
        # dispatched exactly as before — its ``_verify`` keeps its original three-argument
        # signature, so this is transparent for envs with no scoring terminal.
        if evidence is not None:
            # A score-terminal env's ``_verify`` accepts a fourth keyword, ``evidence``; a
            # legacy env's does not. Only score envs ever reach this branch (evidence is None
            # for every legacy env), so forward it dynamically — this keeps the abstract
            # ``_verify`` contract at three args, so legacy overrides stay type-compatible.
            verify_with_evidence = cast(Any, self._verify)
            return verify_with_evidence(
                trajectory, task, terminated=terminated, evidence=evidence
            )
        return self._verify(trajectory, task, terminated=terminated)

    async def close(self) -> None:
        """Release any resources, then hand off to ``_close``."""
        # Tear down every session still open (an env may serve several concurrent
        # episodes) so a stateful in-process tool server doesn't leak per-episode
        # entries `begin_session` created. Snapshot first — end_session mutates the set.
        for session_id in list(self._open_session_ids):
            self.end_session(session_id)
        await self._close()

    @property
    def name(self) -> str:
        return getattr(self, "_registered_name", type(self).__name__)

    @property
    def horizon(self) -> Optional[int]:
        return getattr(self, "_horizon", None)

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
        """Score the recorded trajectory (pure).

        A ``score``-terminal env additionally accepts a keyword ``evidence`` (core-owned
        :class:`~shogym.serve.lifecycle.TerminalEvidence`) and scores from it. A non-score env
        keeps this three-argument signature — the base ``verify`` forwards ``evidence`` only to
        an env that produced it, so a non-score env is never called with it."""

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
