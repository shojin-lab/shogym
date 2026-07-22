"""The episode-serving engine (RFC 008): drive a served env one tool call at a time.

A tool call *is* the step. :class:`ServedEpisode` opens the env's essential MCP sessions
directly (the terminate server + the env-mandatory servers), dispatches each incoming call
to the right session, records a flat :class:`~hgym.trajectory.Trajectory`, gates the
horizon env-side, and runs the env's pure ``verify`` on each call. Feedback rides back on
the ``_meta`` sidecar (episode-level hidden until the terminal result) and every step is
appended to the JSONL trace. No gym ``step``/``Observation`` anywhere.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hgym.envs import make
from hgym.feedback.wire import build_meta, dump_item, select_inband
from hgym.mcp.session import MCPSession
from hgym.mcp.toolset import _open_session_for_spec
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from hgym.task import TaskSpec
from hgym.trace import append_trace, step_record
from hgym.trajectory import Step, Trajectory
from hgym.utils.uuid7 import uuid7


@dataclass
class CallResult:
    """The outcome of one tool call: the tool's functional ``content`` (the observation the
    harness needs), the ``meta`` sidecar (feedback + terminate flag), and whether the episode
    is now over."""

    content: str
    meta: Dict[str, Any]
    terminated: bool


class ServedEpisode:
    """One episode of one env, driven by external tool calls.

    Construct via :meth:`start` (it loads the task and opens the tool sessions), then call
    :meth:`call` per tool invocation until :attr:`terminated`. :meth:`describe` returns the
    task contract to hand the harness at setup.
    """

    def __init__(
        self,
        env,
        env_name: str,
        task_id: Optional[str],
        session_id: str,
        task: Dict[str, Any],
        sessions: Dict[str, MCPSession],
        opened: List[MCPSession],
        trace_path: Optional[Union[str, Path]],
    ) -> None:
        self._env = env
        self._env_name = env_name
        self._task_id = task_id
        self._session_id = session_id
        self._task = task
        self._sessions = sessions  # advertised tool name -> session
        self._opened = opened  # every session opened, for teardown
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._trajectory: Trajectory = []
        self._step = 0
        self._terminated = False
        # The terminal step's feedback in wire form (inference + episode), retained so
        # the in-process `evaluate()` can report the score without a trace file. Same
        # list `result_from_trace` reconstructs from the terminal row.
        self._terminal_feedback: List[Dict[str, Any]] = []
        # Serialize calls: one episode is a single sequential trajectory. `call()`
        # mutates shared step/trajectory/terminated state across an `await`, so
        # concurrent MCP requests on this session must not interleave.
        self._lock = asyncio.Lock()

    @classmethod
    async def start(
        cls,
        env_name: str,
        *,
        task: Optional[Union[int, str]] = None,
        trace_path: Optional[Union[str, Path]] = None,
        env_config: Optional[Dict[str, Any]] = None,
    ) -> "ServedEpisode":
        """Build the env, load the task instance, open the essential MCP sessions, and push
        per-episode state into the (in-process) tool servers."""
        env = make(env_name, config=env_config)
        opened: List[MCPSession] = []
        try:
            task_idx = int(task) if task is not None else None
            task_data = env.load_task(task_idx)
            # Publish the *resolved* task identity so a random-default episode (task
            # omitted) is still attributable: an env that indexes tasks records the
            # chosen index in task_data (Wordle: "task_idx"), so a `hgym serve wordle_v1`
            # run traces a concrete task rather than null.
            if task is not None:
                resolved_task: Optional[str] = str(task)
            elif "task_idx" in task_data:
                resolved_task = str(task_data["task_idx"])
            else:
                resolved_task = None
            session_id = str(uuid7())

            sessions: Dict[str, MCPSession] = {}
            for spec in env.essential_specs():
                session = await _open_session_for_spec(spec, session_id=session_id)
                opened.append(session)
                for tool_config in await session.list_tools():
                    sessions[tool_config.name] = session
            env.begin_session(session_id, task_data)
        except BaseException:
            # Setup failed, so no ServedEpisode is returned for the caller to close:
            # release everything here. Close any opened MCP sessions, then close the
            # env (drops per-episode state begin_session may have pushed before
            # raising). Both are best-effort so the original setup error propagates.
            for session in opened:
                try:
                    await session.close()
                except Exception:
                    pass
            try:
                await env.close()
            except Exception:
                pass
            raise

        return cls(
            env,
            env_name,
            resolved_task,
            session_id,
            task_data,
            sessions,
            opened,
            trace_path,
        )

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def terminal_feedback(self) -> List[Dict[str, Any]]:
        """The terminal step's feedback (wire form), or ``[]`` until the episode ends."""
        return self._terminal_feedback

    def describe(self) -> TaskSpec:
        return self._env.describe(self._task_id)

    async def call(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> CallResult:
        """Execute one tool call as one step; return its result + feedback sidecar.

        After termination (terminate tool or horizon), further calls are a no-op that
        re-report the terminal state."""
        # One episode = one sequential trajectory. Hold the lock across the whole
        # lifecycle so concurrent requests can't interleave the shared step counter,
        # slip past the horizon, or corrupt trace order.
        async with self._lock:
            if self._terminated:
                return CallResult(
                    content="<episode already terminated>",
                    meta=build_meta(terminate=True),
                    terminated=True,
                )

            # Prospective step: don't advance `self._step` until the call actually
            # completes. If `call_tool` is cancelled (harness timeout) or raises, the
            # counter stays put so the next call reuses this number — the trajectory
            # stays contiguous, one Step per completed call.
            step = self._step + 1
            args = dict(arguments or {})
            # `_session_id` is a reserved hidden field the transport injects with the
            # real id. Strip any caller-supplied value before *both* dispatch and Step
            # construction, so a forged id can't run against the real session nor land
            # in the trajectory a verifier reads.
            args.pop("_session_id", None)
            session = self._sessions.get(tool_name)
            if session is None:
                content = f"<unknown tool {tool_name!r}>"
            else:
                result = await session.call_tool(
                    tool_name, args, tool_call_id=f"call-{step}"
                )
                content = result.result
            # The await completed; commit the step atomically with its Step. Everything
            # from here on is synchronous, so no cancellation point can split them.
            self._step = step
            self._trajectory.append(
                Step(index=step, tool=tool_name, arguments=args, result=content)
            )

            horizon = self._env.horizon
            terminated = tool_name == TERMINATE_TOOL_NAME or (
                horizon is not None and step >= horizon
            )
            self._terminated = terminated

            feedback = self._env.verify(self._trajectory, self._task, terminated=terminated)
            items = [*feedback.inference, *feedback.episode]

            if terminated:
                # Retain the terminal feedback so the no-trace `evaluate()` path can
                # report the score directly off the episode (not only via the trace).
                self._terminal_feedback = [dump_item(item) for item in items]

            if self._trace_path is not None:
                append_trace(
                    self._trace_path,
                    step_record(
                        session_id=self._session_id,
                        env_name=self._env_name,
                        task_id=self._task_id,
                        step=step,
                        tool=tool_name,
                        feedback=items,  # trace records everything, in or out of band
                        terminated=terminated,
                    ),
                )

            # Eval-safe default: dense inference feedback is recorded (above) but not
            # surfaced in-band. v0 exposes no per-tool opt-in, so surface_inference
            # stays False; episode feedback rides out only on the terminal result.
            inband = select_inband(items, terminal=terminated, surface_inference=False)
            return CallResult(
                content=content,
                meta=build_meta(inband, terminate=terminated),
                terminated=terminated,
            )

    async def close(self) -> None:
        # Close every MCP session opened for this episode, then let the env tear
        # down its own per-session state. `env.close()` drops in-process server
        # state via `end_session`; out-of-process sessions are reaped by their
        # `close()` above (one subprocess per session).
        for session in self._opened:
            try:
                await session.close()
            except Exception:
                pass
        await self._env.close()
