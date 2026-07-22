"""The episode-serving engine (RFC 008): drive a reset env one tool call at a time.

An external harness owns the loop; hgym owns the env. So there is no ``step(action)`` the
harness calls — a *tool call is the step*. :meth:`ServedEpisode.call` translates one
incoming call into one ``env.step``, which already handles dispatch, horizon, terminate
detection, and ``_verify``. The engine layers on the RFC 008 wire contract: feedback rides
back as a ``_meta`` sidecar (episode feedback hidden until the terminal result), a separate
``hgym/terminate`` flag signals the stop, and every step is appended to the JSONL trace.

Horizon is enforced env-side: ``env.step`` returns ``terminated`` when the budget is spent,
so a harness that keeps calling tools is stopped by the env, not by any loop hgym owns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hgym.envs import make
from hgym.envs.tool_using_env import ToolUsingEnv
from hgym.feedback.wire import FeedbackItem, build_meta, select_inband
from hgym.task import TaskSpec
from hgym.trace import append_trace, step_record
from hgym.types import StepData
from hgym.types.content import ToolCallContentBlock, ToolResultContentBlock


@dataclass
class CallResult:
    """The outcome of one tool call: the tool's functional ``content`` (the observation
    the harness needs), the ``meta`` sidecar (feedback + terminate flag), and whether the
    episode is now over."""

    content: str
    meta: Dict[str, Any]
    terminated: bool


class ServedEpisode:
    """One episode of one env, driven by external tool calls.

    Construct via :meth:`start` (it resets the env). Then call :meth:`call` per tool
    invocation until :attr:`terminated`. :meth:`describe` returns the task contract to
    hand the harness at setup.
    """

    def __init__(
        self,
        env: ToolUsingEnv,
        env_name: str,
        task_id: Optional[str],
        trace_path: Optional[Union[str, Path]],
    ) -> None:
        self._env = env
        self._env_name = env_name
        self._task_id = task_id
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._session_id: str = env._session_id or ""
        self._step = 0
        self._terminated = False

    @classmethod
    async def start(
        cls,
        env_name: str,
        *,
        task: Optional[Union[int, str]] = None,
        trace_path: Optional[Union[str, Path]] = None,
        env_config: Optional[Dict[str, Any]] = None,
    ) -> "ServedEpisode":
        """Build the env, reset it to the given task instance, and return the engine."""
        env = make(env_name, config=env_config)
        if not isinstance(env, ToolUsingEnv):
            raise TypeError(f"{env_name!r} is not a ToolUsingEnv; cannot be served")
        task_idx = int(task) if task is not None else None
        await env.reset(task_idx)
        return cls(env, env_name, None if task is None else str(task), trace_path)

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def session_id(self) -> str:
        return self._session_id

    def describe(self) -> TaskSpec:
        """The task contract to publish to the harness (RFC 008 §3.1)."""
        return self._env.describe(self._task_id)

    async def call(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> CallResult:
        """Execute one tool call as one env step; return its result + feedback sidecar.

        After termination (terminate tool or horizon), further calls are a no-op that
        re-report the terminal state, so a harness that keeps going is handled gracefully.
        """
        if self._terminated:
            return CallResult(
                content="<episode already terminated>",
                meta=build_meta(terminate=True),
                terminated=True,
            )

        self._step += 1
        action: Action = [
            ToolCallContentBlock(
                id=f"call-{self._step}", name=tool_name, arguments=arguments or {}
            )
        ]
        step_data = await self._env.step(action)

        content = _extract_result(step_data, tool_name)
        items: List[FeedbackItem] = [
            *step_data.feedback.inference,
            *step_data.feedback.episode,
        ]
        terminated = step_data.terminated or step_data.truncated
        self._terminated = terminated

        if self._trace_path is not None:
            append_trace(
                self._trace_path,
                step_record(
                    session_id=self._session_id,
                    env_name=self._env_name,
                    task_id=self._task_id,
                    step=self._step,
                    tool=tool_name,
                    feedback=items,  # trace records everything, in or out of band
                    terminated=terminated,
                ),
            )

        # In-band: hide episode-level feedback until the terminal result (RFC 008 §4.4).
        inband = select_inband(items, terminal=terminated)
        return CallResult(
            content=content,
            meta=build_meta(inband, terminate=terminated),
            terminated=terminated,
        )

    async def close(self) -> None:
        await self._env.close()


# `Action` is a list of content blocks (see hgym.types.action); named here for clarity.
Action = List[ToolCallContentBlock]


def _extract_result(step_data: StepData, tool_name: str) -> str:
    """Pull this step's tool result out of the observation's last user turn.

    The env appends tool results as the final user message; with one call per step there
    is exactly one result block. Match by name, falling back to the first block."""
    for message in reversed(step_data.observation.messages):
        if message.role != "user" or not isinstance(message.content, list):
            continue
        blocks = [b for b in message.content if isinstance(b, ToolResultContentBlock)]
        if not blocks:
            continue
        for block in blocks:
            if block.name == tool_name:
                return block.result
        return blocks[0].result
    return ""
