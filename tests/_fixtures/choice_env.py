"""A second ``score``-terminal fixture env whose tool names collide with the first (tests only).

Same names as :mod:`tests._fixtures.score_env` — ``submit``, ``noop``, plus the reserved
``terminate`` every env carries — and a different schema behind ``submit``. Used to
pin multi-env routing: one endpoint cannot register two schemas under one name, and a call must
reach the env its lease names.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shogym.core import Env
from shogym.mcp import MCPServerSpec
from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

from tests._fixtures import choice_mcp

ENV_NAME = "_fixture_choice"
SUBMIT_TOOL = "submit"
HORIZON = 3

_SPEC = MCPServerSpec(
    name="fixture_choice",
    transport="in_process",
    module="tests._fixtures.choice_mcp",
)

_INSTRUCTIONS = "Pick the right option, then call `submit` with your final `choice`."


class _FixtureChoiceEnv(Env):
    """A score-terminal env graded by an integer compare, offline and deterministic."""

    mcp_servers = (_SPEC,)
    function_name = "chooser"
    score_terminal_tool = SUBMIT_TOOL

    def __init__(self, tasks: Optional[List[Dict[str, Any]]] = None) -> None:
        self._tasks: List[Dict[str, Any]] = list(tasks) if tasks else [{"id": "c0", "choice": 1}]
        self._gold: Dict[str, int] = {}
        self.function = FunctionConfig(example_system_template=_INSTRUCTIONS)
        super().__init__(horizon=HORIZON, num_tasks=len(self._tasks))

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._tasks)))
        task = self._tasks[task_idx]
        return {
            "task_idx": task_idx,
            "id": str(task.get("id", task_idx)),
            "choice": int(task.get("choice", 0)),
        }

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        self._gold[session_id] = int(task["choice"])
        choice_mcp.begin_session(session_id, choice=int(task["choice"]))

    def _end_session(self, session_id: str) -> None:
        self._gold.pop(session_id, None)
        choice_mcp.end_session(session_id)

    async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
        correct = req.args is not None and req.args.get("choice") == self._gold.get(
            req.session_id
        )
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict={"correct": bool(correct)},
        )

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: Optional[TerminalEvidence] = None,
    ) -> FeedbackCollection:
        fb = FeedbackCollection()
        if not terminated:
            return fb
        correct = bool(evidence.verdict.get("correct")) if evidence is not None else False
        fb.episode.append(EpisodeFeedback(name="success", value=correct))
        fb.episode.append(EpisodeFeedback(name="reward", value=1.0 if correct else 0.0))
        return fb
