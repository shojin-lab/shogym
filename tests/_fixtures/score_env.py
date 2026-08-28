"""A minimal ``score``-terminal fixture env (tests only).

This is the env the lifecycle/durability/middleware tests drive — NOT a real env. It opts
into the seal transaction the way a migrated env does: it declares
``score_terminal_tool`` and overrides ``finalize``. ``finalize`` grades deterministically
offline (normalized string compare against the session's gold answer), returns core-consumed
:class:`TerminalEvidence`, and ``_verify`` scores from that evidence (never from marker JSON).

Registered under ``_fixture_score``. Importing this module registers it (idempotent within a
process); the registry entry is inert for every other test because nothing else constructs it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from shogym.core import Env
from shogym.envs.registration import _ENV_REGISTRY, register
from shogym.mcp import MCPServerSpec
from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

from tests._fixtures import score_mcp

ENV_NAME = "_fixture_score"
SUBMIT_TOOL = "submit"
HORIZON = 3

_SPEC = MCPServerSpec(
    name="fixture_score",
    transport="in_process",
    module="tests._fixtures.score_mcp",
)

_INSTRUCTIONS = "Answer the question, then call `submit` with your final `answer`."


class _FixtureScoreEnv(Env):
    """A score-terminal env with a deterministic offline finalizer.

    Config: ``tasks`` (list of ``{"id","question","answer"}``); optional ``finalize_hook`` — a
    callable ``(FinalizeRequest, correct: bool) -> None`` a test can use to count invocations
    or block/raise inside ``finalize`` (to exercise the cancellation/close/deadline rules).

    Declares an identity channel, because a stream reads one only where an env says it has one:
    which item says what produced a row is the env's to name, and a replay finds the name on the
    registered class rather than on a live object it does not have. Publishing nothing under it is
    the ordinary case and reads as nothing."""

    identity_feedback_name = "config_digest"

    mcp_servers = (_SPEC,)
    function_name = "solver"
    score_terminal_tool = SUBMIT_TOOL

    def __init__(
        self,
        tasks: Optional[List[Dict[str, Any]]] = None,
        finalize_hook: Optional[Any] = None,
    ) -> None:
        self._tasks: List[Dict[str, Any]] = list(tasks) if tasks else [
            {"id": "q0", "question": "What is 2+2?", "answer": "4"}
        ]
        self._finalize_hook = finalize_hook
        self._gold: Dict[str, str] = {}
        self.finalize_calls = 0
        self.function = FunctionConfig(example_system_template=_INSTRUCTIONS)
        super().__init__(horizon=HORIZON, num_tasks=len(self._tasks))

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._tasks)))
        task = self._tasks[task_idx]
        return {
            "task_idx": task_idx,
            "id": str(task.get("id", task_idx)),
            "question": str(task.get("question", "")),
            "answer": str(task.get("answer", "")),
        }

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        self._gold[session_id] = task["answer"]
        score_mcp.begin_session(session_id, answer=task["answer"])

    def _end_session(self, session_id: str) -> None:
        self._gold.pop(session_id, None)
        score_mcp.end_session(session_id)

    async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
        """Grade the sealed episode. Deterministic + offline: a normalized string compare
        against the session's gold answer. Returns core-owned evidence carrying only a
        public-safe verdict — no oracle (the gold answer is never echoed)."""
        self.finalize_calls += 1
        # A yield so a test can seal, observe the FINALIZING state, then let this proceed.
        await asyncio.sleep(0)
        if req.source == "horizon" or req.args is None:
            # No submission reached the seal: score incorrect (nothing to grade).
            correct = False
        else:
            gold = self._gold.get(req.session_id, "")
            submitted = str(req.args.get("answer", ""))
            correct = submitted.strip().lower() == gold.strip().lower()
        if self._finalize_hook is not None:
            self._finalize_hook(req, correct)
        verdict: Dict[str, Any] = {"correct": correct}
        if req.args is not None and "confidence" in req.args:
            verdict["confidence"] = req.args["confidence"]
        # diagnostic is PRIVATE (durable store only) — safe to reference the gold here.
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict=verdict,
            diagnostic=f"graded source={req.source} correct={correct}",
        )

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: Optional[TerminalEvidence] = None,
    ) -> FeedbackCollection:
        """Score exclusively from the core-owned terminal evidence (never marker JSON)."""
        fb = FeedbackCollection()
        if not terminated:
            return fb
        if evidence is None:
            fb.episode.append(EpisodeFeedback(name="correct", value=False))
            return fb
        fb.episode.append(
            EpisodeFeedback(name="correct", value=bool(evidence.verdict.get("correct")))
        )
        if evidence.finalize_error:
            fb.episode.append(EpisodeFeedback(name="finalize_error", value=True))
        return fb


if ENV_NAME not in _ENV_REGISTRY:
    register(ENV_NAME)(_FixtureScoreEnv)
