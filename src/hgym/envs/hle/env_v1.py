"""``hle`` — Humanity's Last Exam on the env-as-center core (RFC 008, issue #33).

A minimal single-turn reasoning env, and hgym's first **model-graded verifier**. The env
serves one tool, ``submit_answer`` (plus the reserved ``terminate``); the question rides in
the ``TaskSpec`` instructions that ``describe()`` hands the harness. The ``submit_answer``
handler grades server-side (an exact-match fast path, then an injectable LLM judge — see
``mcp_server`` / ``judge``), and this env's pure ``_verify`` parses that verdict off the
recorded trajectory into episode feedback: ``correct`` plus a ``calibration_error`` derived
from the submitted confidence.

This module imports **nothing** heavy at load time — not ``datasets``, not ``openai`` — so
``import hgym`` (which imports this module to register the env) stays offline. The dataset is
loaded lazily only when the registered ``hle`` env is *constructed*, and the default judge's
network client only when it is first *called*.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from hgym.envs.hle import mcp_server as hle_mcp_server
from hgym.envs.hle.judge import DEFAULT_JUDGE_MODEL, Judge, OpenAIJudge
from hgym.envs.registration import register
from hgym.envs.tool_using_env import ToolUsingEnv
from hgym.mcp import MCPServerSpec
from hgym.task import TaskSpec
from hgym.trajectory import Trajectory
from hgym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

# Kept in sync with mcp_server (duplicated here so the verifier needn't import it). Only a
# `submit_answer` step carrying this marker is trusted for terminal credit.
GRADE_MARKER = "hle_grade"
SUBMIT_TOOL_NAME = "submit_answer"

# One `submit_answer`, then `terminate` — a single-turn episode. The horizon also caps a
# harness that keeps submitting: the last graded submission before the cap is scored.
HORIZON = 2

HLE_SPEC = MCPServerSpec(
    name="hle",
    transport="in_process",
    module="hgym.envs.hle.mcp_server",
)

_BASE_INSTRUCTIONS = (
    "You are answering a single question from Humanity's Last Exam (HLE), a benchmark of "
    "expert-level, closed-ended academic questions across many subjects.\n"
    "Read the question, reason carefully, then call `submit_answer` exactly once with your "
    "final `answer` and a `confidence` score from 0 to 100 (how sure you are it is "
    "correct). Then call `terminate` to end the episode.\n"
    "There is no feedback loop and no other tools: your first submitted answer is graded, "
    "so commit to your best answer."
)


@register("hle")
class HleEnv(ToolUsingEnv):
    """Humanity's Last Exam as a single-turn, model-graded hgym env.

    Config (all optional, via ``hgym.make("hle", config=...)`` / ``env_config``):
      - ``task_split``: ``"train"`` (default) or ``"test"`` — a positional 80/20 slice of the
        text-only ``cais/hle`` questions.
      - ``tasks``: an explicit task list (each ``{"question", "answer", ...}``). When given,
        the gated dataset is **not** downloaded — this is how offline tests construct the env.
      - ``judge``: an injected :class:`~hgym.envs.hle.judge.Judge` (a scripted judge for
        offline tests). Default: :class:`~hgym.envs.hle.judge.OpenAIJudge`.
      - ``judge_model`` / ``judge_base_url``: the default judge's model id + endpoint.
    """

    mcp_servers = (HLE_SPEC,)
    function_name = "solver"

    def __init__(
        self,
        task_split: str = "train",
        tasks: Optional[List[Dict[str, Any]]] = None,
        judge: Optional[Judge] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: Optional[str] = None,
    ) -> None:
        self._task_split = task_split
        self._judge = judge
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._tasks: List[Dict[str, Any]] = (
            list(tasks) if tasks is not None else _load_default_tasks(task_split)
        )
        self.function = FunctionConfig(example_system_template=_BASE_INSTRUCTIONS)
        super().__init__(horizon=HORIZON, num_tasks=len(self._tasks))

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if not self._tasks:
            raise ValueError("hle env has no tasks loaded")
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._tasks)))
        if not 0 <= task_idx < len(self._tasks):
            # Reject negatives too: Python's negative indexing would silently serve a
            # valid record while the serve layer records task id `-1` and `describe("-1")`
            # refuses to resolve it — a misattributed run with no published question.
            raise ValueError(
                f"Task index {task_idx} is out of range for {len(self._tasks)} tasks"
            )
        task = self._tasks[task_idx]
        return {
            "task_idx": task_idx,
            "id": str(task.get("id", task_idx)),
            "question": str(task.get("question", "")),
            "answer": str(task.get("answer", "")),
            "answer_type": str(task.get("answer_type", "")),
            "split": self._task_split,
        }

    # ----- session lifecycle -----

    def _judge_for_session(self) -> Judge:
        """The injected judge, or a lazily-built default ``OpenAIJudge`` (no network yet).

        Preflighted at session start (not env construction), so ``hgym.make("hle")``, the tool
        manifest, and ``describe()`` all stay offline and keyless. The default ``OpenAIJudge``
        needs ``OPENAI_API_KEY`` to grade the (almost always free-form) non-exact answers; with
        no key it would raise inside ``submit_answer`` and every non-exact answer would
        fail-closed to ``correct=False``, silently deflating the benchmark. Raise early and
        clearly instead — but only in the "forgot the key" case: an injected ``judge=`` or a
        ``judge_base_url`` override (a keyless OpenAI-compatible endpoint) opts out, which is
        what keeps offline tests and manifest probing network-free."""
        if self._judge is not None:
            return self._judge
        if not self._judge_base_url and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "HLE's default judge (OpenAIJudge) needs OPENAI_API_KEY to grade non-exact "
                "answers, but it is not set. Set OPENAI_API_KEY, pass judge_base_url=... for a "
                "keyless OpenAI-compatible endpoint, or inject judge=... . Without it only exact "
                "string matches score correct and every other answer is scored incorrect."
            )
        return OpenAIJudge(model=self._judge_model, base_url=self._judge_base_url)

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        hle_mcp_server.begin_session(
            session_id,
            question=task["question"],
            correct_answer=task["answer"],
            judge=self._judge_for_session(),
            answer_type=task.get("answer_type", ""),
        )

    def _end_session(self, session_id: str) -> None:
        hle_mcp_server.end_session(session_id)

    # ----- describe: surface this task's question in the instructions -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        spec = super().describe(task_id)
        task = self._resolve_task(task_id)
        if task is None:
            return spec
        parts = [_BASE_INSTRUCTIONS, "", "# Question", str(task.get("question", ""))]
        answer_type = str(task.get("answer_type", ""))
        if answer_type:
            parts += ["", f"(Answer type: {answer_type})"]
        return spec.model_copy(update={"instructions": "\n".join(parts)})

    def _resolve_task(self, task_id: Optional[str]) -> Optional[Dict[str, Any]]:
        """Map the (stringified index) ``task_id`` the serve layer passes to a task dict."""
        if task_id is None:
            return None
        try:
            idx = int(task_id)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(self._tasks):
            return self._tasks[idx]
        return None

    # ----- verify: parse the server-side grade off the recorded trajectory -----

    def _verify(
        self, trajectory: Trajectory, task: Dict[str, Any], *, terminated: bool
    ) -> FeedbackCollection:
        """Score the episode from the judge's verdict, recorded on the ``submit_answer`` step.

        The grade is produced server-side by the tool handler (exact-match or LLM judge); this
        function only *parses* it — pure over the trajectory, like tau2's verifier. Parsing is
        defensive: a missing, forged, or malformed grade scores as incorrect rather than
        raising."""
        return score_trajectory(trajectory, terminated=terminated)


# ----- pure scoring (module-level so it is unit-testable without a judge or the dataset) -----


def score_trajectory(trajectory: Trajectory, *, terminated: bool) -> FeedbackCollection:
    """Build episode feedback from the ``submit_answer`` grade recorded in the trajectory.

    Pure. Emits ``correct`` (bool) always on termination; when a grade is present it also
    emits ``confidence`` (0–1) and ``calibration_error`` — the absolute gap between the
    submitted confidence and the correctness indicator (a per-episode Brier-style term). A
    premature end with no graded submission emits only ``correct = False`` (no confidence to
    calibrate).

    When the recorded grade was produced by a **failed** judge call
    (``judged_by == "llm_judge_error"`` — a mid-run key revocation, rate-limit, or network
    drop that the handler fail-closed to ``correct=False``), an extra ``judge_error = True`` is
    emitted so an analyst can filter judge-infra failures out instead of counting them as
    legitimate zeros. A clean grade (``exact_match`` / ``llm_judge``) emits no ``judge_error``
    (mirroring how tau2's verifier only emits its optional flags when relevant)."""
    fb = FeedbackCollection()
    if not terminated:
        return fb

    grade = _find_grade(trajectory)
    if grade is None:
        # No graded `submit_answer` recorded — the episode ended without a scoreable answer.
        fb.episode.append(EpisodeFeedback(name="correct", value=False))
        return fb

    correct = bool(grade["result"].get("correct"))
    confidence = _confidence_fraction(grade["confidence"])
    fb.episode.append(EpisodeFeedback(name="correct", value=correct))
    if grade["result"].get("judged_by") == "llm_judge_error":
        # The judge itself failed; `correct=False` is fail-closed, not a genuine wrong answer.
        fb.episode.append(EpisodeFeedback(name="judge_error", value=True))
    fb.episode.append(EpisodeFeedback(name="confidence", value=confidence))
    fb.episode.append(
        EpisodeFeedback(
            name="calibration_error",
            value=abs(confidence - (1.0 if correct else 0.0)),
        )
    )
    return fb


def _find_grade(trajectory: Trajectory) -> Optional[Dict[str, Any]]:
    """Return the **first** trusted grade in the trajectory, or None.

    Single-turn: the *first* submitted answer is the graded one (the server already refuses to
    grade a second submission), so the verifier also takes the first marked grade — a harness
    can't inspect a wrong verdict and replace it. Only a ``submit_answer`` step is trusted: it
    is the sole tool whose handler runs the judge, so a marked grade on any *other* recorded
    result (a forged tool output) grants no credit. Scans from the start for a ``submit_answer``
    result that is a JSON object carrying ``GRADE_MARKER``; the confidence is read from the step
    **arguments** (the trustworthy trajectory value the transport strips of any injected
    ``_session_id``), not from the (echoed) result. Any non-JSON / non-object / unmarked /
    non-``submit_answer`` result is skipped, never raised on."""
    for step in trajectory:
        if step.tool != SUBMIT_TOOL_NAME:
            continue
        try:
            payload = json.loads(step.result)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get(GRADE_MARKER) is True:
            return {"result": payload, "confidence": step.arguments.get("confidence")}
    return None


def _confidence_fraction(confidence: Any) -> float:
    """Coerce a 0–100 confidence to a [0, 1] fraction; junk/absent defaults to 1.0."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 100.0
    return max(0.0, min(100.0, value)) / 100.0


def _load_default_tasks(task_split: str) -> List[Dict[str, Any]]:
    """Load the registered env's real tasks from the gated ``cais/hle`` dataset (lazy)."""
    from hgym.envs.hle.data import load_hle_tasks, split_tasks

    return split_tasks(load_hle_tasks(text_only=True), task_split)
