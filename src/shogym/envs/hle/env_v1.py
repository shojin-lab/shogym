"""``hle`` — Humanity's Last Exam on the env-as-center core (RFC 008, issue #33).

A minimal single-turn reasoning env, and shogym's first **model-graded verifier**. The env
serves one tool, ``submit_answer`` (plus the reserved ``terminate``); the question rides in
the ``TaskSpec`` instructions that ``describe()`` hands the harness. ``submit_answer`` is the
env's **score terminal**: calling it validates the args, atomically seals the episode, and
ends it in one step. The serve layer then runs this env's ``finalize`` hook on the sealed
episode — an exact-match fast path, then an injectable LLM judge (see ``judge``) — which
returns core-owned :class:`~shogym.serve.lifecycle.TerminalEvidence` carrying a **public-safe**
verdict. ``_verify`` scores that evidence into episode feedback: ``correct`` plus a
``calibration_error`` derived from the submitted confidence. Because grading runs only on an
already-sealed, un-continuable episode, an agent can never grade, read the verdict, and revise
its answer.

This module imports **nothing** heavy at load time — not ``datasets``, not ``openai`` — so
``import shogym`` (which imports this module to register the env) stays offline. The dataset is
loaded lazily only when the registered ``hle`` env is *constructed*, and the default judge's
network client only when it is first *called*.
"""

from __future__ import annotations

import asyncio
import copy
import os
from typing import Any, Dict, List, Optional

from shogym.core import Env
from shogym.envs.hle.judge import DEFAULT_JUDGE_MODEL, Judge, OpenAIJudge, exact_match
from shogym.envs.registration import register
from shogym.mcp import MCPServerSpec
from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

SUBMIT_TOOL_NAME = "submit_answer"

# `submit_answer` is the single terminal action: submitting seals + grades + ends the episode
# in one step (no separate `terminate`). The horizon is 1 so a single action ends the episode;
# reaching it with no valid submission scores `correct=False` (`zero_unsubmitted`).
HORIZON = 1

HLE_SPEC = MCPServerSpec(
    name="hle",
    transport="in_process",
    module="shogym.envs.hle.mcp_server",
)

_BASE_INSTRUCTIONS = (
    "You are answering a single question from Humanity's Last Exam (HLE), a benchmark of "
    "expert-level, closed-ended academic questions across many subjects.\n"
    "Read the question, reason carefully, then call `submit_answer` exactly once with your "
    "final `answer` and a `confidence` score from 0 to 100 (how sure you are it is "
    "correct). Submitting grades your answer and ends the episode — there is no second "
    "submission, no feedback loop, and no further step to take (do not call `terminate` "
    "afterward), so commit to your best answer."
)


@register("hle")
class HleEnv(Env):
    """Humanity's Last Exam as a single-turn, model-graded shogym env.

    Config (all optional, via ``shogym.make("hle", config=...)`` / ``env_config``):
      - ``task_split``: ``"train"`` (default) or ``"test"`` — a positional 80/20 slice of the
        text-only ``cais/hle`` questions.
      - ``tasks``: an explicit task list (each ``{"question", "answer", ...}``). When given,
        the gated dataset is **not** downloaded — this is how offline tests construct the env.
      - ``judge``: an injected :class:`~shogym.envs.hle.judge.Judge` (a scripted judge for
        offline tests). Default: :class:`~shogym.envs.hle.judge.OpenAIJudge`.
      - ``judge_model`` / ``judge_base_url``: the default judge's model id + endpoint.
      - ``judge_kwargs``: sampling fields for the default judge's chat-completions request
        (``{"reasoning_effort": "low"}``, say). Sent verbatim; omitted entirely when unset. What
        the judge owns is refused when the episode starts: its model and prompt, the SDK's
        ``extra_*`` hatches, and anything that changes the shape of the reply it parses.
    """

    mcp_servers = (HLE_SPEC,)
    function_name = "solver"
    # `submit_answer` is this env's single `score` terminal. The serve layer validates its
    # args, atomically seals the episode, then runs `finalize` (seal-before-verdict), so a
    # graded verdict only ever exists for an already-sealed, un-continuable episode.
    score_terminal_tool = SUBMIT_TOOL_NAME

    def __init__(
        self,
        task_split: str = "train",
        tasks: Optional[List[Dict[str, Any]]] = None,
        judge: Optional[Judge] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: Optional[str] = None,
        judge_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._task_split = task_split
        self._judge = judge
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        # Deep, so an edit to the config mapping the caller still holds cannot change what a
        # later episode of this env is scored with.
        self._judge_kwargs: Dict[str, Any] = copy.deepcopy(dict(judge_kwargs or {}))
        self._tasks: List[Dict[str, Any]] = (
            list(tasks) if tasks is not None else _load_default_tasks(task_split)
        )
        # Per-session grading state (question, gold answer, judge), keyed by session_id so one
        # env instance safely backs many concurrent episodes. `finalize` reads it to grade the
        # sealed submission; `_begin_session`/`_end_session` own its lifecycle.
        self._grading_state: Dict[str, Dict[str, Any]] = {}
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

        Preflighted at session start (not env construction), so ``shogym.make("hle")``, the tool
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
        return OpenAIJudge(
            model=self._judge_model,
            base_url=self._judge_base_url,
            request_kwargs=self._judge_kwargs,
        )

    def _judge_provenance(self) -> Dict[str, str]:
        """The model this env's own judge grades with, plus its ``reasoning_effort`` when set.

        Empty for an injected ``judge=``: only the caller knows what that is, so the env names
        nothing rather than guessing."""
        if self._judge is not None:
            return {}
        provenance = {"judge_model": self._judge_model}
        effort = self._judge_kwargs.get("reasoning_effort")
        if effort:
            provenance["judge_effort"] = str(effort)
        return provenance

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        # Grading runs in `finalize` (on the env), not in the tool handler, so the per-episode
        # question / gold answer / judge live here on the env rather than in the tool server.
        self._grading_state[session_id] = {
            "question": task["question"],
            "correct_answer": task["answer"],
            "judge": self._judge_for_session(),
            "answer_type": task.get("answer_type", ""),
        }

    def _end_session(self, session_id: str) -> None:
        self._grading_state.pop(session_id, None)

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

    # ----- finalize: grade the sealed submission (seal-before-verdict) -----

    # `Env.finalize` is declared as an optional Callable attribute (default None); overriding
    # it with the sanctioned `async def finalize(self, req)` method is the documented opt-in,
    # so the attribute-vs-method variance report here is a false positive.
    async def finalize(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, req: FinalizeRequest
    ) -> TerminalEvidence:
        """Grade the already-sealed episode and return core-owned terminal evidence.

        Runs HLE's grading on the sealed submission: a normalized exact-match fast path first
        (offline, free) and, on a miss, the session's injectable LLM judge. The returned
        ``verdict`` is **public-safe** (``correct``, ``judge_error``, and, when the env built the
        judge itself, which model graded: see :meth:`_judge_provenance`) and is the sole
        thing the agent ever sees (the serve layer stamps provenance / finalization_id / source
        and appends its own ``finalize_error`` flag). The judge's ``reasoning`` /
        ``extracted_answer`` and any exception text are answer oracles: they go **only** in the
        private ``diagnostic`` (durable store / server logs), never in the verdict. Only a
        model-graded episode names a judge: the exact-match fast path publishes none, since no
        model read that answer.

        A judge failure fails **closed** — ``correct=False`` with ``status='finalize_error'`` —
        so a grading-infra failure is a distinguishable, non-oracle zero rather than a crash. A
        terminal with no gradeable submission (the ``horizon`` ``zero_unsubmitted`` path, or a
        lost session) scores ``correct=False`` without invoking the judge. (``terminate``/abort
        is scored by the serve layer directly and never reaches this hook.)"""
        state = self._grading_state.get(req.session_id)
        if req.source != "explicit_tool" or req.args is None or state is None:
            # No scoreable submission reached the seal: no credit, no judge call.
            return TerminalEvidence(
                source=req.source,
                status="ok",
                verdict={"correct": False, "judge_error": False},
                diagnostic=f"no gradeable submission (source={req.source})",
            )

        answer = str(req.args.get("answer", ""))
        gold = state["correct_answer"]
        # Fast path: a normalized exact match short-circuits the LLM judge (offline, free).
        if exact_match(answer, gold):
            return TerminalEvidence(
                source=req.source,
                status="ok",
                verdict={"correct": True, "judge_error": False},
                diagnostic="graded exact_match correct=True",
            )

        judge: Judge = state["judge"]
        question = state["question"]
        provenance = self._judge_provenance()
        try:
            # Offload the (possibly blocking, network-bound) judge to a worker thread so it
            # neither wedges the event loop for concurrent episodes nor defeats the serve
            # layer's finalize deadline.
            verdict = await asyncio.to_thread(
                judge, question=question, correct_answer=gold, response=answer
            )
        except Exception as exc:  # noqa: BLE001 — any judge failure fails closed, never crashes
            return TerminalEvidence(
                source=req.source,
                status="finalize_error",
                # Provenance rides on a fail-closed grade too: which judge could not be reached
                # is what an analyst filtering these zeros needs.
                verdict={"correct": False, "judge_error": True, **provenance},
                # Exception text is PRIVATE — it may echo the answer/gold; keep it off the wire.
                diagnostic=f"judge error: {type(exc).__name__}: {exc}",
            )
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict={
                "correct": bool(verdict.correct),
                "judge_error": False,
                **provenance,
            },
            # extracted_answer / reasoning are answer oracles — private diagnostic only.
            diagnostic=(
                f"graded llm_judge correct={bool(verdict.correct)} "
                f"extracted={verdict.extracted_answer!r}"
            ),
        )

    # ----- verify: score the core-owned terminal evidence -----

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: Optional[TerminalEvidence] = None,
    ) -> FeedbackCollection:
        """Score the episode from the core-owned terminal ``evidence`` (never marker JSON).

        Pure over the evidence + the submitted confidence. The verdict's ``correct`` is the
        authoritative, seal-protected grade; the confidence rides on the validated submit
        args."""
        return score_evidence(evidence, terminated=terminated)


# ----- pure scoring (module-level so it is unit-testable without a judge or the dataset) -----


def score_evidence(
    evidence: Optional[TerminalEvidence], *, terminated: bool
) -> FeedbackCollection:
    """Build episode feedback from the core-owned terminal :class:`TerminalEvidence`.

    Pure. Emits ``correct`` (bool) always on termination; when there was a submission it also
    emits ``confidence`` (0–1, from the validated submit args) and ``calibration_error`` — the
    absolute gap between the submitted confidence and the correctness indicator (a per-episode
    Brier-style term). A terminal with no submission (``terminate``/abort or the
    ``zero_unsubmitted`` horizon) emits only ``correct = False`` — there is no confidence to
    calibrate.

    A **fail-closed** grade (a judge failure, or a serve-layer deadline/crash — any
    ``finalize_error``) also emits ``judge_error = True`` so an analyst can filter grading-infra
    failures out instead of counting them as legitimate zeros. A clean grade emits no
    ``judge_error`` (mirroring how tau2's verifier only emits its optional flags when
    relevant).

    A model-graded episode also emits the **judge provenance** the verdict carries
    (``judge_model``, plus ``judge_effort`` when one was configured), so a score read back off
    the trace says what produced it."""
    fb = FeedbackCollection()
    if not terminated:
        return fb
    if evidence is None:
        # No evidence produced — the episode ended without a scoreable outcome.
        fb.episode.append(EpisodeFeedback(name="correct", value=False))
        return fb

    verdict = evidence.verdict or {}
    correct = bool(verdict.get("correct"))
    fb.episode.append(EpisodeFeedback(name="correct", value=correct))
    if evidence.finalize_error or bool(verdict.get("judge_error")):
        # A grading-infra failure fail-closed to `correct=False`, not a genuine wrong answer.
        fb.episode.append(EpisodeFeedback(name="judge_error", value=True))
    for key in ("judge_model", "judge_effort"):
        value = verdict.get(key)
        if value:
            fb.episode.append(EpisodeFeedback(name=key, value=str(value)))
    # Confidence rides on the validated submit args (explicit_tool only). A no-submission
    # terminal carries none, so there is nothing to calibrate.
    args = evidence.args
    if args is not None:
        confidence = _confidence_fraction(args.get("confidence"))
        fb.episode.append(EpisodeFeedback(name="confidence", value=confidence))
        fb.episode.append(
            EpisodeFeedback(
                name="calibration_error",
                value=abs(confidence - (1.0 if correct else 0.0)),
            )
        )
    return fb


def _confidence_fraction(confidence: Any) -> float:
    """Coerce a 0–100 confidence to a [0, 1] fraction; junk/absent defaults to 1.0."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 100.0
    return max(0.0, min(100.0, value)) / 100.0


def _load_default_tasks(task_split: str) -> List[Dict[str, Any]]:
    """Load the registered env's real tasks from the gated ``cais/hle`` dataset (lazy)."""
    from shogym.envs.hle.data import load_hle_tasks, split_tasks

    return split_tasks(load_hle_tasks(text_only=True), task_split)
