"""``browsecomp_plus`` — BrowseComp-Plus on the env-as-center core (RFC 008, issue #43).

A Deep-Research **retrieval** benchmark: answer OpenAI BrowseComp's reasoning-heavy queries
against a **fixed, human-verified corpus** (served ``search`` / ``get_document`` tools) instead
of the live web — isolating search+reasoning from web noise and making runs reproducible. This
is "HLE with a fixed retrieval corpus": the answer is graded by an LLM judge (as in the HLE
port), and the env adds deterministic **retrieval recall** (off the recorded ``search`` steps)
and **citation** metrics (off the submitted answer the terminal evidence carries), both scored
against the query's relevance judgements (qrels).

``submit_answer`` is the env's ``score`` **terminal**: submitting **seals** the episode, then
the env's ``finalize`` hook runs the LLM judge on the frozen submission (seal-before-verdict),
so a graded verdict only ever exists for an already-sealed, un-continuable episode — the agent
can never grade, read the verdict, then edit and re-grade. ``finalize`` returns core-owned,
**sanitized** :class:`~shogym.serve.lifecycle.TerminalEvidence` (a public ``correct`` verdict
only — never the judge's reasoning, extracted answer, or exception text); the pure ``_verify``
scores from that evidence, plus retrieval recall off the recorded ``search`` steps and citation
metrics off the submitted answer the same terminal evidence carries.

The env **describes** a task (the query + the served tool manifest + a search/turn horizon),
**serves** ``search`` / ``get_document`` / ``submit_answer`` over MCP (see :mod:`mcp_server`),
and **verifies** the recorded trajectory + terminal evidence. No agent loop, model, or prompt
lives here — a harness drives the tools.

This module imports **nothing** heavy at load time — not ``datasets``, not ``openai``, not
``pyserini`` — so ``import shogym`` (which imports this module to register the env) stays offline.
The dataset (decrypted **in memory**; never persisted) loads when the *registered* env is
constructed, which means a default ``make`` cold-downloads the query dataset and both qrel files;
pass ``tasks`` to skip that. What ``make`` and ``describe`` do **not** do is provision or open
the multi-GB BM25 index — that is deferred to session start. The judge's client is built only
when it is first called.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from shogym.core import Env
from shogym.envs.browsecomp_plus import mcp_server as bcp_mcp_server
from shogym.envs.browsecomp_plus.judge import DEFAULT_JUDGE_MODEL, Judge, OpenAIJudge
from shogym.envs.browsecomp_plus.metrics import (
    compute_citation_metrics,
    extract_citations_from_response,
    retrieval_recall,
)
from shogym.envs.browsecomp_plus.searcher import Searcher
from shogym.envs.registration import register
from shogym.mcp import MCPServerSpec
from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

SEARCH_TOOL_NAME = "search"
SUBMIT_TOOL_NAME = "submit_answer"

# Fraction of queries assigned to the train split; the rest is held-out test (positional 80/20,
# like the HLE/wordle ports — BrowseComp-Plus declares no train/test holdout).
TRAIN_FRACTION = 0.8

# Default budget on tool calls per episode (search + get_document + submit_answer). Deep-Research
# runs issue many searches; the horizon caps a runaway harness. Reaching it with no valid
# submission scores `correct=False` (zero-unsubmitted — see `finalize`).
DEFAULT_MAX_TURNS = 50

# Confidence is a 0–100 integer (HLE convention); clamp out-of-range values rather than reject.
MIN_CONFIDENCE = 0
MAX_CONFIDENCE = 100
DEFAULT_CONFIDENCE = 100

BCP_SPEC = MCPServerSpec(
    name="browsecomp_plus",
    transport="in_process",
    module="shogym.envs.browsecomp_plus.mcp_server",
)

_BASE_INSTRUCTIONS = (
    "You are answering a research question from BrowseComp-Plus by retrieving evidence from a "
    "fixed document corpus — you have no web access. Use the `search` tool to find relevant "
    "documents (each hit has a `docid`, `score`, and `snippet`) and `get_document` to read a "
    "document's full text. Reason over the evidence, then call `submit_answer` exactly once with "
    "your final `answer` and a `confidence` from 0 to 100. Cite the docids you relied on in your "
    "answer as `[docid]`. Submitting grades your answer and ends the episode — there is no second "
    "submission and no further step to take (do not call `terminate` afterward), so commit to "
    "your best answer."
)


class BrowseCompPlusEnv(Env):
    """BrowseComp-Plus wrapped as an shogym env.

    Config (all optional, via ``shogym.make("browsecomp_plus", config=...)`` / ``env_config``):
      - ``task_split``: ``"train"`` (default) or ``"test"`` — a positional 80/20 slice.
      - ``tasks``: an explicit task list (each ``{"query", "answer", "qrel_evidence", ...}``).
        When given, the encrypted dataset is **not** downloaded — this is how offline tests
        construct the env (no network, no decryption).
      - ``searcher``: an injected :class:`~shogym.envs.browsecomp_plus.searcher.Searcher` (an
        ``InMemorySearcher`` for offline tests). Default: a pyserini ``BM25Searcher`` over the
        lazily-provisioned prebuilt index.
      - ``judge``: an injected :class:`~shogym.envs.browsecomp_plus.judge.Judge` (a scripted judge
        for offline tests). Default: :class:`~shogym.envs.browsecomp_plus.judge.OpenAIJudge`.
      - ``judge_model`` / ``judge_base_url``: the default judge's model id + endpoint (set both
        to a vLLM Qwen3-32B to grade with upstream's model; the sampling still differs).
      - ``k`` / ``snippet_max_tokens``: retrieval knobs (upstream defaults 5 / 512).
      - ``max_turns``: the tool-call horizon.
    """

    mcp_servers = (BCP_SPEC,)
    function_name = "researcher"
    # `submit_answer` is the env's single `score` terminal: the serve layer validates its args,
    # atomically seals the episode, then runs `finalize` (the LLM judge) on the sealed
    # submission. Every non-score tool leaves the episode OPEN.
    score_terminal_tool = SUBMIT_TOOL_NAME

    def __init__(
        self,
        task_split: str = "train",
        tasks: Optional[List[Dict[str, Any]]] = None,
        searcher: Optional[Searcher] = None,
        judge: Optional[Judge] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_base_url: Optional[str] = None,
        k: int = bcp_mcp_server.DEFAULT_K,
        snippet_max_tokens: int = bcp_mcp_server.DEFAULT_SNIPPET_MAX_TOKENS,
        max_turns: int = DEFAULT_MAX_TURNS,
    ) -> None:
        if task_split not in ("train", "test"):
            raise ValueError(f"unknown task_split {task_split!r}; expected 'train' or 'test'")
        self._task_split = task_split
        self._injected_searcher = searcher
        self._searcher: Optional[Searcher] = searcher
        self._judge = judge
        self._judge_model = judge_model
        self._judge_base_url = judge_base_url
        self._k = int(k)
        self._snippet_max_tokens = int(snippet_max_tokens)
        # Per-session grading inputs `finalize` reads after the seal (the judge, question, and
        # gold answer). Held on the env — not the served MCP server — because grading no longer
        # runs inside a tool handler. Keyed by session_id so concurrent episodes stay isolated.
        self._finalize_state: Dict[str, Dict[str, Any]] = {}
        self._tasks: List[Dict[str, Any]] = (
            list(tasks) if tasks is not None else _load_default_tasks(task_split)
        )
        self.function = FunctionConfig(example_system_template=_BASE_INSTRUCTIONS)
        super().__init__(horizon=max_turns, num_tasks=len(self._tasks))

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if not self._tasks:
            raise ValueError("browsecomp_plus env has no tasks loaded")
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._tasks)))
        if not 0 <= task_idx < len(self._tasks):
            # Reject negatives too: Python's negative indexing would silently serve a valid
            # record while the serve layer records task id `-1` — a misattributed run.
            raise ValueError(
                f"Task index {task_idx} is out of range for {len(self._tasks)} tasks"
            )
        task = self._tasks[task_idx]
        return {
            "task_idx": task_idx,
            "query_id": str(task.get("query_id", task_idx)),
            "query": str(task.get("query", "")),
            "answer": str(task.get("answer", "")),
            "qrel_gold": [str(d) for d in task.get("qrel_gold", [])],
            "qrel_evidence": [str(d) for d in task.get("qrel_evidence", [])],
            "split": self._task_split,
        }

    # ----- session lifecycle -----

    def _searcher_for_session(self) -> Searcher:
        """The injected searcher, or a lazily-built default ``BM25Searcher`` (built once, shared).

        Deferred to session start (not env construction), so ``shogym.make(...)``, the tool
        manifest, and ``describe()`` stay offline: building the BM25 searcher opens the prebuilt
        Lucene index (Java 21 + pyserini) and provisions the corpus, which must not happen just to
        read the contract. The searcher is read-only, so one instance safely backs every episode."""
        if self._searcher is not None:
            return self._searcher
        from shogym.envs.browsecomp_plus.data import bm25_index_path
        from shogym.envs.browsecomp_plus.searcher import BM25Searcher

        self._searcher = BM25Searcher(bm25_index_path())
        return self._searcher

    def _judge_for_session(self) -> Judge:
        """The injected judge, or a lazily-built default ``OpenAIJudge`` (no network yet).

        Preflighted at session start (not env construction), so making the env, probing the tool
        manifest, and ``describe()`` stay offline and keyless. The default ``OpenAIJudge`` needs
        ``OPENAI_API_KEY`` to grade answers; with no key it would fail-close every answer to
        incorrect, silently deflating the benchmark. Raise early and clearly instead — but only in
        the "forgot the key" case: an injected ``judge=`` or a ``judge_base_url`` override (a
        keyless OpenAI-compatible endpoint) opts out, which keeps offline tests network-free."""
        if self._judge is not None:
            return self._judge
        import os

        if not self._judge_base_url and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "BrowseComp-Plus's default judge (OpenAIJudge) needs OPENAI_API_KEY to grade "
                "answers, but it is not set. Set OPENAI_API_KEY, pass judge_base_url=... for a "
                "keyless OpenAI-compatible endpoint (e.g. a vLLM Qwen3-32B), or inject judge=... . "
                "Without it every answer is scored incorrect."
            )
        return OpenAIJudge(model=self._judge_model, base_url=self._judge_base_url)

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        # The served tools only need the searcher; the judge + gold answer are held on the env for
        # `finalize` (grading runs after the seal, not in a served handler).
        bcp_mcp_server.begin_session(
            session_id,
            searcher=self._searcher_for_session(),
            k=self._k,
            snippet_max_tokens=self._snippet_max_tokens,
        )
        self._finalize_state[session_id] = {
            "question": task["query"],
            "correct_answer": task["answer"],
            "judge": self._judge_for_session(),
        }

    def _end_session(self, session_id: str) -> None:
        bcp_mcp_server.end_session(session_id)
        self._finalize_state.pop(session_id, None)

    # ----- describe: surface this task's query in the instructions -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        spec = super().describe(task_id)
        task = self._resolve_task(task_id)
        if task is None:
            return spec
        backend = getattr(self._injected_searcher, "search_type", None) or "BM25"
        parts = [
            _BASE_INSTRUCTIONS,
            "",
            "# Question",
            str(task.get("query", "")),
            "",
            f"(Retriever: {backend}; top-{self._k} per search. Cite supporting docids as [docid].)",
        ]
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

    # ----- finalize: run the LLM judge on the sealed submission (seal-before-verdict) -----

    # `Env.finalize` is declared as an `Optional[Callable]` attribute (default None) so the serve
    # layer can `getattr`/`callable`-probe it; overriding it with the prescribed
    # `async def finalize(self, req)` method is exactly the opt-in the base documents, but pyright
    # reads the attribute-vs-method shape as an incompatible override. Suppress that one rule here.
    async def finalize(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, req: FinalizeRequest
    ) -> TerminalEvidence:
        """Grade the **already-sealed** submission and return core-owned terminal evidence.

        Called by the serve layer only after ``submit_answer`` has atomically sealed the episode
        (source ``explicit_tool``), or when the tool-call horizon is reached with no submission
        (source ``horizon``); an explicit ``terminate`` abort is scored by the core without
        reaching here. The LLM judge grades ``answer`` against the session's gold answer.

        **Sanitized (answer-oracle safe).** The returned ``verdict`` is public — the serve layer
        surfaces it to the agent — so it carries only ``correct``, the echoed ``confidence``, and
        (on a judge-infra failure) ``judge_error``. The judge's ``reasoning`` / ``extracted_answer``
        and any exception text are answer oracles; they go **only** to the private ``diagnostic``
        (durable store / server logs), never to the agent. A judge failure **fails closed** to
        ``correct=False`` with ``judge_error`` flagged (distinguishable in audit from an honest
        wrong answer)."""
        args = req.args or {}
        confidence = _clamp_confidence(args.get("confidence"))
        state = self._finalize_state.get(req.session_id)
        if req.source == "horizon" or req.args is None or state is None:
            # No gradeable submission reached the seal (horizon reached, or state already dropped):
            # score incorrect. Nothing to grade, no confidence to echo.
            return TerminalEvidence(
                source=req.source,
                status="ok",
                verdict={"correct": False},
                diagnostic=f"no submission to grade (source={req.source})",
            )

        answer = str(args.get("answer", ""))
        question = str(state["question"])
        gold = str(state["correct_answer"])
        judge: Judge = state["judge"]
        try:
            # The judge does a blocking network call; run it off the event loop so the loop (and
            # any finalize deadline) is not blocked. Offline scripted judges run trivially here too.
            verdict = await asyncio.to_thread(
                judge, question=question, correct_answer=gold, response=answer
            )
        except Exception as exc:  # noqa: BLE001 — a judge failure fails closed, never crashes
            # Fail closed: score incorrect and flag `judge_error` so an analyst can filter judge
            # infra failures out of honest wrong answers. The exception text is a private
            # diagnostic — never surfaced to the agent.
            return TerminalEvidence(
                source=req.source,
                status="ok",
                verdict={"correct": False, "confidence": confidence, "judge_error": True},
                diagnostic=f"judge error: {type(exc).__name__}: {exc}",
            )
        # A clean grade. `reasoning` / `extracted_answer` are answer oracles — diagnostic only.
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict={"correct": bool(verdict.correct), "confidence": confidence},
            diagnostic=(
                f"judged_by=llm_judge correct={bool(verdict.correct)} "
                f"extracted_answer={verdict.extracted_answer!r} reasoning={verdict.reasoning!r}"
            ),
        )

    # ----- verify: score from the terminal evidence + deterministic trajectory metrics -----

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: Optional[TerminalEvidence] = None,
    ) -> FeedbackCollection:
        """Score the episode: correctness from the core-owned ``evidence`` (never marker JSON),
        plus deterministic retrieval/citation metrics off the recorded trajectory + the task's
        qrels. Pure over its inputs."""
        return score_trajectory(trajectory, task, terminated=terminated, evidence=evidence)


# ----- pure scoring (module-level so it is unit-testable without a judge, searcher, or dataset) --


def score_trajectory(
    trajectory: Trajectory,
    task: Dict[str, Any],
    *,
    terminated: bool,
    evidence: Optional[TerminalEvidence] = None,
) -> FeedbackCollection:
    """Build episode feedback from the terminal ``evidence`` + the recorded trajectory. Pure.

    Correctness comes from the core-owned :class:`TerminalEvidence` the seal transaction
    produced (the judge's verdict) — not from any tool result the agent can forge. The
    retrieval/citation metrics are deterministic over the trajectory + the task's qrels.

    Emits, on termination:
      - ``correct`` (bool) — the judge's verdict (False if the episode ended with no evidence).
      - ``confidence`` (0–1) + ``calibration_error`` (``|confidence − correct|``) — only when a
        submission was graded (HLE-style calibration; omitted on a horizon/abort end).
      - ``judge_error`` (True) — only when the grade was a fail-closed judge failure, or the
        finalize transaction itself failed closed.
      - ``retrieval_recall`` — fraction of the query's evidence docids retrieved across all
        ``search`` steps (BrowseComp-Plus retrieval recall vs ``qrel_evidence``).
      - ``citation_recall`` / ``citation_precision`` / ``num_citations`` — cited-docid metrics vs
        ``qrel_evidence`` (citations parsed from the submitted answer as ``[docid]``).

    Deterministic metrics are emitted whenever the query has evidence qrels, even on a premature
    end — a run that retrieved well but never answered still gets a recall score."""
    fb = FeedbackCollection()
    if not terminated:
        return fb

    evidence_qrels = [str(d) for d in task.get("qrel_evidence", [])]
    verdict = evidence.verdict if evidence is not None else {}
    # The submission that was graded (validated args the serve layer stamped onto the evidence).
    # Present only for an `explicit_tool` (submit) terminal; None for horizon/abort.
    submission = evidence.args if evidence is not None else None

    # --- correctness (core-owned judge verdict) ---
    correct = bool(verdict.get("correct"))
    fb.episode.append(EpisodeFeedback(name="correct", value=correct))
    if verdict.get("judge_error") or (evidence is not None and evidence.finalize_error):
        # The judge (or the finalize transaction) failed; `correct=False` is fail-closed, not a
        # genuine wrong answer — flag it so it can be filtered from honest zeros.
        fb.episode.append(EpisodeFeedback(name="judge_error", value=True))

    # --- confidence / calibration (only when there was a graded submission) ---
    answer_text = ""
    if submission is not None:
        answer_text = str(submission.get("answer", ""))
        confidence = _confidence_fraction(submission.get("confidence"))
        fb.episode.append(EpisodeFeedback(name="confidence", value=confidence))
        fb.episode.append(
            EpisodeFeedback(
                name="calibration_error",
                value=abs(confidence - (1.0 if correct else 0.0)),
            )
        )

    # --- deterministic retrieval + citation metrics (only meaningful with evidence qrels) ---
    if evidence_qrels:
        retrieved = _retrieved_docids(trajectory)
        fb.episode.append(
            EpisodeFeedback(
                name="retrieval_recall", value=retrieval_recall(retrieved, evidence_qrels)
            )
        )
        cited = extract_citations_from_response(answer_text)
        cm = compute_citation_metrics(cited, evidence_qrels)
        fb.episode.append(EpisodeFeedback(name="citation_recall", value=cm["recall"]))
        fb.episode.append(EpisodeFeedback(name="citation_precision", value=cm["precision"]))
        fb.episode.append(EpisodeFeedback(name="num_citations", value=cm["num_citations"]))

    return fb


def _retrieved_docids(trajectory: Trajectory) -> List[str]:
    """Union of docids returned across every ``search`` step (BrowseComp-Plus ``retrieved_docids``).

    Reads each ``search`` result (a JSON list of ``{docid, ...}``) off the trajectory. Malformed
    or non-``search`` results are skipped, never raised on."""
    docids: set[str] = set()
    for step in trajectory:
        if step.tool != SEARCH_TOOL_NAME:
            continue
        try:
            payload = json.loads(step.result)
        except (TypeError, ValueError):
            continue
        if not isinstance(payload, list):
            continue
        for hit in payload:
            if isinstance(hit, dict) and "docid" in hit:
                docids.add(str(hit["docid"]))
    return list(docids)


def _clamp_confidence(confidence: Optional[Any]) -> int:
    """Coerce ``confidence`` to an int in [0, 100]; junk falls back to 100 (the echoed verdict value)."""
    try:
        value = int(confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, value))


def _confidence_fraction(confidence: Any) -> float:
    """Coerce a 0–100 confidence to a [0, 1] fraction; junk/absent defaults to 1.0."""
    try:
        value = float(confidence)
    except (TypeError, ValueError):
        value = 100.0
    return max(0.0, min(100.0, value)) / 100.0


def _load_default_tasks(task_split: str) -> List[Dict[str, Any]]:
    """Load the registered env's real tasks (decrypt queries in memory + join qrels), then split."""
    from shogym.envs.browsecomp_plus.data import load_default_tasks

    return _split_tasks(load_default_tasks(), task_split)


def _split_tasks(tasks: List[Dict[str, Any]], split: str) -> List[Dict[str, Any]]:
    """Positionally split the task list 80/20 into ``train`` / ``test`` (indices are split-relative)."""
    if split not in ("train", "test"):
        raise ValueError(f"unknown task_split {split!r}; expected 'train' or 'test'")
    cut = int(len(tasks) * TRAIN_FRACTION)
    return tasks[:cut] if split == "train" else tasks[cut:]


@register("browsecomp_plus")
class BrowseCompPlusDefault(BrowseCompPlusEnv):
    """The canonical BrowseComp-Plus env (BM25 retriever, LLM judge)."""
