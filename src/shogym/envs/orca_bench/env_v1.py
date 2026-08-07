"""``orca_bench``: ORCA-bench on the env-as-center core, root-cause analysis over telemetry.

One task hands an SRE agent a user-visible complaint and a reported time, and asks for a
structured incident RCA report. The evidence is a recorded observability stack (metrics, traces
and logs from a frozen snapshot, behind Grafana) that the agent queries through a served shell;
the report is graded by the LLM judge that ships inside every task, against one rubric per
plausible root cause.

  - **describe(task_id)** returns the task's ``instruction.md``, and nothing else. The task's
    ``task.toml`` carries the full answer, so the redaction is the load-bearing property here,
    not a nicety (see :mod:`shogym.envs.orca_bench.tasks`).
  - **serve** exposes ``exec`` / ``read_file`` / ``write_file`` on the task container, plus
    ``submit_report`` as the ``score`` terminal. **Phase 2**: constructing the backend raises
    today (see :mod:`shogym.envs.orca_bench.backend`).
  - **finalize (sealed)** runs the task's own verifier over ``/app/report.md`` and turns its
    two output files into a verdict, mapping a judge failure to an **explicit** judge-error grade
    rather than the honest-looking zero upstream writes.
  - **verify (pure)** scores that evidence: ``reward`` (the judge's 0-1 rubric score) and
    ``success`` (the benchmark's strict all-causes ``rca_accuracy``).

This module imports nothing heavy at load time, neither the dataset nor the backend, so
``import shogym`` (which imports it to register the env) stays offline. The dataset downloads on
first *construction*; the judge's key is required only when an episode is *served*.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from shogym.core import Env
from shogym.envs.orca_bench.judge import (
    DEFAULT_JUDGE_EFFORT,
    DEFAULT_JUDGE_MODEL,
    JudgeConfig,
    OrcaVerdict,
    ResolvedJudge,
    judge_provenance,
    parse_verdict,
)
from shogym.envs.orca_bench.tasks import OrcaTaskRef
from shogym.envs.registration import register
from shogym.mcp import MCPServerSpec
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

if TYPE_CHECKING:
    # Only for annotations, imported lazily inside `finalize` so `import shogym` never pulls in
    # the serve layer (mirrors `shogym.core` and the frontier_bench port).
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence

SUBMIT_TOOL_NAME = "submit_report"

# The step budget. Investigating a telemetry stack is many small queries; upstream gives the agent
# a 3600 s wall clock rather than a step cap, so this is generous by design.
DEFAULT_MAX_STEPS = 400

_INSTRUCTIONS = """\
You are an on-call site reliability engineer working inside a task container over a served shell. \
Call `describe` first to read the full task instruction: it states the reported issue, the time \
it was reported, the current time, where the application source is, and the exact four-section \
format your report must take.

Act on the container through these tools:
- `exec(command)` runs a shell command. The telemetry lives behind the Grafana HTTP API the \
instruction names (metrics, traces and logs are all queryable through it); the application source \
is on disk. Returns JSON with `ok`, `exit_code`, `stdout`, `stderr`.
- `read_file(path)` / `write_file(path, content)` read or write a file in the container.

Write your incident RCA report to `/app/report.md`, then call `submit_report`. That **ends the \
episode**: it seals the run and the task's own judge scores the report against the ground-truth \
rubrics. It is one-shot and authoritative (you cannot read the verdict and revise), so finish \
the report first. Note that the reported time is not necessarily when the incident began, there \
may be several root causes, and there may be no incident at all (in which case write an empty \
report)."""


@dataclass(frozen=True)
class _Grading:
    """What an episode fixed about its own grading when it started.

    Both fields are properties of the episode, not of the moment it is graded: a control task
    scores differently (see :meth:`OrcaVerdict.success`), and the judge was resolved when the
    verifier was equipped with it.
    """

    is_control: bool
    judge: ResolvedJudge


@register("orca_bench")
class OrcaBenchEnv(Env):
    """ORCA-bench as a shogym env: 755 telemetry root-cause tasks, judged by the task's verifier.

    Tasks are selectable by **index** (``0..N-1``, the name-sorted order) or by **name**.

    Config (all optional, via ``shogym.make("orca_bench", config=...)`` / ``env_config``):
      - ``task``: the *default* task, by name or index, used when a call omits the selector.
      - ``dataset_dir``: an already-extracted dataset directory. When given, nothing is
        downloaded, which is how offline tests construct the env. Otherwise the pinned revision
        is fetched once into ``~/.cache/shogym/orca_bench`` (see
        :mod:`shogym.envs.orca_bench.dataset`).
      - ``tasks``: an explicit list of :class:`~shogym.envs.orca_bench.tasks.OrcaTaskRef` to
        expose (a slice of the index, e.g. one difficulty tier or one snapshot's group).
      - ``judge_model`` / ``judge_effort`` / ``judge_base_url``: how the task's own verifier is
        invoked. The model is validated at construction; the key is required only when an
        episode is served.
      - ``max_steps``: hard cap on tool calls per episode (the shogym horizon).
    """

    function_name = "exec"
    # `submit_report` is the score terminal: a call seals the episode (validate -> seal ->
    # finalize), so a verdict only ever exists for an already-sealed, un-continuable episode.
    score_terminal_tool = SUBMIT_TOOL_NAME

    def __init__(
        self,
        task: Optional[Union[str, int]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        tasks: Optional[List[OrcaTaskRef]] = None,
        judge_model: str = DEFAULT_JUDGE_MODEL,
        judge_effort: str = DEFAULT_JUDGE_EFFORT,
        judge_base_url: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        from shogym.envs.orca_bench.judge import validate_judge_config

        self._judge = JudgeConfig(
            model=judge_model, effort=judge_effort, base_url=judge_base_url
        )
        # A judge that cannot grade is a configuration error, not a run-time surprise: reject it
        # here, before a dataset download or a single episode. Keyless on purpose (see
        # `preflight_judge`), so construction stays offline.
        validate_judge_config(self._judge)
        self._refs: List[OrcaTaskRef] = (
            list(tasks) if tasks is not None else _load_refs(dataset_dir)
        )
        if not self._refs:
            raise ValueError("orca_bench env has no tasks loaded")
        # A task list is a set of identities. Two entries under one name collapse here and in
        # `_position`, and then the second one answers to the first one's id: `load_task` reports
        # it as the other task and `describe` publishes that other task's name in the footer. The
        # error names both sources, since the whole difficulty is that they look alike.
        self._by_name: Dict[str, OrcaTaskRef] = {}
        for ref in self._refs:
            clash = self._by_name.get(ref.name)
            if clash is not None:
                raise ValueError(
                    f"orca_bench was given the task {ref.name!r} twice, from {clash.task_dir} "
                    f"(dataset index {clash.dataset_index}) and {ref.task_dir} (dataset index "
                    f"{ref.dataset_index}). A task list is a set of identities: the second copy "
                    "would answer to the first one's id."
                )
            self._by_name[ref.name] = ref
        # The served task id is a position in THIS env's task list, which is a slice of the
        # dataset whenever `tasks=` is given. `ref.dataset_index` is the position in the full 755
        # and is never used to select: an id resolved in the wrong space either raises or, worse,
        # silently selects a different task than the one the backend is running.
        self._position = {ref.name: position for position, ref in enumerate(self._refs)}
        # Resolve the default selector eagerly so a bad config fails at construction, not
        # mid-serve. (`_resolve_ref(None)` reads `_default_ref`, so it is set before any call.)
        self._default_ref = self._refs[0] if task is None else self._resolve_ref(task)
        self._max_steps = max_steps
        # Per-episode grading context, keyed by session id so one env instance safely backs many
        # concurrent episodes. `finalize` is handed only a session id, and both of these are
        # things about the episode rather than about the moment it is graded: `success` means
        # something different on a control task (see `OrcaVerdict.success`), and the judge is
        # resolved when the verifier is equipped, not re-read from the environment at readback.
        self._grading: Dict[str, _Grading] = {}
        self.function = FunctionConfig(example_system_template=_INSTRUCTIONS)
        self.mcp_servers = (
            MCPServerSpec(
                name="orca_bench",
                transport="in_process",
                module="shogym.envs.orca_bench.mcp_server",
            ),
        )
        # One step per tool call, plus room for `submit_report`.
        super().__init__(horizon=max_steps + 2, num_tasks=len(self._refs))

    # ----- task selection -----

    @property
    def refs(self) -> List[OrcaTaskRef]:
        """The tasks this env exposes, in name-sorted order. Position is the public task id."""
        return list(self._refs)

    def position_of(self, ref: OrcaTaskRef) -> int:
        """This env's public id for ``ref``: its position in the served task list.

        Not ``ref.dataset_index``, which is the position in the full 755 and only matches when
        the env exposes the whole dataset."""
        return self._position[ref.name]

    def _resolve_ref(self, selector: Optional[Union[str, int]]) -> OrcaTaskRef:
        """Resolve a task name, a position (int or digit string), or ``None`` (the default task).

        A numeric selector is always a position in **this env's** task list, never a
        ``dataset_index``. An out-of-range position raises rather than silently serving another
        task under a bogus public id: the serve layer would record the episode as that id while
        running a different task, and ``describe(id)`` would then refuse to resolve it."""
        if selector is None:
            return self._default_ref
        if isinstance(selector, bool):  # bool is an int subclass, never a valid selector
            raise ValueError(f"invalid orca_bench task selector {selector!r}")
        idx: Optional[int] = None
        if isinstance(selector, int):
            idx = selector
        elif isinstance(selector, str):
            text = selector.strip()
            if text.lstrip("-").isdigit():
                idx = int(text)
            elif text in self._by_name:
                return self._by_name[text]
            else:
                raise ValueError(f"unknown orca_bench task {selector!r}")
        else:
            raise ValueError(f"invalid orca_bench task selector {selector!r}")
        if idx < 0 or idx >= len(self._refs):
            raise ValueError(
                f"task index {idx} is out of range for {len(self._refs)} tasks (use index "
                f"0..{len(self._refs) - 1} or a task name)"
            )
        return self._refs[idx]

    # ----- task lifecycle -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        ref = self._resolve_ref(task_idx)
        return {
            # The id this episode is served and recorded under. The serve layer echoes it back
            # into `describe()` when the caller named no task, so it MUST be a position in this
            # env's task list; `dataset_index` rides along as provenance only.
            "task_idx": self.position_of(ref),
            "dataset_index": ref.dataset_index,
            "task_name": ref.name,
            "difficulty": ref.difficulty,
            "section": ref.section,
            "is_control": ref.is_control,
            "snapshot": ref.snapshot,
            "task_dir": str(ref.task_dir),
        }

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        from shogym.envs.orca_bench import mcp_server
        from shogym.envs.orca_bench.judge import environment_snapshot, preflight_judge

        # One reading of the environment for this whole episode. The preflight decision, the
        # environment the verifier is equipped with, and the endpoint recorded with the score are
        # three statements about one run, so they are derived from one snapshot rather than from
        # three reads of a mapping the process may change in between.
        environ = environment_snapshot()
        # The key is needed to grade, and grading happens at the end of a long, expensive
        # episode: check it before the stack comes up, not after the report is written.
        preflight_judge(self._judge, environ=environ)
        # Resolve the judge HERE, where the verifier is equipped with it: what is audited has to
        # be the reading this episode was given, not the one in force whenever the score is read
        # back.
        resolved = self._judge.resolve(environ)
        self._grading[session_id] = _Grading(
            is_control=bool(task.get("is_control")), judge=resolved
        )
        mcp_server.begin_session(session_id, task_dir=Path(task["task_dir"]), judge=resolved)

    def _end_session(self, session_id: str) -> None:
        from shogym.envs.orca_bench import mcp_server

        self._grading.pop(session_id, None)
        mcp_server.end_session(session_id)

    # ----- describe: the task's instruction.md, and nothing else -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        """Publish the task contract: upstream's ``instruction.md`` plus a provenance footer.

        The footer carries the task's public id and the dataset pin, never a label from
        ``task.toml``. ``section`` alone would give away whether a task is a control (a task with
        no incident to find), and the event times bracket the answer outright, so nothing from
        ``[metadata]`` is published here."""
        from shogym.envs.orca_bench.dataset import DATASET, DATASET_REVISION

        spec = super().describe(task_id)
        ref = self._resolve_ref(task_id)
        parts = [
            ref.instructions().rstrip(),
            "",
            "# This run",
            # The published index is this env's public id (the one a harness passes back to
            # `describe`), not the canonical dataset position.
            f"- task: {ref.name} (index {self.position_of(ref)})",
            f"- pinned: {DATASET} revision {DATASET_REVISION}",
        ]
        return spec.model_copy(update={"instructions": "\n".join(parts)})

    # ----- finalize: grade the sealed episode -----

    async def finalize(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, req: "FinalizeRequest"
    ) -> "TerminalEvidence":
        """Run the task's verifier over the sealed episode's report and return the verdict.

        (``Env.finalize`` is declared a typed *attribute* the serve layer reads via
        ``getattr``/``callable``; overriding it with this method is the documented opt-in, so the
        variable-vs-method override report is a false positive.)

        The verifier is the task's own LLM judge. Its two output files are parsed by
        :func:`~shogym.envs.orca_bench.judge.parse_verdict`, which fails **loudly**: a judge that
        raised, or that returned nothing scoreable, becomes ``judge_error=True`` rather than the
        reward 0.0 upstream writes for both that and a genuinely wrong report. The judge's prompt,
        reasoning and rubrics are answer oracles and stay in the private diagnostic.

        The verdict also carries **which judge produced it**, because changing the judge changes
        the scoring function: a bare number cannot be compared with another run's, or re-read
        later when a model id has moved on. It rides on the verdict rather than only in this
        diagnostic, since the diagnostic is private to the durable store and never reaches the
        trace a result is read back from.
        """
        from shogym.envs.orca_bench import mcp_server
        from shogym.serve.lifecycle import TerminalEvidence

        grading = self._grading.get(req.session_id) or _Grading(
            # No session state: the episode is being graded without having been begun through
            # this env instance, so there is no captured resolution to honor and the current
            # environment is the only reading available. Recorded as such rather than omitted.
            is_control=False,
            judge=self._judge.resolve(),
        )
        payload = await asyncio.to_thread(mcp_server.finalize_session, req.session_id)
        if payload is None:
            # No live session to grade. On the sealed path finalize runs before teardown, so this
            # is an infra failure, not an honest zero: fail closed and say which.
            verdict = OrcaVerdict(
                reward=0.0,
                rca_accuracy=None,
                hallucinate_any=None,
                mode="",
                rca_depth=None,
                judge_error=True,
                judge_error_message="finalize: no live session to verify",
            )
        else:
            verdict = parse_verdict(
                payload.get("reward"),
                payload.get("details"),
                submission_error=str(payload.get("submission_error") or ""),
            )
        provenance = judge_provenance(verdict, grading.judge)
        return TerminalEvidence(
            source=req.source,
            status="finalize_error" if verdict.judge_error else "ok",
            verdict=public_verdict(verdict, is_control=grading.is_control, judge=grading.judge),
            # The judge error text and the rubric depth are diagnostics, not a grade: they belong
            # in the durable store, never on the wire to the agent. The judge provenance is
            # repeated here so a server-side log line stands on its own.
            diagnostic=(
                f"judge {provenance['judge_model']}/{provenance['judge_effort']} "
                f"endpoint={provenance['judge_endpoint']} mode={verdict.mode or '?'} "
                f"rca_depth={verdict.rca_depth} error={verdict.judge_error_message or 'none'} "
                # Which rule derived `success`, which the published numbers cannot show: a
                # control task passes on reward alone, with `rca_accuracy` structurally 0. This
                # is the hidden label, so it stays here, where the serve layer drops it before
                # anything reaches the caller or the trace.
                f"success_rule={'control' if grading.is_control else 'incident'}"
            ),
        )

    # ----- verify -----

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: "Optional[TerminalEvidence]" = None,
    ) -> FeedbackCollection:
        """Score exclusively from the core-owned terminal evidence (never marker JSON)."""
        if not terminated:
            return FeedbackCollection()
        return score_evidence(evidence)


# ----- pure scoring (module-level so it is unit-testable without a stack or a judge) -----


def public_verdict(
    verdict: OrcaVerdict, *, is_control: bool, judge: ResolvedJudge
) -> Dict[str, Any]:
    """The public-safe verdict the serve layer commits as terminal evidence.

    Carries the numbers the benchmark publishes and nothing that reveals the rubrics: the 0-1
    reward, the strict all-causes ``rca_accuracy``, the hallucination flag, the derived
    ``success``, and whether the grade is a judge error at all.

    ``is_control`` decides how ``success`` is derived and is not published as a field; it stays in
    the private diagnostic, which never leaves the durable store. That is not a claim that the
    class is hidden afterwards, and the docstring should not pretend otherwise: a passing control
    shows ``success`` true beside ``rca_accuracy`` false, which an incident task cannot do, and
    ``hallucinate_any`` is absent exactly on controls. Publishing upstream's per-task metrics
    publishes facts about the task's class. What redaction protects is the attempt itself, before
    and during, which is ``describe``'s job and is intact; see the env README on why the port
    accepts the after-the-fact disclosure instead of distorting the numbers.

    Plus the judge provenance, which is not optional: ``judge`` is a required argument precisely
    so a score cannot be published without saying which scoring function produced it. It is a
    :class:`~shogym.envs.orca_bench.judge.ResolvedJudge` rather than the configuration, so the
    endpoint published here is the one the episode's verifier was equipped with.
    """
    out: Dict[str, Any] = {
        "reward": verdict.reward,
        "success": verdict.success(is_control=is_control),
        "judge_error": verdict.judge_error,
        **judge_provenance(verdict, judge),
    }
    if verdict.submission_error:
        # The agent's own artifact, described back to it: not an oracle, and the one thing that
        # explains a zero it may not otherwise understand.
        out["submission_error"] = verdict.submission_error
    if verdict.rca_accuracy is not None:
        out["rca_accuracy"] = verdict.rca_accuracy
    if verdict.hallucinate_any is not None:
        out["hallucinate_any"] = verdict.hallucinate_any
    return out


def score_evidence(evidence: "Optional[TerminalEvidence]") -> FeedbackCollection:
    """Build episode feedback from the core-owned terminal evidence. Pure.

    A graded episode emits ``reward`` (the judge's 0-1 rubric score), ``success`` (the per-task
    binary), ``verified=True``, and, where the benchmark defines them, ``rca_accuracy`` and
    ``hallucinate_any``, so the published per-tier and hallucination numbers are reproducible
    from a trace.

    It also emits the **judge provenance** (``judge_model`` / ``judge_effort`` /
    ``judge_endpoint``). A score is only meaningful next to the scoring function that produced it,
    and the trace is where a result is read back from, so the answer to "scored by what?" has to
    be on that surface rather than in the private diagnostic.

    A **judge error** emits ``judge_error=True`` alongside ``reward=0.0``: upstream records the
    same 0.0 for a failed judge and a wrong report, and this is the flag that tells them apart, so
    grading-infra failures can be filtered out instead of averaged in. A **failed submission** (no
    report, an unreadable one, one too large to judge) is the opposite case and emits
    ``submission_error`` with ``verified=True``: it is the agent's own doing and counts as the zero
    it is, or an agent could leave the exclusion path open for itself by writing nothing. An episode that ended with
    no verdict at all (an ``abort``, or a fail-closed finalize) emits ``verified=False``.
    """
    fb = FeedbackCollection()
    verdict: Dict[str, Any] = (
        evidence.verdict
        if (evidence is not None and isinstance(evidence.verdict, dict))
        else {}
    )
    if "reward" not in verdict:
        # No verdict: an abort (`terminate`/close before `submit_report`) or a fail-closed
        # finalize the serve layer synthesized.
        fb.episode.append(EpisodeFeedback(name="reward", value=0.0))
        fb.episode.append(EpisodeFeedback(name="success", value=False))
        fb.episode.append(EpisodeFeedback(name="verified", value=False))
        if evidence is not None and evidence.finalize_error:
            fb.episode.append(EpisodeFeedback(name="judge_error", value=True))
        return fb

    judge_error = bool(verdict.get("judge_error")) or (
        evidence is not None and evidence.finalize_error
    )
    fb.episode.append(EpisodeFeedback(name="reward", value=_as_float(verdict.get("reward"))))
    fb.episode.append(EpisodeFeedback(name="success", value=bool(verdict.get("success"))))
    fb.episode.append(EpisodeFeedback(name="verified", value=not judge_error))
    if judge_error:
        fb.episode.append(EpisodeFeedback(name="judge_error", value=True))
    for key in ("judge_model", "judge_effort", "judge_endpoint", "submission_error"):
        value = verdict.get(key)
        if value:
            fb.episode.append(EpisodeFeedback(name=key, value=str(value)))
    for key in ("rca_accuracy", "hallucinate_any"):
        if key in verdict:
            fb.episode.append(EpisodeFeedback(name=key, value=bool(verdict[key])))
    return fb


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _load_refs(dataset_dir: Optional[Union[str, Path]]) -> List[OrcaTaskRef]:
    """Index the dataset, downloading the pinned revision on a cold cache (lazy import).

    The provisioned cache is indexed by the **pinned identities**, not by whatever directories
    are in it. `ensure_dataset` already refuses a cache that is not exactly the revision, so this
    is the second half of the same statement: what a task index contains, and therefore how many
    tasks there are and what every id refers to, is decided by the pin rather than by the
    filesystem. An explicit ``dataset_dir`` is a directory the caller vouches for, so it is
    indexed as given."""
    from shogym.envs.orca_bench.dataset import ensure_dataset, pinned_manifest
    from shogym.envs.orca_bench.tasks import load_index

    if dataset_dir is not None:
        return load_index(Path(dataset_dir).expanduser())
    return load_index(ensure_dataset(), names=pinned_manifest())


__all__ = [
    "DEFAULT_MAX_STEPS",
    "SUBMIT_TOOL_NAME",
    "OrcaBenchEnv",
    "public_verdict",
    "score_evidence",
]
