"""``receipts_v1``: one side of an admitted instance, served as an ordinary task.

The environment serves task A (or task B) of a family drawn from a frozen bank: a
clerical table and an instruction that leaves every axis of the hidden convention
undetermined. The agent files one line per record and the episode ends. Filing is
the score terminal, so the episode is sealed before it is graded and an agent can
never grade, read the verdict and revise.

The environment does not deliver a receipt, and the terminal's CONTENT returns no
verdict: it says the filing landed and how many rows it named. The three cells of a
fork are made in one act, after the filing seals, by the bank's own render path (the
record is committed once; a simultaneous first caller may render and discard), and an
experiment decides which one a branch is served. An environment that put the grade
in the terminal's content would be delivering a receipt in every arm, including the
arm that is supposed to be empty.

A TOOL RESULT HAS TWO CHANNELS AND NEITHER CARRIES THE SCORE. The serve layer
attaches episode feedback to the terminal result's ``_meta["shogym/feedback"]``
sidecar by default, which crosses into the agent's own process even where a host
keeps it out of a transcript, and an exact score narrows the drawn convention: on the
ledger a perfect one names it outright. So this env declares
``inband_terminal_feedback = False`` and the sidecar carries the terminate flag and
nothing else. The scalar is still verified, still written to the trace and still
returned by ``evaluate``. The served tests assert the whole result, both channels.

WHAT STAYS CONTROLLER-SIDE. The env holds a bank, and a bank holds the master key,
the drawn conventions and the answer keys. Only the task text and the opaque task
identifier cross to the agent. Nothing here serializes a convention outward. The task
text is itself a stable name for one instance: the surface a lineage sees is a pure
function of the ordinal, so a lineage carrying memory across episodes can recognize
an instance it has already been served, which is the side channel this v0 records
rather than closes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from dataclasses import dataclass

from shogym.core import Env
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts.mcp_server import SUBMIT_TOOL_NAME
from shogym.envs.receipts import streams
from shogym.envs.receipts.protocol import Generator, Instance, NoFiling, SealedSubmission
from shogym.envs.receipts.receipt_ast import GRADED
from shogym.envs.receipts.registry import (
    bank_path,
    bundle_dir,
    bundles,
    is_fixture,
    load_generator,
)
from shogym.envs.registration import register
from shogym.mcp import MCPServerSpec
from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

#: One action ends the episode: the filing. Reaching the horizon with no filing
#: scores zero, and records that no filing arrived rather than a wrong one.
HORIZON = 1

DEFAULT_GENRE = "ledger"
RECEIPTS_SPEC = MCPServerSpec(
    name="receipts",
    transport="in_process",
    module="shogym.envs.receipts.mcp_server",
)

_BASE_INSTRUCTIONS = (
    "You are the duty clerk on a continuing engagement. Read the schedule and the policy "
    "extract below, then call `submit_filing` exactly once with `filing`: one line per "
    "record, the record id, a comma, and the value, in the order the records appear. A "
    "record you are leaving empty still gets a line, with nothing after the comma.\n"
    "Filing ends the episode. There is no second filing and no feedback here, so file your "
    "best answer."
)


@dataclass(frozen=True)
class _Served:
    """What an env was opened onto: a frozen source, its instances, its fork store."""

    digest: str
    instances: tuple[Instance, ...]
    forks: Path
    verified: bool


@register("receipts_v1")
class ReceiptsV1Env(Env):
    """One side of an admitted instance, as a shogym env.

    Config (all optional, via ``shogym.make("receipts_v1", config=...)``):
      - ``genre``: which generator. Default ``"ledger"``.
      - ``side``: ``"a"`` or ``"b"``, which sibling of the family is served.
      - ``bundle``: an admission bundle, by digest or by directory. Default is the
        genre's bundle when it has exactly one. A bundle that does not verify is a
        refusal, not a rebuild, and there is no argument that turns the refusal off:
        an argument that did would make the development name decorative.
    """

    mcp_servers = (RECEIPTS_SPEC,)
    function_name = "clerk"
    score_terminal_tool = SUBMIT_TOOL_NAME
    #: The score does not leave with the terminal result, on either channel. What one
    #: graded receipt is worth is what this environment exists to measure, so the
    #: quantity is delivered by an experiment in a controlled arm and not by the seal.
    #: The scalar is still verified, still written to the trace and still returned by
    #: `evaluate`; it is withheld from the result that crosses to the agent's process.
    inband_terminal_feedback = False

    def __init__(
        self,
        genre: str = DEFAULT_GENRE,
        side: str = "a",
        bundle: Optional[str] = None,
    ) -> None:
        self._genre = genre
        if is_fixture(genre):
            # A gate exhibit wears the protocol so the gates can be exercised through
            # the shipped path. It has no surface and no admission, and refusing it
            # here is what makes "vectors are never dealt" a property of the env
            # rather than of whichever command happened to check.
            raise ValueError(f"{genre!r} is a gate exhibit and is never served")
        self._generator: Generator = load_generator(genre)
        self._side = side.strip().lower()
        if self._side not in ("a", "b"):
            raise ValueError(f"a family has sides a and b, not {side!r}")
        self._served = self._open(genre, bundle)
        self._ordinals: List[int] = [i.ordinal for i in self._served.instances]
        self._built: Dict[int, Instance] = {
            i.ordinal: i for i in self._served.instances
        }
        # Per-session grading state, keyed by session id, so one env instance
        # safely backs many concurrent episodes.
        self._grading_state: Dict[str, Dict[str, Any]] = {}
        self.function = FunctionConfig(example_system_template=_BASE_INSTRUCTIONS)
        super().__init__(horizon=HORIZON, num_tasks=len(self._ordinals))

    def _open(self, genre: str, ref: Optional[str]) -> _Served:
        """The verified bundle this env serves. Overridden only by the development
        environment, which is a different registered name."""
        return _open_bundle(genre, ref, self._generator)

    @property
    def dealable(self) -> bool:
        """Whether what this env serves cleared every admission stage.

        True for anything the production environment opened, because opening it is
        what established it. The development environment is False.
        """
        return self._served.verified

    def instance(self, ordinal: int) -> Instance:
        """The admitted instance at `ordinal`, as the source recomputed it.

        The instances an env serves are the ones verification rebuilt while it was
        establishing that this source holds them, so there is no second rebuild here
        that could produce anything different.
        """
        key = int(ordinal)
        if key not in self._built:
            raise KeyError(f"instance {key} is not in what this env serves")
        return self._built[key]

    def _ordinal_for(self, task_idx: int) -> int:
        if not 0 <= task_idx < len(self._ordinals):
            raise ValueError(
                f"Task index {task_idx} is out of range for {len(self._ordinals)} tasks"
            )
        return self._ordinals[task_idx]

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._ordinals)))
        ordinal = self._ordinal_for(task_idx)
        task = self.instance(ordinal).side(self._side)
        return {
            "task_idx": task_idx,
            "ordinal": ordinal,
            "genre": self._genre,
            "side": self._side,
            "task_id": task.task_id,
            "n_rows": task.n_rows,
        }

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        self._grading_state[session_id] = {"ordinal": int(task["ordinal"])}

    def _end_session(self, session_id: str) -> None:
        self._grading_state.pop(session_id, None)

    # ----- describe: the schedule and the policy extract -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        """Publish the task, under its opaque identifier.

        The selector a caller passes is a position in this env's own list and stays
        controller-side. What goes out is the task's HMAC identifier, which encodes
        no seed, no family index and no draw ordinal, so a harness cannot condition
        on family position or draw order across runs.
        """
        spec = super().describe(task_id)
        idx = _as_index(task_id)
        if idx is None or not 0 <= idx < len(self._ordinals):
            # Refused rather than answered thinly. The base instructions tell the
            # agent to read a schedule and a policy extract, so a spec published
            # without one is a task that says to read something that is not there.
            raise KeyError(
                f"{task_id!r} is not a position in what this env serves, which holds "
                f"{len(self._ordinals)} tasks"
            )
        task = self.instance(self._ordinal_for(idx)).side(self._side)
        return spec.model_copy(
            update={
                "task_id": task.task_id,
                "instructions": "\n".join([_BASE_INSTRUCTIONS, "", task.text]),
            }
        )

    # ----- finalize: read and grade the sealed filing -----

    async def finalize(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, req: FinalizeRequest
    ) -> TerminalEvidence:
        """Canonicalize the sealed filing and grade it, returning core-owned evidence.

        The public verdict says the filing landed and how many rows it named. It
        does not say what the filing scored: what one graded receipt is worth is the
        quantity this environment exists to measure, and a verdict at the terminal
        is a receipt nobody asked for.
        """
        state = self._grading_state.get(req.session_id)
        if req.source != "explicit_tool":
            # An abort or a horizon: no filing arrived, which is an outcome and not a
            # failure. The score is zero and the reason code says which.
            return TerminalEvidence(
                source=req.source,
                status="ok",
                verdict={"filed": False, "rows": 0},
                diagnostic=f"no filing reached the seal (source={req.source})",
            )
        if req.args is None or state is None:
            # A filing DID arrive and this cannot grade it: without the session's
            # ordinal there is no instance to render against, so no fork can be
            # committed. Reporting it as an ordinary empty filing would put a scored
            # zero on a link that was never rendered, so it fails closed instead.
            missing = "its arguments" if req.args is None else "its episode's state"
            return TerminalEvidence(
                source=req.source,
                status="finalize_error",
                verdict={"filed": False, "rows": 0},
                diagnostic=f"a filing reached the seal without {missing}",
            )
        instance = self.instance(state["ordinal"])
        task = instance.side(self._side)
        raw = req.args.get("filing")
        # The one post-A act: canonicalize, render the three cells, check the
        # envelope, hash and commit. A retry or a resume replays the committed bytes,
        # because rendering again is how two branches of one fork come to differ.
        fork = bank_mod.fork_for(
            self._generator, instance, self._side, raw,
            self._served.forks, self._served.digest,
        )
        canonical = self._generator.parse_and_canonicalize(task, raw)
        # How many rows the filing NAMED, not how many the table has. The canonical
        # form allocates one value per printed row including the omissions, so its
        # length is the table's size and would tell an agent it had filed rows it
        # never mentioned.
        rows = (
            sum(1 for filed in canonical.filed if filed)
            if isinstance(canonical, SealedSubmission)
            else 0
        )
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict={"filed": canonical.is_filing, "rows": rows},
            diagnostic=(
                f"no filing ({canonical.reason}); fork {fork.digests[GRADED][:12]}"
                if isinstance(canonical, NoFiling)
                else f"component score {fork.component_score:.6f}, "
                f"fork {fork.digests[GRADED][:12]}, replayed {fork.replayed}"
            ),
        )

    # ----- verify -----

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: Optional[TerminalEvidence] = None,
    ) -> FeedbackCollection:
        instance = self.instance(int(task["ordinal"]))
        return score_evidence(self._generator, instance, self._side, evidence,
                              terminated=terminated)


# ----- pure scoring, module level so it is unit-testable without a server -----


def score_evidence(
    generator: Generator,
    instance: Instance,
    side: str,
    evidence: Optional[TerminalEvidence],
    *,
    terminated: bool,
) -> FeedbackCollection:
    """Build episode feedback from the sealed filing.

    ``component_score`` is the scalar the chain seals: the fraction of rows the
    filing got right, equal weight per row, over the printed row count.
    ``no_filing`` is the reason code when nothing scorable arrived, which is what
    keeps an unanswered task mechanically distinct from a badly answered one.

    A TERMINAL THAT FAILED CLOSED CARRIES NO SCORE. The one post-A act is the
    render, the envelope check, the hash and the commit together, and a scalar is the
    chain's record that all of it happened. When that act failed, this can still
    re-parse the filing and arrive at a number, and the number would say the link
    produced a fork it does not have: no cell was committed, so no branch of it can be
    served. So the failure is reported as ``grade_error`` and nothing else, and a
    consumer that wants the score has to notice its absence rather than a flag beside
    it.

    WHICH FAILURE, EXACTLY. This covers what ``finalize`` owns: parsing, rendering,
    judging and publishing the fork, and a lost session. It does NOT cover a failure
    of the core verifier that runs after ``finalize`` returned, because by then the
    fork IS committed and it is a valid one: the cells were rendered from the filing
    that sealed and judged before they were written. The core's own contract on that
    path is to publish no feedback at all, so an episode whose verifier failed carries
    an empty feedback list rather than ``grade_error``, alongside a committed fork.
    ``finalize_error`` therefore means "the terminal transaction failed", and it does
    not by itself mean "no fork" or "grade_error is present". The served tests pin
    both shapes.
    """
    fb = FeedbackCollection()
    if not terminated:
        return fb
    if evidence is not None and evidence.finalize_error:
        fb.episode.append(EpisodeFeedback(name="grade_error", value=True))
        return fb
    task = instance.side(side)
    args = None if evidence is None else evidence.args
    raw = None if args is None else args.get("filing")
    canonical = generator.parse_and_canonicalize(task, raw)
    score, outcomes = generator.score(task, canonical)
    fb.episode.append(EpisodeFeedback(name="component_score", value=score))
    fb.episode.append(EpisodeFeedback(name="solved", value=bool(outcomes) and score >= 1.0))
    if isinstance(canonical, NoFiling):
        fb.episode.append(EpisodeFeedback(name="no_filing", value=canonical.reason))
    else:
        fb.episode.append(
            EpisodeFeedback(name="rows_filed", value=float(canonical.filed_rows))
        )
        fb.episode.append(
            EpisodeFeedback(name="rows_omitted", value=float(len(canonical.omissions)))
        )
    return fb


def _bank_source(genre: str, bank: Optional[str]) -> Path:
    """Where a development bank file is."""
    return Path(bank) if bank else bank_path(genre)


def _bundle_root(genre: str, ref: Optional[str]) -> Path:
    """Which bundle directory a reference names, or the genre's only one."""
    if ref:
        candidate = Path(ref)
        return candidate if candidate.exists() else bundle_dir(genre) / ref
    held = bundles(genre)
    if not held:
        raise FileNotFoundError(
            f"no admission bundle for {genre!r} under {bundle_dir(genre)}. Materialize a "
            f"bank, record a room screen and a review pack, and freeze them with "
            f"`shogym receipts bundle {genre} ...`. This environment does not build its own."
        )
    if len(held) > 1:
        raise ValueError(
            f"{genre!r} has {len(held)} bundles and no digest was named; say which one: "
            + ", ".join(item.name[:16] for item in held[:4])
        )
    return held[0]


def _open_bundle(genre: str, ref: Optional[str], generator: Generator) -> _Served:
    """The verified bundle this env serves. There is no fallback and no invention.

    Verification is the whole of eligibility: one operation, recomputing the
    population, the screen and the review coverage from the bundle's own files and
    the running code. Nothing is composed beside it, so there is no second answer
    that could disagree with this one.
    """
    root = _bundle_root(genre, ref)
    opened = bundle_mod.load(root)
    checked = bundle_mod.verify(opened, generator)
    if not checked.verified:
        raise ValueError(
            f"the bundle at {root} does not verify: " + "; ".join(checked.problems[:3])
            + ". Use the development environment if you are working locally."
        )
    # Forks are written where the bundle is NOT: a bundle is frozen at its digest, and
    # anything written inside one would make it disagree with its own manifest.
    return _Served(
        digest=opened.digest,
        instances=checked.instances,
        forks=root.parent / "forks",
        verified=True,
    )


def _as_index(task_id: Optional[str]) -> Optional[int]:
    if task_id is None:
        return None
    try:
        return int(task_id)
    except (TypeError, ValueError):
        return None


@register("receipts_dev_v1")
class ReceiptsDevEnv(ReceiptsV1Env):
    """The same environment, serving a bank that has never been bundled.

    THE ONE UNBUNDLED PATH, and it says so: nothing it serves has an admission
    bundle, so nothing it serves has a verified screen, a human read, or a code pin.
    It exists so that the registered environment can refuse everything short of a
    bundle outright instead of carrying a flag that turns the refusal off, and so
    that "I was developing" and "this was dealt" are two different names in a trace
    rather than one name with an argument.
    """

    def __init__(
        self,
        genre: str = DEFAULT_GENRE,
        side: str = "a",
        bank: Optional[str] = None,
    ) -> None:
        super().__init__(genre=genre, side=side, bundle=bank)

    def _open(self, genre: str, ref: Optional[str]) -> _Served:
        path = _bank_source(genre, ref)
        if not path.is_file():
            raise FileNotFoundError(f"no bank for {genre!r} at {path}")
        held = bank_mod.load_bank(path)
        # The population is still recomputed rather than listed: development changes
        # what a bank is worth, not what it holds.
        found = bank_mod.population(held, self._generator)
        return _Served(
            # The bank file AND the code that renders from it. A bank file names the
            # generator, the key and a count, so an edit to the generator that changes
            # a rendered cell leaves it, the task identifier and the filing digest
            # untouched, and the fork store would replay the cells the old code wrote.
            # Production is safe because the bundle digest pins the code; here the
            # digest has to carry it.
            digest=streams.digest(
                path.read_bytes(),
                bank_mod.current_code_digest(self._generator).encode(),
            ),
            instances=found.instances,
            forks=path.parent / f"{genre}-forks",
            verified=False,
        )


__all__ = ["HORIZON", "ReceiptsDevEnv", "ReceiptsV1Env", "score_evidence"]
