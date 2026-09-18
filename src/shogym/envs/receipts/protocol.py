"""The generator protocol: one genre, one hidden convention, two sibling tasks.

A FAMILY is one generator plus one draw of its hidden convention. The generator
emits sibling tasks A and B over different surface data, both scored under that
same drawn convention, so the family relation holds BY CONSTRUCTION rather than by
assertion. Nothing about the relation is asserted in prose and then hoped for: the
same convention object scores both sides, and a check can read it.

A generator supplies:

  SHAPE     the table's columns and how their values are invented, one row per
            case, every value drawn from a domain-separated stream.
  AXES      the hidden decisions, each a named axis with an interchangeable option
            set. The sampler LAW is public and lives in `draw_convention`; the
            live draws are not.
  parse_and_canonicalize
            the mechanical reading of what the agent filed. Identifier matching,
            row order, duplicates, omissions, extras, invalid tokens, case and
            whitespace. It returns a reason-coded NoFiling rather than raising, so
            a scorable zero and an absent filing stay mechanically distinct.
  score     pure and deterministic, returning the sealed scalar and the per-row
            outcomes. The SAME canonical filing feeds the scorer and both
            renderers, so a receipt can never grade something the score did not.
  render_receipt / render_placebo / render_oracle
            the three cells, each a ReceiptAST. One shared serializer turns an AST
            into bytes.
  parse_oracle
            reads a convention back out of an oracle cell, so the rule template is
            checkably lossless.
  describe  the instruction the agent sees, leaving every axis genuinely
            undetermined.
  RECEIPT_POLICY
            which of the graded task's rows the receipt reports a verdict and a
            correction on: every one of them, or the q the committed mask drew. It
            is a registration, priced by the gate and stated in the task text, and a
            generator that declares none is refused at registration.

THE TRUST BOUNDARY. Everything in this module is controller-side. Generator code,
the sampler's live draws, the convention, the answer key, the oracle renderer and
the bank never reach a lineage sandbox. What crosses is rendered bytes and an
opaque task identifier. `Task.key` and `Instance.convention` are answer oracles and
are named here so it is obvious which fields must never be serialized outward.

THE SAMPLER LAW IS PUBLISHED. `draw_convention` draws each axis uniformly and
independently from a keyed stream, so the support is the whole option product. A
sampler generated from a latent, where two resolved axes pin the rest, has the same
option sets and a far smaller support, and everything downstream that counts
assignments is then wrong about how much there is to learn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, Union, runtime_checkable

from shogym.envs.receipts import streams
from shogym.envs.receipts.receipt_ast import Envelope, ReceiptAST

#: The only scoring shape the shared gate prices. Gate H optimizes over the sibling
#: task's legal action space by taking each row's posterior mode, which is the exact
#: Bayes action when and only when the component score is an unweighted mean of
#: per-row exact matches. A generator whose score weighted its rows, or scored them
#: jointly, would be gated against an objective that is not its own, and could clear
#: H on room it does not have. Declaring the shape is what lets the gate refuse
#: rather than quietly misprice.
ROW_ADDITIVE_EQUAL_WEIGHT = "row_additive_equal_weight"
SCORING_SHAPES = (ROW_ADDITIVE_EQUAL_WEIGHT,)

#: Why a filing did not produce a scorable submission. Reason-coded, never raised,
#: because the chain's failure taxonomy has to tell "answered badly" from "did not
#: answer" and an exception erases the difference.
NO_FILING_REASONS = (
    "empty",
    "unreadable",
    "no_known_identifier",
)

#: The two shapes a receipt policy can have. EVERY_ROW is what every family had: a
#: verdict and a same-row correction on all of the graded task's rows. SAMPLED_ROWS
#: reports q of n rows in full and says nothing at all on the other n - q, which carry
#: the row's committed neutral tokens in both slots.
EVERY_ROW = "every-row"
SAMPLED_ROWS = "sampled-rows"
POLICY_SHAPES = (EVERY_ROW, SAMPLED_ROWS)


@dataclass(frozen=True)
class ReceiptPolicy:
    """Which of the graded task's rows the receipt reports on, as one registration.

    WHY THIS IS A REGISTRATION AND NOT A RENDERER'S CHOICE. What a receipt reports is
    the whole of what one link can teach, so it is priced by the gate, stated in the
    task text, committed with the instance, and compared against by the judge. A
    generator free to decide per render what it reported would be a generator whose
    measured room and served room are two different things.

    `reported` and `rows` are q and n for a sampled policy and are zero for the full
    one, whose row count is whatever the task prints. A generator declares one of
    `REGISTERED_POLICIES` and one that declares none is refused at registration,
    exactly as an undeclared copy profile is.
    """

    name: str
    shape: str
    reported: int = 0
    rows: int = 0

    def __post_init__(self) -> None:
        if self.shape not in POLICY_SHAPES:
            raise ValueError(
                f"a receipt policy is one of {POLICY_SHAPES}, not {self.shape!r}"
            )
        if self.shape == EVERY_ROW and (self.reported or self.rows):
            raise ValueError(
                "the full receipt reports every row the task prints, so it names no "
                "count of its own"
            )
        if self.shape == SAMPLED_ROWS and not 1 <= self.reported < self.rows:
            raise ValueError(
                f"a sampled receipt reports between one and {self.rows - 1} of "
                f"{self.rows} rows, not {self.reported}"
            )

    @property
    def samples(self) -> bool:
        """Whether this policy reports fewer rows than the task prints."""
        return self.shape == SAMPLED_ROWS

    def as_record(self) -> dict[str, object]:
        """The policy as one canonical value, for the instance commitment."""
        return {
            "name": self.name,
            "shape": self.shape,
            "reported": int(self.reported),
            "rows": int(self.rows),
        }


#: THE REGISTERED POLICIES, and the whole of them. A policy outside this tuple has not
#: been priced by any gate, so nothing may be registered under it.
FULL_RECEIPT = ReceiptPolicy(name="full", shape=EVERY_ROW)
SAMPLED_FOUR_OF_TWENTY_FOUR = ReceiptPolicy(
    name="sampled-4-of-24", shape=SAMPLED_ROWS, reported=4, rows=24
)
REGISTERED_POLICIES = (FULL_RECEIPT, SAMPLED_FOUR_OF_TWENTY_FOUR)


def policy_of(generator: "Generator") -> ReceiptPolicy:
    """The receipt policy this generator declares, refused when it declares none.

    Refused rather than defaulted. Which rows a receipt reports decides what the gate
    prices, what the task text promises and what the judge accepts, and a default here
    would be that decision taken about a family by whoever wrote this module.
    """
    declared = getattr(generator, "RECEIPT_POLICY", None)
    name = getattr(generator, "name", type(generator).__name__)
    if declared is None:
        raise ValueError(
            f"{name!r} declares no RECEIPT_POLICY. Which rows a receipt reports is what "
            f"the gate prices and what the task text says, so a generator declares one; "
            f"the registered policies are "
            f"{', '.join(p.name for p in REGISTERED_POLICIES)}"
        )
    if not isinstance(declared, ReceiptPolicy) or declared not in REGISTERED_POLICIES:
        shown = getattr(declared, "name", declared)
        raise ValueError(
            f"{name!r} declares receipt policy {shown!r}; the registered policies are "
            f"{', '.join(p.name for p in REGISTERED_POLICIES)}"
        )
    return declared


def require_policy(generator: "Generator") -> "Generator":
    """Refuse a generator that does not declare a registered policy. Returns it."""
    policy_of(generator)
    return generator


def receipt_mask(
    policy: ReceiptPolicy,
    master: bytes,
    generator: str,
    ordinal: int,
    label: str,
    n_rows: int,
) -> tuple[int, ...]:
    """The rows this task's receipt reports on, as printed positions in order.

    DRAWN BEFORE ANY FILING AND FROM A STREAM OF ITS OWN, so it cannot depend on what
    the agent filed and cannot depend on the drawn convention. It is a function of the
    master key and of the generator, the ordinal and the sibling label alone, so a
    re-render after a crash reproduces the same rows and the two branches of one fork
    are served the same cell.

    A MASK THAT OMITS AN AXIS IS NOT REDRAWN. What is registered is the law and not the
    realized receipt: redrawing until every hidden decision has a witness would make the
    mask a function of the convention, and the arithmetic the gate does averages over
    the uniform law rather than over a filtered one.
    """
    if not policy.samples:
        return tuple(range(int(n_rows)))
    if int(n_rows) != policy.rows:
        raise ValueError(
            f"policy {policy.name!r} reports {policy.reported} of {policy.rows} rows "
            f"and this task prints {n_rows}"
        )
    stream = streams.rng(master, streams.RECEIPT_MASK, generator, ordinal, label)
    return tuple(sorted(stream.sample(range(int(n_rows)), policy.reported)))


class ConstructionExhausted(RuntimeError):
    """A generator's bounded search for a valid instance ran out.

    IT IS NOT A SKIPPED ORDINAL. A generator that searches for a table its own filter
    accepts can fail to find one, and the two ways of reporting that are not
    interchangeable. An ordinary exception out of `build_table` reaches `bank.population`
    outside the admission guard and leaves the caller with a traceback where it was
    promised a count; treating it as a rejection would silently move the population,
    because the bank would fill from later ordinals and the passing fraction it prints
    would be measured against a different rule than the one it says it used.

    So it has a name. Materialization stops on it and says which ordinal exhausted,
    verification recognises it as the same whole-bank failure rather than as a bundle
    that cannot be read, and the provenance record beside the bank keeps the key the
    attempt was made under, so a failed attempt is reproducible instead of lost.
    """


@dataclass(frozen=True)
class Axis:
    """One hidden decision: a name and an interchangeable option set."""

    name: str
    options: tuple[str, ...]
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.options) < 2:
            raise ValueError(f"axis {self.name!r} needs at least two options")
        if len(set(self.options)) != len(self.options):
            raise ValueError(f"axis {self.name!r} repeats an option")

    @property
    def arity(self) -> int:
        return len(self.options)


@dataclass(frozen=True)
class Column:
    """One column of the generated table, and how its values are invented."""

    name: str
    invented: str


@dataclass(frozen=True)
class Shape:
    """The table's shape: its columns, its row count, and what a row stands for."""

    columns: tuple[Column, ...]
    rows: int
    case: str
    note: str = ""


@dataclass(frozen=True)
class SealedSubmission:
    """A filing, read mechanically into one canonical value per printed row.

    `values` is one entry per row in printed order, empty where the agent filed
    nothing for that row. The counts beside it are what the reading had to decide,
    kept so a low score can be told apart from a malformed filing.
    """

    values: tuple[str, ...]
    #: The row mask, and it is not redundant with `values`. Filing an empty value
    #: and filing nothing at all are different acts, and they have to stay
    #: different: an axis option can BE the empty value, so without the mask an
    #: agent that filed nothing would collect every row that option covers for free.
    filed: tuple[bool, ...] = ()
    filed_rows: int = 0
    duplicates: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()
    omissions: tuple[str, ...] = ()

    @property
    def is_filing(self) -> bool:
        return True


@dataclass(frozen=True)
class NoFiling:
    """No scorable submission, with the reason recorded rather than thrown away."""

    reason: str

    def __post_init__(self) -> None:
        if self.reason not in NO_FILING_REASONS:
            raise ValueError(f"unknown no-filing reason {self.reason!r}")

    @property
    def is_filing(self) -> bool:
        return False


Filing = Union[SealedSubmission, NoFiling]


@dataclass(frozen=True)
class RowOutcome:
    """What the filing did on one row. The receipt grades this, not a stored choice."""

    ordinal: int
    identifier: str
    filed: str
    correct: str
    matched: bool
    #: Whether the agent filed a line for this row at all. Distinct from an empty
    #: `filed`, because filing nothing and filing an empty value are different acts.
    was_filed: bool = False


@dataclass(frozen=True)
class PublicTask:
    """Everything about a task that is not an answer oracle.

    This is the type the placebo renderer and the task description take, and it
    exists so that neither CAN read the drawn convention: there is no field on it
    through which the key could arrive. A renderer that took the full task would be
    trusted not to look, and the placebo arm's whole value is that it does not have
    to be trusted.
    """

    label: str
    task_id: str
    surface: str
    table: Any
    n_rows: int
    #: The rows this task's receipt reports on. Drawn from its own stream before any
    #: filing and independently of the convention, so it is not an answer oracle and
    #: it is carried here the way the row count is.
    mask: tuple[int, ...] = ()


@dataclass(frozen=True)
class Task:
    """One of the two sibling tasks.

    `table` is the generator's own row structure and is opaque here. `key` is the
    answer under the instance's drawn convention: an answer oracle, controller-side
    only. `task_id` is the opaque identifier that crosses the boundary.

    `mask` is the receipt's committed row selection, and it is a REQUIRED field with
    no default. Every counterfactual render in this package retasks a task under
    another convention by building a new one of these, and a default would let such a
    rebuild silently drop the rows the receipt reports on: the cell it produced would
    then be compared against an expected cell built from the same dropped mask, and
    both would be wrong together. Naming it at every construction is what makes a
    forgotten mask a refusal rather than a quiet second instrument.
    """

    label: str
    task_id: str
    surface: str
    table: Any
    text: str
    key: tuple[str, ...]
    mask: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        return len(self.key)

    def public(self) -> PublicTask:
        """The half of this task that may cross to a keyless renderer."""
        return PublicTask(
            label=self.label,
            task_id=self.task_id,
            surface=self.surface,
            table=self.table,
            n_rows=self.n_rows,
            mask=self.mask,
        )


@dataclass(frozen=True)
class Instance:
    """One family: a generator, a drawn convention, and the sibling tasks under it."""

    generator: str
    genre: str
    ordinal: int
    convention: Mapping[str, str]
    a: Task
    b: Task
    envelope: Envelope

    def side(self, label: str) -> Task:
        key = label.strip().lower()
        if key in ("a", "0"):
            return self.a
        if key in ("b", "1"):
            return self.b
        raise ValueError(f"a family has sides a and b, not {label!r}")


@runtime_checkable
class Generator(Protocol):
    """What a genre module implements. See this module's docstring for the parts."""

    name: str
    genre: str
    SHAPE: Shape
    AXES: tuple[Axis, ...]
    #: Which of the registered scoring shapes this generator's `score` implements.
    SCORING: str
    #: Which registered family of maps the copy screen prices this generator against.
    #: Declared rather than inferred, and refused at registration when it is absent:
    #: the bar and the family of maps it is a maximum over are one registration, so a
    #: family priced under the wrong one reports a number that measured nothing.
    #: `copy_profiles` holds the registered names and what each one enumerates.
    COPY_PROFILE: str
    #: Which rows of the graded task this family's receipt reports on. Declared rather
    #: than assumed, and refused at registration when it is absent: what a receipt
    #: reports is what the gate prices and what the task text promises, so a family
    #: that named none would be gated against one instrument and served as another.
    #: `REGISTERED_POLICIES` holds the registered values.
    RECEIPT_POLICY: ReceiptPolicy

    def surface_for(self, ordinal: int, label: str) -> str:
        """Which surface data pool this instance uses for side A or side B."""
        ...

    def build_table(self, master: bytes, ordinal: int, label: str) -> Any:
        """The table for one side, from that side's own stream."""
        ...

    def table_record(self, table: Any) -> Any:
        """One table as a canonical, hashable value. Every field anything reads.

        The table is opaque to this package by design, and the bank's commitment to
        an instance is what fixation and a bundle compare against. A commitment that
        left the tables out is a commitment two rebuilds can satisfy while the rows
        an agent is served differ, which is the one thing fixation exists to catch,
        so a family says what its table is and the bank hashes what it said.

        Every field the description, the parser, the scorer, the answer ranks and the
        renderers read has to be in it. A field that is read and not recorded is a
        field that can move between two rebuilds without the digest noticing.
        """
        ...

    def build_envelope(self, master: bytes, ordinal: int) -> Envelope:
        """The registered envelope, with this instance's committed filler and neutrals."""
        ...

    def key_for(self, table: Any, convention: Mapping[str, str]) -> tuple[str, ...]:
        """The correct answer for every row under one convention."""
        ...

    def parse_and_canonicalize(self, task: Task, raw: object) -> Filing:
        """Read what the agent filed into one canonical value per row."""
        ...

    def score(self, task: Task, canonical: Filing) -> tuple[float, tuple[RowOutcome, ...]]:
        """The sealed scalar and the per-row outcomes. Pure."""
        ...

    def describe(self, task: PublicTask) -> str:
        """The instruction the agent sees. Every axis stays undetermined.

        It takes the public half of the task, so the instruction cannot be written
        from the answer even by accident.
        """
        ...

    def render_receipt(
        self,
        task: Task,
        canonical: Filing,
        truth: Sequence[str],
        feedback: Any,
    ) -> ReceiptAST:
        """Per-row verdicts on what the filing did. No axis labels.

        `feedback` is the immutable `render.Feedback`: the declared policy, the
        committed mask, and the neutral tokens a suppressed position prints. It is
        snapshotted before this callback runs and the same snapshot builds the cell
        this one is compared against, so a renderer cannot widen what it reports by
        rewriting what it was handed.
        """
        ...

    def render_placebo(
        self, task: PublicTask, canonical: Filing, envelope: Envelope
    ) -> ReceiptAST:
        """The inert cell: congruent with the graded one outside the registered slots.

        It takes the PUBLIC task and the envelope, so there is no argument here
        through which the hidden rule could reach it. Not "does not look at the key"
        as a promise the author makes, but "has no key to look at" as a fact about
        the signature.
        """
        ...

    def render_oracle(
        self, task_id: str, convention: Mapping[str, str], row_count: int = 0
    ) -> ReceiptAST:
        """The drawn rule, from the registered declarative template."""
        ...

    def parse_oracle(self, ast: ReceiptAST) -> dict[str, str]:
        """The convention an oracle cell states, read back out of it."""
        ...

    def row_identifiers(self, table: Any) -> tuple[str, ...]:
        """The identifier the receipt prints for each row, in printed order."""
        ...

    def row_classes(self, table: Any) -> tuple[str, ...]:
        """The class a reader can see each row belongs to, from the receipt and task."""
        ...

    def normalize_answer(self, value: str) -> str:
        """The canonical form of one answer, as this family's SCORER compares them.

        Gate H optimizes over the sibling task's legal actions, and two spellings the
        scorer treats as the same answer are one action, not two. Coding raw strings
        would count an alias as a separate choice, spread the posterior across
        spellings nobody can act on separately, and report room that is not in the
        legal problem.

        It has to be the function the scorer actually uses, not a second one that
        agrees today, so a generator implements it once and its `score` calls it.
        """
        ...

    def surface_templates(self) -> tuple[str, ...]:
        """Every surface this family can draw, named.

        The review pack has to cover each of them, so the set has to be enumerable
        without drawing the whole bank.
        """
        ...

    def answer_ranks(self, table: Any) -> tuple[str, ...] | None:
        """The family's COMPLETE ordered legal answers for this table, as the task
        publishes them, or None where the declared profile consumes none.

        Complete, and ordered, and public. The copy screen needs it because the
        cheapest transfer a reader can do is not lexical: it is to read the two band
        tables the two tasks print and map first to first. A screen that built its
        relabels from the tokens one drawn key happened to realize would miss that
        map entirely, and would call two vocabularies of different size a bijection.

        NONE IS FOR A PROFILE THAT CONSUMES NO RANKS, and it is not the same as an
        empty tuple. A family whose answers are whole invented words has no complete
        ordered vocabulary to publish: an empty tuple would be a fictitious one, and a
        list of the words one drawn key realized would be a transfer nobody could
        perform. Such a family declares a profile whose maps are built from something
        else and returns None here, which `copy_profiles` refuses to confuse with a
        family that publishes ranks and forgot to.
        """
        ...

    def row_label(self, table: Any) -> tuple[str, ...]:
        """What each printed row is labelled by: the scored row, or an axis.

        A row labelled by axis states its own constraint and is what gate S refuses.
        An ordinary generator labels every row by the record it grades.
        """
        ...


# --------------------------------------------------------------------------
# the sampler law, and the draw
# --------------------------------------------------------------------------


def draw_convention(
    axes: Sequence[Axis],
    master: bytes,
    ordinal: int,
    support: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, str]:
    """Draw one convention.

    The law is public: independent uniform draws over the declared option sets, so
    the support is the whole product and knowing every option on every other axis
    says nothing about this one. The key is not public, so the law can be inspected
    while a live draw stays hidden.

    A generator that declares a smaller SUPPORT is drawn uniformly from that instead,
    and this is the sampler's source of truth rather than a label the gates read
    while the draw goes elsewhere. Gating one distribution and materializing another
    is the failure the correlated exhibit exists to show, and an exhibit that did it
    itself would demonstrate nothing.
    """
    rng = streams.rng(master, streams.CONVENTION, ordinal)
    if support:
        return dict(rng.choice(list(support)))
    return {axis.name: rng.choice(list(axis.options)) for axis in axes}


def conventions(axes: Sequence[Axis]) -> list[dict[str, str]]:
    """Every convention in the space, enumerated in a fixed order."""
    out: list[dict[str, str]] = [{}]
    for axis in axes:
        out = [dict(c, **{axis.name: o}) for c in out for o in axis.options]
    return out


def support_of(generator: "Generator") -> list[dict[str, str]]:
    """The conventions this generator's sampler can actually draw.

    Almost always the full product of the declared axes, which is what
    `draw_convention` samples and what a family is supposed to have. A generator may
    declare a SUPPORT that is smaller, and one does: the correlated exhibit, whose
    whole point is that the declared axis catalogue overstates the rule space. The
    gates read the support rather than the product, so that exhibit is gated on the
    space it really has instead of on the space it advertises.
    """
    declared = getattr(generator, "SUPPORT", None)
    if declared is None:
        return conventions(generator.AXES)
    return [dict(c) for c in declared]


def draw(generator: Generator, master: bytes, ordinal: int) -> Instance:
    """One family: the drawn convention, and sibling tasks A and B scored under it.

    THE CONVENTION IS FROZEN BEFORE EITHER SIBLING IS BUILT. What makes the family
    relation hold by construction is that one convention scores both sides, and a
    plain dictionary handed to A's callback and then to B's is a dictionary either
    callback can edit in between: A's key computed under one rule and B's under
    another, with every named check still passing because each side is internally
    consistent on its own. So the draw is snapshotted into a read-only mapping here,
    and that is what both siblings and the stored instance get. A generator that
    writes to it raises where the mistake is rather than producing an unrelated pair.
    """
    support = support_of(generator)
    declared = getattr(generator, "SUPPORT", None)
    drawn = draw_convention(
        generator.AXES, master, ordinal, support if declared is not None else None
    )
    if declared is not None:
        reachable = {tuple(sorted(c.items())) for c in support}
        if tuple(sorted(drawn.items())) not in reachable:
            raise ValueError(
                f"{generator.name!r} drew a convention outside its declared support"
            )
    convention = MappingProxyType(dict(drawn))
    return Instance(
        generator=generator.name,
        genre=generator.genre,
        ordinal=int(ordinal),
        convention=convention,
        a=_task(generator, master, ordinal, "A", convention),
        b=_task(generator, master, ordinal, "B", convention),
        envelope=generator.build_envelope(master, ordinal),
    )


def _task(
    generator: Generator,
    master: bytes,
    ordinal: int,
    label: str,
    convention: Mapping[str, str],
) -> Task:
    table = generator.build_table(master, ordinal, label)
    key = tuple(generator.key_for(table, convention))
    task_id = streams.task_identifier(master, generator.name, ordinal, label)
    surface = generator.surface_for(ordinal, label)
    # The mask is drawn here, from the key and the coordinates, and before the text is
    # written or any filing exists. The convention is already in hand at this point and
    # is deliberately not one of the coordinates: the stream is domain-separated, so
    # what the receipt reports on is independent of what it would report.
    mask = receipt_mask(
        policy_of(generator), master, generator.name, ordinal, label, len(key)
    )
    public = PublicTask(
        label=label,
        task_id=task_id,
        surface=surface,
        table=table,
        n_rows=len(key),
        mask=mask,
    )
    return Task(
        label=label,
        task_id=task_id,
        surface=surface,
        table=table,
        text=generator.describe(public),
        key=key,
        mask=mask,
    )


# --------------------------------------------------------------------------
# the describe lint
# --------------------------------------------------------------------------


def _tokens(option: str) -> set[str]:
    return {option, option.replace("_", " "), option.replace("_", "-")}


def option_mentions(axes: Sequence[Axis], text: str) -> list[tuple[str, str]]:
    """Axis options the text names outright.

    A task text that prints an option token has answered its own question. Full
    underdetermination needs a human read and the invariance render; this catches
    the failure a machine can see.
    """
    low = text.lower()
    hits: list[tuple[str, str]] = []
    for axis in axes:
        for option in axis.options:
            for token in _tokens(option.lower()):
                if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(token), low):
                    hits.append((axis.name, option))
                    break
    return hits


__all__ = [
    "EVERY_ROW",
    "FULL_RECEIPT",
    "NO_FILING_REASONS",
    "POLICY_SHAPES",
    "REGISTERED_POLICIES",
    "SAMPLED_FOUR_OF_TWENTY_FOUR",
    "SAMPLED_ROWS",
    "Axis",
    "ConstructionExhausted",
    "Column",
    "Filing",
    "Generator",
    "Instance",
    "NoFiling",
    "PublicTask",
    "ReceiptPolicy",
    "RowOutcome",
    "SealedSubmission",
    "Shape",
    "ROW_ADDITIVE_EQUAL_WEIGHT",
    "SCORING_SHAPES",
    "Task",
    "conventions",
    "draw",
    "policy_of",
    "receipt_mask",
    "require_policy",
    "support_of",
    "draw_convention",
    "option_mentions",
]
