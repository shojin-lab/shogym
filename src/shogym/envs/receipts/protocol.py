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


@dataclass(frozen=True)
class Task:
    """One of the two sibling tasks.

    `table` is the generator's own row structure and is opaque here. `key` is the
    answer under the instance's drawn convention: an answer oracle, controller-side
    only. `task_id` is the opaque identifier that crosses the boundary.
    """

    label: str
    task_id: str
    surface: str
    table: Any
    text: str
    key: tuple[str, ...]

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
        self, task: Task, canonical: Filing, truth: Sequence[str]
    ) -> ReceiptAST:
        """Per-row verdicts on what the filing did. No axis labels."""
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

    def answer_ranks(self, table: Any) -> tuple[str, ...]:
        """The family's COMPLETE ordered legal answers for this table, as the task
        publishes them.

        Complete, and ordered, and public. The copy screen needs it because the
        cheapest transfer a reader can do is not lexical: it is to read the two band
        tables the two tasks print and map first to first. A screen that built its
        relabels from the tokens one drawn key happened to realize would miss that
        map entirely, and would call two vocabularies of different size a bijection.
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
    public = PublicTask(
        label=label, task_id=task_id, surface=surface, table=table, n_rows=len(key)
    )
    return Task(
        label=label,
        task_id=task_id,
        surface=surface,
        table=table,
        text=generator.describe(public),
        key=key,
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
    "NO_FILING_REASONS",
    "Axis",
    "Column",
    "Filing",
    "Generator",
    "Instance",
    "NoFiling",
    "PublicTask",
    "RowOutcome",
    "SealedSubmission",
    "Shape",
    "ROW_ADDITIVE_EQUAL_WEIGHT",
    "SCORING_SHAPES",
    "Task",
    "conventions",
    "draw",
    "support_of",
    "draw_convention",
    "option_mentions",
]
