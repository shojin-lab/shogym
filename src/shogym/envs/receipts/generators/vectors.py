"""The hand-checkable vectors, as real generators, so the gates are tested through the CLI.

These are not families and they are never dealt. They are the instrument's own gate
vectors, ported so that `shogym receipts gate <name>` exercises the shipped code path
end to end: a generator renders a receipt, the serializer makes bytes, the observer
reads the printed rows back, and the gate scores them. A gate validated only by
hand-built matrices would be a gate nobody had run.

Every expected value is arithmetic, and every one of them is the instrument's:

  slots-c3 / c4 / c6   one row per axis, each row storing its own axis's option
                       verbatim on a verdict-only receipt. Two blocks per axis, so
                       R fails at every arity; the receipt is labelled by axis, so
                       S fails; ceiling 2/c, placebo 1/c, and headroom exactly zero
                       because floor and ceiling are the same partition.
  merge                one axis of three options and two rows whose readouts merge
                       different pairs. Intersecting the two merges separates all
                       three, so R passes at three blocks.
  one-row              the same axis carried by a single row. One row on a
                       verdict-only receipt can produce at most two signatures, so
                       R fails however rich the readout is.
  binary               one axis of two options. R is not asked of it, and with no
                       wider axis the instance fails for want of one.
  copies               twenty copies of one row plus one other. Item count with an
                       unchanged readout is worth exactly nothing, and the block
                       counts are identical to the one-copy twin.
  affine               the correlated sampler, reproduced deliberately. The same
                       option sets, but every row is generated from two latents as
                       (a * row + b) mod c. Anyone counting assignments reports a
                       large rule space and is wrong: the support is the two
                       latents, and two resolved rows pin every other row. It is
                       asserted REJECTED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from shogym.envs.receipts.protocol import (
    Axis,
    Column,
    Filing,
    ROW_ADDITIVE_EQUAL_WEIGHT,
    NoFiling,
    PublicTask,
    RowOutcome,
    SealedSubmission,
    Shape,
    Task,
)
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse as parse_oracle_cell
from shogym.envs.receipts.oracle import render as render_oracle_cell
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    PLACEBO,
    Envelope,
    ReceiptAST,
    ReceiptRow,
    Slot,
    SlotSpec,
    envelope_size_for,
)
from shogym.envs.receipts.protocol import support_of as _support_of
from shogym.receipts import AXIS_LABEL, ROW_LABEL

IDENTIFIER_WIDTH = 8
OBSERVED_WIDTH = 10
VERDICT_WIDTH = 4
CORRECTION_WIDTH = 10
#: One fixed literal, so a vector's placebo can never coincide with a verdict.
NEUTRAL_TOKEN = "----------"


@dataclass(frozen=True)
class VectorTable:
    """A vector's table is nothing but its row identifiers: there is no surface to invent."""

    rows: tuple[str, ...]


class VectorGenerator:
    """One hand-checkable vector, wearing the generator protocol.

    `readout` is the whole design: what row `i` reads under a convention. A verbatim
    readout returns the option itself and pins its axis at two blocks; a merging one
    returns a value several options share and is what lifts the partition past the
    pin.
    """

    def __init__(
        self,
        name: str,
        genre: str,
        axes: Sequence[Axis],
        n_rows: int,
        readout: Callable[[Mapping[str, str], int], str],
        label: str = ROW_LABEL,
        corrections: bool = False,
        support: Sequence[Mapping[str, str]] | None = None,
    ) -> None:
        self.name = name
        self.genre = genre
        self.SCORING: str = ROW_ADDITIVE_EQUAL_WEIGHT
        self.AXES = tuple(axes)
        self._n_rows = n_rows
        self._readout = readout
        self._label = label
        self._corrections = corrections
        if support is not None:
            self.SUPPORT: tuple[Mapping[str, str], ...] = tuple(support)
        self.SHAPE = Shape(
            columns=(Column("row", "a running number"),),
            rows=n_rows,
            case="one reported row",
            note="a gate vector, not a family",
        )
        verdict = SlotSpec("verdict", VERDICT_WIDTH, vocabulary=("PASS", "FAIL"))
        self.SLOTS: tuple[SlotSpec, ...] = (
            (verdict, SlotSpec("correction", CORRECTION_WIDTH, allows_answers=True,
                               allows_empty=True))
            if corrections
            else (verdict,)
        )
        self.ORACLE = OracleTemplate(
            head=("THE RULE",  ""),
            sentences={axis.name: axis.name + " is {}." for axis in self.AXES},
            phrases={
                axis.name: {o: f"the option written {o}" for o in axis.options}
                for axis in self.AXES
            },
        )
        self._envelope_size = envelope_size_for(
            max_rows=n_rows,
            identifier_width=IDENTIFIER_WIDTH,
            observed_width=OBSERVED_WIDTH,
            slots=self.SLOTS,
            body_allowance=400,
        )

    # ----- the instance -----

    def surface_for(self, ordinal: int, label: str) -> str:
        return "a" if label.upper() == "A" else "b"

    def surface_templates(self) -> tuple[str, ...]:
        return ("a", "b")

    def build_table(self, master: bytes, ordinal: int, label: str) -> VectorTable:
        side = self.surface_for(ordinal, label)
        return VectorTable(rows=tuple(f"{side}{i:03d}" for i in range(self._n_rows)))

    def table_record(self, table: VectorTable) -> dict[str, Any]:
        """A vector's whole table: its row identifiers, and nothing else exists."""
        return {"rows": list(table.rows)}

    def build_envelope(self, master: bytes, ordinal: int) -> Envelope:
        return Envelope(
            size=self._envelope_size,
            identifier_width=IDENTIFIER_WIDTH,
            observed_width=OBSERVED_WIDTH,
            slots=self.SLOTS,
            filler="Z" * self._envelope_size,
            column_titles=tuple("mn"[i] for i in range(len(self.SLOTS))),
            neutral={spec.name: (NEUTRAL_TOKEN[: spec.width],) * self._n_rows
                     for spec in self.SLOTS},
        )

    def normalize_answer(self, value: str) -> str:
        return value.strip().lower()

    def key_for(
        self, table: VectorTable, convention: Mapping[str, str]
    ) -> tuple[str, ...]:
        return tuple(self._readout(convention, i) for i in range(len(table.rows)))

    def row_identifiers(self, table: VectorTable) -> tuple[str, ...]:
        return table.rows

    def row_classes(self, table: VectorTable) -> tuple[str, ...]:
        return ("row",) * len(table.rows)

    def answer_ranks(self, table: VectorTable) -> tuple[str, ...]:
        seen: list[str] = []
        for convention in _support_of(self):
            for value in self.key_for(table, convention):
                if value not in seen:
                    seen.append(value)
        return tuple(seen)

    def row_label(self, table: VectorTable) -> tuple[str, ...]:
        return (self._label,) * len(table.rows)

    # ----- reading and scoring -----

    def parse_and_canonicalize(self, task: Task, raw: object) -> Filing:
        if raw is None or not isinstance(raw, str):
            return NoFiling("unreadable")
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return NoFiling("empty")
        identifiers = self.row_identifiers(task.table)
        position = {i.lower(): n for n, i in enumerate(identifiers)}
        values = [""] * len(identifiers)
        seen: set[int] = set()
        for line in lines:
            head, _, tail = line.partition(",")
            index = position.get(head.strip().lower())
            if index is None or index in seen:
                continue
            values[index] = tail.strip()
            seen.add(index)
        if not seen:
            return NoFiling("no_known_identifier")
        return SealedSubmission(
            values=tuple(values),
            filed=tuple(i in seen for i in range(len(identifiers))),
            filed_rows=len(seen),
            omissions=tuple(identifiers[i] for i in range(len(identifiers)) if i not in seen),
        )

    def score(
        self, task: Task, canonical: Filing
    ) -> tuple[float, tuple[RowOutcome, ...]]:
        identifiers = self.row_identifiers(task.table)
        truth = task.key if task.key else ("",) * len(identifiers)
        submitted = canonical if isinstance(canonical, SealedSubmission) else None
        outcomes: list[RowOutcome] = []
        for i, identifier in enumerate(identifiers):
            got = "" if submitted is None else submitted.values[i]
            was_filed = False if submitted is None else submitted.filed[i]
            want = truth[i] if i < len(truth) else ""
            outcomes.append(
                RowOutcome(
                    ordinal=i + 1,
                    identifier=identifier,
                    filed=got,
                    correct=want,
                    matched=was_filed
                    and self.normalize_answer(got) == self.normalize_answer(want),
                    was_filed=was_filed,
                )
            )
        if not outcomes:
            return 0.0, ()
        return (
            round(sum(1 for o in outcomes if o.matched) / float(len(outcomes)), 6),
            tuple(outcomes),
        )

    # ----- the task text and the three cells -----

    def describe(self, task: PublicTask) -> str:
        return (
            "A gate vector, not a task anyone is meant to solve. File one line per "
            "row: the row id, a comma, and a value.\n\n"
            + "\n".join(self.row_identifiers(task.table))
        )

    def render_receipt(
        self, task: Task, canonical: Filing, truth: Sequence[str]
    ) -> ReceiptAST:
        graded = Task(
            label=task.label, task_id=task.task_id, surface=task.surface, table=task.table,
            text=task.text, key=tuple(truth),
        )
        _, outcomes = self.score(graded, canonical)
        rows = []
        for outcome in outcomes:
            slots = [Slot("verdict", "PASS" if outcome.matched else "FAIL")]
            if self._corrections:
                slots.append(
                    Slot("correction", "" if outcome.matched else outcome.correct)
                )
            rows.append(
                ReceiptRow(
                    ordinal=outcome.ordinal,
                    identifier=outcome.identifier,
                    observed=outcome.filed,
                    slots=tuple(slots),
                )
            )
        return ReceiptAST(
            kind=GRADED, task_id=task.task_id, row_count=len(rows), rows=tuple(rows)
        )

    def render_placebo(
        self, task: PublicTask, canonical: Filing, envelope: Envelope
    ) -> ReceiptAST:
        # A keyless task standing in for the scorer's shape. There is no key on the
        # argument and none is invented here: the placebo needs the filing echoed back
        # and the row identities, and nothing else.
        blind = Task(
            label=task.label, task_id=task.task_id, surface=task.surface, table=task.table,
            text="", key=(),
        )
        _, outcomes = self.score(blind, canonical)
        rows = tuple(
            ReceiptRow(
                ordinal=o.ordinal,
                identifier=o.identifier,
                observed=o.filed,
                slots=tuple(
                    Slot(spec.name, envelope.neutral[spec.name][o.ordinal - 1])
                    for spec in self.SLOTS
                ),
            )
            for o in outcomes
        )
        return ReceiptAST(
            kind=PLACEBO, task_id=task.task_id, row_count=len(rows), rows=rows
        )

    def render_oracle(
        self, task_id: str, convention: Mapping[str, str], row_count: int = 0
    ) -> ReceiptAST:
        return render_oracle_cell(self.ORACLE, task_id, convention, row_count)

    def parse_oracle(self, ast: ReceiptAST) -> dict[str, str]:
        return parse_oracle_cell(self.ORACLE, ast)


# --------------------------------------------------------------------------
# the vectors
# --------------------------------------------------------------------------


def _named_slots(c: int, n_axes: int = 2) -> VectorGenerator:
    """One row per axis, each storing its own axis's option. The known-dead shape."""
    axes = tuple(
        Axis(f"ax{i}", tuple(f"o{j}" for j in range(c)), "a slot stored verbatim")
        for i in range(n_axes)
    )

    def readout(convention: Mapping[str, str], row: int) -> str:
        # Distinct axes get disjoint value ranges, so a row never accidentally agrees
        # with another axis's option.
        return f"ax{row}:{convention[f'ax{row}']}"

    return VectorGenerator(
        name=f"slots-c{c}",
        genre="named slots stored verbatim",
        axes=axes,
        n_rows=n_axes,
        readout=readout,
        label=AXIS_LABEL,
    )


def _merge() -> VectorGenerator:
    """Two rows whose readouts merge different pairs; intersecting them separates three."""
    axis = Axis("k", ("o0", "o1", "o2"), "one axis, carried by two merging rows")
    table = {"o0": ("m0", "m0"), "o1": ("m0", "m1"), "o2": ("m1", "m1")}

    def readout(convention: Mapping[str, str], row: int) -> str:
        return table[convention["k"]][row]

    return VectorGenerator(
        name="merge", genre="crossed merging readouts", axes=(axis,), n_rows=2,
        readout=readout,
    )


def _one_row() -> VectorGenerator:
    """The same axis carried by a single row. Two signatures, whatever the readout."""
    axis = Axis("k", ("o0", "o1", "o2"), "one axis, carried by one row")

    def readout(convention: Mapping[str, str], row: int) -> str:
        return convention["k"]

    return VectorGenerator(
        name="one-row", genre="one axis on one row", axes=(axis,), n_rows=1,
        readout=readout,
    )


def _binary() -> VectorGenerator:
    """A binary axis alone. R is not asked of it, and nothing else can carry it."""
    axis = Axis("bin", ("a", "b"), "a binary axis")

    def readout(convention: Mapping[str, str], row: int) -> str:
        return convention["bin"]

    return VectorGenerator(
        name="binary", genre="a binary axis alone", axes=(axis,), n_rows=2,
        readout=readout,
    )


def _copies(n: int = 20) -> VectorGenerator:
    """Copies of one row plus one other. Item count with an unchanged readout is free."""
    axes = (
        Axis("ax0", tuple(f"o{j}" for j in range(4)), "the copied axis"),
        Axis("ax1", tuple(f"o{j}" for j in range(4)), "the other axis"),
    )

    def readout(convention: Mapping[str, str], row: int) -> str:
        return f"ax0:{convention['ax0']}" if row < n else f"ax1:{convention['ax1']}"

    return VectorGenerator(
        name=f"copies-{n}", genre="copies of one readout", axes=axes, n_rows=n + 1,
        readout=readout,
    )


def _affine(c: int = 3, n_rows: int = 4) -> VectorGenerator:
    """The correlated sampler, over the same declared slot catalogue as the source.

    Four declared slot axes of three options each: an assignment space of 81, which
    is what anyone counting assignments would report. The SAMPLER reaches nine of
    them, one per latent pair, because every slot is `(a * i + b) mod c`. Two resolved
    slots pin the rest.

    Declaring the slots as axes and the reachable assignments as the support is what
    makes this the source failure shape rather than an affine readout over two
    independent axes. The gates read the support, so they gate the nine.
    """
    axes = tuple(
        Axis(f"slot{i}", tuple(f"v{j}" for j in range(c)), "a slot the latent fills")
        for i in range(n_rows)
    )
    support = [
        {f"slot{i}": f"v{(a * i + b) % c}" for i in range(n_rows)}
        for a in range(c)
        for b in range(c)
    ]

    def readout(convention: Mapping[str, str], row: int) -> str:
        return convention[f"slot{row}"]

    return VectorGenerator(
        name="affine", genre="a correlated latent sampler", axes=axes, n_rows=n_rows,
        readout=readout, support=support,
    )


VECTORS: dict[str, VectorGenerator] = {
    generator.name: generator
    for generator in (
        _named_slots(3),
        _named_slots(4),
        _named_slots(6),
        _merge(),
        _one_row(),
        _binary(),
        _copies(20),
        _copies(1),
        _affine(),
    )
}


def generator_for(name: str) -> VectorGenerator:
    if name not in VECTORS:
        raise KeyError(f"no vector named {name!r}; this build carries {sorted(VECTORS)}")
    return VECTORS[name]


__all__ = ["VECTORS", "VectorGenerator", "VectorTable", "generator_for"]
