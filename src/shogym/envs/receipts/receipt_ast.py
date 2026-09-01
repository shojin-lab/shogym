"""The receipt as a structure, and the one serializer that turns it into bytes.

A renderer returns a ReceiptAST. It does not return text, and it does not choose
its own layout. One serializer, shared by every family, turns an AST into the bytes
the agent reads.

WHY THE RENDERER DOES NOT OWN THE BYTES. The gates ask what a receipt can tell an
agent, and the answer is a property of what the agent actually reads. A renderer
that returned text could tell the gates one thing and the agent another: it could
order rows informatively, pad one class of row differently from another, or print
an identifier that correlates with the drawn option, and a gate reading the
renderer's declared intentions would see none of it. Handing the AST to one
serializer makes the bytes a function of the structure, so the structure is a
faithful object to gate.

THE ENVELOPE. The three cells a fork can serve, graded, placebo and oracle, are
read into a context, so they share one envelope: a single registered size that does
not depend on the drawn convention, reached by padding with a committed filler
stream. Graded and placebo are structurally congruent, meaning identical wrapper,
identifiers, order, offsets, whitespace and padding, and they differ only inside
registered fixed-width SLOTS. The oracle shares the size and the outer wrapper but
has no rows to align.

Fixed-width everywhere is what makes that checkable rather than hoped for. Every
row line is the same length, built from registered widths, so a slot occupies a
known byte range on a known line and the envelope check can mask those ranges and
assert the rest never moves. A layout that stripped trailing spaces, or sized a
column from the longest value in it, would put the answer key into the byte count.

ASCII only. Byte offsets and character offsets have to be the same number for the
slot arithmetic to mean anything, so the serializer refuses a value it cannot
encode as ASCII rather than silently shifting every offset after it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

GRADED = "graded"
PLACEBO = "placebo"
ORACLE = "oracle"
KINDS = (GRADED, PLACEBO, ORACLE)

#: The wrapper's first line, identical across every family and every cell.
BANNER = "SUBMISSION RECEIPT"
#: What a truncated observed value ends with, so truncation is visible and fixed width.
TRUNCATION_MARK = "~"
#: Column gap, and the width the printed ordinal takes.
GAP = "  "
ORDINAL_WIDTH = 4
#: How the padding is introduced. Fixed text, so it costs the same bytes every time.
FILLER_LEAD = "reference block: "


@dataclass(frozen=True)
class SlotSpec:
    """A registered fixed-width field that graded and placebo are allowed to differ in.

    The GRAMMAR is the point. A slot is not a free string: it prints a value drawn
    from a closed, registered set, and admission checks every value the slot realizes
    over the whole convention support against that set. Without it a renderer can put
    the drawn rule in a slot as a short code, print no axis name and no option token,
    and hand the child a lookup table while the run records induction.

    `vocabulary` is the literal set. `allows_answers` additionally admits any value
    the scorer can produce as a correct answer, which is what a correction slot is
    for. `allows_empty` admits the empty string.
    """

    name: str
    width: int
    vocabulary: tuple[str, ...] = ()
    allows_answers: bool = False
    allows_empty: bool = False

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError(f"slot {self.name!r} needs a positive width")
        if not self.vocabulary and not self.allows_answers and not self.allows_empty:
            raise ValueError(
                f"slot {self.name!r} registers no grammar; a slot that may print "
                "anything can print the rule"
            )

    def allowed(self, answers: frozenset[str]) -> frozenset[str]:
        """Everything this slot may print, given the family's answer vocabulary."""
        out = set(self.vocabulary)
        if self.allows_answers:
            out |= set(answers)
        if self.allows_empty:
            out.add("")
        return frozenset(out)


@dataclass(frozen=True)
class Slot:
    """One slot's value on one row."""

    name: str
    value: str


@dataclass(frozen=True)
class ReceiptRow:
    """One printed row: its position, the identifier it names, what was filed, its slots."""

    ordinal: int
    identifier: str
    observed: str
    slots: tuple[Slot, ...]


@dataclass(frozen=True)
class ReceiptAST:
    """A rendered cell, before it is bytes.

    `rows` is empty for the oracle, which has no row alignment to share. `body` is
    the prose beneath the rows: nothing for a graded or placebo cell, the rule
    template for an oracle.
    """

    kind: str
    task_id: str
    row_count: int
    rows: tuple[ReceiptRow, ...] = ()
    body: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"a cell is one of {KINDS}, not {self.kind!r}")
        if self.kind == ORACLE and self.rows:
            raise ValueError("the oracle cell has no rows to align")

    def slot_names(self) -> tuple[str, ...]:
        return tuple(slot.name for slot in self.rows[0].slots) if self.rows else ()


@dataclass(frozen=True)
class Envelope:
    """The registered shape every cell of one family pads to.

    `size` is convention-independent by construction: it is computed from the
    family's maximum row count and the registered column widths, never from the
    values a particular draw happened to produce.
    """

    size: int
    identifier_width: int
    observed_width: int
    slots: tuple[SlotSpec, ...]
    filler: str
    #: One title per slot, shared by every cell kind. There is deliberately no way
    #: to give the graded and placebo cells different headers: a column title is
    #: part of the wrapper, and a wrapper that moved between the two would be a
    #: difference the registered slots do not account for.
    column_titles: tuple[str, ...] = ()
    #: The placebo's slot values: slot name to one neutral token per row position.
    #: Drawn from the family's filler alphabet by the committed stream, fixed
    #: before launch, so a placebo slot carries no more than its own width.
    neutral: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("an envelope registers at least one slot")
        if len({s.name for s in self.slots}) != len(self.slots):
            raise ValueError("slot names have to be distinct")

    @property
    def row_line_width(self) -> int:
        """Bytes in one row line, newline included. The same for every row, always."""
        widths = [ORDINAL_WIDTH, self.identifier_width, self.observed_width]
        widths += [s.width for s in self.slots]
        return len(GAP) + sum(widths) + len(GAP) * (len(widths) - 1) + 1

    def slot_span(self, name: str) -> tuple[int, int]:
        """Where one slot sits inside a row line, as a byte range."""
        offset = len(GAP) + ORDINAL_WIDTH + len(GAP) + self.identifier_width
        offset += len(GAP) + self.observed_width
        for spec in self.slots:
            offset += len(GAP)
            if spec.name == name:
                return offset, offset + spec.width
            offset += spec.width
        raise KeyError(f"no registered slot named {name!r}")

    def header_line(self) -> str:
        """The column titles. One line, the same bytes in every cell that has rows."""
        titles = self.column_titles or tuple(spec.name for spec in self.slots)
        if len(titles) != len(self.slots):
            raise ValueError(
                f"the envelope declares {len(titles)} column titles for "
                f"{len(self.slots)} slots"
            )
        cells = [
            _fit("#", ORDINAL_WIDTH),
            _fit("record", self.identifier_width),
            _fit("filed", self.observed_width),
        ]
        cells += [_fit(t, spec.width) for t, spec in zip(titles, self.slots)]
        return GAP + GAP.join(cells)


def frozen_envelope(envelope: Envelope) -> Envelope:
    """A copy of the registered envelope nothing downstream can rewrite.

    `Envelope` is frozen at the dataclass level, which stops a field being replaced
    and does nothing about a mutable mapping inside one. The committed neutral tokens
    are exactly such a mapping, and they are the pre-launch commitment the placebo
    arm's whole meaning rests on: a renderer handed the live dictionary can rewrite
    what it is about to be compared against, and the comparison then passes.

    Handing family code a proxy over tuples makes that a failure rather than a
    rewrite, and building the expected cell from this copy makes the comparison
    independent of anything the renderer does.
    """
    return Envelope(
        size=envelope.size,
        identifier_width=envelope.identifier_width,
        observed_width=envelope.observed_width,
        slots=tuple(envelope.slots),
        filler=envelope.filler,
        column_titles=tuple(envelope.column_titles),
        neutral=MappingProxyType(
            {name: tuple(values) for name, values in envelope.neutral.items()}
        ),
    )


def _fit(value: str, width: int) -> str:
    """One field at exactly `width` bytes: padded, or truncated with the fixed mark."""
    text = "" if value is None else str(value)
    if not text.isascii():
        raise ValueError(f"a receipt field has to be ascii, got {text!r}")
    if len(text) > width:
        if width <= len(TRUNCATION_MARK):
            return text[:width]
        return text[: width - len(TRUNCATION_MARK)] + TRUNCATION_MARK
    return text.ljust(width)


def _row_cells(row: ReceiptRow, envelope: Envelope) -> list[str]:
    """One row's registered fields, at their registered widths, in canonical order.

    The schema is closed. A row carries exactly the registered slot names, once
    each, and anything else is refused rather than dropped. Silently ignoring an
    unregistered slot would let a renderer carry a field the bytes never show and
    the gates would read it; letting a duplicate name through would let the last one
    win in a dict while an earlier one was what a reader of the structure saw.
    """
    names = [slot.name for slot in row.slots]
    registered = [spec.name for spec in envelope.slots]
    if len(names) != len(set(names)):
        raise ValueError(f"row {row.ordinal} repeats a slot name: {names}")
    if set(names) != set(registered):
        raise ValueError(
            f"row {row.ordinal} carries slots {sorted(names)} against the registered "
            f"{sorted(registered)}"
        )
    by_name = {slot.name: slot.value for slot in row.slots}
    return [_fit(by_name[spec.name], spec.width) for spec in envelope.slots]


def wrapper_lines(ast: ReceiptAST) -> list[str]:
    """The outer wrapper, identical in shape across all three cells."""
    return [BANNER, f"reference: {ast.task_id}", f"records: {ast.row_count:04d}", ""]


def body_of(ast: ReceiptAST, envelope: Envelope) -> str:
    """Everything before the padding, as one string."""
    lines = wrapper_lines(ast)
    if ast.rows:
        lines.append(envelope.header_line())
        for row in ast.rows:
            cells = [
                _fit(str(row.ordinal), ORDINAL_WIDTH),
                _fit(row.identifier, envelope.identifier_width),
                _fit(row.observed, envelope.observed_width),
            ]
            cells.extend(_row_cells(row, envelope))
            lines.append(GAP + GAP.join(cells))
    lines.append("")
    lines.extend(ast.body)
    lines.append("")
    return "\n".join(lines)


def serialize(ast: ReceiptAST, envelope: Envelope) -> bytes:
    """The bytes the agent reads. Deterministic, and the same code for every family."""
    text = body_of(ast, envelope)
    used = len(text.encode("ascii")) + len(FILLER_LEAD) + 1
    room = envelope.size - used
    if room < 0:
        raise ValueError(
            f"the {ast.kind} cell needs {used - envelope.size} bytes more than the "
            f"registered envelope of {envelope.size}"
        )
    if room > len(envelope.filler):
        raise ValueError(
            f"the committed filler is {len(envelope.filler)} bytes and this cell needs "
            f"{room}; commit a longer stream before launch"
        )
    out = (text + FILLER_LEAD + envelope.filler[:room] + "\n").encode("ascii")
    if len(out) != envelope.size:
        raise ValueError(f"serialized {len(out)} bytes against an envelope of {envelope.size}")
    return out


def rows_region(ast: ReceiptAST, envelope: Envelope) -> tuple[int, int]:
    """Where the row block starts and ends in the serialized bytes."""
    head = len("\n".join(wrapper_lines(ast))) + 1
    if not ast.rows:
        return head, head
    head += len(envelope.header_line()) + 1
    return head, head + len(ast.rows) * envelope.row_line_width


def row_lines(payload: bytes, ast: ReceiptAST, envelope: Envelope) -> tuple[bytes, ...]:
    """The serialized bytes of each printed row, sliced out of the payload.

    This is what a reader of one row actually reads: the ordinal, the identifier, the
    echoed filing and the registered fields, all at their registered widths and
    already truncated. Reading the structure instead would read values the bytes
    never carried.
    """
    start, end = rows_region(ast, envelope)
    width = envelope.row_line_width
    return tuple(
        payload[start + i * width: start + (i + 1) * width] for i in range(len(ast.rows))
    )


def slot_ranges(ast: ReceiptAST, envelope: Envelope) -> list[tuple[int, int]]:
    """Every registered slot's byte range in the serialized cell, in order.

    This is what the envelope check masks: outside these ranges, a graded cell and
    a placebo cell of the same instance and filing must be byte-identical, and a
    graded cell must not move when the drawn convention moves.
    """
    start, _ = rows_region(ast, envelope)
    out: list[tuple[int, int]] = []
    for index in range(len(ast.rows)):
        line = start + index * envelope.row_line_width
        for spec in envelope.slots:
            lo, hi = envelope.slot_span(spec.name)
            out.append((line + lo, line + hi))
    return out


def mask_slots(payload: bytes, ranges: Iterable[tuple[int, int]], fill: bytes = b"?") -> bytes:
    """The payload with every slot range blanked, for comparing everything else."""
    out = bytearray(payload)
    for lo, hi in ranges:
        out[lo:hi] = fill * (hi - lo)
    return bytes(out)


def envelope_size_for(
    max_rows: int,
    identifier_width: int,
    observed_width: int,
    slots: Sequence[SlotSpec],
    body_allowance: int,
    task_id_width: int = 16,
) -> int:
    """The registered size for a family, from its maximum shape and nothing else.

    Convention-independent because every input is: the family's largest row count,
    the widths registered for its columns, and how much room its rule template
    needs. A size derived from one draw's values would move with the draw.
    """
    probe = Envelope(
        size=1,
        identifier_width=identifier_width,
        observed_width=observed_width,
        slots=tuple(slots),
        filler="",
    )
    wrapper = len(BANNER) + 1
    wrapper += len("reference: ") + task_id_width + 1
    wrapper += len("records: 0000") + 1
    wrapper += 1
    header = len(probe.header_line()) + 1
    rows = max_rows * probe.row_line_width
    tail = 1 + body_allowance + 1 + len(FILLER_LEAD) + 1
    return wrapper + header + rows + tail


__all__ = [
    "BANNER",
    "GRADED",
    "KINDS",
    "ORACLE",
    "PLACEBO",
    "Envelope",
    "ReceiptAST",
    "ReceiptRow",
    "Slot",
    "SlotSpec",
    "body_of",
    "envelope_size_for",
    "frozen_envelope",
    "mask_slots",
    "row_lines",
    "rows_region",
    "serialize",
    "slot_ranges",
]
