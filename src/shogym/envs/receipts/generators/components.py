"""The components genre: counting islands on small boards, under an unknown contact rule.

A schedule of 24 separate 5 by 5 boards, each with one to six occupied cells printed as
coordinate pairs, and one question per board: how many islands are there. The mechanics
of an island are completely public. What is not public is the one decision that says
which two occupied cells touch:

    contact_kernel   3 options   a shared side, a shared corner only, or either

That is three conventions, drawn uniformly. Sibling schedules A and B are two different
sets of boards scored under the same drawn rule.

WHY ONE TERNARY DECISION IS ENOUGH HERE. The answer a row prints is a count the rule
COMPUTES, not the option a reader chose, and the recipe puts six rows in each of three
strata: six where a shared side answers differently from the other two, six where a
corner-only rule does, six where the combined rule does, and six invariant boards that
every rule answers alike. Every pair of rules therefore differs on exactly twelve of the
twenty four rows, no single row names the drawn rule, and the corrected rows together
name it exactly. A reader with no receipt can do no better than each row's modal count,
which is three quarters; a reader who has resolved the rule scores one.

WHY A RELATION AND NOT ANOTHER PROCEDURE. The hidden decision here selects a spatial
adjacency predicate and the answer counts the equivalence classes its paths generate.
Transitivity is the whole of the work: two cells with no direct contact can still share
an island, and a cycle merges nothing extra, so the count is not a sum of contacts and
cannot be reached by subtracting edges from vertices. Six vertices with five side edges
can still be two islands.

WHY THE BOARDS ARE SMALL AND THE ARITHMETIC IS NONE, AND WHAT THAT IS NOT AN ANSWER TO.
A schedule holds at most 129 occupied cells and 305 unordered pairs, and every answer is
one digit. That is a deliberate choice about what a copy has to do once it has the rule.
It is NOT a repair for the low oracle grade the fourth engineering run recorded: that
run's loss was in uptake and in scope, not in execution, and the repairs to those are the
registered wording this genre inherits. Small boards buy a clean separation between a
copy that never took the rule and a copy that took it and miscounted. They do not
establish that a reader will use the oracle's prose, which is what the model screen is
for.

ONE USE PER CHAIN. Three options is a small support, and the release restriction is part
of the design rather than a caution: if a previous draw were known and repetition
forbidden, two rules would remain and the lookup floor's single all-passed bit separates
those two outright, so the headroom this genre is admitted on would be zero. The bank's
master key is separate from every other genre's for the same reason, because the
convention stream's coordinates carry the ordinal and not the generator name.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts import streams
from shogym.envs.receipts.copy_profiles import ORDERED_TOKENS
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse as parse_oracle_cell
from shogym.envs.receipts.oracle import render as render_oracle_cell
from shogym.envs.receipts.protocol import (
    ROW_ADDITIVE_EQUAL_WEIGHT,
    Axis,
    Column,
    ConstructionExhausted,
    Filing,
    PublicTask,
    RowOutcome,
    Shape,
    Task,
)
from shogym.envs.receipts.receipt_ast import (
    Envelope,
    ReceiptAST,
    SlotSpec,
    envelope_size_for,
)
from shogym.envs.receipts.render import graded_receipt, placebo_receipt
from shogym.receipts import ROW_LABEL

# ----- the board ---------------------------------------------------------------
# One size, one notation. Coordinates are one-based because that is how the schedule
# prints them, and the printed form is the only form a reader ever sees.

SIDE = 5
MIN_CELLS = 1
MAX_CELLS = 6
#: Every cell of the board, in the order the catalogue enumerates subsets of it.
BOARD: tuple[tuple[int, int], ...] = tuple(
    (row, column) for row in range(1, SIDE + 1) for column in range(1, SIDE + 1)
)

#: The three contact rules, as the axis spells them.
SIDE_CONTACTS = "side_contacts"
CORNER_CONTACTS = "corner_contacts"
COMBINED_CONTACTS = "combined_contacts"

AXES: tuple[Axis, ...] = (
    Axis(
        "contact_kernel",
        (SIDE_CONTACTS, CORNER_CONTACTS, COMBINED_CONTACTS),
        "which two occupied cells count as touching",
    ),
)

#: The stratum letters, which are private construction labels and never printed. A row's
#: stratum is the one rule whose count differs from the other two on that board.
STRATA = ("S", "C", "U")
#: Which option each stratum letter is the odd one out for, in the option order above.
STRATUM_OPTION = {"S": SIDE_CONTACTS, "C": CORNER_CONTACTS, "U": COMBINED_CONTACTS}

#: The complete public answer vocabulary, in its published order. A board of one to six
#: cells has one to six islands, so this is the whole of what any row can be.
ANSWER_RANKS: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")

#: What the receipt prints for a row filed with an empty value, and for a row filed with
#: nothing at all. Two different acts, two different tokens, and neither is a legal
#: answer: every board has at least one island, so no canonical answer is ever empty.
BLANK_TOKEN = "(empty)"
UNFILED_TOKEN = "(unfiled)"

ROWS = 24
#: Six rows in each informative stratum, then one invariant control of each size.
STRATUM_ROWS = 6
CONTROL_SIZES = (1, 2, 3, 4, 5, 6)

SHAPE = Shape(
    columns=(
        Column("identifier", "sibling prefix and ten keyed hexadecimal digits"),
        Column("occupied cells", "keyed subset of one through six cells on a 5 by 5 board"),
    ),
    rows=ROWS,
    case="one independent nonempty occupied-cell pattern",
    note="six rows in each merging stratum and six separated-cell controls",
)

#: The one surface. There is one notation and one semantic template here, and a second
#: organisation name would vary nothing a reader could act on.
SURFACE = "occupied_cells_5x5"

# ----- the registered envelope constants ---------------------------------------
# Family maxima and fixed widths, never a value read off one draw. The coordinate list
# can run to 35 bytes and belongs in the task table: an observed value is the submitted
# count, not the input pattern, so no receipt field has to hold a board.

IDENTIFIER_WIDTH = 12
OBSERVED_WIDTH = 16
VERDICT_WIDTH = 4
CORRECTION_WIDTH = 12
VERDICT_TOKENS = ("PASS", "FAIL")
#: No blank-answer display token in the correction grammar. A board's canonical answer is
#: a positive count, so the correction slot never has an empty truth to stand in for.
SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("verdict", VERDICT_WIDTH, vocabulary=VERDICT_TOKENS),
    SlotSpec("correction", CORRECTION_WIDTH, allows_answers=True, allows_empty=True),
)
COLUMN_TITLES = ("mark", "note")
#: Four consonants and nothing else. A neutral token drawn from these can never normalize
#: to a count, to PASS or FAIL, or to either status token.
FILLER_ALPHABET = "qzxv"
ORACLE_BODY_ALLOWANCE = 1100
ENVELOPE_SIZE = envelope_size_for(
    max_rows=ROWS,
    identifier_width=IDENTIFIER_WIDTH,
    observed_width=OBSERVED_WIDTH,
    slots=SLOTS,
    body_allowance=ORACLE_BODY_ALLOWANCE,
)

# ----- the bounded construction's registered bounds -----------------------------

#: How many proposals one ordinal may try, and how many orderings each surviving
#: proposal may try. Both are registered: a keyed sweep of ordinals 0 through 63 under
#: the public test master needed proposal indices from 2 to 1047, so a bound of 1024
#: would have turned one ordinal into a whole-bank failure. Running out is a named
#: failure and never a widened bound.
MAX_PROPOSALS = 4096
MAX_ORDERINGS = 64
#: No count may appear more than this many times in a side's answers under any rule.
MAX_ANSWER_FREQUENCY = 8
#: The most correct rows a copy of A's answers may reach on B, over every row move and
#: every bijection of the six counts. Twelve of twenty four is the registered half.
MAX_COPY_ROWS = 12

#: What a provenance record says this generator's construction was bounded by.
CONSTRUCTION_BOUNDS: dict[str, int] = {
    "proposals": MAX_PROPOSALS,
    "orderings": MAX_ORDERINGS,
    "rows": ROWS,
    "answer_frequency": MAX_ANSWER_FREQUENCY,
    "copy_rows": MAX_COPY_ROWS,
}

Board = tuple[tuple[int, int], ...]
ClassId = tuple[int, ...]


# --------------------------------------------------------------------------
# the contact rules, and the count they generate
# --------------------------------------------------------------------------


def linked(one: tuple[int, int], other: tuple[int, int], option: str) -> bool:
    """Whether two distinct occupied cells touch under one rule.

    Boards do not wrap and a border cell obeys the same rule as any other, so this is
    arithmetic on the coordinate difference and nothing else.
    """
    down = abs(one[0] - other[0])
    across = abs(one[1] - other[1])
    if option == SIDE_CONTACTS:
        return down + across == 1
    if option == CORNER_CONTACTS:
        return down == 1 and across == 1
    if option == COMBINED_CONTACTS:
        return max(down, across) == 1
    raise ValueError(f"no contact rule named {option!r}")


def check_board(cells: Sequence[tuple[int, int]]) -> None:
    """Refuse a board that is not one, rather than giving it an answer.

    Invalid generator data has no island count, and assigning it one would put a number
    on the schedule that no rule produced. Every condition here is a public property of
    the printed row.
    """
    if not MIN_CELLS <= len(cells) <= MAX_CELLS:
        raise ValueError(
            f"a board carries {MIN_CELLS} to {MAX_CELLS} occupied cells, not {len(cells)}"
        )
    for cell in cells:
        if len(cell) != 2 or not all(1 <= part <= SIDE for part in cell):
            raise ValueError(f"{cell!r} is not a cell of a {SIDE} by {SIDE} board")
    if len(set(cells)) != len(cells):
        raise ValueError("a board lists each occupied cell once")


def _roots(cells: Sequence[tuple[int, int]], option: str) -> int:
    """Islands by union-find over every unordered pair. No validation, no shortcut.

    Start with one set per occupied cell, union the endpoints of every permitted contact,
    and count the sets that are left. Subtracting contacts from cells would be wrong: a
    contact whose endpoints already share a root merges nothing, which is exactly what a
    cycle does.
    """
    size = len(cells)
    parent = list(range(size))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    islands = size
    for i in range(size):
        one = cells[i]
        for j in range(i + 1, size):
            if not linked(one, cells[j], option):
                continue
            left, right = find(i), find(j)
            if left != right:
                parent[left] = right
                islands -= 1
    return islands


def components_of(cells: Sequence[tuple[int, int]], option: str) -> int:
    """The number of islands on one valid board under one rule."""
    check_board(cells)
    return _roots(cells, option)


def key_for(table: "ComponentsTable", convention: Mapping[str, str]) -> tuple[str, ...]:
    """The correct answer for every row under one convention: one island count per row."""
    option = convention["contact_kernel"]
    return tuple(str(components_of(row.cells, option)) for row in table.rows)


ALL_CONVENTIONS: tuple[dict[str, str], ...] = tuple(
    {"contact_kernel": option} for option in AXES[0].options
)


# --------------------------------------------------------------------------
# the catalogue: every board, classified once
# --------------------------------------------------------------------------


def counts_of(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int]:
    """The three island counts of one board, in the axis's own option order."""
    return (
        _roots(cells, SIDE_CONTACTS),
        _roots(cells, CORNER_CONTACTS),
        _roots(cells, COMBINED_CONTACTS),
    )


def stratum_of(triple: tuple[int, int, int]) -> str:
    """Which rule this board's counts single out, or the empty string for neither.

    A board is informative when exactly two of the three counts agree: the third rule is
    the one it distinguishes. Three distinct counts would hand a reader more than the
    recipe prices, and three equal counts distinguish nothing.
    """
    side, corner, combined = triple
    agreements = (side == corner) + (side == combined) + (corner == combined)
    if agreements != 1:
        return ""
    if corner == combined:
        return "S"
    if side == combined:
        return "C"
    return "U"


def side_structure(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int]:
    """(side contacts, cells, side islands) of one board, for the cycle witness."""
    contacts = sum(
        1
        for i in range(len(cells))
        for j in range(i + 1, len(cells))
        if linked(cells[i], cells[j], SIDE_CONTACTS)
    )
    return contacts, len(cells), _roots(cells, SIDE_CONTACTS)


def has_side_cycle(cells: Sequence[tuple[int, int]]) -> bool:
    """Whether the side-contact graph of this board closes a cycle.

    A forest has exactly cells minus islands edges, so more than that is a cycle, and a
    cycle is what defeats reading the island count off a contact tally.
    """
    contacts, size, islands = side_structure(cells)
    return contacts > size - islands


def class_identifier(cells: Sequence[tuple[int, int]]) -> ClassId:
    """This board's colour-preserving graph class, as an exact isomorphism invariant.

    Every unordered pair of occupied cells carries a colour: 1 for a shared side, 2 for a
    shared corner, 0 for neither. A class preserves all three colours, so two boards in
    one class are the same labelled contact graph however they sit on the board, and
    translating, rotating, reflecting or otherwise re-embedding a graph cannot produce a
    second class.

    The invariant is exact rather than heuristic. Any colour-preserving isomorphism
    preserves each vertex's side degree and corner degree, so it maps each degree group
    onto itself; minimizing the upper-triangle colour tuple over the permutations INSIDE
    those groups therefore minimizes over every isomorphism, and two boards get the same
    identifier exactly when they are isomorphic. The vertex count leads it so that boards
    of different sizes never collide.
    """
    size = len(cells)
    colours = [[_colour(cells[i], cells[j]) for j in range(size)] for i in range(size)]
    degrees = [
        (
            sum(1 for j in range(size) if colours[i][j] == 1),
            sum(1 for j in range(size) if colours[i][j] == 2),
        )
        for i in range(size)
    ]
    ordered = sorted(range(size), key=lambda i: degrees[i])
    groups = [
        tuple(run)
        for _, run in itertools.groupby(ordered, key=lambda i: degrees[i])
    ]
    pairs = _PAIRS[size]
    if all(len(group) == 1 for group in groups):
        order = [group[0] for group in groups]
        return (size,) + tuple(colours[order[i]][order[j]] for i, j in pairs)
    best = min(
        tuple(
            colours[order[i]][order[j]]
            for i, j in pairs
        )
        for order in (
            [vertex for group in arrangement for vertex in group]
            for arrangement in itertools.product(
                *(itertools.permutations(g) for g in groups)
            )
        )
    )
    return (size,) + best


def _colour(one: tuple[int, int], other: tuple[int, int]) -> int:
    """1 for a shared side, 2 for a shared corner, 0 for neither."""
    down = abs(one[0] - other[0])
    across = abs(one[1] - other[1])
    if down + across == 1:
        return 1
    if down == 1 and across == 1:
        return 2
    return 0


#: The upper-triangle index pairs for each board size, built once.
_PAIRS = {
    size: tuple((i, j) for i in range(size) for j in range(i + 1, size))
    for size in range(MIN_CELLS, MAX_CELLS + 1)
}


@dataclass(frozen=True)
class Catalogue:
    """Every eligible board on the 5 by 5 grid, classified and grouped.

    `classes` maps each stratum letter to its sorted class identifiers, `embeddings` maps
    a class identifier to every board that realizes it in sorted order, `counts` and
    `cycles` are the two class invariants the construction filter reads, and `controls`
    maps a size to every board of that size with no contact at all under any rule.

    It is code-derived: the whole of it comes from enumerating subsets of the board and
    nothing is read from a data file. Held in memory once per process, and clearing the
    cache rebuilds an identical object.
    """

    classes: Mapping[str, tuple[ClassId, ...]]
    embeddings: Mapping[ClassId, tuple[Board, ...]]
    counts: Mapping[ClassId, tuple[int, int, int]]
    cycles: Mapping[ClassId, bool]
    controls: Mapping[int, tuple[Board, ...]]


@lru_cache(maxsize=1)
def catalogue() -> Catalogue:
    """The finite catalogue, enumerated exactly and cached in memory.

    Every subset of one to six of the twenty five cells, in increasing size and then
    lexicographic order: 245505 of them. Boards whose three counts are all equal are
    contacts-free controls when the count equals the cell count and are otherwise
    discarded; boards with three distinct counts are discarded; the rest are the
    informative pools, grouped by colour-preserving graph class.
    """
    grouped: dict[str, dict[ClassId, list[Board]]] = {letter: {} for letter in STRATA}
    counts: dict[ClassId, tuple[int, int, int]] = {}
    cycles: dict[ClassId, bool] = {}
    controls: dict[int, list[Board]] = {size: [] for size in CONTROL_SIZES}
    for size in range(MIN_CELLS, MAX_CELLS + 1):
        for cells in itertools.combinations(BOARD, size):
            triple = counts_of(cells)
            letter = stratum_of(triple)
            if not letter:
                if triple == (size, size, size):
                    controls[size].append(cells)
                continue
            identifier = class_identifier(cells)
            pool = grouped[letter].setdefault(identifier, [])
            if not pool:
                counts[identifier] = triple
                cycles[identifier] = has_side_cycle(cells)
            pool.append(cells)
    return Catalogue(
        classes={
            letter: tuple(sorted(grouped[letter])) for letter in STRATA
        },
        embeddings={
            identifier: tuple(sorted(boards))
            for letter in STRATA
            for identifier, boards in grouped[letter].items()
        },
        counts=dict(counts),
        cycles=dict(cycles),
        controls={size: tuple(sorted(boards)) for size, boards in controls.items()},
    )


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentsRow:
    """One printed row: its identifier and its occupied cells in printed order."""

    row_id: str
    cells: Board


@dataclass(frozen=True)
class ComponentsTable:
    """One side's schedule. Immutable, and its printed body is a function of its rows.

    The body is rendered on demand rather than stored, so there is no derived copy that
    could drift from the rows the bank commits to: what a reader sees is computed from
    exactly the fields `table_record` hashes.
    """

    surface: str
    rows: tuple[ComponentsRow, ...]

    @property
    def body(self) -> str:
        """The printed table: the header, the rule line, and one line per row."""
        lines = [TABLE_HEAD, TABLE_RULE]
        lines += [
            "| %s | %s |" % (row.row_id, print_cells(row.cells)) for row in self.rows
        ]
        return "\n".join(lines)


TABLE_HEAD = "| identifier | occupied cells |"
TABLE_RULE = "| ---------- | -------------- |"


def print_cells(cells: Iterable[tuple[int, int]]) -> str:
    """The occupied cells as the schedule prints them: `(row,column)` separated by spaces."""
    return " ".join("(%d,%d)" % (row, column) for row, column in cells)


# --------------------------------------------------------------------------
# the bounded pair construction
# --------------------------------------------------------------------------


def _side_rng(
    master: bytes, label: str, ordinal: int, proposal: int, *suffix: object
) -> random.Random:
    """One construction stream. Every coordinate tuple begins the same way.

    The side's own label separates A's draws from B's, the generator name separates this
    genre's from another's under one key, and the ordinal and proposal separate one
    attempt from the next, so a rejected proposal takes nothing from the streams the
    accepted one reads.
    """
    stream = streams.SURFACE_A if label == "A" else streams.SURFACE_B
    return streams.rng(
        master, stream, "components", int(ordinal), int(proposal), *suffix
    )


def _choose_classes(
    master: bytes, ordinal: int, proposal: int
) -> dict[str, dict[str, list[ClassId]]]:
    """Six classes per stratum per side, with no class shared within or across siblings."""
    held = catalogue()
    chosen: dict[str, dict[str, list[ClassId]]] = {"A": {}, "B": {}}
    for letter in STRATA:
        library = list(held.classes[letter])
        first = _side_rng(master, "A", ordinal, proposal, "classes", letter).sample(
            library, STRATUM_ROWS
        )
        taken = set(first)
        second = _side_rng(master, "B", ordinal, proposal, "classes", letter).sample(
            [identifier for identifier in library if identifier not in taken], STRATUM_ROWS
        )
        chosen["A"][letter] = first
        chosen["B"][letter] = second
    return chosen


def _proposal_triples(
    chosen: Mapping[str, Mapping[str, Sequence[ClassId]]], side: str
) -> tuple[tuple[tuple[int, int, int], ...], bool]:
    """One side's 24 count triples in recipe order, and whether it holds a side cycle.

    Both are class invariants, so the whole of step four and step five is decided before
    a single coordinate embedding is drawn. Which embedding a class takes cannot change a
    count, a frequency or a copy maximum.
    """
    held = catalogue()
    triples: list[tuple[int, int, int]] = []
    cycle = False
    for letter in STRATA:
        for identifier in chosen[side][letter]:
            triples.append(held.counts[identifier])
            cycle = cycle or held.cycles[identifier]
    triples += [(size, size, size) for size in CONTROL_SIZES]
    return tuple(triples), cycle


def _boards(
    master: bytes,
    ordinal: int,
    proposal: int,
    side: str,
    chosen: Mapping[str, Mapping[str, Sequence[ClassId]]],
    forbidden: Mapping[int, Board] | None = None,
) -> list[Board]:
    """One side's 24 boards in recipe order: the informative embeddings, then the controls."""
    held = catalogue()
    out: list[Board] = []
    for letter in STRATA:
        for position, identifier in enumerate(chosen[side][letter]):
            picker = _side_rng(
                master, side, ordinal, proposal, "embedding", letter, position
            )
            out.append(picker.choice(list(held.embeddings[identifier])))
    for size in CONTROL_SIZES:
        pool = list(held.controls[size])
        if forbidden is not None and size in forbidden:
            pool = [board for board in pool if board != forbidden[size]]
        picker = _side_rng(master, side, ordinal, proposal, "control", size)
        out.append(picker.choice(pool))
    return out


def _ordered(master: bytes, ordinal: int, proposal: int, ordering: int, side: str,
             values: Sequence[Any]) -> list[Any]:
    """One side's rows in a proposed printed order: a fresh copy, shuffled in place."""
    shuffled = list(values)
    _side_rng(master, side, ordinal, proposal, "order", ordering).shuffle(shuffled)
    return shuffled


def _identifiers(master: bytes, ordinal: int, proposal: int, ordering: int,
                 side: str) -> list[str]:
    """One identifier per row in printed order: the prefix and ten hexadecimal digits."""
    picker = _side_rng(master, side, ordinal, proposal, "identifiers", ordering)
    return [
        "%s%010X" % (side, value)
        for value in picker.sample(range(16 ** 10), ROWS)
    ]


@lru_cache(maxsize=32)
def build_pair(master: bytes, ordinal: int) -> tuple[ComponentsTable, ComponentsTable]:
    """Both sibling schedules for one ordinal, from the keyed streams and nothing else.

    PURE, AND BOTH SIDES AT ONCE. Every acceptance condition is a statement about the
    PAIR: the classes B may draw depend on the classes A drew, and the copy maximum is a
    property of the two answer vectors together, so neither side can be accepted alone
    and the two `build_table` calls read one answer out of here.

    THE FILTER NEVER CONSULTS THE DRAWN CONVENTION. Every predicate quantifies over all
    three rules, so acceptance is a function of the two schedules alone. A filter that
    read the live draw would make the fact that this pair was served informative, and an
    invariant description would not repair that.

    Running out of proposals is `ConstructionExhausted`: a whole-bank failure with a name
    and a reproducible count of what refused what, never a skipped ordinal and never a
    widened bound. The cache is controller-side and small; clearing it rebuilds the same
    tables.
    """
    from shogym.envs.receipts.generators import components_audit as audit

    refused = {"frequency": 0, "cycle": 0, "ordering": 0}
    for proposal in range(MAX_PROPOSALS):
        chosen = _choose_classes(bytes(master), int(ordinal), proposal)
        triples = {side: _proposal_triples(chosen, side) for side in ("A", "B")}
        if any(audit.frequency_refusal(triples[side][0]) for side in ("A", "B")):
            refused["frequency"] += 1
            continue
        if any(not triples[side][1] for side in ("A", "B")):
            refused["cycle"] += 1
            continue
        for ordering in range(MAX_ORDERINGS):
            laid = {
                side: _ordered(
                    bytes(master), int(ordinal), proposal, ordering, side, triples[side][0]
                )
                for side in ("A", "B")
            }
            if audit.copy_refusal(laid["A"], laid["B"], MAX_COPY_ROWS):
                continue
            return _commit(bytes(master), int(ordinal), proposal, ordering, chosen, laid)
        refused["ordering"] += 1
    raise ConstructionExhausted(
        "no components pair at ordinal %d cleared the construction filter in %d proposals "
        "and %d orderings each; %s"
        % (
            int(ordinal),
            MAX_PROPOSALS,
            MAX_ORDERINGS,
            ", ".join("%s refused %d" % (name, n) for name, n in sorted(refused.items())),
        )
    )


def _commit(
    master: bytes,
    ordinal: int,
    proposal: int,
    ordering: int,
    chosen: Mapping[str, Mapping[str, Sequence[ClassId]]],
    laid: Mapping[str, Sequence[tuple[int, int, int]]],
) -> tuple[ComponentsTable, ComponentsTable]:
    """The two tables of an accepted proposal and ordering, with their printed order.

    The boards are drawn here rather than during the search, because the classes fix
    every acceptance condition and each embedding comes from its own stream: drawing them
    only for the pair that was accepted takes nothing from any other stream and leaves
    the result identical.
    """
    a_boards = _boards(master, ordinal, proposal, "A", chosen)
    a_controls = {
        len(board): board for board in a_boards[len(STRATA) * STRATUM_ROWS:]
    }
    b_boards = _boards(master, ordinal, proposal, "B", chosen, forbidden=a_controls)
    tables: list[ComponentsTable] = []
    for side, boards in (("A", a_boards), ("B", b_boards)):
        order = _ordered(master, ordinal, proposal, ordering, side, boards)
        expected = list(laid[side])
        if [counts_of(board) for board in order] != expected:
            raise RuntimeError(
                "the accepted ordering at components ordinal %d does not reproduce the "
                "answers it was accepted on, so the copy bound was read against a "
                "schedule that is not the one being committed" % ordinal
            )
        names = _identifiers(master, ordinal, proposal, ordering, side)
        tables.append(
            ComponentsTable(
                surface=SURFACE,
                rows=tuple(
                    ComponentsRow(row_id=name, cells=board)
                    for name, board in zip(names, order)
                ),
            )
        )
    return tables[0], tables[1]


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------

TASK_TEMPLATE = """PATTERN SCHEDULE
reference: {task_id}

File the number of islands in each row of this schedule. Each row is
a separate 5 by 5 board. Its occupied cells are listed as (row,column),
with both coordinates numbered 1 through 5. Row numbers increase
downward and column numbers increase to the right. All unlisted cells
are empty. Each list contains one through six different occupied cells,
printed in increasing row order and then increasing column order.

One contact rule applies throughout the schedule. The possible rules are:
  * Link cells that share a side: one coordinate differs by one and the
    other coordinate is equal.
  * Link cells that touch diagonally at a corner: both coordinates
    differ by one. Sharing a side does not count under this rule.
  * Link cells that share a side or touch diagonally at a corner.
The schedule does not specify which of these rules applies.

Only occupied cells can be linked. Contacts are undirected. Diagonal
contacts, when permitted, do not depend on whether the two intervening
side-neighbour cells are occupied. Boards do not wrap. Cells on the
border obey the same contact rule as all other cells. There are no
marked cells, connections between different boards, or connections
through empty cells.

An island is a maximal set of occupied cells connected by a sequence
of permitted contacts. An isolated occupied cell is one island. Count
each island once. A cycle does not create another island or another
merge. The starting cell, traversal order, and printed order cannot
change the answer. There is no selection among tied islands and no
fallback rule. Every board is valid and nonempty.

Use one of the exact values 1, 2, 3, 4, 5, 6. File text with one
identifier,value line per row, for example ABC12345678,2 when that
identifier occurs in the table. Do not file explanations or fences.
The score is the number of correct rows divided by 24, rounded to six
decimal places. Every row has equal weight. No partial credit is given
for an incorrect count.

For reading text filings, leading and trailing whitespace on each line
is removed and blank lines are discarded. Identifiers and values then
have runs of whitespace replaced by one space and surrounding whitespace
removed. Remaining characters outside printable ASCII are replaced by
question marks. Identifier matching and value comparison ignore ASCII
letter case. No numerical conversion is performed: 01, +1, 1.0, and one
are not the value 1.

If any line contains a comma, each line is split at its first comma.
The part before it is the identifier; everything after it is the value.
A line naming a known identifier without a comma files an empty value
in this mode. Unknown identifiers are ignored. For a repeated known
identifier the first submitted value wins, even if empty or invalid.
An omitted row receives no credit. Filing an empty value also receives
no credit, but is recorded separately from an omission.

As a fixed alternative, text containing no commas is read in table order
only when it contains exactly 24 nonblank lines, one value per line.
Any other comma-free text is not a filing for these rows. Unreadable
input, empty input, or input identifying no row receives zero credit.

{scope}

{body}
"""


# --------------------------------------------------------------------------
# the oracle template, and reading it back
# --------------------------------------------------------------------------

#: THE HEADING NAMES THE SCOPE BY THE SCHEDULES THE READER FILES.
#: A heading that scoped the rule to a word for the body of work it was drawn for was
#: read, and what it did was measured: of 119 readers given a correct statement of the
#: rule, 64 restated it correctly, wrote it down, and then filed the held-out task under
#: a different rule, because they had taken the scope word for a name printed on the
#: surface. So the scope is named by what the reader does, which is file this schedule
#: and then the next one, and the schedule after those is said in the same breath to be
#: scored under conventions of its own.
ORACLE_HEAD: tuple[str, ...] = (
    "HOUSE CONVENTIONS FOR THIS SCHEDULE AND THE NEXT ONE YOU FILE (they",
    "govern the schedule above and the next schedule you are given, whichever",
    "organisation it names; the schedule after that is scored under",
    "conventions of its own)",
    "",
)

#: What each option means, in words, with no phrase contained in another: a reader of
#: prose cannot tell two options apart when one of their phrases sits inside the other.
CONTACT_WORDS = {
    SIDE_CONTACTS: "a side is shared",
    CORNER_CONTACTS: "only one corner is shared",
    COMBINED_CONTACTS: "at least one boundary point is shared",
}

ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences={"contact_kernel": "Treat two occupied cells as adjacent exactly when {}."},
    phrases={"contact_kernel": CONTACT_WORDS},
)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class ComponentsGenerator:
    """The components genre, as the generator protocol wants it."""

    name = "components"
    genre = "geometric contact connectivity"
    SHAPE = SHAPE
    AXES = AXES
    SCORING: str = ROW_ADDITIVE_EQUAL_WEIGHT
    #: An island count is a token of a complete published vocabulary: the task prints the
    #: six legal values in their order, so the cheapest transfer a reader can make is a
    #: map between the two published orders, which is what the ordered-token profile
    #: prices. The stronger all-bijection bound is an added check rather than a change of
    #: profile: see `components_audit.check_bijection_copy`.
    COPY_PROFILE: str = ORDERED_TOKENS
    BLANK_TOKEN = BLANK_TOKEN
    UNFILED_TOKEN = UNFILED_TOKEN
    CONSTRUCTION_BOUNDS = CONSTRUCTION_BOUNDS

    # ----- the instance -----

    def surface_for(self, ordinal: int, label: str) -> str:
        return SURFACE

    def surface_templates(self) -> tuple[str, ...]:
        return (SURFACE,)

    def build_table(self, master: bytes, ordinal: int, label: str) -> ComponentsTable:
        """This side of the pair the construction filter accepted for this ordinal."""
        first, second = build_pair(bytes(master), int(ordinal))
        return first if label.upper() == "A" else second

    def table_record(self, table: ComponentsTable) -> dict[str, Any]:
        """Every field of a components table, as one canonical value.

        The surface names the presentation, and each row carries its printed identifier
        and its ordered coordinate pairs. The printed body is not a field: it is rendered
        on demand from exactly these rows, so hashing them commits to the bytes a reader
        is served without storing a second copy that could drift away from them.
        """
        return {
            "surface": table.surface,
            "rows": [
                {
                    "row_id": row.row_id,
                    "cells": [[cell[0], cell[1]] for cell in row.cells],
                }
                for row in table.rows
            ],
        }

    def build_envelope(self, master: bytes, ordinal: int) -> Envelope:
        """The registered envelope, with this instance's committed filler and neutrals.

        The coordinates name this generator, so two genres under one key never share a
        filler stream, and both are fixed before any filing exists.
        """
        neutral: dict[str, tuple[str, ...]] = {}
        for spec in SLOTS:
            neutral[spec.name] = tuple(
                streams.filler_stream(
                    master, FILLER_ALPHABET, spec.width,
                    "components", ordinal, "neutral", spec.name, row,
                )
                for row in range(ROWS)
            )
        return Envelope(
            size=ENVELOPE_SIZE,
            identifier_width=IDENTIFIER_WIDTH,
            observed_width=OBSERVED_WIDTH,
            slots=SLOTS,
            filler=streams.filler_stream(
                master, FILLER_ALPHABET, ENVELOPE_SIZE, "components", ordinal, "pad"
            ),
            column_titles=COLUMN_TITLES,
            neutral=neutral,
        )

    def key_for(
        self, table: ComponentsTable, convention: Mapping[str, str]
    ) -> tuple[str, ...]:
        return key_for(table, convention)

    def normalize_answer(self, value: str) -> str:
        # The same function the scorer compares with, so the two cannot drift apart.
        return shared_filing.fold(value)

    def row_identifiers(self, table: ComponentsTable) -> tuple[str, ...]:
        return tuple(row.row_id for row in table.rows)

    def row_classes(self, table: ComponentsTable) -> tuple[str, ...]:
        # One declared class. A reader of the schedule sees 24 occupied-cell patterns and
        # no printed distinction between them; the construction strata are private and
        # are never labelled, ordered or otherwise shown.
        return ("occupied_pattern",) * len(table.rows)

    def answer_ranks(self, table: ComponentsTable) -> tuple[str, ...]:
        """The complete published vocabulary, in the order the task prints it.

        Complete because a board of one to six cells has one to six islands and the task
        names all six; public because the task names them; ordered because they are
        counts. The copy screen needs exactly this to build the maps between two
        published orders that its bar is read against.
        """
        return ANSWER_RANKS

    def row_label(self, table: ComponentsTable) -> tuple[str, ...]:
        # Every row names the board it grades. Nothing on this receipt names an axis.
        return (ROW_LABEL,) * len(table.rows)

    # ----- reading the filing -----

    def parse_and_canonicalize(self, task: Task, raw: object) -> Filing:
        """One canonical value per printed row, by the shared registered rules.

        The reading is the one in `filing.py`: a line is a row identifier, a comma and a
        count; identifiers match without case after whitespace is collapsed; the first
        line for an identifier wins; a comma-free filing is read positionally only when
        it has exactly one line per printed row. A known identifier whose value is not a
        legal count stays filed and wrong, because refusing to read it would turn a bad
        answer into no answer.
        """
        return shared_filing.parse_rows(self.row_identifiers(task.table), raw)

    def score(self, task: Task, canonical: Filing) -> tuple[float, tuple[RowOutcome, ...]]:
        """The sealed scalar and what the filing did on every row.

        Equal weight per row, the denominator is the printed row count, rounded to six
        places. A row counts only if it was FILED: no correct count is ever the empty
        string, and the difference between a filed empty value and an omission is what
        the receipt prints, so the scorer keeps it too.
        """
        identifiers = self.row_identifiers(task.table)
        truth = task.key if task.key else ("",) * len(identifiers)
        return shared_filing.score_rows(
            identifiers, truth, canonical, self.normalize_answer
        )

    # ----- the task text -----

    def describe(self, task: PublicTask) -> str:
        """The mechanics, which schedules share a rule, and the schedule itself.

        It takes the PUBLIC task, so there is no argument here the drawn rule could
        arrive through, and the scope sentence is chosen by the sibling label and by
        nothing else: the same bytes go to every arm of a fork.
        """
        table: ComponentsTable = task.table
        return TASK_TEMPLATE.format(
            task_id=task.task_id,
            scope=scope_sentence(task.label),
            body=table.body,
        )

    # ----- the three cells -----

    def render_receipt(
        self, task: Task, canonical: Filing, truth: Sequence[str]
    ) -> ReceiptAST:
        """One verdict per board, on what the filing did.

        The rows are built by the shared grader from the scorer's own outcomes, so what a
        row says is not a choice this module makes. A failed row carries that row's own
        count and nothing else: not the number of cells, not a rule name, not another
        row's answer. A passed row carries nothing, because the observed column already
        prints the value that passed.
        """
        graded = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text=task.text, key=tuple(truth),
        )
        _, outcomes = self.score(graded, canonical)
        return graded_receipt(task.task_id, outcomes, BLANK_TOKEN, UNFILED_TOKEN)

    def render_placebo(
        self, task: PublicTask, canonical: Filing, envelope: Envelope
    ) -> ReceiptAST:
        """The inert cell: congruent with the graded one outside the registered slots.

        It takes the PUBLIC task and the envelope, so there is no argument here through
        which the hidden rule could reach it. The keyless task supplies the row identities
        and the filed values and nothing else, and both slots carry this instance's
        committed neutral tokens on every row.
        """
        blind = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text="", key=(),
        )
        _, outcomes = self.score(blind, canonical)
        return placebo_receipt(task.task_id, outcomes, envelope, BLANK_TOKEN, UNFILED_TOKEN)

    #: The declared phrase table. Rendering and reading both go through it.
    ORACLE = ORACLE_TEMPLATE

    def render_oracle(
        self, task_id: str, convention: Mapping[str, str], row_count: int = 0
    ) -> ReceiptAST:
        """The drawn contact rule, from the declared phrases, with no rows to align."""
        return render_oracle_cell(ORACLE_TEMPLATE, task_id, convention, row_count)

    def parse_oracle(self, ast: ReceiptAST) -> dict[str, str]:
        return parse_oracle_cell(ORACLE_TEMPLATE, ast)


GENERATOR = ComponentsGenerator()


__all__ = [
    "ALL_CONVENTIONS",
    "ANSWER_RANKS",
    "AXES",
    "BLANK_TOKEN",
    "BOARD",
    "COMBINED_CONTACTS",
    "CONSTRUCTION_BOUNDS",
    "CONTACT_WORDS",
    "CONTROL_SIZES",
    "CORNER_CONTACTS",
    "ENVELOPE_SIZE",
    "FILLER_ALPHABET",
    "GENERATOR",
    "MAX_ANSWER_FREQUENCY",
    "MAX_CELLS",
    "MAX_COPY_ROWS",
    "MAX_ORDERINGS",
    "MAX_PROPOSALS",
    "MIN_CELLS",
    "ORACLE_HEAD",
    "ORACLE_TEMPLATE",
    "ROWS",
    "SHAPE",
    "SIDE",
    "SIDE_CONTACTS",
    "SLOTS",
    "STRATA",
    "STRATUM_OPTION",
    "STRATUM_ROWS",
    "SURFACE",
    "UNFILED_TOKEN",
    "Board",
    "Catalogue",
    "ClassId",
    "ComponentsGenerator",
    "ComponentsRow",
    "ComponentsTable",
    "build_pair",
    "catalogue",
    "check_board",
    "class_identifier",
    "components_of",
    "counts_of",
    "has_side_cycle",
    "key_for",
    "linked",
    "print_cells",
    "side_structure",
    "stratum_of",
]
