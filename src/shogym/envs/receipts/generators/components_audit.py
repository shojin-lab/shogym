"""A second implementation of the island count, and the checks the pair has to pass.

WHY A SECOND IMPLEMENTATION. `judge_cells` compares what the renderer printed with what
the scorer computed, so it catches a renderer that disagrees with the scorer and cannot
catch a scorer that is wrong. A connectivity routine that subtracted contacts from cells,
or stopped at direct contacts instead of following paths, or let a corner-only rule
inherit side contacts, would be carried identically into the key, the receipt, the
placebo and every gate, and every named check would pass on it. The serializer also fits
an overwide field by truncating it rather than refusing it. Both are mistakes an author
makes and cannot see, and the only thing that finds them is another implementation that
was not written from the first.

So this module never imports the production contact predicate or its union-find. It
builds its own coordinate neighbourhoods from the option's own name, keeps only the
neighbours that are on the board and occupied, and floods from each unvisited cell. It
shares the board size, the option identifiers, the registered bounds and the dataclasses,
which are declarations rather than computation: `side_contacts` names the four steps that
share a side and `corner_contacts` the four that share only a corner, so an edited
production table is a disagreement here rather than a change both sides make together.

THE CONSTRUCTION PREDICATES ARE PURE AND CONVENTION FREE, AND THEY ARE NOT THE CHECKS.
`frequency_refusal` and `copy_refusal` take answer vectors and return a reason or the
empty string. They never draw, never see the live convention, and quantify over all three
rules. They are separate from the admission dispatch below because the constructor calls
them while it is building the pair the checks will later be run on, and a predicate that
reached back into admission would recurse through `draw` into itself.
"""

from __future__ import annotations

import itertools
from collections import Counter
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from shogym.envs.receipts.admission import (
    REGISTERED_MAX_FLIP_SCORE,
    REGISTERED_MIN_LEVERAGE,
)
from shogym.envs.receipts.checks import (
    CheckResult,
    axis_leverage,
    check_exercise,
    copy_scores,
)
from shogym.envs.receipts.generators.components import (
    ANSWER_RANKS,
    AXES,
    BLANK_TOKEN,
    CONTROL_SIZES,
    MAX_ANSWER_FREQUENCY,
    MAX_CELLS,
    MAX_COPY_ROWS,
    MIN_CELLS,
    ROWS,
    SIDE,
    STRATUM_ROWS,
    UNFILED_TOKEN,
    ComponentsTable,
)
from shogym.envs.receipts.filing import fold, scope_sentence
from shogym.envs.receipts.observe import observe
from shogym.envs.receipts.protocol import Instance, Task
from shogym.envs.receipts.receipt_ast import (
    GAP,
    ORDINAL_WIDTH,
    frozen_envelope,
    row_lines,
    serialize,
)
from shogym.envs.receipts.render import FAIL_TOKEN, PASS_TOKEN, feedback_for
from shogym.receipts import gate

#: The option order this module reads answer vectors in. Taken from the axis rather than
#: written again, because a vector's positions have to mean the same thing on both sides
#: of every comparison here.
OPTIONS: tuple[str, ...] = tuple(AXES[0].options)

# ----- the registered quantities this module holds a pair to ---------------------
#
# The room arithmetic the recipe produces, stated as numbers rather than derived from the
# thing being checked. Six rows in each of three strata make every pair of rules differ
# on exactly twelve rows, which fixes all of it: the whole receipt resolves three blocks,
# no informative row is evident, the ceiling is one, the lookup floor is five sixths and
# the headroom is one sixth. A pair that produces anything else was not built by the
# recipe this genre is admitted under.

EXPECTED_MOVEMENT = 12
EXPECTED_BLOCKS = 3
EXPECTED_CEILING = 1.0
EXPECTED_FLOOR = 5.0 / 6.0
EXPECTED_HEADROOM = 1.0 / 6.0
#: What the specified case-lookup procedure earns with the public modal fallback: each
#: row's modal count over the three rules is right on the eighteen informative rows two
#: times in three and on the six controls always.
EXPECTED_LOOKUP_GRADE = 0.75
#: How close a computed quantity has to sit to the one above. These are exact rationals
#: in the arithmetic and floating point is what carries them.
TOLERANCE = 1e-9

#: The eight symmetries of a square board, as the maps a whole board is carried by.
SQUARE_SYMMETRIES = ("identity", "quarter", "half", "three_quarter",
                     "flip_rows", "flip_columns", "transpose", "anti_transpose")

# ----- the independent neighbourhood ---------------------------------------------

#: The coordinate steps that share a side, and the ones that share only a corner. Written
#: as steps rather than as a distance test, which is the other route to the same relation.
SIDE_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNER_STEPS = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def steps(option: str) -> tuple[tuple[int, int], ...]:
    """Which coordinate steps one option's own name says are a contact."""
    head = str(option).split("_")[0]
    if head == "side":
        return SIDE_STEPS
    if head == "corner":
        return CORNER_STEPS
    if head == "combined":
        return SIDE_STEPS + CORNER_STEPS
    raise ValueError(f"no contact rule named {option!r}")


def on_board(cell: tuple[int, int]) -> bool:
    """Whether a coordinate is a cell of the board. Nothing wraps: it is off or it is on."""
    return 1 <= cell[0] <= SIDE and 1 <= cell[1] <= SIDE


def board_refusal(cells: Sequence[tuple[int, int]]) -> str:
    """Why this is not a board, or the empty string. Independent of the production guard."""
    if len(cells) < MIN_CELLS:
        return "a board with no occupied cell has no island count"
    if len(cells) > MAX_CELLS:
        return f"a board carries at most {MAX_CELLS} occupied cells and this one carries {len(cells)}"
    for cell in cells:
        pair = tuple(cell)
        if len(pair) != 2 or not on_board((pair[0], pair[1])):
            return f"{pair!r} is off a {SIDE} by {SIDE} board"
    if len({tuple(cell) for cell in cells}) != len(cells):
        return "a board lists an occupied cell twice"
    return ""


def neighbourhood(
    cells: Sequence[tuple[int, int]], option: str
) -> dict[tuple[int, int], set[tuple[int, int]]]:
    """Each occupied cell's occupied neighbours under one rule, built from the steps.

    A step off the board reaches nothing, which is what "boards do not wrap" means, and a
    step onto an empty cell reaches nothing, which is what "only occupied cells can be
    linked" means. Nothing here consults the intervening cells of a diagonal.
    """
    occupied = {(cell[0], cell[1]) for cell in cells}
    moves = steps(option)
    out: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for cell in occupied:
        reached = set()
        for down, across in moves:
            other = (cell[0] + down, cell[1] + across)
            if on_board(other) and other in occupied:
                reached.add(other)
        out[cell] = reached
    return out


def flood_components(cells: Sequence[tuple[int, int]], option: str) -> int:
    """Islands by flood fill from each unvisited occupied cell.

    A different route to the same number: the production evaluator unions the endpoints of
    every permitted contact and counts the sets that remain, and this walks paths. Neither
    counts contacts, which is what a board with a cycle would punish.
    """
    refusal = board_refusal(cells)
    if refusal:
        raise ValueError(refusal)
    neighbours = neighbourhood(cells, option)
    seen: set[tuple[int, int]] = set()
    islands = 0
    for start in sorted(neighbours):
        if start in seen:
            continue
        islands += 1
        seen.add(start)
        stack = [start]
        while stack:
            here = stack.pop()
            for other in neighbours[here]:
                if other not in seen:
                    seen.add(other)
                    stack.append(other)
    return islands


def flood_counts(cells: Sequence[tuple[int, int]]) -> tuple[int, int, int]:
    """One board's three island counts, in the axis's option order."""
    first, second, third = (flood_components(cells, option) for option in OPTIONS)
    return first, second, third


def pair_colour(one: tuple[int, int], other: tuple[int, int]) -> int:
    """1 for a shared side, 2 for a shared corner only, 0 for neither, from the steps."""
    step = (other[0] - one[0], other[1] - one[1])
    if step in SIDE_STEPS:
        return 1
    if step in CORNER_STEPS:
        return 2
    return 0


@lru_cache(maxsize=8192)
def graph_signature(cells: tuple[tuple[int, int], ...]) -> tuple[int, ...]:
    """The board's coloured contact graph, minimized over EVERY vertex order.

    Brute force over all orders rather than over the orders a degree argument leaves, so
    this establishes the production classifier's grouping rather than assuming it: two
    boards share a signature exactly when some relabelling carries one graph onto the
    other with all three colours preserved. At most six vertices, so at most 720 orders.
    """
    size = len(cells)
    return (size,) + min(
        tuple(
            pair_colour(cells[order[i]], cells[order[j]])
            for i in range(size)
            for j in range(i + 1, size)
        )
        for order in itertools.permutations(range(size))
    )


def carried(cells: Sequence[tuple[int, int]], symmetry: str) -> tuple[tuple[int, int], ...]:
    """One board under one of the eight symmetries of the square, sorted as printed."""
    top = SIDE + 1
    moved: list[tuple[int, int]] = []
    for row, column in cells:
        if symmetry == "identity":
            made = (row, column)
        elif symmetry == "quarter":
            made = (column, top - row)
        elif symmetry == "half":
            made = (top - row, top - column)
        elif symmetry == "three_quarter":
            made = (top - column, row)
        elif symmetry == "flip_rows":
            made = (top - row, column)
        elif symmetry == "flip_columns":
            made = (row, top - column)
        elif symmetry == "transpose":
            made = (column, row)
        elif symmetry == "anti_transpose":
            made = (top - column, top - row)
        else:
            raise ValueError(f"no square symmetry named {symmetry!r}")
        moved.append(made)
    return tuple(sorted(moved))


# --------------------------------------------------------------------------
# the construction predicates: pure, and about the pair alone
# --------------------------------------------------------------------------


def frequency_refusal(triples: Sequence[tuple[int, int, int]]) -> str:
    """Why one side's answers are too concentrated, or the empty string.

    Under each of the three rules separately, because the cap is what keeps any one count
    from being a filing worth making on its own, and a side can be flat under one rule and
    piled up under another.
    """
    for position, option in enumerate(OPTIONS):
        counted = Counter(triple[position] for triple in triples)
        worst, seen = max(counted.items(), key=lambda item: (item[1], item[0]))
        if seen > MAX_ANSWER_FREQUENCY:
            return (
                "under %s the answer %d occurs %d times, over the registered %d"
                % (option, worst, seen, MAX_ANSWER_FREQUENCY)
            )
    return ""


def row_moves(values: Sequence[int]) -> list[list[int]]:
    """The dihedral rearrangements of one filing: every rotation, and of its reversal."""
    out: list[list[int]] = []
    for base in (list(values), list(reversed(values))):
        for shift in range(len(base)):
            out.append(base[shift:] + base[:shift])
    return out


#: Every bijection of the six published counts, as index permutations.
BIJECTIONS = np.array(list(itertools.permutations(range(len(ANSWER_RANKS)))), dtype=np.int64)


def contingency(source: Sequence[int], target: Sequence[int]) -> np.ndarray:
    """One contingency matrix per row move: how often source value a meets target value b.

    The rows are counts from one to six, so the matrix is six by six and its entries sum
    to the row count. A maximum-weight assignment on it is the best any bijection of the
    six values can do for that row move.
    """
    moves = np.array(row_moves(list(source)), dtype=np.int64) - 1
    into = np.asarray(list(target), dtype=np.int64) - 1
    matrices = np.zeros((moves.shape[0], len(ANSWER_RANKS), len(ANSWER_RANKS)), dtype=np.int64)
    for position in range(moves.shape[0]):
        np.add.at(matrices[position], (moves[position], into), 1)
    return matrices


def _assignment_bound(matrices: np.ndarray) -> np.ndarray:
    """Per row move, an upper bound on its assignment: the sum of its row maxima.

    A perfect matching takes one entry from each row, so it cannot beat this. When the
    bound is already inside the registered ceiling the exact maximum is too, which is what
    keeps the construction search from enumerating 720 bijections for every candidate.
    """
    return matrices.max(axis=2).sum(axis=1)


def _exact_maximum(matrices: np.ndarray) -> tuple[int, int, tuple[int, ...]]:
    """The exact maximum over row moves and bijections, and a transformation reaching it.

    Enumerating all 720 bijections is exact, and the row-maxima bound orders the search so
    that a row move that cannot beat what is already held is never enumerated.
    """
    bound = _assignment_bound(matrices)
    best, best_move, best_map = -1, 0, tuple(range(len(ANSWER_RANKS)))
    rows = np.arange(len(ANSWER_RANKS))
    for move in np.argsort(-bound):
        if int(bound[move]) <= best:
            break
        scored = matrices[int(move)][rows[None, :], BIJECTIONS].sum(axis=1)
        top = int(scored.argmax())
        if int(scored[top]) > best:
            best = int(scored[top])
            best_move = int(move)
            best_map = tuple(int(value) for value in BIJECTIONS[top])
    return best, best_move, best_map


def bijection_maximum(
    source: Sequence[int], target: Sequence[int]
) -> tuple[int, int, tuple[int, ...]]:
    """Correct rows at best, the row move that reached it, and the bijection it used."""
    return _exact_maximum(contingency(source, target))


def copy_refusal(
    a_triples: Sequence[tuple[int, int, int]],
    b_triples: Sequence[tuple[int, int, int]],
    ceiling: int = MAX_COPY_ROWS,
) -> str:
    """Why one proposed ordering lets A's answers be copied onto B, or the empty string.

    Under every rule, because the copy an agent can make does not depend on which rule was
    drawn, and over the whole closed transformation family: every rotation of A's filing
    and of its reversal, composed with every bijection of the six counts.
    """
    for position, option in enumerate(OPTIONS):
        source = [triple[position] for triple in a_triples]
        target = [triple[position] for triple in b_triples]
        matrices = contingency(source, target)
        if int(_assignment_bound(matrices).max()) <= ceiling:
            continue
        best, _, _ = _exact_maximum(matrices)
        if best > ceiling:
            return (
                "under %s a copy of A's answers reaches %d of %d rows on B, over the "
                "registered %d" % (option, best, len(target), ceiling)
            )
    return ""


# --------------------------------------------------------------------------
# reading a table the way the audit reads it
# --------------------------------------------------------------------------


def boards(table: ComponentsTable) -> tuple[tuple[tuple[int, int], ...], ...]:
    return tuple(tuple(row.cells) for row in table.rows)


def classify(cells: Sequence[tuple[int, int]]) -> str:
    """What this board is, from its own counts: a stratum letter, `control`, or nothing.

    Derived, never read off a label. The printed schedule carries no stratum, no ordering
    by stratum and no mark, so the audit recovers the recipe from the boards themselves or
    it has not checked the recipe at all.
    """
    side, corner, combined = flood_counts(cells)
    if side == corner == combined:
        return "control" if side == len(cells) else ""
    agreements = (side == corner) + (side == combined) + (corner == combined)
    if agreements != 1:
        return ""
    if corner == combined:
        return "S"
    if side == combined:
        return "C"
    return "U"


def modal_filing(table: ComponentsTable) -> tuple[str, ...]:
    """Each row's modal count over the three rules, which is the public no-receipt action.

    Ties cannot arise here: every board either answers alike under all three rules or
    singles out exactly one, so the mode is the answer the other two share.
    """
    out: list[str] = []
    for cells in boards(table):
        seen = Counter(flood_counts(cells))
        out.append(str(seen.most_common(1)[0][0]))
    return tuple(out)


def _retasked(task: Task, convention: Mapping[str, str], generator) -> Task:
    return Task(
        label=task.label,
        task_id=task.task_id,
        surface=task.surface,
        table=task.table,
        text=task.text,
        key=tuple(generator.key_for(task.table, convention)),
        mask=task.mask,
    )


def retasked_instance(generator, instance: Instance, convention: Mapping[str, str]) -> Instance:
    """The same pair as if this rule had been drawn, for a counterfactual reference.

    It is a reference filing's instance and never a bank member: nothing here fixes it,
    hashes it or offers it to admission, and `check_fixation` deliberately stays on the
    actual draw, where comparing a rebuild against the original means something.
    """
    frozen = MappingProxyType(dict(convention))
    return Instance(
        generator=instance.generator,
        genre=instance.genre,
        ordinal=instance.ordinal,
        convention=frozen,
        a=_retasked(instance.a, frozen, generator),
        b=_retasked(instance.b, frozen, generator),
        envelope=instance.envelope,
    )


def _printed_columns(generator, instance: Instance, side: str) -> str:
    """What the serialized graded cell says the identifiers and corrections are.

    The serializer fits an overwide field by truncating it rather than refusing it, so a
    value longer than its slot becomes a shorter legal-looking one and every other check
    still passes. This reads the columns back out of the bytes an agent would receive.
    """
    task = instance.side(side)
    envelope = frozen_envelope(instance.envelope)
    raw = "\n".join(
        "%s," % identifier for identifier in generator.row_identifiers(task.table)
    )
    canonical = generator.parse_and_canonicalize(task, raw)
    ast = generator.render_receipt(
        task, canonical, task.key, feedback_for(generator, task, envelope)
    )
    payload = serialize(ast, envelope)
    identifier_start = len(GAP) + ORDINAL_WIDTH + len(GAP)
    identifier_end = identifier_start + envelope.identifier_width
    low, high = envelope.slot_span("correction")
    for line, row in zip(row_lines(payload, ast, envelope), ast.rows):
        shown = line[identifier_start:identifier_end].decode("ascii").strip()
        correction = line[low:high].decode("ascii").strip()
        wanted = {slot.name: slot.value for slot in row.slots}["correction"]
        if shown != row.identifier:
            return (
                f"side {side.upper()} prints the identifier {shown!r} where the row names "
                f"{row.identifier!r}"
            )
        if correction != wanted:
            return (
                f"side {side.upper()} prints the correction {correction!r} where the row's "
                f"own answer is {wanted!r}"
            )
    return ""


# --------------------------------------------------------------------------
# the six checks admission adds for this genre
# --------------------------------------------------------------------------


def check_semantics(generator, instance: Instance) -> CheckResult:
    """The production island count, recomputed by a flood fill on every row and every rule.

    It fails if the two implementations disagree on any row under any rule, if a board is
    empty, out of range, repeated or larger than six cells, if a key value is outside the
    published vocabulary, or if a printed identifier or correction changes under serialize
    and read back.
    """
    reports: list[str] = []
    legal = set(ANSWER_RANKS)
    for side in ("a", "b"):
        table = instance.side(side).table
        for position, cells in enumerate(boards(table)):
            refusal = board_refusal(cells)
            if refusal:
                return CheckResult(
                    "components_semantics", False,
                    f"side {side.upper()} row {position + 1}: {refusal}",
                )
        for option in OPTIONS:
            produced = tuple(generator.key_for(table, {"contact_kernel": option}))
            expected = tuple(str(flood_components(cells, option)) for cells in boards(table))
            if produced != expected:
                wrong = next(
                    n for n, (x, y) in enumerate(zip(produced, expected)) if x != y
                )
                return CheckResult(
                    "components_semantics", False,
                    f"side {side.upper()} row {wrong + 1} ({boards(table)[wrong]}) is scored "
                    f"{produced[wrong]!r} under {option} and an independent flood fill makes "
                    f"it {expected[wrong]!r}",
                )
            outside = sorted(set(produced) - legal)
            if outside:
                return CheckResult(
                    "components_semantics", False,
                    f"side {side.upper()} answers {outside} under {option}, outside the "
                    f"published vocabulary {list(ANSWER_RANKS)}",
                )
        reports.append(
            "%s %d boards, largest %d cells"
            % (side.upper(), len(table.rows), max(len(cells) for cells in boards(table)))
        )
    for side in ("a", "b"):
        wrong = _printed_columns(generator, instance, side)
        if wrong:
            return CheckResult("components_semantics", False, wrong)
    return CheckResult(
        "components_semantics", True,
        "an independent flood fill agrees with the scorer on all %d rows of both sides "
        "under all %d rules; %s" % (ROWS, len(OPTIONS), "; ".join(reports)),
    )


def check_shape(generator, instance: Instance) -> CheckResult:
    """The recipe, recovered from the boards rather than read off a label.

    It fails if either side is not six rows in each of the three informative strata and six
    separated controls, if a count is missing from a side's answers under some rule, if a
    count occurs more than eight times, if a side holds no side-contact cycle, if an
    informative graph repeats within or across the siblings, or if an exact board repeats
    across them.
    """
    seen: dict[str, dict[str, int]] = {}
    signatures: dict[str, list[tuple[int, ...]]] = {}
    for side in ("a", "b"):
        table = instance.side(side).table
        kinds = Counter(classify(cells) for cells in boards(table))
        if any(letter not in kinds or kinds[letter] != STRATUM_ROWS for letter in ("S", "C", "U")):
            return CheckResult(
                "components_shape", False,
                "side %s holds %s and the recipe is %d rows in each of S, C and U"
                % (side.upper(), dict(sorted(kinds.items())), STRATUM_ROWS),
            )
        if kinds.get("control", 0) != len(CONTROL_SIZES):
            return CheckResult(
                "components_shape", False,
                "side %s holds %d separated controls and the recipe is %d"
                % (side.upper(), kinds.get("control", 0), len(CONTROL_SIZES)),
            )
        control_sizes = sorted(
            len(cells) for cells in boards(table) if classify(cells) == "control"
        )
        if tuple(control_sizes) != CONTROL_SIZES:
            return CheckResult(
                "components_shape", False,
                "side %s holds controls of sizes %s and the recipe is one of each of %s"
                % (side.upper(), control_sizes, list(CONTROL_SIZES)),
            )
        triples = [flood_counts(cells) for cells in boards(table)]
        refusal = frequency_refusal(triples)
        if refusal:
            return CheckResult(
                "components_shape", False, f"side {side.upper()}: {refusal}"
            )
        for position, option in enumerate(OPTIONS):
            realized = {str(triple[position]) for triple in triples}
            missing = sorted(set(ANSWER_RANKS) - realized)
            if missing:
                return CheckResult(
                    "components_shape", False,
                    f"side {side.upper()} never answers {missing} under {option}, so the "
                    "published vocabulary is not realized",
                )
        cycles = [
            cells for cells in boards(table)
            if classify(cells) in ("S", "C", "U") and _has_side_cycle(cells)
        ]
        if not cycles:
            return CheckResult(
                "components_shape", False,
                f"side {side.upper()} holds no informative board whose side contacts close "
                "a cycle, so nothing on it refuses a count read off a contact tally",
            )
        informative = [
            graph_signature(cells) for cells in boards(table)
            if classify(cells) in ("S", "C", "U")
        ]
        if len(set(informative)) != len(informative):
            return CheckResult(
                "components_shape", False,
                f"side {side.upper()} repeats an informative contact graph",
            )
        signatures[side] = informative
        seen[side] = {"cycles": len(cycles)}
    shared = set(signatures["a"]) & set(signatures["b"])
    if shared:
        return CheckResult(
            "components_shape", False,
            "%d informative contact graphs occur on both siblings" % len(shared),
        )
    repeats = set(boards(instance.a.table)) & set(boards(instance.b.table))
    if repeats:
        return CheckResult(
            "components_shape", False,
            "%d exact boards occur on both siblings" % len(repeats),
        )
    return CheckResult(
        "components_shape", True,
        "both sides hold %d rows in each stratum and %d separated controls, realize every "
        "count under every rule, share no informative graph and no board; side cycles A %d, "
        "B %d" % (STRATUM_ROWS, len(CONTROL_SIZES), seen["a"]["cycles"], seen["b"]["cycles"]),
    )


def _has_side_cycle(cells: Sequence[tuple[int, int]]) -> bool:
    """Whether this board's side contacts close a cycle, counted independently."""
    contacts = sum(
        1
        for i in range(len(cells))
        for j in range(i + 1, len(cells))
        if pair_colour(cells[i], cells[j]) == 1
    )
    return contacts > len(cells) - flood_components(cells, OPTIONS[0])


def check_support(generator, instance: Instance) -> CheckResult:
    """The room arithmetic, rerun at every rule rather than only at the one drawn.

    It fails if under any reference rule, on either sibling, the three keys are not
    distinct after normalization, the three graded receipts are not distinct as bytes, a
    pair of keys moves other than twelve rows, the receipt resolves other than three
    blocks, any row is evident, the ceiling is not one, the lookup floor is not five
    sixths, the headroom is not one sixth, an axis is not exercised, a one-rule-wrong
    filing beats its bar, the axis carries too little leverage, or the registered copy
    calculation reports a maximum over its bar.

    Nothing here calls admission or fixation. A counterfactual instance is a reference
    filing and not a candidate, so fixing it would compare a rebuild of the actual draw
    against a rule that was not drawn.
    """
    lines: list[str] = []
    for reference in OPTIONS:
        alternative = retasked_instance(generator, instance, {"contact_kernel": reference})
        for side in ("a", "b"):
            table = alternative.side(side).table
            keys = {
                option: tuple(
                    generator.normalize_answer(value)
                    for value in generator.key_for(table, {"contact_kernel": option})
                )
                for option in OPTIONS
            }
            if len(set(keys.values())) != len(OPTIONS):
                return CheckResult(
                    "components_support", False,
                    f"under reference {reference}, side {side.upper()} normalizes two rules "
                    "to one key, so two options are one action",
                )
            task = alternative.side(side)
            filed = "\n".join(
                "%s,%s" % (identifier, value)
                for identifier, value in zip(generator.row_identifiers(table), task.key)
            )
            canonical = generator.parse_and_canonicalize(task, filed)
            envelope = frozen_envelope(alternative.envelope)
            rendered = {
                option: serialize(
                    generator.render_receipt(
                        task,
                        canonical,
                        generator.key_for(table, {"contact_kernel": option}),
                        feedback_for(generator, task, envelope),
                    ),
                    envelope,
                )
                for option in OPTIONS
            }
            if len(set(rendered.values())) != len(OPTIONS):
                return CheckResult(
                    "components_support", False,
                    f"under reference {reference}, side {side.upper()} serializes two rules "
                    "to one receipt, so the bytes alias what the keys separate",
                )
            for one, other in itertools.combinations(OPTIONS, 2):
                moved = sum(1 for x, y in zip(keys[one], keys[other]) if x != y)
                if moved != EXPECTED_MOVEMENT:
                    return CheckResult(
                        "components_support", False,
                        "under reference %s, side %s moves %d rows between %s and %s and the "
                        "recipe moves %d"
                        % (reference, side.upper(), moved, one, other, EXPECTED_MOVEMENT),
                    )
        result = gate(observe(generator, alternative, "a"))
        blocks = result.blocks.get(AXES[0].name, 0)
        if blocks != EXPECTED_BLOCKS:
            return CheckResult(
                "components_support", False,
                "under reference %s the receipt resolves %d blocks and the recipe resolves %d"
                % (reference, blocks, EXPECTED_BLOCKS),
            )
        if result.n_evident:
            return CheckResult(
                "components_support", False,
                "under reference %s, %d rows are evident: a row with two output classes on a "
                "three-option axis hands nothing over" % (reference, result.n_evident),
            )
        for name, got, want in (
            ("ceiling", result.ceiling, EXPECTED_CEILING),
            ("floor", result.floor, EXPECTED_FLOOR),
            ("headroom", result.ceiling - result.floor, EXPECTED_HEADROOM),
        ):
            if abs(got - want) > TOLERANCE:
                return CheckResult(
                    "components_support", False,
                    "under reference %s the %s is %.6f and the recipe gives %.6f"
                    % (reference, name, got, want),
                )
        exercised = check_exercise(generator, alternative)
        if not exercised.passed:
            return CheckResult(
                "components_support", False,
                f"under reference {reference}, {exercised.detail}",
            )
        scores = copy_scores(generator, alternative)
        leverage = axis_leverage(generator, alternative)
        registered = max(
            scores[name] for name in ("identity", "permutation", "relabel", "closure")
        )
        if registered > MAX_COPY_ROWS / float(ROWS) + TOLERANCE:
            return CheckResult(
                "components_support", False,
                "under reference %s the registered copy family earns %.6f on B, over %.6f"
                % (reference, registered, MAX_COPY_ROWS / float(ROWS)),
            )
        # THE TWO NUMBERS BELOW WERE PRINTED AND NOT READ. The shared `copy` check reads
        # them at the rule that was drawn, and this walks the other two, which is the
        # whole reason for rerunning the arithmetic at every reference: a pair whose
        # counterfactual rule costs nothing to get wrong is a pair the link cannot be
        # scored on whenever that rule is the one drawn, and every ordinal draws its own.
        if scores["option_flip"] > REGISTERED_MAX_FLIP_SCORE + TOLERANCE:
            return CheckResult(
                "components_support", False,
                "under reference %s a filing made under one of the other two rules still "
                "earns %.6f on B, over the registered %.6f"
                % (reference, scores["option_flip"], REGISTERED_MAX_FLIP_SCORE),
            )
        weak = sorted(
            name for name, value in leverage.items()
            if value < REGISTERED_MIN_LEVERAGE - TOLERANCE
        )
        if weak:
            return CheckResult(
                "components_support", False,
                "under reference %s B pays %.6f for %s and the registered leverage is "
                "%.6f" % (reference, leverage[weak[0]], ", ".join(weak),
                          REGISTERED_MIN_LEVERAGE),
            )
        lines.append(
            "%s blocks %d, evident %d, ceiling %.6f, floor %.6f, H %.6f, copy %.6f, "
            "one rule wrong %.6f, leverage %.6f"
            % (
                reference, blocks, result.n_evident, result.ceiling, result.floor,
                result.ceiling - result.floor, registered, scores["option_flip"],
                leverage[AXES[0].name],
            )
        )
    return CheckResult("components_support", True, "; ".join(lines))


def check_bijection_copy(generator, instance: Instance) -> CheckResult:
    """What a copy of one sibling's answers earns on the other, over every value bijection.

    It fails if, under any rule and in either direction, some rotation of one filing or of
    its reversal composed with some bijection of the six counts reaches more than twelve of
    twenty four rows. The registered ordered-token family is six cyclic maps; this is all
    720, so the bound holds whether or not a reader treats the counts as ordered.
    """
    lines: list[str] = []
    for option in OPTIONS:
        convention = {"contact_kernel": option}
        keys = {
            side: [
                int(value)
                for value in generator.key_for(instance.side(side).table, convention)
            ]
            for side in ("a", "b")
        }
        for name, source, target in (
            ("A to B", keys["a"], keys["b"]),
            ("B to A", keys["b"], keys["a"]),
        ):
            best, move, mapping = bijection_maximum(source, target)
            if best > MAX_COPY_ROWS:
                return CheckResult(
                    "components_bijection_copy", False,
                    "under %s, %s reaches %d of %d rows with row move %d and the value map "
                    "%s, over the registered %d"
                    % (option, name, best, len(target), move,
                       _map_text(mapping), MAX_COPY_ROWS),
                )
            lines.append(
                "%s %s %d/%d at row move %d under %s"
                % (option, name, best, len(target), move, _map_text(mapping))
            )
    return CheckResult("components_bijection_copy", True, "; ".join(lines))


def _map_text(mapping: Sequence[int]) -> str:
    """One value bijection as the substitution it is."""
    return " ".join(
        "%s>%s" % (ANSWER_RANKS[position], ANSWER_RANKS[image])
        for position, image in enumerate(mapping)
    )


def check_analogy(generator, instance: Instance) -> CheckResult:
    """What recognizing A's own boards again on B would earn, and what it would not.

    It fails if any informative board on A shares a colour-preserving contact graph with an
    informative board on B, which is what would let a reader carry a correction across
    without inferring the rule at all, or if the public modal fallback that remains does
    not score exactly three quarters on the sibling. The invariant controls DO recur by
    construction and are counted separately: they are reusable counting cases and they are
    charged in that three quarters.
    """
    informative = {
        side: {
            graph_signature(cells): cells
            for cells in boards(instance.side(side).table)
            if classify(cells) in ("S", "C", "U")
        }
        for side in ("a", "b")
    }
    shared = sorted(set(informative["a"]) & set(informative["b"]))
    if shared:
        return CheckResult(
            "components_analogy", False,
            "%d informative contact graphs occur on both siblings, so a correction on A "
            "transfers to B by recognition; the first is A %s and B %s"
            % (len(shared), informative["a"][shared[0]], informative["b"][shared[0]]),
        )
    controls = {
        side: Counter(
            graph_signature(cells)
            for cells in boards(instance.side(side).table)
            if classify(cells) == "control"
        )
        for side in ("a", "b")
    }
    lines: list[str] = []
    for side in ("a", "b"):
        other = instance.side(side)
        filed = "\n".join(
            "%s,%s" % (identifier, value)
            for identifier, value in zip(
                generator.row_identifiers(other.table), modal_filing(other.table)
            )
        )
        canonical = generator.parse_and_canonicalize(other, filed)
        grade = generator.score(other, canonical)[0]
        if abs(grade - EXPECTED_LOOKUP_GRADE) > TOLERANCE:
            return CheckResult(
                "components_analogy", False,
                "the public modal fallback scores %.6f on side %s and the recipe gives %.6f"
                % (grade, side.upper(), EXPECTED_LOOKUP_GRADE),
            )
        lines.append("%s modal fallback %.6f" % (side.upper(), grade))
    return CheckResult(
        "components_analogy", True,
        "no informative contact graph is shared, %d invariant control graphs recur, %s"
        % (len(set(controls["a"]) & set(controls["b"])), "; ".join(lines)),
    )


def check_public_contract(generator, instance: Instance) -> CheckResult:
    """What the public surface promises, held against what it prints.

    It fails if a committed neutral token normalizes onto any legal answer, either verdict
    or either status token; if a task text does not carry its own registered scope sentence
    or carries the other sibling's; if a row identifier does not survive the shared
    normalization or collides with another after folding; if a printed coordinate list is
    not what the row holds; if the published answer ranks are not the complete vocabulary;
    if permuting a board's cell order changes any count; if one of the eight symmetries of
    the square changes any count; or if relabelling a board's vertices changes its contact
    graph signature.
    """
    forbidden = {fold(value) for value in ANSWER_RANKS}
    forbidden |= {fold(PASS_TOKEN), fold(FAIL_TOKEN)}
    forbidden |= {fold(BLANK_TOKEN), fold(UNFILED_TOKEN)}
    for name, tokens in sorted(instance.envelope.neutral.items()):
        for position, token in enumerate(tokens):
            if fold(token) in forbidden:
                return CheckResult(
                    "components_public_contract", False,
                    "the committed %s token on row %d normalizes to %r, which is a legal "
                    "answer, a verdict or a status" % (name, position + 1, fold(token)),
                )
    for side in ("a", "b"):
        task = instance.side(side)
        mine = scope_sentence(task.label)
        theirs = scope_sentence("B" if task.label.upper() == "A" else "A")
        if mine not in task.text:
            return CheckResult(
                "components_public_contract", False,
                f"task {task.label} does not carry the registered scope sentence for its side",
            )
        if theirs in task.text:
            return CheckResult(
                "components_public_contract", False,
                f"task {task.label} carries the other sibling's scope sentence",
            )
        identifiers = generator.row_identifiers(task.table)
        if len({fold(identifier) for identifier in identifiers}) != len(identifiers):
            return CheckResult(
                "components_public_contract", False,
                f"task {task.label} has two row identifiers the shared reading folds together",
            )
        for identifier in identifiers:
            if fold(identifier) != identifier.lower() or not identifier.isascii():
                return CheckResult(
                    "components_public_contract", False,
                    f"the row identifier {identifier!r} does not survive the shared reading",
                )
        for row in task.table.rows:
            printed = " ".join("(%d,%d)" % cell for cell in row.cells)
            if printed not in task.text:
                return CheckResult(
                    "components_public_contract", False,
                    f"task {task.label} does not print the cells of row {row.row_id}",
                )
        if tuple(generator.answer_ranks(task.table)) != ANSWER_RANKS:
            return CheckResult(
                "components_public_contract", False,
                "the published answer ranks are not the complete vocabulary",
            )
        for cells in boards(task.table):
            base = flood_counts(cells)
            for order in (tuple(reversed(cells)), tuple(sorted(cells, key=lambda c: c[1]))):
                if flood_counts(order) != base:
                    return CheckResult(
                        "components_public_contract", False,
                        f"the printed order of {cells} changes its counts, so the notation "
                        "is part of the answer",
                    )
                if graph_signature(tuple(order)) != graph_signature(tuple(cells)):
                    return CheckResult(
                        "components_public_contract", False,
                        f"relabelling the vertices of {cells} changes its contact graph",
                    )
            for symmetry in SQUARE_SYMMETRIES:
                if flood_counts(carried(cells, symmetry)) != base:
                    return CheckResult(
                        "components_public_contract", False,
                        f"carrying {cells} by the {symmetry} of the square changes its counts",
                    )
    counted = sum(len(tokens) for tokens in instance.envelope.neutral.values())
    return CheckResult(
        "components_public_contract", True,
        "%d committed neutral tokens sit outside the whole vocabulary, both task texts "
        "carry their own scope sentence, every identifier and coordinate list survives the "
        "shared reading, and all %d rows of both sides are invariant under coordinate order "
        "and the %d symmetries of the square"
        % (counted, ROWS, len(SQUARE_SYMMETRIES)),
    )


#: The six checks this genre adds to the eleven every family runs, in reading order.
CHECKS = (
    ("components_semantics", check_semantics),
    ("components_shape", check_shape),
    ("components_support", check_support),
    ("components_bijection_copy", check_bijection_copy),
    ("components_analogy", check_analogy),
    ("components_public_contract", check_public_contract),
)


def pair_report(generator, instance: Instance) -> dict[str, object]:
    """The detail a release audit wants about one admitted pair, computed independently."""
    out: dict[str, object] = {"ordinal": instance.ordinal}
    for side in ("a", "b"):
        table = instance.side(side).table
        out[side] = {
            "boards": [list(map(list, cells)) for cells in boards(table)],
            "strata": [classify(cells) for cells in boards(table)],
            "counts": {
                option: [flood_components(cells, option) for cells in boards(table)]
                for option in OPTIONS
            },
            "signatures": [list(graph_signature(cells)) for cells in boards(table)],
        }
    out["copy"] = {
        option: {
            direction: list(
                bijection_maximum(
                    [int(v) for v in generator.key_for(
                        instance.side(source).table, {"contact_kernel": option})],
                    [int(v) for v in generator.key_for(
                        instance.side(target).table, {"contact_kernel": option})],
                )[:2]
            )
            for direction, source, target in (("a_to_b", "a", "b"), ("b_to_a", "b", "a"))
        }
        for option in OPTIONS
    }
    return out


__all__ = [
    "BIJECTIONS",
    "CHECKS",
    "CORNER_STEPS",
    "EXPECTED_BLOCKS",
    "EXPECTED_CEILING",
    "EXPECTED_FLOOR",
    "EXPECTED_HEADROOM",
    "EXPECTED_LOOKUP_GRADE",
    "EXPECTED_MOVEMENT",
    "OPTIONS",
    "SIDE_STEPS",
    "SQUARE_SYMMETRIES",
    "TOLERANCE",
    "bijection_maximum",
    "board_refusal",
    "boards",
    "carried",
    "check_analogy",
    "check_bijection_copy",
    "check_public_contract",
    "check_semantics",
    "check_shape",
    "check_support",
    "classify",
    "contingency",
    "copy_refusal",
    "flood_components",
    "flood_counts",
    "frequency_refusal",
    "graph_signature",
    "modal_filing",
    "neighbourhood",
    "on_board",
    "pair_colour",
    "pair_report",
    "retasked_instance",
    "row_moves",
    "steps",
]
