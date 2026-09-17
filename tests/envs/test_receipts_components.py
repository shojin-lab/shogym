"""The components genre: its geometry, its construction, its cells and its evidence.

Each test states the failure it is here to catch, because a test whose failure condition
is "something changed" is a test nobody can act on.
"""

from __future__ import annotations

import inspect
import itertools
import json
import random
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import pytest

from shogym.envs.receipts import admission
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import checks, copy_profiles
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.generators import components
from shogym.envs.receipts.generators import components_audit as audit
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse_body as parse_oracle_body
from shogym.envs.receipts.protocol import (
    ConstructionExhausted,
    Instance,
    NoFiling,
    SealedSubmission,
    Task,
    draw,
    option_mentions,
)
from shogym.envs.receipts.receipt_ast import (
    ReceiptAST,
    frozen_envelope,
    mask_slots,
    serialize,
    slot_ranges,
)
from shogym.envs.receipts.render import judge_cells

GENERATOR = components.GENERATOR
#: A private fixed master. Test keys only: nothing here is a permissible production key.
MASTER = bytes(range(32))
OPTIONS = tuple(components.AXES[0].options)


@lru_cache(maxsize=8)
def _drawn(ordinal: int) -> Instance:
    return draw(GENERATOR, MASTER, ordinal)


def _cells(text: str) -> tuple[tuple[int, int], ...]:
    """A board written the way the schedule prints it."""
    return tuple(
        (int(part.strip("()").split(",")[0]), int(part.strip("()").split(",")[1]))
        for part in text.split()
    )


#: The eight worked boards of the specification, and the keys it states for them. W3 is
#: the cycle witness and W8 is the one-cell board.
WORKED: tuple[tuple[str, str, tuple[int, int, int]], ...] = (
    ("W1", "(1,1) (2,2) (3,3) (4,4) (5,2) (5,5)", (6, 2, 2)),
    ("W2", "(1,2) (3,2) (4,2) (5,2) (5,3) (5,4)", (2, 5, 2)),
    ("W3", "(1,2) (2,3) (2,4) (3,3) (3,4) (4,4)", (2, 2, 1)),
    ("W4", "(1,3) (2,2) (2,4) (3,1) (3,3)", (5, 1, 1)),
    ("W5", "(2,2) (3,4) (4,3) (4,5) (5,4)", (5, 2, 2)),
    ("W6", "(2,2) (3,2) (4,1) (4,2) (5,1)", (1, 3, 1)),
    ("W7", "(1,5) (2,1) (3,2) (3,4) (4,2) (5,5)", (5, 5, 4)),
    ("W8", "(1,1)", (1, 1, 1)),
)


# ----- 1: the geometry -------------------------------------------------------


def test_the_eight_worked_boards_have_the_keys_the_specification_states() -> None:
    """The three rules on the eight boards the design was written against.

    It fails if any of the eight has a different key under any rule, if a diagonal touch
    is treated as a shared side, if a board wraps at a border, or if a shared side is
    counted as a contact under the corner-only rule.
    """
    for name, printed, expected in WORKED:
        cells = _cells(printed)
        assert components.counts_of(cells) == expected, name
        assert audit.flood_counts(cells) == expected, name

    diagonal = ((1, 1), (2, 2))
    assert components.components_of(diagonal, components.SIDE_CONTACTS) == 2
    assert components.components_of(diagonal, components.CORNER_CONTACTS) == 1
    assert components.components_of(diagonal, components.COMBINED_CONTACTS) == 1

    shared_side = ((1, 1), (1, 2))
    assert components.components_of(shared_side, components.SIDE_CONTACTS) == 1
    assert components.components_of(shared_side, components.CORNER_CONTACTS) == 2
    assert components.components_of(shared_side, components.COMBINED_CONTACTS) == 1

    for across, down in ((((1, 1), (1, 5)), ((1, 1), (5, 1))), ):
        for board in (across, down):
            for option in OPTIONS:
                assert components.components_of(board, option) == 2, board

    # A diagonal whose two intervening side neighbours are occupied is still a diagonal.
    blocked = ((1, 1), (1, 2), (2, 1), (2, 2))
    assert components.components_of(blocked, components.CORNER_CONTACTS) == 2
    assert components.components_of(blocked, components.SIDE_CONTACTS) == 1

    # A board is refused rather than given an answer.
    for wrong in ((), ((0, 1),), ((1, 6),), ((1, 1), (1, 1)),
                  tuple((1, c) for c in range(1, 6)) + ((2, 1), (2, 2))):
        with pytest.raises(ValueError):
            components.components_of(wrong, components.SIDE_CONTACTS)
        assert audit.board_refusal(wrong)


# ----- 2: the cycle ----------------------------------------------------------


def test_a_cycle_merges_nothing_and_a_path_merges_everything_along_it() -> None:
    """Transitivity, and what a contact tally would have said instead.

    It fails if a side answer is computed as cells minus contacts, if unioning two cells
    that already share an island reduces the count, or if two cells with no direct
    contact are left in different islands when a path joins them.
    """
    w3 = _cells(WORKED[2][1])
    contacts, cellcount, islands = components.side_structure(w3)
    assert (contacts, cellcount, islands) == (5, 6, 2)
    assert cellcount - contacts == 1
    assert components.components_of(w3, components.SIDE_CONTACTS) == 2
    assert components.has_side_cycle(w3)

    block = ((1, 1), (1, 2), (2, 1), (2, 2))
    assert components.side_structure(block) == (4, 4, 1)
    assert components.components_of(block, components.SIDE_CONTACTS) == 1
    assert len(block) - 4 == 0

    chain = ((1, 1), (1, 2), (1, 3))
    assert components.components_of(chain, components.SIDE_CONTACTS) == 1
    assert components.linked(chain[0], chain[2], components.SIDE_CONTACTS) is False
    assert not components.has_side_cycle(chain)


# ----- 3: the independent evaluator ------------------------------------------


def test_union_find_and_an_independent_flood_fill_agree_on_every_eligible_board() -> None:
    """Both implementations, on all 245505 boards of one to six cells, under all rules.

    It fails if the production union-find and an independent flood fill over
    independently built coordinate neighbourhoods disagree anywhere: 736515 comparisons.
    A scorer that is wrong is carried identically into the key, the receipt, the placebo
    and every gate, so nothing but a second implementation can find it.
    """
    seen = 0
    for size in range(components.MIN_CELLS, components.MAX_CELLS + 1):
        for cells in itertools.combinations(components.BOARD, size):
            for option in OPTIONS:
                assert components.components_of(cells, option) == audit.flood_components(
                    cells, option
                ), (cells, option)
            seen += 1
    assert seen == 245505


# ----- 4: the catalogue ------------------------------------------------------


def test_the_catalogue_is_the_exact_one_and_its_classes_are_isomorphism_classes() -> None:
    """The three libraries, their embeddings, and what a class identifier means.

    It fails if the informative class counts are not 46, 96 and 78, if a class holds a
    repeated coordinate embedding or one that is not in its stratum, if the control pools
    are not the contact-free boards of each size, or if the production class identifier
    is not the same equivalence as an independent minimization over every vertex order.
    """
    held = components.catalogue()
    assert {letter: len(held.classes[letter]) for letter in components.STRATA} == {
        "S": 46, "C": 96, "U": 78
    }
    assert sum(len(held.embeddings[c]) for letter in components.STRATA
               for c in held.classes[letter]) == 169053
    assert {size: len(held.controls[size]) for size in components.CONTROL_SIZES} == {
        1: 25, 2: 228, 3: 964, 4: 1987, 5: 1974, 6: 978
    }
    for size, pool in held.controls.items():
        assert list(pool) == sorted(set(pool))
        for board in pool:
            assert len(board) == size
            assert components.counts_of(board) == (size, size, size)

    signatures: dict[tuple[int, ...], tuple[int, ...]] = {}
    for letter in components.STRATA:
        assert list(held.classes[letter]) == sorted(held.classes[letter])
        for identifier in held.classes[letter]:
            boards = held.embeddings[identifier]
            assert list(boards) == sorted(set(boards))
            sampled = (boards[0], boards[len(boards) // 2], boards[-1])
            for board in sampled:
                assert components.class_identifier(board) == identifier
                assert components.stratum_of(components.counts_of(board)) == letter
                assert components.counts_of(board) == held.counts[identifier]
                assert components.has_side_cycle(board) == held.cycles[identifier]
            made = {audit.graph_signature(board) for board in sampled}
            assert len(made) == 1
            signatures[identifier] = made.pop()
    assert len(set(signatures.values())) == len(signatures)

    # The same graph in another place, another rotation and another reflection is the
    # same class, and that is what a class is for.
    for _, printed, _ in WORKED:
        cells = _cells(printed)
        if components.stratum_of(components.counts_of(cells)) == "":
            continue
        for symmetry in audit.SQUARE_SYMMETRIES:
            carried = audit.carried(cells, symmetry)
            assert components.class_identifier(carried) == components.class_identifier(cells)
            assert audit.graph_signature(carried) == audit.graph_signature(cells)


# ----- 5: the construction ---------------------------------------------------


def test_a_generated_pair_is_the_recipe_and_shares_no_graph_or_board() -> None:
    """What the bounded search is allowed to accept.

    It fails if either side is not six rows in each of the three informative strata and
    six separated controls, if a count is missing from a side's answers under some rule,
    if a count occurs more than eight times, if neither informative board on a side closes
    a side-contact cycle, or if an informative contact graph or an exact board occurs on
    both siblings.
    """
    for ordinal in range(4):
        instance = _drawn(ordinal)
        result = audit.check_shape(GENERATOR, instance)
        assert result.passed, result.detail
        for side in ("a", "b"):
            table = instance.side(side).table
            assert len(table.rows) == components.ROWS
            kinds = [audit.classify(cells) for cells in audit.boards(table)]
            assert kinds.count("control") == len(components.CONTROL_SIZES)
            for letter in components.STRATA:
                assert kinds.count(letter) == components.STRATUM_ROWS
            for option in OPTIONS:
                key = GENERATOR.key_for(table, {"contact_kernel": option})
                assert set(key) == set(components.ANSWER_RANKS)
                assert max(key.count(value) for value in set(key)) <= 8
        assert not (set(audit.boards(instance.a.table))
                    & set(audit.boards(instance.b.table)))


def test_the_registered_checkpoints_of_the_bounded_search_reproduce() -> None:
    """The proposal and ordering the public test master reaches at four ordinals.

    It fails if the keyed class sampling, the embedding draw, the control draw, the
    frequency cap, the cycle requirement, the ordering shuffle or the copy bound moves:
    all four checkpoints are a function of every one of them. They are deterministic test
    fixtures under a private master and are not a promise about any other key.
    """
    assert (components.MAX_PROPOSALS, components.MAX_ORDERINGS) == (4096, 64)
    assert _search(0) == (205, 27)
    assert _search(1) == (28, 1)
    assert _search(2) == (276, 6)
    assert _search(42) == (1047, 3)


def _search(ordinal: int) -> tuple[int, int]:
    """Which proposal and ordering the registered search accepts for one ordinal."""
    for proposal in range(components.MAX_PROPOSALS):
        chosen = components._choose_classes(MASTER, ordinal, proposal)
        triples = {
            side: components._proposal_triples(chosen, side) for side in ("A", "B")
        }
        if any(audit.frequency_refusal(triples[side][0]) for side in ("A", "B")):
            continue
        if any(not triples[side][1] for side in ("A", "B")):
            continue
        for ordering in range(components.MAX_ORDERINGS):
            laid = {
                side: components._ordered(
                    MASTER, ordinal, proposal, ordering, side, triples[side][0]
                )
                for side in ("A", "B")
            }
            if not audit.copy_refusal(laid["A"], laid["B"], components.MAX_COPY_ROWS):
                return proposal, ordering
    raise AssertionError("the registered search found nothing")


# ----- 6: determinism --------------------------------------------------------


def test_nothing_public_moves_when_the_cache_the_call_order_or_the_rule_moves() -> None:
    """The public surface is a function of the key and the ordinal and of nothing else.

    It fails if clearing the construction cache changes a table, if asking for B before A
    changes either of them, if the drawn rule changes a table, an identifier, a task text
    or the envelope, or if any construction entry point takes a convention at all.
    """
    first = GENERATOR.build_table(MASTER, 3, "A")
    second = GENERATOR.build_table(MASTER, 3, "B")
    components.build_pair.cache_clear()
    components.catalogue.cache_clear()
    again_b = GENERATOR.build_table(MASTER, 3, "B")
    again_a = GENERATOR.build_table(MASTER, 3, "A")
    assert (again_a, again_b) == (first, second)
    assert again_a.body == first.body

    instance = _drawn(3)
    for option in OPTIONS:
        convention = MappingProxyType({"contact_kernel": option})
        for side in ("a", "b"):
            task = instance.side(side)
            rebuilt = Task(
                label=task.label, task_id=task.task_id, surface=task.surface,
                table=task.table, text="", key=tuple(
                    GENERATOR.key_for(task.table, convention)
                ),
            )
            assert GENERATOR.describe(rebuilt.public()) == task.text
            assert GENERATOR.row_identifiers(rebuilt.table) == GENERATOR.row_identifiers(
                task.table
            )
    assert GENERATOR.build_envelope(MASTER, 3) == instance.envelope

    for name in ("build_pair", "_choose_classes", "_proposal_triples", "_boards",
                 "_commit", "_identifiers", "_ordered"):
        signature = inspect.signature(getattr(components, name))
        assert not any(
            "convention" in parameter or "kernel" in parameter
            for parameter in signature.parameters
        ), name


# ----- 7: the support --------------------------------------------------------


def test_the_room_is_the_same_under_every_reference_rule() -> None:
    """The counterfactual references, and the arithmetic the recipe fixes.

    It fails if under any reference on either sibling the three keys alias after
    normalization or after serialization, if a pair of keys moves other than twelve rows,
    if the receipt resolves other than three blocks, if any row is evident, or if the
    ceiling, the lookup floor or the headroom is not one, five sixths and one sixth.
    """
    for ordinal in range(3):
        instance = _drawn(ordinal)
        result = audit.check_support(GENERATOR, instance)
        assert result.passed, result.detail
        assert "ceiling 1.000000" in result.detail
        assert "floor 0.833333" in result.detail
        assert "H 0.166667" in result.detail

    # A pair whose strata are not the recipe does not produce that arithmetic, and the
    # check says so rather than passing on a number nobody read.
    instance = _drawn(0)
    flattened = _replace_rows(
        instance, "b",
        {n: components.catalogue().controls[1][n % 25] for n in range(6)},
    )
    refused = audit.check_support(GENERATOR, flattened)
    assert not refused.passed
    assert "moves" in refused.detail or "blocks" in refused.detail


def _replace_rows(
    instance: Instance, side: str, replacements: Mapping[int, tuple[tuple[int, int], ...]]
) -> Instance:
    """The same instance with some of one side's boards swapped, for a refusal fixture."""
    task = instance.side(side)
    rows = list(task.table.rows)
    for position, cells in replacements.items():
        rows[position] = components.ComponentsRow(
            row_id=rows[position].row_id, cells=cells
        )
    table = components.ComponentsTable(surface=task.table.surface, rows=tuple(rows))
    made = Task(
        label=task.label, task_id=task.task_id, surface=task.surface, table=table,
        text=GENERATOR.describe(
            Task(
                label=task.label, task_id=task.task_id, surface=task.surface,
                table=table, text="", key=(),
            ).public()
        ),
        key=tuple(GENERATOR.key_for(table, instance.convention)),
    )
    return Instance(
        generator=instance.generator, genre=instance.genre, ordinal=instance.ordinal,
        convention=instance.convention,
        a=made if side == "a" else instance.a,
        b=made if side == "b" else instance.b,
        envelope=instance.envelope,
    )


# ----- 8: the copy families --------------------------------------------------


def test_the_registered_closure_and_the_all_bijection_optimizer_are_what_they_claim() -> None:
    """What each family enumerates, and whether the fast maximum is the exact one.

    It fails if the registered row moves are not the 48 dihedral rearrangements, if the
    registered token family is not the six cyclic maps this complete vocabulary
    generates, or if the independent all-bijection optimizer disagrees with a brute-force
    enumeration of all 720 bijections against all 48 row moves on any fixture.
    """
    instance = _drawn(0)
    key = list(instance.a.key)
    moves = copy_profiles.permutations(key, components.ROWS)
    assert len(moves) == 2 * components.ROWS == 48
    assert len(audit.row_moves(key)) == 48
    assert moves[0] == key
    assert sorted(tuple(m) for m in moves) == sorted(
        tuple(m) for m in audit.row_moves(key)
    )

    maps = copy_profiles.token_maps(
        key, components.ANSWER_RANKS, components.ANSWER_RANKS
    )
    assert len(maps) == 6
    rotations = [
        {
            token: components.ANSWER_RANKS[(n + shift) % 6]
            for n, token in enumerate(components.ANSWER_RANKS)
        }
        for shift in range(6)
    ]
    assert sorted(tuple(sorted(m.items())) for m in maps) == sorted(
        tuple(sorted(m.items())) for m in rotations
    )
    assert len(copy_profiles.relabellings(GENERATOR, instance, key, components.ROWS)) == 6

    for source, target in (
        ([1, 2, 3, 4, 5, 6] * 4, [6, 5, 4, 3, 2, 1] * 4),
        ([int(v) for v in instance.a.key], [int(v) for v in instance.b.key]),
        ([1] * 12 + [2] * 12, [2] * 12 + [1] * 12),
        ([1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6] * 2, [3, 1, 4, 1, 5, 9 % 6 + 1] * 4),
    ):
        assert audit.bijection_maximum(source, target)[0] == _brute_force(source, target)


def _brute_force(source: Sequence[int], target: Sequence[int]) -> int:
    """Correct rows at best, by enumerating every row move against every bijection."""
    best = 0
    for moved in audit.row_moves(list(source)):
        for mapping in itertools.permutations(range(1, 7)):
            best = max(
                best,
                sum(1 for x, y in zip(moved, target) if mapping[x - 1] == y),
            )
    return best


# ----- 9: what admission refuses ---------------------------------------------


def test_a_pair_sharing_a_graph_or_over_the_copy_bound_cannot_enter_a_bank() -> None:
    """Two arrangements the eleven common checks cannot see, and what they cost.

    It fails if a pair holding the same informative contact graph on both siblings is
    admitted, or if a printed order whose all-bijection copy maximum is over twelve of
    twenty four is admitted, in either case while the eleven common checks pass on it. It
    also fails if such a pair can fill a bank.
    """
    instance = _drawn(0)
    disguised = _isomorphic_sibling(instance)
    assert tuple(disguised.b.key) == tuple(instance.b.key)
    failed = _refuses_a_bank(disguised)
    assert "components_analogy" in failed
    assert "components_shape" in failed

    copied = _over_the_copy_bound(instance)
    failed = _refuses_a_bank(copied)
    assert "components_bijection_copy" in failed


#: The eleven checks every family runs, which are what an arrangement has to survive for
#: this test to be about the ones this genre adds.
COMMON = (
    "exercise", "materiality", "copy", "fixation", "envelope", "graded", "placebo",
    "neutral", "oracle", "lint", "invariance",
)


def _refuses_a_bank(arranged: Instance) -> tuple[str, ...]:
    """The arranged pair put through the checks, admission and a bank fill.

    The construction is stubbed to hand back the arranged tables, so the pair is what the
    key builds and `fixation` rebuilds to it: an arrangement that could not survive
    fixation would be refused for not being a rebuild of its own key rather than for what
    it does, and this test is about what it does. Returns the checks that refused it.
    """
    tables = (arranged.a.table, arranged.b.table)
    held = components.build_pair
    try:
        components.build_pair = lambda *_: tables  # type: ignore[assignment]
        results = {r.name: r for r in checks.run_checks(
            GENERATOR, arranged, MASTER,
            max_copy_score=admission.REGISTERED_MAX_COPY_SCORE,
            max_flip_score=admission.REGISTERED_MAX_FLIP_SCORE,
            min_leverage=admission.REGISTERED_MIN_LEVERAGE,
        )}
        assert all(results[name].passed for name in COMMON), [
            results[name].detail for name in COMMON if not results[name].passed
        ]
        report = admission.report(GENERATOR, arranged, MASTER, admission.Thresholds())
        assert not report.admitted
        bank = bank_mod.Bank(
            generator=GENERATOR.name, genre=GENERATOR.genre,
            renderer=bank_mod.RENDERER_CONFIGURATION, master=MASTER, size=1,
        )
        with pytest.raises(ValueError):
            bank_mod.population(bank, GENERATOR)
        return report.failed_checks
    finally:
        components.build_pair = held


def _isomorphic_sibling(instance: Instance) -> Instance:
    """One of B's informative boards swapped for a re-embedding of one of A's.

    The replacement has A's contact graph and B's own counts, so every key on both sides
    is exactly what it was and the eleven common checks compute the same numbers. What
    changes is that a reader holding A's correction can recognise the board again.
    """
    a_boards = [
        cells for cells in audit.boards(instance.a.table)
        if audit.classify(cells) in components.STRATA
    ]
    b_boards = audit.boards(instance.b.table)
    for position, target in enumerate(b_boards):
        if audit.classify(target) not in components.STRATA:
            continue
        for source in a_boards:
            if audit.flood_counts(source) != audit.flood_counts(target):
                continue
            for symmetry in audit.SQUARE_SYMMETRIES:
                moved = audit.carried(source, symmetry)
                if moved in b_boards or moved in set(audit.boards(instance.a.table)):
                    continue
                return _replace_rows(instance, "b", {position: moved})
    raise AssertionError("no informative board of A matched one of B's counts")


def _over_the_copy_bound(instance: Instance) -> Instance:
    """B's rows in a printed order the construction filter refused.

    The search accepts the first ordering whose all-bijection maximum is at most twelve of
    twenty four, so an ordering it passed over is the arrangement this wants. A rotation
    of B would not do: rotating B is the same comparison as rotating A's filing the other
    way and that is already one of the 48 registered row moves, so the maximum cannot move
    under one. An independent shuffle is the lever step five actually pulls.

    The keys are the same multiset however the rows are laid out, so materiality, the
    one-rule-wrong score and the axis leverage are untouched, and the registered closure
    is six cyclic maps where this is all 720: that gap is what the added check is for.
    """
    for seed in range(64):
        rows = list(instance.b.table.rows)
        random.Random(seed).shuffle(rows)
        arranged = _replace_rows(
            instance, "b", {n: rows[n].cells for n in range(components.ROWS)}
        )
        if audit.check_bijection_copy(GENERATOR, arranged).passed:
            continue
        registered = checks.check_copy(
            GENERATOR, arranged,
            admission.REGISTERED_MAX_COPY_SCORE,
            admission.REGISTERED_MAX_FLIP_SCORE,
            admission.REGISTERED_MIN_LEVERAGE,
        )
        if registered.passed:
            return arranged
    raise AssertionError("no shuffle of B went over the bound with the common checks green")


# ----- 10: the reading -------------------------------------------------------


def test_the_reading_is_the_registered_one_on_this_genre_too() -> None:
    """Case, whitespace, duplicates, omissions, positional text and numeric spellings.

    It fails if case or outer whitespace changes a grade, if a later duplicate replaces
    the first value, if an unknown identifier is read, if an omission is credited, if
    comma-free prose gets a positional reading without one line per printed row, if a
    NoFiling reason changes, or if any of 01, +1, 1.0 and one is read as the value 1.
    """
    instance = _drawn(0)
    task = instance.a
    identifiers = list(GENERATOR.row_identifiers(task.table))
    key = list(task.key)

    perfect = "\n".join("%s,%s" % kv for kv in zip(identifiers, key))
    assert GENERATOR.score(task, GENERATOR.parse_and_canonicalize(task, perfect))[0] == 1.0

    shouted = "\n".join(
        "  %s ,  %s  " % (i.lower(), v) for i, v in zip(identifiers, key)
    )
    assert GENERATOR.score(task, GENERATOR.parse_and_canonicalize(task, shouted))[0] == 1.0

    doubled = perfect + "\n" + "\n".join("%s,9" % i for i in identifiers)
    read = GENERATOR.parse_and_canonicalize(task, doubled)
    assert isinstance(read, SealedSubmission)
    assert list(read.values) == key
    assert len(read.duplicates) == len(identifiers)

    unknown = GENERATOR.parse_and_canonicalize(task, "NOSUCHROW,3")
    assert isinstance(unknown, NoFiling) and unknown.reason == "no_known_identifier"
    assert isinstance(GENERATOR.parse_and_canonicalize(task, ""), NoFiling)
    assert GENERATOR.parse_and_canonicalize(task, "").reason == "empty"
    assert GENERATOR.parse_and_canonicalize(task, {"a": 1}).reason == "unreadable"

    positional = "\n".join(key)
    assert GENERATOR.score(
        task, GENERATOR.parse_and_canonicalize(task, positional)
    )[0] == 1.0
    short = "\n".join(key[:12])
    assert isinstance(GENERATOR.parse_and_canonicalize(task, short), NoFiling)
    prose = "I could not work out which contact rule applies"
    assert isinstance(GENERATOR.parse_and_canonicalize(task, prose), NoFiling)

    omitted = "\n".join("%s,%s" % kv for kv in zip(identifiers[:12], key[:12]))
    read = GENERATOR.parse_and_canonicalize(task, omitted)
    assert isinstance(read, SealedSubmission)
    assert len(read.omissions) == 12
    assert GENERATOR.score(task, read)[0] == 0.5

    for spelling in ("01", "+1", "1.0", "one", " 1 1", "1st"):
        row = next(n for n, value in enumerate(key) if value == "1")
        filed = "%s,%s" % (identifiers[row], spelling)
        read = GENERATOR.parse_and_canonicalize(task, filed)
        _, outcomes = GENERATOR.score(task, read)
        assert not outcomes[row].matched, spelling
        assert outcomes[row].was_filed


# ----- 11: the score ---------------------------------------------------------


def test_the_score_is_a_fraction_of_twenty_four_rows_rounded_to_six_places() -> None:
    """The denominator, the empty value, the absent filing and the rounding.

    It fails if a partial filing changes the denominator from the printed row count, if a
    filed empty value earns credit, if a NoFiling reports other than one outcome per
    printed row, or if a score loses the shared six-place rounding.
    """
    instance = _drawn(0)
    task = instance.a
    identifiers = list(GENERATOR.row_identifiers(task.table))
    key = list(task.key)

    for correct in (0, 1, 7, 23, 24):
        filed = "\n".join(
            "%s,%s" % (identifiers[n], key[n] if n < correct else "")
            for n in range(components.ROWS)
        )
        read = GENERATOR.parse_and_canonicalize(task, filed)
        score, outcomes = GENERATOR.score(task, read)
        assert len(outcomes) == components.ROWS
        assert score == round(correct / 24.0, 6)
        assert sum(1 for o in outcomes if o.matched) == correct
        assert all(o.was_filed for o in outcomes)

    score, outcomes = GENERATOR.score(task, NoFiling("empty"))
    assert score == 0.0
    assert len(outcomes) == components.ROWS
    assert not any(o.was_filed for o in outcomes)
    assert [o.correct for o in outcomes] == key

    sevenths = "\n".join(
        "%s,%s" % (identifiers[n], key[n] if n < 7 else "9") for n in range(components.ROWS)
    )
    assert GENERATOR.score(
        task, GENERATOR.parse_and_canonicalize(task, sevenths)
    )[0] == 0.291667


# ----- 12: the receipt -------------------------------------------------------


def test_a_failed_row_corrects_to_its_own_truth_and_a_sibling_uses_its_own() -> None:
    """What a graded row is allowed to say, on the drawn rule and on every other.

    It fails if a failed row prints anything but that row's own supplied truth, if a
    passed row prints a correction, or if either sibling's rendering uses the other's
    identifiers or the other's key.
    """
    instance = _drawn(1)
    for side in ("a", "b"):
        task = instance.side(side)
        identifiers = list(GENERATOR.row_identifiers(task.table))
        half = "\n".join(
            "%s,%s" % (identifiers[n], task.key[n] if n % 2 else "9")
            for n in range(components.ROWS)
        )
        canonical = GENERATOR.parse_and_canonicalize(task, half)
        for option in OPTIONS:
            truth = GENERATOR.key_for(task.table, {"contact_kernel": option})
            ast = GENERATOR.render_receipt(task, canonical, truth)
            assert [row.identifier for row in ast.rows] == identifiers
            for position, row in enumerate(ast.rows):
                slots = {slot.name: slot.value for slot in row.slots}
                if slots["verdict"] == "PASS":
                    assert slots["correction"] == ""
                else:
                    assert slots["correction"] == truth[position]
                assert slots["correction"] != components.BLANK_TOKEN

    a_identifiers = set(GENERATOR.row_identifiers(instance.a.table))
    b_ast = GENERATOR.render_receipt(
        instance.b,
        GENERATOR.parse_and_canonicalize(instance.b, ""),
        instance.b.key,
    )
    assert not a_identifiers & {row.identifier for row in b_ast.rows}
    assert [
        {slot.name: slot.value for slot in row.slots}["correction"] for row in b_ast.rows
    ] == list(instance.b.key)


# ----- 13: the bytes ---------------------------------------------------------


def test_every_cell_is_congruent_under_every_filing_class_and_every_draw() -> None:
    """The envelope, the two registered slots, and what may move inside them.

    It fails if a byte outside the two registered slots differs between the graded and
    placebo cells, if the JSON-encoded lengths of the two differ, if a cell is not 2657
    bytes, or if a canonical identifier or answer is truncated in the bytes an agent
    reads.
    """
    instance = _drawn(0)
    assert components.ENVELOPE_SIZE == 2657
    envelope = frozen_envelope(instance.envelope)
    assert envelope.row_line_width == 59
    assert envelope.slot_span("verdict") == (40, 44)
    assert envelope.slot_span("correction") == (46, 58)

    judged_count = 0
    for side in ("a", "b"):
        for shape in checks.FILING_CLASSES:
            raw = checks.filing_of(GENERATOR, instance, side, shape)
            for option in OPTIONS:
                convention = {"contact_kernel": option}
                task = instance.side(side)
                retasked = Task(
                    label=task.label, task_id=task.task_id, surface=task.surface,
                    table=task.table, text=task.text,
                    key=tuple(GENERATOR.key_for(task.table, convention)),
                )
                canonical = GENERATOR.parse_and_canonicalize(retasked, raw)
                judged = judge_cells(
                    GENERATOR, retasked, canonical, convention, envelope
                )
                assert judged.problems == ()
                graded = judged.payloads["graded"]
                placebo = judged.payloads["placebo"]
                oracle = judged.payloads["oracle"]
                assert len(graded) == len(placebo) == len(oracle) == 2657
                ranges = slot_ranges(judged.asts["graded"], envelope)
                assert mask_slots(graded, ranges) == mask_slots(placebo, ranges)
                assert len(json.dumps(graded.decode("ascii"))) == len(
                    json.dumps(placebo.decode("ascii"))
                )
                text = graded.decode("ascii")
                for identifier in GENERATOR.row_identifiers(task.table):
                    assert identifier in text
                judged_count += 1
    assert judged_count == 2 * len(checks.FILING_CLASSES) * len(OPTIONS)

    # No canonical answer and no identifier is ever near its field's width.
    assert max(len(v) for v in components.ANSWER_RANKS) < components.CORRECTION_WIDTH
    assert all(
        len(row.row_id) < components.IDENTIFIER_WIDTH
        for row in instance.a.table.rows
    )


# ----- 14: the oracle --------------------------------------------------------


def test_the_oracle_is_the_registered_cell_and_states_exactly_the_drawn_rule() -> None:
    """The phrase table, the parse back, the head, and what an extra sentence costs.

    It fails if one option's phrase contains another's, if any supported body does not
    parse back to the rule it was rendered from, if an appended sentence or a changed
    head survives the shared exact-template comparison, or if the registered scope
    wording moves.
    """
    template = components.ORACLE_TEMPLATE
    assert set(template.phrases["contact_kernel"]) == set(OPTIONS)
    for one, other in itertools.permutations(OPTIONS, 2):
        assert (
            template.phrases["contact_kernel"][one]
            not in template.phrases["contact_kernel"][other]
        )
    with pytest.raises(ValueError):
        OracleTemplate(
            head=components.ORACLE_HEAD,
            sentences=template.sentences,
            phrases={"contact_kernel": dict(
                template.phrases["contact_kernel"], corner_contacts="a side is shared too"
            )},
        )

    instance = _drawn(0)
    for option in OPTIONS:
        convention = {"contact_kernel": option}
        ast = GENERATOR.render_oracle(instance.a.task_id, convention, components.ROWS)
        assert GENERATOR.parse_oracle(ast) == convention
        assert parse_oracle_body(template, ast.body) == convention
        assert ast.body[:len(components.ORACLE_HEAD)] == components.ORACLE_HEAD
        rendered = serialize(ast, frozen_envelope(instance.envelope))
        assert len(rendered) == 2657
        assert option not in rendered.decode("ascii")

    assert components.ORACLE_HEAD[0].startswith(
        "HOUSE CONVENTIONS FOR THIS SCHEDULE AND THE NEXT ONE YOU FILE"
    )
    assert scope_sentence("A") in instance.a.text
    assert scope_sentence("B") in instance.b.text
    assert scope_sentence("B") not in instance.a.text

    class _Chatty:
        """A renderer that states the rule correctly and adds a line of coaching."""

        def __getattr__(self, name: str) -> object:
            return getattr(GENERATOR, name)

        def render_oracle(self, task_id, convention, row_count=0) -> ReceiptAST:
            honest = GENERATOR.render_oracle(task_id, convention, row_count)
            return ReceiptAST(
                kind=honest.kind, task_id=honest.task_id, row_count=honest.row_count,
                body=honest.body + ("Count the islands carefully.",),
            )

    chatty = _Chatty()
    task = instance.a
    canonical = GENERATOR.parse_and_canonicalize(task, "")
    judged = judge_cells(
        chatty, task, canonical, instance.convention, instance.envelope
    )
    assert any("not the cell the registered template renders" in p for p in judged.problems)


# ----- 15: the public contract -----------------------------------------------


def test_the_public_contract_holds_on_the_surface_and_in_the_neutral_tokens() -> None:
    """What the schedule may print, and what the placebo may not.

    It fails if a committed neutral token normalizes onto a legal answer, a verdict or a
    status token, if either task text names an option of the axis, if permuting a board's
    printed coordinates changes a count, or if one of the eight symmetries of the square
    changes a count.
    """
    for ordinal in range(2):
        instance = _drawn(ordinal)
        result = audit.check_public_contract(GENERATOR, instance)
        assert result.passed, result.detail
        assert checks.check_neutral(GENERATOR, instance).passed
        assert checks.check_lint(GENERATOR, instance).passed
        for task in (instance.a, instance.b):
            assert option_mentions(components.AXES, task.text) == []
            assert task.text.startswith("PATTERN SCHEDULE")
            assert components.TABLE_HEAD in task.text
        assert components.FILLER_ALPHABET == "qzxv"
        for tokens in instance.envelope.neutral.values():
            assert all(set(token) <= set("qzxv") for token in tokens)


# ----- 16: construction exhaustion -------------------------------------------


def test_a_construction_that_runs_out_is_a_named_whole_bank_failure() -> None:
    """Exhaustion is not a skipped ordinal and not an unnamed exception.

    It fails if a bounded search that cannot produce a pair skips the ordinal, reaches a
    caller as an exception with no name of its own, silently draws another master, or
    loses the count of what refused what.
    """
    components.build_pair.cache_clear()
    held = audit.frequency_refusal
    bound = components.MAX_PROPOSALS
    try:
        components.MAX_PROPOSALS = 8
        audit.frequency_refusal = lambda *_: "a fixture refusal"  # type: ignore[assignment]
        with pytest.raises(ConstructionExhausted) as exhausted:
            components.build_pair(MASTER, 5)
        said = str(exhausted.value)
        assert "ordinal 5" in said and "8 proposals" in said
        assert "frequency refused 8" in said

        bank = bank_mod.Bank(
            generator=GENERATOR.name, genre=GENERATOR.genre,
            renderer=bank_mod.RENDERER_CONFIGURATION, master=MASTER, size=2,
        )
        with pytest.raises(ConstructionExhausted) as named:
            bank_mod.population(bank, GENERATOR)
        assert "ordinal 0" in str(named.value)
        assert "holding 0 of 2" in str(named.value)
    finally:
        audit.frequency_refusal = held
        components.MAX_PROPOSALS = bound
        components.build_pair.cache_clear()
    assert components.build_pair(MASTER, 5)


# ----- the registration, and the labels it did not move ----------------------


def test_the_roster_carries_the_third_genre_under_the_labels_the_second_one_set() -> None:
    """Registration, the declared profile, and why neither shared label moved.

    It fails if the genre is not reachable through the one door every command and every
    bundle reaches a generator through, if it is registered as a gate vector rather than
    a family, if a generator that declares no copy profile is admitted through that door,
    or if the executable gate label or the renderer label moved.

    Neither moved, and that is a claim with a reason. A label is what a family was
    admitted under, and this genre registers the receipt geometry ledger already
    registers: identifier 12, observed 16, verdict 4, correction 12, and an envelope of
    2657 bytes. The six checks it adds are dispatched by genre name, so nothing any other
    family is admitted under changed, and a bank frozen under these labels is still gated
    on cells of the shape they name.
    """
    from shogym.envs.receipts.registry import (
        FIXTURES,
        GENRES,
        is_fixture,
        load_generator,
    )
    from shogym.envs.receipts.generators import ledger, soundchange

    assert GENRES["components"] == "shogym.envs.receipts.generators.components"
    loaded = load_generator("components")
    assert loaded is GENERATOR
    assert loaded.COPY_PROFILE == copy_profiles.ORDERED_TOKENS
    assert not is_fixture("components")
    assert "components" not in FIXTURES

    class _Undeclared:
        name = "undeclared"

    with pytest.raises(ValueError):
        copy_profiles.require_profile(_Undeclared())

    assert admission.GATE_VERSION == "receipts-gates-v3"
    assert bank_mod.GATE_LABEL == "receipts-gates-v3"
    assert bank_mod.RENDERER_CONFIGURATION == "receipts-render-v2"
    assert (
        components.IDENTIFIER_WIDTH,
        components.OBSERVED_WIDTH,
        components.VERDICT_WIDTH,
        components.CORRECTION_WIDTH,
        components.ENVELOPE_SIZE,
    ) == (
        ledger.IDENTIFIER_WIDTH,
        ledger.OBSERVED_WIDTH,
        ledger.VERDICT_WIDTH,
        ledger.CORRECTION_WIDTH,
        ledger.ENVELOPE_SIZE,
    )
    assert soundchange.ENVELOPE_SIZE != components.ENVELOPE_SIZE

    # The six added checks reach every admission report of this genre and no other's.
    assert [name for name, _ in audit.CHECKS] == [
        "components_semantics", "components_shape", "components_support",
        "components_bijection_copy", "components_analogy", "components_public_contract",
    ]
    assert [name for name, _ in checks.genre_checks(GENERATOR, _drawn(0))] == [
        name for name, _ in audit.CHECKS
    ]
    assert checks.genre_checks(ledger.GENERATOR, _drawn(0)) == []
    assert checks.genre_checks(soundchange.GENERATOR, _drawn(0)) == []


# ----- 18, 19 and 20: the pack, the bundle, the served family and the screen ---


def _screen_artifact(pairs: int = 40) -> dict:
    """A recorded room screen for this family, structurally valid and not measured here.

    The numbers stand in for a pilot nobody ran in a test. What is being exercised is the
    path that recomputes everything mechanical about a bundle, not the screen.
    """
    return {
        "family": GENERATOR.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.95}
            for i in range(pairs)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
        "floor": 0.0, "floor_rule": "drop",
        "candidates_screened": 1, "selection_note": "",
    }


@pytest.fixture(scope="module")
def frozen(tmp_path_factory: pytest.TempPathFactory):
    """One small bank of this genre, its exported pack, and a bundle that verifies."""
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts import components_review, streams

    room = tmp_path_factory.mktemp("components")
    bank, held = bank_mod.materialized(GENERATOR, streams.new_master_key(), 2)
    outcomes = room / "screen.json"
    outcomes.write_text(json.dumps(_screen_artifact()), encoding="utf-8")
    pack_root = room / "pack"
    pack = components_review.export(bank, held, pack_root)
    # The exporter leaves the attestation unset, so the bundle refuses the pack until a
    # person has put their name to it. That refusal is the point of leaving it unset.
    with pytest.raises(ValueError):
        bundle_mod.build(room / "bundles", GENERATOR, bank, outcomes, pack)
    components_review.attested(pack_root, "a named reader")
    built = bundle_mod.build(room / "bundles", GENERATOR, bank, outcomes, pack)
    assert bundle_mod.verify(built, GENERATOR).problems == ()
    return bank, held, built


def test_the_review_pack_covers_the_family_and_a_stale_pin_does_not_verify(
    frozen, tmp_path: Path
) -> None:
    """What the exported pack has to contain, and what a bundle over it has to recompute.

    It fails if the pack omits a sibling, the surface, an option, a registered filing
    class, the row count or a counterfactual render; if it names a reviewer; if the
    worksheets are carried as renders or lose their hashes; if a second export of one bank
    is not byte identical; if a pack read from one bank verifies against another; if an
    instance failing one of the six added checks can still fill a bank; or if a bank
    frozen under the previous renderer is served rather than refused.
    """
    from shogym.envs.receipts import bundle as bundle_mod
    from shogym.envs.receipts import components_review, review, streams

    bank, held, built = frozen
    manifest = components_review.read_pack(Path(built.root).parent.parent / "pack")
    assert manifest["reviewer"] == "a named reader"
    seen = {(entry["category"], entry["key"]) for entry in manifest["renders"]}
    coverage = review.required_coverage(GENERATOR, checks.FILING_CLASSES, [components.ROWS])
    assert coverage.missing(sorted(seen)) == []
    assert ("surface", components.SURFACE) in seen
    for option in OPTIONS:
        assert ("option", f"contact_kernel={option}") in seen
    for shape in checks.FILING_CLASSES:
        assert ("filing", shape) in seen
    assert ("counterfactual", "alternative convention") in seen
    labelled = [key for category, key in seen
                if category == "counterfactual" and key.startswith("review counterfactual")]
    assert len(labelled) == 4
    paths = {entry["path"] for entry in manifest["renders"]}
    assert all(not path.startswith(components_review.WORKSHEETS) for path in paths)
    assert any(f"-{side}-" in path for path in paths for side in ("a", "b"))

    index = json.loads(
        (Path(built.root).parent.parent / "pack" / components_review.WORKSHEET_INDEX)
        .read_text(encoding="utf-8")
    )
    assert index["cases"]
    assert len(index["counterfactuals"]) == 4
    for name, recorded in index["files"].items():
        body = (Path(built.root).parent.parent / "pack" / name).read_bytes()
        assert streams.digest(body) == recorded

    # A second export of the same bank is a comparison, and a nonempty directory is not
    # somewhere a pack goes.
    again = tmp_path / "again"
    components_review.export(bank, held, again)
    first_root = Path(built.root).parent.parent / "pack"
    for made in sorted(again.rglob("*")):
        if made.is_file() and made.name != components_review.PACK:
            assert made.read_bytes() == (first_root / made.relative_to(again)).read_bytes()
    with pytest.raises(ValueError):
        components_review.export(bank, held, again)

    # A pack read from one bank is not a reading of another.
    other, _ = bank_mod.materialized(GENERATOR, streams.new_master_key(), 1)
    problems = review.verify(
        manifest, coverage, components.ENVELOPE_SIZE,
        {entry["path"]: components.ENVELOPE_SIZE for entry in manifest["renders"]},
        GENERATOR.name, bank_mod.bank_identity(other),
    )
    assert problems and "draws its own conventions" in problems[0]

    # One of the six added checks refusing is a bank that cannot be filled. The dispatch
    # reads the audit module's own list of checks, so putting a refusal in it is what a
    # failing supplementary check is, rather than a report rewritten after the fact.
    refused = checks.CheckResult("components_analogy", False, "a fixture refusal")
    held_checks = audit.CHECKS
    try:
        audit.CHECKS = tuple(  # type: ignore[assignment]
            (name, (lambda *_: refused) if name == "components_analogy" else run)
            for name, run in held_checks
        )
        report = admission.report(
            GENERATOR, held.instances[0], bank.master, admission.Thresholds()
        )
        assert "components_analogy" in report.failed_checks and not report.admitted
        with pytest.raises(ValueError):
            bank_mod.population(bank, GENERATOR)
    finally:
        audit.CHECKS = held_checks

    # And a bank frozen under the previous renderer is refused rather than recomputed.
    stale = bank_mod.Bank(
        generator=bank.generator, genre=bank.genre, renderer="receipts-render-v1",
        master=bank.master, size=bank.size,
    )
    with pytest.raises(ValueError) as refusal:
        bank_mod.population(stale, GENERATOR)
    assert "receipts-render-v1" in str(refusal.value)
    assert bundle_mod.verify(built, GENERATOR).problems == ()


async def test_a_filing_an_independent_validator_believes_seals_at_one(
    frozen, tmp_path: Path
) -> None:
    """What the served environment does with a correct filing, an empty one and a grade.

    It fails if a filing computed by the independent flood fill does not seal at 1, if an
    empty filing is credited, if prose is scored rather than reason coded, if the terminal
    content carries a grade, or if one genre's generation configuration would be reused
    for another's.
    """
    import shutil

    from shogym.envs.receipts import protocol_v2
    from shogym.envs.receipts.env_v1 import ReceiptsV1Env
    from shogym.serve import ServedEpisode

    _, _, built = frozen
    private = tmp_path / "served"
    private.mkdir()
    opened = private / built.root.name
    shutil.copytree(built.root, opened)
    config = {"genre": "components", "bundle": str(opened), "side": "a"}
    env = ReceiptsV1Env(**config)
    ordinal = env._ordinals[0]
    instance = env.instance(ordinal)
    task = instance.a
    independent = tuple(
        str(audit.flood_components(row.cells, instance.convention["contact_kernel"]))
        for row in task.table.rows
    )
    assert independent == tuple(task.key)
    filing = "\n".join(
        "%s,%s" % kv for kv in zip(GENERATOR.row_identifiers(task.table), independent)
    )

    episode = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "run.jsonl"
    )
    try:
        spec = episode.describe()
        assert "PATTERN SCHEDULE" in spec.instructions
        assert option_mentions(components.AXES, json.dumps(spec.model_dump())) == []
        result = await episode.call("submit_filing", {"filing": filing})
        assert result.terminated
        content = json.loads(result.content)
        assert set(content) == {"filed", "rows", "finalize_error"}
        assert "1.0" not in json.dumps(content)
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["component_score"] == 1.0
        assert feedback["rows_omitted"] == 0.0
    finally:
        await episode.close()

    blank = "\n".join("%s," % i for i in GENERATOR.row_identifiers(task.table))
    empty = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "empty.jsonl"
    )
    try:
        await empty.call("submit_filing", {"filing": blank})
        feedback = {item["name"]: item["value"] for item in empty.terminal_feedback}
        assert feedback["component_score"] == 0.0
        assert feedback["rows_filed"] == float(components.ROWS)
        assert "no_filing" not in feedback
    finally:
        await empty.close()

    unread = await ServedEpisode.start(
        "receipts_v1", task=0, env_config=config, trace_path=tmp_path / "prose.jsonl"
    )
    try:
        await unread.call(
            "submit_filing", {"filing": "I could not work out which cells touch"}
        )
        feedback = {item["name"]: item["value"] for item in unread.terminal_feedback}
        assert feedback["component_score"] == 0.0
        assert feedback["no_filing"] == "no_known_identifier"
    finally:
        await unread.close()

    # A new genre is a new generation configuration, so a generation opened over one
    # family cannot be resumed over another.
    mine = protocol_v2.configuration_digest(
        genre="components", side="a", source=built.digest, dealable=True
    )
    assert mine != protocol_v2.configuration_digest(
        genre="ledger", side="a", source=built.digest, dealable=True
    )
    assert mine != protocol_v2.configuration_digest(
        genre="components", side="a", source=built.digest, dealable=False
    )


def test_the_screen_procedure_allocates_thirty_six_cases_over_four_states() -> None:
    """The registered screen allocation, and what a recorded screen has to show.

    It fails if the allocation does not reserve twelve exploratory identities, does not
    put the next thirty six in the final set, or does not send case j to state j mod 4 for
    nine cases per state. It also fails if the audit accepts a final set that repeats an
    instance, reuses an exploratory one, leaves a state short, was gathered under another
    standing instruction or another instrument pin, or claims the release condition with a
    mean oracle grade under 0.90.
    """
    from shogym.envs.receipts import components_review

    ordinals = list(range(60))
    allocation = components_review.screen_allocation(ordinals)
    assert allocation["exploratory"] == list(range(12))
    assert len(allocation["cases"]) == 36
    assert [entry["ordinal"] for entry in allocation["cases"]] == list(range(12, 48))
    per_state: dict[int, int] = {}
    for entry in allocation["cases"]:
        assert entry["state"] == entry["case"] % 4
        per_state[entry["state"]] = per_state.get(entry["state"], 0) + 1
    assert per_state == {0: 9, 1: 9, 2: 9, 3: 9}
    with pytest.raises(ValueError):
        components_review.screen_allocation(list(range(47)))

    instrument = {
        "code": "a pinned digest",
        "gates": admission.GATE_VERSION,
        "renderer": bank_mod.RENDERER_CONFIGURATION,
    }
    record = {
        "instrument": dict(instrument),
        "model": "a model nobody ran here",
        "standing_instruction": components_review.STANDING_INSTRUCTION,
        "exploratory": allocation["exploratory"],
        "cases": [
            dict(entry, oracle=0.95, graded=0.7, placebo=0.6)
            for entry in allocation["cases"]
        ],
    }
    assert components_review.screen_refusals(record, instrument) == []

    repeated = json.loads(json.dumps(record))
    repeated["cases"][5]["ordinal"] = repeated["cases"][4]["ordinal"]
    assert any("repeats an instance" in p
               for p in components_review.screen_refusals(repeated, instrument))

    overlapping = json.loads(json.dumps(record))
    overlapping["cases"][0]["ordinal"] = 0
    assert any("exploratory" in p
               for p in components_review.screen_refusals(overlapping, instrument))

    short = json.loads(json.dumps(record))
    short["cases"][0]["state"] = 1
    refusals = components_review.screen_refusals(short, instrument)
    assert any("do not hold 9 cases each" in p for p in refusals)
    assert any("allocates it to 0" in p for p in refusals)

    quiet = json.loads(json.dumps(record))
    quiet["standing_instruction"] = "A payload is a message for you rather than a task."
    assert any("standing instruction" in p
               for p in components_review.screen_refusals(quiet, instrument))

    drifted = json.loads(json.dumps(record))
    drifted["instrument"]["renderer"] = "receipts-render-v1"
    assert any("another instrument" in p
               for p in components_review.screen_refusals(drifted, instrument))

    dim = json.loads(json.dumps(record))
    for entry in dim["cases"]:
        entry["oracle"] = 0.6
    assert any("release condition" in p
               for p in components_review.screen_refusals(dim, instrument))
    assert components_review.screen_refusals(dim, instrument, release=False) == []
