"""The appworld port's scorer and its three payload classes: what each carries, and what none does.

Offline and upstream-free. Everything here is a pure function of a backlog, a key and a filing,
so the whole score range is reachable without a world and the invariants can be checked over many
submissions rather than argued about.

The properties defended here are the ones that make two arms one comparison. The three classes are
the same length on the wire. None of them prints an expected value, a rule, or the name of a
choice. The digest is a function of the agent's own submission alone. And the drawn receipt's
verdicts do not move when the real ones do, which is the executable form of "it carries nothing".
"""

from __future__ import annotations

import datetime as dt
import json
import re

import pytest

from shogym.envs.appworld import ledger, payload, world
from shogym.envs.appworld.scorer import (
    AMBIGUOUS,
    ASSERTION,
    LEDGER,
    NOT_FILED,
    NOT_SET,
    OTHER,
    PINNED,
    Key,
    draw_key,
    score,
)

REFERENCE = dt.date(2023, 5, 18)
TASK = "5238afc_1"
CHECKS = [("aw.001", True), ("aw.002", False), ("aw.003", None)]


@pytest.fixture(scope="module")
def backlog() -> ledger.Backlog:
    built = ledger.build_backlog(7, REFERENCE)
    assert built is not None
    return built


def filing(backlog: ledger.Backlog, key: Key, *, correct: int = 29, **overrides):
    """A filing that gets the first ``correct`` requests right and the rest wrong."""
    expected = backlog.key(key.convention)
    lines = []
    for position, (request, band) in enumerate(zip(backlog.requests, expected)):
        wrong = next(b for b in ledger.BANDS if b != band)
        lines.append(f"{request.reference}, {band if position < correct else wrong}")
    fields = {
        "filed": True,
        "rows": 1,
        "lines": tuple(lines),
        "section": key.slots[0],
        "color": key.slots[1],
        "unit": key.slots[2],
        "priority": key.slots[3],
    }
    fields.update(overrides)
    return world.Filing(**fields)


def scored(backlog: ledger.Backlog, key: Key, **kwargs):
    return score(
        backlog=backlog, key=key, filing=filing(backlog, key, **kwargs), assertions=CHECKS
    )


# ----- the key -----


def test_a_key_is_a_function_of_the_task_and_the_draw_and_nothing_else() -> None:
    assert draw_key(TASK, 0) == draw_key(TASK, 0)
    assert draw_key(TASK, 0) != draw_key(TASK, 1)
    assert draw_key(TASK, 0) != draw_key("5238afc_3", 0)
    key = draw_key(TASK, 0)
    assert key.convention in ledger.CONVENTIONS
    for slot, drawn in zip(world.SLOTS, key.slots):
        assert drawn in slot.options


def test_the_draw_spreads_over_the_whole_space() -> None:
    # Uniform over the conventions, not tilted toward the one a reader reaches for unprompted.
    # The tilt is what makes the measurement possible: the graded arm is usually told its default
    # was wrong, and a draw that followed the tilt would stop telling it that.
    from collections import Counter

    drawn = Counter(draw_key(f"task_{n}", 0).convention for n in range(3200))
    assert len(drawn) == 64
    assert min(drawn.values()) > 20 and max(drawn.values()) < 90
    assert drawn[ledger.REFERENCE_CONVENTION] < 120


# ----- scoring -----


def test_a_filing_that_matches_the_drawn_convention_scores_everything() -> None:
    built = ledger.build_backlog(7, REFERENCE)
    assert built is not None
    key = draw_key(TASK, 0)
    verdicts = scored(built, key)
    assert verdicts.ledger_fraction == 1.0
    assert verdicts.pinned_fraction == 1.0
    assert verdicts.exercise_fraction == 1.0
    assert verdicts.parse_fraction == 1.0


def test_a_filing_that_matches_a_different_convention_does_not(backlog: ledger.Backlog) -> None:
    key = draw_key(TASK, 0)
    other = next(c for c in ledger.CONVENTIONS if c != key.convention)
    verdicts = score(
        backlog=backlog,
        key=key,
        filing=filing(backlog, Key(convention=other, slots=key.slots)),
        assertions=CHECKS,
    )
    assert 0.0 <= verdicts.ledger_fraction < 1.0
    # The stored slots were right, and they are scored on their own: a ledger the agent got wrong
    # says nothing about them.
    assert verdicts.pinned_fraction == 1.0


def test_nothing_filed_is_every_item_wrong_rather_than_every_item_dropped(
    backlog: ledger.Backlog,
) -> None:
    # The denominator is the whole item set. Dropping what the agent did not attempt would let a
    # contrast between two arms be carried entirely by how many lines each arm bothered to write.
    verdicts = score(
        backlog=backlog, key=draw_key(TASK, 0), filing=world.EMPTY_FILING, assertions=CHECKS
    )
    assert verdicts.ledger_fraction == 0.0
    assert verdicts.pinned_fraction == 0.0
    assert verdicts.exercise_fraction == 0.0
    ledger_rows = [item for item in verdicts.items if item.kind == LEDGER]
    assert len(ledger_rows) == 29
    assert {item.observed for item in ledger_rows} == {NOT_FILED}
    assert {item.observed for item in verdicts.items if item.kind == PINNED} == {NOT_SET}


def test_the_filing_rate_is_reported_separately(backlog: ledger.Backlog) -> None:
    key = draw_key(TASK, 0)
    partial = filing(backlog, key)._replace(lines=tuple(filing(backlog, key).lines[:10]))
    verdicts = score(backlog=backlog, key=key, filing=partial, assertions=CHECKS)
    assert verdicts.exercise_fraction == pytest.approx(10 / 29)
    assert verdicts.ledger_fraction == pytest.approx(10 / 29)


@pytest.mark.parametrize(
    "line, observed",
    [
        ("{ref} , {band}", "{band}"),
        ("{ref}: {band}", "{band}"),
        ("  {BAND} -- {ref}  ", "{band}"),
        ("{ref}, Routine, Urgent", AMBIGUOUS),
        ("{ref}, Overdue", OTHER),
    ],
)
def test_the_parser_is_lenient_about_shape_and_strict_about_the_answer(
    backlog: ledger.Backlog, line: str, observed: str
) -> None:
    # A line counts if it names one request and one band, in any order, in any case, with any
    # separator. Two bands is a wrong answer rather than a parse failure: the agent acted.
    key = draw_key(TASK, 0)
    request = backlog.requests[0]
    band = backlog.key(key.convention)[0]
    written = line.format(ref=request.reference, band=band, BAND=band.upper())
    verdicts = score(
        backlog=backlog,
        key=key,
        filing=world.EMPTY_FILING._replace(filed=True, rows=1, lines=(written,)),
        assertions=CHECKS,
    )
    row = next(item for item in verdicts.items if item.check_id == request.reference)
    assert row.observed == observed.format(band=band)
    assert verdicts.parse_fraction == 1.0


def test_a_line_naming_no_request_is_a_shape_error_and_never_a_lesson(
    backlog: ledger.Backlog,
) -> None:
    key = draw_key(TASK, 0)
    lines = ("here is the log:", f"{backlog.requests[0].reference}, Routine")
    verdicts = score(
        backlog=backlog,
        key=key,
        filing=world.EMPTY_FILING._replace(filed=True, rows=1, lines=lines),
        assertions=CHECKS,
    )
    assert verdicts.parse_fraction == pytest.approx(0.5)
    assert verdicts.exercise_fraction == pytest.approx(1 / 29)


def test_a_request_named_twice_is_one_answer_and_a_duplicate(backlog: ledger.Backlog) -> None:
    # A second line cannot revise the first. Revising after the fact is what the seal exists to
    # prevent, and a parser that allowed it would let it back in through the description.
    key = draw_key(TASK, 0)
    request = backlog.requests[0]
    band = backlog.key(key.convention)[0]
    other = next(b for b in ledger.BANDS if b != band)
    verdicts = score(
        backlog=backlog,
        key=key,
        filing=world.EMPTY_FILING._replace(
            filed=True,
            rows=1,
            lines=(f"{request.reference}, {band}", f"{request.reference}, {other}"),
        ),
        assertions=CHECKS,
    )
    row = next(item for item in verdicts.items if item.check_id == request.reference)
    assert row.observed == band
    assert row.passed


def test_a_slot_left_at_the_api_s_own_default_reads_as_unexercised(
    backlog: ledger.Backlog,
) -> None:
    # The world writes a value into these columns whether the agent chose one or not, so "left
    # alone" and "chosen wrong" arrive looking the same. Every default is struck from its option
    # set and read back as unexercised: scoring one as a wrong guess would count an omission as an
    # attempt and put the slot's filing rate inside its compliance rate.
    key = draw_key(TASK, 0)
    defaults = {slot.check_id: slot.default for slot in world.SLOTS}
    verdicts = scored(backlog, key, color="charcoal", priority="medium")
    rows = {item.check_id: item for item in verdicts.items if item.kind == PINNED}
    assert defaults["fr.label.color"] == "charcoal"
    assert defaults["fr.log.priority"] == "medium"
    assert rows["fr.label.color"].observed == NOT_SET
    assert rows["fr.log.priority"].observed == NOT_SET
    assert not rows["fr.label.color"].passed and not rows["fr.log.priority"].passed
    # And a default is never in its own option set, so it cannot be what the draw asked for.
    for slot in world.SLOTS:
        assert slot.default not in slot.options


def test_a_stored_slot_is_scored_on_itself(backlog: ledger.Backlog) -> None:
    key = draw_key(TASK, 0)
    verdicts = scored(backlog, key, color=None, unit="minutes")
    rows = {item.check_id: item for item in verdicts.items if item.kind == PINNED}
    assert rows["fr.label.color"].observed == NOT_SET
    assert not rows["fr.label.color"].passed
    assert rows["fr.log.unit"].observed == "minutes"
    assert rows["fr.log.unit"].passed == (key.slots[2] == "minutes")


# ----- the three payload classes -----

CLASSES = (payload.GRADED, payload.DIGEST, payload.DRAWN)


@pytest.mark.parametrize("correct", [0, 7, 29])
def test_the_three_classes_are_the_same_length_on_the_wire(
    backlog: ledger.Backlog, correct: int
) -> None:
    # The match is what makes two arms one comparison: an agent that reads a short answer in one
    # arm and a long one in the other has been told which arm it is in before reading a word of
    # either. Measured as the answer is encoded, because that is the form the agent reads.
    key = draw_key(TASK, 0)
    verdicts = scored(backlog, key, correct=correct)
    rendered = [
        payload.render(task_id=TASK, verdicts=verdicts, cell=cell) for cell in CLASSES
    ]
    assert len({len(text.encode()) for text in rendered}) == 1
    assert len({len(json.dumps(text)) for text in rendered}) == 1


def test_the_match_survives_a_world_whose_values_are_not_ascii(
    backlog: ledger.Backlog,
) -> None:
    # The corpus carries cells with zero-width code points in them, and a value that reached a
    # payload would change its byte count and, under a JSON encoder that escapes, change it by
    # more in one class than another. Nothing from the world reaches a payload: a value outside
    # the published option set is rendered as a token from the payload's own vocabulary.
    key = draw_key(TASK, 0)
    hostile = "​Ińbound—x"
    verdicts = scored(backlog, key, section=hostile, color=hostile)
    section = next(item for item in verdicts.items if item.check_id == "fr.log.section")
    assert section.observed == OTHER
    rendered = [
        payload.render(task_id=TASK, verdicts=verdicts, cell=cell) for cell in CLASSES
    ]
    for text in rendered:
        assert text.isascii()
        assert hostile not in text
    assert len({len(text.encode()) for text in rendered}) == 1


@pytest.mark.parametrize("cell", CLASSES)
def test_no_payload_names_an_expected_value_a_rule_or_a_choice(
    backlog: ledger.Backlog, cell: str
) -> None:
    key = draw_key(TASK, 0)
    text = payload.render(task_id=TASK, verdicts=scored(backlog, key, correct=3), cell=cell)
    # Not the drawn convention's options, not the axis names they belong to, and not the words
    # the paragraph leaves open. The verdict column is the only place reference information
    # appears at all.
    for axis in ("anchor", "basis", "boundary", "missing"):
        assert axis not in text
    for option in (*ledger.ROLES, *ledger.BASIS_OPTIONS, *ledger.BOUNDARY_OPTIONS):
        assert not re.search(rf"\b{option}\b", text), option
    for band in backlog.key(key.convention):
        # A band may appear as something the agent wrote; it may never appear as an answer, so a
        # payload for a filing that wrote nothing has no band in it at all.
        pass
    empty = payload.render(
        task_id=TASK,
        verdicts=score(
            backlog=backlog, key=key, filing=world.EMPTY_FILING, assertions=CHECKS
        ),
        cell=cell,
    )
    for band in ledger.BANDS:
        assert band not in empty


@pytest.mark.parametrize("cell", CLASSES)
def test_the_row_count_is_a_task_constant(backlog: ledger.Backlog, cell: str) -> None:
    # Fixed before the agent acts and identical for every possible submission, so it carries no
    # bits about the score.
    key = draw_key(TASK, 0)
    header = {
        payload.render(task_id=TASK, verdicts=scored(backlog, key, correct=n), cell=cell)
        .splitlines()[2]
        for n in (0, 5, 29)
    }
    assert len(header) == 1
    assert header.pop() == "checks: 36   (assertions 3, ledger 29, pinned 4)"


def test_the_receipt_says_pass_or_fail_and_the_digest_says_neither(
    backlog: ledger.Backlog,
) -> None:
    key = draw_key(TASK, 0)
    verdicts = scored(backlog, key, correct=15)
    receipt = payload.render(task_id=TASK, verdicts=verdicts, cell=payload.GRADED)
    digest = payload.render(task_id=TASK, verdicts=verdicts, cell=payload.DIGEST)
    assert receipt.count(payload.PASS) + receipt.count(payload.FAIL) == len(verdicts.items)
    assert payload.PASS not in digest and payload.FAIL not in digest
    # Every column but the last is the same in both.
    stripped = [
        [line[:-8] for line in text.splitlines()[3:]] for text in (receipt, digest)
    ]
    assert stripped[0] == stripped[1]


def test_the_digest_is_a_function_of_the_submission_alone(backlog: ledger.Backlog) -> None:
    # Both the task identity and the submission are already in the agent's own transcript, so a
    # digest carries no information about the key by construction, and no reading of its wording
    # is needed to establish that. Two different keys over one submission give one digest column.
    first, second = draw_key(TASK, 0), draw_key(TASK, 3)
    assert first.convention != second.convention
    submission = filing(backlog, first, correct=11)
    columns = {
        _last_column(
            payload.render(
                task_id=TASK,
                verdicts=score(
                    backlog=backlog, key=key, filing=submission, assertions=CHECKS
                ),
                cell=payload.DIGEST,
            )
        )
        for key in (first, second)
    }
    assert len(columns) == 1


def test_a_known_wrong_and_a_correct_submission_render_the_same_shaped_digest(
    backlog: ledger.Backlog,
) -> None:
    key = draw_key(TASK, 0)
    lengths = {
        len(
            payload.render(
                task_id=TASK, verdicts=scored(backlog, key, correct=n), cell=payload.DIGEST
            ).encode()
        )
        for n in (0, 29)
    }
    assert len(lengths) == 1


# ----- the drawn receipt -----


def test_the_drawn_verdicts_do_not_move_when_the_real_ones_do(
    backlog: ledger.Backlog,
) -> None:
    # The executable form of "the count carries nothing": re-render the payload against a
    # different submission and neither the number of passes nor which rows carry them moves.
    key = draw_key(TASK, 0)
    columns = {
        _last_column(
            payload.render(
                task_id=TASK, verdicts=scored(backlog, key, correct=n), cell=payload.DRAWN
            )
        )
        for n in (0, 9, 29)
    }
    assert len(columns) == 1


def test_the_drawn_verdicts_do_not_move_when_the_key_does(backlog: ledger.Backlog) -> None:
    first, second = draw_key(TASK, 0), draw_key(TASK, 3)
    submission = filing(backlog, first, correct=11)
    columns = {
        _last_column(
            payload.render(
                task_id=TASK,
                verdicts=score(
                    backlog=backlog, key=key, filing=submission, assertions=CHECKS
                ),
                cell=payload.DRAWN,
            )
        )
        for key in (first, second)
    }
    assert len(columns) == 1


def test_a_drawn_receipt_keeps_the_base_task_s_own_checks(backlog: ledger.Backlog) -> None:
    # An assertion says something about the base task and nothing about the key, so there is
    # nothing in it to destroy and a drawn verdict on one would be a lie about the world rather
    # than a payload that carries no rule.
    key = draw_key(TASK, 0)
    verdicts = scored(backlog, key)
    text = payload.render(task_id=TASK, verdicts=verdicts, cell=payload.DRAWN)
    rows = [line for line in text.splitlines() if f"   {ASSERTION}   " in line]
    assert len(rows) == len(CHECKS)
    assert rows[0].endswith(payload.PASS + "   ")
    assert rows[1].strip().endswith(payload.FAIL)


def test_the_drawn_pass_count_comes_from_the_frozen_table(backlog: ledger.Backlog) -> None:
    # Over many tasks the counts a drawn receipt states have to look like the counts real ones
    # state, or the payload is identifiable by its count alone.
    key = draw_key(TASK, 0)
    verdicts = scored(backlog, key)
    dated = {item.check_id for item in verdicts.items if item.dated}
    counts = []
    for task in range(300):
        text = payload.render(task_id=f"t{task}", verdicts=verdicts, cell=payload.DRAWN)
        passing = sum(
            1
            for line in text.splitlines()[4:]
            if line.split()[1] == LEDGER
            and line.split()[2] in dated
            and line.strip().endswith(payload.PASS)
        )
        counts.append(passing)
    assert min(counts) == 0
    assert max(counts) == ledger.DATED
    table = payload.pass_counts()
    expected = sum(n * w for n, w in enumerate(table)) / sum(table)
    assert abs(sum(counts) / len(counts) - expected) < 2.0


def _last_column(text: str) -> tuple:
    """The rendered marks, one per row, and nothing around them."""
    return tuple(line.split()[-1] for line in text.splitlines()[4:] if line.strip())
