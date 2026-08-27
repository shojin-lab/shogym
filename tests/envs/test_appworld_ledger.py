"""The appworld port's backlog generator: the gates it enforces and the determinism it promises.

Offline and upstream-free. The generator is a pure function of a seed and a date, and everything
the measurement rests on is a property of what it produces: that the 64 conventions give 64
different answer keys, that no single choice is decided by one request, and that the same seed
gives the same backlog in any process on any machine.
"""

from __future__ import annotations

import datetime as dt
import zlib

import pytest

from shogym.envs.appworld import adapter, ledger, world

# The reference date all but a handful of the split's tasks carry, so a backlog built against it
# is the shape the roster is mostly made of.
REFERENCE = dt.date(2023, 5, 18)


@pytest.fixture(scope="module")
def backlog() -> ledger.Backlog:
    built = ledger.build_backlog(7, REFERENCE)
    assert built is not None
    return built


# ----- the convention space -----


def test_the_space_is_the_four_axes_crossed() -> None:
    assert len(ledger.CONVENTIONS) == 4 * 4 * 2 * 2 == 64
    assert len(set(ledger.CONVENTIONS)) == 64
    # Read off the option sets rather than written down twice: a design that grew an option would
    # otherwise have a test still asserting the old arity.
    assert len(ledger.ROLES) == 4
    assert len(ledger.BASIS_OPTIONS) == 4
    assert len(ledger.BOUNDARY_OPTIONS) == 2
    assert len(ledger.MISSING_OPTIONS) == 2


def test_a_count_on_a_printed_figure_is_where_the_boundary_rule_bites() -> None:
    # The printed ranges share their ends, which is the whole reason the axis exists. Away from a
    # printed figure the two rules agree; on one they do not.
    for cut, below, above in zip(ledger.CUTS, ledger.BANDS, ledger.BANDS[1:]):
        assert ledger.band_of(cut, "lower") == below
        assert ledger.band_of(cut, "upper") == above
        assert ledger.band_of(cut - 1, "lower") == ledger.band_of(cut - 1, "upper") == below
    assert ledger.band_of(0, "lower") == ledger.BANDS[0]
    assert ledger.band_of(999, "upper") == ledger.BANDS[-1]


# ----- the gates a shipped backlog has to clear -----


def test_every_convention_gives_a_different_answer_key(backlog: ledger.Backlog) -> None:
    # Two conventions with one key are two conventions no verdict vector can tell apart, however
    # well it is read.
    keys = {backlog.key(convention) for convention in ledger.CONVENTIONS}
    assert len(keys) == 64


def test_no_axis_is_decided_by_a_single_request(backlog: ledger.Backlog) -> None:
    # One request produces two signatures whatever its readout, so an axis carried by one request
    # is capped at two blocks however rich that request is. The undated request is the exception
    # by construction: it is the only place `missing` is read, so a flip on it moves exactly one
    # row and no backlog can do better.
    distinct, flips = backlog.audit()
    assert distinct
    assert min(flips[axis] for axis in ("anchor", "basis", "boundary")) >= 2
    assert flips["missing"] == 1


def test_one_verdict_vector_separates_every_option_on_every_axis(
    backlog: ledger.Backlog,
) -> None:
    """The property the whole instrument rests on, stated as a number.

    Take one convention as the drawn one and move a single axis through its options. Each move
    gives a verdict vector over the requests. If two options give the *same* vector, no reader can
    tell them apart from one grade however well it reads, and compliance on that axis is capped at
    two over its option count whatever the agent does. Counting the distinct vectors is exact, so
    the cap is checkable before any model is run rather than argued about after."""
    axes = {
        "anchor": ledger.ROLES,
        "basis": ledger.BASIS_OPTIONS,
        "boundary": ledger.BOUNDARY_OPTIONS,
        "missing": ledger.MISSING_OPTIONS,
    }
    positions = {c: i for i, c in enumerate(ledger.CONVENTIONS)}
    for drawn in ledger.CONVENTIONS:
        key = backlog.keys[positions[drawn]]
        for axis, options in axes.items():
            vectors = {
                tuple((backlog.keys[positions[drawn._replace(**{axis: option})]] == key).tolist())
                for option in options
            }
            # Full resolution: as many distinct vectors as the axis has options, so a reader that
            # evaluates the readouts can separate all of them.
            assert len(vectors) == len(options), (drawn, axis)


def test_a_backlog_holds_the_requests_it_says_it_does(backlog: ledger.Backlog) -> None:
    assert len(backlog.requests) == ledger.DATED + ledger.UNDATED == 29
    assert sum(1 for request in backlog.requests if request.dates is None) == ledger.UNDATED
    assert len({request.reference for request in backlog.requests}) == 29
    assert backlog.keys.shape == (64, 29)


def test_a_reference_carries_its_position_and_nothing_about_its_dates(
    backlog: ledger.Backlog,
) -> None:
    # References are assigned after the requests are shuffled, so they run in printed order and
    # say nothing a reader cannot already see. What matters is that they say nothing else: a
    # reference that sorted by a recorded date, or that separated the requests built around the
    # printed figures from the rest, would let a reader group the backlog without evaluating a
    # single readout.
    references = [request.reference for request in backlog.requests]
    assert references == sorted(references)
    for role in ledger.ROLES:
        dated = [r.dates[role] for r in backlog.requests if r.dates is not None]
        assert dated != sorted(dated)
        assert dated != sorted(dated, reverse=True)


def test_the_project_description_carries_the_week_and_the_closures_and_nothing_else(
    backlog: ledger.Backlog,
) -> None:
    # A seeded row that named a band would set the leak to one on every task at once, silently.
    description = backlog.description
    assert description.startswith("Worked: Monday to Friday. Closed: ")
    assert len(backlog.closures) == 3
    for closure in backlog.closures:
        assert closure.isoformat() in description
        assert closure.weekday() < 5
    for band in ledger.BANDS:
        assert band not in description
    for section in ledger.SECTIONS:
        assert section not in description


# ----- determinism -----


def test_the_same_seed_gives_the_same_backlog() -> None:
    first = ledger.build_backlog(11, REFERENCE)
    second = ledger.build_backlog(11, REFERENCE)
    assert first is not None and second is not None
    assert first.requests == second.requests
    assert first.closures == second.closures
    assert (first.keys == second.keys).all()


def test_different_seeds_give_different_backlogs() -> None:
    first = ledger.build_backlog(11, REFERENCE)
    second = ledger.build_backlog(12, REFERENCE)
    assert first is not None and second is not None
    assert first.requests != second.requests


def test_a_seeded_row_carries_dates_and_nothing_a_reader_could_read_a_band_off(
    backlog: ledger.Backlog,
) -> None:
    for request in backlog.requests:
        text = world.request_description(request)
        if request.dates is None:
            assert text == ""
            continue
        assert sorted(line.split(":")[0] for line in text.splitlines()) == sorted(ledger.ROLES)
        for band in ledger.BANDS:
            assert band not in text


# ----- the served roster and the frozen table beside it -----


def test_the_manifest_is_a_roster_of_distinct_tasks() -> None:
    served = adapter.task_ids()
    assert served
    assert len(set(served)) == len(served)
    assert all(task_id.rsplit("_", 1)[-1] in {"1", "2", "3"} for task_id in served)


@pytest.mark.parametrize("position", [0, 1, 100, 200, 317])
def test_a_task_in_the_manifest_really_has_an_admissible_backlog(position: int) -> None:
    # The manifest is the roster and the generator is what admitted it; a task that is listed and
    # has no backlog is the two disagreeing, which would stop a run three tasks in.
    task_id = adapter.task_ids()[position]
    built = ledger.build_backlog(zlib.crc32(task_id.encode()), REFERENCE)
    assert built is not None, task_id
    distinct, flips = built.audit()
    assert distinct
    assert min(flips[axis] for axis in ("anchor", "basis", "boundary")) >= 2


def test_the_frozen_pass_count_table_covers_the_dated_requests() -> None:
    from shogym.envs.appworld.payload import pass_counts

    counts = pass_counts()
    # One weight per possible number of passing dated requests, zero through all of them.
    assert len(counts) == ledger.DATED + 1
    assert sum(counts) > 0
    assert all(weight >= 0 for weight in counts)
    # Not a tidy distribution, and that is the point of drawing from it: a real receipt says
    # everything failed often enough that a payload drawn from a symmetric distribution would be
    # identifiable by its count on sight.
    assert counts[0] / sum(counts) > 0.1


def test_the_table_is_the_one_this_generator_produces(backlog: ledger.Backlog) -> None:
    # Enumerated the same way the frozen table was, on one backlog: the shape has to be the
    # roster's own, or a drawn receipt's count comes from a distribution real receipts do not
    # have.
    marginal = ledger.pass_count_marginal([backlog])
    assert len(marginal) == ledger.DATED + 1
    assert sum(marginal) == len(ledger.CONVENTIONS)
    # The reference convention agrees with itself on every dated request.
    assert marginal[-1] >= 1
