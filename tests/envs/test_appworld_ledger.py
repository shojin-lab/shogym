"""The appworld port's backlog generator: the gates it enforces and the determinism it promises.

Offline and upstream-free. The generator is a pure function of a seed and a date, and everything
the measurement rests on is a property of what it produces: that the 64 conventions give 64
different answer keys, that no single choice is decided by one request, and that the same seed
gives the same backlog in any process on any machine.

The seed and the date are production's own. The seed is the task identity, which these tests can
compute; the date is the task's own specification, which lives in a 134 MB corpus these tests do
not download, so it is committed beside them as a table and checked against the real corpus by
``test_appworld_table.py``. A roster accepted at a date no episode is served is a roster nobody
tested.
"""

from __future__ import annotations

import datetime as dt
import os
import zlib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest

from shogym.envs.appworld import adapter, ledger, world

# Deselected from the per-push CI step with the rest of AppWorld's tests, and run by the job that
# runs nightly and on request.
pytestmark = pytest.mark.appworld

# The reference date all but a handful of the split's tasks carry, so a backlog built against it
# is the shape the roster is mostly made of. It is what the *generator's* own properties are
# stated at, below, and it is deliberately not what the roster is accepted against: 31 of
# the 318 served tasks carry another date, and production reads each task's own.
REFERENCE = dt.date(2023, 5, 18)

#: Task id to the datetime its specification carries, read off the pinned corpus and committed
#: beside these tests. These tests are offline and a corpus is 134 MB of download, so the dates
#: production reads have to arrive some other way; ``test_appworld_table.py`` checks this table
#: against the live corpus wherever there is one, which is what stops it drifting into fiction.
TASK_DATES = Path(__file__).with_name("appworld_task_dates.tsv")


def _dates() -> Dict[str, dt.date]:
    """The committed table, as the dates production would build backlogs against."""
    table: Dict[str, dt.date] = {}
    for line in TASK_DATES.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        task_id, moment = line.split("\t")
        # Parsed exactly as `AppWorldEnv._backlog` parses it, so what these tests feed the
        # generator is what a run feeds it and not a second reading of the same string.
        table[task_id] = dt.datetime.fromisoformat(moment).date()
    return table


def _production_inputs(task_id: str) -> Tuple[int, dt.date]:
    """The two arguments production draws a task's backlog from: its seed and its own date."""
    return zlib.crc32(task_id.encode()), _dates()[task_id]


def _off_reference() -> Tuple[str, ...]:
    """The served tasks whose date is not the one the rest share.

    A backlog built at :data:`REFERENCE` is right for 287 of the 318 and wrong for these."""
    return tuple(task for task, date in _dates().items() if date != REFERENCE)


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


def test_moving_one_choice_at_a_time_separates_every_option_on_every_axis(
    backlog: ledger.Backlog,
) -> None:
    """The admission gate's own property, stated exactly and not one word further.

    Hold three choices at the drawn convention and move the fourth through its options. Each move
    gives a verdict vector over the requests, and this asserts they are all different, which is
    what stops an axis being carried by a single request: one request produces two signatures
    whatever its readout, and compliance on such an axis is capped at two over its option count
    whatever the agent does.

    **This is a statement about one axis at a time, and it is not identifiability.** Nothing here
    says the drawn convention can be recovered from a receipt; three choices are already correct
    in every vector compared. What one receipt actually leaves standing is a different quantity,
    it is enumerated below, and it is not one."""
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
            assert len(vectors) == len(options), (drawn, axis)


def test_one_receipt_narrows_the_convention_and_does_not_name_it(
    backlog: ledger.Backlog,
) -> None:
    """The quantity the one-axis gate above does *not* establish, enumerated.

    A receipt is the vector of bits saying, per request, whether the drawn convention would have
    written what the agent wrote. Conventions sharing a vector are conventions no reader can
    separate however well it reads. The map is **not** injective and cannot be made so: some pairs
    agree on every request the world can supply, and measuring at 28, 34, 40 and 48 dated requests
    found no backlog at any length whose map is one-to-one.

    So the claim this instrument supports is matching the drawn convention closely, not naming it,
    and these are the numbers that say by how much: 64 conventions down to a handful."""
    profile = ledger.posterior_profile(backlog, ledger.REFERENCE_CONVENTION)
    assert not profile.distinct == len(ledger.CONVENTIONS), "injectivity is not claimed"
    # The floors are the roster's measured behaviour, not aspirations: over the served roster the
    # means are about 48 distinct vectors, a largest class of about 8, and a posterior of about 3.
    assert profile.distinct >= 32
    assert profile.largest <= 16
    assert profile.mean_size <= 8.0
    # And a receipt is worth something: 64 down to a few is most of what there was to learn.
    assert profile.mean_size < len(ledger.CONVENTIONS) / 8


@pytest.mark.parametrize("position", [0, 40, 120, 240])
def test_the_narrowing_holds_across_the_roster(position: int) -> None:
    # At the sampled task's own date, which is what a run builds it against. A floor asserted at
    # one hard-coded date is a floor for a backlog no episode is ever served.
    task_id = adapter.task_ids()[position]
    built = ledger.build_backlog(*_production_inputs(task_id))
    assert built is not None
    profile = ledger.posterior_profile(built, ledger.REFERENCE_CONVENTION)
    assert profile.distinct >= 32, (task_id, profile)
    assert profile.mean_size <= 8.0, (task_id, profile)


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


def test_the_committed_dates_are_one_per_served_task_and_no_others() -> None:
    """The table and the manifest are one roster, or the tests below are testing another one.

    Cheap, unconditional, and the thing that makes a committed table safe to reason from offline:
    an entry missing here is a task no roster test could have covered, and an entry that is here
    and not in the manifest is a task no run serves."""
    assert tuple(_dates()) == adapter.task_ids()


def _admissible(task_id: str) -> None:
    """Assert one task clears the gate that put it in the manifest, at its own date.

    The manifest is the roster and the generator is what admitted it; a task that is listed and
    has no backlog is the two disagreeing, which would stop a run at whatever position it sits
    at. Admission is two claims: a backlog exists, and no axis of it is carried by a single
    request."""
    built = ledger.build_backlog(*_production_inputs(task_id))
    assert built is not None, task_id
    distinct, flips = built.audit()
    assert distinct, task_id
    assert min(flips[axis] for axis in ("anchor", "basis", "boundary")) >= 2, task_id


@pytest.mark.parametrize("task_id", _off_reference())
def test_a_task_that_does_not_carry_the_reference_date_is_still_admissible(task_id: str) -> None:
    """Every served task whose date is its own, all 31 of them, and not a sample of them.

    A backlog built at the single reference date, which is the date 287 of the 318 happen to
    carry, is a property of a backlog no episode is served for the other 31. The whole
    subpopulation is covered here because it is small; the whole roster is covered by the test
    below, which is minutes."""
    _admissible(task_id)


@pytest.mark.skipif(
    os.environ.get("SHOGYM_CHECK_ROSTER") != "1",
    reason="four minutes of computation; set SHOGYM_CHECK_ROSTER=1 to run all 318",
)
def test_every_task_in_the_manifest_has_an_admissible_backlog_at_its_own_date() -> None:
    """All 318, at production's inputs, with nothing sampled.

    Opt-in for the same reason the frozen table's full check is (see
    ``tests/envs/test_appworld_table.py``): a backlog takes about three quarters of a second to
    draw, so this is four minutes of computation whose result only moves when the manifest, the
    seed rule, the generator's constants or the pinned corpus's dates move. That is when to run
    it:

        SHOGYM_CHECK_ROSTER=1 pytest tests/envs/test_appworld_ledger.py

    The default run covers the five sampled positions and the whole off-reference subpopulation,
    which is every task whose inputs the cheap test could get wrong."""
    for task_id in adapter.task_ids():
        _admissible(task_id)


@pytest.mark.parametrize("position", [0, 1, 100, 200, 317])
def test_a_task_in_the_manifest_really_has_an_admissible_backlog(position: int) -> None:
    _admissible(adapter.task_ids()[position])


def test_the_frozen_pass_count_table_covers_the_dated_requests() -> None:
    from shogym.envs.appworld.payload import pass_counts

    counts = pass_counts()
    # Enumerated over the tasks that are actually served, and no others. A table built before the
    # roster was settled carries the excluded tasks' outcomes and is a distribution the run never
    # produces, which is exactly what a drawn receipt must not be drawn from.
    assert sum(counts) == len(adapter.task_ids()) * len(ledger.CONVENTIONS)
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


def test_the_draw_cannot_depend_on_the_machine_that_runs_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The docstring promises a backlog is the same on any machine. This is what makes that true.

    A float64 dot product on the greedy cover's scoring path is summed by BLAS in whatever order
    its kernel picks, which is enough to make one served task's backlog exist on arm64 and not on
    x86_64: the seeded ledger an episode is graded against becomes a property of the CPU.

    Checked by construction rather than by running anywhere else, because two machines is not a
    thing a test suite has. If every array on the scoring path is an integer array, no kernel's
    summation order can reach the result: integer addition is associative and exact. So this runs
    a real draw and asserts the dtypes of what actually flowed through it."""
    seen: list[tuple] = []
    original = ledger._cover_gains

    def recording(pool, cover, scale):
        result = original(pool, cover, scale)
        seen.append((pool.dtype, cover.dtype, result.dtype, type(scale)))
        return result

    monkeypatch.setattr(ledger, "_cover_gains", recording)
    built = ledger.build_backlog(zlib.crc32(b"machine-independence"), dt.date(2023, 5, 18))
    assert built is not None

    assert seen, "the scoring path never ran, so this asserted nothing"
    for pool_dtype, cover_dtype, gains_dtype, scale_type in seen:
        for dtype in (pool_dtype, cover_dtype, gains_dtype):
            assert np.issubdtype(dtype, np.integer), f"{dtype} on the scoring path is not integral"
        # The common denominator is a Python int, so `3 * scale // (1 + cover)` stays integral
        # rather than being promoted to float by a stray numpy float scalar.
        assert scale_type is int
