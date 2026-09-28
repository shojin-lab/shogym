"""The room screen: what one graded receipt was worth, and what a floor is for.

Closed-form throughout. Every expected value is arithmetic on the three branch
scores, so a failure means the screen's logic drifted.

Two versions of the bars are registered. `screen` is the registered one and `screen_v1`
the first, which every record without a `bars` field was made under. The first version's
tests are kept as they were, calling `screen_v1`, because a record made under it has to
rerun to the verdict it had.
"""

from __future__ import annotations

import importlib
import math

import pytest

from shogym.receipts import (
    Outcomes,
    ScreenRecord,
    ScreenResult,
    ScreenResultV1,
    contrasts,
    floored_ratio,
    screen,
    screen_v1,
    sd_influence,
)


#: Every screen states the sample it was required to have. These are unit tests over
#: tiny hand-checkable samples, so the bar is the smallest a screen may register.
_PAIRS = 2


def _outcomes(placebo, graded, oracle, ideal=None) -> Outcomes:
    return Outcomes(
        placebo=tuple(placebo),
        graded=tuple(graded),
        oracle=tuple(oracle),
        ideal=tuple(ideal) if ideal is not None else (),
    )


def test_the_contrasts_are_the_two_differences_against_the_placebo() -> None:
    out = _outcomes([0.4, 0.5], [0.6, 0.5], [0.9, 1.0])
    gain, room = contrasts(out)
    assert list(gain) == pytest.approx([0.2, 0.0])
    assert list(room) == pytest.approx([0.5, 0.5])


def test_the_ratio_is_the_aggregate_gain_over_the_aggregate_room() -> None:
    # Per-pair ratios are 0.4 and 0.0, whose mean is 0.2. The estimand is not
    # that: it is 0.1 / 0.5, the ratio of the two pooled means.
    result = screen_v1("f", _outcomes([0.4, 0.5], [0.6, 0.5], [0.9, 1.0]),
                    min_room=0.0, min_ratio=0.0, min_pairs=_PAIRS)
    assert result.gain == pytest.approx(0.1)
    assert result.room == pytest.approx(0.5)
    assert result.ratio == pytest.approx(0.2)
    assert result.verdict


def test_a_family_at_the_ceiling_has_no_room_and_fails() -> None:
    # Every pair is already solved without a receipt, so the oracle adds nothing.
    result = screen_v1("saturated", _outcomes([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
                    min_room=0.05, min_ratio=0.1, min_pairs=_PAIRS)
    assert result.room == pytest.approx(0.0)
    assert result.saturated == pytest.approx(1.0)
    assert not result.room_pass
    assert not result.verdict
    assert math.isnan(result.ratio)
    assert any("nothing for a receipt to carry" in r for r in result.reasons)


def test_the_drop_rule_refuses_a_ratio_under_the_floor() -> None:
    result = screen_v1("thin", _outcomes([0.5, 0.5], [0.55, 0.55], [0.56, 0.56]),
                    min_room=0.0, min_ratio=0.1, min_pairs=_PAIRS, floor=0.15,
                    floor_rule="drop")
    assert math.isnan(result.ratio)
    assert result.floor_binds
    assert not result.verdict
    assert any("not a number" in r for r in result.reasons)


def test_the_clamp_rule_divides_by_the_floor_instead() -> None:
    assert floored_ratio(0.06, 0.06, 0.15, "clamp") == pytest.approx(0.4)
    assert floored_ratio(0.06, 0.06, 0.15, "none") == pytest.approx(1.0)
    assert math.isnan(floored_ratio(0.06, 0.06, 0.15, "drop"))
    # a denominator at zero has no ratio under any rule, floor or no floor
    for rule in ("drop", "clamp", "none"):
        assert math.isnan(floored_ratio(0.0, 0.0, 0.0, rule))


def test_an_unknown_floor_rule_is_refused() -> None:
    with pytest.raises(ValueError):
        floored_ratio(1.0, 1.0, 0.1, "shrug")
    with pytest.raises(ValueError):
        screen_v1("f", _outcomes([0.1], [0.2], [0.3]), min_room=0.0, min_ratio=0.0,
               min_pairs=1, floor_rule="shrug")


def test_the_influence_sd_is_the_delta_method_quantity() -> None:
    # With the two contrasts uncorrelated and the ratio at one, the influence SD
    # is sqrt(s_x^2 + s_y^2) over the room.
    got = sd_influence(s_x=0.3, s_y=0.4, r_xy=0.0, ratio=1.0, room=0.5)
    assert got == pytest.approx(math.sqrt(0.09 + 0.16) / 0.5)


def test_the_screen_needs_matching_branches_and_at_least_one_pair() -> None:
    with pytest.raises(ValueError):
        Outcomes(placebo=(), graded=(), oracle=())
    with pytest.raises(ValueError):
        Outcomes(placebo=(0.1, 0.2), graded=(0.3,), oracle=(0.4, 0.5))


def test_outcomes_read_back_from_rows() -> None:
    out = Outcomes.from_rows(
        [{"placebo": 0.1, "graded": 0.2, "oracle": 0.3},
         {"placebo": 0.4, "graded": 0.5, "oracle": 0.6}]
    )
    assert out.n_pairs == 2
    assert out.graded == (0.2, 0.5)


def test_a_sample_below_the_registered_bar_does_not_pass() -> None:
    """One pair is a number, not a result: its influence SD is not even defined."""
    result = screen_v1("f", _outcomes([0.4], [0.6], [0.9]), min_room=0.1, min_ratio=0.3,
                    min_pairs=8)
    assert not result.verdict
    assert any("pairs against a required 8" in r for r in result.reasons)
    with pytest.raises(ValueError, match="fewer than two pairs"):
        screen_v1("f", _outcomes([0.4], [0.6], [0.9]), min_room=0.1, min_ratio=0.3,
               min_pairs=1)


def test_the_thresholds_reach_the_printed_report() -> None:
    result = screen_v1("f", _outcomes([0.2, 0.2], [0.4, 0.4], [0.8, 0.8]),
                    min_room=0.3, min_ratio=0.9, min_pairs=_PAIRS)
    text = "\n".join(result.lines())
    assert "0.9000" in text
    assert "REJECTED" in text
    assert not result.ratio_pass


def test_the_verdict_does_not_depend_on_the_order_the_rows_were_written_in() -> None:
    """The bootstrap addresses positions, so two serializations of one multiset of
    paired observations drew different resamples and could report different intervals.
    The room interval's lower bound decides dealability, which made the verdict a fact
    about the order rows happened to be written in rather than about the sample.
    """
    rows = [(1.0, 0.5, 0.0)] * 2 + [(0.4, 0.475, 0.55)] * 34
    moved = list(rows)
    moved.insert(7, moved.pop(0))
    moved.insert(34, moved.pop(0))

    def run(order):
        return screen_v1(
            "f",
            _outcomes(
                [r[0] for r in order], [r[1] for r in order], [r[2] for r in order]
            ),
        )

    first, second = run(rows), run(moved)
    assert [r[0] for r in rows] != [r[0] for r in moved]
    for field in ("room", "gain", "ratio", "room_low", "room_high", "verdict"):
        assert getattr(first, field) == getattr(second, field), field


def test_a_sample_exactly_at_a_registered_bar_clears_it() -> None:
    """The bars are registered as inclusive, so the comparison has to be.

    Room and ratio are means of binary floats. A sample that sits exactly at a bar in
    decimal arrives a fraction under it: 36 pairs of oracle 0.35 against placebo 0.30
    average to 0.049999999999999996, which prints as 0.0500 and used to fail a bar of
    0.05, and the same pairs put the ratio a fraction under 0.25. One registered
    resolution decides both, here and again when a bundle re-verifies.
    """
    # The oracle and gap bars are held at zero here so that the verdict is decided by
    # the two bars this is about. Their own edge is held in the tests below.
    edges = {"min_oracle": 0.0, "min_learning_gap": 0.0}
    room_edge = screen_v1("f", _outcomes([0.3] * 36, [0.32] * 36, [0.35] * 36), **edges)
    assert room_edge.room < 0.05  # the float really is under the bar
    assert room_edge.room_pass and room_edge.verdict

    both = screen_v1("f", _outcomes([0.4] * 36, [0.4125] * 36, [0.45] * 36), **edges)
    assert both.room < 0.05 and both.ratio < 0.25
    assert both.room_pass and both.ratio_pass and both.verdict

    # A sample genuinely under the bar still fails: the resolution is not a discount.
    under = screen_v1("f", _outcomes([0.3] * 36, [0.31] * 36, [0.34] * 36), **edges)
    assert not under.room_pass and not under.verdict


def test_a_constant_sample_reports_no_spread_rather_than_a_non_number() -> None:
    """A correlation of a constant is NaN, and an SD of a deterministic sample is zero.

    `sd_influence` multiplied that NaN by the zero standard deviations it came from, so
    a passing deterministic screen reported a non-number where it promises a standard
    deviation.
    """
    result = screen_v1("f", _outcomes([0.4] * 36, [0.6] * 36, [0.9] * 36))
    assert result.verdict
    assert math.isnan(result.r_xy)  # the correlation genuinely is not defined
    assert math.isfinite(result.sd_influence)
    assert result.sd_influence == pytest.approx(0.0, abs=1e-12)


def test_a_selected_screen_is_scored_and_is_not_deal_evidence() -> None:
    """Selection is disclosed, unadjusted, and therefore not dealable.

    Nothing corrects for it: the interval, the bars and the diagnostic verdict are
    identical for one candidate and for a thousand, so a selected winner clears this
    arithmetic on exactly what one candidate clears it on. Until an adjustment is
    registered, that means a selected record is scored and printed and is not evidence
    a bundle may be frozen on. `dealable_selection` is where the two part company.
    """
    rows = _outcomes([0.4] * 36, [0.6] * 36, [0.9] * 36)
    one = screen_v1("f", rows)
    many = screen_v1("f", rows, candidates_screened=1000, selection_note="best of a thousand")
    # The arithmetic really is unchanged, which is the reason for the rule.
    assert one.verdict and many.verdict
    assert (one.room, one.ratio, one.room_low) == (many.room, many.ratio, many.room_low)
    assert any("no selection adjustment" in reason for reason in many.reasons)
    assert not any("no selection adjustment" in reason for reason in one.reasons)

    def record(count: int, note: str) -> ScreenRecord:
        return ScreenRecord.from_payload({
            "family": "f", "model": "m",
            "task_seeds": [str(n) for n in range(36)],
            "pairs": [
                {"instance": f"t{n:02d}", "filing": f"f{n:02d}",
                 "placebo": 0.4, "graded": 0.6, "oracle": 0.9, "ideal": 0.82}
                for n in range(36)
            ],
            "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
            "min_oracle": 0.90, "min_learning_gap": 0.10,
            "floor": 0.0, "floor_rule": "drop",
            "candidates_screened": count, "selection_note": note,
        })

    assert record(1, "").dealable_selection
    assert not record(2, "best of two").dealable_selection
    assert not record(1000000, "best of a million").dealable_selection



def test_the_oracle_bar_refuses_a_family_whose_oracle_copies_did_not_execute() -> None:
    """The room the ratio divides by has to be a room this model actually took.

    FAILS IF a screen passes while the oracle copies averaged under 0.90 on the
    held-out task. A low oracle level is as consistent with copies that could not carry
    out a rule they were handed as with a family that leaves nothing to carry, and the
    ratio reads the same either way.
    """
    weak = screen_v1("f", _outcomes([0.2] * 36, [0.5] * 36, [0.7] * 36, [0.9] * 36))
    assert weak.room_pass and weak.ratio_pass
    assert not weak.oracle_pass and not weak.verdict
    assert any("told the rule" in reason for reason in weak.reasons)
    strong = screen_v1("f", _outcomes([0.2] * 36, [0.5] * 36, [0.95] * 36, [0.9] * 36))
    assert strong.oracle_pass and strong.verdict
    assert strong.oracle == pytest.approx(0.95)


def test_the_gap_bar_refuses_a_family_already_at_its_receipt_s_own_ceiling() -> None:
    """A graded level at the ideal has nothing left for a later step to read better.

    FAILS IF a screen passes while the graded level sits within 0.10 of the level a
    perfect reader of the SAME receipts reaches, or while the paired interval on that
    gap reaches zero. The other three bars cannot see this: a family at its own ceiling
    clears them emphatically, which is exactly the saturation a chain would then be run
    on for nothing.
    """
    saturated = screen_v1(
        "f", _outcomes([0.45] * 36, [0.96] * 36, [0.99] * 36, [1.0] * 36)
    )
    assert saturated.room_pass and saturated.ratio_pass and saturated.oracle_pass
    assert not saturated.gap_pass and not saturated.verdict
    assert any("nothing left for a later step" in r for r in saturated.reasons)

    room_to_read = screen_v1(
        "f", _outcomes([0.45] * 36, [0.65] * 36, [0.99] * 36, [0.82] * 36)
    )
    assert room_to_read.gap == pytest.approx(0.17)
    assert room_to_read.gap_pass and room_to_read.verdict

    # The point estimate is not enough on its own: the paired interval has to clear
    # zero, and a sample that straddles it does not establish the gap.
    straddling = screen_v1(
        "f",
        _outcomes(
            [0.45] * 36,
            [0.6 if n % 2 else 0.95 for n in range(36)],
            [0.99] * 36,
            [0.9 if n % 2 else 0.6 for n in range(36)],
        ),
    )
    assert not straddling.gap_pass and not straddling.verdict


def test_the_first_version_s_bars_are_the_numbers_they_were() -> None:
    """Five bars in the first version, and each is the number it was registered at.

    FAILS IF a first-version bar moves, or if one of them stops being part of what
    `registered` means for a first-version result. A record made under that version is
    judged against these numbers for as long as it is read.
    """
    from shogym.receipts.screen import (
        REGISTERED_MIN_LEARNING_GAP,
        REGISTERED_MIN_PAIRS,
        V1_MIN_ORACLE,
        V1_MIN_RATIO,
        V1_MIN_ROOM,
    )

    assert (V1_MIN_ROOM, V1_MIN_RATIO, REGISTERED_MIN_PAIRS) == (0.05, 0.25, 36)
    assert (V1_MIN_ORACLE, REGISTERED_MIN_LEARNING_GAP) == (0.90, 0.10)
    rows = _outcomes([0.45] * 36, [0.65] * 36, [0.99] * 36, [0.82] * 36)
    assert screen_v1("f", rows).registered
    assert not screen_v1("f", rows, min_oracle=0.5).registered
    assert not screen_v1("f", rows, min_learning_gap=0.01).registered


# ----- the registered version: three intervals, and no oracle bar -----


def _cycled(n: int, *values: float) -> list[float]:
    """``n`` values that cycle through ``values``, so a sample has a spread to resample."""
    return [values[i % len(values)] for i in range(n)]


def test_the_registered_bars_are_the_interval_version() -> None:
    """The registered version is the second, and it has two numbers and three tests.

    FAILS IF new records stop being made under the second version, the sample or the gap
    floor moves, the interval stops being 90 percent over 2,000 resamples, or a name that
    says a superseded bar is registered comes back.
    """
    from shogym.receipts.screen import (
        BARS_V1,
        BARS_V2,
        BOOTSTRAP_DRAWS,
        BOOTSTRAP_MASS,
        REGISTERED_BARS,
        REGISTERED_MIN_LEARNING_GAP,
        REGISTERED_MIN_PAIRS,
    )

    assert (BARS_V1, BARS_V2) == ("receipts-screen-v1", "receipts-screen-v2")
    assert REGISTERED_BARS == BARS_V2
    assert (REGISTERED_MIN_PAIRS, REGISTERED_MIN_LEARNING_GAP) == (36, 0.10)
    assert (BOOTSTRAP_DRAWS, BOOTSTRAP_MASS) == (2000, 0.90)
    # The modules themselves: `from shogym.receipts import screen` is the screen function,
    # which never had these names, so asking it would pass whatever the module held.
    for name in ("shogym.receipts.screen", "shogym.receipts"):
        module = importlib.import_module(name)
        for gone in ("REGISTERED_MIN_ROOM", "REGISTERED_MIN_RATIO", "REGISTERED_MIN_ORACLE"):
            assert not hasattr(module, gone), (name, gone)
    rows = _outcomes([0.2] * 36, [0.5] * 36, [0.6] * 36, [0.9] * 36)
    assert screen("f", rows).registered
    assert not screen("f", rows, min_pairs=40).registered
    assert not screen("f", rows, min_learning_gap=0.05).registered


def test_the_oracle_level_is_reported_and_decides_nothing() -> None:
    """An oracle at 0.60 is under the first version's 0.90 and is no bar in the registered one.

    FAILS IF the registered screen refuses a family on its oracle level, or stops
    reporting that level.
    """
    rows = _outcomes([0.2] * 36, [0.5] * 36, [0.6] * 36, [0.9] * 36)
    registered = screen("f", rows)
    assert isinstance(registered, ScreenResult)
    assert registered.verdict and registered.reasons == ()
    assert registered.oracle == pytest.approx(0.6)
    assert "ORACLE executed level       0.6000   (reported, not a bar)" in registered.lines()
    first = screen_v1("f", rows)
    assert isinstance(first, ScreenResultV1)
    assert not first.oracle_pass and not first.verdict


def test_the_feedback_effect_has_to_be_shown_above_zero_and_no_ratio_is_asked() -> None:
    """The interval on the feedback effect replaces the ratio's bar of 0.25.

    FAILS IF a small effect the whole sample shows is refused for being a small fraction
    of the room, or if an effect whose interval reaches zero is admitted.
    """
    small = _outcomes([0.4] * 36, [0.42] * 36, [0.9] * 36, [0.9] * 36)
    shown = screen("f", small)
    assert shown.effect == pytest.approx(0.02)
    assert shown.effect_low > 0.0 and shown.effect_pass and shown.verdict
    assert not hasattr(shown, "ratio")
    # The first version refuses the same sample, on a ratio of 0.04 against its 0.25.
    assert not screen_v1("f", small).ratio_pass

    noisy = _outcomes([0.4] * 36, _cycled(36, 0.6, 0.2), [0.9] * 36, [0.9] * 36)
    unshown = screen("f", noisy)
    assert unshown.effect_low <= 0.0 < unshown.effect_high
    assert not unshown.effect_pass and not unshown.verdict
    assert unshown.room_pass and unshown.gap_pass
    assert any(
        "does not establish that the copy with feedback did better" in reason
        for reason in unshown.reasons
    )


def test_the_room_has_to_be_shown_above_zero_and_has_no_point_floor() -> None:
    """A room of 0.03 the whole sample shows clears the registered screen.

    FAILS IF the first version's floor of 0.05 comes back as a point bar, or if a room
    whose interval reaches zero is admitted.
    """
    narrow = _outcomes([0.3] * 36, [0.32] * 36, [0.33] * 36, [0.9] * 36)
    shown = screen("f", narrow)
    assert shown.room == pytest.approx(0.03)
    assert shown.room_pass and shown.verdict
    assert not screen_v1("f", narrow, min_ratio=0.0).room_pass

    noisy = _outcomes([0.4] * 36, [0.6] * 36, _cycled(36, 0.8, 0.0), [0.9] * 36)
    unshown = screen("f", noisy)
    assert unshown.room_low <= 0.0
    assert not unshown.room_pass and not unshown.verdict
    assert any("any room at all" in reason for reason in unshown.reasons)


def test_the_gap_holds_its_floor_and_its_interval_under_the_registered_bars() -> None:
    """The gap floor is unchanged: at least 0.10, inclusively, with the interval above zero.

    FAILS IF a gap under 0.10 passes, a gap exactly at 0.10 in decimal fails on the last
    bit of a float, or a gap whose interval reaches zero passes on its mean.
    """
    under = screen("f", _outcomes([0.3] * 36, [0.6] * 36, [0.95] * 36, [0.69] * 36))
    assert not under.gap_pass and not under.verdict
    assert any("nothing left for a later step" in reason for reason in under.reasons)

    edge = screen("f", _outcomes([0.3] * 36, [0.6] * 36, [0.95] * 36, [0.7] * 36))
    assert edge.gap < 0.10  # the float really is under the floor
    assert edge.gap_pass and edge.verdict

    straddling = screen(
        "f",
        _outcomes(
            [0.1] * 36,
            [0.2] * 18 + [0.98] * 18,
            [1.0] * 36,
            [0.9] * 18 + [0.5] * 18,
        ),
    )
    assert straddling.gap >= 0.10
    assert straddling.gap_low <= 0.0
    assert not straddling.gap_pass and not straddling.verdict
    assert any("the gap interval reaches" in reason for reason in straddling.reasons)


def test_the_three_intervals_are_paired_over_one_set_of_resampled_tasks() -> None:
    """Each resample draws one set of tasks and reads all three contrasts off it.

    FAILS IF the intervals stop being 90 percent percentile intervals over 2,000 resamples
    seeded from the sample rule, or stop sharing their resamples. Drawn this way the room
    and gap intervals are the first version's own, to the last bit.
    """
    import numpy as np

    from shogym.receipts.screen import (
        BOOTSTRAP_DRAWS,
        REGISTERED_MIN_PAIRS,
        canonical_order,
        learning_gap,
    )

    rows = _outcomes(
        [(i * 7 % 10) / 20 for i in range(36)],
        [0.5 + (i * 3 % 10) / 20 for i in range(36)],
        [0.6 + (i * 5 % 9) / 20 for i in range(36)],
        [0.95 - (i % 4) / 40 for i in range(36)],
    )
    result = screen("f", rows)
    ordered = canonical_order(rows)
    gain, room = contrasts(ordered)
    picks = np.random.default_rng(REGISTERED_MIN_PAIRS).integers(
        0, 36, size=(BOOTSTRAP_DRAWS, 36)
    )
    for values, low, high in (
        (gain, result.effect_low, result.effect_high),
        (room, result.room_low, result.room_high),
        (learning_gap(ordered), result.gap_low, result.gap_high),
    ):
        means = values[picks].mean(axis=1)
        assert (low, high) == (
            float(np.quantile(means, 0.05)),
            float(np.quantile(means, 0.95)),
        )
    first = screen_v1("f", rows)
    assert (first.room_low, first.room_high) == (result.room_low, result.room_high)
    assert (first.gap_low, first.gap_high) == (result.gap_low, result.gap_high)


def test_the_registered_verdict_does_not_depend_on_the_order_rows_were_written_in() -> None:
    rows = [(0.4, 0.6 if i % 3 else 0.35, 0.9 - (i % 5) / 20, 0.95) for i in range(36)]
    moved = list(rows)
    moved.insert(7, moved.pop(0))
    moved.insert(30, moved.pop(2))

    def run(order):
        return screen("f", _outcomes(*([row[k] for row in order] for k in range(4))))

    first, second = run(rows), run(moved)
    for field in ("effect", "effect_low", "effect_high", "room", "room_low", "room_high",
                  "gap", "gap_low", "gap_high", "verdict"):
        assert getattr(first, field) == getattr(second, field), field


# ----- a record says which bars it was made under -----


def _record(rows: list[tuple[float, float, float, float]], **changes) -> dict:
    """A record made under the registered bars over ``rows``, with ``changes`` applied."""
    payload = {
        "family": "f",
        "model": "m",
        "task_seeds": [str(n) for n in range(len(rows))],
        "pairs": [
            {"instance": f"t{n:02d}", "filing": f"f{n:02d}", "placebo": placebo,
             "graded": graded, "oracle": oracle, "ideal": ideal}
            for n, (placebo, graded, oracle, ideal) in enumerate(rows)
        ],
        "bars": "receipts-screen-v2",
        "min_pairs": 36,
        "min_learning_gap": 0.10,
        "candidates_screened": 1,
        "selection_note": "",
    }
    payload.update(changes)
    return payload


def _first_record(rows: list[tuple[float, float, float, float]], **changes) -> dict:
    """The same run as a record made under the first version, which names no bars."""
    payload = {k: v for k, v in _record(rows).items() if k != "bars"}
    payload.update(
        min_room=0.05, min_ratio=0.25, min_oracle=0.90, floor=0.0, floor_rule="drop"
    )
    payload.update(changes)
    return payload


_STEADY = [(0.4, 0.6, 0.95, 0.82)] * 36


def test_a_record_states_the_bars_it_was_made_under_and_is_rerun_under_them() -> None:
    registered = ScreenRecord.from_payload(_record(_STEADY))
    assert registered.bars == "receipts-screen-v2"
    assert registered.registered and registered.overrides() == []
    assert registered.dealable_selection
    assert (registered.min_room, registered.min_ratio, registered.min_oracle) == (None,) * 3
    assert (registered.floor, registered.floor_rule) == (None, None)
    assert isinstance(registered.result("f"), ScreenResult)

    first = ScreenRecord.from_payload(_first_record(_STEADY))
    assert first.bars == "receipts-screen-v1"
    assert first.registered and first.overrides() == []
    rerun = first.result("f")
    assert isinstance(rerun, ScreenResultV1)
    assert rerun == screen_v1("f", _outcomes(*([row[k] for row in _STEADY] for k in range(4))))

    moved = ScreenRecord.from_payload(_record(_STEADY, min_learning_gap=0.05))
    assert not moved.registered
    assert moved.overrides() == ["min_learning_gap=0.05 against the registered 0.1"]


@pytest.mark.parametrize(
    "change,expected",
    [
        ({"min_room": 0.05}, "names exactly"),
        ({"min_ratio": 0.25}, "names exactly"),
        ({"min_oracle": 0.90}, "names exactly"),
        ({"floor": 0.0, "floor_rule": "drop"}, "names exactly"),
        ({"bars": "receipts-screen-v1"}, "names no bars"),
        ({"bars": "receipts-screen-v3"}, "not 'receipts-screen-v3'"),
        ({"bars": 2}, "not 2"),
        ({"min_learning_gap": "0.1"}, "is a number"),
        ({"candidates_screened": 3}, "says nothing about the selection"),
    ],
)
def test_a_registered_record_carries_its_own_fields_and_no_others(change, expected) -> None:
    """A first-version bar on a registered record is a field nothing reads, and is refused."""
    with pytest.raises(ValueError, match=expected):
        ScreenRecord.from_payload(_record(_STEADY, **change))


@pytest.mark.parametrize("field", ["bars", "min_pairs", "min_learning_gap",
                                   "candidates_screened", "selection_note"])
def test_every_registered_decision_input_is_required(field) -> None:
    with pytest.raises(ValueError, match="names exactly"):
        ScreenRecord.from_payload(
            {k: v for k, v in _record(_STEADY).items() if k != field}
        )


def test_a_record_cannot_hold_bars_its_version_does_not_have() -> None:
    from shogym.receipts import ScreenRun

    run = ScreenRun.from_payload(
        {k: v for k, v in _record(_STEADY).items()
         if k in ("family", "model", "task_seeds", "pairs")}
    )
    common = dict(run=run, min_pairs=36, min_learning_gap=0.10, candidates_screened=1,
                  selection_note="")
    with pytest.raises(ValueError, match="states no room"):
        ScreenRecord(bars="receipts-screen-v2", min_room=0.05, **common)
    with pytest.raises(ValueError, match="states its room"):
        ScreenRecord(bars="receipts-screen-v1", **common)
    with pytest.raises(ValueError, match="made under one of"):
        ScreenRecord(bars="receipts-screen-v0", **common)


def test_a_first_version_record_keeps_the_verdict_it_had() -> None:
    """Each record is judged by its own version, and the two versions can disagree.

    FAILS IF a record made under the first bars is read under the registered ones, in
    either direction: a small effect the first version refused on its ratio stays
    refused, and a noisy effect it admitted stays admitted.
    """
    small = [(0.4, 0.42, 0.95, 0.9)] * 36
    assert not ScreenRecord.from_payload(_first_record(small)).result("f").verdict
    assert ScreenRecord.from_payload(_record(small)).result("f").verdict

    # Twenty five tasks the copy with feedback solved and eleven it did not: a ratio of
    # 0.31 over a room of 0.30, and a feedback effect whose interval reaches zero.
    noisy = [(0.6, 1.0, 0.9, 1.0)] * 25 + [(0.6, 0.0, 0.9, 1.0)] * 11
    kept = ScreenRecord.from_payload(_first_record(noisy)).result("f")
    assert isinstance(kept, ScreenResultV1)
    assert kept.ratio >= 0.25 and kept.verdict
    now = ScreenRecord.from_payload(_record(noisy)).result("f")
    assert isinstance(now, ScreenResult)
    assert now.effect_low <= 0.0 and not now.verdict
