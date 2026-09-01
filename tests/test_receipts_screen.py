"""The room screen: what one graded receipt was worth, and what a floor is for.

Closed-form throughout. Every expected value is arithmetic on the three branch
scores, so a failure means the screen's logic drifted.
"""

from __future__ import annotations

import math

import pytest

from shogym.receipts import (
    Outcomes,
    ScreenRecord,
    contrasts,
    floored_ratio,
    screen,
    sd_influence,
)


#: Every screen states the sample it was required to have. These are unit tests over
#: tiny hand-checkable samples, so the bar is the smallest a screen may register.
_PAIRS = 2


def _outcomes(placebo, graded, oracle) -> Outcomes:
    return Outcomes(placebo=tuple(placebo), graded=tuple(graded), oracle=tuple(oracle))


def test_the_contrasts_are_the_two_differences_against_the_placebo() -> None:
    out = _outcomes([0.4, 0.5], [0.6, 0.5], [0.9, 1.0])
    gain, room = contrasts(out)
    assert list(gain) == pytest.approx([0.2, 0.0])
    assert list(room) == pytest.approx([0.5, 0.5])


def test_the_ratio_is_the_aggregate_gain_over_the_aggregate_room() -> None:
    # Per-pair ratios are 0.4 and 0.0, whose mean is 0.2. The estimand is not
    # that: it is 0.1 / 0.5, the ratio of the two pooled means.
    result = screen("f", _outcomes([0.4, 0.5], [0.6, 0.5], [0.9, 1.0]),
                    min_room=0.0, min_ratio=0.0, min_pairs=_PAIRS)
    assert result.gain == pytest.approx(0.1)
    assert result.room == pytest.approx(0.5)
    assert result.ratio == pytest.approx(0.2)
    assert result.verdict


def test_a_family_at_the_ceiling_has_no_room_and_fails() -> None:
    # Every pair is already solved without a receipt, so the oracle adds nothing.
    result = screen("saturated", _outcomes([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
                    min_room=0.05, min_ratio=0.1, min_pairs=_PAIRS)
    assert result.room == pytest.approx(0.0)
    assert result.saturated == pytest.approx(1.0)
    assert not result.room_pass
    assert not result.verdict
    assert math.isnan(result.ratio)
    assert any("nothing for a receipt to carry" in r for r in result.reasons)


def test_the_drop_rule_refuses_a_ratio_under_the_floor() -> None:
    result = screen("thin", _outcomes([0.5, 0.5], [0.55, 0.55], [0.56, 0.56]),
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
        screen("f", _outcomes([0.1], [0.2], [0.3]), min_room=0.0, min_ratio=0.0,
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
    result = screen("f", _outcomes([0.4], [0.6], [0.9]), min_room=0.1, min_ratio=0.3,
                    min_pairs=8)
    assert not result.verdict
    assert any("pairs against a required 8" in r for r in result.reasons)
    with pytest.raises(ValueError, match="fewer than two pairs"):
        screen("f", _outcomes([0.4], [0.6], [0.9]), min_room=0.1, min_ratio=0.3,
               min_pairs=1)


def test_the_thresholds_reach_the_printed_report() -> None:
    result = screen("f", _outcomes([0.2, 0.2], [0.4, 0.4], [0.8, 0.8]),
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
        return screen(
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
    room_edge = screen("f", _outcomes([0.3] * 36, [0.32] * 36, [0.35] * 36))
    assert room_edge.room < 0.05  # the float really is under the bar
    assert room_edge.room_pass and room_edge.verdict

    both = screen("f", _outcomes([0.4] * 36, [0.4125] * 36, [0.45] * 36))
    assert both.room < 0.05 and both.ratio < 0.25
    assert both.room_pass and both.ratio_pass and both.verdict

    # A sample genuinely under the bar still fails: the resolution is not a discount.
    under = screen("f", _outcomes([0.3] * 36, [0.31] * 36, [0.34] * 36))
    assert not under.room_pass and not under.verdict


def test_a_constant_sample_reports_no_spread_rather_than_a_non_number() -> None:
    """A correlation of a constant is NaN, and an SD of a deterministic sample is zero.

    `sd_influence` multiplied that NaN by the zero standard deviations it came from, so
    a passing deterministic screen reported a non-number where it promises a standard
    deviation.
    """
    result = screen("f", _outcomes([0.4] * 36, [0.6] * 36, [0.9] * 36))
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
    one = screen("f", rows)
    many = screen("f", rows, candidates_screened=1000, selection_note="best of a thousand")
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
                 "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
                for n in range(36)
            ],
            "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
            "floor": 0.0, "floor_rule": "drop",
            "candidates_screened": count, "selection_note": note,
        })

    assert record(1, "").dealable_selection
    assert not record(2, "best of two").dealable_selection
    assert not record(1000000, "best of a million").dealable_selection

