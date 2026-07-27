"""yc_bench's verifier is a pure function over the core-owned terminal verdict (RFC 008).

No yc-bench install and no serving here — build the verdict dict ``finalize`` would read off
the sim and check the episode feedback. ``score_verdict`` is yc_bench-free, so these run in the
offline core suite. The terminal-state gate (credit a solvent state only on a genuine
``horizon_end`` / bankruptcy) is asserted directly.
"""

from __future__ import annotations

from hgym.envs.yc_bench.env_v1 import score_verdict

_UNSET = object()


def _default_terminal_reason(survived: bool, horizon_reached: bool):
    """Mirror ``read_final_state``: bankruptcy wins over horizon; solvent + pre-horizon = None."""
    if not survived:
        return "bankruptcy"
    if horizon_reached:
        return "horizon_end"
    return None


def _verdict(
    *,
    survived: bool = True,
    final_funds_cents: int = 12_345_678,
    tasks_succeeded: int = 3,
    tasks_failed: int = 1,
    horizon_reached: bool = True,
    terminal_reason: object = _UNSET,
) -> dict:
    if terminal_reason is _UNSET:
        terminal_reason = _default_terminal_reason(survived, horizon_reached)
    return {
        "survived": survived,
        "final_funds_cents": final_funds_cents,
        "tasks_succeeded": tasks_succeeded,
        "tasks_failed": tasks_failed,
        "horizon_reached": horizon_reached,
        "terminal_reason": terminal_reason,
    }


def _episode(fb) -> dict:
    return {f.name: f.value for f in fb.episode}


def test_solvent_horizon_end_is_success() -> None:
    ep = _episode(score_verdict(_verdict(), terminated=True))
    assert ep["reward"] == 12_345_678.0
    assert ep["final_funds_cents"] == 12_345_678.0
    assert ep["survived"] is True
    assert ep["horizon_reached"] is True
    assert ep["success"] is True
    assert ep["tasks_succeeded"] == 3.0
    assert ep["tasks_failed"] == 1.0


def test_solvent_pre_horizon_submit_is_premature_zero() -> None:
    # Anti-gaming (the terminal-state gate): `submit` is callable at any time. A solvent
    # submission *before* the horizon (terminal_reason is None) must NOT bank the current funds —
    # it scores a premature zero, otherwise submitting on turn one would beat most real runs.
    ep = _episode(
        score_verdict(
            _verdict(survived=True, final_funds_cents=20_000_000, horizon_reached=False),
            terminated=True,
        )
    )
    assert ep["reward"] == 0.0
    assert ep["final_funds_cents"] == 0.0
    assert ep["survived"] is False
    assert ep["success"] is False


def test_bankruptcy_is_terminal_and_credits_negative_funds() -> None:
    # Bankruptcy is a genuine terminal outcome (terminal_reason == "bankruptcy"): the negative
    # final funds are credited (no gaming benefit — reward is negative), not survived.
    ep = _episode(
        score_verdict(
            _verdict(survived=False, final_funds_cents=-500, horizon_reached=False),
            terminated=True,
        )
    )
    assert ep["reward"] == -500.0
    assert ep["survived"] is False
    assert ep["success"] is False


def test_no_feedback_until_terminated() -> None:
    assert score_verdict(_verdict(), terminated=False).episode == []


def test_missing_verdict_scores_premature_zero() -> None:
    # No scoring submission reached the seal (an abort, or a fail-closed finalize): no verdict.
    ep = _episode(score_verdict(None, terminated=True))
    assert ep["reward"] == 0.0
    assert ep["survived"] is False
    assert ep["success"] is False
    assert ep["final_funds_cents"] == 0.0
    assert "tasks_succeeded" not in ep  # nothing to report on a premature end


def test_aborted_verdict_scores_premature_zero() -> None:
    # The serve layer synthesizes `{"correct": False, "aborted": True}` for a `terminate` abort;
    # it has no terminal_reason, so the gate scores it zero.
    ep = _episode(
        score_verdict({"correct": False, "aborted": True}, terminated=True)
    )
    assert ep["reward"] == 0.0
    assert ep["survived"] is False
    assert ep["success"] is False


def test_malformed_verdict_fields_do_not_crash() -> None:
    bad = {
        "terminal_reason": "horizon_end",  # genuine terminal state → exercise coercion
        "final_funds_cents": "oops",
        "survived": "yes",
        "horizon_reached": None,
        "tasks_succeeded": None,
        "tasks_failed": "nan",
    }
    ep = _episode(score_verdict(bad, terminated=True))
    # funds coerces to 0.0; NaN/None counts coerce to 0.0 — none of it raises.
    assert ep["reward"] == 0.0
    assert ep["tasks_succeeded"] == 0.0
    assert ep["tasks_failed"] == 0.0
