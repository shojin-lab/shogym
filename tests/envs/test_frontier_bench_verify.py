"""frontier_bench's verifier scores from core-owned terminal evidence (RFC 008, seal contract).

No Docker and no serving here — build a :class:`TerminalEvidence` by hand and check the episode
feedback :func:`score_evidence` derives from ``evidence.verdict``. The reward is produced by the
env's ``finalize`` hook over the container end-state (the seal makes it non-forgeable and
one-shot); this scorer only *parses* it, so it is Docker-free and runs in the offline core suite.
"""

from __future__ import annotations

from shogym.envs.frontier_bench.env_v1 import score_evidence
from shogym.serve.lifecycle import TerminalEvidence, fail_closed_verdict


def _verdict_evidence(
    *, reward: float, reward_found: bool = True, source: str = "explicit_tool"
) -> TerminalEvidence:
    """The evidence `finalize` returns for a real verifier run."""
    return TerminalEvidence(
        source=source,  # type: ignore[arg-type]
        status="ok",
        verdict={
            "reward": reward,
            "success": reward >= 1.0,
            "reward_found": reward_found,
            "artifacts_collected": {"/app/output/x.csv": True},
        },
    )


def _episode(fb) -> dict:
    return {f.name: f.value for f in fb.episode}


def test_passing_verifier_is_success() -> None:
    ep = _episode(score_evidence(_verdict_evidence(reward=1.0)))
    assert ep["reward"] == 1.0
    assert ep["success"] is True
    assert ep["verified"] is True
    assert ep["reward_found"] is True


def test_failing_verifier_is_not_success() -> None:
    ep = _episode(score_evidence(_verdict_evidence(reward=0.0)))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert ep["verified"] is True


def test_horizon_evidence_is_scored_like_a_done_verdict() -> None:
    # on_horizon = finalize_current_state: the horizon runs the verifier over the container
    # end-state, so its evidence carries a real reward and is `verified` just like an explicit
    # `done`.
    ep = _episode(score_evidence(_verdict_evidence(reward=1.0, source="horizon")))
    assert ep["reward"] == 1.0 and ep["success"] is True and ep["verified"] is True


def test_abort_evidence_scores_zero_unverified() -> None:
    # `terminate` (abort) — the base synthesizes a no-score abort verdict; no verifier ran.
    abort = TerminalEvidence(
        source="abort", status="ok", verdict={"correct": False, "aborted": True}
    )
    ep = _episode(score_evidence(abort))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert ep["verified"] is False


def test_none_evidence_scores_zero_unverified() -> None:
    # Defensive: no evidence at all (should not happen on the sealed path) scores 0, unverified.
    ep = _episode(score_evidence(None))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert ep["verified"] is False


def test_fail_closed_finalize_flags_the_error() -> None:
    # A crash/cancel/timeout mid-verify: the serve layer commits a fail-closed verdict. It scores
    # 0, unverified, and flags `finalize_error` so infra failures are distinguishable from honest
    # zeros.
    fc = TerminalEvidence(
        source="explicit_tool", status="finalize_error", verdict=fail_closed_verdict()
    )
    ep = _episode(score_evidence(fc))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert ep["verified"] is False
    assert ep["finalize_error"] is True


def test_malformed_reward_does_not_crash() -> None:
    bad = TerminalEvidence(
        source="explicit_tool",
        status="ok",
        verdict={"reward": "oops", "reward_found": True},
    )
    ep = _episode(score_evidence(bad))
    assert ep["reward"] == 0.0  # coerces to 0.0, does not raise
    assert ep["success"] is False
    assert ep["verified"] is True  # a genuine verifier verdict was present


def test_boolean_reward_coerces() -> None:
    # A boolean reward (True) coerces to 1.0 — guards the isinstance(bool) branch.
    ev = TerminalEvidence(
        source="explicit_tool",
        status="ok",
        verdict={"reward": True, "reward_found": True},
    )
    ep = _episode(score_evidence(ev))
    assert ep["reward"] == 1.0 and ep["success"] is True


def test_non_finite_reward_coerces_to_zero() -> None:
    ev = TerminalEvidence(
        source="explicit_tool",
        status="ok",
        verdict={"reward": float("inf"), "reward_found": True},
    )
    ep = _episode(score_evidence(ev))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert ep["verified"] is True
