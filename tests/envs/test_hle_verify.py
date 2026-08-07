"""The HLE verifier scores the core-owned terminal evidence, not marker JSON (RFC 008).

No judge and no serving here — build a :class:`TerminalEvidence` by hand and check the episode
feedback ``score_evidence`` derives from it. It is dataset-free and judge-free, so these run in
the offline core suite (no ``hle`` extra).
"""

from __future__ import annotations

from typing import Optional

from shogym.envs.hle.env_v1 import score_evidence
from shogym.serve.lifecycle import TerminalEvidence


def _evidence(
    *,
    correct: bool,
    confidence: Optional[int] = 100,
    judge_error: bool = False,
    status: str = "ok",
    source: str = "explicit_tool",
    submitted: bool = True,
) -> TerminalEvidence:
    """A core-owned evidence like the serve layer commits: a public-safe verdict plus the
    validated submit args (present only for a real submission)."""
    verdict = {"correct": correct, "judge_error": judge_error}
    if submitted:
        args = {"answer": "x"}
        if confidence is not None:
            args["confidence"] = confidence
    else:
        args = None
    return TerminalEvidence(source=source, status=status, verdict=verdict, args=args)


def _episode(fb) -> dict:
    return {f.name: f.value for f in fb.episode}


def test_correct_full_confidence_scores_zero_calibration_error() -> None:
    ep = _episode(score_evidence(_evidence(correct=True, confidence=100), terminated=True))
    assert ep["correct"] is True
    assert ep["confidence"] == 1.0
    assert ep["calibration_error"] == 0.0


def test_wrong_high_confidence_is_maximally_miscalibrated() -> None:
    ep = _episode(score_evidence(_evidence(correct=False, confidence=100), terminated=True))
    assert ep["correct"] is False
    assert ep["calibration_error"] == 1.0  # |1.0 - 0.0|


def test_calibration_error_tracks_confidence() -> None:
    # Correct but only 40% confident -> |0.4 - 1.0| = 0.6.
    ep = _episode(score_evidence(_evidence(correct=True, confidence=40), terminated=True))
    assert ep["confidence"] == 0.4
    assert abs(ep["calibration_error"] - 0.6) < 1e-9
    # Wrong and 30% confident -> |0.3 - 0.0| = 0.3.
    ep2 = _episode(score_evidence(_evidence(correct=False, confidence=30), terminated=True))
    assert abs(ep2["calibration_error"] - 0.3) < 1e-9


def test_no_feedback_until_terminated() -> None:
    assert score_evidence(_evidence(correct=True), terminated=False).episode == []


def test_no_evidence_scores_incorrect() -> None:
    # A terminal with no evidence at all (e.g. a legacy/none-evidence path) scores incorrect.
    ep = _episode(score_evidence(None, terminated=True))
    assert ep["correct"] is False
    assert "calibration_error" not in ep
    assert "confidence" not in ep


def test_no_submission_scores_premature_incorrect() -> None:
    # Terminated without a submission (horizon `zero_unsubmitted` / abort): incorrect, and no
    # confidence to calibrate.
    ev = _evidence(correct=False, source="horizon", submitted=False)
    ep = _episode(score_evidence(ev, terminated=True))
    assert ep["correct"] is False
    assert "calibration_error" not in ep
    assert "confidence" not in ep


def test_abort_evidence_scores_incorrect_only() -> None:
    # The serve layer synthesizes an abort verdict {correct: False, aborted: True} with no
    # submit args; the verifier scores just `correct=False`.
    ev = TerminalEvidence(
        source="abort", status="ok", verdict={"correct": False, "aborted": True}, args=None
    )
    ep = _episode(score_evidence(ev, terminated=True))
    assert ep["correct"] is False
    assert "confidence" not in ep


def test_judge_error_status_sets_judge_error_flag() -> None:
    # A grade fail-closed by a broken judge (status=finalize_error) is labelled: correct stays
    # False (so aggregation is unchanged), judge_error=True lets an analyst filter it out — and
    # the submitted confidence still calibrates.
    ev = _evidence(correct=False, confidence=70, judge_error=True, status="finalize_error")
    ep = _episode(score_evidence(ev, terminated=True))
    assert ep["correct"] is False
    assert ep["judge_error"] is True
    assert ep["confidence"] == 0.7
    assert abs(ep["calibration_error"] - 0.7) < 1e-9


def test_serve_layer_fail_closed_verdict_is_labelled_judge_error() -> None:
    # A serve-layer fail-closed (deadline/crash) sets finalize_error on the verdict WITHOUT a
    # judge_error key; the verifier still labels it judge_error from evidence.finalize_error.
    ev = TerminalEvidence(
        source="explicit_tool",
        status="finalize_error",
        verdict={"correct": False, "finalize_error": True},
        args={"answer": "x", "confidence": 50},
    )
    ep = _episode(score_evidence(ev, terminated=True))
    assert ep["correct"] is False
    assert ep["judge_error"] is True
    assert abs(ep["calibration_error"] - 0.5) < 1e-9


def test_clean_grades_do_not_set_judge_error() -> None:
    for correct in (True, False):
        ep = _episode(score_evidence(_evidence(correct=correct), terminated=True))
        assert "judge_error" not in ep


def test_missing_confidence_argument_defaults_to_full() -> None:
    # A submission with no confidence recorded defaults confidence to 1.0.
    ep = _episode(score_evidence(_evidence(correct=True, confidence=None), terminated=True))
    assert ep["confidence"] == 1.0
    assert ep["calibration_error"] == 0.0
