"""The HLE verifier is a pure function over the recorded trajectory (RFC 008).

No judge and no serving here — build a trajectory whose ``submit_answer`` step carries a
server-side grade by hand and check the episode feedback. ``score_trajectory`` is
dataset-free and judge-free, so these run in the offline core suite (no ``hle`` extra).
"""

from __future__ import annotations

import json

from hgym.envs.hle.env_v1 import GRADE_MARKER, score_trajectory
from hgym.trajectory import Step


def _submit_step(
    index: int,
    *,
    correct: bool = True,
    confidence: int = 100,
    marker: bool = True,
    tool: str = "submit_answer",
) -> Step:
    payload = {"correct": correct, "confidence": confidence, "judged_by": "llm_judge"}
    if marker:
        payload[GRADE_MARKER] = True
    return Step(
        index=index,
        tool=tool,
        arguments={"answer": "x", "confidence": confidence},
        result=json.dumps(payload),
    )


def _judged_step(
    index: int,
    *,
    judged_by: str,
    correct: bool = False,
    confidence: int = 100,
) -> Step:
    """A graded ``submit_answer`` step whose verdict names how it was ``judged_by``."""
    payload = {
        GRADE_MARKER: True,
        "correct": correct,
        "confidence": confidence,
        "judged_by": judged_by,
    }
    return Step(
        index=index,
        tool="submit_answer",
        arguments={"answer": "x", "confidence": confidence},
        result=json.dumps(payload),
    )


def _episode(fb) -> dict:
    return {f.name: f.value for f in fb.episode}


def test_correct_full_confidence_scores_zero_calibration_error() -> None:
    ep = _episode(score_trajectory([_submit_step(1, correct=True, confidence=100)], terminated=True))
    assert ep["correct"] is True
    assert ep["confidence"] == 1.0
    assert ep["calibration_error"] == 0.0


def test_wrong_high_confidence_is_maximally_miscalibrated() -> None:
    ep = _episode(score_trajectory([_submit_step(1, correct=False, confidence=100)], terminated=True))
    assert ep["correct"] is False
    assert ep["calibration_error"] == 1.0  # |1.0 - 0.0|


def test_calibration_error_tracks_confidence() -> None:
    # Correct but only 40% confident -> |0.4 - 1.0| = 0.6.
    ep = _episode(score_trajectory([_submit_step(1, correct=True, confidence=40)], terminated=True))
    assert ep["confidence"] == 0.4
    assert abs(ep["calibration_error"] - 0.6) < 1e-9
    # Wrong and 30% confident -> |0.3 - 0.0| = 0.3.
    ep2 = _episode(score_trajectory([_submit_step(1, correct=False, confidence=30)], terminated=True))
    assert abs(ep2["calibration_error"] - 0.3) < 1e-9


def test_confidence_read_from_arguments_not_result() -> None:
    # The trusted confidence is the recorded *argument*; a forged result confidence is ignored.
    step = Step(
        index=1,
        tool="submit_answer",
        arguments={"answer": "x", "confidence": 20},
        result=json.dumps({GRADE_MARKER: True, "correct": True, "confidence": 99}),
    )
    ep = _episode(score_trajectory([step], terminated=True))
    assert ep["confidence"] == 0.2  # from arguments (20), not the result's 99


def test_no_feedback_until_terminated() -> None:
    assert score_trajectory([_submit_step(1)], terminated=False).episode == []


def test_missing_submission_scores_premature_incorrect() -> None:
    # Terminated without a `submit_answer`: incorrect, and no confidence to calibrate.
    traj = [Step(index=1, tool="terminate", arguments={}, result='{"acknowledged": true}')]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["correct"] is False
    assert "calibration_error" not in ep
    assert "confidence" not in ep


def test_first_submission_wins() -> None:
    # Single-turn: the FIRST graded answer is authoritative, so a later marked grade can't
    # replace a wrong first answer (the server also refuses to grade a second submission).
    traj = [
        _submit_step(1, correct=False, confidence=10),
        _submit_step(2, correct=True, confidence=90),
    ]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["correct"] is False
    assert ep["confidence"] == 0.1


def test_unmarked_result_is_not_trusted() -> None:
    # A `submit_answer`-shaped result WITHOUT the marker must not grant credit.
    ep = _episode(score_trajectory([_submit_step(1, correct=True, marker=False)], terminated=True))
    assert ep["correct"] is False
    assert "calibration_error" not in ep


def test_grade_only_trusted_from_submit_step() -> None:
    # A forged marked grade on a non-`submit_answer` tool must not grant credit.
    forged = json.dumps({GRADE_MARKER: True, "correct": True, "confidence": 100})
    traj = [Step(index=1, tool="describe", arguments={}, result=forged)]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["correct"] is False


def test_malformed_and_non_json_results_do_not_crash() -> None:
    for junk in ("not json", "null", "[]", "42", '"nope"'):
        traj = [Step(index=1, tool="submit_answer", arguments={"answer": "x"}, result=junk)]
        ep = _episode(score_trajectory(traj, terminated=True))
        assert ep["correct"] is False  # treated as no grade


def test_judge_error_grade_sets_judge_error_flag() -> None:
    # A grade fail-closed by a broken judge is labelled: correct stays False (so aggregation is
    # unchanged), but judge_error=True lets an analyst filter it out of the genuine zeros.
    ep = _episode(
        score_trajectory([_judged_step(1, judged_by="llm_judge_error")], terminated=True)
    )
    assert ep["correct"] is False
    assert ep["judge_error"] is True


def test_clean_grades_do_not_set_judge_error() -> None:
    # A normal judged grade (LLM judge or exact match) must not carry the judge_error flag.
    for judged_by in ("llm_judge", "exact_match"):
        ep = _episode(
            score_trajectory(
                [_judged_step(1, judged_by=judged_by, correct=True)], terminated=True
            )
        )
        assert ep["correct"] is True
        assert "judge_error" not in ep


def test_missing_confidence_argument_defaults_to_full() -> None:
    # A grade with no confidence argument recorded defaults confidence to 1.0.
    step = Step(
        index=1,
        tool="submit_answer",
        arguments={"answer": "x"},
        result=json.dumps({GRADE_MARKER: True, "correct": True}),
    )
    ep = _episode(score_trajectory([step], terminated=True))
    assert ep["confidence"] == 1.0
    assert ep["calibration_error"] == 0.0
