"""tau2's verifier is a pure function over the terminal verdict (RFC 008).

No tau2 install and no serving here — build a verdict by hand (as a ``done``-step trajectory
for the legacy ``score_trajectory``, or as core-owned ``TerminalEvidence`` for the migrated
``score_evidence``) and check the episode feedback. Both scorers are tau2-free, so these run
in the offline core suite. The parity tests are the **migration fidelity gate**: for the same
verdict, the new evidence-based path emits byte-identical feedback to the legacy path.
"""

from __future__ import annotations

import json

from hgym.envs.tau2.env_v1 import VERDICT_MARKER, score_evidence, score_trajectory
from hgym.serve.lifecycle import TerminalEvidence
from hgym.trajectory import Step


def _verdict_step(
    index: int,
    *,
    reward: float = 1.0,
    db_match: bool | None = True,
    action_match_proportion: float | None = 1.0,
    marker: bool = True,
) -> Step:
    payload = {
        "reward": reward,
        "db_match": db_match,
        "action_match_proportion": action_match_proportion,
    }
    if marker:
        payload[VERDICT_MARKER] = True
    return Step(index=index, tool="done", arguments={}, result=json.dumps(payload))


def _tool_step(index: int, tool: str = "create_task") -> Step:
    return Step(index=index, tool=tool, arguments={"title": "x"}, result='{"ok": true}')


def _episode(fb) -> dict:
    return {f.name: f.value for f in fb.episode}


def test_full_credit_verdict() -> None:
    traj = [_tool_step(1), _verdict_step(2)]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["reward"] == 1.0
    assert ep["success"] is True
    assert ep["db_match"] is True
    assert ep["action_match_proportion"] == 1.0


def test_partial_verdict_is_not_success() -> None:
    traj = [
        _tool_step(1),
        _verdict_step(2, reward=0.0, db_match=False, action_match_proportion=0.5),
    ]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert ep["db_match"] is False
    assert ep["action_match_proportion"] == 0.5


def test_no_feedback_until_terminated() -> None:
    traj = [_tool_step(1), _verdict_step(2)]
    assert score_trajectory(traj, terminated=False).episode == []


def test_missing_verdict_scores_premature_zero() -> None:
    # Harness terminated without ever calling `done`: no verdict recorded.
    traj = [_tool_step(1), _tool_step(2)]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert "db_match" not in ep  # nothing to report


def test_latest_verdict_wins() -> None:
    # If two verdicts are present, the most recent one is authoritative.
    traj = [
        _verdict_step(1, reward=0.0, db_match=False, action_match_proportion=0.0),
        _verdict_step(2, reward=1.0, db_match=True, action_match_proportion=1.0),
    ]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["reward"] == 1.0 and ep["success"] is True


def test_unmarked_result_is_ignored() -> None:
    # A `done`-shaped result WITHOUT the verdict marker must not be trusted as a verdict:
    # a forged full-credit payload lacking the marker scores as premature zero.
    forged = json.dumps({"reward": 1.0, "db_match": True, "action_match_proportion": 1.0})
    traj = [Step(index=1, tool="done", arguments={}, result=forged)]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["reward"] == 0.0
    assert ep["success"] is False


def test_malformed_verdict_fields_do_not_crash() -> None:
    # Marker present but fields junk: reward coerces to 0.0, non-bool db_match / non-numeric
    # action proportion are dropped rather than raising.
    bad = json.dumps(
        {VERDICT_MARKER: True, "reward": "oops", "db_match": "yes", "action_match_proportion": None}
    )
    traj = [Step(index=1, tool="done", arguments={}, result=bad)]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert "db_match" not in ep
    assert "action_match_proportion" not in ep


def test_verdict_only_trusted_from_done_step() -> None:
    # A forged verdict marker on an ordinary domain-tool result must NOT grant terminal
    # credit — only the `done` step (which runs tau2's evaluator) is trusted. Here a
    # `create_task` result carries a full-credit marked payload, then the episode terminates
    # with no real `done` verdict: it must score as premature zero.
    forged = json.dumps(
        {VERDICT_MARKER: True, "reward": 1.0, "db_match": True, "action_match_proportion": 1.0}
    )
    traj = [Step(index=1, tool="create_task", arguments={}, result=forged)]
    ep = _episode(score_trajectory(traj, terminated=True))
    assert ep["reward"] == 0.0
    assert ep["success"] is False
    assert "db_match" not in ep


def test_non_json_results_do_not_crash() -> None:
    for junk in ("not json", "null", "[]", "42", '"nope"'):
        traj = [Step(index=1, tool="done", arguments={}, result=junk)]
        ep = _episode(score_trajectory(traj, terminated=True))
        assert ep["reward"] == 0.0  # treated as no verdict


# ----- migration fidelity gate: score_evidence == score_trajectory for the same verdict -----


def _evidence(verdict: dict, *, source: str = "explicit_tool", status: str = "ok") -> TerminalEvidence:
    return TerminalEvidence(source=source, status=status, verdict=verdict)  # type: ignore[arg-type]


def _verdict(reward=1.0, db_match=True, action_match_proportion=1.0) -> dict:
    return {
        VERDICT_MARKER: True,
        "reward": reward,
        "db_match": db_match,
        "action_match_proportion": action_match_proportion,
    }


def test_score_evidence_matches_score_trajectory_full_credit() -> None:
    # The seal migration must be score-preserving: for a real verdict, the evidence-based scorer
    # emits byte-identical episode feedback to the legacy trajectory-based scorer.
    v = _verdict()
    traj = [_tool_step(1), Step(index=2, tool="done", arguments={}, result=json.dumps(v))]
    assert _episode(score_evidence(_evidence(v))) == _episode(
        score_trajectory(traj, terminated=True)
    )


def test_score_evidence_matches_score_trajectory_partial_and_edge_verdicts() -> None:
    for v in (
        _verdict(reward=0.0, db_match=False, action_match_proportion=0.5),
        _verdict(reward=0.4, db_match=None, action_match_proportion=None),
        {VERDICT_MARKER: True, "reward": None, "db_match": "junk"},  # malformed fields
    ):
        traj = [Step(index=1, tool="done", arguments={}, result=json.dumps(v))]
        assert _episode(score_evidence(_evidence(v))) == _episode(
            score_trajectory(traj, terminated=True)
        ), v


def test_score_evidence_abort_scores_premature_zero() -> None:
    # The core-synthesized abort verdict (a `terminate` with no tau2 reward) reads as a premature
    # zero — identical to the legacy "no verdict recorded" outcome.
    ev = _evidence({"correct": False, "aborted": True}, source="abort")
    ep = _episode(score_evidence(ev))
    assert ep["reward"] == 0.0 and ep["success"] is False
    assert "db_match" not in ep and "action_match_proportion" not in ep
    assert "eval_error" not in ep


def test_score_evidence_flags_eval_error() -> None:
    # An evaluator failure fails closed to reward 0 AND flags eval_error, so infra failures are
    # distinguishable in audit data from a genuine reward-0 run. finalize_error status alone
    # (with no verdict flag) also raises the flag.
    ev = _evidence({VERDICT_MARKER: True, "reward": 0.0, "eval_error": True}, status="finalize_error")
    ep = _episode(score_evidence(ev))
    assert ep["reward"] == 0.0 and ep["success"] is False and ep["eval_error"] is True

    ev2 = _evidence({VERDICT_MARKER: True, "reward": 0.0}, status="finalize_error")
    assert _episode(score_evidence(ev2))["eval_error"] is True
