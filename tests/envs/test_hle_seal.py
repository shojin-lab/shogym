"""RFC 009 (HLE-only prototype): the seal-before-verdict terminal transaction.

Drives a served ``hle`` episode with a scripted judge (no network, no key) and exercises the
lifecycle the prototype adds: validate-before-seal, seal-before-judge + post-seal tombstone,
close() racing the finalizer, post-seal cancellation, payload sanitization, the
``zero_unsubmitted`` horizon, and the ``terminal_kind`` manifest gate. Only HLE marks a
``score`` terminal, so these behaviours engage for HLE alone — see
``test_serve_episode.py`` for proof the non-score engine is byte-identical.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("datasets", reason="hle extra not installed")

from fastmcp import Client  # noqa: E402

from hgym.envs.hle import mcp_server  # noqa: E402
from hgym.envs.hle.env_v1 import GRADE_MARKER  # noqa: E402
from hgym.envs.hle.judge import JudgeResult  # noqa: E402
from hgym.serve import ServedEpisode  # noqa: E402
from hgym.serve.episode import LifecycleState  # noqa: E402
from hgym.serve.server import build_server  # noqa: E402

_TASKS = [
    {
        "id": "q_geo",
        "question": "What is the capital of France?",
        "answer": "Paris",
        "answer_type": "exactMatch",
        "category": "geography",
    }
]


class _ScriptedJudge:
    """Deterministic judge: grades correct iff the response mentions 'light'; counts calls so
    a test can prove the judge ran exactly once (or not at all)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, question: str, correct_answer: str, response: str) -> JudgeResult:
        self.calls += 1
        return JudgeResult(correct="light" in response.lower(), extracted_answer="Paris")


class _StateProbeJudge:
    """Records the episode's lifecycle state at the instant the judge is invoked, so a test
    can prove the episode was already SEALED before the evaluator ran."""

    def __init__(self, episode_ref: dict) -> None:
        self.calls = 0
        self.state_when_called: object = None
        self._ref = episode_ref

    def __call__(self, *, question: str, correct_answer: str, response: str) -> JudgeResult:
        self.calls += 1
        self.state_when_called = self._ref["episode"]._state
        return JudgeResult(correct="light" in response.lower())


def _config(judge) -> dict:
    return {"tasks": _TASKS, "judge": judge}


def _feedback(episode: ServedEpisode) -> dict:
    return {item["name"]: item["value"] for item in episode.terminal_feedback}


# ----- validate -> seal ordering -----


async def test_invalid_terminal_call_is_a_validation_error_while_open() -> None:
    # An invalid submit_answer (missing `answer`) is a NORMAL validation error while the
    # episode is still OPEN: not sealed, no verdict, no judge call, no evidence.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        res = await episode.call("submit_answer", {"confidence": 90})  # no `answer`
        assert res.terminated is False
        payload = json.loads(res.content)
        assert payload["validation_error"] is True

        assert episode._state is LifecycleState.OPEN  # NOT sealed
        assert episode.terminated is False
        assert episode._finalization is None
        assert episode.terminal_feedback == []  # no verdict recorded
        assert judge.calls == 0

        # The harness may correct and re-submit: a valid answer now seals + grades.
        res2 = await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})
        assert res2.terminated is True
        assert _feedback(episode)["correct"] is True
    finally:
        await episode.close()


async def test_schema_invalid_terminal_call_does_not_seal() -> None:
    # The complete advertised schema is enforced BEFORE the seal: a non-integer `confidence`
    # or an unknown extra field (which FastMCP would reject downstream) is a normal validation
    # error while OPEN — never a sealed finalizer that irrevocably scores 0. The harness can
    # correct and re-submit; the episode was never spent.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        for bad in (
            {"answer": "Paris", "confidence": "lots"},  # wrong type
            {"answer": "Paris", "surprise": True},  # additionalProperties: false
        ):
            res = await episode.call("submit_answer", bad)
            assert res.terminated is False
            assert json.loads(res.content)["validation_error"] is True
            assert episode._state is LifecycleState.OPEN  # NOT sealed
            assert episode._finalization is None
            assert episode.terminal_feedback == []
            assert judge.calls == 0

        # A now-valid submission seals + grades — the episode was never consumed.
        res = await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})
        assert res.terminated is True
        assert _feedback(episode)["correct"] is True
    finally:
        await episode.close()


async def test_schema_invalid_terminal_call_through_served_interface_does_not_seal() -> None:
    # The same guard end-to-end through build_server + a FastMCP Client (the real served
    # interface a harness drives): malformed terminal args do not seal or score the episode.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        server = build_server(episode)
        async with Client(server) as client:
            res = await client.call_tool(
                "submit_answer", {"answer": "Paris", "confidence": "lots"}
            )
            payload = json.loads(res.content[0].text)  # type: ignore[union-attr]
            assert payload["validation_error"] is True
            assert episode._state is LifecycleState.OPEN
            assert episode.terminated is False
            assert judge.calls == 0

            # A valid submit through the same client then seals + grades.
            ok = await client.call_tool(
                "submit_answer", {"answer": "Paris", "confidence": 80}
            )
            assert json.loads(ok.content[0].text) == {  # type: ignore[union-attr]
                "correct": True,
                "judge_error": False,
            }
            assert episode.terminated is True
    finally:
        await episode.close()


async def test_blank_answer_is_a_validation_error_while_open() -> None:
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        res = await episode.call("submit_answer", {"answer": "   ", "confidence": 90})
        assert res.terminated is False
        assert json.loads(res.content)["validation_error"] is True
        assert episode._state is LifecycleState.OPEN
        assert judge.calls == 0
    finally:
        await episode.close()


async def test_judge_runs_only_after_the_episode_is_sealed() -> None:
    # Seal-before-verdict: the evaluator observes an already-SEALED (FINALIZING) episode, so a
    # verdict is only ever produced for a sealed, un-continuable episode.
    ref: dict = {}
    judge = _StateProbeJudge(ref)
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    ref["episode"] = episode
    try:
        res = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 60}
        )
        assert res.terminated is True
        assert judge.calls == 1
        # The judge saw the episode already sealed (never OPEN).
        assert judge.state_when_called in (
            LifecycleState.SEALED,
            LifecycleState.FINALIZING,
        )
    finally:
        await episode.close()


# ----- close() participates in the lifecycle (B3) -----


async def test_close_race_keeps_judge_session_alive_and_tears_down_once() -> None:
    # close() racing an in-flight finalizer must WAIT for the judge to commit evidence before
    # teardown — the HLE session stays alive through the judge, and teardown runs exactly once.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    sess = episode._sessions["submit_answer"]
    real_call = sess.call_tool
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def blocking(*a, **k):
        in_flight.set()  # sealed + finalize dispatched; judge about to run
        await release.wait()
        return await real_call(*a, **k)

    sess.call_tool = blocking  # type: ignore[method-assign]

    submit = asyncio.create_task(
        episode.call("submit_answer", {"answer": "The City of Light", "confidence": 50})
    )
    await asyncio.wait_for(in_flight.wait(), timeout=1.0)
    assert episode._state is LifecycleState.FINALIZING

    close_task = asyncio.create_task(episode.close())
    await asyncio.sleep(0.02)  # let close() reach its await on the finalization
    assert not close_task.done()  # close is WAITING for the finalizer, not tearing down
    assert episode._teardown_runs == 0
    assert episode._session_id in mcp_server._sessions  # judge session still alive

    release.set()
    result = await submit
    await close_task

    assert result.terminated is True
    assert judge.calls == 1  # exactly one judge invocation
    assert episode._teardown_runs == 1  # teardown ran once, after evidence
    assert episode._state is LifecycleState.CLOSED
    assert _feedback(episode)["correct"] is True  # session survived -> graded, not fail-closed


# ----- post-seal cancellation (distinct from the close race) -----


async def test_post_seal_cancellation_awaits_the_single_finalization() -> None:
    # Once SEALED/FINALIZING, cancelling the awaiting request must NOT abandon or re-dispatch
    # the judge: the lifecycle retains and awaits the single in-flight finalization, commits,
    # then tears down. Exactly one judge invocation; never a second.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    sess = episode._sessions["submit_answer"]
    real_call = sess.call_tool
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def blocking(*a, **k):
        in_flight.set()
        await release.wait()
        return await real_call(*a, **k)

    sess.call_tool = blocking  # type: ignore[method-assign]

    submit = asyncio.create_task(
        episode.call("submit_answer", {"answer": "The City of Light", "confidence": 50})
    )
    await asyncio.wait_for(in_flight.wait(), timeout=1.0)
    finalization = episode._finalization
    assert finalization is not None and not finalization.done()
    assert episode.terminal_feedback == []  # not committed yet
    assert episode._teardown_runs == 0

    # Client cancellation / disconnect: cancel the awaiting request.
    submit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submit

    # The single finalization survived the cancellation (shielded); no re-dispatch.
    assert not finalization.done()
    assert episode._teardown_runs == 0

    release.set()
    result = await finalization  # the retained finalization runs to completion

    assert result.terminated is True
    assert judge.calls == 1  # exactly one judge invocation, never a second
    fb = _feedback(episode)
    assert fb["correct"] is True  # evidence committed to a safe outcome
    assert episode._state is LifecycleState.CLOSED  # teardown-after-commit
    assert episode._teardown_runs == 1

    await episode.close()  # idempotent


async def test_finalize_deadline_fails_closed() -> None:
    # The deadline arm of the cancellation rule: if the evaluator overruns its deadline, the
    # episode fails closed to a `finalize_error` verdict (correct=False) rather than hanging.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start(
        "hle", task=0, env_config=_config(judge), finalize_deadline=0.05
    )
    sess = episode._sessions["submit_answer"]
    real_call = sess.call_tool

    async def slow(*a, **k):
        await asyncio.sleep(0.3)  # exceeds the 0.05s finalize deadline
        return await real_call(*a, **k)

    sess.call_tool = slow  # type: ignore[method-assign]
    try:
        result = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 50}
        )
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False
        assert payload["judge_error"] is True  # fail-closed, flagged, no oracle leaked
        assert _feedback(episode)["correct"] is False
        await asyncio.sleep(0.35)  # let the orphaned dispatch drain cleanly
    finally:
        await episode.close()


# ----- payload sanitization -----


async def test_terminal_payload_exposes_no_reasoning_or_diagnostics() -> None:
    # The sanitized terminal payload carries only the score + judge_error — never the judge's
    # reasoning / extracted_answer / judged_by (answer oracles).
    class _ChattyJudge:
        calls = 0

        def __call__(self, *, question, correct_answer, response) -> JudgeResult:
            return JudgeResult(
                correct="light" in response.lower(),
                extracted_answer="SECRET_GOLD_Paris",
                reasoning="SECRET_REASONING matches the gold answer",
            )

    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ChattyJudge()))
    try:
        result = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 50}
        )
        assert result.terminated is True
        assert set(json.loads(result.content)) == {"correct", "judge_error"}
        assert "SECRET_REASONING" not in result.content
        assert "SECRET_GOLD" not in result.content
        assert "reasoning" not in result.content
        assert "extracted_answer" not in result.content
        assert "judged_by" not in result.content
    finally:
        await episode.close()


# ----- horizon policy: zero_unsubmitted -----


async def test_zero_unsubmitted_horizon_scores_incorrect() -> None:
    # Reaching the horizon with no valid submission scores correct=False (no confidence to
    # calibrate). The judge is never invoked.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        # RFC 009: submit_answer is terminal, so the horizon is 1 — a single non-submit call
        # reaches it without ever submitting an answer.
        r1 = await episode.call("noop", {})
        assert r1.terminated is True  # step 1 == horizon

        fb = _feedback(episode)
        assert fb["correct"] is False
        assert "calibration_error" not in fb  # no submission -> nothing to calibrate
        assert judge.calls == 0
    finally:
        await episode.close()


# ----- terminal_kind manifest gate -----


# ----- trust source: the sealed server verdict, distinct from the sanitized agent view -----


async def test_credit_comes_from_the_sealed_server_verdict_not_the_agent_view() -> None:
    # The RFC invariant: credit derives from the AUTHORITATIVE, server-produced marked grade
    # recorded on the sealed submit_answer step — a DIFFERENT object from the sanitized
    # payload the agent sees. Prove the two are distinct and the agent view is not the trust
    # source.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        result = await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        assert result.terminated is True

        # Agent view: sanitized, no trust marker, no diagnostics.
        agent_view = json.loads(result.content)
        assert agent_view == {"correct": False, "judge_error": False}
        assert GRADE_MARKER not in agent_view

        # Authoritative source: the recorded submit_answer step carries the full,
        # server-owned marked grade (hle_grade + judged_by) — never shown to the agent. This
        # is what _verify trusts, and it is a different object than the agent view.
        recorded = json.loads(episode._trajectory[-1].result)
        assert episode._trajectory[-1].tool == "submit_answer"
        assert recorded[GRADE_MARKER] is True
        assert "judged_by" in recorded
        assert recorded["correct"] is False
        assert result.content != episode._trajectory[-1].result  # projection != trust source

        # The verifier's verdict tracks the server grade (not any agent-controlled field).
        assert _feedback(episode)["correct"] is False
    finally:
        await episode.close()


async def test_post_seal_tombstoned_call_records_no_trajectory_step() -> None:
    # A tombstoned post-seal call does no inward dispatch and records NO trajectory step, so a
    # forged marker on such a call can never enter the trajectory the verifier reads. Combined
    # with the verifier trusting only server-produced submit_answer results (see
    # test_hle_verify.test_grade_only_trusted_from_submit_step), the agent has no forge path.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})  # seals
        recorded = len(episode._trajectory)
        # Every post-seal call is tombstoned: no dispatch, no new step recorded.
        await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        await episode.call("terminate", {})
        await episode.call("noop", {})
        assert len(episode._trajectory) == recorded  # nothing recorded after the seal
    finally:
        await episode.close()


async def test_manifest_marks_submit_answer_score_and_terminate_abort() -> None:
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        spec = episode.describe()
        by_name = {t.name: t for t in spec.tools}
        assert by_name["submit_answer"].terminal_kind == "score"
        assert by_name["terminate"].terminal_kind == "abort"
        # Exactly one score terminal.
        assert sum(t.terminal_kind == "score" for t in spec.tools) == 1
        assert spec.contract_version == 2
    finally:
        await episode.close()
