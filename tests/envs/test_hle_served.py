"""End-to-end: drive a served ``hle`` episode through shogym's serve layer with a scripted
policy + a scripted (injectable) judge, and check the seal-before-verdict terminal transaction
and the model-graded score — without any network.

``submit_answer`` is the env's ``score`` terminal: calling it validates the args, atomically
seals the episode, runs the env's ``finalize`` (exact-match fast path, then the injected
judge), and ends the episode in one step. These tests cover that flow: validate-before-seal,
seal-before-judge, payload sanitization (no answer oracles reach the agent), the
``zero_unsubmitted`` horizon, and the manifest gate. Only HLE marks a ``score`` terminal, so
the generic lifecycle mechanics (close-race, cancellation, durability, recovery) are proven
once on the fixture in ``tests/test_terminal_lifecycle.py``.

Gated on the ``hle`` extra (``importorskip("datasets")``) so the offline core suite stays
green when the extra isn't installed. The tasks and the judge are injected via ``env_config``,
so nothing here downloads the gated ``cais/hle`` dataset or needs an API key.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="hle extra not installed")

from fastmcp import Client  # noqa: E402

from shogym.envs.hle.judge import DEFAULT_JUDGE_MODEL, JudgeResult  # noqa: E402
from shogym.serve import LifecycleState, ServedEpisode  # noqa: E402
from shogym.serve.server import build_server  # noqa: E402

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
    """A deterministic judge that records how often it is called (to prove the exact-match
    fast path short-circuits it) and grades correct iff the response mentions 'light'."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, question: str, correct_answer: str, response: str) -> JudgeResult:
        self.calls += 1
        return JudgeResult(correct="light" in response.lower(), extracted_answer="Paris")


def _config(judge) -> dict:
    return {"tasks": _TASKS, "judge": judge}


def _feedback(episode: ServedEpisode) -> dict:
    return {item["name"]: item["value"] for item in episode.terminal_feedback}


async def test_describe_surfaces_the_question_and_tools() -> None:
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        spec = episode.describe()
        assert {"submit_answer", "terminate"} <= {t.name for t in spec.tools}
        assert "capital of France" in spec.instructions
    finally:
        await episode.close()


async def test_manifest_marks_submit_answer_score_and_terminate_abort() -> None:
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        spec = episode.describe()
        by_name = {t.name: t for t in spec.tools}
        assert by_name["submit_answer"].terminal_kind == "score"
        assert by_name["terminate"].terminal_kind == "abort"
        assert sum(t.terminal_kind == "score" for t in spec.tools) == 1
        assert spec.contract_version == 2
        assert episode.seal_enabled is True
    finally:
        await episode.close()


# ----- the seal flow: submit_answer is the terminal action -----


async def test_exact_match_scores_correct_without_calling_the_judge() -> None:
    # submit_answer is the `score` terminal: the call itself seals + finalizes, so it returns
    # terminated=True and a *sanitized* payload — never the judge's reasoning/extracted answer.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        result = await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})
        assert result.terminated
        payload = json.loads(result.content)
        assert payload["correct"] is True
        assert payload["judge_error"] is False
        assert judge.calls == 0  # fast path short-circuited the judge

        fb = _feedback(episode)
        assert fb["correct"] is True
        assert fb["confidence"] == 0.9
        assert abs(fb["calibration_error"] - 0.1) < 1e-9
    finally:
        await episode.close()


async def test_llm_judge_path_grades_a_paraphrase_correct() -> None:
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        result = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 40}
        )
        assert result.terminated
        payload = json.loads(result.content)
        assert payload["correct"] is True
        assert payload["judge_error"] is False
        assert judge.calls == 1  # exact match missed, so the judge ran

        fb = _feedback(episode)
        assert fb["correct"] is True
        assert abs(fb["calibration_error"] - 0.6) < 1e-9  # |0.4 - 1.0|
    finally:
        await episode.close()


async def test_wrong_confident_answer_is_maximally_miscalibrated() -> None:
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        result = await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        assert result.terminated
        assert json.loads(result.content)["correct"] is False
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert fb["calibration_error"] == 1.0
    finally:
        await episode.close()


async def test_negative_task_index_is_rejected() -> None:
    # A negative index must not silently serve `self._tasks[-1]` (a misattributed run).
    import shogym

    env = shogym.make("hle", config=_config(_ScriptedJudge()))
    with pytest.raises(ValueError, match="out of range"):
        env._load_task(-1)
    with pytest.raises(Exception):
        await ServedEpisode.start("hle", task=-1, env_config=_config(_ScriptedJudge()))


# ----- validate -> seal ordering -----


async def test_invalid_terminal_call_is_a_validation_error_while_open() -> None:
    # An invalid submit (missing `answer`, wrong type, extra field, blank answer) is a NORMAL
    # validation error while the episode is still OPEN: not sealed, no verdict, no judge call.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        for bad in (
            {"confidence": 90},  # missing `answer`
            {"answer": "Paris", "confidence": "lots"},  # wrong type
            {"answer": "Paris", "surprise": True},  # additionalProperties: false
            {"answer": "   ", "confidence": 90},  # blank required string
        ):
            res = await episode.call("submit_answer", bad)
            assert res.terminated is False
            assert json.loads(res.content)["validation_error"] is True
            assert episode._state is LifecycleState.OPEN  # NOT sealed
            assert episode._finalization is None
            assert episode.terminal_feedback == []
            assert judge.calls == 0

        # A now-valid submission seals + grades — the episode was never consumed.
        ok = await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})
        assert ok.terminated is True
        assert _feedback(episode)["correct"] is True
    finally:
        await episode.close()


async def test_validation_error_through_served_interface_does_not_seal() -> None:
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
            assert json.loads(res.content[0].text)["validation_error"] is True
            assert episode._state is LifecycleState.OPEN
            assert judge.calls == 0

            ok = await client.call_tool("submit_answer", {"answer": "Paris", "confidence": 80})
            assert json.loads(ok.content[0].text)["correct"] is True
            assert episode.terminated is True
    finally:
        await episode.close()


async def test_judge_runs_only_after_the_episode_is_sealed() -> None:
    # Seal-before-verdict: the evaluator observes an already-sealed (FINALIZING) episode, so a
    # verdict is only ever produced for a sealed, un-continuable episode.
    ref: dict = {}

    class _StateProbeJudge:
        def __init__(self) -> None:
            self.state_when_called = None

        def __call__(self, *, question, correct_answer, response) -> JudgeResult:
            self.state_when_called = ref["episode"]._state
            return JudgeResult(correct="light" in response.lower())

    judge = _StateProbeJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    ref["episode"] = episode
    try:
        res = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 60}
        )
        assert res.terminated is True
        assert judge.state_when_called is LifecycleState.FINALIZING
    finally:
        await episode.close()


# ----- post-seal tombstone -----


async def test_second_submission_cannot_replace_the_first_answer() -> None:
    # The first submit_answer SEALS the episode. A harness cannot submit a wrong answer, read
    # `correct: false`, then submit a replacement: the second call is tombstoned (no inward
    # dispatch, no second judge run) and the first verdict stands.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        first = await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        assert first.terminated
        assert json.loads(first.content)["correct"] is False
        calls_after_first = judge.calls  # "Berlin" missed exact-match, so the judge ran once

        second = await episode.call("submit_answer", {"answer": "Paris", "confidence": 100})
        assert second.terminated
        assert "sealed" in second.content  # generic tombstone, no grade
        assert judge.calls == calls_after_first  # the judge was NOT invoked a second time

        assert _feedback(episode)["correct"] is False  # the first, wrong answer stands
    finally:
        await episode.close()


async def test_post_seal_tombstoned_call_records_no_trajectory_step() -> None:
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})  # seals
        recorded = len(episode._trajectory)
        await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        await episode.call("terminate", {})
        assert len(episode._trajectory) == recorded  # nothing dispatched or recorded post-seal
    finally:
        await episode.close()


# ----- horizon policy: zero_unsubmitted -----


async def test_premature_terminate_scores_incorrect() -> None:
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        result = await episode.call("terminate", {})
        assert result.terminated
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert "calibration_error" not in fb
    finally:
        await episode.close()


async def test_zero_unsubmitted_horizon_scores_incorrect() -> None:
    # Reaching the horizon with no valid submission scores correct=False. `submit_answer` is
    # terminal, so the horizon is 1: a single non-submit call reaches it without submitting.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        r1 = await episode.call("noop", {})  # an unknown/ordinary tool; step 1 == horizon
        assert r1.terminated is True
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert "calibration_error" not in fb  # no submission -> nothing to calibrate
        assert judge.calls == 0
    finally:
        await episode.close()


# ----- judge failures fail closed, no oracle leaked -----


async def test_judge_exception_fails_closed_and_labels_judge_error() -> None:
    class _BoomJudge:
        def __call__(self, *, question, correct_answer, response) -> JudgeResult:
            raise RuntimeError("SECRET judge unavailable")

    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_BoomJudge()))
    try:
        result = await episode.call("submit_answer", {"answer": "somewhere", "confidence": 70})
        assert result.terminated
        payload = json.loads(result.content)
        assert payload["correct"] is False
        assert payload["judge_error"] is True
        assert "SECRET" not in result.content  # the exception text never reaches the agent
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert fb["judge_error"] is True
    finally:
        await episode.close()


async def test_terminal_payload_exposes_no_reasoning_or_diagnostics() -> None:
    # The sanitized terminal payload carries only the public-safe score fields — never the
    # judge's reasoning / extracted_answer / judged_by (answer oracles).
    class _ChattyJudge:
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
        payload = json.loads(result.content)
        assert set(payload) == {"correct", "judge_error", "finalize_error"}
        assert "SECRET_REASONING" not in result.content
        assert "SECRET_GOLD" not in result.content
        assert "reasoning" not in result.content
        assert "extracted_answer" not in result.content
        assert "judged_by" not in result.content
    finally:
        await episode.close()


async def test_finalize_deadline_fails_closed() -> None:
    # If the judge overruns the finalize deadline, the episode fails closed to a judge_error
    # verdict (correct=False) rather than hanging. (The judge runs in a worker thread, so the
    # deadline timer fires even while it blocks.)
    class _SlowJudge:
        def __call__(self, *, question, correct_answer, response) -> JudgeResult:
            time.sleep(0.3)  # exceeds the 0.05s deadline
            return JudgeResult(correct=True)

    episode = await ServedEpisode.start(
        "hle", task=0, env_config=_config(_SlowJudge()), finalize_deadline=0.05
    )
    try:
        result = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 50}
        )
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False
        assert payload["finalize_error"] is True  # fail-closed, no oracle leaked
        assert _feedback(episode)["judge_error"] is True
        await asyncio.sleep(0.35)  # let the orphaned judge drain cleanly
    finally:
        await episode.close()


# ----- keyless / preflight (unchanged grading semantics) -----


async def test_preflight_raises_without_key_for_default_judge(monkeypatch) -> None:
    # Default judge + no OPENAI_API_KEY: beginning the episode must raise early with a clear,
    # actionable message rather than silently scoring ~0.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        await ServedEpisode.start("hle", task=0, env_config={"tasks": _TASKS})
    msg = str(excinfo.value)
    assert "OPENAI_API_KEY" in msg
    assert "judge_base_url" in msg
    assert "judge=" in msg


async def test_preflight_does_not_fire_with_injected_judge(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        assert episode.describe() is not None
    finally:
        await episode.close()


async def test_preflight_does_not_fire_with_base_url(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {"tasks": _TASKS, "judge_base_url": "http://localhost:11434/v1"}
    episode = await ServedEpisode.start("hle", task=0, env_config=config)
    try:
        assert episode.describe() is not None
    finally:
        await episode.close()


async def test_keyless_base_url_grades_via_llm_judge_not_error(monkeypatch) -> None:
    # The keyless-endpoint contract, end to end: with OPENAI_API_KEY unset and a judge_base_url
    # set, a non-exact submit_answer must actually invoke the default judge (correct=True, no
    # judge_error) rather than fail-closing because the SDK client refused to construct. We
    # patch `openai.OpenAI` so nothing hits the network.
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    constructed: dict = {}

    class _FakeMessage:
        content = (
            "extracted_final_answer: The City of Light\n"
            "reasoning: matches the gold answer Paris\n"
            "correct: yes\n"
            "confidence: 95"
        )

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeCompletion:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, *, model, messages):
            return _FakeCompletion()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        chat = _FakeChat()

    def _fake_openai(*, base_url=None, api_key=None):
        constructed["base_url"] = base_url
        constructed["api_key"] = api_key
        assert api_key, "OpenAIJudge must supply a non-empty api_key for a keyless base_url"
        return _FakeClient()

    monkeypatch.setattr("openai.OpenAI", _fake_openai)

    config = {"tasks": _TASKS, "judge_base_url": "http://localhost:11434/v1"}
    episode = await ServedEpisode.start("hle", task=0, env_config=config)
    try:
        result = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 40}
        )
        assert result.terminated
        payload = json.loads(result.content)
        assert payload["correct"] is True
        assert payload["judge_error"] is False
        assert constructed["base_url"] == "http://localhost:11434/v1"
        assert constructed["api_key"]  # non-empty placeholder (no real key was set)

        fb = _feedback(episode)
        assert fb["correct"] is True
        assert "judge_error" not in fb
    finally:
        await episode.close()


# ----- what the default judge is asked, and what the score says graded it -----


def _patch_openai_recording_the_request(monkeypatch, calls: list) -> None:
    """Point the default judge at a stand-in client that records the request it is sent.

    The judge builds its client with ``from openai import OpenAI`` at call time, so patching the
    SDK constructor is what stands between these tests and the network. The recorded kwargs are
    the request the env's own judge actually built, config and all."""
    pytest.importorskip("openai")

    reply = (
        "extracted_final_answer: The City of Light\n"
        "reasoning: matches the gold answer Paris\n"
        "correct: yes\n"
        "confidence: 95"
    )

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=reply))])

    def _fake_openai(*, base_url=None, api_key=None):
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    monkeypatch.setattr("openai.OpenAI", _fake_openai)


async def _grade_with_default_judge(monkeypatch, calls: list, **config) -> dict:
    """Grade one non-exact submission with the env's own judge; return the terminal feedback."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-used")
    _patch_openai_recording_the_request(monkeypatch, calls)
    episode = await ServedEpisode.start("hle", task=0, env_config={"tasks": _TASKS, **config})
    try:
        result = await episode.call(
            "submit_answer", {"answer": "The City of Light", "confidence": 40}
        )
        assert result.terminated
        assert json.loads(result.content)["judge_error"] is False
        return _feedback(episode)
    finally:
        await episode.close()


async def test_unconfigured_env_grades_with_the_default_model_and_records_it(monkeypatch) -> None:
    # Nothing specified: the env resolves DEFAULT_JUDGE_MODEL, sends the same two fields it has
    # always sent (no effort, and no null standing in for one), and the score names what graded.
    calls: list = []
    fb = await _grade_with_default_judge(monkeypatch, calls)

    assert calls[0]["model"] == DEFAULT_JUDGE_MODEL
    assert set(calls[0]) == {"model", "messages"}
    assert fb["judge_model"] == DEFAULT_JUDGE_MODEL
    assert "judge_effort" not in fb  # nothing was configured, so nothing is claimed


async def test_judge_kwargs_reach_the_request_and_ride_out_with_the_score(monkeypatch) -> None:
    # An explicit judge_model still overrides the default, and judge_kwargs reach the request
    # they were always meant to reach. Both are scoring decisions, so both are recorded.
    calls: list = []
    fb = await _grade_with_default_judge(
        monkeypatch,
        calls,
        judge_model="judge-model-x",
        judge_kwargs={"reasoning_effort": "low"},
    )

    assert calls[0]["model"] == "judge-model-x"
    assert calls[0]["reasoning_effort"] == "low"
    assert fb["judge_model"] == "judge-model-x"
    assert fb["judge_effort"] == "low"


async def test_injected_judge_and_the_fast_path_claim_no_judge_model() -> None:
    # The env only names a judge it built itself. An injected judge is the caller's to describe,
    # and an exact match was read by no model at all: neither may borrow a model's name.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        await episode.call("submit_answer", {"answer": "The City of Light", "confidence": 40})
        assert judge.calls == 1
        assert "judge_model" not in _feedback(episode)
    finally:
        await episode.close()

    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})
        assert "judge_model" not in _feedback(episode)
    finally:
        await episode.close()


async def test_make_and_describe_are_keyless(monkeypatch) -> None:
    # Constructing the env, reading describe(), and probing the tool manifest must stay offline
    # and keyless with the default judge — only session-begin preflights the key.
    import shogym

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = shogym.make("hle", config={"tasks": _TASKS})  # default judge, no key: must not raise
    spec = env.describe("0")
    assert "capital of France" in spec.instructions
    assert {"submit_answer", "terminate"} <= {t.name for t in spec.tools}
