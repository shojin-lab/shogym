"""End-to-end: drive a served ``hle`` episode through hgym's serve layer with a scripted
policy + a scripted (injectable) judge, and check the server-side grade flows into episode
feedback — exercising the model-graded ``submit_answer`` handler without any network.

Gated on the ``hle`` extra (``importorskip("datasets")``) so the offline core suite stays
green when the extra isn't installed. The tasks and the judge are injected via ``env_config``,
so nothing here downloads the gated ``cais/hle`` dataset or needs an API key.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("datasets", reason="hle extra not installed")

from hgym.envs.hle import mcp_server  # noqa: E402
from hgym.envs.hle.judge import JudgeResult  # noqa: E402
from hgym.serve import ServedEpisode  # noqa: E402

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


async def test_exact_match_scores_correct_without_calling_the_judge() -> None:
    # RFC 009: submit_answer is the `score` terminal. The call itself seals + finalizes, so
    # it returns terminated=True and a *sanitized* payload (score + judge_error only), never
    # the judge's reasoning/extracted_answer.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        result = await episode.call("submit_answer", {"answer": "Paris", "confidence": 90})
        assert result.terminated
        payload = json.loads(result.content)
        assert payload == {"correct": True, "judge_error": False}
        assert mcp_server.GRADE_MARKER not in payload  # marker stays server-side
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
        await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        await episode.call("terminate", {})
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert fb["calibration_error"] == 1.0
    finally:
        await episode.close()


async def test_negative_task_index_is_rejected() -> None:
    # A negative index must not silently serve `self._tasks[-1]` (a misattributed run). The
    # `_load_task` guard rejects it authoritatively at the resolution point (the serve layer's
    # seeding also rejects negatives first, so no episode is ever built either way).
    import hgym

    env = hgym.make("hle", config=_config(_ScriptedJudge()))
    with pytest.raises(ValueError, match="out of range"):
        env._load_task(-1)
    # End to end: starting an episode at task -1 raises rather than serving the last record.
    with pytest.raises(Exception):
        await ServedEpisode.start("hle", task=-1, env_config=_config(_ScriptedJudge()))


async def test_second_submission_cannot_replace_the_first_answer() -> None:
    # RFC 009: the first submit_answer SEALS the episode. A harness cannot submit a wrong
    # answer, read `correct: false`, then submit a replacement: the second call is tombstoned
    # (no inward dispatch, no second judge run) and the first verdict stands.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(judge))
    try:
        first = await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        assert first.terminated
        assert json.loads(first.content)["correct"] is False
        calls_after_first = judge.calls  # "Berlin" missed exact-match, so the judge ran once

        # Second submission (the correct answer) hits the post-seal tombstone: a generic
        # terminal notice, no grade, no dispatch, and no second judge run.
        second = await episode.call("submit_answer", {"answer": "Paris", "confidence": 100})
        assert second.terminated
        assert "sealed" in second.content
        assert judge.calls == calls_after_first  # the judge was NOT invoked a second time

        fb = _feedback(episode)
        assert fb["correct"] is False  # the first, wrong answer stands
    finally:
        await episode.close()


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


async def test_judge_exception_scores_incorrect_without_crashing() -> None:
    class _BoomJudge:
        def __call__(self, *, question: str, correct_answer: str, response: str) -> JudgeResult:
            raise RuntimeError("judge unavailable")

    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_BoomJudge()))
    try:
        result = await episode.call("submit_answer", {"answer": "somewhere", "confidence": 70})
        assert result.terminated
        payload = json.loads(result.content)
        # Sanitized: fail-closed to incorrect, with the judge_error flag set (no exception
        # text, no reasoning leaked to the agent).
        assert payload["correct"] is False
        assert payload["judge_error"] is True
        assert _feedback(episode)["correct"] is False
    finally:
        await episode.close()


async def test_judge_error_is_labelled_in_episode_feedback() -> None:
    # A mid-run judge failure fail-closes to correct=False (unchanged), but the verifier now
    # also flags judge_error=True so it isn't miscounted as a genuine wrong answer.
    class _BoomJudge:
        def __call__(self, *, question: str, correct_answer: str, response: str) -> JudgeResult:
            raise RuntimeError("judge unavailable")

    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_BoomJudge()))
    try:
        await episode.call("submit_answer", {"answer": "somewhere", "confidence": 70})
        await episode.call("terminate", {})
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert fb["judge_error"] is True
    finally:
        await episode.close()


async def test_preflight_raises_without_key_for_default_judge(monkeypatch) -> None:
    # Default judge (no judge=, no judge_base_url) + no OPENAI_API_KEY: beginning the episode
    # must raise early with a clear, actionable message rather than silently scoring ~0.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError) as excinfo:
        await ServedEpisode.start("hle", task=0, env_config={"tasks": _TASKS})
    msg = str(excinfo.value)
    assert "OPENAI_API_KEY" in msg
    assert "judge_base_url" in msg
    assert "judge=" in msg


async def test_preflight_does_not_fire_with_injected_judge(monkeypatch) -> None:
    # An injected scripted judge opts out of the preflight (keeps offline tests keyless).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    episode = await ServedEpisode.start("hle", task=0, env_config=_config(_ScriptedJudge()))
    try:
        assert episode.describe() is not None
    finally:
        await episode.close()


async def test_preflight_does_not_fire_with_base_url(monkeypatch) -> None:
    # A judge_base_url override (a keyless OpenAI-compatible endpoint) opts out too. The
    # default judge is built but never *called* here (no submit_answer), so no network happens.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {"tasks": _TASKS, "judge_base_url": "http://localhost:11434/v1"}
    episode = await ServedEpisode.start("hle", task=0, env_config=config)
    try:
        assert episode.describe() is not None
    finally:
        await episode.close()


async def test_keyless_base_url_grades_via_llm_judge_not_error(monkeypatch) -> None:
    # The keyless-endpoint contract, end to end: with OPENAI_API_KEY unset and a judge_base_url
    # set, a non-exact submit_answer must actually invoke the default judge (judged_by=llm_judge)
    # rather than fail-closing to llm_judge_error because the SDK client refused to construct.
    # We patch `openai.OpenAI` so nothing hits the network; the fake asserts it was handed a
    # non-empty api_key and returns a canned, correctly-formatted judge reply.
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
        # The client constructed and the judge ran: correct, no judge_error.
        assert payload["correct"] is True
        assert payload["judge_error"] is False
        assert constructed["base_url"] == "http://localhost:11434/v1"
        assert constructed["api_key"]  # non-empty placeholder (no real key was set)

        fb = _feedback(episode)
        assert fb["correct"] is True
        assert "judge_error" not in fb
    finally:
        await episode.close()


async def test_make_and_describe_are_keyless(monkeypatch) -> None:
    # Constructing the env, reading describe(), and probing the tool manifest must stay offline
    # and keyless with the default judge — only session-begin preflights the key.
    import hgym

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = hgym.make("hle", config={"tasks": _TASKS})  # default judge, no key: must not raise
    spec = env.describe("0")
    assert "capital of France" in spec.instructions
    assert {"submit_answer", "terminate"} <= {t.name for t in spec.tools}
