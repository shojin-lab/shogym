"""Unit tests for the HLE judge's pure pieces: the exact-match fast path, the parser for the
LLM judge's structured reply, and the request the judge builds. All dependency-free (no
``datasets``, no network: the request tests drive an injected stand-in client), so they run in
the offline core suite.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from shogym.envs.hle.judge import (
    DEFAULT_JUDGE_MODEL,
    OpenAIJudge,
    exact_match,
    normalize_answer,
    parse_judge_response,
)

_REPLY = (
    "extracted_final_answer: Paris\nreasoning: matches the gold answer\ncorrect: yes\n"
    "confidence: 95"
)


def _recording_client(calls: List[Dict[str, Any]], reply: str = _REPLY) -> Any:
    """A stand-in for the OpenAI client that records each request and answers with ``reply``.

    Injected as ``client=``, so nothing imports ``openai`` or reaches the network, and the
    recorded kwargs are the request the judge actually built."""

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_exact_match_normalizes_case_space_and_punctuation() -> None:
    assert exact_match("Paris", "paris")
    assert exact_match("  Paris.  ", "Paris")
    assert exact_match("the  answer", "the answer")
    assert exact_match("A)", "a")


def test_exact_match_rejects_semantic_differences() -> None:
    # The fast path must never grant credit an exact comparison wouldn't.
    assert not exact_match("The City of Light", "Paris")
    assert not exact_match("42 km", "42")
    assert not exact_match("berlin", "paris")


def test_empty_gold_never_matches() -> None:
    assert not exact_match("", "")
    assert not exact_match("anything", "")
    assert normalize_answer(None) == ""


def test_parse_judge_yes_no() -> None:
    yes = parse_judge_response(
        "extracted_final_answer: Paris\nreasoning: matches\ncorrect: yes\nconfidence: 95"
    )
    assert yes.correct is True
    assert yes.extracted_answer == "Paris"
    assert "matches" in yes.reasoning

    no = parse_judge_response("extracted_final_answer: Berlin\ncorrect: no\nconfidence: 80")
    assert no.correct is False
    assert no.extracted_answer == "Berlin"


def test_parse_judge_is_case_insensitive_and_defaults_false() -> None:
    assert parse_judge_response("Correct: YES").correct is True
    # No parseable verdict -> not correct (never grant credit on malformed judge output).
    assert parse_judge_response("the model rambled without a verdict").correct is False
    assert parse_judge_response("").correct is False


def test_parse_judge_ignores_correct_echoed_inside_reasoning() -> None:
    # The reasoning line is model-generated and echoes the agent-controlled response; an
    # embedded "correct: yes" there must NOT override the final "correct: no" verdict.
    reply = (
        "extracted_final_answer: Berlin\n"
        "reasoning: the response claimed correct: yes but that is wrong\n"
        "correct: no\n"
        "confidence: 90"
    )
    assert parse_judge_response(reply).correct is False


def test_parse_judge_fails_closed_on_conflicting_verdicts() -> None:
    # Two line-anchored `correct:` fields disagreeing -> fail closed (no credit).
    assert parse_judge_response("correct: yes\ncorrect: no").correct is False


def test_default_judge_model_is_the_measured_one() -> None:
    # The default judge model is a scoring function, not a preference: changing it changes every
    # HLE number a caller who does not pin one will measure. Pinned here so the change is a
    # deliberate edit to a test that says so, never a drive-by.
    assert DEFAULT_JUDGE_MODEL == "gpt-5.6-luna"
    assert OpenAIJudge().model == DEFAULT_JUDGE_MODEL


def test_request_kwargs_reach_the_create_call() -> None:
    calls: List[Dict[str, Any]] = []
    judge = OpenAIJudge(
        model="judge-model-x",
        client=_recording_client(calls),
        request_kwargs={"reasoning_effort": "low"},
    )

    result = judge(question="Capital of France?", correct_answer="Paris", response="Paris")

    assert result.correct is True
    assert len(calls) == 1
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["model"] == "judge-model-x"


def test_without_request_kwargs_the_request_carries_only_model_and_messages() -> None:
    # The no-kwargs request must be exactly what it was before the pass-through existed: no
    # extra field, and no null standing in for an unset one. An OpenAI-compatible server behind
    # `base_url` can reject a field it does not implement, and a rejected request grades nothing.
    calls: List[Dict[str, Any]] = []
    judge = OpenAIJudge(model="judge-model-x", client=_recording_client(calls))

    judge(question="q", correct_answer="Paris", response="Paris")

    assert set(calls[0]) == {"model", "messages"}
    assert calls[0]["messages"][0]["role"] == "user"


def test_request_kwargs_refuse_the_fields_the_judge_supplies_itself() -> None:
    # A colliding key raises a TypeError inside the call, and a judge that raises fails closed,
    # so the run would land as a page of zeros. It has to be an error at construction instead.
    with pytest.raises(ValueError) as excinfo:
        OpenAIJudge(request_kwargs={"model": "someone-elses-model"})
    assert "model" in str(excinfo.value)
    with pytest.raises(ValueError):
        OpenAIJudge(request_kwargs={"messages": []})


def test_request_kwargs_are_frozen_at_construction() -> None:
    # Grading has to stay one function for the length of a run, so the judge holds its own copy:
    # editing the mapping afterwards cannot change what the next episode is scored with.
    kwargs: Dict[str, Any] = {"reasoning_effort": "low"}
    judge = OpenAIJudge(client=_recording_client([]), request_kwargs=kwargs)
    kwargs["reasoning_effort"] = "xhigh"

    assert judge.request_kwargs == {"reasoning_effort": "low"}
    with pytest.raises(TypeError):
        judge.request_kwargs["reasoning_effort"] = "xhigh"  # type: ignore[index]


def test_ensure_client_uses_placeholder_key_for_keyless_base_url(monkeypatch) -> None:
    # A keyless local OpenAI-compatible endpoint (judge_base_url set, OPENAI_API_KEY unset) must
    # construct: the SDK refuses a missing/empty api_key even against a local base_url, so the
    # judge supplies a non-empty placeholder. Patch the SDK constructor so nothing hits network.
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    captured: dict = {}

    def _fake_openai(*, base_url=None, api_key=None):
        captured["base_url"] = base_url
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr("openai.OpenAI", _fake_openai)

    judge = OpenAIJudge(base_url="http://localhost:11434/v1")
    client = judge._ensure_client()  # must not raise
    assert client is not None
    assert captured["base_url"] == "http://localhost:11434/v1"
    assert captured["api_key"]  # non-empty


def test_ensure_client_prefers_real_key_when_set(monkeypatch) -> None:
    # When OPENAI_API_KEY is present it is used verbatim (the placeholder is only a fallback).
    pytest.importorskip("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-key-xyz")

    captured: dict = {}

    def _fake_openai(*, base_url=None, api_key=None):
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr("openai.OpenAI", _fake_openai)

    OpenAIJudge()._ensure_client()
    assert captured["api_key"] == "sk-real-key-xyz"
