"""Unit tests for the HLE judge's pure pieces: the exact-match fast path and the parser for
the LLM judge's structured reply. Both are dependency-free (no ``datasets``, no network), so
they run in the offline core suite.
"""

from __future__ import annotations

import pytest

from hgym.envs.hle.judge import (
    OpenAIJudge,
    exact_match,
    normalize_answer,
    parse_judge_response,
)


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
