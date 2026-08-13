"""Unit tests for the HLE judge's pure pieces: the exact-match fast path, the parser for the
LLM judge's structured reply, and the request the judge builds. All dependency-free (no
``datasets``, no network: the request tests inject a stand-in client), so they run in the
offline core suite. The one test that reads the serialized wire body needs ``openai`` and skips
without it; it still reaches nothing, driving the SDK over an httpx ``MockTransport``.
"""

from __future__ import annotations

import copy
import json
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
    """A stand-in OpenAI client that records each request and answers with ``reply``.

    Injected as ``client=``, so nothing imports ``openai`` or reaches the network. Each request
    is recorded as a deep copy, so a recording is a snapshot of that call rather than a view of
    whatever the judge holds now."""

    def create(**kwargs: Any) -> Any:
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _wire_probe(reply: str = _REPLY):
    """A real SDK client over an httpx ``MockTransport``: the bodies it records are what was
    serialized onto the wire, not what was handed to ``create``. Still no network."""
    pytest.importorskip("openai")
    import httpx
    from openai import OpenAI

    bodies: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-probe",
                "object": "chat.completion",
                "created": 0,
                "model": "probe",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": reply},
                    }
                ],
            },
        )

    client = OpenAI(
        api_key="sk-probe",
        base_url="http://probe.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, bodies


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
    # Changing the default changes every HLE number a caller who pins no judge will measure, so
    # it is pinned here: moving it takes a deliberate edit to this test.
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
    # The no-kwargs request must stay byte-identical to the pre-pass-through one: no extra field
    # and no null, for endpoints behind `base_url` that reject a field they do not implement.
    calls: List[Dict[str, Any]] = []
    judge = OpenAIJudge(model="judge-model-x", client=_recording_client(calls))

    judge(question="q", correct_answer="Paris", response="Paris")

    assert set(calls[0]) == {"model", "messages"}
    assert calls[0]["messages"][0]["role"] == "user"


def test_the_wire_body_is_the_judge_prompt_plus_exactly_what_was_configured() -> None:
    # What `create` is handed is not the request: the SDK builds the body, and its `extra_*`
    # hatches merge over the named parameters. So this reads the serialized body itself, which
    # is the only place a rewritten model or prompt would be visible.
    client, bodies = _wire_probe()
    OpenAIJudge(model="judge-model-x", client=client)(
        question="Capital of France?", correct_answer="Paris", response="Paris"
    )
    assert set(bodies[0]) == {"model", "messages"}
    assert bodies[0]["model"] == "judge-model-x"
    assert "[correct_answer]: Paris" in bodies[0]["messages"][0]["content"]

    client, bodies = _wire_probe()
    OpenAIJudge(
        model="judge-model-x", client=client, request_kwargs={"reasoning_effort": "low"}
    )(question="Capital of France?", correct_answer="Paris", response="Paris")
    assert set(bodies[0]) == {"model", "messages", "reasoning_effort"}
    assert bodies[0]["model"] == "judge-model-x"
    assert bodies[0]["reasoning_effort"] == "low"


@pytest.mark.parametrize(
    "field, value",
    [
        # What the judge asks. A collision raises inside the call, and a judge that raises fails
        # closed on every non-exact answer.
        ("model", "someone-elses-model"),
        ("messages", []),
        # The SDK merges these over the named parameters, so they can rewrite the model, the
        # prompt, or the effort while the recorded provenance still reports what was configured.
        ("extra_body", {"model": "wire-model"}),
        ("extra_headers", {"x-model": "wire-model"}),
        ("extra_query", {"model": "wire-model"}),
        # The shape of the reply the parser reads. Each of these either raises or costs the reply
        # its verdict line, which fails closed as a wrong answer rather than as a judge error.
        ("stream", True),
        ("n", 2),
        ("response_format", {"type": "json_object"}),
        ("stop", ["correct:"]),
        ("max_tokens", 4),
        ("max_completion_tokens", 4),
        ("tools", []),
        ("tool_choice", "required"),
    ],
)
def test_request_kwargs_refuse_the_fields_the_judge_owns(field, value) -> None:
    with pytest.raises(ValueError) as excinfo:
        OpenAIJudge(request_kwargs={field: value})
    assert field in str(excinfo.value)


def test_sampling_kwargs_are_still_accepted() -> None:
    # The refusal is a line around the judge's own contract, not a whitelist: how the model is
    # sampled stays the caller's to set.
    calls: List[Dict[str, Any]] = []
    judge = OpenAIJudge(
        client=_recording_client(calls),
        request_kwargs={"reasoning_effort": "low", "temperature": 0, "seed": 7},
    )

    judge(question="q", correct_answer="Paris", response="Paris")

    assert calls[0]["temperature"] == 0
    assert calls[0]["seed"] == 7


def test_request_kwargs_are_deep_copied_at_construction() -> None:
    # A shallow copy leaves nested values shared: editing one after construction would change
    # what a later episode is scored with, and no score would show it.
    calls: List[Dict[str, Any]] = []
    kwargs: Dict[str, Any] = {"metadata": {"run": "a"}}
    judge = OpenAIJudge(client=_recording_client(calls), request_kwargs=kwargs)

    judge(question="q", correct_answer="Paris", response="Paris")
    kwargs["metadata"]["run"] = "b"  # the caller's copy
    judge.request_kwargs["metadata"]["run"] = "c"  # what the property handed back
    judge(question="q", correct_answer="Paris", response="Paris")

    assert calls[0]["metadata"] == {"run": "a"}
    assert calls[1]["metadata"] == {"run": "a"}
    assert judge.request_kwargs["metadata"] == {"run": "a"}


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
