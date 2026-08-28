"""Unit tests for the HLE judge's pure pieces: the exact-match fast path, the parser for the
LLM judge's structured reply, and the request the judge builds. All dependency-free (no
``datasets``, no network: the request tests inject a stand-in client), so they run in the
offline core suite. The one test that reads the serialized wire body needs ``openai`` and skips
without it; it still reaches nothing, driving the SDK over an httpx ``MockTransport``.
"""

from __future__ import annotations

import copy
import json
from types import MappingProxyType, SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from shogym.envs.hle import judge as judge_module
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


def _recording_client(
    calls: List[Dict[str, Any]], reply: str = _REPLY, reported_model: Optional[str] = None
) -> Any:
    """A stand-in OpenAI client that records each request and answers with ``reply``.

    Injected as ``client=``, so nothing imports ``openai`` or reaches the network. Each request
    is recorded as a deep copy, so a recording is a snapshot of that call rather than a view of
    whatever the judge holds now. The response reports ``reported_model``, or echoes the
    requested id the way a provider does."""

    def create(**kwargs: Any) -> Any:
        calls.append(copy.deepcopy(kwargs))
        return SimpleNamespace(
            model=reported_model if reported_model is not None else kwargs.get("model", ""),
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))],
        )

    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def _wire_probe(reply: str = _REPLY):
    """A real SDK client over an httpx ``MockTransport``: what it records is the request that was
    serialized, not what was handed to ``create``. Still no network.

    The body is recorded parsed, so tests over it assert the request field for field rather than
    byte for byte."""
    pytest.importorskip("openai")
    import httpx
    from openai import OpenAI

    requests: List[Dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(
            {
                "method": request.method,
                "path": request.url.path,
                "body": json.loads(request.content),
            }
        )
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
    return client, requests


def test_exact_match_normalizes_case_space_and_punctuation() -> None:
    assert exact_match("Paris", "paris")
    assert exact_match("  Paris.  ", "Paris")
    assert exact_match("the  answer", "the answer")
    assert exact_match("A)", "a")


def test_exact_match_rejects_semantic_differences() -> None:
    # Semantic differences the fast path does reject. It is not airtight: `_STRIP_CHARS`
    # includes `!` and brackets, so exact_match("5", "5!") is True — see issue #139.
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
    # The no-kwargs request must carry the same fields the pre-pass-through one did: no extra
    # field and no null, for endpoints behind `base_url` that reject fields they do not know.
    calls: List[Dict[str, Any]] = []
    judge = OpenAIJudge(model="judge-model-x", client=_recording_client(calls))

    judge(question="q", correct_answer="Paris", response="Paris")

    assert set(calls[0]) == {"model", "messages"}
    assert calls[0]["messages"][0]["role"] == "user"


def test_the_wire_request_is_the_judge_prompt_plus_exactly_what_was_configured() -> None:
    # What `create` is handed is not the request: the SDK builds it, and its `extra_*` hatches
    # merge over the named parameters. So this reads the serialized request, which is the only
    # place a rewritten model or prompt would be visible. The body is compared as parsed JSON,
    # so what is pinned is the request field for field, not its exact bytes.
    client, requests = _wire_probe()
    OpenAIJudge(model="judge-model-x", client=client)(
        question="Capital of France?", correct_answer="Paris", response="Paris"
    )
    assert requests[0]["method"] == "POST"
    assert requests[0]["path"] == "/v1/chat/completions"
    assert set(requests[0]["body"]) == {"model", "messages"}
    assert requests[0]["body"]["model"] == "judge-model-x"
    assert "[correct_answer]: Paris" in requests[0]["body"]["messages"][0]["content"]

    # An allowlisted sampling field reaches the wire, and nothing else joins it.
    client, requests = _wire_probe()
    OpenAIJudge(
        model="judge-model-x", client=client, request_kwargs={"reasoning_effort": "low"}
    )(question="Capital of France?", correct_answer="Paris", response="Paris")
    assert set(requests[0]["body"]) == {"model", "messages", "reasoning_effort"}
    assert requests[0]["body"]["model"] == "judge-model-x"
    assert requests[0]["body"]["reasoning_effort"] == "low"

    # A field outside the allowlist never gets as far as a request.
    client, requests = _wire_probe()
    with pytest.raises(ValueError):
        OpenAIJudge(model="judge-model-x", client=client, request_kwargs={"functions": []})
    assert requests == []


@pytest.mark.parametrize(
    "field, value",
    [
        # What the judge asks.
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
        # The legacy function-calling API: the same content-less reply as `tools`, under names
        # an exclusion list written against the current API does not mention.
        ("functions", [{"name": "grade", "parameters": {}}]),
        ("function_call", {"name": "grade"}),
        # Other ways to be answered with something that is not the text the parser reads.
        ("audio", {"voice": "alloy", "format": "wav"}),
        ("modalities", ["text", "audio"]),
        ("prediction", {"type": "content", "content": "correct: yes"}),
        ("web_search_options", {}),
        # A token ban is a stop sequence by another route: it can remove the verdict's own words.
        ("logit_bias", {"9891": -100}),
        # Not a request field at all, and not sampling either.
        ("timeout", 0.001),
        # The point of an allowlist: a name that does not exist yet is refused by default.
        ("some_future_sdk_parameter", "whatever"),
    ],
)
def test_only_sampling_settings_are_settable(field, value) -> None:
    with pytest.raises(ValueError) as excinfo:
        OpenAIJudge(request_kwargs={field: value})
    message = str(excinfo.value)
    assert repr(field) in message
    # The error has to say what IS allowed, or a caller cannot act on it.
    assert "reasoning_effort" in message and "temperature" in message


def test_every_allowlisted_sampling_setting_reaches_the_call() -> None:
    calls: List[Dict[str, Any]] = []
    sampling: Dict[str, Any] = {
        "reasoning_effort": "low",
        "temperature": 0,
        "top_p": 0.9,
        "seed": 7,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
    }
    judge = OpenAIJudge(client=_recording_client(calls), request_kwargs=dict(sampling))

    judge(question="q", correct_answer="Paris", response="Paris")

    assert {k: calls[0][k] for k in sampling} == sampling
    assert set(sampling) == set(judge_module._ALLOWED_REQUEST_FIELDS), (
        "the allowlist grew or shrank; decide deliberately and update this test"
    )


def _allow_a_structured_sampling_field(monkeypatch) -> None:
    """Widen the allowlist for one test, so the copying is tested rather than today's list.

    Every field the allowlist admits is a scalar right now, which closes the nested-sharing hole
    structurally. The copies are what keep it closed if a structured sampling field is ever
    added, and that is what these two tests are for."""
    monkeypatch.setattr(
        judge_module,
        "_ALLOWED_REQUEST_FIELDS",
        MappingProxyType({**judge_module._ALLOWED_REQUEST_FIELDS, "logit_bias": "test only"}),
    )


def test_request_kwargs_are_deep_copied_at_construction(monkeypatch) -> None:
    # A shallow copy leaves nested values shared: editing one after construction would change
    # what a later episode is scored with, and no score would show it.
    _allow_a_structured_sampling_field(monkeypatch)
    calls: List[Dict[str, Any]] = []
    kwargs: Dict[str, Any] = {"logit_bias": {"9891": -1}}
    judge = OpenAIJudge(client=_recording_client(calls), request_kwargs=kwargs)

    judge(question="q", correct_answer="Paris", response="Paris")
    kwargs["logit_bias"]["9891"] = -2  # the caller's copy
    judge.request_kwargs["logit_bias"]["9891"] = -3  # what the property handed back
    judge(question="q", correct_answer="Paris", response="Paris")

    assert calls[0]["logit_bias"] == {"9891": -1}
    assert calls[1]["logit_bias"] == {"9891": -1}
    assert judge.request_kwargs["logit_bias"] == {"9891": -1}


def test_a_client_that_edits_what_it_is_handed_cannot_change_a_later_request(monkeypatch) -> None:
    # The client is arbitrary (an injected one is caller code), and it is handed the kwargs
    # themselves. Without a copy per call, one request rewrites the next.
    _allow_a_structured_sampling_field(monkeypatch)
    sent: List[Any] = []

    def create(**kwargs: Any) -> Any:
        sent.append(copy.deepcopy(kwargs["logit_bias"]))
        kwargs["logit_bias"]["9891"] = -100
        return SimpleNamespace(
            model="probe", choices=[SimpleNamespace(message=SimpleNamespace(content=_REPLY))]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    judge = OpenAIJudge(client=client, request_kwargs={"logit_bias": {"9891": -1}})

    judge(question="q", correct_answer="Paris", response="Paris")
    judge(question="q", correct_answer="Paris", response="Paris")

    assert sent == [{"9891": -1}, {"9891": -1}]


def test_the_verdict_carries_the_model_that_answered() -> None:
    # Provenance names what ran, so the judge has to read it off the response rather than repeat
    # the id it asked for: an alias, a router, or a local endpoint can answer as something else.
    calls: List[Dict[str, Any]] = []
    judge = OpenAIJudge(
        model="configured-alias",
        client=_recording_client(calls, reported_model="actually-ran"),
    )

    verdict = judge(question="q", correct_answer="Paris", response="Paris")

    assert calls[0]["model"] == "configured-alias"  # what was asked for
    assert verdict.model == "actually-ran"  # what answered


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
