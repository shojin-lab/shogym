"""Unit tests for the inference seam (ModelClient / CompletionRequest)."""

from __future__ import annotations

from hgym.models import CompletionRequest, build_extra_kwargs, split_model


def test_split_model_with_provider() -> None:
    assert split_model("openai/gpt-5.4-nano") == ("openai", "gpt-5.4-nano")
    assert split_model("openrouter/anthropic/claude") == (
        "openrouter",
        "anthropic/claude",
    )


def test_split_model_bare() -> None:
    assert split_model("gpt-5.4-nano") == (None, "gpt-5.4-nano")


def test_build_extra_kwargs_no_tools_only_params() -> None:
    req = CompletionRequest(
        model="openai/x", messages=[], params={"temperature": 0.7, "max_tokens": 256}
    )
    kw = build_extra_kwargs(req)
    assert kw == {"temperature": 0.7, "max_tokens": 256}
    assert "tools" not in kw and "tool_choice" not in kw


def test_build_extra_kwargs_with_tools() -> None:
    tools = [{"type": "function", "function": {"name": "t"}}]
    req = CompletionRequest(
        model="openai/x",
        messages=[],
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        params={"temperature": 0.2},
    )
    kw = build_extra_kwargs(req)
    assert kw["tools"] == tools
    assert kw["tool_choice"] == "required"
    assert kw["parallel_tool_calls"] is False
    assert kw["temperature"] == 0.2


def test_build_extra_kwargs_tool_choice_defaults_auto() -> None:
    req = CompletionRequest(model="openai/x", messages=[], tools=[])
    assert build_extra_kwargs(req)["tool_choice"] == "auto"


def test_build_extra_kwargs_omits_none_parallel() -> None:
    req = CompletionRequest(model="openai/x", messages=[], tools=[], parallel_tool_calls=None)
    assert "parallel_tool_calls" not in build_extra_kwargs(req)
