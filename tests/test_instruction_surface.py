"""The instruction surface (RFC 002): a loaded harness's system_template overrides
the function's own example_system_template, while the env still owns the variables.

Offline throughout: a capturing fake ``ModelClient`` records the rendered messages.
"""

from __future__ import annotations

from types import SimpleNamespace

from pydantic import BaseModel

from hgym.agents.openai.agent import OpenAIAgent
from hgym.agents.openai.utils import parse_observation
from hgym.harness import Harness
from hgym.models import CompletionRequest
from hgym.types import (
    FunctionConfigChat,
    FunctionConfigs,
    Observation,
    TextResultContentBlock,
)


class _SystemVars(BaseModel):
    value: str


class _CapturingClient:
    def __init__(self) -> None:
        self.request: CompletionRequest | None = None

    async def complete(self, request: CompletionRequest):
        self.request = request
        message = SimpleNamespace(tool_calls=None, content="ok", audio=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _vars_obs(value: str) -> Observation:
    """An observation whose env supplies system *variables* (a dict matching a schema)."""
    return Observation(
        function_name="agent",
        system=[TextResultContentBlock(value={"value": value})],
        messages=[],
    )


def test_parse_observation_prefers_override_template() -> None:
    # The env owns the variable (``value``); the function template and the harness
    # override differ only in wording. The override should win, with the same variable.
    fc = FunctionConfigChat(
        system_schema=_SystemVars, example_system_template="ENV: {{ value }}"
    )
    obs = _vars_obs("hello")

    default = parse_observation(obs, fc)
    overridden = parse_observation(obs, fc, "HARNESS: {{ value }}")

    assert default[0]["content"] == "ENV: hello"
    assert overridden[0]["content"] == "HARNESS: hello"


def test_override_emits_system_even_without_function_template() -> None:
    # Function declares no template; the harness supplies a (variable-free) one — a
    # system message must still be emitted (the gate now considers the override).
    fc = FunctionConfigChat()
    obs = Observation(function_name="agent", messages=[])
    messages = parse_observation(obs, fc, "ONLY-HARNESS INSTRUCTIONS")
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "ONLY-HARNESS INSTRUCTIONS"


def test_none_override_defers_to_function_template() -> None:
    fc = FunctionConfigChat(
        system_schema=_SystemVars, example_system_template="ENV: {{ value }}"
    )
    obs = _vars_obs("x")
    assert parse_observation(obs, fc, None)[0]["content"] == "ENV: x"


async def test_agent_applies_system_template_override() -> None:
    functions = FunctionConfigs()
    functions["agent"] = FunctionConfigChat(
        system_schema=_SystemVars, example_system_template="ENV: {{ value }}"
    )
    client = _CapturingClient()
    agent = OpenAIAgent(
        "openai/gpt-5.4-nano",
        functions,
        system_template="HARNESS: {{ value }}",
        client=client,
    )
    await agent.act(_vars_obs("world"))
    assert client.request is not None
    assert client.request.messages[0]["content"] == "HARNESS: world"


async def test_from_harness_wires_inference_and_instruction() -> None:
    functions = FunctionConfigs()
    functions["agent"] = FunctionConfigChat(
        system_schema=_SystemVars, example_system_template="ENV: {{ value }}"
    )
    harness = Harness(
        model="openai/gpt-5.4-nano",
        inference_params={"temperature": 0.1},
        system_template="HARNESS: {{ value }}",
    )
    client = _CapturingClient()
    agent = OpenAIAgent.from_harness(harness, functions, client=client)
    await agent.act(_vars_obs("world"))

    assert client.request is not None
    assert client.request.model == "openai/gpt-5.4-nano"
    assert client.request.params == {"temperature": 0.1}
    assert client.request.messages[0]["content"] == "HARNESS: world"
