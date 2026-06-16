"""OpenAIAgent routes model + inference params through the ModelClient seam.

Offline: a capturing fake ``ModelClient`` records the ``CompletionRequest`` and
returns a canned response, so ``act()`` is exercised end to end with no network.
"""

from __future__ import annotations

from types import SimpleNamespace

from hgym.agents.openai.agent import OpenAIAgent
from hgym.models import CompletionRequest
from hgym.types import FunctionConfigChat, FunctionConfigs, Observation


class _CapturingClient:
    def __init__(self) -> None:
        self.request: CompletionRequest | None = None

    async def complete(self, request: CompletionRequest):
        self.request = request
        message = SimpleNamespace(tool_calls=None, content="ok", audio=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def test_agent_passes_model_and_inference_params() -> None:
    functions = FunctionConfigs()
    functions["agent"] = FunctionConfigChat()
    client = _CapturingClient()
    agent = OpenAIAgent(
        "openai/gpt-5.4-nano",
        functions,
        inference_params={"temperature": 0.3, "max_tokens": 128},
        client=client,
    )

    # Empty observation: parse_observation yields no messages, so no template
    # rendering — we only care that the request carries the inference surface.
    obs = Observation(function_name="agent", messages=[])
    action = await agent.act(obs)

    assert client.request is not None
    assert client.request.model == "openai/gpt-5.4-nano"
    assert client.request.params == {"temperature": 0.3, "max_tokens": 128}
    # The canned text response parses to one text content block.
    assert len(action) == 1
