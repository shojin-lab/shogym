"""The inference seam: a thin ``ModelClient`` over the OpenAI-compatible wire.

RFC 001 Section 6 (the inference surface): the model id and the inference-API request
config that travels with it (temperature, max_tokens, reasoning_effort, ...) are one
optimizable surface, distinct from the messages and the tools. ``CompletionRequest``
carries exactly that, and ``ModelClient`` is the one seam every gateway choice hides
behind (a ``base_url`` points the default client at OpenAI, vLLM, Ollama, OpenRouter,
a proxy, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable

from openai import AsyncOpenAI


def split_model(model: str) -> Tuple[Optional[str], str]:
    """Split a ``provider/model`` string (inspect-ai convention).

    ``"openai/gpt-5.4-nano"`` -> ``("openai", "gpt-5.4-nano")``;
    a bare ``"gpt-5.4-nano"`` -> ``(None, "gpt-5.4-nano")``.
    """
    if "/" in model:
        provider, _, name = model.partition("/")
        return provider, name
    return None, model


@dataclass
class CompletionRequest:
    """Everything a chat-completion call takes: the messages, the tools, and the
    inference surface (``model`` + ``params``)."""

    model: str
    messages: List[Any]
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    parallel_tool_calls: Optional[bool] = None
    params: Dict[str, Any] = field(default_factory=dict)  # temperature, max_tokens, ...


@runtime_checkable
class ModelClient(Protocol):
    async def complete(self, request: CompletionRequest) -> Any: ...


def build_extra_kwargs(request: CompletionRequest) -> Dict[str, Any]:
    """The non-messages, non-model kwargs for a chat completion: the inference params,
    plus tools/tool_choice/parallel only when tools are present (matching the prior
    agent behavior, so an env with no tools sends none of these)."""
    kwargs: Dict[str, Any] = dict(request.params)
    if request.tools is not None:
        kwargs["tools"] = request.tools
        kwargs["tool_choice"] = (
            request.tool_choice if request.tool_choice is not None else "auto"
        )
        if request.parallel_tool_calls is not None:
            kwargs["parallel_tool_calls"] = request.parallel_tool_calls
    return kwargs


class OpenAICompatClient:
    """``ModelClient`` over the OpenAI-compatible wire schema (one dependency:
    ``openai``). A ``base_url`` covers OpenAI, vLLM, Ollama, OpenRouter, Vercel, a
    LiteLLM proxy, etc. The ``provider/`` prefix on the model string is stripped (the
    base_url selects the endpoint); richer provider routing is a later extra."""

    def __init__(
        self, *, base_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        client_kwargs: Dict[str, Any] = {}
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        if api_key is not None:
            client_kwargs["api_key"] = api_key
        self._client = AsyncOpenAI(**client_kwargs)

    async def complete(self, request: CompletionRequest) -> Any:
        _, name = split_model(request.model)
        return await self._client.chat.completions.create(
            model=name,
            messages=request.messages,
            **build_extra_kwargs(request),
        )
