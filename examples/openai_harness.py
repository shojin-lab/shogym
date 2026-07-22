"""A ~one-screen example harness: drive a served hgym env as an MCP client (RFC 008 §6).

This is deliberately *not* part of ``hgym`` core — it is one client among the real ones
(Claude Code, Codex, pi, Hermes). It reads the task via the ``describe`` tool, then loops
model -> tool calls -> tool results until the env signals ``hgym/terminate``. The model is
injected as a ``chat`` coroutine so the loop can be exercised offline with a scripted
policy; :func:`openai_chat` is the real (network) implementation.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from fastmcp import Client

from hgym.feedback import parse_meta

ToolCall = Tuple[str, Dict[str, Any]]
# (instructions, tools_manifest, transcript) -> the tool calls to issue this turn.
ChatFn = Callable[[str, List[Dict[str, Any]], List[Dict[str, Any]]], Awaitable[List[ToolCall]]]


async def run_episode(client: Client, chat: ChatFn, *, max_steps: int = 30) -> None:
    """Read the task, then loop until the env terminates (or ``max_steps``)."""
    spec = (await client.call_tool("describe", {})).data
    instructions: str = spec["instructions"]
    tools: List[Dict[str, Any]] = spec["tools"]
    transcript: List[Dict[str, Any]] = []

    for _ in range(max_steps):
        calls = await chat(instructions, tools, transcript)
        if not calls:
            break
        for name, args in calls:
            result = await client.call_tool(name, args, raise_on_error=False)
            content = result.content[0].text if result.content else ""
            _, terminated = parse_meta(result.meta or {})
            transcript.append({"tool": name, "args": args, "result": content})
            if terminated:
                return


def openai_chat(model: str, *, base_url: Optional[str] = None) -> ChatFn:
    """A real model policy over the OpenAI-compatible wire (network; not used in tests)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=base_url)

    def _openai_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]

    async def chat(
        instructions: str, tools: List[Dict[str, Any]], transcript: List[Dict[str, Any]]
    ) -> List[ToolCall]:
        messages: List[Dict[str, Any]] = [{"role": "system", "content": instructions}]
        for turn in transcript:
            messages.append(
                {"role": "user", "content": f"{turn['tool']} -> {turn['result']}"}
            )
        response = await client.chat.completions.create(
            model=model.split("/")[-1],
            messages=messages,
            tools=_openai_tools(tools),
            tool_choice="auto",
            parallel_tool_calls=False,
        )
        tool_calls = response.choices[0].message.tool_calls or []
        return [
            (tc.function.name, json.loads(tc.function.arguments or "{}"))
            for tc in tool_calls
        ]

    return chat
