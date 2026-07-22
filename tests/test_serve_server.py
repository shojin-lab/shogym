"""The FastMCP server (RFC 008 §5/§6): a real MCP client drives the served Wordle env
in-process — dynamic tool schemas, the describe tool + resource, and _meta passthrough."""

from __future__ import annotations

import json

from fastmcp import Client

import hgym
from hgym.feedback import parse_meta
from hgym.serve import ServedEpisode
from hgym.serve.server import build_server


def _answer(task_idx: int) -> str:
    return hgym.make("wordle_v1")._words[task_idx]


async def test_server_advertises_env_tools_with_schemas() -> None:
    episode = await ServedEpisode.start("wordle_v1", task=0)
    try:
        async with Client(build_server(episode)) as client:
            tools = {t.name: t for t in await client.list_tools()}
            assert {"guess", "terminate", "describe"} <= set(tools)
            # The env's real argument schema is advertised (not the synthesized one).
            assert tools["guess"].inputSchema["properties"]["word"]["type"] == "string"
    finally:
        await episode.close()


async def test_describe_tool_and_resource_agree() -> None:
    episode = await ServedEpisode.start("wordle_v1", task=0)
    try:
        async with Client(build_server(episode)) as client:
            from_tool = (await client.call_tool("describe", {})).data
            resource = await client.read_resource("hgym://task")
            from_resource = json.loads(resource[0].text)
            assert from_tool["env_name"] == "wordle_v1"
            assert from_tool == from_resource
    finally:
        await episode.close()


async def test_guess_then_terminate_over_mcp_client() -> None:
    episode = await ServedEpisode.start("wordle_v1", task=0)
    try:
        async with Client(build_server(episode)) as client:
            res = await client.call_tool("guess", {"word": _answer(0)})
            payload = json.loads(res.content[0].text)
            assert payload["solved"] is True
            _, terminated = parse_meta(res.meta or {})
            assert terminated is False  # solving doesn't auto-terminate

            end = await client.call_tool("terminate", {})
            items, terminated = parse_meta(end.meta or {})
            assert terminated is True
            assert any(i.name == "check_answer" and i.value is True for i in items)
    finally:
        await episode.close()
