"""The FastMCP server (RFC 008 §5/§6): a real MCP client drives the served Wordle env
in-process — dynamic tool schemas, the describe tool + resource, and _meta passthrough."""

from __future__ import annotations

import json

import pytest

from fastmcp import Client, FastMCP

import shogym
from shogym.feedback import parse_meta
from shogym.serve import ServedEpisode
from shogym.serve.server import build_server


def _answer(task_idx: int) -> str:
    return shogym.make("wordle_v1")._words[task_idx]


def test_build_server_rejects_describe_tool_collision() -> None:
    # `describe` is a reserved control tool; an env that declares one would have it
    # silently replaced by FastMCP, so build_server must reject the collision.
    from shogym.task import TaskSpec, ToolManifest

    class _FakeEpisode:
        def describe(self):
            return TaskSpec(
                env_name="x",
                instructions="",
                tools=[ToolManifest(
                    name="describe", description="d",
                    input_schema={"type": "object", "properties": {}},
                )],
            )

    with pytest.raises(ValueError, match="reserved control tool"):
        build_server(_FakeEpisode())  # type: ignore[arg-type]


async def test_passthrough_tool_handles_nonidentifier_and_optional_args() -> None:
    # A tool schema's property names need not be Python identifiers, and an omitted
    # optional must not be forwarded as None. The passthrough adapter must handle both
    # (the old signature-synthesis raised SyntaxError / forwarded {"note": None}).
    from fastmcp.tools import ToolResult

    from shogym.serve.server import _build_tool
    from shogym.task import ToolManifest

    seen: dict = {}

    async def dispatch(name, arguments):
        seen["name"], seen["args"] = name, arguments
        return ToolResult(content="ok")

    manifest = ToolManifest(
        name="greet",
        description="d",
        input_schema={
            "type": "object",
            "properties": {"user-name": {"type": "string"}, "note": {"type": "string"}},
            "required": ["user-name"],
            "additionalProperties": False,
        },
    )
    server = FastMCP(name="t")
    server.add_tool(_build_tool(manifest, dispatch))
    async with Client(server) as client:
        await client.call_tool("greet", {"user-name": "ann"})  # omit optional 'note'

    assert seen["name"] == "greet"
    assert seen["args"] == {"user-name": "ann"}  # non-identifier preserved, no None note


async def test_server_advertises_env_tools_with_schemas() -> None:
    episode = await ServedEpisode.start("wordle_v1", task=0)
    try:
        async with Client(build_server(episode)) as client:
            tools = {t.name: t for t in await client.list_tools()}
            assert {"guess", "terminate", "describe"} <= set(tools)
            assert tools["guess"].inputSchema["properties"]["word"]["type"] == "string"
    finally:
        await episode.close()


async def test_describe_tool_and_resource_agree() -> None:
    episode = await ServedEpisode.start("wordle_v1", task=0)
    try:
        async with Client(build_server(episode)) as client:
            from_tool = (await client.call_tool("describe", {})).data
            resource = await client.read_resource("shogym://task")
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
            assert terminated is False

            end = await client.call_tool("terminate", {})
            items, terminated = parse_meta(end.meta or {})
            assert terminated is True
            assert any(i.name == "check_answer" and i.value is True for i in items)
    finally:
        await episode.close()
