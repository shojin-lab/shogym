"""Wrap a :class:`ServedEpisode` in a FastMCP server (RFC 008 §5, §6).

The env's essential tools (from its ``TaskSpec``) become FastMCP tools whose bodies run
one env step; the task contract is published both as the ``hgym://task`` resource and a
``describe`` tool (the lowest-friction path for a harness that only calls tools). Feedback
rides back on each tool result's ``_meta`` (RFC 008 §4). ``run_stdio`` is what
``hgym serve`` executes and what Claude Code / Codex / pi / Hermes spawn as their MCP
server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from fastmcp import FastMCP
from fastmcp.tools import Tool
from fastmcp.tools.tool import ToolResult

from hgym.serve.episode import ServedEpisode
from hgym.task import ToolManifest

_Dispatch = Callable[[str, Dict[str, Any]], Awaitable[ToolResult]]


def _make_tool_fn(manifest: ToolManifest, dispatch: _Dispatch) -> Callable[..., Any]:
    """Build a coroutine with real named parameters from the tool's JSON Schema.

    FastMCP infers a tool from a function signature and rejects ``**kwargs``, so we
    synthesize one parameter per schema property (env-authored names, trusted) whose body
    forwards the collected arguments to ``dispatch``. The advertised schema is overridden
    with the env's own in :func:`_build_tool`, so richer typing survives."""
    props = list(manifest.input_schema.get("properties", {}))
    required = set(manifest.input_schema.get("required", []))
    params = ", ".join(p if p in required else f"{p}=None" for p in props)
    collect = "{" + ", ".join(f"{p!r}: {p}" for p in props) + "}"
    src = f"async def _tool({params}):\n    return await _dispatch({manifest.name!r}, {collect})\n"
    namespace: Dict[str, Any] = {"_dispatch": dispatch}
    exec(src, namespace)  # noqa: S102 — names come from the env's own schema, not input
    fn = namespace["_tool"]
    fn.__name__ = manifest.name
    return fn


def _build_tool(manifest: ToolManifest, dispatch: _Dispatch) -> Tool:
    tool = Tool.from_function(
        _make_tool_fn(manifest, dispatch),
        name=manifest.name,
        description=manifest.description,
    )
    # Advertise the env's real argument schema (the synthesized signature is untyped).
    return tool.model_copy(update={"parameters": manifest.input_schema})


def build_server(episode: ServedEpisode, *, name: Optional[str] = None) -> FastMCP:
    """Build a FastMCP server exposing ``episode``'s tools, ``describe``, and the task
    resource. The same object is served over stdio (`hgym serve`) or driven in-process by
    a FastMCP ``Client`` (the tests and the example harness)."""
    spec = episode.describe()
    server: FastMCP = FastMCP(name=name or f"hgym:{spec.env_name}")

    async def dispatch(tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        result = await episode.call(tool_name, arguments)
        return ToolResult(content=result.content, meta=result.meta or None)

    for manifest in spec.tools:
        server.add_tool(_build_tool(manifest, dispatch))

    @server.tool(name="describe")
    async def describe() -> Dict[str, Any]:
        """Return the task contract (TaskSpec) as a JSON object."""
        return episode.describe().model_dump()

    @server.resource("hgym://task")
    async def task_resource() -> Dict[str, Any]:
        return episode.describe().model_dump()

    return server


async def run_stdio(
    env_name: str,
    *,
    task: Optional[Union[int, str]] = None,
    trace_path: Optional[Union[str, Path]] = None,
) -> None:
    """Start an episode and serve it over stdio until the client disconnects.

    This is the body of ``hgym serve`` — a harness spawns it as its MCP server."""
    episode = await ServedEpisode.start(env_name, task=task, trace_path=trace_path)
    server = build_server(episode)
    try:
        await server.run_async(transport="stdio")
    finally:
        await episode.close()
