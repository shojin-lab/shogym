"""Wrap a :class:`ServedEpisode` in a FastMCP server (RFC 008 §5, §6).

The env's essential tools (from its ``TaskSpec``) become FastMCP tools whose bodies run one
served step; the task contract is published as the ``hgym://task`` resource and a
``describe`` tool. Feedback rides each tool result's ``_meta``. ``run_stdio`` is what
``hgym serve`` executes and what an external harness spawns as its MCP server.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional, Union

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from pydantic import PrivateAttr

from hgym.serve.episode import ServedEpisode
from hgym.task import ToolManifest

_Dispatch = Callable[[str, Dict[str, Any]], Awaitable[ToolResult]]

# Server-added control tool; an env may not expose a tool of this name (FastMCP would
# silently replace it, so the manifest would list a tool that dispatches to the control
# endpoint). `terminate` is already reserved by the always-present terminate server.
_DESCRIBE_TOOL = "describe"


class _PassthroughTool(Tool):
    """A FastMCP tool that forwards the caller's arguments to the served episode verbatim.

    We can't use ``Tool.from_function`` here: FastMCP derives the tool from a Python
    signature and rejects ``**kwargs``, but a JSON-Schema property name need not be a valid
    Python identifier (``{"user-name": ...}`` would make ``def _tool(user-name)`` a
    ``SyntaxError``), and synthesizing ``def _tool(prop=None, ...)`` also forwards an
    *omitted* optional as ``None`` rather than leaving it absent. Overriding ``run`` to
    pass the raw argument mapping straight through avoids both: arbitrary schema names
    work, and only the keys the caller actually sent reach ``dispatch``.
    """

    _dispatch: _Dispatch = PrivateAttr()

    def __init__(self, *, dispatch: _Dispatch, **data: Any) -> None:
        super().__init__(**data)
        self._dispatch = dispatch

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        return await self._dispatch(self.name, arguments)


def _build_tool(manifest: ToolManifest, dispatch: _Dispatch) -> Tool:
    """Expose one env tool: advertise its real JSON-Schema, forward raw args to dispatch."""
    return _PassthroughTool(
        dispatch=dispatch,
        name=manifest.name,
        description=manifest.description,
        parameters=manifest.input_schema,
    )


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
        if manifest.name == _DESCRIBE_TOOL:
            raise ValueError(
                f"env tool name {_DESCRIBE_TOOL!r} collides with the server's reserved "
                f"control tool; an env may not expose a tool named {_DESCRIBE_TOOL!r}"
            )
        server.add_tool(_build_tool(manifest, dispatch))

    @server.tool(name=_DESCRIBE_TOOL)
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
    """Start an episode and serve it over stdio until the client disconnects — the body of
    ``hgym serve``, which a harness spawns as its MCP server."""
    episode = await ServedEpisode.start(env_name, task=task, trace_path=trace_path)
    try:
        # Inside the guard: build_server can raise (e.g. a reserved-name collision) after
        # the episode has opened sessions + pushed state that must be released.
        server = build_server(episode)
        await server.run_async(transport="stdio")
    finally:
        await episode.close()
