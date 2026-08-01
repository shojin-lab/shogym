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
from fastmcp.server.middleware import Middleware
from fastmcp.tools import Tool, ToolResult
from pydantic import PrivateAttr

from hgym.feedback.wire import build_meta
from hgym.serve.episode import ServedEpisode
from hgym.task import ToolManifest

_Dispatch = Callable[[str, Dict[str, Any]], Awaitable[ToolResult]]

# Server-added control tool; an env may not expose a tool of this name (FastMCP would
# silently replace it, so the manifest would list a tool that dispatches to the control
# endpoint). `terminate` is already reserved by the always-present terminate server.
_DESCRIBE_TOOL = "describe"


class _IngressGate(Middleware):
    """Request-level ingress gate: the post-seal tombstone at the MCP request boundary.

    A universal post-seal tombstone cannot live at the per-tool dispatcher: an **unknown**
    tool never reaches the Python dispatcher (FastMCP rejects it first), and `describe` /
    `tools/list` / resource reads don't route through ``episode.call`` at all. So the gate sits
    at the MCP request boundary as FastMCP middleware.

    **Post-seal method policy** (the invariant is "no request reaches task/env *state* after the
    seal"):

    - **Tombstoned** (``on_call_tool`` returns the generic terminal tombstone with **no inward
      dispatch and no recorded step**): every ``tools/call`` — including a repeat of the score
      terminal, the reserved ``terminate``, and any *unknown* tool — and, by the same hook, any
      ToolSearch/deferred-tool activation (no such tool exists in this checkout, but any future
      one is a ``tools/call`` and is covered here).
    - **Read-only, allowed** (pass straight through): ``describe`` (the immutable task spec),
      ``tools/list`` (``on_list_tools``), ``resources/read`` (``on_read_resource``), and
      ``resources/list``/``resources/templates/list`` — they expose no task state and no
      verdict, and harnesses legitimately re-read them.

    The gate only ever fires for a **sealed** seal-enabled episode; a non-seal episode is
    always OPEN, so the gate is inert and the server behaves exactly as before.
    """

    def __init__(self, episode: ServedEpisode) -> None:
        self._episode = episode

    async def on_call_tool(self, context, call_next):  # type: ignore[override]
        if self._episode.sealed and context.message.name != _DESCRIBE_TOOL:
            return ToolResult(
                content="<episode sealed; no further tool calls are dispatched>",
                meta=build_meta(terminate=True) or None,
            )
        return await call_next(context)


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


def build_tool(
    manifest: ToolManifest,
    dispatch: _Dispatch,
    *,
    name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> Tool:
    """Expose one env tool: advertise its real JSON-Schema, forward raw args to dispatch.

    ``name`` and ``parameters`` override what is advertised, for a server that publishes the
    same env tool under a different public name or behind a wrapper schema. ``dispatch``
    receives the **advertised** name, so such a caller maps it back itself."""
    return _PassthroughTool(
        dispatch=dispatch,
        name=name if name is not None else manifest.name,
        description=manifest.description,
        parameters=parameters if parameters is not None else manifest.input_schema,
    )


# Kept for callers that reached for this helper while it was private.
_build_tool = build_tool


def build_server(episode: ServedEpisode, *, name: Optional[str] = None) -> FastMCP:
    """Build a FastMCP server exposing ``episode``'s tools, ``describe``, and the task
    resource. The same object is served over stdio (`hgym serve`) or driven in-process by
    a FastMCP ``Client`` (the tests and the example harness)."""
    spec = episode.describe()
    server: FastMCP = FastMCP(name=name or f"hgym:{spec.env_name}")
    # Request-level ingress gate. Inert for a non-seal episode (always OPEN); tombstones every
    # post-seal `tools/call` (incl. unknown tools) for a seal-enabled one.
    server.add_middleware(_IngressGate(episode))

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
    # Restart recovery runs transport-independently inside ServedEpisode.start (at store
    # construction), so `hgym serve`, `evaluate()`, and every in-process caller share it — no
    # extra step is needed here.
    episode = await ServedEpisode.start(env_name, task=task, trace_path=trace_path)
    try:
        # Inside the guard: build_server can raise (e.g. a reserved-name collision) after
        # the episode has opened sessions + pushed state that must be released.
        server = build_server(episode)
        await server.run_async(transport="stdio")
    finally:
        await episode.close()
