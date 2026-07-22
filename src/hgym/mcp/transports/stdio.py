"""Stdio MCP transport backed by ``fastmcp.client.transports.StdioTransport``.

Spec ``transport="stdio"`` launches a subprocess from ``spec.command`` (argv)
and speaks MCP over its stdin/stdout. This is the transport a standalone tool
server uses — a FastMCP server run out-of-process, e.g.::

    MCPServerSpec(
        name="dictionary",
        transport="stdio",
        command=["uv", "run", "python", "/path/to/dictionary_server.py"],
        env={"LOG_LEVEL": "warning"},
    )

Each session spawns its own process (``keep_alive=False``) so per-episode
sessions stay isolated and ``close()`` terminates the process. The session
contract itself lives in :mod:`hgym.mcp.transports._client_session`; this
module only constructs the subprocess-backed client.
"""

from __future__ import annotations

from typing import Optional

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from hgym.mcp.transports._client_session import ClientMCPSession
from hgym.mcp.types import MCPServerSpec


class StdioMCPSession(ClientMCPSession):
    """Episode-scoped session over a subprocess MCP server."""


async def open_stdio(spec: MCPServerSpec, *, session_id: str) -> StdioMCPSession:
    """Open a stdio MCP session for ``spec`` keyed by ``session_id``.

    Spawns the server subprocess from ``spec.command`` and completes the MCP
    handshake before returning. Raises if the transport is wrong, the command
    is empty, or the subprocess fails to start / handshake.
    """
    if spec.transport != "stdio":
        raise ValueError(
            f"open_stdio requires transport=`stdio`, got {spec.transport!r}"
        )
    # `command` presence is enforced by Pydantic; guard the empty-list case and
    # narrow the type for the splat below.
    if not spec.command:
        raise ValueError("`MCPServerSpec.command` is required for `stdio` transport")

    transport = StdioTransport(
        command=spec.command[0],
        args=list(spec.command[1:]),
        env=dict(spec.env) if spec.env else None,
        # One subprocess per session: don't let fastmcp keep the process alive
        # for reuse across context-manager entries — `close()` must reap it.
        keep_alive=False,
    )
    client: Optional[Client] = Client(transport=transport)
    try:
        await client.__aenter__()
    except BaseException:
        client = None
        raise

    return StdioMCPSession(client=client, session_id=session_id, spec=spec)
