"""Open a per-episode MCP session for a server spec.

``_open_session_for_spec`` is the one primitive the env-as-center paths build on: the env
base probes tools through it, and the serving layer opens each essential server's session
through it. It dispatches to the transport opener for the spec's transport.
"""

from __future__ import annotations

from hgym.mcp.session import MCPSession
from hgym.mcp.transports import open_in_process, open_stdio
from hgym.mcp.types import MCPServerSpec


async def _open_session_for_spec(spec: MCPServerSpec, *, session_id: str) -> MCPSession:
    if spec.transport == "in_process":
        return await open_in_process(spec, session_id=session_id)
    if spec.transport == "stdio":
        return await open_stdio(spec, session_id=session_id)
    if spec.transport == "streamable_http":
        raise NotImplementedError("streamable_http MCP transport is not yet implemented")
    raise ValueError(f"unknown transport: {spec.transport!r}")  # unreachable
