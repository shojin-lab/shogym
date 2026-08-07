"""MCP integration for shogym — the tool substrate.

Environments expose their tools as MCP servers; the env base and the serving layer open
per-episode sessions (``_open_session_for_spec``) and call tools over the session protocol.
Transports: in-process and stdio.
"""

from shogym.mcp.session import MCPSession
from shogym.mcp.types import (
    MCPServerSpec,
    MCPTransport,
    ToolNameConflictError,
    UnknownToolError,
)

__all__ = [
    "MCPServerSpec",
    "MCPSession",
    "MCPTransport",
    "ToolNameConflictError",
    "UnknownToolError",
]
