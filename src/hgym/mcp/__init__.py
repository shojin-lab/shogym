"""MCP integration for hgym.

This subpackage exposes the contract used by `ToolUsingEnv` to source tools
from MCP servers. PR 1 lands the types and session protocol; transports and
the toolset land in subsequent PRs.
"""

from hgym.mcp.config import load_mcp_server_specs, load_mcp_toolset
from hgym.mcp.session import MCPSession
from hgym.mcp.toolset import MCPToolset
from hgym.mcp.types import (
    MCPServerSpec,
    MCPTransport,
    ToolNameConflictError,
    UnknownToolError,
)

__all__ = [
    "MCPServerSpec",
    "MCPSession",
    "MCPToolset",
    "MCPTransport",
    "ToolNameConflictError",
    "UnknownToolError",
    "load_mcp_server_specs",
    "load_mcp_toolset",
]
