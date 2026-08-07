"""Transport implementations for MCP sessions.

``in_process`` and ``stdio`` ship; ``streamable_http`` lands later.
"""

from shogym.mcp.transports.in_process import (
    InProcessMCPSession,
    open_in_process,
)
from shogym.mcp.transports.stdio import (
    StdioMCPSession,
    open_stdio,
)

__all__ = [
    "InProcessMCPSession",
    "open_in_process",
    "StdioMCPSession",
    "open_stdio",
]
