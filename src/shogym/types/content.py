"""Tool content blocks used by the MCP layer.

Minimal (RFC 008): the env-as-center path only moves tool calls and tool results across
the MCP boundary — no messages, observations, images, audio, or thinking blocks.
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel


class ToolCallContentBlock(BaseModel):
    """A tool invocation: its arguments, a correlation id, and the tool name."""

    arguments: Dict[str, Any]
    id: str
    name: Optional[str] = None


class ToolResultContentBlock(BaseModel):
    """The result of a tool execution, correlated to its call by ``id``."""

    result: str
    id: str
    name: Optional[str] = None
