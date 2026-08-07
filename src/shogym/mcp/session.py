"""The MCP session protocol used to probe and call an env's tools."""

from typing import Any, Dict, List, Protocol, runtime_checkable

from shogym.types.config import ToolConfig
from shogym.types.content import ToolResultContentBlock


@runtime_checkable
class MCPSession(Protocol):
    """Per-episode handle to a running MCP server.

    Implementations are returned by transport-specific openers (e.g.
    ``open_in_process``). The session is keyed to a single ``session_id`` for the
    duration of an episode; a fresh session is opened per episode.

    Concrete implementations must:
      - inject the episode's ``session_id`` as a hidden ``_session_id``
        argument into ``call_tool`` invocations **only for tools whose input
        schema declares ``_session_id``**. Tools that don't declare it must be
        called without it: a strict schema (``additionalProperties: false``)
        would otherwise reject the call. (The env layer strips ``_session_id``
        from the recorded trajectory.)
      - surface tool errors — including failures while lazily listing tools —
        as ``ToolResultContentBlock`` results rather than raising, so that one
        failing tool does not kill the episode
      - be idempotent on ``close``
    """

    @property
    def session_id(self) -> str:
        """The episode-scoped session id this handle is bound to."""
        ...

    async def list_tools(self) -> List[ToolConfig]:
        """Return the tools advertised by the underlying MCP server."""
        ...

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
        *,
        tool_call_id: str,
    ) -> ToolResultContentBlock:
        """Dispatch a tool call and return the result block.

        ``tool_call_id`` is the id from the originating
        ``ToolCallContentBlock`` so the resulting ``ToolResultContentBlock``
        can be correlated.
        """
        ...

    async def close(self) -> None:
        """Tear down the session. Idempotent."""
        ...
