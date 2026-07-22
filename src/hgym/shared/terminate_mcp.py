"""Built-in ``terminate`` tool — the reserved episode-completion signal.

Every env serves this server (see ``ToolUsingEnv.essential_specs``). Calling ``terminate``
signals the episode is over (alongside the horizon); the result is a no-op acknowledgement —
the terminate call itself is the terminal signal, detected by name.

The name ``terminate`` is reserved — no env-mandatory or user-supplied server may expose a
tool with that name.
"""

from typing import Any, Dict

from fastmcp import FastMCP

server: FastMCP = FastMCP(name="terminate")

TERMINATE_TOOL_NAME = "terminate"


@server.tool(name=TERMINATE_TOOL_NAME)
def terminate(_session_id: str) -> Dict[str, Any]:
    """End the current episode.

    Returns a stub acknowledgement; the env layer detects the call and ends
    the episode.
    """
    return {"acknowledged": True}
