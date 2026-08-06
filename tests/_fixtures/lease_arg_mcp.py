"""In-process MCP server for an env whose own tool takes an argument named ``lease``.

Nothing here is special to the serving layer: this is an ordinary env tool that happens to
name one of its arguments with a word the stream also uses to route calls when several
episodes are live. The handler echoes the argument straight back, so a test can see exactly
what reached the env.
"""

from __future__ import annotations

from typing import Any, Dict

from fastmcp import FastMCP

server: FastMCP = FastMCP(name="fixture_lease_arg")


@server.tool
def lookup_lease(lease: str, _session_id: str) -> Dict[str, Any]:
    """Look up a lease agreement by its identifier (the env's own notion of a lease)."""
    return {"looked_up": lease}
