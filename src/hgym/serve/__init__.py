"""Serving an environment to an external harness (RFC 008 §3.2, §5).

:class:`ServedEpisode` is the transport-independent engine: it opens the env's essential
MCP sessions, turns one incoming tool call into one recorded step, and returns the tool's
result plus the feedback ``_meta`` sidecar. PR4 wraps it in a FastMCP stdio server
(`hgym serve`); the engine itself is exercised in-process, no subprocess required.
"""

from hgym.serve.episode import CallResult, ServedEpisode

__all__ = ["CallResult", "ServedEpisode"]
