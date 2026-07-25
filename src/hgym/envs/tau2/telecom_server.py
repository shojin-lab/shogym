"""In-process MCP server for tau2's ``telecom`` domain (non-solo; user simulator).

Thin per-domain module exposing the ``server`` attribute the in-process transport resolves.
Requires the ``tau2`` extra; imported only when a ``tau2_telecom`` env is constructed or served.
Running an episode drives tau2's user simulator (an LLM) — pass
``user_llm_args={"mock_response": "..."}`` via env config for a deterministic offline user.
"""

from __future__ import annotations

from hgym.envs.tau2.mcp_server import begin_session, build_domain_server, end_session

__all__ = ["server", "begin_session", "end_session"]

server = build_domain_server("telecom", solo_mode=False)
