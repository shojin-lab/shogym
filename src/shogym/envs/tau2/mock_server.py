"""In-process MCP server for tau2's ``mock`` domain (solo mode).

Thin per-domain module: it exposes the ``server`` attribute the in-process transport
resolves, built by the shared bridge factory. Importing it constructs a tau2 ``mock``
environment to enumerate the domain's tools, so it requires the ``tau2`` extra — but it is
only imported when a ``tau2_mock`` env is constructed (manifest probe) or served.

Solo mode uses tau2's ``DummyUser`` (no user-simulator LLM), so the whole slice — engine,
tools, and evaluator — runs without a model call; ``db_match`` / ``action`` scoring is
deterministic. Importing tau2's registry still pulls in litellm, which reaches for a model-cost
map unless ``LITELLM_LOCAL_MODEL_COST_MAP=true``, so this is keyless rather than network-free.
"""

from __future__ import annotations

from shogym.envs.tau2.mcp_server import begin_session, build_domain_server, end_session

__all__ = ["server", "begin_session", "end_session"]

server = build_domain_server("mock", solo_mode=True)
