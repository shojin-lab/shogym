"""In-process MCP server for tau2's ``banking_knowledge`` domain (non-solo).

Thin per-domain module exposing the ``server`` attribute the in-process transport resolves.
Pinned to the ``bm25_grep`` retrieval variant (rank-bm25, no OpenAI embeddings) so the tool
manifest builds — and the env constructs/serves — fully offline. The benchmark-default
``alltools`` variant (dense embeddings) requires an OpenAI key at construction time and is a
keyed follow-up; switching variants would also change the published tool manifest, so it is
fixed here to match. Requires the ``tau2`` extra (with ``rank-bm25``).
"""

from __future__ import annotations

from shogym.envs.tau2.mcp_server import begin_session, build_domain_server, end_session

__all__ = ["server", "begin_session", "end_session"]

server = build_domain_server(
    "banking_knowledge", solo_mode=False, env_kwargs={"retrieval_variant": "bm25_grep"}
)
