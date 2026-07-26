"""A ``Policy`` drives one dispensed task by issuing native tool calls to the broker.

The real "policy" in this study is **Claude Code** (spawned as a subprocess against the broker
over MCP — see :mod:`.arms`). This module provides the *in-process* seam used for keyless,
deterministic runs:

  - :class:`StubPolicy` — a fixed, model-free policy that exercises the exact plumbing
    (``get_task`` → native tool calls → ``done`` → authoritative seal score) without any model
    spend or credentials. It is what the smoke test and the default held-out eval use to prove
    the measurement spine end-to-end.

A policy receives the task ``framing`` (what ``get_task`` returned) and the ``broker`` and plays
the task by ``await broker.dispatch(tool, args)``. It never sees a target or a score — the broker
scores authoritatively after ``done``.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Protocol

from .broker import Broker


def _text(tool_result: Any) -> str:
    """Extract the text payload from a FastMCP ToolResult (content is a block list or str)."""
    c = getattr(tool_result, "content", tool_result)
    if isinstance(c, str):
        return c
    return "".join(getattr(b, "text", "") for b in c)


class Policy(Protocol):
    async def play(self, framing: Dict[str, Any], broker: Broker) -> None:
        """Carry out one dispensed task by issuing tool calls to ``broker``."""
        ...


class StubPolicy:
    """A deterministic, model-free policy: it does a couple of read-only ``api_search`` probes
    (to exercise tool routing), then calls ``done`` to submit whatever state exists. It does not
    actually complete the workflow, so it scores near-zero — but that is fine: the point is to
    drive the *authoritative scoring path* end-to-end without a model. It proves the seal scores
    the sealed world and the broker records it; it does not pretend to be a capable agent."""

    def __init__(self, probes: List[str] | None = None) -> None:
        self.probes = probes or ["messages", "spreadsheet rows"]
        self.calls: List[Dict[str, Any]] = []

    async def play(self, framing: Dict[str, Any], broker: Broker) -> None:
        tool_names = {t["name"] for t in framing.get("tools", [])}
        if "api_search" in tool_names:
            for q in self.probes:
                out = json.loads(_text(await broker.dispatch("api_search", {"query": q, "top_k": 3})))
                self.calls.append({"tool": "api_search", "query": q, "terminated": out.get("terminated")})
        # Submit + score. `done` is AutomationBench's score terminal: it seals and finalizes.
        done_name = "done" if "done" in tool_names else "terminate"
        out = json.loads(_text(await broker.dispatch(done_name, {})))
        self.calls.append({"tool": done_name, "terminated": out.get("terminated"),
                           "result": out.get("result")})
