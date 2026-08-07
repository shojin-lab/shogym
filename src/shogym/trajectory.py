"""The trajectory an env verifies over (RFC 008).

A tool call *is* the step, so the trajectory is just the flat sequence of tool calls the
harness made and the results the env's tools returned — no messages, observations, or
agent state. ``verify`` is a pure function over this list.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Step:
    """One tool call and its result. ``index`` is 1-based within the episode."""

    index: int
    tool: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    result: str = ""


Trajectory = List[Step]
