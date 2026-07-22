"""The task specification an environment publishes about itself (RFC 008 §3.1).

An env does three things: **describe** a task, **serve** essential tools (MCP), and
**verify**. This is the *describe* half — the read-only contract a harness (Claude Code,
Codex, pi, Hermes, or a small example loop) reads once to configure itself. Fully
JSON-serializable, so a later PR publishes it verbatim as an MCP resource.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ToolProvenance = Literal["env-mandatory", "reserved"]


class ReferenceTemplate(BaseModel):
    """A template a harness MAY render — advisory, not injected by hgym. ``variables_schema``
    is the JSON Schema of the variables the env promises to fill if it is rendered."""

    role: Literal["system", "user"]
    template: str
    variables_schema: Optional[Dict[str, Any]] = None


class ToolManifest(BaseModel):
    """One advertised tool: name, description (instruction content the model reads), and
    the JSON Schema of its arguments. ``provenance`` flags the reserved ``terminate`` tool
    so a harness finds the stop tool without hard-coding its name."""

    name: str
    description: str
    input_schema: Dict[str, Any]
    provenance: ToolProvenance = "env-mandatory"


class TaskSpec(BaseModel):
    """Everything a harness needs to run one task instance.

    - ``instructions``: the durable task framing (the rendered system template).
    - ``tools``: the essential-tool manifest (env-mandatory + the reserved terminate).
    - ``reference_templates``: optional advisory templates + their variable schemas.
    - ``horizon``: the step budget, enforced env-side by tool-gating (a later PR).
    """

    env_name: str
    task_id: Optional[str] = None
    instructions: str
    tools: List[ToolManifest] = Field(default_factory=list)
    reference_templates: List[ReferenceTemplate] = Field(default_factory=list)
    horizon: Optional[int] = None
