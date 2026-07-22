"""The task specification an environment publishes about itself (RFC 008 §3.1).

In the env-as-center design, an environment does exactly three things: it
**describes** a task, it **serves** essential tools (MCP), and it **verifies**. This
module is the *describe* half — the read-only contract a harness (Claude Code, Codex,
pi, Hermes, or a small example loop) reads once at setup to configure itself.

A :class:`TaskSpec` is fully JSON-serializable (schemas are carried as JSON Schema
dicts, never as class references), because a later PR publishes it verbatim as an MCP
resource (``hgym://task``) and via a ``describe`` tool. Nothing here opens a session or
makes a model call; it only reflects the env's static configuration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ToolProvenance = Literal["env-mandatory", "reserved"]


class ReferenceTemplate(BaseModel):
    """A template a harness MAY render — advisory, not injected by hgym.

    The env owns the *shape* (``variables_schema``: the JSON Schema of the variables it
    promises to fill if the template is rendered); the harness owns whether to use the
    template at all. hgym never renders instructions into a model call itself — in the
    env-as-center design it makes no model calls.
    """

    role: Literal["system", "user"]
    template: str
    variables_schema: Optional[Dict[str, Any]] = None


class ToolManifest(BaseModel):
    """One advertised tool: its name, its description (instruction content the model
    reads to decide when to call it), and the JSON Schema of its arguments.

    ``provenance`` distinguishes the reserved ``terminate`` tool (the universal
    completion signal) from ordinary env-mandatory tools, so a harness can find the
    stop tool without hard-coding its name.
    """

    name: str
    description: str
    input_schema: Dict[str, Any]
    provenance: ToolProvenance = "env-mandatory"


class TaskSpec(BaseModel):
    """Everything a harness needs to run one task instance, and nothing it doesn't.

    - ``instructions``: the durable task framing to hand the harness (for the built-in
      envs this is the rendered system template — the rules plus the tool docs).
    - ``tools``: the essential-tool manifest (env-mandatory + the reserved terminate).
    - ``reference_templates``: optional advisory templates + their variable schemas,
      for a harness that wants the structured system/user split rather than the flat
      ``instructions`` blob.
    - ``output_schema``: JSON Schema of a structured final answer, if the task wants one.
    - ``horizon``: the step budget; enforced env-side by tool-gating (a later PR), not
      by any loop hgym owns.
    """

    env_name: str
    task_id: Optional[str] = None
    instructions: str
    tools: List[ToolManifest] = Field(default_factory=list)
    reference_templates: List[ReferenceTemplate] = Field(default_factory=list)
    output_schema: Optional[Dict[str, Any]] = None
    horizon: Optional[int] = None
