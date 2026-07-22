"""Config types for the env-as-center core.

Minimal by design (RFC 008): an env declares only its **tool manifest** (probed from its
MCP servers) and its **instruction interface** — advisory templates plus the pydantic
schemas of the variables they reference. Content is the harness's to own; the env owns
the shape. No agent-loop / metric machinery lives here anymore.
"""

from typing import Any, Dict, List, Literal, Optional, Type

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class ParametersSchema(BaseModel):
    """JSON-Schema for a tool's arguments (the `input_schema` of a manifest entry)."""

    type: Literal["object"]
    properties: Dict[str, Any]
    required: List[str]
    additionalProperties: Optional[bool] = Field(default=False)


class ToolConfig(BaseModel):
    """A probed tool: its name, its description, and its argument schema."""

    description: str
    parameters: ParametersSchema
    name: str

    @field_serializer("name")
    def serialize_name(self, value: str) -> None:
        return None

    @field_serializer("parameters")
    def serialize_parameters(self, value: ParametersSchema) -> str:
        return f"tools/{self.name}.json"


class FunctionConfig(BaseModel):
    """The env's advisory instruction interface (RFC 008 §3.1).

    ``example_*_template`` are minijinja templates a harness MAY render; the ``*_schema``
    pointers declare the variables those templates reference. The env declares this shape;
    the harness owns whether/how to use it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_schema: Optional[Type[BaseModel]] = None
    user_schema: Optional[Type[BaseModel]] = None
    example_system_template: Optional[str] = None
    example_user_template: Optional[str] = None
