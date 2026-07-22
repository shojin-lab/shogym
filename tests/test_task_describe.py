"""`Env.describe()` publishes the env's task contract as a `TaskSpec` (RFC 008 §3.1).

Offline: `describe()` reflects static config probed at construction — no session, no
model call — so these assertions run right after `make(...)`.
"""

from __future__ import annotations

import hgym
from hgym.task import ReferenceTemplate, TaskSpec, ToolManifest


def _wordle_spec() -> TaskSpec:
    return hgym.make("wordle_v1").describe(task_id="7")


def test_describe_returns_taskspec_with_identity() -> None:
    spec = _wordle_spec()
    assert isinstance(spec, TaskSpec)
    assert spec.env_name == "wordle_v1"  # stamped by make()
    assert spec.task_id == "7"
    assert spec.horizon == 6  # MAX_GUESSES


def test_instructions_are_the_rendered_system_framing() -> None:
    spec = _wordle_spec()
    # The durable task framing: the rules and the tool docs, not a per-turn message.
    assert "Wordle" in spec.instructions
    assert "5-letter" in spec.instructions
    assert "{{" not in spec.instructions  # rendered, not raw template


def test_tool_manifest_covers_guess_and_reserved_terminate() -> None:
    spec = _wordle_spec()
    by_name = {t.name: t for t in spec.tools}
    assert isinstance(spec.tools[0], ToolManifest)

    assert "guess" in by_name and "terminate" in by_name
    # terminate is flagged reserved so a harness finds the stop tool without a hardcode.
    assert by_name["terminate"].provenance == "reserved"
    assert by_name["guess"].provenance == "env-mandatory"

    # The guess tool carries a real JSON Schema for its arguments.
    guess_schema = by_name["guess"].input_schema
    assert guess_schema["type"] == "object"
    assert "word" in guess_schema["properties"]
    assert guess_schema["required"] == ["word"]


def test_reference_templates_carry_shape_and_schema() -> None:
    spec = _wordle_spec()
    by_role = {t.role: t for t in spec.reference_templates}
    assert isinstance(spec.reference_templates[0], ReferenceTemplate)

    assert set(by_role) == {"system", "user"}
    # The env owns the *shape*: the system template's variable schema is published.
    sys_schema = by_role["system"].variables_schema
    assert sys_schema is not None
    assert "remaining_guesses" in sys_schema["properties"]
    # The user template references `feedback`.
    assert "{{ feedback }}" in by_role["user"].template


def test_taskspec_is_json_serializable() -> None:
    # A later PR publishes this verbatim as an MCP resource, so it must round-trip.
    spec = _wordle_spec()
    dumped = spec.model_dump_json()
    restored = TaskSpec.model_validate_json(dumped)
    assert restored == spec
