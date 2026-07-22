"""`Env.describe()` publishes the task contract as a `TaskSpec` (RFC 008 §3.1).

Offline: `describe()` reflects static config probed at construction — no session, no
model call.
"""

from __future__ import annotations

import hgym
from hgym.task import ReferenceTemplate, TaskSpec, ToolManifest


def _spec() -> TaskSpec:
    return hgym.make("wordle_v1").describe(task_id="7")


def test_identity_and_horizon() -> None:
    spec = _spec()
    assert isinstance(spec, TaskSpec)
    assert spec.env_name == "wordle_v1"
    assert spec.task_id == "7"
    assert spec.horizon == 6


def test_instructions_are_rendered_framing() -> None:
    spec = _spec()
    assert "Wordle" in spec.instructions and "5-letter" in spec.instructions
    assert "{{" not in spec.instructions  # rendered, not raw template


def test_tool_manifest_covers_guess_and_reserved_terminate() -> None:
    spec = _spec()
    assert isinstance(spec.tools[0], ToolManifest)
    by_name = {t.name: t for t in spec.tools}
    assert {"guess", "terminate"} <= set(by_name)
    assert by_name["terminate"].provenance == "reserved"
    assert by_name["guess"].provenance == "env-mandatory"
    guess_schema = by_name["guess"].input_schema
    assert guess_schema["type"] == "object"
    assert "word" in guess_schema["properties"]
    assert guess_schema["required"] == ["word"]


def test_reference_templates_carry_shape_and_schema() -> None:
    spec = _spec()
    assert isinstance(spec.reference_templates[0], ReferenceTemplate)
    by_role = {t.role: t for t in spec.reference_templates}
    assert set(by_role) == {"system", "user"}
    sys_schema = by_role["system"].variables_schema
    assert sys_schema is not None
    assert "remaining_guesses" in sys_schema["properties"]
    assert "{{ feedback }}" in by_role["user"].template


def test_taskspec_json_roundtrips() -> None:
    spec = _spec()
    assert TaskSpec.model_validate_json(spec.model_dump_json()) == spec
