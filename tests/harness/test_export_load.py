"""Round-trip tests for the harness directory format (export -> load)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from hgym.harness import Harness, export_harness, load_harness


def test_export_writes_expected_layout(tmp_path: Path) -> None:
    out = export_harness("wordle_v1", "openai/gpt-5.4-nano", tmp_path / "h")

    assert (out / "harness.toml").exists()
    # wordle_v1 declares a system template, so it must be exported.
    assert (out / "instruction" / "system.minijinja").exists()
    # Baseline harness: no extras, so no tools.toml.
    assert not (out / "tools.toml").exists()

    with (out / "harness.toml").open("rb") as f:
        doc = tomllib.load(f)
    assert doc["inference"]["model"] == "openai/gpt-5.4-nano"
    assert doc["limits"]["horizon"] == 6  # MAX_GUESSES


def test_roundtrip_preserves_fields(tmp_path: Path) -> None:
    out = export_harness(
        "wordle_v1",
        "openai/gpt-5.4-nano",
        tmp_path / "h",
        inference_params={"temperature": 0.7, "max_tokens": 2048},
    )
    h = load_harness(out)

    assert isinstance(h, Harness)
    assert h.model == "openai/gpt-5.4-nano"
    assert h.inference_params == {"temperature": 0.7, "max_tokens": 2048}
    assert h.horizon == 6
    assert h.extra_specs == []
    assert h.system_template is not None and len(h.system_template) > 0
    # The exported template text must equal what load reads back.
    assert h.system_template == (out / "instruction" / "system.minijinja").read_text()


def test_load_missing_harness_toml_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_harness(tmp_path / "does_not_exist")


def test_load_missing_model_raises(tmp_path: Path) -> None:
    (tmp_path / "harness.toml").write_text("[limits]\nhorizon = 3\n")
    with pytest.raises(ValueError, match="inference.model"):
        load_harness(tmp_path)


def test_load_reads_optional_extras(tmp_path: Path) -> None:
    # A harness the optimizer has extended with a tool-surface extras file. Extras are
    # optimizer-authored, so they must use an isolated transport (see the tool-surface
    # tests for the guardrail that enforces this).
    export_harness("wordle_v1", "openai/gpt-5.4-nano", tmp_path)
    (tmp_path / "tools.toml").write_text(
        '[[mcp_servers]]\nname = "think"\ntransport = "stdio"\n'
        'command = ["python", "-m", "think_mcp"]\n'
    )
    h = load_harness(tmp_path)
    assert len(h.extra_specs) == 1
    assert h.extra_specs[0].name == "think"
    assert h.extra_specs[0].transport == "stdio"
