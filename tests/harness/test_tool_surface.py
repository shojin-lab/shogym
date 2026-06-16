"""The tool surface (RFC 002): extras round-trip, and the isolation guardrail that
keeps optimizer-authored extras out of the host process."""

from __future__ import annotations

from pathlib import Path

import pytest

from hgym.harness import ExtraIsolationError, export_harness, load_harness
from hgym.mcp.types import MCPServerSpec


def _stdio(name: str) -> MCPServerSpec:
    return MCPServerSpec(
        name=name, transport="stdio", command=["python", "-m", f"{name}_mcp"]
    )


def test_export_writes_tools_toml_for_extras(tmp_path: Path) -> None:
    out = export_harness(
        "wordle_v1",
        "openai/gpt-5.4-nano",
        tmp_path / "h",
        extra_specs=[_stdio("think"), _stdio("search")],
    )
    assert (out / "tools.toml").exists()


def test_extras_roundtrip(tmp_path: Path) -> None:
    specs = [
        MCPServerSpec(
            name="think",
            transport="stdio",
            command=["python", "-m", "think_mcp"],
            env={"LOG": "1"},
        ),
        MCPServerSpec(
            name="docs", transport="streamable_http", url="https://example.com/mcp"
        ),
    ]
    out = export_harness(
        "wordle_v1", "openai/gpt-5.4-nano", tmp_path / "h", extra_specs=specs
    )
    h = load_harness(out)
    assert h.extra_specs == specs


def test_export_rejects_in_process_extra(tmp_path: Path) -> None:
    bad = MCPServerSpec(
        name="think", transport="in_process", module="think_mcp"
    )
    with pytest.raises(ExtraIsolationError, match="think"):
        export_harness(
            "wordle_v1", "openai/gpt-5.4-nano", tmp_path / "h", extra_specs=[bad]
        )
    # Nothing is written when validation fails up front.
    assert not (tmp_path / "h" / "tools.toml").exists()


def test_load_rejects_in_process_extra(tmp_path: Path) -> None:
    # An extras file hand-written (or tampered) with a non-isolated transport.
    export_harness("wordle_v1", "openai/gpt-5.4-nano", tmp_path)
    (tmp_path / "tools.toml").write_text(
        '[[mcp_servers]]\nname = "think"\ntransport = "in_process"\n'
        'module = "think_mcp"\n'
    )
    with pytest.raises(ExtraIsolationError, match="think"):
        load_harness(tmp_path)


def test_baseline_export_still_writes_no_tools_toml(tmp_path: Path) -> None:
    out = export_harness("wordle_v1", "openai/gpt-5.4-nano", tmp_path / "h")
    assert not (out / "tools.toml").exists()
