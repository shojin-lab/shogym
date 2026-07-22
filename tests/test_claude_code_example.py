"""Guard tests for the Claude Code example (RFC 008): the config and command builders
stay consistent with the `hgym serve` CLI and the tool namespacing. The example itself
needs the `claude` CLI + network, so it is not run here."""

from __future__ import annotations

import json
from pathlib import Path

from examples.wordle.claude_code.run import (
    ALLOWED_TOOLS,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    build_claude_command,
    write_mcp_config,
)

_EXAMPLE_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "wordle" / "claude_code"
)


def test_checked_in_mcp_config_spawns_hgym_serve() -> None:
    config = json.loads((_EXAMPLE_DIR / ".mcp.json").read_text())
    server = config["mcpServers"]["wordle"]
    assert server["command"] == "hgym"
    assert server["args"][:3] == ["serve", "wordle_v1", "--task"]
    assert "--trace" in server["args"]


def test_write_mcp_config_bakes_task_and_trace(tmp_path: Path) -> None:
    out = write_mcp_config(tmp_path / ".mcp.json", task="5", trace_path=tmp_path / "t.jsonl")
    server = json.loads(out.read_text())["mcpServers"]["wordle"]
    # Invoked through the current interpreter so it doesn't depend on `hgym` being on PATH.
    assert server["args"][:3] == ["-m", "hgym.cli", "serve"]
    assert "5" in server["args"] and str(tmp_path / "t.jsonl") in server["args"]


def test_build_claude_command_shape() -> None:
    cmd = build_claude_command(Path("cfg.json"))
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "cfg.json"
    # Defaults to the latest Sonnet, configurable via the --model arg.
    assert DEFAULT_MODEL == "claude-sonnet-5"
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    # Reasoning effort defaults to the lowest level.
    assert DEFAULT_EFFORT == "low"
    assert cmd[cmd.index("--effort") + 1] == "low"
    # The three MCP tools are pre-allowed, namespaced the way Claude Code expects.
    assert ALLOWED_TOOLS.split(",") == [
        "mcp__wordle__describe",
        "mcp__wordle__guess",
        "mcp__wordle__terminate",
    ]


def test_model_and_effort_are_configurable() -> None:
    cmd = build_claude_command(Path("cfg.json"), "opus", "high")
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "high"


def test_transcript_flag_adds_stream_json() -> None:
    plain = build_claude_command(Path("cfg.json"))
    assert "--output-format" not in plain

    streamed = build_claude_command(Path("cfg.json"), transcript=True)
    assert streamed[streamed.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in streamed
