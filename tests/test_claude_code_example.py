"""Guard tests for the Claude Code example (RFC 008): the config and command builders stay
consistent with the `hgym serve` CLI and the tool namespacing. The example itself needs the
`claude` CLI + network, so it is not run here."""

from __future__ import annotations

import json
from pathlib import Path

import examples.wordle.claude_code.run as run_mod
from examples.wordle.claude_code.run import (
    ALLOWED_TOOLS,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    build_claude_command,
    evaluate_claude,
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


def test_write_mcp_config_uses_current_interpreter(tmp_path: Path) -> None:
    out = write_mcp_config(tmp_path / ".mcp.json", task="5", trace_path=tmp_path / "t.jsonl")
    server = json.loads(out.read_text())["mcpServers"]["wordle"]
    # Invoked through the current interpreter so it doesn't depend on `hgym` being on PATH.
    assert server["args"][:3] == ["-m", "hgym.cli", "serve"]
    assert "5" in server["args"] and str(tmp_path / "t.jsonl") in server["args"]


def test_build_claude_command_shape() -> None:
    cmd = build_claude_command(Path("cfg.json"))
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "cfg.json"
    assert DEFAULT_MODEL == "claude-sonnet-5"
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert DEFAULT_EFFORT == "low"
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert ALLOWED_TOOLS.split(",") == [
        "mcp__wordle__describe",
        "mcp__wordle__guess",
        "mcp__wordle__terminate",
    ]


def test_build_claude_command_isolates_mcp_config() -> None:
    # --strict-mcp-config keeps inherited user/project MCP servers out of the demo so its
    # trace is attributable solely to the configured `hgym serve` process.
    assert "--strict-mcp-config" in build_claude_command(Path("cfg.json"))


def test_build_claude_command_disables_builtin_tools() -> None:
    # --allowedTools only auto-approves; built-in Read/Bash stay available and could take
    # untraced side-channel actions (e.g. read the word corpus). `--tools ""` removes the
    # built-in set so only the three MCP tools remain reachable and the run stays fully
    # trace-attributable.
    cmd = build_claude_command(Path("cfg.json"))
    assert cmd[cmd.index("--tools") + 1] == ""
    # MCP tools are still allowed.
    assert cmd[cmd.index("--allowedTools") + 1] == ALLOWED_TOOLS


def test_evaluate_claude_reports_unterminated_when_no_trace(
    tmp_path: Path, monkeypatch,
) -> None:
    # A zero-tool-call run (text-only / declined) exits 0 without writing a trace; the
    # evaluation must report terminated=False, not raise FileNotFoundError.
    monkeypatch.setattr(run_mod.subprocess, "run", lambda *a, **k: None)
    result = evaluate_claude("0", workdir=tmp_path)
    assert result.terminated is False
    assert result.env == "wordle_v1" and result.task == "0"
    assert not (tmp_path / "wordle.jsonl").exists()


def test_model_and_effort_are_configurable() -> None:
    cmd = build_claude_command(Path("cfg.json"), "opus", "high")
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "high"


def test_transcript_flag_adds_stream_json() -> None:
    assert "--output-format" not in build_claude_command(Path("cfg.json"))
    streamed = build_claude_command(Path("cfg.json"), transcript=True)
    assert streamed[streamed.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in streamed
