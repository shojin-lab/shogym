"""Guard tests for the AutomationBench Claude Code example (RFC 008): the config + command
builders stay consistent with the ``hgym serve`` CLI and the tool namespacing. The example itself
needs the ``claude`` CLI + network, so it is not run here — these builders are pure and offline
(they never provision the upstream source)."""

from __future__ import annotations

import json
from pathlib import Path

from examples.automationbench.claude_code.run import (
    ALLOWED_TOOLS,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    ENV_NAME,
    build_claude_command,
    write_mcp_config,
)

_EXAMPLE_DIR = (
    Path(__file__).resolve().parent.parent / "examples" / "automationbench" / "claude_code"
)


def test_checked_in_mcp_config_spawns_hgym_serve() -> None:
    config = json.loads((_EXAMPLE_DIR / ".mcp.json").read_text())
    server = config["mcpServers"]["automationbench"]
    assert server["args"][:5] == ["run", "python", "-m", "hgym.cli", "serve"]
    assert "automationbench" in server["args"] and "--task" in server["args"]
    assert "--trace" in server["args"]


def test_write_mcp_config_uses_current_interpreter(tmp_path: Path) -> None:
    out = write_mcp_config(tmp_path / ".mcp.json", task="3", trace_path=tmp_path / "t.jsonl")
    server = json.loads(out.read_text())["mcpServers"]["automationbench"]
    # Invoked through the current interpreter so it doesn't depend on `hgym` being on PATH.
    assert server["args"][:3] == ["-m", "hgym.cli", "serve"]
    assert server["args"][3] == ENV_NAME
    assert "3" in server["args"] and str(tmp_path / "t.jsonl") in server["args"]


def test_build_claude_command_shape() -> None:
    cmd = build_claude_command(Path("cfg.json"))
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "cfg.json"
    assert DEFAULT_MODEL == "claude-sonnet-5"
    assert cmd[cmd.index("--model") + 1] == DEFAULT_MODEL
    assert DEFAULT_EFFORT == "low"
    assert cmd[cmd.index("--effort") + 1] == DEFAULT_EFFORT
    # "All tools from this server" — the glob-free ``mcp__<server>__*`` form Claude Code requires.
    assert ALLOWED_TOOLS == "mcp__automationbench__*"


def test_build_claude_command_isolates_and_locks_down_builtins() -> None:
    cmd = build_claude_command(Path("cfg.json"))
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    disallowed = cmd[cmd.index("--disallowedTools") + 1]
    assert "Bash" in disallowed and "WebFetch" in disallowed
