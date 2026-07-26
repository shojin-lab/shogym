"""Guard tests for the frontier_bench Claude Code example (RFC 008): the config + command
builders stay consistent with the `hgym serve` CLI and the tool namespacing. The example itself
needs Docker + the `claude` CLI + network, so it is not run here (these builders don't touch
Docker)."""

from __future__ import annotations

import json
from pathlib import Path

from examples.frontier_bench.claude_code.run import (
    ALLOWED_TOOLS,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    ENV_NAME,
    SERVER_KEY,
    build_claude_command,
    write_mcp_config,
)


def test_write_mcp_config_uses_current_interpreter(tmp_path: Path) -> None:
    out = write_mcp_config(tmp_path / ".mcp.json", task="0", trace_path=tmp_path / "t.jsonl")
    server = json.loads(out.read_text())["mcpServers"][SERVER_KEY]
    # Invoked through the current interpreter so it doesn't depend on `hgym` being on PATH.
    assert server["args"][:4] == ["-m", "hgym.cli", "serve", ENV_NAME]
    assert "0" in server["args"] and str(tmp_path / "t.jsonl") in server["args"]


def test_build_claude_command_shape() -> None:
    cmd = build_claude_command(Path("cfg.json"))
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "cfg.json"
    assert DEFAULT_MODEL == "claude-sonnet-5"
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert DEFAULT_EFFORT == "medium"
    assert cmd[cmd.index("--effort") + 1] == "medium"
    # "all tools from this server" is the glob-free __* form.
    assert ALLOWED_TOOLS == f"mcp__{SERVER_KEY}__*"


def test_build_claude_command_isolates_and_locks_down() -> None:
    cmd = build_claude_command(Path("cfg.json"))
    # --strict-mcp-config keeps inherited MCP servers out; built-ins are denied so the agent's
    # only shell is the served `exec` (inside the container), keeping the trace attributable.
    assert "--strict-mcp-config" in cmd
    denied = cmd[cmd.index("--disallowedTools") + 1]
    assert "Bash" in denied and "WebFetch" in denied
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
