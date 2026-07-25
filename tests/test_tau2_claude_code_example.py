"""Guard tests for the tau2 Claude Code example (RFC 008 / issue #31): the config and command
builders stay consistent with the ``hgym serve`` CLI, the tool namespacing, and the
**corrected** ``claude`` invocation. The example itself needs the ``claude`` CLI + (for real
domains) network, so it is not run here. These are offline: ``run`` imports only ``hgym``
(which registers the tau2 envs without importing tau2), so no tau2 extra is required."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

import examples.tau2.claude_code.run as run_mod
from examples.tau2.claude_code.run import (
    ALLOWED_TOOLS,
    DEFAULT_DOMAIN,
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    DISALLOWED_TOOLS,
    build_claude_command,
    env_name,
    evaluate_claude,
    write_mcp_config,
)

_EXAMPLE_DIR = Path(__file__).resolve().parent.parent / "examples" / "tau2" / "claude_code"


def test_defaults_to_offline_mock_domain() -> None:
    assert DEFAULT_DOMAIN == "mock"
    assert env_name("mock") == "tau2_mock"
    assert env_name("telecom") == "tau2_telecom"


def test_checked_in_mcp_config_serves_tau2_mock_via_uv_run() -> None:
    config = json.loads((_EXAMPLE_DIR / ".mcp.json").read_text())
    server = config["mcpServers"]["tau2"]
    # Launch the server through `uv run python` so it resolves the project's `.venv`
    # interpreter (where hgym + the tau2 extra live) — a bare `python` on PATH need not have
    # either. Assert the command, not just the args.
    assert server["command"] == "uv"
    assert server["args"][:6] == ["run", "python", "-m", "hgym.cli", "serve", "tau2_mock"]
    assert "--trace" in server["args"]


def test_write_mcp_config_uses_current_interpreter_and_domain(tmp_path: Path) -> None:
    out = write_mcp_config(
        tmp_path / ".mcp.json", domain="telecom", task="5", trace_path=tmp_path / "t.jsonl"
    )
    server = json.loads(out.read_text())["mcpServers"]["tau2"]
    # Invoked through the current interpreter so it doesn't depend on `hgym` being on PATH.
    assert server["command"] == sys.executable
    assert server["args"][:4] == ["-m", "hgym.cli", "serve", "tau2_telecom"]
    assert "5" in server["args"] and str(tmp_path / "t.jsonl") in server["args"]


def test_allowed_tools_is_tau2_server_wildcard() -> None:
    # All tau2 tools allowed via the correct glob-free-prefix form (a bare `mcp__tau2` is
    # rejected by Claude Code's allow-rule grammar).
    assert ALLOWED_TOOLS == "mcp__tau2__*"


def test_build_claude_command_shape() -> None:
    cmd = build_claude_command(Path("cfg.json"))
    assert cmd[0] == "claude" and "-p" in cmd
    assert cmd[cmd.index("--mcp-config") + 1] == "cfg.json"
    assert DEFAULT_MODEL == "claude-sonnet-5"
    assert cmd[cmd.index("--model") + 1] == "claude-sonnet-5"
    assert DEFAULT_EFFORT == "low"
    assert cmd[cmd.index("--effort") + 1] == "low"
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__tau2__*"


def test_build_claude_command_isolates_mcp_config() -> None:
    # --strict-mcp-config keeps inherited user/project MCP servers out so the trace is
    # attributable solely to the configured `hgym serve` process.
    assert "--strict-mcp-config" in build_claude_command(Path("cfg.json"))


def test_build_claude_command_does_not_strip_mcp_tools() -> None:
    # The corrected invocation must NOT pass `--tools ""` (in current Claude Code that removes
    # the MCP tools too, leaving the agent with an empty toolset).
    cmd = build_claude_command(Path("cfg.json"))
    assert "--tools" not in cmd
    # MCP tools allowed; the run is non-interactive via `dontAsk` (deny anything not allowed).
    assert cmd[cmd.index("--allowedTools") + 1] == ALLOWED_TOOLS
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"


def test_build_claude_command_denies_builtins_for_trace_attribution() -> None:
    # Built-in tools (Bash/Read/… — untraced side channels) must be explicitly denied so the
    # episode's score is attributable to the MCP tool surface alone, while the MCP tools stay
    # reachable. `dontAsk` denies by default, and the explicit disallow closes its read-only
    # exemption (Read/Glob/Grep).
    cmd = build_claude_command(Path("cfg.json"))
    denied = cmd[cmd.index("--disallowedTools") + 1]
    assert denied == DISALLOWED_TOOLS
    for builtin in ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "WebFetch"):
        assert builtin in denied.split(",")
    # The MCP tools are not caught by the deny list.
    assert not any(t.startswith("mcp__") for t in denied.split(","))


def test_evaluate_claude_reports_unterminated_when_no_trace(tmp_path: Path, monkeypatch) -> None:
    # A zero-tool-call run (text-only / declined) exits 0 without writing a trace; the
    # evaluation must report terminated=False for the right env, not raise FileNotFoundError.
    # (Preflight is an external-env check; stub it here so this stays an offline unit test.)
    monkeypatch.setattr(run_mod, "preflight", lambda domain: None)
    monkeypatch.setattr(run_mod.subprocess, "run", lambda *a, **k: None)
    result = evaluate_claude("0", domain="mock", workdir=tmp_path)
    assert result.terminated is False
    assert result.env == "tau2_mock" and result.task == "0"
    assert not (tmp_path / "tau2_mock.jsonl").exists()


def test_ensure_tau2_data_respects_existing_env(monkeypatch) -> None:
    # If TAU2_DATA_DIR is already set, ensure_tau2_data must be a no-op — never shell out.
    monkeypatch.setenv("TAU2_DATA_DIR", "/some/where")

    def _fail(*a, **k):
        raise AssertionError("must not clone when TAU2_DATA_DIR is set")

    monkeypatch.setattr(run_mod.subprocess, "run", _fail)
    run_mod.ensure_tau2_data()
    assert os.environ["TAU2_DATA_DIR"] == "/some/where"


def test_ensure_tau2_data_uses_cache_without_cloning(tmp_path, monkeypatch) -> None:
    # When the cache already holds the data, ensure_tau2_data points TAU2_DATA_DIR at it
    # without cloning.
    monkeypatch.delenv("TAU2_DATA_DIR", raising=False)
    marker = tmp_path / "data" / "tau2" / "domains" / "mock" / "tasks.json"
    marker.parent.mkdir(parents=True)
    marker.write_text("[]")
    monkeypatch.setattr(run_mod, "_tau2_data_cache", lambda: tmp_path)

    def _fail(*a, **k):
        raise AssertionError("must not clone when the cache is already populated")

    monkeypatch.setattr(run_mod.subprocess, "run", _fail)
    run_mod.ensure_tau2_data()
    assert os.environ["TAU2_DATA_DIR"] == str(tmp_path / "data")


def test_preflight_reports_missing_tau2_with_fix(monkeypatch) -> None:
    # If the server interpreter can't import tau2, preflight must fail fast with an actionable
    # message (the interpreter path + the `uv sync` fix) rather than letting Claude Code
    # connect to a toolless, crashed server. (Stub data provisioning — external.)
    monkeypatch.setattr(run_mod, "ensure_tau2_data", lambda: None)

    def _boom(name: str):
        raise ImportError(f"No module named {name!r}")

    monkeypatch.setattr(run_mod.importlib, "import_module", _boom)
    with pytest.raises(SystemExit) as excinfo:
        run_mod.preflight("mock")
    msg = str(excinfo.value)
    assert "uv sync" in msg
    assert sys.executable in msg


def test_preflight_reports_unconstructible_env_with_data_hint(monkeypatch) -> None:
    # tau2 imports but the env won't build (typically missing data): the message must point
    # at TAU2_DATA_DIR, not surface as a served-server crash.
    monkeypatch.setattr(run_mod, "ensure_tau2_data", lambda: None)
    monkeypatch.setattr(run_mod, "_assert_server_can_import_tau2", lambda: None)

    def _boom(env: str):
        raise FileNotFoundError("domains/mock/tasks.json")

    monkeypatch.setattr(run_mod.hgym, "make", _boom)
    with pytest.raises(SystemExit) as excinfo:
        run_mod.preflight("mock")
    assert "TAU2_DATA_DIR" in str(excinfo.value)


def test_preflight_requires_key_for_non_solo_domain(monkeypatch) -> None:
    # A non-solo domain (its user simulator is an OpenAI LLM) must fail fast without a key,
    # not blow up mid-episode at the first user turn. mock (solo) must NOT require a key.
    monkeypatch.setattr(run_mod, "ensure_tau2_data", lambda: None)
    monkeypatch.setattr(run_mod, "_assert_server_can_import_tau2", lambda: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _Env:
        def __init__(self, solo: bool):
            self.solo_mode = solo

    monkeypatch.setattr(run_mod.hgym, "make", lambda env: _Env("telecom" not in env))
    with pytest.raises(SystemExit) as excinfo:
        run_mod.preflight("telecom")
    assert "OPENAI_API_KEY" in str(excinfo.value)
    # A solo domain does not require the key.
    run_mod.preflight("mock")


def test_model_and_effort_are_configurable() -> None:
    cmd = build_claude_command(Path("cfg.json"), "opus", "high")
    assert cmd[cmd.index("--model") + 1] == "opus"
    assert cmd[cmd.index("--effort") + 1] == "high"


def test_transcript_flag_adds_stream_json() -> None:
    assert "--output-format" not in build_claude_command(Path("cfg.json"))
    streamed = build_claude_command(Path("cfg.json"), transcript=True)
    assert streamed[streamed.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in streamed
