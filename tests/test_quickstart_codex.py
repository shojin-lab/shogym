"""Guard tests for the Codex quickstart: the checked-in wiring is what the docs promise.

Configuration and served surface, and nothing beyond them. A whole generation needs a durable
service and a worker to run it, and that arc is exercised once where the gateway is tested rather
than five times over here. What a quickstart owns is the wiring: a harness config that spawns the
server, one variable that names the env, and a prompt that describes the loop the server actually
serves.

The surface is built against ``wordle_v1``, the env this quickstart ships as its default: it needs
no extra, no key and no download, so this stays offline. Nothing here spawns the ``codex`` CLI or
spends a token.
"""

from __future__ import annotations

import importlib
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("temporalio")

import shogym  # noqa: E402
from fastmcp import Client  # noqa: E402

from examples.codex import serve as serve_mod  # noqa: E402
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    PULL_TOOL,
    StreamGateway,
    build_gateway_server,
    terminal_manifest,
)

_QUICKSTART = Path(__file__).resolve().parent.parent / "examples" / "codex"

TEST_ENV = "wordle_v1"


def test_project_config_spawns_the_serve_script() -> None:
    config = tomllib.loads((_QUICKSTART / ".codex" / "config.toml").read_text())
    # Codex namespaces a server's tools under its name, and it layers this file only when the
    # working directory is this one, which is what makes the relative `serve.py` resolve.
    assert list(config["mcp_servers"]) == ["shogym"]
    server = config["mcp_servers"]["shogym"]
    assert server["command"] == "uv"
    assert server["args"] == ["run", "python", "serve.py"]
    assert (_QUICKSTART / "serve.py").is_file()
    # The load-bearing line: `codex exec` has nobody to approve a tool call, so anything short
    # of "approve" turns every call into "user cancelled MCP tool call". "auto" is not enough.
    assert server["default_tools_approval_mode"] == "approve"
    # Codex's defaults (10s startup, 60s per tool call) cover neither a startup that builds the
    # env and starts the durable service nor a terminal call that seals and grades before it
    # answers.
    assert server["startup_timeout_sec"] > 10.0
    assert server["tool_timeout_sec"] > 60.0


def test_project_config_touches_nothing_global() -> None:
    """The quickstart's whole config surface is this one server. A project layer cannot carry
    provider, auth or telemetry keys anyway, and it should not try to carry anything else."""
    config = tomllib.loads((_QUICKSTART / ".codex" / "config.toml").read_text())
    assert list(config) == ["mcp_servers"]


def test_the_one_variable_names_a_registered_env() -> None:
    assert serve_mod.ENV in shogym.registered_envs()
    # One index, because one launch is one episode over one env at one task.
    assert type(serve_mod.TASK) is int


def test_the_env_it_ships_with_is_one_a_bare_install_can_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default in the file, read with the override unset.

    The quickstart is an install and a key and nothing else, so the env this ships with has to be
    one that needs no extra: an env whose upstream import is behind one fails before the server
    is reached. ``SHOGYM_ENV`` wins when it is set, so the shipped value is what is left when it
    is not, and this reloads the module without it rather than reading whatever the suite was
    started with."""
    monkeypatch.delenv("SHOGYM_ENV", raising=False)
    assert importlib.reload(serve_mod).ENV == "wordle_v1"


def test_prompt_drives_the_pull_loop() -> None:
    prompt = (_QUICKSTART / "PROMPT.txt").read_text()
    assert PULL_TOOL in prompt and "done" in prompt
    # The wrapper is closed, so an agent that never puts the attempt in its calls gets nowhere.
    assert "attempt_id" in prompt


def test_run_dirs_are_fresh_per_launch(tmp_path: Path) -> None:
    # A generation writes its manifest once and refuses a directory that already holds one, and
    # the run's durable history is a file in that directory, so the quickstart must not hand out
    # the same directory twice. Two allocations in one process are two inside one second, which
    # is the case a name stamped in whole seconds gets wrong.
    first = serve_mod.new_run_dir(TEST_ENV, tmp_path)
    second = serve_mod.new_run_dir(TEST_ENV, tmp_path)
    assert first.parent == tmp_path and TEST_ENV in first.name
    assert first != second


async def test_the_served_surface_is_the_loop_the_prompt_describes() -> None:
    """``pull`` plus the env's own tools, each in the wrapper the prompt tells the agent to fill.

    Built in process from a real episode. Only the served surface is read here, which is decided
    by the env's manifest and never by the stream, so no durable service is involved."""
    episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        spec = episode.describe()
        gateway = StreamGateway(
            None,  # type: ignore[arg-type]  # listing tools reaches no stream
            episode,
            spec,
            terminal_manifest(spec),
            initial_cursor="0" * 32,
        )
        async with Client(build_gateway_server(gateway, name="shogym")) as client:
            schemas = {tool.name: tool.inputSchema for tool in await client.list_tools()}
    finally:
        await episode.close()

    assert set(schemas) == {PULL_TOOL, "guess", "terminate"}
    assert schemas[PULL_TOOL]["properties"] == {}
    for tool in ("guess", "terminate"):
        assert set(schemas[tool]["properties"]) == {"attempt_id", "arguments"}
        assert schemas[tool]["required"] == ["attempt_id", "arguments"]
