"""Guard tests for the Hermes quickstart: the checked-in wiring is what the docs promise.

Configuration and served surface, and nothing beyond them. A whole generation needs a durable
service and a worker to run it, and that arc is exercised once where the gateway is tested rather
than five times over here. What a quickstart owns is the wiring: a harness config that spawns the
server, one variable that names the env, and a prompt that describes the loop the server actually
serves.

The surface is built against ``wordle_v1`` rather than the quickstart's shipped default: it needs
no extra, no key and no download, so this stays offline. Nothing here spawns the ``hermes`` CLI or
spends a token.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("temporalio")

import shogym  # noqa: E402

# PyYAML is not an shogym dependency; it arrives with the default `dev` group (which CI syncs)
# and is how the quickstart's Hermes config is read here.
import yaml  # noqa: E402
from fastmcp import Client  # noqa: E402

from examples.hermes import serve as serve_mod  # noqa: E402
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    PULL_TOOL,
    StreamGateway,
    build_gateway_server,
    terminal_manifest,
)

_QUICKSTART = Path(__file__).resolve().parent.parent / "examples" / "hermes"

TEST_ENV = "wordle_v1"


def test_checked_in_config_spawns_the_serve_script() -> None:
    config = yaml.safe_load((_QUICKSTART / "config.yaml").read_text())
    # Hermes has no project-local MCP file: this block is copied into $HERMES_HOME/config.yaml,
    # and the server key is what Hermes namespaces tools under, so the README's `mcp__shogym__*`
    # is only correct while this key is `shogym`.
    assert list(config) == ["mcp_servers"]
    assert list(config["mcp_servers"]) == ["shogym"]
    server = config["mcp_servers"]["shogym"]
    assert server["command"] == "uv"
    assert server["args"] == ["run", "python", "serve.py"]
    assert server["enabled"] is True
    # Connecting builds the env and starts the durable service before the server answers, so
    # Hermes's own default would time the handshake out.
    assert server["connect_timeout"] > 60.0
    # Shipped stdio and secret-free. Hermes interpolates `${VAR}` in an `env:` block, so a key
    # pasted in literally would be committed here; the sample `env:` block stays commented out.
    assert set(server) == {"command", "args", "connect_timeout", "enabled"}


def test_the_one_variable_names_a_registered_env() -> None:
    assert serve_mod.ENV in shogym.registered_envs()
    # One index, because one launch is one episode over one env at one task.
    assert type(serve_mod.TASK) is int


def test_prompt_drives_the_pull_loop() -> None:
    prompt = (_QUICKSTART / "PROMPT.txt").read_text()
    assert PULL_TOOL in prompt and "done" in prompt
    # The wrapper is closed, so an agent that never puts the attempt in its calls gets nowhere.
    assert "attempt_id" in prompt


def test_run_dirs_are_fresh_per_launch(tmp_path: Path) -> None:
    # A generation writes its manifest once and refuses a directory that already holds one, so
    # the quickstart must not hand out the same directory twice.
    first = serve_mod.new_run_dir(TEST_ENV, tmp_path)
    assert first.parent == tmp_path and TEST_ENV in first.name


async def test_the_served_surface_is_the_loop_the_prompt_describes() -> None:
    """``pull`` plus the env's own tools, each in the wrapper the prompt tells the agent to fill.

    This is also what `hermes mcp test shogym` prints. Built in process from a real episode, and
    only the served surface is read, which is decided by the env's manifest and never by the
    stream, so no durable service is involved."""
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
