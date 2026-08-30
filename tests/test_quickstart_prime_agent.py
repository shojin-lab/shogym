"""Guard tests for the Prime Agent quickstart: the checked-in wiring is what the docs promise.

Configuration and served surface, and nothing beyond them. A whole generation needs a durable
service and a worker to run it, and that arc is exercised once where the gateway is tested rather
than five times over here. What a quickstart owns is the wiring: a settings entry and a skill that
agree on one endpoint, one variable that names the env, and a prompt that describes the loop the
server actually serves.

The skill package is read rather than imported. It is the only Python in this repo written for
somebody else's interpreter: it lives in Prime Agent's kernel venv and imports ``rlm``, which
is not an shogym dependency and is not on PyPI at all. ``ast`` gets the same facts without it.

The surface is built against ``wordle_v1`` rather than the quickstart's shipped default: it needs
no extra, no key and no download, so this stays offline. Nothing here spawns the ``prime-agent``
CLI or spends a token.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("temporalio")

import shogym  # noqa: E402
from fastmcp import Client  # noqa: E402

from examples.prime_agent import serve as serve_mod  # noqa: E402
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    PULL_TOOL,
    StreamGateway,
    build_gateway_server,
    terminal_manifest,
)

_QUICKSTART = Path(__file__).resolve().parent.parent / "examples" / "prime_agent"
_SETTINGS = _QUICKSTART / ".prime" / "agent" / "settings.json"
_SKILL = _QUICKSTART / ".prime" / "agent" / "skills" / "shogym-stream"

TEST_ENV = "wordle_v1"


def _skill_class_attrs() -> Dict[str, str]:
    """The `McpIntegration` subclass's string attributes, read out of the source."""
    module = ast.parse((_SKILL / "src" / "shogym_stream" / "__init__.py").read_text())
    classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
    assert len(classes) == 1, "the skill is one integration class"
    return {
        target.id: node.value.value
        for node in classes[0].body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def test_the_server_is_http_because_stdio_is_dropped() -> None:
    """Prime Agent's host skips every ``mcpServers`` entry whose type is not ``http``
    ("stdio servers self-manage in Python"), and its kernel-side client only speaks
    streamable HTTP. A stdio entry here would be silently ignored, not rejected."""
    settings = json.loads(_SETTINGS.read_text())
    assert list(settings) == ["mcpServers"], "the quickstart's whole config surface"
    assert list(settings["mcpServers"]) == ["shogym"]
    server = settings["mcpServers"]["shogym"]
    assert server["type"] == "http"
    assert server["url"].startswith("http://127.0.0.1:")


def test_a_static_bearer_token_is_declared_on_both_sides() -> None:
    """There is no unauthenticated path into ``McpIntegration``: it resolves a token before
    every connection and raises ``NotEnabled`` without one. ``serve.py`` authenticates nobody,
    so the token is a formality, but a formality both sides have to name identically."""
    settings = json.loads(_SETTINGS.read_text())
    attrs = _skill_class_attrs()
    assert settings["mcpServers"]["shogym"]["bearerTokenEnvVar"] == attrs["bearer_token_env"]
    assert "oauth" not in settings["mcpServers"]["shogym"], "no browser login in a quickstart"


def test_the_skill_settings_and_server_agree_on_one_url() -> None:
    """Three files name the endpoint and nothing reconciles them at runtime: the host's URL
    wins when the settings entry is present, the skill's own is the fallback when it is not."""
    settings = json.loads(_SETTINGS.read_text())
    attrs = _skill_class_attrs()
    assert attrs["url"] == settings["mcpServers"]["shogym"]["url"]
    assert attrs["url"] == f"http://127.0.0.1:{serve_mod.PORT}/mcp"
    assert attrs["server"] == "shogym"


def test_the_skill_is_discoverable_as_a_python_backed_skill() -> None:
    """Prime Agent's discovery rules, all three of them: a directory with a ``SKILL.md`` whose
    frontmatter ``name`` matches it, a ``pyproject.toml`` (which is what makes it Python-backed),
    and ``src/<import name>/__init__.py`` where the import name is the directory name with
    hyphens turned into underscores."""
    assert _SKILL.name == "shogym-stream"
    assert (_SKILL / "pyproject.toml").is_file()
    assert (_SKILL / "src" / "shogym_stream" / "__init__.py").is_file()

    front = (_SKILL / "SKILL.md").read_text().split("---")[1]
    fields = dict(
        line.split(":", 1) for line in front.strip().splitlines() if ":" in line and line[0] != " "
    )
    assert fields["name"].strip() == _SKILL.name
    assert fields["description"].strip(), "a skill with no description is not loaded at all"


def test_the_one_variable_names_a_registered_env() -> None:
    assert serve_mod.ENV in shogym.registered_envs()
    # One index, because one launch is one episode over one env at one task.
    assert type(serve_mod.TASK) is int


def test_prompt_drives_the_pull_loop_from_the_kernel() -> None:
    prompt = (_QUICKSTART / "PROMPT.txt").read_text()
    assert PULL_TOOL in prompt and "done" in prompt
    # The wrapper is closed, so an agent that never puts the attempt in its calls gets nowhere.
    assert "attempt_id" in prompt
    # The agent is never handed tools here: it imports the skill and calls it in Python, and
    # every answer arrives as a JSON string it has to parse.
    assert "shogym_stream" in prompt and "json.loads" in prompt


def test_run_dirs_are_fresh_per_launch(tmp_path: Path) -> None:
    # A generation writes its manifest once and refuses a directory that already holds one, so
    # the quickstart must not hand out the same directory twice.
    first = serve_mod.new_run_dir(TEST_ENV, tmp_path)
    assert first.parent == tmp_path and TEST_ENV in first.name


async def test_the_served_surface_is_the_loop_the_prompt_describes() -> None:
    """``pull`` plus the env's own tools, each in the wrapper the prompt tells the agent to fill.

    The kernel binds one async method per tool discovered here. Built in process from a real
    episode, and only the served surface is read, which is decided by the env's manifest and
    never by the stream, so no durable service is involved."""
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
