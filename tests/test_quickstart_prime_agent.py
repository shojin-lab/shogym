"""Guard tests for the Prime Agent quickstart: the checked-in wiring is what the docs promise.

Configuration and served surface, and nothing beyond them. A whole generation needs a durable
service and a worker to run it, and that arc is exercised once where the gateway is tested rather
than five times over here. What a quickstart owns is the wiring: a settings entry and a skill that
agree on one endpoint, one variable that names the env, and a prompt that describes the loop the
server actually serves.

The skill package is read rather than imported. It is the only Python in this repo written for
somebody else's interpreter: it lives in Prime Agent's kernel venv and imports ``rlm``, which
is not an shogym dependency and is not on PyPI at all. ``ast`` gets the same facts without it.

The surface is built against ``wordle_v1``, the env this quickstart ships as its default: it needs
no extra, no key and no download, so this stays offline. Nothing here spawns the ``prime-agent``
CLI or spends a token.
"""

from __future__ import annotations

import ast
import importlib
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

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


def test_prompt_drives_the_pull_loop_from_the_kernel() -> None:
    prompt = (_QUICKSTART / "PROMPT.txt").read_text()
    assert PULL_TOOL in prompt and "done" in prompt
    # The wrapper is closed, so an agent that never puts the attempt in its calls gets nowhere.
    assert "attempt_id" in prompt
    # The agent is never handed tools here: it imports the skill and calls it in Python, and
    # every answer arrives as a JSON string it has to parse.
    assert "shogym_stream" in prompt and "json.loads" in prompt


def test_run_dirs_are_fresh_per_launch(tmp_path: Path) -> None:
    # A generation writes its manifest once and refuses a directory that already holds one, and
    # the run's durable history is a file in that directory, so the quickstart must not hand out
    # the same directory twice. Two allocations in one process are two inside one second, which
    # is the case a name stamped in whole seconds gets wrong.
    first = serve_mod.new_run_dir(TEST_ENV, tmp_path)
    second = serve_mod.new_run_dir(TEST_ENV, tmp_path)
    assert first.parent == tmp_path and TEST_ENV in first.name
    assert first != second


class _Episode:
    """A world that records what it was started with and what became of it.

    The launch touches nothing else on an episode: it hands it to the gateway and, on the one
    path where there is no gateway to let it go, closes it.
    """

    def __init__(self, **started: Any) -> None:
        self.started = started
        self.closed_with: List[bool] = []

    async def close(self, *, finalize: bool = True) -> None:
        self.closed_with.append(finalize)


class _Gateway:
    """The generation, as far as the launch is concerned: it is closed, and it says so."""

    def __init__(self) -> None:
        self.queue_closed = False
        self.stopped = False

    async def close_queue(self) -> None:
        self.queue_closed = True

    async def aclose(self) -> None:
        self.stopped = True


def _launch_doubles(
    monkeypatch: pytest.MonkeyPatch, run_dir: Path, *, opening: Optional[Exception] = None
) -> Dict[str, Any]:
    """Replace everything the launch reaches, and keep what it was asked for.

    Nothing durable starts, nothing binds a port, and no env is built. What is left is the
    wiring this file owns: what the episode is given, what the gateway is given, and what is
    done with each of them on the way out.
    """
    made: Dict[str, Any] = {"gateway": _Gateway()}

    async def start(env: str, **kwargs: Any) -> _Episode:
        made["episode"] = _Episode(env=env, **kwargs)
        return made["episode"]

    @asynccontextmanager
    async def client(*, run_directory: Any = None) -> AsyncIterator[Any]:
        made["client_run_directory"] = run_directory
        yield SimpleNamespace()

    @asynccontextmanager
    async def worker(client: Any, *, activities: Any = ()) -> AsyncIterator[Any]:
        yield SimpleNamespace()

    async def opened(client: Any, episode: Any, **kwargs: Any) -> _Gateway:
        made["gateway_run_directory"] = kwargs.get("run_directory")
        if opening is not None:
            raise opening
        return made["gateway"]

    def served(gateway: Any) -> Any:
        async def run_async(**kwargs: Any) -> None:
            made["served"] = kwargs

        return SimpleNamespace(run_async=run_async)

    monkeypatch.setattr(serve_mod, "new_run_dir", lambda: run_dir)
    monkeypatch.setattr(serve_mod, "ServedEpisode", SimpleNamespace(start=start))
    monkeypatch.setattr(
        serve_mod, "environment_terminal", lambda episode: SimpleNamespace(activities=[])
    )
    monkeypatch.setattr(serve_mod, "durable_client", client)
    monkeypatch.setattr(serve_mod, "stream_worker", worker)
    monkeypatch.setattr(serve_mod, "open_gateway", opened)
    monkeypatch.setattr(serve_mod, "build_gateway_server", served)
    return made


async def test_the_launch_gives_the_run_to_the_episode_as_well_as_to_the_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This file is ``run_stdio_v2`` with the transport swapped, so it holds the same lifecycle.

    The run directory is the episode's as well as the generation's. An episode picks the store
    its durable records go in when it is constructed, so a directory handed only to the gateway
    arrives after that choice was made and leaves the records in a store shared with every
    session this machine has served, where a reader of this run cannot find them.

    And the world is let go of rather than ended. Stopping the gateway is what releases it, and
    an ordinary close would read the untouched lifecycle as an episode that stopped without a
    seal and claim an abort verdict for it, beside whatever the generation committed.
    """
    run_dir = tmp_path / "run"
    made = _launch_doubles(monkeypatch, run_dir)

    await serve_mod.main(["8971"])

    assert made["episode"].started["run_directory"] == run_dir
    assert made["client_run_directory"] == run_dir
    assert made["gateway_run_directory"] == run_dir
    # The queue is closed before anything is served, the server ran, and the gateway is what
    # stopped: nothing ended this episode.
    assert made["gateway"].queue_closed is True
    assert made["served"]["port"] == 8971
    assert made["gateway"].stopped is True
    assert made["episode"].closed_with == []


async def test_a_launch_that_never_opened_a_generation_releases_its_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one path that closes the episode here closes it the way the stream would.

    A world nothing is serving still has to be released, and releasing it is not ending the
    attempt it was opened for: an abort verdict claimed on the way out of a failed launch is a
    result for an attempt nobody worked.
    """
    made = _launch_doubles(monkeypatch, tmp_path / "run", opening=RuntimeError("no service"))

    with pytest.raises(RuntimeError):
        await serve_mod.main([])

    assert made["episode"].closed_with == [False]
    assert made["gateway"].stopped is False


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
