"""Guard tests for the Prime Agent quickstart: the served stream really does hand out tasks,
record one row each, and read them back.

Driven against ``wordle_v1`` rather than the quickstart's shipped default: it needs no extra, no
key and no download, so this stays offline. Nothing here spawns the ``prime-agent`` CLI or
spends a token; the harness half is checked as configuration.

The skill package is read rather than imported. It is the only Python in this repo written for
somebody else's interpreter: it lives in Prime Agent's kernel venv and imports ``rlm``, which
is not an shogym dependency and is not on PyPI at all. ``ast`` gets the same facts without it.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict

import shogym
from fastmcp import Client

from examples.prime_agent import results as results_mod
from examples.prime_agent import serve as serve_mod
from shogym.serve.stream import build_stream_server

_QUICKSTART = Path(__file__).resolve().parent.parent / "examples" / "prime_agent"
_SETTINGS = _QUICKSTART / ".prime" / "agent" / "settings.json"
_SKILL = _QUICKSTART / ".prime" / "agent" / "skills" / "shogym-stream"

TEST_ENV = "wordle_v1"


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


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
    so the token is a formality -- but a formality both sides have to name identically."""
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
    assert serve_mod.TASKS and all(type(i) is int for i in serve_mod.TASKS)


def test_prompt_drives_the_stream_loop_from_the_kernel() -> None:
    prompt = (_QUICKSTART / "PROMPT.txt").read_text()
    assert "get_task" in prompt and "done" in prompt
    # The agent is never handed tools here: it imports the skill and calls it in Python, and
    # every answer arrives as a JSON string it has to parse.
    assert "shogym_stream" in prompt and "json.loads" in prompt


def test_run_dirs_are_fresh_per_run(tmp_path: Path) -> None:
    # A stream refuses a directory another run recorded into, so the quickstart must not hand
    # out one twice.
    first = serve_mod.new_run_dir(TEST_ENV, tmp_path)
    assert first.parent == tmp_path and TEST_ENV in first.name


async def test_stream_serves_tasks_and_records_one_row_each(tmp_path: Path) -> None:
    prov = tmp_path / "prov"
    stream = serve_mod.build_stream(env=TEST_ENV, tasks=[0, 1], prov_dir=prov)
    async with stream:
        client_server = build_stream_server(stream, name="shogym")
        async with Client(client_server) as client:
            names = {tool.name for tool in await client.list_tools()}
            # The stream's control tools plus the env's own surface, on one endpoint.
            assert {"get_task", "queue_info", "guess", "terminate"} <= names

            task = _payload(await client.call_tool("get_task", {}))
            # Redaction is structural: there is no field the index or the target could ride on.
            assert set(task) == {"env", "instructions", "budget", "tools"}
            assert task["env"] == TEST_ENV
            assert {t["name"] for t in task["tools"]} == {"guess", "terminate"}

            played = _payload(await client.call_tool("guess", {"word": "crane"}))
            assert played["terminated"] is False

            ended = _payload(await client.call_tool("terminate", {}))
            # The practice default: the env's own published verdict comes back, and
            # nothing else does. No identity, no queue state, no stream fields.
            assert ended["terminated"] is True
            assert "feedback" in ended
            assert {f["name"] for f in ended["feedback"]} == {
                "partial_credit",
                "check_answer",
                "count_turns",
            }
            assert set(ended) <= {"content", "terminated", "hint", "feedback"}

            second = _payload(await client.call_tool("get_task", {}))
            assert second["env"] == TEST_ENV
            _payload(await client.call_tool("terminate", {}))

            assert _payload(await client.call_tool("get_task", {}))["done"] is True

    # Read back with the quickstart's own reader, off disk, after the stream is closed.
    recorded = results_mod.rows(prov)
    assert [row.position for row in recorded] == [0, 1]
    assert [row.task_idx for row in recorded] == [0, 1]
    assert all(row.closure == "aborted" for row in recorded)
    # Scoring happened server-side: the rows carry the env's numbers and the regime stamp.
    assert all(row.score is not None and row.score.reward is not None for row in recorded)
    assert all(row.feedback_regime == "immediate" for row in recorded)
    assert all(any(item["name"] == "check_answer" for item in row.observed) for row in recorded)


async def test_unsealed_task_is_recorded_when_the_run_ends(tmp_path: Path) -> None:
    """A disconnect mid-task still owes a row: the drain ends it and records it."""
    prov = tmp_path / "prov"
    stream = serve_mod.build_stream(env=TEST_ENV, tasks=[0], prov_dir=prov)
    async with stream:
        async with Client(build_stream_server(stream)) as client:
            await client.call_tool("get_task", {})

    recorded = results_mod.rows(prov)
    assert [row.closure for row in recorded] == ["drained"]
    assert recorded[0].score is not None
