"""Guard tests for the Pi quickstart: the served stream really does hand out tasks, record one
row each, and read them back.

Driven against ``wordle_v1`` rather than the quickstart's shipped default: it needs no extra, no
key and no download, so this stays offline. Nothing here spawns the ``pi`` CLI, installs the MCP
bridge or spends a token; the harness half is checked as configuration.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import shogym
from fastmcp import Client

from examples.quickstarts.pi import results as results_mod
from examples.quickstarts.pi import serve as serve_mod
from shogym.serve.stream import build_stream_server

_QUICKSTART = Path(__file__).resolve().parent.parent / "examples" / "quickstarts" / "pi"

TEST_ENV = "wordle_v1"


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


def test_checked_in_mcp_config_spawns_the_serve_script() -> None:
    config = json.loads((_QUICKSTART / ".pi" / "mcp.json").read_text())
    # The server key is what the bridge namespaces tools under, so `mcp_shogym_*` in the README
    # and PROMPT is only correct while this key is `shogym`.
    assert list(config["mcpServers"]) == ["shogym"]
    server = config["mcpServers"]["shogym"]
    assert server["transport"] == "stdio"
    assert server["command"] == "uv"
    assert server["args"] == ["run", "python", "serve.py"]
    # The bridge's own default is "lazy", which never starts the server in a `pi -p` run.
    assert server["lifecycle"] == "eager"
    assert (_QUICKSTART / "serve.py").is_file()


def test_the_bridge_is_pinned_to_an_exact_version() -> None:
    # Pi has no built-in MCP client; this quickstart adds a third-party one. A range would let a
    # new release of someone else's code into the agent without an edit here.
    settings = json.loads((_QUICKSTART / ".pi" / "settings.json").read_text())
    assert len(settings["packages"]) == 1
    assert re.fullmatch(r"npm:pi-mcp-extension@\d+\.\d+\.\d+", settings["packages"][0])


def test_the_one_variable_names_a_registered_env() -> None:
    assert serve_mod.ENV in shogym.registered_envs()
    assert serve_mod.TASKS and all(type(i) is int for i in serve_mod.TASKS)


def test_prompt_drives_the_stream_loop_under_the_bridge_prefix() -> None:
    prompt = (_QUICKSTART / "PROMPT.txt").read_text()
    server_key = next(iter(json.loads((_QUICKSTART / ".pi" / "mcp.json").read_text())["mcpServers"]))
    # The bridge renames every served tool `mcp_<server>_<tool>`, so the prompt has to ask for
    # the prefixed name, not the wire name.
    assert f"mcp_{server_key}_get_task" in prompt and "done" in prompt


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
            # The stream's control tools plus the env's own surface, on one endpoint. These are
            # the wire names; the bridge is what prefixes them on the way to the model.
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
            # nothing else does -- no identity, no queue state, no stream fields.
            assert ended["terminated"] is True
            assert "feedback" in ended
            assert {f["name"] for f in ended["feedback"]} == {"partial_credit", "check_answer", "count_turns"}
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
