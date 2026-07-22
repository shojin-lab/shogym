"""Harness-agnostic evaluation (RFC 008 §7): evaluate() drives a served env with an
in-process harness and reads the terminal feedback off the trace. Offline — the example
harness loop runs against a scripted policy (no network)."""

from __future__ import annotations

import pytest
from pathlib import Path
from typing import Any, Dict, List

import hgym
from examples.openai_harness import run_episode
from hgym.evaluate import result_from_trace


def _answer(task_idx: int) -> str:
    return hgym.make("wordle_v1")._words[task_idx]


def _scripted_chat(answer: str):
    """A deterministic policy: guess the answer, then terminate."""

    async def chat(
        instructions: str, tools: List[Dict[str, Any]], transcript: List[Dict[str, Any]]
    ):
        if not transcript:
            return [("guess", {"word": answer})]
        return [("terminate", {})]

    return chat


async def test_evaluate_solves_wordle_offline(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"

    async def harness(client) -> None:
        await run_episode(client, _scripted_chat(_answer(0)))

    result = await hgym.evaluate("wordle_v1", task=0, harness=harness, trace_path=trace)
    assert result.env == "wordle_v1"
    assert result.task == "0"
    assert result.terminated is True
    assert result.value("check_answer") is True
    assert result.value("partial_credit") == 1.0
    assert result.trace_path == str(trace)


async def test_evaluate_without_trace_reports_terminal_feedback() -> None:
    # The default public API (README quickstart) calls evaluate() without a trace_path;
    # it must still surface the terminal score, not an empty feedback list.
    async def harness(client) -> None:
        await run_episode(client, _scripted_chat(_answer(2)))

    result = await hgym.evaluate("wordle_v1", task=2, harness=harness)
    assert result.terminated is True
    assert result.trace_path is None
    assert result.value("check_answer") is True
    assert result.value("partial_credit") == 1.0


async def test_evaluate_closes_episode_when_build_server_fails(monkeypatch) -> None:
    # build_server() can raise after start() opened sessions + pushed state; evaluate()
    # must still close the episode (drop the pushed session state), not leak it.
    import importlib

    from hgym.envs.wordle import mcp_server

    # `hgym.evaluate` the attribute is the re-exported function, so reach the module.
    ev = importlib.import_module("hgym.evaluate")

    def boom(episode, **kwargs):
        raise ValueError("build failed")

    monkeypatch.setattr(ev, "build_server", boom)

    async def noop(client) -> None:
        return None

    before = set(mcp_server.sessions)
    with pytest.raises(ValueError, match="build failed"):
        await hgym.evaluate("wordle_v1", task=0, harness=noop)
    assert set(mcp_server.sessions) == before  # episode.close() ran despite the failure


async def test_evaluate_scopes_result_to_its_own_session(tmp_path: Path) -> None:
    trace = tmp_path / "shared.jsonl"

    async def solver(client) -> None:
        await run_episode(client, _scripted_chat(_answer(0)))

    r0 = await hgym.evaluate("wordle_v1", task=0, harness=solver, trace_path=trace)
    assert r0.value("check_answer") is True  # first run wrote a solved terminal row

    async def noop(client) -> None:
        return None

    # Same append-only trace, but this run makes no calls — it must NOT inherit the
    # first run's terminal result.
    r1 = await hgym.evaluate("wordle_v1", task=1, harness=noop, trace_path=trace)
    assert r1.task == "1"
    assert r1.terminated is False
    assert r1.value("check_answer") is None


def test_result_from_trace_scopes_external_read_by_identity(tmp_path: Path) -> None:
    # The external path has no session id; env/task must still scope the read so a reused
    # append-only trace can't let a prior task's terminal row supply this task's result.
    from hgym.trace import append_trace, step_record
    from hgym.types import EpisodeFeedback

    trace = tmp_path / "shared.jsonl"
    append_trace(trace, step_record(  # a completed task-0 episode
        session_id="s0", env_name="wordle_v1", task_id="0", step=1, tool="terminate",
        feedback=[EpisodeFeedback(name="check_answer", value=True)], terminated=True,
    ))
    # Reading for task 1 (which wrote nothing) must not inherit task 0's terminal row.
    r = result_from_trace(trace, env="wordle_v1", task="1")
    assert r.task == "1"
    assert r.terminated is False
    assert r.value("check_answer") is None


def test_result_from_trace_scopes_to_latest_same_task_episode(tmp_path: Path) -> None:
    # Two runs of the same env/task in one append-only trace: a read must reflect the
    # latest episode, not inherit an earlier run's terminal row (env/task aren't unique).
    from hgym.trace import append_trace, step_record
    from hgym.types import EpisodeFeedback

    trace = tmp_path / "shared.jsonl"
    append_trace(trace, step_record(  # run A: completed task 0
        session_id="a", env_name="wordle_v1", task_id="0", step=1, tool="terminate",
        feedback=[EpisodeFeedback(name="check_answer", value=True)], terminated=True,
    ))
    append_trace(trace, step_record(  # run B: task 0 again, guessed but never terminated
        session_id="b", env_name="wordle_v1", task_id="0", step=1, tool="guess",
    ))
    r = result_from_trace(trace, env="wordle_v1", task="0")
    assert r.terminated is False  # scoped to run B (latest), which didn't terminate
    assert r.value("check_answer") is None


def test_result_from_trace_reads_terminal_row(tmp_path: Path) -> None:
    from hgym.trace import append_trace, step_record
    from hgym.types import EpisodeFeedback

    trace = tmp_path / "ext.jsonl"
    append_trace(trace, step_record(session_id="s", env_name="wordle_v1", task_id="9",
                                    step=1, tool="guess"))
    append_trace(trace, step_record(session_id="s", env_name="wordle_v1", task_id="9",
                                    step=2, tool="terminate",
                                    feedback=[EpisodeFeedback(name="check_answer", value=False)],
                                    terminated=True))
    result = result_from_trace(trace)
    assert result.env == "wordle_v1" and result.task == "9"
    assert result.terminated is True
    assert result.value("check_answer") is False
