"""Harness-agnostic evaluation (RFC 008 §7): evaluate() drives a served env with an
in-process harness and reads the terminal feedback off the trace. Offline — the example
harness loop runs against a scripted policy (no network)."""

from __future__ import annotations

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


async def test_evaluate_without_trace_reports_termination(tmp_path: Path) -> None:
    async def harness(client) -> None:
        await run_episode(client, _scripted_chat(_answer(2)))

    result = await hgym.evaluate("wordle_v1", task=2, harness=harness)
    assert result.terminated is True
    assert result.feedback == []  # no trace file -> engine-only view, no feedback rows


def test_result_from_trace_reads_terminal_row(tmp_path: Path) -> None:
    # A trace an external harness (Claude Code) would have written by spawning hgym serve.
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
