"""The episode-serving engine (RFC 008): drive Wordle one tool call at a time, in
process (no subprocess), and check the wire contract — feedback sidecar, the visibility
rule, horizon-as-terminal, and the trace.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import hgym
from hgym.feedback import parse_meta
from hgym.serve import ServedEpisode
from hgym.trace import load_traces


def _answer(task_idx: int) -> str:
    # The default env loads a deterministic word list; instances agree on it.
    return hgym.make("wordle_v1")._words[task_idx]


async def test_describe_available_after_start() -> None:
    ep = await ServedEpisode.start("wordle_v1", task=3)
    try:
        spec = ep.describe()
        assert spec.env_name == "wordle_v1"
        assert spec.task_id == "3"
        assert {t.name for t in spec.tools} >= {"guess", "terminate"}
    finally:
        await ep.close()


async def test_solve_then_terminate_surfaces_episode_feedback_only_at_end(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start("wordle_v1", task=0, trace_path=trace_path)
    try:
        # A correct guess: functional result says solved; mid-episode feedback is the
        # inference-level format_reward, and NO episode-level reward is surfaced yet.
        res = await ep.call("guess", {"word": _answer(0)})
        assert not res.terminated
        payload = json.loads(res.content)
        assert payload["solved"] is True and payload["score"] == "GGGGG"

        items, terminate = parse_meta(res.meta)
        assert terminate is False
        names = {i.name for i in items}
        assert "format_reward" in names  # inference feedback surfaces mid-episode
        assert "check_answer" not in names  # episode feedback stays hidden until the end

        # Terminate: now the episode-level verdict surfaces, with the stop flag.
        end = await ep.call("terminate")
        assert end.terminated is True
        items, terminate = parse_meta(end.meta)
        assert terminate is True
        by_name = {i.name: i.value for i in items}
        assert by_name["check_answer"] is True
    finally:
        await ep.close()

    rows = load_traces(trace_path)
    assert [r["tool"] for r in rows] == ["guess", "terminate"]
    assert rows[-1]["terminated"] is True
    assert any(f["name"] == "check_answer" for f in rows[-1]["feedback"])


async def test_horizon_terminates_env_side_without_a_terminate_call(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start("wordle_v1", task=1, trace_path=trace_path)
    try:
        # "aaaaa" is a valid-format guess but never the target, so six of them exhaust
        # the budget. The env stops the episode on the sixth call — no terminate needed.
        terminated_at = None
        for i in range(1, 7):
            res = await ep.call("guess", {"word": "aaaaa"})
            if res.terminated:
                terminated_at = i
                break
        assert terminated_at == 6

        # Horizon is a terminal boundary: episode feedback surfaces on that final result.
        items, terminate = parse_meta(res.meta)
        assert terminate is True
        assert {i.name for i in items} >= {"check_answer", "count_turns"}

        # Further calls after termination are a graceful no-op.
        after = await ep.call("guess", {"word": "aaaaa"})
        assert after.terminated is True
    finally:
        await ep.close()

    rows = load_traces(trace_path)
    assert len(rows) == 6  # the post-termination call is not stepped/recorded
    assert rows[-1]["terminated"] is True


async def test_start_rejects_non_toolusing_env() -> None:
    with pytest.raises(ValueError):
        await ServedEpisode.start("does_not_exist")
