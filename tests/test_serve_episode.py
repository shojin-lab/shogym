"""The episode-serving engine (RFC 008): drive Wordle one tool call at a time, in process
(no subprocess). Checks the wire contract — feedback sidecar, visibility rule,
horizon-as-terminal, and the trace."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import hgym
from hgym.feedback import parse_meta
from hgym.serve import ServedEpisode
from hgym.trace import load_traces


def _answer(task_idx: int) -> str:
    return hgym.make("wordle_v1")._words[task_idx]


async def test_describe_available_after_start() -> None:
    ep = await ServedEpisode.start("wordle_v1", task=3)
    try:
        spec = ep.describe()
        assert spec.env_name == "wordle_v1" and spec.task_id == "3"
        assert {t.name for t in spec.tools} >= {"guess", "terminate"}
    finally:
        await ep.close()


async def test_solve_then_terminate_surfaces_episode_feedback_only_at_end(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start("wordle_v1", task=0, trace_path=trace_path)
    try:
        res = await ep.call("guess", {"word": _answer(0)})
        assert not res.terminated
        payload = json.loads(res.content)
        assert payload["solved"] is True and payload["score"] == "GGGGG"
        items, terminate = parse_meta(res.meta)
        assert terminate is False
        # Eval-safe default: dense inference feedback (format_reward) is recorded-only,
        # not surfaced in-band; episode feedback is hidden until the terminal result.
        assert items == []

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
    # format_reward is recorded on the guess row even though it never surfaced in-band.
    assert any(f["name"] == "format_reward" for f in rows[0]["feedback"])
    assert any(f["name"] == "check_answer" for f in rows[-1]["feedback"])


async def test_cancelled_call_does_not_advance_step(tmp_path: Path) -> None:
    # A cancelled/timed-out in-flight call must not leave the step counter advanced
    # with no trajectory entry: the next call has to be step 1, not 2 (contiguous,
    # one Step per completed call).
    ep = await ServedEpisode.start("wordle_v1", task=0)
    try:
        session = ep._sessions["guess"]
        real_call = session.call_tool
        blocker = asyncio.Event()  # never set → the call blocks until cancelled

        async def blocking(*a, **k):
            await blocker.wait()
            return await real_call(*a, **k)

        session.call_tool = blocking  # type: ignore[method-assign]
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ep.call("guess", {"word": _answer(0)}), timeout=0.05)

        assert ep._step == 0  # not advanced by the cancelled call
        assert ep._trajectory == []

        session.call_tool = real_call  # type: ignore[method-assign]
        res = await ep.call("guess", {"word": _answer(0)})
        assert ep._step == 1 and ep._trajectory[0].index == 1  # step 1, not 2
        assert json.loads(res.content)["solved"] is True
    finally:
        await ep.close()


async def test_call_strips_forged_reserved_session_id(tmp_path: Path) -> None:
    # A caller-supplied `_session_id` is a forged reserved field: it must not run
    # against the real session, and it must not be recorded in the trajectory a
    # verifier reads (which would let it score an input never actually sent).
    ep = await ServedEpisode.start("wordle_v1", task=0)
    try:
        res = await ep.call("guess", {"word": _answer(0), "_session_id": "forged"})
        # Ran against the real session (the correct guess still solves), and the
        # recorded step carries only the agent's semantic args — no `_session_id`.
        assert json.loads(res.content)["solved"] is True
        assert ep._trajectory[0].arguments == {"word": _answer(0)}
        assert "_session_id" not in ep._trajectory[0].arguments
    finally:
        await ep.close()


async def test_random_default_task_is_attributed(tmp_path: Path) -> None:
    # With `task` omitted, load_task() still picks a concrete instance; the resolved
    # id must be published (not None) so a random episode stays reproducible/groupable.
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start("wordle_v1", trace_path=trace)  # no task=
    try:
        task_id = ep.describe().task_id
        assert task_id is not None and task_id.isdigit()  # concrete resolved index
        await ep.call("terminate")
    finally:
        await ep.close()
    rows = load_traces(trace)
    assert rows[-1]["task_id"] == task_id  # trace records the resolved id, not null


async def test_start_cleans_up_env_when_setup_fails(monkeypatch) -> None:
    # If setup raises after begin_session pushed per-episode state, start() returns no
    # ServedEpisode for the caller to close — so it must close the env itself, dropping
    # that state. Otherwise the in-process server leaks a session entry permanently.
    from hgym.envs.wordle import env_v1, mcp_server

    original = env_v1.WordleV1Env._begin_session

    def failing(self, session_id, task):
        original(self, session_id, task)  # push the target into the server, then fail
        raise RuntimeError("boom during setup")

    monkeypatch.setattr(env_v1.WordleV1Env, "_begin_session", failing)

    before = set(mcp_server.sessions)
    with pytest.raises(RuntimeError, match="boom during setup"):
        await ServedEpisode.start("wordle_v1", task=0)
    # env.close() ran on the failure path and dropped the pushed session state.
    assert set(mcp_server.sessions) == before


async def test_concurrent_calls_are_serialized(tmp_path: Path) -> None:
    # One episode is a single sequential trajectory. Firing many calls concurrently
    # must not interleave the shared step counter or run past the horizon: the
    # per-episode lock serializes them into unique, ordered steps that stop at 6.
    trace_path = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start("wordle_v1", task=1, trace_path=trace_path)
    try:
        results = await asyncio.gather(
            *[ep.call("guess", {"word": "aaaaa"}) for _ in range(10)]
        )
    finally:
        await ep.close()

    assert len(results) == 10  # every call returns
    rows = load_traces(trace_path)
    # Serialized + horizon-bounded: exactly steps 1..6, none duplicated or beyond it.
    assert [r["step"] for r in rows] == [1, 2, 3, 4, 5, 6]
    assert [r["terminated"] for r in rows] == [False] * 5 + [True]


async def test_horizon_terminates_env_side_without_a_terminate_call(tmp_path: Path) -> None:
    trace_path = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start("wordle_v1", task=1, trace_path=trace_path)
    try:
        terminated_at = None
        for i in range(1, 7):
            res = await ep.call("guess", {"word": "aaaaa"})
            if res.terminated:
                terminated_at = i
                break
        assert terminated_at == 6
        items, terminate = parse_meta(res.meta)
        assert terminate is True
        assert {i.name for i in items} >= {"check_answer", "count_turns"}

        after = await ep.call("guess", {"word": "aaaaa"})
        assert after.terminated is True
    finally:
        await ep.close()

    rows = load_traces(trace_path)
    assert len(rows) == 6  # the post-termination call is not stepped/recorded
    assert rows[-1]["terminated"] is True


async def test_start_rejects_unknown_env() -> None:
    with pytest.raises(ValueError):
        await ServedEpisode.start("does_not_exist")
