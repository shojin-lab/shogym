"""The episode-serving engine (RFC 008): drive Wordle one tool call at a time, in process
(no subprocess). Checks the wire contract — feedback sidecar, visibility rule,
horizon-as-terminal, and the trace."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import pytest

import shogym
from shogym.feedback import parse_meta
from shogym.serve import ServedEpisode
from shogym.serve.lifecycle import LifecycleState
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.trace import load_traces
from tests._fixtures import score_env, score_mcp


def _answer(task_idx: int) -> str:
    return shogym.make("wordle_v1")._words[task_idx]


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


async def test_a_cancelled_call_is_adopted_rather_than_forgotten(tmp_path: Path) -> None:
    # A cancelled/timed-out call must not leave the step counter advanced with no trajectory
    # entry, and it must not leave the *operation* unaccounted for either. The operation is the
    # episode's, not the caller's: an abandoned await does not stop a call that is already in the
    # env, so what lands is recorded, and the next call follows it rather than taking its number.
    ep = await ServedEpisode.start("wordle_v1", task=0)
    try:
        session = ep._sessions["guess"]
        real_call = session.call_tool
        blocker = asyncio.Event()

        async def blocking(*a, **k):
            await blocker.wait()
            return await real_call(*a, **k)

        session.call_tool = blocking  # type: ignore[method-assign]
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(ep.call("guess", {"word": _answer(0)}), timeout=0.05)

        # Nothing is committed while it is still running: the step and the trajectory entry are
        # written when the operation lands, not when it is issued.
        assert ep._step == 0
        assert ep._trajectory == []

        # And it lands. The caller is gone; the episode is not.
        blocker.set()
        session.call_tool = real_call  # type: ignore[method-assign]
        res = await ep.call("guess", {"word": _answer(0)})
        # Two contiguous steps, one per completed call: the adopted one and this one.
        assert ep._step == 2
        assert [entry.index for entry in ep._trajectory] == [1, 2]
        assert json.loads(res.content)["solved"] is True
    finally:
        await ep.close()


async def test_a_second_call_waits_for_an_abandoned_one_to_land() -> None:
    """The gate a lock could not hold.

    ``asyncio.Lock`` is released as a cancellation unwinds, so a caller that went away left the
    episode open to the next call while its own operation was still in the env.

    The handler here is a synchronous FastMCP tool, so it runs in a worker thread: cancelling the
    coroutine that awaits it abandons the await and not the operation, which is the shape of
    AppWorld's own ``execute`` and the reason a cancellable async substitute cannot show this."""
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    try:
        first = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert not score_mcp.landed.is_set()

        second = asyncio.ensure_future(ep.call("noop", {}))
        await asyncio.sleep(0.1)
        # Held out, and by the operation rather than by the lock the cancellation dropped.
        assert not second.done()
        assert not ep._lock.locked()

        score_mcp.released.set()
        await second
        # Both are in the trajectory, in the order they ran.
        assert [entry.tool for entry in ep._trajectory] == ["block", "noop"]
        assert ep._step == 2
    finally:
        score_mcp.released.set()
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
    from shogym.envs.wordle import env_v1, mcp_server

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


async def test_an_episode_enforces_the_contract_it_advertises() -> None:
    # The contract this episode publishes and the contract it enforces are one object: the score
    # terminal is found by comparing the name a call arrives under against the names in it, and a
    # terminal call's arguments are validated against the schema in it. Both of those are the
    # env's own values, and a JSON scalar has subclasses the models coerce away at construction
    # but not on assignment — so a name or a `const` can serialise as ordinary text and answer a
    # comparison its own way.
    #
    # Advertised, they are what a client is shown and what an agent acts on. Enforced, the same
    # values match nothing the agent can send: the terminal is dispatched as an ordinary step and
    # seals nothing, or the argument the framing described is refused. Both end as a task the
    # agent is recorded as having played badly. So the snapshot is normalised to the wire form.
    from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

    class _NeverEqual(str):
        def __eq__(self, other: object) -> bool:
            return False

        def __ne__(self, other: object) -> bool:
            return True

        __hash__ = str.__hash__

    class _PublishesSubclasses(_FixtureScoreEnv):
        def describe(self, task_id=None):
            spec = super().describe(task_id)
            for manifest in spec.tools:
                if manifest.name == SUBMIT_TOOL:
                    manifest.name = _NeverEqual(SUBMIT_TOOL)
                    schema = json.loads(json.dumps(manifest.input_schema))
                    schema["properties"]["answer"]["const"] = _NeverEqual("4")
                    manifest.input_schema = schema
            return spec

    tasks = [{"id": "q0", "question": "2+2?", "answer": "4"}]
    ep = await ServedEpisode.open_env(
        _PublishesSubclasses(tasks=tasks), env_name=ENV_NAME, task=0
    )
    try:
        published = ep.describe()
        assert all(type(tool.name) is str for tool in published.tools)
        submit = next(t for t in published.tools if t.name == SUBMIT_TOOL)
        assert type(submit.input_schema["properties"]["answer"]["const"]) is str
        # The advertised terminal, called with the advertised-correct argument, ends the task.
        result = await ep.call(SUBMIT_TOOL, {"answer": "4"})
        assert result.terminated is True, "the advertised score terminal sealed nothing"
        assert "validation_error" not in result.content
    finally:
        await ep.close()


async def test_a_cancelled_waiter_does_not_take_the_call_it_was_waiting_for() -> None:
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    try:
        first = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        waiter = asyncio.ensure_future(ep.call("noop", {}))
        await asyncio.sleep(0.1)
        assert not waiter.done()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        # The cancellation is the waiter's own and goes back to it: nothing of `noop` ran.
        assert [entry.tool for entry in ep._trajectory] == []
        score_mcp.released.set()
        await first
        assert [entry.tool for entry in ep._trajectory] == ["block"]
        assert ep._step == 1
    finally:
        score_mcp.released.set()
        await ep.close()


async def test_only_a_forced_terminal_overtakes_an_accepted_call() -> None:
    # The bypass exists for the wall clock: the episode a deadline is for is the one whose
    # ordinary call is not coming back.
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    try:
        running = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        agent = asyncio.ensure_future(ep.call(score_env.SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.1)
        assert not agent.done(), "an agent submission overtook a call already accepted"
        # The wall clock's terminal is the one that does not queue.
        score_mcp.released.set()
        await running
        await agent
        assert ep.terminated is True
    finally:
        score_mcp.released.set()
        await ep.close()


async def test_a_forced_terminal_ends_a_blocked_non_seal_episode() -> None:
    # A deadline has to be enforceable for every env this layer serves. On a non-seal env
    # `terminate` is an ordinary call, so made to queue it could only fire once the thing it was
    # timing had already finished: the one episode a wall clock exists for is the one it could
    # never end.
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    ep._seal_enabled = False  # the same episode, driven down the non-seal path
    try:
        running = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        ended = await asyncio.wait_for(
            ep.call(TERMINATE_TOOL_NAME, {}, forced=True), timeout=2.0
        )
        assert ended.terminated is True
        score_mcp.released.set()
        with contextlib.suppress(BaseException):
            await running
    finally:
        score_mcp.released.set()
        await ep.close()


async def test_an_env_that_cancels_from_end_session_does_not_cancel_the_caller() -> None:
    # `_teardown` and `wait_finalized` run third-party lifecycle code.
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)

    def refuse(_session_id: str) -> None:
        raise asyncio.CancelledError()

    ep._env._end_session = refuse  # type: ignore[method-assign]
    result = await ep.call(score_env.SUBMIT_TOOL, {"answer": "4"})
    assert result.terminated is True
    assert json.loads(result.content)["correct"] is True
    await ep.close()
    await ep.close()


async def test_a_cancelled_close_does_not_score_over_a_running_finalizer() -> None:
    held = asyncio.Event()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    real_finalize = ep._finalize

    async def slow(req: Any) -> Any:
        await held.wait()
        return await real_finalize(req)

    ep._finalize = slow  # type: ignore[assignment]
    submitting = asyncio.ensure_future(ep.call(score_env.SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.sleep(0.05)
    closing = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    with contextlib.suppress(BaseException):
        await closing
    # Turns for a teardown arranged in front of the finalizer to have run, if one was.
    for _ in range(20):
        await asyncio.sleep(0.01)
    # Nothing has been torn down under the finalizer: it is still the owner of the env.
    assert ep._env._gold, "the session was released while finalize was still grading"
    held.set()
    result = await submitting
    assert json.loads(result.content)["correct"] is True
    await ep.close()


async def test_a_cancelled_horizon_call_still_seals_at_the_horizon() -> None:
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    try:
        for _ in range(score_env.HORIZON - 1):
            await ep.call("noop", {})
        assert ep._step == score_env.HORIZON - 1
        reaching = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        reaching.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reaching
        score_mcp.released.set()
        # The dispatch commits the step that reaches the budget and seals as part of committing
        # it, so there is nothing left for a caller to decide.
        for _ in range(100):
            if ep.terminated:
                break
            await asyncio.sleep(0.02)
        assert ep._step == score_env.HORIZON
        assert ep.terminated is True
        await ep.wait_finalized()
        assert ep._state is not LifecycleState.OPEN
    finally:
        score_mcp.released.set()
        await ep.close()


async def test_a_forced_legacy_terminal_does_not_start_a_second_dispatch() -> None:
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    ep._seal_enabled = False
    try:
        running = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        ended = await asyncio.wait_for(
            ep.call(TERMINATE_TOOL_NAME, {}, forced=True), timeout=2.0
        )
        assert ended.terminated is True
        assert ep.terminated is True
        # One dispatch, and it is still the blocked one.
        assert [entry.tool for entry in ep._trajectory] == []
        score_mcp.released.set()
        late = await running
        # The overtaken call is tombstoned when it lands: nothing appended, nothing un-terminated.
        assert late.tombstoned is True
        assert ep.terminated is True
        assert [entry.tool for entry in ep._trajectory] == []
    finally:
        score_mcp.released.set()
        await ep.close()
