"""How a dispensed task ends, and what that costs the numbers.

Every dispensed task lands exactly one :class:`ResultRow`. These tests pin the closure each
terminal path produces and — the point of the taxonomy — that a row the agent did not earn
carries ``score=None``, so it can be counted but never averaged.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, List, Optional

import pytest

import hgym.serve.stream as stream_module
from hgym.serve.episode import ServedEpisode
from hgym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from hgym.serve.stream import (
    _TASK_OVER,
    ResultRow,
    Score,
    TaskRef,
    TaskStream,
    read_dispenses,
    read_results,
    reconcile,
)
from hgym.task import TaskSpec, ToolManifest
from hgym.types import EpisodeFeedback, FeedbackCollection
from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [
    {"id": "q0", "question": "2+2?", "answer": "4"},
    {"id": "q1", "question": "3+3?", "answer": "6"},
]


def _stream(tmp_path: Path, indices: List[int], factory=None, **kwargs: Any) -> TaskStream:
    return TaskStream(
        factory or (lambda _name: _FixtureScoreEnv(tasks=TASKS)),
        [TaskRef(ENV_NAME, i) for i in indices],
        prov_dir=tmp_path / "prov",
        **kwargs,
    )


# ----- the closure taxonomy -----


async def test_agent_submits_is_sealed_and_scored(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score == Score(
        reward=None,
        success=True,
        feedback=[{"name": "correct", "value": True, "level": "episode"}],
    )


async def test_agent_terminates_is_aborted_and_still_scored(tmp_path: Path) -> None:
    # Giving up is a choice the agent made, so its zero is earned and stays in the average.
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        await stream.dispatch("terminate", {})
    (row,) = stream.results
    assert row.closure == "aborted"
    assert row.score is not None and row.score.success is False


async def test_stream_forcing_the_terminal_is_drained_and_scored(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
    (row,) = stream.results
    assert row.closure == "drained"
    assert row.score is not None


async def test_a_failed_finalization_is_unscored(tmp_path: Path) -> None:
    # THE case the taxonomy exists for: the evaluator blew up, so the env stands behind no
    # verdict. The core still fails closed to `correct=False`, and averaging that would book an
    # infrastructure failure as an earned zero.
    def explode(_req: FinalizeRequest, _correct: bool) -> None:
        raise RuntimeError("evaluator exploded")

    stream = _stream(
        tmp_path, [0], factory=lambda _n: _FixtureScoreEnv(tasks=TASKS, finalize_hook=explode)
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    (row,) = stream.results
    assert row.closure == "finalize_error"
    assert row.score is None
    # The fail-closed verdict is still on the row, as evidence — just not as a score.
    assert {"name": "correct", "value": False, "level": "episode"} in row.observed
    assert row.diagnostic is not None


async def test_a_call_that_ended_nothing_cannot_say_how_the_task_ended(tmp_path: Path) -> None:
    # Ordinary concurrent MCP ingress: the agent's `submit` is accepted and its finalizer is
    # still grading when a `terminate` arrives. The episode answers that second call with the
    # post-seal tombstone — `terminated`, like every call after the first terminal — and the
    # tombstone comes back FIRST, because the submission is the one still waiting. Attributing
    # the ending to whichever response reaches the seal first files the agent's earned, scored
    # submission as a task it gave up on.
    release = asyncio.Event()

    class _SlowFinalize(_FixtureScoreEnv):
        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            await release.wait()
            return await super().finalize(req)

    stream = _stream(tmp_path, [0], factory=lambda _n: _SlowFinalize(tasks=TASKS))
    await stream.__aenter__()
    try:
        await stream.get_task()
        submit = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.05)  # the submission is inside its blocked finalizer
        abort = asyncio.ensure_future(stream.dispatch("terminate", {}))
        await asyncio.sleep(0.05)  # the tombstoned abort is what reaches the seal first
        release.set()
        answers = await asyncio.gather(submit, abort)
        # Both are answered with the same redacted constant: which one ended the task is not
        # something either caller may read back off its own response.
        assert [r.content[0].text for r in answers] == [_TASK_OVER] * 2  # type: ignore[union-attr]
    finally:
        release.set()
        await stream.aclose()

    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["sealed"]


async def test_a_terminal_the_stream_did_not_drive_is_not_drained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The same mistake one level in. The drain finds the episode not yet terminated, so it forces
    # the env's terminal — but the agent's own `submit` was queued on the episode's lock behind an
    # ordinary call and wins it, so the forced call is tombstoned and drove nothing. Crediting the
    # drain for it records an outcome the agent earned as one the stream imposed.
    hold = asyncio.Event()
    original = ServedEpisode._dispatch_step

    async def _slow_step(self: ServedEpisode, *args: Any, **kwargs: Any) -> Any:
        await hold.wait()  # holds the EPISODE's lock, which is what queues the two terminals
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(ServedEpisode, "_dispatch_step", _slow_step)

    stream = _stream(tmp_path, [0])
    await stream.__aenter__()
    try:
        await stream.get_task()
        step = asyncio.ensure_future(stream.dispatch("noop", {}))
        await asyncio.sleep(0.05)  # the ordinary call holds the episode's lock
        submit = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.05)  # queued behind it
        closing = asyncio.ensure_future(stream.aclose())
        await asyncio.sleep(0.05)  # the drain's forced terminal queues behind the submission
        hold.set()
        await asyncio.gather(step, submit, closing)
    finally:
        hold.set()
        await stream.aclose()

    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True


async def test_a_forced_terminal_the_env_fails_is_not_a_clean_agent_seal(
    tmp_path: Path,
) -> None:
    # The stream drives the env's terminal for an agent that stopped short — and the env raises
    # once that call has already ended the episode. A non-seal env is the plain shape of it: its
    # `verify` runs inline, after the step is committed as terminal and with no transaction to
    # stamp the failure. Reading the missing return value as "nothing was forced" says the agent
    # ended the task itself: the row lands `sealed` with `success` null, which in a run that
    # finished means the env published no such field, while the queue serves on against an
    # evaluator that will raise for every task in it.
    class _RaisesOnTheTerminalStep(_FixtureScoreEnv):
        score_terminal_tool = None  # a non-seal env: no seal transaction, `verify` runs inline

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if terminated:
                raise RuntimeError("the evaluator exploded on the terminal step")
            return FeedbackCollection()

    stream = _stream(tmp_path, [0, 1], factory=lambda _n: _RaisesOnTheTerminalStep(tasks=TASKS))
    await stream.get_task()
    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.get_task()  # the drain forces the terminal; the queue may not go on

    (row,) = stream.results
    assert row.closure == "finalize_error", "a task the env failed to end is not an agent's seal"
    assert row.score is None
    assert "evaluator exploded" in (row.diagnostic or ""), "the failure must be on the row"
    assert stream.stopped
    with pytest.raises(RuntimeError, match="failed while the stream ended a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, RuntimeError)
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["finalize_error"]


async def test_a_forced_abort_the_env_fails_is_not_an_earned_give_up(tmp_path: Path) -> None:
    # The same defect on a seal-enabled env, where it costs more. The terminal transaction
    # commits and the episode then serializes the feedback it is about to hand over, so a value
    # the wire refuses raises from a forced call that has already ended the task. `terminal_tool`
    # is read off the episode, which sealed by `abort` — so the row does not merely lose its
    # summary, it says the agent *chose* to give up, and `aborted` is a scored closure, so that
    # invented choice goes into the average.
    class _UnserializableOnAbort(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            if terminated and evidence is not None and evidence.source == "abort":
                fb.episode.append(EpisodeFeedback(name="reward", value=float("nan")))
            return fb

    stream = _stream(tmp_path, [0, 1], factory=lambda _n: _UnserializableOnAbort(tasks=TASKS))
    await stream.get_task()
    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.get_task()

    (row,) = stream.results
    assert row.closure == "finalize_error", "the stream's forced abort is not the agent's"
    assert row.score is None
    assert "must be finite" in (row.diagnostic or "")
    with pytest.raises(RuntimeError, match="failed while the stream ended a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, ValueError)


class _Unrenderable(RuntimeError):
    """A failure whose own message cannot be built — what an exception formatted lazily from
    state that is gone by the time it is asked for becomes at the handler that caught it."""

    def __str__(self) -> str:
        raise ValueError("this message cannot be built")


class _RaisesUnrenderably(_FixtureScoreEnv):
    """A non-seal env — `verify` runs inline on the terminal step — that fails there with a
    failure this record cannot describe."""

    score_terminal_tool = None

    def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
        if terminated:
            raise _Unrenderable("the evaluator exploded, and cannot say so")
        return FeedbackCollection()


async def test_a_failure_this_record_cannot_describe_still_lands_its_row(tmp_path: Path) -> None:
    # The same forced terminal, failing with an exception whose own `__str__` raises. Describing
    # a *contained* failure runs the env's code a second time, outside the handler that just
    # contained it, and here that description is an argument to both the row and the stop, so an
    # unguarded one takes them with it: nothing is written for a task that was dispensed,
    # `reconcile` reports a crash that never happened, and the run blames its own storage for the
    # env's failure.
    stream = _stream(tmp_path, [0, 1], factory=lambda _n: _RaisesUnrenderably(tasks=TASKS))
    await stream.get_task()
    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.get_task()  # the drain forces the terminal; the queue may not go on

    (row,) = stream.results
    assert row.closure == "finalize_error", "a failure it cannot describe still ended the task"
    assert row.score is None
    assert "_Unrenderable" in (row.diagnostic or ""), "the type survives when the message cannot"
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["finalize_error"]
    assert reconcile(tmp_path / "prov") == [], "a dispense answered by a row is not a crash"
    with pytest.raises(RuntimeError, match="failed while the stream ended a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, _Unrenderable)


async def test_a_timed_out_task_still_reports_a_failure_it_cannot_describe(
    tmp_path: Path,
) -> None:
    # The deadline owns the closure, so the row is classified without reading the failure at all
    # — which leaves the *stop* as the first thing to describe it, and that runs after the append
    # with the row already durable. Unguarded, the description raises from the claimed tail: the
    # row stands in the file with nothing saying the stream should stop, the queue is served on
    # against an env that failed, and the tail keeps that failure for every later joiner while
    # the drain blames the storage for a row that landed.
    stream = _stream(
        tmp_path, [0, 1], factory=lambda _n: _RaisesUnrenderably(tasks=TASKS), deadline=0.01
    )
    await stream.get_task()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if stream.stopped:
            break

    (row,) = stream.results
    assert row.closure == "timeout", "the deadline classifies the task it ended"
    assert row.diagnostic == "the per-episode deadline elapsed before the task was sealed"
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["timeout"]
    with pytest.raises(RuntimeError, match="failed while the stream ended a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, _Unrenderable), (
        "the stop must name the env's failure, not whatever describing it raised"
    )


def _values_the_record_may_not_read(name: str) -> tuple:
    """Values ``name`` is not allowed to carry, given what this record reads it as.

    Derived from the module's own constants, so a name added to ``_RESERVED_FEEDBACK_NAMES``
    is covered without touching this test. A name outside the two summary families falls to
    the last case, which is the right default: the record owns that name, so *nothing* an env
    publishes under it may be read — a plausible ``True`` least of all."""
    if name in stream_module._REWARD_NAMES:  # read as a number
        return ("false", "0", True)  # `bool` is an `int`, so a number test has to exclude it
    if name in stream_module._SUCCESS_NAMES:  # read as a bool
        return ("false", "0", 1.0)
    # Owned outright: a truthy string, a falsy number, and a value of exactly the right type —
    # the last is the sharpest, since a name the record owns is not the env's to answer even
    # when it answers it in the right shape.
    return ("false", 0.0, True)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (name, value)
        for name in stream_module._RESERVED_FEEDBACK_NAMES
        for value in _values_the_record_may_not_read(name)
    ],
)
async def test_a_reserved_feedback_name_cannot_steer_the_row(
    tmp_path: Path, name: str, value: Any
) -> None:
    """The guard. Every feedback name this record gives a meaning to, published with a value of
    a type it is not allowed to have, against a task the agent got WRONG.

    The rule is one rule for all of them: a wrong-typed reserved value either changes nothing
    about the row, or leaves it unscored and stops the stream loudly. What it may never do is
    quietly produce a *different* answer — `bool("false")` is `True` and `bool(0.25)` is too, so
    truthiness on any of these names is how a run comes to disagree with the raw feedback printed
    beside it. Both instances this repository has had — a coerced summary value, and a coerced
    `finalize_error` turning a clean finalization into an unscored infrastructure failure — fail
    this test against their own unfixed source."""

    class _AlsoPublishes(_FixtureScoreEnv):
        def _verify(self, trajectory: Any, task: Any, **kwargs: Any) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, **kwargs)
            if kwargs.get("terminated"):
                fb.episode.append(EpisodeFeedback(name=name, value=value))
            return fb

    # The clean answer this row has to keep: a wrong submission, sealed and scored False.
    stream = _stream(tmp_path / "clean", [0])
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "5"})
    (clean,) = stream.results
    assert clean.closure == "sealed" and clean.score is not None
    assert clean.score.success is False

    published = _stream(
        tmp_path / "published", [0], factory=lambda _n: _AlsoPublishes(tasks=TASKS)
    )
    stopped: Optional[BaseException] = None
    try:
        async with published:
            await published.get_task()
            await published.dispatch(SUBMIT_TOOL, {"answer": "5"})
    except RuntimeError as exc:  # the loud stop an unreadable summary owes
        stopped = exc
    (row,) = published.results

    # Evidence is kept verbatim either way — the row always says what the env published.
    assert {"name": name, "value": value, "level": "episode"} in row.observed
    assert row.closure == clean.closure, "a published value reclassified the row"
    if row.score is None:
        assert stopped is not None, "the row was left unscored without stopping the stream"
        assert row.diagnostic is not None
    else:
        assert stopped is None
        assert row.score.reward == clean.score.reward
        assert row.score.success == clean.score.success


async def test_the_deadline_is_a_timeout_and_unscored(tmp_path: Path) -> None:
    # A task nobody ever ends: the stream's own clock must end it, since a hung agent never
    # calls back for the deadline to be noticed at the next request.
    stream = _stream(tmp_path, [0, 1], deadline=0.05)
    async with stream:
        await stream.get_task()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if stream.results:
                break
        (row,) = stream.results
        assert row.closure == "timeout"
        assert row.score is None
        # The queue keeps draining: the next task is dispensed normally.
        assert await stream.get_task() is not None
    assert [row.closure for row in stream.results] == ["timeout", "drained"]


async def test_the_deadline_holds_while_a_terminal_call_is_in_flight(tmp_path: Path) -> None:
    # The deadline is a clock on the agent, and an env must not be able to spend it. Grading that
    # outlasts the deadline is the reachable case: the agent's terminal arrives in time, the
    # env's finalizer runs long, and if the stream cannot arbitrate meanwhile the row lands
    # `sealed` with a score — a task that ran over the harness's wall clock recorded as an
    # ordinary one. The clock is per episode, so the whole queue can be spent this way.
    release = asyncio.Event()

    class _SlowFinalize(_FixtureScoreEnv):
        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            await release.wait()
            return await super().finalize(req)

    stream = _stream(
        tmp_path, [0], factory=lambda _n: _SlowFinalize(tasks=TASKS), deadline=0.05
    )
    async with stream:
        await stream.get_task()
        call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.15)  # three deadlines' worth, with the finalizer still running
        release.set()
        await asyncio.wait_for(call, timeout=5)
    (row,) = stream.results
    assert row.closure == "timeout", "an in-flight env call must not outrun the deadline"
    assert row.score is None, "an outcome that missed the wall clock was not earned"


async def test_a_task_finished_in_time_is_not_timed_out(tmp_path: Path) -> None:
    # The deadline must not fire on a task the agent finished inside it.
    stream = _stream(tmp_path, [0], deadline=5.0)
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert [row.closure for row in stream.results] == ["sealed"]


async def test_the_deadline_starts_when_the_task_is_handed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The clock is a budget for the agent, and recording the dispense is not the agent's work:
    # it happens before `get_task` returns, so the agent cannot see the task, act on it, or wait
    # it out. A clock started before that write charges storage latency to the agent, and a
    # volume slower than the deadline hands out a task that has already run out of time.
    real_append = stream_module._append_jsonl

    def _slow_dispense(path: Path, record: Any, **kwargs: Any) -> None:
        if path.name == "dispenses.jsonl":
            time.sleep(0.6)  # well past the deadline below
        real_append(path, record, **kwargs)

    monkeypatch.setattr(stream_module, "_append_jsonl", _slow_dispense)
    stream = _stream(tmp_path, [0, 1], deadline=0.25)
    async with stream:
        task = await stream.get_task()
        assert task is not None
        await asyncio.sleep(0.05)  # the agent thinking, comfortably inside its budget
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        (row,) = stream.results
        assert row.closure == "sealed", "the write the agent waited on spent the agent's clock"
        assert row.score is not None and row.score.success is True

        # ...and the clock is started, not skipped: the next task, left unanswered, still ends.
        assert await stream.get_task() is not None
        for _ in range(200):
            await asyncio.sleep(0.01)
            if len(stream.results) == 2:
                break
    assert [row.closure for row in stream.results] == ["sealed", "timeout"]


async def test_unscored_rows_are_structurally_unaggregatable(tmp_path: Path) -> None:
    # The property the whole taxonomy buys: a mean over `score` cannot see the failures.
    def explode(_req: FinalizeRequest, _correct: bool) -> None:
        raise RuntimeError("boom")

    good = _stream(tmp_path / "a", [0])
    async with good:
        await good.get_task()
        await good.dispatch(SUBMIT_TOOL, {"answer": "4"})
    bad = _stream(
        tmp_path / "b", [0], factory=lambda _n: _FixtureScoreEnv(tasks=TASKS, finalize_hook=explode)
    )
    async with bad:
        await bad.get_task()
        await bad.dispatch(SUBMIT_TOOL, {"answer": "4"})

    rows = [*good.results, *bad.results]
    scored = [row.score for row in rows if row.score is not None]
    assert len(rows) == 2 and len(scored) == 1
    assert [score.success for score in scored] == [True]


# ----- cancellation -----


async def test_cancelling_get_task_leaves_the_position_replayable(tmp_path: Path) -> None:
    # Cancelled while the episode was still opening: nothing was handed out, so no dispense
    # record exists and the position is still there to play.
    stream = _stream(tmp_path, [0, 1])
    async with stream:
        pending = asyncio.ensure_future(stream.get_task())
        await asyncio.sleep(0)  # let it reach the first await inside episode startup
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert read_dispenses(tmp_path / "prov") == []
        assert stream.queue_info().consumed == 0
        # The stream is still usable and the cancelled position replays.
        dispensed = await stream.get_task()
        assert dispensed is not None
    assert [row.position for row in stream.results] == [0]


async def test_cancelling_a_call_in_flight_still_lands_exactly_one_row(tmp_path: Path) -> None:
    # The episode shields its finalization, so a cancelled dispatch cannot abandon the evaluator
    # or produce a second one. The task is still live afterwards, and the drain records it once.
    release = asyncio.Event()

    class _SlowFinalize(_FixtureScoreEnv):
        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            await release.wait()
            return await super().finalize(req)

    stream = _stream(tmp_path, [0], factory=lambda _n: _SlowFinalize(tasks=TASKS))
    async with stream:
        await stream.get_task()
        call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.05)
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        release.set()
    assert len(stream.results) == 1
    assert len(read_results(tmp_path / "prov")) == 1
    assert stream.results[0].score is not None  # the shielded evaluator still produced it


async def test_a_cancelled_terminal_call_is_sealed_not_drained(tmp_path: Path) -> None:
    # The agent's submission was accepted and its (shielded) finalization is still running when
    # the caller goes away. The drain must adopt that outcome, not force a second terminal over
    # the top of it: forcing one reads the post-seal tombstone and would file the agent's
    # correct, scored answer as an unfinished task the stream drained.
    release = asyncio.Event()

    class _SlowFinalize(_FixtureScoreEnv):
        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            await release.wait()
            return await super().finalize(req)

    stream = _stream(tmp_path, [0], factory=lambda _n: _SlowFinalize(tasks=TASKS))
    await stream.__aenter__()
    try:
        await stream.get_task()
        call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.05)  # inside the blocked finalizer
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        closing = asyncio.ensure_future(stream.aclose())
        await asyncio.sleep(0.05)  # the drain must wait for the finalization, not race it
        release.set()
        await closing
    finally:
        await stream.aclose()

    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    assert row.observed == [{"name": "correct", "value": True, "level": "episode"}]
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["sealed"]


async def test_a_call_after_a_cancelled_terminal_is_still_scored_from_it(
    tmp_path: Path,
) -> None:
    # The same adoption at the entry point no drain is involved in: the client that lost the
    # response retries its call. It reaches the entry the cancellation left unclaimed, the
    # episode tombstones it as terminated, and the seal runs from there — so this read has to
    # wait for the verdict too, or a retry files the agent's solve as an empty, unscored row.
    release = asyncio.Event()

    class _SlowFinalize(_FixtureScoreEnv):
        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            await release.wait()
            return await super().finalize(req)

    stream = _stream(tmp_path, [0], factory=lambda _n: _SlowFinalize(tasks=TASKS))
    await stream.__aenter__()
    try:
        await stream.get_task()
        call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.05)  # inside the blocked finalizer
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call

        retry = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.05)
        waited, recorded = not retry.done(), stream.results
        release.set()
        await retry
        assert waited, "the retry read the episode without waiting for its verdict"
        assert recorded == (), "a row was published before the verdict landed"
    finally:
        release.set()  # never leave the evaluator blocked, however this test ends
        await stream.aclose()

    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True


async def test_cancelling_the_drain_leaves_the_task_sealable(tmp_path: Path) -> None:
    # A shutdown cancelled while it was forcing a terminal must not strand the task: an entry
    # marked sealed with no row is invisible to a later drain, so its durable dispense would be
    # reported as a crash that never happened.
    release = asyncio.Event()

    class _SlowNoArgTerminal(_FixtureScoreEnv):
        """A score terminal the stream can force with no arguments (the shape a real env with a
        no-arg `done` has), whose grading blocks — so the drain really seals here."""

        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            spec.tools = [
                ToolManifest(
                    name=m.name,
                    description=m.description,
                    input_schema=(
                        {**m.input_schema, "required": []}
                        if m.name == SUBMIT_TOOL
                        else m.input_schema
                    ),
                    terminal_kind=m.terminal_kind,
                )
                for m in spec.tools
            ]
            return spec

        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            await release.wait()
            return await super().finalize(req)

    stream = _stream(tmp_path, [0], factory=lambda _n: _SlowNoArgTerminal(tasks=TASKS))
    await stream.__aenter__()
    await stream.get_task()

    closing = asyncio.ensure_future(stream.aclose())
    await asyncio.sleep(0.05)  # mid-seal, inside the forced terminal's finalizer
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()

    await stream.aclose()  # the retry completes the shutdown the cancelled one started
    assert len(stream.results) == 1
    assert len(read_results(tmp_path / "prov")) == 1
    # The dispense is answered, so recovery reports no abandoned task.
    assert reconcile(tmp_path / "prov") == []


async def test_a_cancelled_drain_still_releases_what_only_the_stream_holds(
    tmp_path: Path,
) -> None:
    # Draining is work a retry can finish; releasing is not. The catalog envs and the deadline
    # watchdog are held by this object and by nothing else, so a shutdown that loses its caller
    # on the way out would leave an env holding MCP sessions and subprocesses, and a watchdog
    # running against a stream nobody serves — with no later call obliged to arrive. Nothing here
    # retries, deliberately.
    release = asyncio.Event()
    release.set()  # cleared below, so only the catalog env's close is the one that blocks

    class _SlowClose(_FixtureScoreEnv):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.released = False

        async def close(self) -> None:
            await release.wait()
            await super().close()
            self.released = True

    built: List[_SlowClose] = []

    def factory(_name: str) -> _SlowClose:
        env = _SlowClose(tasks=TASKS)
        built.append(env)
        return env

    stream = _stream(tmp_path, [0], factory=factory, deadline=5.0)
    await stream.__aenter__()
    await stream.get_task()
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    watching = [task for task in asyncio.all_tasks() if "_watch_deadlines" in repr(task)]
    assert watching, "the deadline watchdog should be running for this stream"

    release.clear()
    closing = asyncio.ensure_future(stream.aclose())
    await asyncio.sleep(0.05)  # inside the catalog env's close
    closing.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await closing

    # No retry, deliberately: the release must not be owed to a call that may never come.
    for _ in range(200):
        await asyncio.sleep(0.01)
        if built[0].released and all(task.done() for task in watching):
            break
    assert built[0].released, "the catalog env was left open with nothing able to close it"
    assert all(task.done() for task in watching), "the deadline watchdog outlived the stream"


class _SlowRelease(_FixtureScoreEnv):
    """An env whose teardown blocks, so a caller can be lost in the *other* half of a seal: the
    window after the row is durable, while the episode is being let go."""

    def __init__(
        self, closing: asyncio.Event, release: asyncio.Event, summary: Any, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._closing = closing
        self._release = release
        self._summary = summary
        self.released = False

    def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
        fb = FeedbackCollection()
        if terminated:
            fb.episode.append(EpisodeFeedback(name="success", value=self._summary))
        return fb

    async def close(self) -> None:
        self._closing.set()
        await self._release.wait()
        await super().close()
        self.released = True


async def _lose_the_caller_mid_release(
    tmp_path: Path, indices: List[int], summary: Any
) -> Any:
    """Answer the first task, then lose the caller once its row is durable and the episode is
    being let go. Unblocked before the await so a regression *fails* rather than hanging."""
    closing, release = asyncio.Event(), asyncio.Event()
    built: List[_SlowRelease] = []

    def factory(_name: str) -> _SlowRelease:
        env = _SlowRelease(closing, release, summary, tasks=TASKS)
        built.append(env)
        return env

    stream = _stream(tmp_path, indices, factory=factory)
    await stream.get_task()
    call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.wait_for(closing.wait(), timeout=5)
    call.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(call, timeout=5)
    await asyncio.sleep(0.05)
    return stream, built


async def test_a_release_that_loses_its_caller_still_closes_the_episode(
    tmp_path: Path,
) -> None:
    # A seal abandoned *before* its row lands hands the claim back, because nothing was lost.
    # After the row lands there is nothing left to hand back — and `_release` drops the entry
    # from the registry on its way out, so a release abandoned partway would leave an episode
    # nothing holds a handle on, its MCP sessions and its env open for the life of the process
    # and no later drain able to reach them.
    stream, built = await _lose_the_caller_mid_release(tmp_path, [0], True)

    assert built[1].released, "the episode's env was left open with nothing able to close it"
    assert not stream.stopped, "an ordinary cancellation must not end the run"
    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    await stream.aclose()


async def test_a_release_that_loses_its_caller_still_stops_an_unheadlinable_run(
    tmp_path: Path,
) -> None:
    # The other half of that tail is the stop an unheadlinable summary owes, and it is the whole
    # reason the offending row is written first. Losing it leaves the run reporting itself
    # complete while serving the rest of the queue against an env whose headline it has already
    # refused to read.
    stream, built = await _lose_the_caller_mid_release(tmp_path, [0, 1], "false")

    assert stream.stopped, "a lost caller must not cost the run its stop"
    assert built[1].released
    (row,) = stream.results
    assert row.score is None
    assert row.diagnostic is not None and "cannot headline" in row.diagnostic
    with pytest.raises(RuntimeError, match="cannot headline"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="cannot headline"):
        await stream.aclose()


async def test_a_result_row_is_fsynced_before_it_is_relied_on(tmp_path: Path) -> None:
    # `reconcile` treats a missing result as a crash, so a result that only reached the page
    # cache turns a sealed, scored task into a false `broker_abort` after a host crash.
    synced: List[Any] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        try:
            info = os.fstat(fd)
            synced.append((info.st_dev, info.st_ino))
        except OSError:  # pragma: no cover — defensive
            pass
        real_fsync(fd)

    prov = tmp_path / "prov"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", spy)
        async with _stream(tmp_path, [0]) as stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    def ident(path: Path) -> Any:
        info = os.stat(path)
        return (info.st_dev, info.st_ino)

    assert ident(prov / "results.jsonl") in synced
    assert ident(prov / "dispenses.jsonl") in synced
    # The directory entries themselves, so a crash cannot lose a freshly created file.
    assert ident(prov) in synced


async def test_closing_at_the_deadline_records_one_row(tmp_path: Path) -> None:
    # The drain and the deadline race for the same task; exactly one of them may finalize it.
    stream = _stream(tmp_path, [0], deadline=0.05)
    async with stream:
        await stream.get_task()
        await asyncio.sleep(0.05)
    assert len(stream.results) == 1
    assert len(read_results(tmp_path / "prov")) == 1


class _ClosingEnv(_FixtureScoreEnv):
    """Records that its close ran, so a release the stream owes nobody else can be checked."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.closed = False

    async def close(self) -> None:
        await super().close()
        self.closed = True


def _unwritable_results(tmp_path: Path) -> Path:
    prov = tmp_path / "prov"
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "results.jsonl").mkdir()  # a directory: no row can be appended to it
    return prov


async def test_a_deadline_that_cannot_record_its_row_reports_a_stopped_run(
    tmp_path: Path,
) -> None:
    # The watchdog is the one path to a seal with no caller to raise at. A row it cannot write
    # still has to reach the harness — as this stream's stop, in the words the run-level report
    # uses. Failing the watchdog task instead leaves that report unreachable: `aclose` awaits
    # that task inside its `finally`, so the raw storage error escapes in place of the
    # `RuntimeError`, and the failed task stays claimed, so every later attempt raises it again.
    _unwritable_results(tmp_path)
    built: List[_ClosingEnv] = []

    def factory(_name: str) -> _ClosingEnv:
        env = _ClosingEnv(tasks=TASKS)
        built.append(env)
        return env

    stream = _stream(tmp_path, [0, 1], factory=factory, deadline=0.01)
    await stream.get_task()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if stream.stopped:
            break
    assert stream.stopped, "a row the deadline could not record must stop the stream"

    for attempt in ("first", "second"):
        with pytest.raises(RuntimeError, match="record is incomplete") as raised:
            await stream.aclose()
        assert isinstance(raised.value.__cause__, OSError), (
            f"the {attempt} close reported the stop without the failure that caused it"
        )
    assert built and all(env.closed for env in built), "an env was left open by the release"


async def test_a_seal_retried_after_a_failed_append_records_what_it_first_reached(
    tmp_path: Path,
) -> None:
    # The deadline seals the task, its row cannot be written, and the claim is handed back so a
    # later drain can retry the append. What that retry may not do is compose a *second* row: the
    # first attempt already forced this episode's terminal, so a fresh reading finds a task that
    # ended by `terminate` and files the deadline's timeout as an abort the agent chose — a
    # scored closure, carrying whatever summary that reading implies. The harness's own timeout
    # would be recorded as an outcome the agent earned, which is the one thing an unscored
    # closure exists to prevent. The answer the first attempt reached is the true one.
    prov = _unwritable_results(tmp_path)
    stream = _stream(tmp_path, [0, 1], deadline=0.01)
    await stream.get_task()
    for _ in range(200):
        await asyncio.sleep(0.01)
        if stream.stopped:
            break
    assert stream.stopped, "a row the deadline could not record must stop the stream"
    assert stream.queue_info().in_flight == 1, "a failed seal must hand its claim back to retry"
    (prov / "results.jsonl").rmdir()  # the storage recovers before the drain retries

    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()
    (row,) = stream.results
    assert row.closure == "timeout", "the retry reclassified a task the deadline had ended"
    assert row.score is None, "a task that ran out of time may not be scored by its retry"
    assert row.diagnostic == "the per-episode deadline elapsed before the task was sealed"
    assert [r.closure for r in read_results(prov)] == ["timeout"]


async def test_a_dispense_after_a_seal_that_failed_reports_the_stop(tmp_path: Path) -> None:
    # The same contract on the caller-facing path: `get_task` drains what is still live before
    # dispensing, and a seal it cannot finish has already recorded the stop. So the caller is
    # told what stopped the run, not whichever storage error one abandoned episode happened to
    # raise on the way out.
    stream = _stream(tmp_path, [0, 1])
    await stream.get_task()
    (tmp_path / "prov" / "results.jsonl").mkdir()  # the abandoned task's row cannot land

    with pytest.raises(RuntimeError, match="missing an outcome") as raised:
        await stream.get_task()
    assert isinstance(raised.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()


# ----- durability -----


async def test_a_dispense_is_durable_before_the_task_is_handed_out(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        (record,) = read_dispenses(tmp_path / "prov")
        assert record["position"] == 0 and record["task_idx"] == 0
        assert record["lease"] and "answer" not in record
    assert reconcile(tmp_path / "prov") == []  # sealed, so nothing is outstanding


async def test_a_dispense_that_cannot_be_recorded_costs_the_queue_nothing(
    tmp_path: Path,
) -> None:
    # The dispense record is what makes a crash reconcilable, so nothing about a dispense may be
    # committed before it lands. Stepping the position over a write that failed retires a task
    # that was never handed out — no row, no episode any drain can reach, and a queue that reads
    # as cleanly served one task short.
    class _Tracked(_FixtureScoreEnv):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.closed = False

        async def close(self) -> None:
            self.closed = True
            await super().close()

    built: List[_Tracked] = []

    def factory(_name: str) -> _Tracked:
        env = _Tracked(tasks=TASKS)
        built.append(env)
        return env

    stream = _stream(tmp_path, [0, 1], factory=factory)
    before = stream.queue_info()
    real_append = stream_module._append_jsonl

    def refuse_dispenses(path: Path, record: Any, **kwargs: Any) -> None:
        if path.name == "dispenses.jsonl":
            raise OSError("no space left on device")
        real_append(path, record, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(stream_module, "_append_jsonl", refuse_dispenses)
        with pytest.raises(OSError):
            await stream.get_task()

    # Nothing was spent: the position, the sequence and the count are where they were.
    assert stream.queue_info() == before
    assert read_dispenses(tmp_path / "prov") == []
    assert stream.results == ()
    # ...and the episode that was opened for it is closed, not left holding an env the registry
    # never learned about.
    assert built[-1].closed, "the episode opened for the refused dispense leaked its env"

    # A provenance directory that cannot be appended to is not a per-task problem: the rest of
    # the queue would be served over a record that already lost a task. So the run stops, and
    # says so at both boundaries rather than draining as if it were complete.
    assert stream.stopped
    with pytest.raises(RuntimeError, match="could not be recorded as dispensed"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="could not record a dispense"):
        await stream.aclose()


def test_a_killed_stream_leaves_a_reconcilable_dispense(tmp_path: Path) -> None:
    # A real hard kill: the child dispenses a task and dies with no chance to drain. The
    # dispense record is what turns that into a countable `broker_abort` instead of silence.
    prov = tmp_path / "prov"
    script = textwrap.dedent(
        f"""
        import asyncio, os
        from hgym.serve.stream import TaskRef, TaskStream
        from tests._fixtures.score_env import ENV_NAME, _FixtureScoreEnv

        async def main():
            stream = TaskStream(
                lambda _n: _FixtureScoreEnv(tasks=[{{"id": "q0", "question": "?", "answer": "4"}}]),
                [TaskRef(ENV_NAME, 0)],
                prov_dir={str(prov)!r},
            )
            await stream.get_task()
            os._exit(9)  # SIGKILL-equivalent: no drain, no atexit, no flush

        asyncio.run(main())
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parent.parent,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 9, completed.stderr

    assert read_results(prov) == []  # it never sealed anything
    (outstanding,) = reconcile(prov)
    assert outstanding.closure == "broker_abort"
    assert outstanding.score is None
    assert outstanding.position == 0 and outstanding.task_idx == 0


def _tear_final_append(path: Path, keep: int = 12) -> None:
    """Leave the file as a death partway through its last append does: every record before the
    last one intact and terminated, then a prefix of the last with no terminator."""
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    committed = raw.rfind(b"\n", 0, len(raw) - 1) + 1
    path.write_bytes(raw[: committed + keep])


async def test_a_torn_final_append_costs_the_log_only_that_record(tmp_path: Path) -> None:
    # The record that survives a crash is the whole point of these files, so the crash may not
    # be what makes them unreadable. A record whose terminator never landed was never committed
    # — nothing was published on the strength of it — while every record before it was, and
    # recovery has to be able to reach them: this is precisely when `reconcile` is asked to turn
    # an unanswered dispense into the `broker_abort` it is owed.
    prov = tmp_path / "prov"
    async with _stream(tmp_path, [0, 1]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert len(read_results(prov)) == 2

    _tear_final_append(prov / "results.jsonl")

    (survivor,) = read_results(prov)
    assert survivor.position == 0 and survivor.score is not None
    (outstanding,) = reconcile(prov)  # the dispense whose result was lost mid-write
    assert outstanding.position == 1 and outstanding.closure == "broker_abort"
    assert outstanding.score is None
    resumed = _stream(tmp_path, [0, 1], resume=True)
    assert resumed.queue_info().remaining == 1
    await resumed.aclose()

    # ...and the same for the dispense log, which is read back by the same reader.
    _tear_final_append(prov / "dispenses.jsonl")
    assert [record["position"] for record in read_dispenses(prov)] == [0]


async def test_a_committed_record_that_cannot_be_parsed_still_fails_loudly(
    tmp_path: Path,
) -> None:
    # The other half of the same boundary, and the one that must not become permissive: a
    # terminated line is a record something downstream was already told about, so a reader that
    # skipped it would drop a task's outcome and report the rest as the whole run.
    prov = tmp_path / "prov"
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    with (prov / "results.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"seq": 2, "lease": "hal\n')  # committed, and not a record

    with pytest.raises(ValueError, match="line 2 is not a JSON record"):
        read_results(prov)


async def test_a_run_continuing_a_torn_log_does_not_commit_the_fragment(
    tmp_path: Path,
) -> None:
    # A resumed run appends to the file the crash left, and an append that lands straight onto
    # an unterminated fragment fuses the two into one terminated line — turning a record nobody
    # was told about into a committed one that cannot be parsed, which is the state no reader
    # may skip. So the fragment goes before the append, not after it.
    prov = tmp_path / "prov"
    async with _stream(tmp_path, [0, 1]) as first:
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "4"})
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "6"})
    _tear_final_append(prov / "results.jsonl", keep=30)

    async with _stream(tmp_path, [0, 1], resume=True) as second:
        await second.get_task()
        await second.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert [row.position for row in second.results] == [1]
    assert [row.position for row in read_results(prov)] == [0, 1]  # parses, and nothing fused


async def test_a_log_whose_first_record_was_torn_is_still_synced_into_its_directory(
    tmp_path: Path,
) -> None:
    # A file that exists is not the same as a file that holds something. An append that died
    # inside a log's *first* record leaves the file created and the entry naming it unsynced —
    # that sync follows the append that never finished — so the next committed record is still
    # the first this file holds, and a crash could otherwise take the whole log with it while
    # every write reported success.
    prov = tmp_path / "prov"
    prov.mkdir()
    log = prov / "results.jsonl"
    log.write_bytes(b'{"seq": 1, "lea')  # a first append that died partway

    synced: List[Any] = []
    real_fsync = os.fsync

    def spy(fd: int) -> None:
        info = os.fstat(fd)
        synced.append((info.st_dev, info.st_ino))
        real_fsync(fd)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", spy)
        stream_module._append_jsonl(log, {"seq": 1}, durable=True)

    info = os.stat(prov)
    assert (info.st_dev, info.st_ino) in synced, "the entry naming the log was never synced"
    assert stream_module._read_jsonl(log) == [{"seq": 1}]


async def test_a_dispense_is_synced_before_the_byte_that_commits_it(tmp_path: Path) -> None:
    # What makes "terminated" mean "committed" rather than "probably committed": the terminator
    # is written after the record it terminates is already on disk. Handed to the kernel
    # together — as one write, which a record carrying a lot of published feedback still is — a
    # crash could persist the block holding the newline and lose one in the middle, and a torn
    # write would read back as a whole record. Pinned here on the dispense log — the file whose
    # whole purpose is to be read back after the crash that interrupted it, and the one the base
    # commit's result-row test cannot cover because it does not exist there.
    seen: List[bytes] = []
    real_fsync = os.fsync
    dispenses = tmp_path / "prov" / "dispenses.jsonl"

    def spy(fd: int) -> None:
        real_fsync(fd)
        if dispenses.is_file():
            seen.append(dispenses.read_bytes()[-1:])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", spy)
        async with _stream(tmp_path, [0]) as stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    ends = [byte for byte in seen if byte]
    assert ends, "the dispense record was never fsync'd"
    assert ends[0] != b"\n", "the record and its terminator were made durable together"
    assert ends[-1] == b"\n", "the terminator that commits the record was never fsync'd"


def _the_commit_of(path: Path) -> Any:
    """An fsync that fails once, on the sync that commits a record to ``path``.

    The terminator is already flushed when it fails, so every reader takes the record as
    committed — while the caller is told the record was not written. That is the one state the
    split terminator cannot settle on its own: a crash leaves no return value to disagree with,
    an error does."""
    real_fsync = os.fsync
    fired: List[bool] = []

    def spy(fd: int) -> None:
        real_fsync(fd)
        if fired:
            return
        try:
            info, target = os.fstat(fd), path.stat()
        except OSError:
            return
        if (info.st_dev, info.st_ino) != (target.st_dev, target.st_ino):
            return
        if path.read_bytes()[-1:] != b"\n":
            return  # the record itself, not the byte that commits it
        fired.append(True)
        raise OSError(f"cannot commit {path.name}")

    return spy


async def test_a_result_whose_commit_could_not_be_confirmed_is_not_recorded_twice(
    tmp_path: Path,
) -> None:
    # `_record` claims the row only once the append returns, so an append that raises with the
    # terminator already visible leaves the log holding a record the stream never published: the
    # seal hands the claim back and the drain retries the append. Two rows for one dispensed
    # task, same lease and same seq, and a consumer scoring the file counts the outcome twice.
    prov = tmp_path / "prov"
    stream = _stream(tmp_path, [0])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", _the_commit_of(prov / "results.jsonl"))
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        assert stream.stopped
        with pytest.raises(RuntimeError, match="record is incomplete"):
            await stream.aclose()

    rows = read_results(prov)
    assert len(rows) == 1, "one dispensed task, one row — the retry recorded it twice"
    assert len({(row.lease, row.seq) for row in rows}) == 1
    assert [(row.lease, row.seq) for row in rows] == [
        (row.lease, row.seq) for row in stream.results
    ], "the file and the in-memory view must claim the same outcomes"
    assert len(read_dispenses(prov)) == 1
    assert reconcile(prov) == []


async def test_a_dispense_whose_commit_could_not_be_confirmed_is_not_reconciled_as_a_crash(
    tmp_path: Path,
) -> None:
    # The same boundary on the other log, where the harm is the mirror image: nothing retries a
    # dispense — the stream stops and the position is never consumed — so a record left behind by
    # the failed append is a task that was never handed out. `reconcile` reads it as a stream
    # that died mid-task and manufactures a `broker_abort` against a queue position that never
    # ran.
    prov = tmp_path / "prov"
    stream = _stream(tmp_path, [0])
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(os, "fsync", _the_commit_of(prov / "dispenses.jsonl"))
        with pytest.raises(OSError):
            await stream.get_task()
        assert stream.stopped
        with pytest.raises(RuntimeError, match="could not record a dispense"):
            await stream.aclose()

    assert read_dispenses(prov) == [], "a task that was never handed out was recorded as one"
    assert read_results(prov) == []
    assert reconcile(prov) == [], "a crash was reported against a task that never ran"


async def test_resume_replays_by_position_not_by_task_index(tmp_path: Path) -> None:
    # The same index queued twice must play twice, so resume cannot key on the index: the first
    # run seals position 0, and the resumed run owes position 1 — the SAME task index.
    async with _stream(tmp_path, [1, 1]) as first:
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert [row.position for row in first.results] == [0]

    async with _stream(tmp_path, [1, 1], resume=True) as second:
        assert second.queue_info().remaining == 1
        await second.get_task()
        await second.dispatch(SUBMIT_TOOL, {"answer": "6"})
        assert await second.get_task() is None
    assert [row.position for row in second.results] == [1]
    assert [row.position for row in read_results(tmp_path / "prov")] == [0, 1]


async def test_resume_replays_a_position_that_was_abandoned(tmp_path: Path) -> None:
    # An abandoned dispense has no authoritative outcome, so the position is owed, not spent.
    stream = _stream(tmp_path, [0, 1])
    await stream.get_task()  # dispensed, never sealed — the process "dies" here
    assert len(read_dispenses(tmp_path / "prov")) == 1
    assert len(reconcile(tmp_path / "prov")) == 1
    (abandoned,) = read_dispenses(tmp_path / "prov")

    async with _stream(tmp_path, [0, 1], resume=True) as resumed:
        assert resumed.queue_info().remaining == 2
        dispensed = await resumed.get_task()
        assert dispensed is not None
    assert [row.position for row in resumed.results] == [0]

    # ...and the replay is numbered past the dispense it replaces, not over it. The abandoned
    # dispense is durable and holds a `seq` no result answers, so a replay numbered from the
    # results alone takes that same number — leaving `reconcile`'s `broker_abort` and the row
    # the replay earned for the same position sharing one identifier, which is exactly the order
    # identity the record keeps `seq` for.
    seqs = [record["seq"] for record in read_dispenses(tmp_path / "prov")]
    assert seqs == sorted(set(seqs)), f"a resumed dispense reused a recorded seq: {seqs}"
    (outstanding,) = reconcile(tmp_path / "prov")
    assert outstanding.seq == abandoned["seq"]
    assert {row.seq for row in resumed.results}.isdisjoint({outstanding.seq})


async def test_resume_refuses_a_queue_the_provenance_was_not_recorded_against(
    tmp_path: Path,
) -> None:
    # A recorded position is only meaningful next to its own queue. Pointing a DIFFERENT queue at
    # the same provenance directory must not retire task 1 on the strength of a row that task 0
    # earned — the mistake surfaces at construction, before anything is spent.
    async with _stream(tmp_path, [0]) as first:
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert [row.position for row in first.results] == [0]

    with pytest.raises(ValueError, match="position 0"):
        _stream(tmp_path, [1], resume=True)


async def test_resume_refuses_a_queue_too_short_for_its_provenance(tmp_path: Path) -> None:
    # The same mistake in its other shape: a queue that no longer reaches a recorded position.
    async with _stream(tmp_path, [0, 1]) as first:
        for answer in ("4", "6"):
            await first.get_task()
            await first.dispatch(SUBMIT_TOOL, {"answer": answer})

    with pytest.raises(ValueError, match="position 1"):
        _stream(tmp_path, [0], resume=True)


async def test_resume_refuses_a_queue_that_disagrees_with_an_abandoned_dispense(
    tmp_path: Path,
) -> None:
    # A dispense with no result is replayed on resume and paired by `reconcile`; both read its
    # position against the queue, so it is checked even though it sealed nothing.
    stream = _stream(tmp_path, [0])
    await stream.get_task()  # dispensed, never sealed
    assert read_results(tmp_path / "prov") == []

    with pytest.raises(ValueError, match="dispense record"):
        _stream(tmp_path, [1], resume=True)
    await stream.aclose()


async def test_a_recorded_provenance_directory_is_refused_unless_it_is_resumed(
    tmp_path: Path,
) -> None:
    # Without `resume` a stream numbers from the start of its own queue, so a second run into one
    # directory files its rows under the first run's positions with nothing on a row to separate
    # them: the file records a one-task queue as two outcomes while `results` shows one. Both
    # readings are faithful and they disagree, so the call that produced them is refused.
    built: List[_FixtureScoreEnv] = []

    def factory(_name: str) -> _FixtureScoreEnv:
        env = _FixtureScoreEnv(tasks=TASKS)
        built.append(env)
        return env

    async with _stream(tmp_path, [0]) as first:
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert [row.position for row in first.results] == [0]

    with pytest.raises(ValueError, match="already holds records"):
        _stream(tmp_path, [0], factory=factory)
    # Refused before a single env is built, so there is nothing to clean up after it — and the
    # record it protected is untouched: not appended to, and not truncated either.
    assert built == []
    assert [row.position for row in read_results(tmp_path / "prov")] == [0]

    # `resume` is the way to say the continuation was meant, and it still works.
    async with _stream(tmp_path, [0, 1], resume=True) as resumed:
        await resumed.get_task()
        await resumed.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert [row.position for row in read_results(tmp_path / "prov")] == [0, 1]


async def test_an_abandoned_dispense_alone_refuses_a_rerun(tmp_path: Path) -> None:
    # The likelier half, and the one a results-only check would miss: a crash is exactly when a
    # directory gets reused. The abandoned dispense the rerun knows nothing about would become a
    # `reconcile` broker_abort beside the row the rerun earns for the same position — a crash
    # reported against a task that ran, over a queue holding one task.
    stream = _stream(tmp_path, [0])
    await stream.get_task()  # dispensed, never sealed — the process "dies" here
    assert read_results(tmp_path / "prov") == []
    assert len(read_dispenses(tmp_path / "prov")) == 1

    with pytest.raises(ValueError, match="dispenses.jsonl"):
        _stream(tmp_path, [0])
    await stream.aclose()


def test_rows_round_trip_through_the_provenance_file(tmp_path: Path) -> None:
    row = ResultRow(
        seq=1,
        lease="ab",
        position=0,
        env="e",
        task_idx=3,
        closure="timeout",
        score=None,
        observed=[{"name": "correct", "value": False, "level": "episode"}],
        diagnostic="d",
    )
    assert ResultRow.from_wire(json.loads(json.dumps(row.to_wire()))) == row


async def test_rejects_a_deadline_that_cannot_be_enforced(tmp_path: Path) -> None:
    # Nonpositive is the obvious half. The other half is that `now - started >= deadline` is
    # false forever against NaN and against infinity, so either would be accepted, start a
    # watchdog that can never fire, and leave the caller believing a clock was set on a run
    # where nothing was ever timed out. `None` is the way to serve without one, and it is the
    # only way — a disabled deadline should be visible in the call, not in the arithmetic.
    for value in (0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="deadline must be a finite positive"):
            _stream(tmp_path, [0], deadline=value)
    async with _stream(tmp_path, [0], deadline=None) as stream:
        assert await stream.get_task() is not None
