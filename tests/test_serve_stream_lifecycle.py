"""How a dispensed task ends, and what that costs the numbers.

Every dispensed task lands exactly one :class:`ResultRow`. These tests pin the closure each
terminal path produces and — the point of the taxonomy — that a row the agent did not earn
carries ``score=None``, so it can be counted but never averaged.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

import shogym.serve.stream as stream_module
from shogym.serve.episode import ServedEpisode
from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from shogym.serve.stream import (
    _TASK_OVER,
    ResultRow,
    Score,
    TaskRef,
    TaskStream,
    read_dispenses,
    read_results,
    reconcile,
)
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.task import TaskSpec, ToolManifest
from shogym.types import EpisodeFeedback, FeedbackCollection
from tests._fixtures.score_env import ENV_NAME, HORIZON, SUBMIT_TOOL, _FixtureScoreEnv

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


def _slow_no_arg_terminal(release: asyncio.Event) -> Any:
    """An env whose score terminal the stream can force with no arguments (the shape a real env
    with a no-arg ``done`` has) and whose grading blocks until ``release`` — so a drain really
    does seal inside it, which is where a cancellation can land."""

    class _SlowNoArgTerminal(_FixtureScoreEnv):
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

    return _SlowNoArgTerminal


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


# ----- a call the harness could not carry -----


def _rekeyed(schema: Dict[str, Any], wrap: Any) -> Dict[str, Any]:
    """The same submit schema with its first property key replaced by an instance of ``wrap``.

    A ``str`` subclass is how a plugin value of this shape reaches the serving layer: the models
    coerce one away at construction and do not validate on assignment, so a manifest mutated after
    it was built carries the subclass through verbatim. Both halves are rekeyed, because a key
    named in ``required`` is looked up in ``properties`` and a schema names the same argument in
    both."""
    props = dict(schema.get("properties") or {})
    key, value = next(iter(props.items()))
    del props[key]
    props[wrap(key)] = value
    required = [wrap(name) if name == key else name for name in schema.get("required") or []]
    return {**schema, "properties": props, "required": required}


def _publishes_a_terminal_schema_that_cannot_be_used() -> Tuple[Any, Any]:
    """An env whose submit schema is exactly what the wire carries and still cannot validate a
    call: a ``$ref`` that points nowhere.

    Nothing above notices — it is plain JSON, it serialises identically wherever the contract is
    compared, and a server advertises it verbatim — and resolving it while checking a terminal
    call raises out of the validator. So the agent is told to call a terminal that answers every
    call with an exception, and so is the stream when it drives that same terminal itself.

    Nothing is armed: this is what the env publishes from the start, which is the shape a real env
    ships when a schema names a definition it never included. (A key whose *own* code misbehaved
    would not reach here any more — an episode enforces the contract in wire form, so an env
    object cannot be what a terminal call is validated against; see
    :func:`shogym.serve.episode._core_spec`.)"""

    def arm(stream: Any = None) -> None:
        # Injected onto the live episode rather than published, because a contract whose schema
        # this layer cannot *execute* is refused at construction now: a document that raises out
        # of the validator will raise on every call any agent can make, so it is an env-wide
        # contract failure rather than a task-local one, and no task is dispensed on it (see
        # `test_a_schema_this_layer_cannot_execute_is_refused_before_it_is_advertised`). What is
        # under test here is what happens when a terminal call raises *mid-episode*, so the
        # unusable schema is put where the seal reads it, on a task that was properly dispensed.
        # Same reasoning as the injections rounds six and ten introduced: the rule is what is
        # under test, and it has to hold for whichever input reaches it next.
        for live in stream._live.values():
            live.episode._score_schemas[SUBMIT_TOOL] = {"$ref": "#/definitions/answer"}

    return (lambda _name: _FixtureScoreEnv(tasks=TASKS)), arm


def _raises_verifying_mid_episode() -> Tuple[Any, Any]:
    """An env whose ``verify`` raises on every non-terminal step. The step is already committed to
    the trajectory when it runs, so the budget is spent on a call that produced nothing."""

    class _Env(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if not terminated:
                raise RuntimeError("this env cannot verify a step")
            return super()._verify(trajectory, task, terminated=terminated, evidence=evidence)

    return (lambda _name: _Env(tasks=TASKS)), (lambda stream=None: None)


def _cannot_read_its_horizon() -> Tuple[Any, Any]:
    """An env whose ``horizon`` raises once the task is out. The episode reads it on every
    ordinary call to decide whether *that* call is the one that reaches the budget, so a horizon
    that cannot be read is an episode that cannot tell whether the task just ended."""
    armed = [False]

    class _Env(_FixtureScoreEnv):
        @property
        def horizon(self) -> Any:
            if armed[0]:
                raise RuntimeError("this env cannot say what its budget is")
            return HORIZON

        @horizon.setter
        def horizon(self, value: Any) -> None:
            pass

    def arm(stream: Any = None) -> None:
        armed[0] = True

    return (lambda _name: _Env(tasks=TASKS)), arm


@pytest.mark.parametrize(
    ("build", "tool", "args", "failure", "stops"),
    [
        pytest.param(
            _publishes_a_terminal_schema_that_cannot_be_used,
            SUBMIT_TOOL,
            {"answer": "4"},
            "definitions/answer",
            True,
            id="the terminal call cannot be validated",
        ),
        pytest.param(
            _raises_verifying_mid_episode,
            "noop",
            {},
            "this env cannot verify a step",
            False,
            id="an ordinary call cannot be verified",
        ),
        pytest.param(
            _cannot_read_its_horizon,
            "noop",
            {},
            "this env cannot say what its budget is",
            False,
            id="the budget cannot be read",
        ),
    ],
)
async def test_a_call_the_harness_lost_is_not_a_task_the_agent_played_out(
    tmp_path: Path, build: Any, tool: str, args: Dict[str, Any], failure: str, stops: bool
) -> None:
    """A call that raises *before* the episode has ended, at each of the places the harness's own
    machinery reads something the env published.

    The exception is still the env's answer to a call the agent can make again, so it goes back
    verbatim and the task stays live. What it may not also be is forgotten. The call reached no
    result, so nothing it was for is on the episode's record — and if the agent never does end the
    task, the drain drives the terminal itself and files the row in a *scored* closure. The first
    of these cases is the sharpest: the agent submitted the right answer, the harness dropped it
    validating it, and the run recorded a task the agent answered wrong.

    What that costs is this task's score, and ``stops`` is which of these cases costs more than
    that: the first case breaks the terminal the *stream* then drives as well, and an env that
    cannot end this task will not end the next one either.

    The class of what comes back is the env's business — a broken schema raises the validator's
    error and a broken callback raises its own — so what is asserted here is that it comes back
    at all, and that the row afterwards names it."""
    factory, arm = build()
    stream = _stream(tmp_path, [0], factory=factory)
    await stream.get_task()
    arm(stream)
    with pytest.raises(Exception):  # noqa: B017 — see above; the type is the env's, not ours
        await stream.dispatch(tool, args)
    # Nothing ended, so nothing is refused yet: the task is still the agent's to finish.
    assert stream.queue_info().in_flight == 1
    assert not stream.stopped

    # It does not finish it, so the stream ends the task — and what the stream ends is not an
    # outcome the agent produced.
    if stops:
        with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
            await stream.aclose()
    else:
        await stream.aclose()
    (row,) = stream.results
    assert row.closure == "finalize_error", "a call the harness lost is not a wrong answer"
    assert row.score is None, "an unearned outcome may never be averaged in"
    # What actually failed is on the row, which is the only place a maintainer will find it: the
    # agent was answered with the redacted constant, and nothing else in the run says the call
    # was ever made.
    assert failure in (row.diagnostic or ""), "the row must name the failure behind it"
    # And the row is the whole answer the lost call itself is owed. `score=None` cannot be
    # averaged into anything, so the record is already honest about this task without the queue
    # behind it paying for one mid-episode failure — the stop belongs to the terminal that also
    # failed, not to the call that did.
    assert stream.stopped is stops
    # The durable record says the same, and it *answers* the dispense: a task handed out and left
    # unscored is not something recovery has to find, because the row is right there.
    durable = read_results(tmp_path / "prov")
    assert [(r.closure, r.score) for r in durable] == [("finalize_error", None)]
    assert reconcile(tmp_path / "prov") == []


@pytest.mark.parametrize(
    ("build", "failure"),
    [
        pytest.param(
            _raises_verifying_mid_episode,
            "cannot verify a step",
            id="an ordinary call cannot be verified",
        ),
        pytest.param(
            _cannot_read_its_horizon,
            "cannot say what its budget is",
            id="the budget cannot be read",
        ),
    ],
)
async def test_a_lost_call_the_agent_recovers_from_is_still_the_agent_s_task(
    tmp_path: Path, build: Any, failure: str
) -> None:
    # The other side of that line, and the one over-correcting would cross. A mid-episode failure
    # is not a verdict about anything: the agent calls again, ends the task itself, and what its
    # own terminal says is what it earned. Refusing the task there would take a run the agent
    # completed and record it as an infrastructure failure.
    #
    # Both boundaries a live call can be lost at, because the loss is kept on the entry and the
    # seal reads it: one that stayed unrecovered lands `finalize_error` with the failure on the
    # row (above), and the same kept loss must *not* follow the agent onto a task it went on to
    # finish. A recovered task carries no scar — no diagnostic beside a score the agent earned,
    # because a row that both scores a success and names a failure is one no reader can act on:
    # it reads as a result arrived at unsoundly, and this one was not.
    factory, arm = build()
    stream = _stream(tmp_path, [0], factory=factory)
    async with stream:
        await stream.get_task()
        arm(stream)
        with pytest.raises(RuntimeError, match=failure):
            await stream.dispatch("noop", {})
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    assert row.diagnostic is None, "a recovered task's earned row carries no failure"
    assert not stream.stopped
    # And the durable record is the same row: the scar would be there to read if it were kept.
    (durable,) = read_results(tmp_path / "prov")
    assert durable.closure == "sealed" and durable.diagnostic is None
    assert durable.score is not None and durable.score.success is True


def _loses_one_call() -> Tuple[Any, Any]:
    """An env that drops exactly one call and then behaves — the transient fault a long run
    actually meets (a session that hiccups once), as opposed to one that will lose a call in
    every task it is given."""
    armed = [False]

    class _Env(_FixtureScoreEnv):
        @property
        def horizon(self) -> Any:
            if armed[0]:
                armed[0] = False  # one call is lost; the next one is answered
                raise RuntimeError("this env lost one call")
            return HORIZON

        @horizon.setter
        def horizon(self, value: Any) -> None:
            pass

    def arm(stream: Any = None) -> None:
        armed[0] = True

    return (lambda _name: _Env(tasks=TASKS)), arm


async def test_a_lost_call_costs_its_own_task_and_not_the_queue_behind_it(
    tmp_path: Path,
) -> None:
    """What the unscored row buys, and the reason it is the whole answer: `score=None` cannot be
    averaged into anything, so the record is honest about the task that lost a call without the
    tasks *after* it paying for that call. A stop here would read one blip mid-episode as a
    verdict on every task still queued."""
    factory, arm = _loses_one_call()
    stream = _stream(tmp_path, [0, 1], factory=factory)
    async with stream:
        await stream.get_task()
        arm(stream)
        with pytest.raises(RuntimeError, match="lost one call"):
            await stream.dispatch("noop", {})
        # The agent gives up on this one and pulls the next, so the stream ends the first itself.
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})
    lost, played = stream.results
    assert lost.closure == "finalize_error" and lost.score is None, (
        "the task whose call was lost is not scored"
    )
    assert played.closure == "sealed", "the next task was served, on a fresh episode"
    assert played.score is not None and played.score.success is True, (
        "and it is scored on what the agent actually did"
    )
    assert not stream.stopped
    assert [(r.closure, r.score is None) for r in read_results(tmp_path / "prov")] == [
        ("finalize_error", True),
        ("sealed", False),
    ]
    assert reconcile(tmp_path / "prov") == []


async def test_a_forced_score_terminal_that_raises_is_not_settled_by_the_abort(
    tmp_path: Path,
) -> None:
    # The same env with no agent call anywhere in it. The drain drives the env's score terminal,
    # which raises, and falls back to the reserved abort — which succeeds, and whose fail-closed
    # `correct=False` is what the row would headline. Reading the abort as an answer to the
    # refusal records a verdict for a task whose only scoring terminal cannot be called, for every
    # task of that env in the queue.
    factory, arm = _publishes_a_terminal_schema_that_cannot_be_used()
    stream = _stream(tmp_path, [0], factory=factory)
    await stream.get_task()
    arm(stream)
    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.aclose()
    (row,) = stream.results
    assert row.closure == "finalize_error", "an abort the stream imposed is not a verdict"
    assert row.score is None
    assert "no verdict" in (row.diagnostic or "")
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["finalize_error"]


async def test_a_required_key_this_schema_cannot_name_is_still_a_validation_error(
    tmp_path: Path,
) -> None:
    # The narrow half: a schema key whose `repr` raises, reached only where a refusal names the
    # argument it is about. Nothing about the contract is unusable here — the caller sent a blank
    # string and can send a real one — so the refusal has to survive its own decoration. Unguarded
    # it becomes an exception instead, and the submission the agent could have corrected is lost.
    class _Key(str):
        def __repr__(self) -> str:
            raise RuntimeError("this schema key cannot be named")

    class _Env(_FixtureScoreEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            for manifest in spec.tools:
                if manifest.name == SUBMIT_TOOL:
                    manifest.input_schema = _rekeyed(manifest.input_schema, _Key)
            return spec

    stream = _stream(tmp_path, [0], factory=lambda _name: _Env(tasks=TASKS))
    async with stream:
        await stream.get_task()
        blank = await stream.dispatch(SUBMIT_TOOL, {"answer": "   "})
        refusal = json.loads(json.loads(blank.content[0].text)["content"])
        assert refusal["validation_error"] is True
        assert refusal["error"].endswith("must be a non-blank string")
        assert "cannot be named" not in refusal["error"], "the env's failure is not the caller's"
        # The episode was never touched, so the answer the agent sends next still earns its score.
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    assert not stream.stopped


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
    # The class is still named, which is the part of a failure that never needed the env's help.
    assert "_Unrenderable: <unrenderable message>" in (row.diagnostic or "")
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["finalize_error"]
    assert reconcile(tmp_path / "prov") == [], "a dispense answered by a row is not a crash"
    assert stream.stopped, "a run that lost an outcome reported itself complete"
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


async def test_an_unformattable_env_failure_still_answers_the_agent_the_same_way(
    tmp_path: Path,
) -> None:
    # The same failure on the agent's own terminal, where it costs more than a row. The stop is
    # recorded from `dispatch`, and building its message raised — so the stop never happened
    # (the queue served on and `aclose` returned clean), and the exception the formatter raised
    # went back to the *agent* in place of the one constant every ending answers with.
    stream = _stream(tmp_path, [0, 1], factory=lambda _n: _RaisesUnrenderably(tasks=TASKS))
    await stream.get_task()
    answer = await stream.dispatch(TERMINATE_TOOL_NAME, {})
    assert answer.content[0].text == _TASK_OVER, "an env failure changed the agent's answer"  # type: ignore[union-attr]
    assert stream.stopped, "the stop the message described was never published"
    with pytest.raises(RuntimeError, match="raised while ending a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, _Unrenderable), "the failure itself is kept"
    assert len(read_results(tmp_path / "prov")) == 1


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


class _StreamClock:
    """The stream's monotonic clock, moved by the test instead of by the runner.

    What the test below is about is *when* the deadline's clock starts, so what it needs is a
    dispense write slower than the deadline, not real seconds spent on one. Left on the wall
    clock it also has to complete its own work inside that same deadline, and that margin is not
    a property of the code under test: a loaded runner loses it and files the passing behaviour
    as a `timeout`. Here the budget is spent only by :meth:`advance`, so load cannot decide the
    outcome in either direction, and the case the test exists to catch (the write's latency
    charged to the agent) is decided by arithmetic rather than by a race.

    Only the stream's own reads move: ``shogym.serve.stream`` reads this clock in exactly two
    places, the dispense that stamps an episode's start and the watchdog scan that enforces the
    deadline against it, which are the two ends of the property. Every other attribute is the
    real module's, `time.time` for the record timestamps included, and everything outside that
    module (this file's `asyncio.sleep`s, the event loop itself) is untouched.
    """

    def __init__(self) -> None:
        # From zero, not from `time.monotonic()`: differences of small exact-ish floats stay
        # exact, where the same additions on top of a six-figure uptime can land a hair under
        # the deadline they were meant to be one whole budget past.
        self._now = 0.0
        self.reads = 0

    def monotonic(self) -> float:
        self.reads += 1
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


async def _a_watchdog_scan(clock: _StreamClock) -> None:
    """Wait until the deadline watchdog has read ``clock`` once, so that an assertion about what
    it left alone is about a scan that happened.

    While the test is awaiting, the watchdog is the only reader of the stream's clock, and one
    scan considers every live entry inside a single critical section, so a read is proof this
    episode was weighed at the current time. Without that proof the assertion after it is
    vacuous: an episode nothing looked at is not an episode that survived. Bounded, and loud when
    the bound elapses, for the same reason.
    """
    seen = clock.reads
    for _ in range(400):
        if clock.reads > seen:
            break
        await asyncio.sleep(0.01)
    assert clock.reads > seen, "the deadline watchdog never read the clock"


async def test_the_deadline_starts_when_the_task_is_handed_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The clock is a budget for the agent, and recording the dispense is not the agent's work:
    # it happens before `get_task` returns, so the agent cannot see the task, act on it, or wait
    # it out. A clock started before that write charges storage latency to the agent, and a
    # volume slower than the deadline hands out a task that has already run out of time.
    clock = _StreamClock()
    monkeypatch.setattr(stream_module, "time", clock)
    real_append = stream_module._append_jsonl

    def _slow_dispense(path: Path, record: Any, **kwargs: Any) -> None:
        if path.name == "dispenses.jsonl":
            clock.advance(0.6)  # a volume well past the deadline below, at no real cost
        real_append(path, record, **kwargs)

    monkeypatch.setattr(stream_module, "_append_jsonl", _slow_dispense)
    stream = _stream(tmp_path, [0, 1], deadline=0.25)
    async with stream:
        task = await stream.get_task()
        assert task is not None
        # Waited for rather than assumed: the watchdog now weighs this episode with the whole
        # write behind it, and charged to the agent the task is 0.6s old against a 0.25s deadline
        # before it is ever handed out, so this is the scan that reaps it.
        await _a_watchdog_scan(clock)
        assert stream.results == (), "the write the agent waited on spent the agent's clock"
        # The budget is the agent's to spend, however long the runner takes to get here: no
        # amount of real time moves this clock.
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        (row,) = stream.results
        assert row.closure == "sealed"
        assert row.score is not None and row.score.success is True

        # ...and the clock is started, not skipped: the next task, left unanswered, still ends.
        # Its budget is spent outright here rather than waited out, which is the same enforcement
        # the runner's load used to have a say in.
        assert await stream.get_task() is not None
        clock.advance(0.3)  # past the whole deadline, with no call back from the agent
        for _ in range(400):
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
    stream = _stream(tmp_path, [0], factory=lambda _n: _slow_no_arg_terminal(release)(tasks=TASKS))
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
    # The seal is held as a task precisely so a caller can go away without taking the rest of it
    # with them, and the release is part of that rest: a row that landed makes the seal final, so
    # every later `_seal` returns it without reaching a release left behind the cancellation, and
    # the episode's MCP sessions and env would stay open for the life of the process.
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


async def _stopped_inside_the_release(
    tmp_path: Path, indices: List[int], summary: Any, *, catalog_blocks: bool = True
) -> Any:
    """Answer the first task and hold the world still *inside* its release: the row is durable,
    the episode has not been let go, and any stop that row owes has not been published."""
    closing, release = asyncio.Event(), asyncio.Event()
    built: List[Any] = []

    def factory(_name: str) -> Any:
        # The catalog env is built first, and a test about shutdown needs it to close fast —
        # otherwise the stream's own release is what `aclose` is waiting on, and the seal it was
        # supposed to wait for proves nothing.
        env: Any = (
            _SlowRelease(closing, release, summary, tasks=TASKS)
            if built or catalog_blocks
            else _FixtureScoreEnv(tasks=TASKS)
        )
        built.append(env)
        return env

    stream = _stream(tmp_path, indices, factory=factory)
    await stream.get_task()
    call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.wait_for(closing.wait(), timeout=5)
    return stream, built, release, call


async def test_a_dispense_waits_for_the_seal_that_still_holds_the_slot(tmp_path: Path) -> None:
    # A task is not finished with its slot when its row lands: the episode behind that row is
    # still open, and its env with it. Dispensing there hands out a second episode against
    # `max_in_flight=1` and starts an agent's clock on a task whose predecessor is still being
    # torn down.
    stream, built, release, call = await _stopped_inside_the_release(tmp_path, [0, 1], True)
    nxt = asyncio.ensure_future(stream.get_task())
    await asyncio.sleep(0.05)

    assert not nxt.done(), "the next task was dispensed while the previous episode was releasing"
    assert not built[1].released
    assert len(read_dispenses(tmp_path / "prov")) == 1, "the durable dispense count moved on"

    release.set()
    assert await asyncio.wait_for(nxt, timeout=5) is not None
    assert built[1].released, "the dispense did not wait for the release to finish"
    assert len(read_dispenses(tmp_path / "prov")) == 2
    await call
    await stream.aclose()


async def test_a_dispense_cannot_step_over_a_stop_the_seal_has_not_published(
    tmp_path: Path,
) -> None:
    # The stop an unheadlinable summary owes is recorded *after* the release, deliberately: the
    # row is the evidence and it lands first. So between the row and the stop there is a stream
    # that has already refused to read this env's headline and has not said so yet. A dispense
    # that reads the row as the end of the seal serves the rest of the queue through that gap,
    # against exactly the env-integrity failure the stop exists to end the queue on.
    stream, built, release, call = await _stopped_inside_the_release(tmp_path, [0, 1], "false")
    nxt = asyncio.ensure_future(stream.get_task())
    await asyncio.sleep(0.05)

    assert not nxt.done(), "a task was dispensed before the seal published its stop"
    assert not stream.stopped

    release.set()
    with pytest.raises(RuntimeError, match="cannot headline"):
        await asyncio.wait_for(nxt, timeout=5)
    await call

    assert stream.stopped
    assert len(read_dispenses(tmp_path / "prov")) == 1, "the queue was served past the stop"
    assert len(stream.results) == 1
    with pytest.raises(RuntimeError, match="cannot headline"):
        await stream.aclose()


async def test_shutdown_waits_for_the_episode_a_seal_is_still_releasing(
    tmp_path: Path,
) -> None:
    # The drain's promise is that every episode this stream dispensed is finished when it
    # returns. A seal whose row has landed is still holding an env and an MCP session until its
    # release returns, so a drain that stops looking at the row returns over a live episode —
    # and over the terminal call that is still waiting to be answered.
    stream, built, release, call = await _stopped_inside_the_release(
        tmp_path, [0], True, catalog_blocks=False
    )
    shutdown = asyncio.ensure_future(stream.aclose())
    await asyncio.sleep(0.05)

    assert not shutdown.done(), "shutdown returned while an episode was still releasing"
    assert not built[1].released

    release.set()
    await asyncio.wait_for(shutdown, timeout=5)
    assert built[1].released, "the drain returned before the episode was let go"
    assert call.done(), "the drain returned while the terminal call was still pending"
    await call


async def test_a_cancelled_drain_records_the_closure_the_drain_actually_forced(
    tmp_path: Path,
) -> None:
    # A shutdown cancelled mid-seal must not change how the task is *classified*. Restarting the
    # seal re-reads an episode the first attempt already terminated, and an episode that ended
    # only because the stream forced its terminal then reads as one the agent ended itself — so
    # a row the stream drained would be filed as earned. The cancelled run must land the same
    # closure the uncancelled one does.
    release = asyncio.Event()
    stream = _stream(tmp_path, [0], factory=lambda _n: _slow_no_arg_terminal(release)(tasks=TASKS))
    await stream.__aenter__()
    await stream.get_task()
    closing = asyncio.ensure_future(stream.aclose())
    await asyncio.sleep(0.05)  # mid-seal, inside the forced terminal's finalizer
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    await stream.aclose()

    # The same run with nothing cancelled, as the reference classification.
    unblocked = asyncio.Event()
    unblocked.set()
    control = _stream(
        tmp_path / "control", [0], factory=lambda _n: _slow_no_arg_terminal(unblocked)(tasks=TASKS)
    )
    async with control:
        await control.get_task()

    (row,) = stream.results
    assert row.closure == control.results[0].closure == "drained"


async def test_shutdown_is_not_bypassed_by_a_dispense_in_flight(tmp_path: Path) -> None:
    # `get_task` opens the spans and the episode with the registry free, so a whole shutdown can
    # complete inside that window. It must not then publish a live task: the drain is over and
    # the watchdog is stopped, so nothing would ever seal it and its durable dispense would read
    # back as a crash the stream never had.
    opening = asyncio.Event()
    release = asyncio.Event()

    class _BlockingSpan:
        """An extension slow enough to hold the dispense open across a shutdown."""

        namespace = "test.blocking"

        async def begin(self, ref: TaskRef) -> Any:
            opening.set()
            await release.wait()
            return self

        @property
        def dispensed(self) -> Dict[str, Any]:
            return {}

        async def finalize(self, completed: Any) -> Dict[str, Any]:
            return {}

    stream = _stream(
        tmp_path, [0, 1], provenance=[_BlockingSpan()], provenance_timeout=None, deadline=0.05
    )
    await stream.__aenter__()
    dispensing = asyncio.ensure_future(stream.get_task())
    await opening.wait()
    await stream.aclose()  # runs to completion while the dispense is still opening
    release.set()

    with pytest.raises(RuntimeError, match="closed"):
        await dispensing
    assert stream.queue_info().in_flight == 0
    # Nothing was exposed, so nothing is owed: no dispense record, and the position is unspent.
    assert read_dispenses(tmp_path / "prov") == []
    assert stream.queue_info().consumed == 0
    assert reconcile(tmp_path / "prov") == []
    await asyncio.sleep(0.15)  # past the deadline: no stranded task to seal late
    assert stream.results == ()


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


class _Interrupted(BaseException):
    """Not an ``Exception``, so every containment boundary in the stream lets it out — the shape
    that fails a seal outright rather than being recorded inside the row it was composing."""


class _RecordsThenInterruptsOnce:
    """A span that reports the closure each seal hands it, and interrupts the first one."""

    namespace = "test.interrupting"

    def __init__(self) -> None:
        self.closures: List[str] = []

    async def begin(self, ref: TaskRef) -> Any:
        return self

    @property
    def dispensed(self) -> Dict[str, Any]:
        return {}

    async def finalize(self, completed: Any) -> Dict[str, Any]:
        self.closures.append(completed.closure)
        if len(self.closures) == 1:
            raise _Interrupted("the seal is interrupted while this span is closing")
        return {}


class _RecordsWhatItWasHanded:
    """A span that closes cleanly and says so, in a payload naming the span that returned it."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.closures: List[str] = []

    async def begin(self, ref: TaskRef) -> Any:
        return self

    @property
    def dispensed(self) -> Dict[str, Any]:
        return {"opened": self.namespace}

    async def finalize(self, completed: Any) -> Dict[str, Any]:
        self.closures.append(completed.closure)
        return {"closed": self.namespace}


class _InterruptsTheFirstTerminal(_FixtureScoreEnv):
    """A non-seal env, so `_verify` runs inline and what it raises reaches the seal through the
    tool call itself. It interrupts the first terminal the stream drives and answers every one
    after it: what fails a seal *before* it has any verdict to compose a row from, and what a
    recomposition would be answered by rather than interrupted."""

    score_terminal_tool = None

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.terminals = 0

    def _verify(
        self, trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None
    ) -> FeedbackCollection:
        if terminated:
            self.terminals += 1
            if self.terminals == 1:
                raise _Interrupted("the seal is interrupted before it has a verdict")
        return super()._verify(trajectory, task, terminated=terminated, evidence=evidence)


async def test_a_seal_that_failed_before_its_row_is_not_recomposed_by_the_retry(
    tmp_path: Path,
) -> None:
    # The sibling of the test above, one step earlier, and the worse half. A seal that fails at
    # the *append* keeps its composed row, so the retry writes what the first attempt reached. A
    # seal that fails before it has composed anything kept nothing, so the retry re-entered the
    # composition: it drove a second terminal into an episode the first attempt had already
    # ended, read the ending back off that, and filed a task the stream drained under a scored
    # closure the agent never earned, running every span's `finalize` on the way. Composing is the
    # half that may not run twice, so a failure in it stands an unscored row in and the retry
    # writes that instead.
    env = _InterruptsTheFirstTerminal(tasks=TASKS)
    span = _RecordsWhatItWasHanded("test.span")
    stream = _stream(tmp_path, [0], factory=lambda _n: env, provenance=[span])
    assert await stream.get_task() is not None

    with pytest.raises(_Interrupted):
        await stream.aclose()  # the seal fails before it has read any verdict at all
    (live,) = stream._live.values()  # noqa: SLF001
    assert live.pending_row is not None, "a hand-back left the retry nothing but a recomposition"
    # The hand-back is the claim and nothing else: the task stays *ended*, which is what a late
    # call is refused on, so the drain window does not reopen from the inside.
    assert live.ended is True and live.sealed is False

    with pytest.raises(RuntimeError, match="seal failed before its row was recorded"):
        await stream.aclose()  # the retry the hand-back exists for

    assert env.terminals == 1, "the retry drove a second terminal"
    assert span.closures == [], "the retry composed a second row"
    assert [(r.closure, r.score) for r in stream.results] == [("finalize_error", None)], (
        "a task the stream drained was recorded under a closure the agent earned"
    )
    assert [(r.closure, r.score) for r in read_results(tmp_path / "prov")] == [
        ("finalize_error", None)
    ]
    # The row says the seal produced none, what failed it, and which ending it was reaching for —
    # the drain's, here, which the closure alone can no longer say.
    diagnostic = stream.results[0].diagnostic or ""
    assert diagnostic.startswith("the seal failed before it composed a row")
    assert "_Interrupted" in diagnostic and "sealing for 'drained'" in diagnostic
    # One dispense, one row: `reconcile` has nothing left to answer as a crash, and the row's
    # span entry keeps the shape an orderly row has — a `dispensed` with exactly one of `sealed`
    # or `error` beside it.
    assert reconcile(tmp_path / "prov") == []
    (entry,) = stream.results[0].extensions.values()
    assert set(entry) == {"dispensed", "error"}


async def test_a_span_that_closed_before_the_seal_failed_keeps_what_it_returned(
    tmp_path: Path,
) -> None:
    # The spans are closed one at a time, so a failure that escapes the finalization leaves three
    # different spans behind: one that closed and returned its payload, the one the failure came
    # out of, and one that was never called at all. Answering for all three with that one failure
    # drops the payload the first span returned and files its namespace under a failure another
    # extension raised, and records a span that was never asked as one whose `finalize` failed.
    # The row is the only account these extensions get, and both of those are false in it.
    closed = _RecordsWhatItWasHanded("test.closed")
    interrupting = _RecordsThenInterruptsOnce()
    unreached = _RecordsWhatItWasHanded("test.unreached")
    stream = _stream(tmp_path, [0], provenance=[closed, interrupting, unreached])
    assert await stream.get_task() is not None

    with pytest.raises(_Interrupted):
        await stream.aclose()  # the seal fails inside the finalization, with one span closed
    with pytest.raises(RuntimeError, match="seal failed before its row was recorded"):
        await stream.aclose()  # the retry the hand-back exists for

    assert closed.closures == ["drained"], "a span that had closed was finalized a second time"
    assert unreached.closures == [], "a span the seal never reached was finalized anyway"
    (row,) = read_results(tmp_path / "prov")
    assert row.extensions["test.closed"] == {
        "dispensed": {"opened": "test.closed"},
        "sealed": {"closed": "test.closed"},
    }, "a span that closed was recorded under a failure another extension raised"
    interrupted = row.extensions["test.interrupting"]
    assert "sealed" not in interrupted, "a span that never returned was recorded as if it had"
    assert "_Interrupted" in interrupted["error"]
    never = row.extensions["test.unreached"]
    assert "sealed" not in never
    assert never["error"].startswith("the seal failed before this span was finalized"), (
        "a span whose `finalize` was never called was recorded as one that failed"
    )


async def test_a_seal_the_extensions_could_not_finish_keeps_the_outcome_it_earned(
    tmp_path: Path,
) -> None:
    # The extensions are the last thing a seal runs, so a failure that escapes them arrives with
    # the episode already sealed, its verdict already read and its closure already classified. An
    # extension may not change a task's outcome, which is what finalizing them after the
    # classification is for, and neither may a failure the stream cannot contain. Standing an
    # unscored row in over the top of one files a task the agent solved as an infrastructure
    # failure, for a verdict the seal was holding when it failed.
    span = _RecordsThenInterruptsOnce()
    stream = _stream(tmp_path, [0], provenance=[span])
    assert await stream.get_task() is not None

    with pytest.raises(_Interrupted):
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    with pytest.raises(RuntimeError, match="seal failed before its row was recorded"):
        await stream.aclose()  # the retry the hand-back exists for

    assert span.closures == ["sealed"], "the retry finalized the span a second time"
    (row,) = read_results(tmp_path / "prov")
    assert row.closure == "sealed", "a task the agent sealed was recorded as a failed seal"
    assert row.score is not None and row.score.success is True, (
        "an outcome the agent earned was dropped by an extension that could not be finalized"
    )
    assert reconcile(tmp_path / "prov") == []


async def test_a_seal_that_failed_short_of_its_row_is_not_reported_as_a_storage_failure(
    tmp_path: Path,
) -> None:
    # A failed seal and a failed *write* stop the stream for different reasons, and the stop's
    # words are all an operator gets. A seal that failed above the append leaves its row on the
    # entry and the retry writes it: the storage never refused anything, `reconcile` has nothing
    # left to answer, and the record the run ends with is complete. Reported as the append's
    # failure it sends that operator to `results.jsonl` for the one part of this that worked, and
    # goes on calling the record incomplete after the row completing it has landed.
    span = _RecordsThenInterruptsOnce()
    stream = _stream(tmp_path, [0], provenance=[span])
    assert await stream.get_task() is not None

    with pytest.raises(_Interrupted):
        await stream.aclose()
    with pytest.raises(RuntimeError, match="seal failed before its row was recorded") as closing:
        await stream.aclose()  # the retry writes the row, and the stop is still owed
    assert isinstance(closing.value.__cause__, _Interrupted)
    assert "results.jsonl" not in str(closing.value)
    assert "record is incomplete" not in str(closing.value)
    # The record it stopped over is complete: the row is durable, and no dispense is unanswered.
    assert len(read_results(tmp_path / "prov")) == 1
    assert reconcile(tmp_path / "prov") == []
    # The same failure in the same words at the other boundary a stop is reported from.
    with pytest.raises(RuntimeError, match="seal failed before its row was recorded") as pulling:
        await stream.get_task()
    assert "results.jsonl" not in str(pulling.value)


async def test_a_released_episode_owing_only_a_row_is_not_reported_in_flight(
    tmp_path: Path,
) -> None:
    # A registry entry stopped meaning "an episode is live" when it began outliving its own
    # release. A seal that failed on the storage keeps its composed row, hands its claim back so
    # a later drain can retry the append, and is released in the meantime — so the entry reads
    # unsealed with its env already closed. Counting it tells a harness whose stream is stopped
    # and closed that a task is still being played, and tells an agent there is still something
    # to answer; there is not, and what the entry is owed is a row.
    prov = _unwritable_results(tmp_path)
    built: List[_ClosingEnv] = []

    def factory(_name: str) -> _ClosingEnv:
        env = _ClosingEnv(tasks=TASKS)
        built.append(env)
        return env

    stream = _stream(tmp_path, [0], factory=factory)
    await stream.get_task()
    assert stream.queue_info().in_flight == 1, "a live episode must be counted"
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    # The seal failed and handed its claim back, and nothing has released the episode yet: this
    # one really is still in flight. The count that changed is the one after the close.
    assert stream.queue_info().in_flight == 1

    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()
    assert stream.stopped and built[1].closed
    assert stream.queue_info().in_flight == 0, "a closed stream reported a released episode live"

    # ...and the count is a *discrimination*, not a deletion. The entry is still there with its
    # composed row on it, so a drain that meets working storage still lands the row it owes —
    # which is the whole reason the entry outlives its release.
    (prov / "results.jsonl").rmdir()
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()
    (row,) = read_results(prov)
    assert row.closure == "sealed" and row.score is not None and row.score.success is True
    assert stream.queue_info().in_flight == 0


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
        from shogym.serve.stream import TaskRef, TaskStream
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


# ------------------------------------------------------------------------------------------
# Cancellation an env or an extension *emits*
#
# `asyncio.CancelledError` inherits from `BaseException`, so it walks straight through an
# `except Exception` written to contain everything. In this module that handler shape means two
# unrelated things — a boundary whose contract is that nothing escapes, and a deliberate
# passthrough where a caller's cancellation must end that caller — and the same exception object
# is legitimate at one and a defect at the other. These pin the containment side: a `CancelledError`
# raised by third-party code is that code failing, and it may not change what the harness answers,
# what it records, or whether it can be shut down.


class _CancelsOnClose(_FixtureScoreEnv):
    """An env whose teardown raises `CancelledError` — what an env inherits by awaiting a child
    task that was cancelled, and what it can simply raise."""

    def __init__(self, cancels: bool, summary: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cancels = cancels
        self._summary = summary
        self.closes = 0

    def _verify(
        self, trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None
    ) -> FeedbackCollection:
        fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
        if terminated and self._summary is not None:
            fb.episode.append(EpisodeFeedback(name="success", value=self._summary))
        return fb

    async def close(self) -> None:
        await super().close()
        self.closes += 1
        if self._cancels:
            raise asyncio.CancelledError()


def _cancelling_close_factory(which: str, **kwargs: Any) -> Tuple[Any, List[Any]]:
    """A factory in which exactly one instance's teardown cancels.

    The constructor builds one long-lived *catalog* env to read the published contract, and the
    factory is called again per dispense for the *episode* env — two different instances closed
    by two different teardowns, so which one raises picks the site under test."""
    built: List[Any] = []

    def factory(_name: str) -> Any:
        nth = len(built)
        env = _CancelsOnClose(
            cancels=(nth == 0) if which == "catalog" else (nth > 0), tasks=TASKS, **kwargs
        )
        built.append(env)
        return env

    return factory, built


@pytest.mark.parametrize(
    "summary, headlinable", [(None, True), ("false", False)], ids=["scorable", "unheadlinable"]
)
async def test_an_episode_teardown_that_cancels_still_answers_and_still_stops(
    tmp_path: Path, summary: Any, headlinable: bool
) -> None:
    # The per-episode release runs as the entry's own claimed task and its callers join it
    # through a shield, so a `CancelledError` observed inside it is not cancellation of the
    # caller — the shield already separated the two. Letting it out lands it in the middle of
    # `_run_seal`, PAST the durable append and BEFORE the stop that seal still owes: the agent's
    # terminating call answers with a traceback instead of the constant, the stop is never
    # published, and `aclose` then reports a clean run over an env whose headline this record has
    # already refused to read. The row is durable either way; what is at stake is everything the
    # seal does after it.
    factory, built = _cancelling_close_factory("episode", summary=summary)
    stream = _stream(tmp_path, [0, 1], factory=factory)
    await stream.get_task()

    answer = await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert answer.content[0].text == _TASK_OVER, "a cancelling teardown changed the agent's answer"  # type: ignore[union-attr]
    assert built[1].closes == 1, "the episode's env was never closed, so nothing was under test"
    (row,) = stream.results
    assert row.closure == "sealed"
    assert reconcile(tmp_path / "prov") == [], "the durable dispense was left unanswered"

    if headlinable:
        assert row.score is not None and row.score.success is True
        assert not stream.stopped, "a teardown failure is not the run's outcome"
        await stream.aclose()
    else:
        assert row.score is None
        assert row.diagnostic is not None and "cannot headline" in row.diagnostic
        assert stream.stopped, "the stop the seal owed was lost with the cancelled release"
        with pytest.raises(RuntimeError, match="cannot headline"):
            await stream.get_task()
        with pytest.raises(RuntimeError, match="cannot headline"):
            await stream.aclose()


async def test_a_catalog_teardown_that_cancels_still_lets_the_stream_close(
    tmp_path: Path,
) -> None:
    # The stream's own release is claimed once and every later `aclose` joins that same task, so
    # a cancelled one is not a failure to retry — it is the claim, and it answers every arrival
    # for the rest of the process. The catalog entry is popped before the close, so nothing can
    # even find the env again: shutdown would have no orderly exit and no way back.
    factory, built = _cancelling_close_factory("catalog")
    stream = _stream(tmp_path, [0], factory=factory)
    await stream.get_task()
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    await stream.aclose()
    await stream.aclose()  # a poisoned release raises here, and on every attempt after it
    assert built[0].closes == 1, "the catalog env was never closed, so nothing was under test"
    assert len(stream.results) == 1
    with pytest.raises(RuntimeError, match="this stream is closed"):
        await stream.get_task()


class _CancelsOnTheTerminalStep(_FixtureScoreEnv):
    """A non-seal env — no seal transaction, so `_verify` runs inline and its failure reaches
    the stream through the tool call itself. (A `score` env's terminal transaction contains any
    evaluator failure, cancellation included, and fails the verdict closed instead.)"""

    score_terminal_tool = None

    def _verify(
        self, trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None
    ) -> FeedbackCollection:
        if terminated:
            raise asyncio.CancelledError()
        return FeedbackCollection()


async def test_an_env_that_cancels_on_the_agents_terminal_answers_the_same_way(
    tmp_path: Path,
) -> None:
    # The terminal is committed and the env is what failed, so a row is owed and the stream must
    # stop — the same answer an env raising anything else gets, and for the same reason. Read as
    # cancellation instead it skips all of it: `dispatch` raises at the agent on exactly the call
    # the redaction exists for, the stop is never recorded so the queue serves on against an env
    # that raises for every task in it, and `aclose` returns clean having lost an outcome.
    stream = _stream(tmp_path, [0, 1], factory=lambda _n: _CancelsOnTheTerminalStep(tasks=TASKS))
    await stream.get_task()

    answer = await stream.dispatch(TERMINATE_TOOL_NAME, {})
    assert answer.content[0].text == _TASK_OVER, "an env failure changed the agent's answer"  # type: ignore[union-attr]
    assert stream.stopped, "the stream served on against an env that raised at the terminal"
    assert len(stream.results) == 1
    with pytest.raises(RuntimeError, match="raised while ending a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, asyncio.CancelledError)
    assert len(read_results(tmp_path / "prov")) == 1


async def test_an_env_that_cancels_on_a_forced_terminal_still_records_and_stops(
    tmp_path: Path,
) -> None:
    # The same env on the stream's own forced terminal, which is where it costs most: this runs
    # inside the seal task, and the seal is a task precisely so a lost caller cannot restart it.
    # A cancellation out of here cancels the claim itself — no row is ever composed, the entry
    # keeps a seal nothing can complete, and every later drain re-awaits it and raises again, so
    # an orderly shutdown reconciles as a crash it never was.
    stream = _stream(tmp_path, [0, 1], factory=lambda _n: _CancelsOnTheTerminalStep(tasks=TASKS))
    await stream.get_task()

    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.get_task()  # the drain forces the terminal; the queue may not go on
    (row,) = stream.results
    assert row.closure == "finalize_error", "an env the stream could not end is not an agent seal"
    assert row.score is None
    assert "CancelledError" in (row.diagnostic or ""), "the failure must be on the row"
    with pytest.raises(RuntimeError, match="failed while the stream ended a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, asyncio.CancelledError)
    assert [r.closure for r in read_results(tmp_path / "prov")] == ["finalize_error"]
    assert reconcile(tmp_path / "prov") == []


async def test_a_refused_dispense_reports_the_refusal_a_cancelling_teardown_cannot_replace(
    tmp_path: Path,
) -> None:
    # A manifest that drifts is refused before anything is exposed, and the episode opened for it
    # is released on the way out — teardown, whose failure must not stand in for the refusal every
    # caller of `get_task` is told to expect. Nothing was spent, so the damage is only the answer;
    # but the answer is the whole contract here, and a `CancelledError` also ends whoever is
    # dispensing rather than being an error they can act on.
    built: List[Any] = []

    class _DriftsAndCancels(_CancelsOnClose):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            if self._cancels:  # the episode instance, never the catalog one
                spec.tools = [
                    *spec.tools,
                    ToolManifest(
                        name="hint",
                        description="Ask for a hint.",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ]
            return spec

    def factory(_name: str) -> Any:
        env = _DriftsAndCancels(cancels=bool(built), tasks=TASKS)
        built.append(env)
        return env

    stream = _stream(tmp_path, [0, 1], factory=factory)
    with pytest.raises(RuntimeError, match="different tool manifest"):
        await stream.get_task()
    assert built[1].closes == 1, "the refused episode's env was never closed"
    assert stream.results == ()
    assert not (tmp_path / "prov" / "results.jsonl").exists()


def test_a_failed_constructor_reports_its_own_error_when_the_catalog_cancels(
    tmp_path: Path,
) -> None:
    # Sync, and outside a running loop on purpose: that is where `_close_on_owning_loop` runs the
    # close to completion rather than scheduling it, so the env's failure reaches the cleanup.
    # This is the one path whose whole job is not to mask the error being raised.
    prov = tmp_path / "prov"
    prov.mkdir(parents=True)
    (prov / "dispenses.jsonl").write_text(
        json.dumps(
            {
                "seq": 1,
                "position": 0,
                "env": "some_other_env",
                "task_idx": 0,
                "lease": "x",
                "dispensed_at": 0.0,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="but this queue holds") as refused:
        TaskStream(
            lambda _n: _CancelsOnClose(cancels=True, tasks=TASKS),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=prov,
            resume=True,
        )
    notes = getattr(refused.value, "__notes__", [])
    assert any("could not be closed" in note for note in notes), (
        "a cleanup that could not finish must be reported on the error being raised"
    )


# ----- the structural guard -----
#
# Three passes of this review found the same defect at three different handlers, which makes the
# handler the wrong unit to fix. The rule now lives in one place — `_must_propagate` — and these
# two keep it from being bypassed: the first is behavioural, over every third-party surface the
# stream calls; the second refuses the handler shape that hid it, at the source.


class _CancellingSpan:
    dispensed = {"opened": True}

    async def finalize(self, completed: Any) -> Dict[str, Any]:
        raise asyncio.CancelledError()


class _CancellingProvenance:
    namespace = "test.cancels"

    def __init__(self, where: str) -> None:
        self._where = where

    async def begin(self, ref: TaskRef) -> Any:
        if self._where == "begin":
            raise asyncio.CancelledError()
        return _CancellingSpan()


def _stream_whose(surface: str, tmp_path: Path) -> Tuple[TaskStream, str, Dict[str, Any]]:
    """A stream in which exactly one third-party surface raises `CancelledError`, plus the tool
    that ends its task."""
    if surface == "an episode teardown":
        factory, _ = _cancelling_close_factory("episode")
        return _stream(tmp_path, [0, 1], factory=factory), SUBMIT_TOOL, {"answer": "4"}
    if surface == "the catalog teardown":
        factory, _ = _cancelling_close_factory("catalog")
        return _stream(tmp_path, [0, 1], factory=factory), SUBMIT_TOOL, {"answer": "4"}
    if surface == "a terminal step":
        return (
            _stream(tmp_path, [0, 1], factory=lambda _n: _CancelsOnTheTerminalStep(tasks=TASKS)),
            TERMINATE_TOOL_NAME,
            {},
        )
    if surface == "a span opening":
        return (
            _stream(tmp_path, [0, 1], provenance=[_CancellingProvenance("begin")]),
            SUBMIT_TOOL,
            {"answer": "4"},
        )
    if surface == "a span closing":
        return (
            _stream(tmp_path, [0, 1], provenance=[_CancellingProvenance("finalize")]),
            SUBMIT_TOOL,
            {"answer": "4"},
        )
    raise AssertionError(f"unknown surface {surface!r}")


@pytest.mark.parametrize("ended_by", ["the agent", "the drain"])
@pytest.mark.parametrize(
    "surface",
    [
        "an episode teardown",
        "the catalog teardown",
        "a terminal step",
        "a span opening",
        "a span closing",
    ],
)
async def test_no_cancellation_an_env_or_extension_raises_reaches_the_harness(
    tmp_path: Path, surface: str, ended_by: str
) -> None:
    # One statement over every third-party surface this module calls: whatever an env or an
    # extension raises, the harness's own API never answers with cancellation. It may raise the
    # loud integrity error, it may return normally — but a caller of `get_task`, `dispatch` or
    # `aclose` is never *cancelled* by code it was only supposed to be running.
    stream, tool, args = _stream_whose(surface, tmp_path)
    cancelled: List[str] = []

    async def call(label: str, coro: Any) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            cancelled.append(label)
        except Exception:  # noqa: BLE001 — only cancellation is under test here
            pass

    await call("get_task", stream.get_task())
    if ended_by == "the agent":
        await call("dispatch", stream.dispatch(tool, dict(args)))
    await call("aclose", stream.aclose())
    await call("aclose again", stream.aclose())

    assert cancelled == [], f"{surface} cancelled the harness's own caller"


def test_a_containment_boundary_may_not_catch_exception_and_leave_cancellation_to_chance() -> None:
    # The source-level half, because the behavioural half above can only cover the surfaces that
    # exist today. Any `try` that runs an env's or an extension's code is a containment boundary,
    # and `except Exception` there is silently *not* a decision about `CancelledError` — which is
    # exactly how this landed three times. So the shape is refused: such a boundary catches
    # `BaseException` and asks `_must_propagate`, or it names `asyncio.CancelledError` itself.
    # Elsewhere `except Exception` keeps its meaning, a deliberate passthrough of the caller's own
    # cancellation, and is left alone.
    #
    # A tripwire, not a proof: it keys on the names this module gives its third-party handles, so
    # renaming one would slip past. It fails at the moment the shape is written, which a comment
    # cannot do.
    handles = frozenset({"env", "episode", "undispensed", "span", "extension"})
    source = Path(stream_module.__file__).read_text()
    offenders: List[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Try):
            continue
        mentioned = set()
        for statement in node.body:
            for inner in ast.walk(statement):
                if isinstance(inner, ast.Name):
                    mentioned.add(inner.id)
                elif isinstance(inner, ast.Attribute):
                    mentioned.add(inner.attr)
        if not mentioned & handles:
            continue

        def caught(handler: ast.ExceptHandler) -> List[str]:
            kinds = handler.type
            listed = kinds.elts if isinstance(kinds, ast.Tuple) else [kinds]
            return [ast.unparse(k) for k in listed if k is not None]

        names = [name for handler in node.handlers for name in caught(handler)]
        if "Exception" in names and not any("CancelledError" in name for name in names):
            offenders.append(f"line {node.handlers[0].lineno}: {sorted(set(names))}")
    assert offenders == [], (
        "these boundaries run an env's or an extension's code and catch `Exception`, so a "
        "`CancelledError` it raises escapes a handler meant to contain everything: " + "; ".join(offenders)
    )
