"""The task stream (``hgym.serve.stream``): dispense a queue, seal and score each task, and
serve the whole thing over one MCP endpoint.

Driven against the real score-terminal fixture env — a full episode per dispensed task, with
the stream (never the caller) owning the seal.
"""

from __future__ import annotations

import ast
import asyncio
import errno
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pytest
from fastmcp import Client

import hgym.serve.stream as stream_module
from hgym.serve.lifecycle import FinalizeRequest, TerminalEvidence
from hgym.serve.stream import (
    QueueInfo,
    TaskRef,
    TaskStream,
    build_stream_server,
    read_dispenses,
    read_results,
    reconcile,
)
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from hgym.task import TaskSpec, ToolManifest
from hgym.types import EpisodeFeedback, FeedbackCollection, InferenceFeedback
from tests._fixtures.score_env import ENV_NAME, HORIZON, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [
    {"id": "q0", "question": "2+2?", "answer": "4"},
    {"id": "q1", "question": "3+3?", "answer": "6"},
    {"id": "q2", "question": "5+5?", "answer": "10"},
]


def _env_for(_name: str) -> _FixtureScoreEnv:
    """A FRESH env per episode — the stream's ownership contract."""
    return _FixtureScoreEnv(tasks=TASKS)


def _stream(tmp_path: Path, indices: List[int], **kwargs: Any) -> TaskStream:
    return TaskStream(
        _env_for,
        [TaskRef(ENV_NAME, i) for i in indices],
        prov_dir=tmp_path / "prov",
        **kwargs,
    )


def _rows(tmp_path: Path) -> List[Dict[str, Any]]:
    path = tmp_path / "prov" / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _published(feedback: Sequence[Dict[str, Any]], name: str) -> Any:
    """What a verbatim feedback record says under ``name`` at episode level — the same
    first-match lookup ``FeedbackCollection.get`` and ``EvalResult.value`` answer with."""
    return next(
        (
            item["value"]
            for item in feedback
            if item.get("level") == "episode" and item["name"] == name
        ),
        None,
    )


def _episode_item(name: str, value: Any) -> Dict[str, Any]:
    """One episode-level feedback item in wire form, as a row records it."""
    return {"name": name, "value": value, "level": "episode"}


def _terminal_text(result: Any) -> str:
    """The one text block a stream's tool response carries."""
    return result.content[0].text


async def _clean_terminal_response(tmp_path: Path) -> str:
    """What a caller gets back when a task ends and every part of ending it worked — the answer
    every *other* way of ending one has to be indistinguishable from.

    Taken from a real run rather than written out as a literal, so each comparison asserts "the
    same thing an ordinary seal returns" rather than "the string this test expected"."""
    async with _stream(tmp_path / "control", [0]) as stream:
        await stream.get_task()
        return _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))


async def _exhausted_response(tmp_path: Path) -> Dict[str, Any]:
    """What the endpoint answers ``get_task`` with once the queue really is empty — the answer a
    stopped stream has to be indistinguishable from, taken from a real run for the same reason
    :func:`_clean_terminal_response` is."""
    async with _stream(tmp_path / "exhausted", [0]) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            await client.call_tool("get_task", {})
            await client.call_tool(SUBMIT_TOOL, {"answer": "4"})
            out = await client.call_tool("get_task", {})
    return json.loads(out.content[0].text)  # type: ignore[union-attr]


async def _reads_as_exhausted(
    tmp_path: Path, *, consumed: int, in_flight: int = 0
) -> Dict[str, Any]:
    """The exact payload a stopped stream owes ``get_task``: what an exhausted queue answers,
    differing only in numbers the caller already holds — the count of tasks it played, and the
    count of its own episodes the stream still has open.

    Compared whole, values included. Key sets alone cannot see the failure this pins: a stopped
    stream relaying its live queue counts would answer ``done: true`` beside a non-zero
    ``remaining``, contradicting itself inside one object while every key stayed in place.

    ``in_flight`` is deliberately *not* part of what the redaction composes. It counts the
    caller's own open episodes, so a fixed zero is not a redaction but a false statement about
    the caller's own work — the one that makes a worker stop while a lease of its own is still
    callable, leaving the task it could have answered to be force-scored at the drain (see
    ``test_the_end_of_the_queue_is_not_told_as_nothing_being_live`` in the leases suite). It also
    hides nothing: a stop seals no episode by itself, so for every stop an env can *cause* — a
    summary the record cannot headline, an env that raises while ending a task, a drifted
    manifest — the row lands, the episode is released, and this number is the same one an
    exhausted queue reports. The exception is a stop the storage caused: a row that cannot be
    appended keeps its entry until the drain, and no env can make a full disk conditional on its
    own verdict."""
    return {
        **await _exhausted_response(tmp_path),
        "consumed": consumed,
        "in_flight": in_flight,
    }


class _TrackedEnv(_FixtureScoreEnv):
    """A fixture env whose release is observable, for the paths where nothing else holds it."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


def _tracked_factory(built: List[_TrackedEnv]) -> Any:
    """A factory that records every env it hands out, in build order: the catalog env first,
    then one per dispensed task."""

    def factory(_name: str) -> _TrackedEnv:
        env = _TrackedEnv(tasks=TASKS)
        built.append(env)
        return env

    return factory


async def test_dispensed_framing_is_redacted(tmp_path: Path) -> None:
    # The framing carries what the agent needs to act and nothing that identifies the task: no
    # index, no target, no lease. The env-static parts (instructions, tool schemas) legitimately
    # mention the word "answer"; what must be absent is the task's own gold value and index.
    async with _stream(tmp_path, [2]) as stream:
        task = await stream.get_task()
        assert task is not None
        assert set(task.to_wire()) == {"env", "instructions", "budget", "tools"}
        assert SUBMIT_TOOL in {t["name"] for t in task.tools}
        assert task.instructions == _FixtureScoreEnv(tasks=TASKS).describe().instructions
        assert "10" not in task.instructions  # TASKS[2]["answer"]


async def test_serves_a_queue_end_to_end(tmp_path: Path) -> None:
    # Three tasks, answered correctly, then exhaustion. One row per dispensed task, in order,
    # each scored off the sealed episode's own evidence.
    async with _stream(tmp_path, [0, 1, 2]) as stream:
        for expected in ("4", "6", "10"):
            task = await stream.get_task()
            assert task is not None
            result = await stream.dispatch(SUBMIT_TOOL, {"answer": expected})
            payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
            assert payload["terminated"] is True
            # Scored on the row the harness keeps — never handed back to whoever answered.
            row = stream.results[-1]
            assert row.closure == "sealed"
            assert row.score is not None
            assert _published(row.score.feedback, "correct") is True
        assert await stream.get_task() is None

    rows = _rows(tmp_path)
    assert [row["seq"] for row in rows] == [1, 2, 3]
    assert [row["task_idx"] for row in rows] == [0, 1, 2]
    assert all(row["score"]["success"] is True for row in rows)
    assert all(row["closure"] == "sealed" for row in rows)
    assert len({row["lease"] for row in rows}) == 3


async def test_a_terminating_call_returns_no_provenance_to_the_caller(tmp_path: Path) -> None:
    # The seal's row identifies the task (lease, queue position, index) and carries the env's
    # raw feedback, which is where a grader's internals and the target itself show up. A queue
    # may repeat an index, so either would let a caller recognise a task it has already played
    # and replay it against the scorer. What comes back over MCP is the env's own tool response
    # and the fact that the task ended — the row stays with the harness.
    class _LeakyFeedback(_FixtureScoreEnv):
        """An env whose terminal feedback names the target, as a grader's often does."""

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            if terminated:
                fb.episode.append(EpisodeFeedback(name="gold", value=task["answer"]))
            return fb

    stream = TaskStream(
        lambda _name: _LeakyFeedback(tasks=TASKS),
        [TaskRef(ENV_NAME, 2)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            await client.call_tool("get_task", {})
            out = await client.call_tool(SUBMIT_TOOL, {"answer": "not the answer"})
            seen = out.content[0].text  # type: ignore[union-attr]

    payload = json.loads(seen)
    assert payload["terminated"] is True
    assert set(payload) == {"content", "terminated", "hint"}
    for leaked in ("lease", "position", "task_idx", "seq", "closure", "feedback", "gold", "10"):
        assert leaked not in seen, f"{leaked!r} reached the caller: {seen}"

    # ...and the harness still has all of it, on the row and in the file.
    row = stream.results[0]
    assert row.task_idx == 2 and row.position == 0 and row.lease
    assert row.score is not None and _published(row.score.feedback, "gold") == "10"
    assert row.score.success is False
    assert _rows(tmp_path) == [row.to_wire()]


async def test_a_terminating_call_returns_no_verdict_to_the_caller(tmp_path: Path) -> None:
    # A single served episode surfaces its verdict and its episode feedback on the terminal
    # result, and that is safe there: the episode is over, so nothing the agent learns can reach
    # its own behaviour. A stream's terminal is mid-run. The queue below plays the SAME index
    # twice, so a verdict on the first would be the signal that identifies the second — answer
    # once, read `correct`, replay. The response is therefore a constant: right and wrong come
    # back byte-identical, and the reward the harness recorded is nowhere in either.
    class _Scored(_FixtureScoreEnv):
        """Grades with partial credit, so `reward` rides both the verdict and the feedback."""

        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            evidence = await super().finalize(req)
            evidence.verdict = {**evidence.verdict, "reward": 0.75}
            return evidence

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            if terminated and evidence is not None:
                fb.episode.append(
                    EpisodeFeedback(name="reward", value=evidence.verdict.get("reward", 0.0))
                )
            return fb

    stream = TaskStream(
        lambda _name: _Scored(tasks=TASKS),
        [TaskRef(ENV_NAME, 2), TaskRef(ENV_NAME, 2)],
        prov_dir=tmp_path / "prov",
    )
    seen: List[str] = []
    async with stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            for answer in ("not the answer", "10"):
                await client.call_tool("get_task", {})
                out = await client.call_tool(SUBMIT_TOOL, {"answer": answer})
                seen.append(out.content[0].text)  # type: ignore[union-attr]

    # Byte-identical is the whole property: no encoding of the outcome survives, in the content
    # or in which keys are present. The named fields are asserted too, so a future payload that
    # is merely constant-shaped but verdict-bearing still fails.
    wrong, right = seen
    assert wrong == right, f"the outcome is readable off the response: {wrong} vs {right}"
    for leaked in ("correct", "reward", "0.75", "verdict", "finalize_error"):
        assert leaked not in wrong, f"{leaked!r} reached the caller: {wrong}"
    assert json.loads(wrong)["terminated"] is True  # the caller still learns the task ended

    # The harness kept every bit of it, and the two attempts are distinguishable there.
    scores = [row.score for row in stream.results]
    assert all(score is not None for score in scores)
    assert [score.success for score in scores if score is not None] == [False, True]
    assert [score.reward for score in scores if score is not None] == [0.75, 0.75]
    assert [row.task_idx for row in stream.results] == [2, 2]


async def test_the_feedback_sidecar_never_rides_out_on_a_terminating_call(
    tmp_path: Path,
) -> None:
    # The other half of the same channel. A served episode carries its terminal episode feedback
    # out on the `_meta` sidecar; relaying it here — as the single-episode server deliberately
    # does — would hand back `correct` through the side door the redacted content just closed.
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        result = await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        assert result.meta is None
        # Not vacuous: the sealed episode really did produce terminal feedback to withhold.
        score = stream.results[-1].score
        assert score is not None and _published(score.feedback, "correct") is True


async def test_an_ordinary_call_returns_the_env_response_unchanged(tmp_path: Path) -> None:
    # Redaction stops at the terminal. A mid-episode response is the agent's observation and
    # only the env can produce it, so it passes through verbatim.
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        result = await stream.dispatch("noop", {})
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert set(payload) == {"content", "terminated"}
        assert payload["terminated"] is False
        assert json.loads(payload["content"]) == {"ok": True}  # the env's own tool response
        assert result.meta is None


async def test_reaching_the_horizon_is_redacted_like_a_submission(tmp_path: Path) -> None:
    # A terminal the agent never asked for. The budget-reaching call returns the env's terminal
    # payload — the verdict again, for a score env — so the boundary is the call that ends the
    # task, not the tool that was named.
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        for _ in range(HORIZON - 1):
            assert json.loads(
                (await stream.dispatch("noop", {})).content[0].text  # type: ignore[union-attr]
            )["terminated"] is False
        last = (await stream.dispatch("noop", {})).content[0].text  # type: ignore[union-attr]
    assert json.loads(last)["terminated"] is True
    assert "correct" not in last and "ok" not in last
    score = stream.results[-1].score  # the harness has the outcome
    assert score is not None and _published(score.feedback, "correct") is False


async def test_a_failing_tool_cannot_relay_env_text_to_the_caller(tmp_path: Path) -> None:
    # Same rule, the other way in: an env that cannot load a task says which one in the
    # exception, and MCP relays a tool exception verbatim. The caller learns the call failed.
    class _RefusesToLoad(_FixtureScoreEnv):
        def load_task(self, task_idx):  # type: ignore[override]
            raise RuntimeError(f"no such task {task_idx}; gold answers live in split heldout")

    stream = TaskStream(
        lambda _name: _RefusesToLoad(tasks=TASKS),
        [TaskRef(ENV_NAME, 2)],
        prov_dir=tmp_path / "prov",
    )
    try:
        server = build_stream_server(stream)
        async with Client(server) as client:
            out = await client.call_tool("get_task", {}, raise_on_error=False)
        assert out.is_error
        seen = out.content[0].text  # type: ignore[union-attr]
        for leaked in ("task 2", "gold", "heldout"):
            assert leaked not in seen, f"{leaked!r} reached the caller: {seen}"
    finally:
        await stream.aclose()


async def _play_one_task(stream: TaskStream, answer: str) -> Tuple[Any, Any]:
    """Play one task over the real endpoint: pull it, end it with ``answer``, then make the call
    an agent makes next. Returns both raw MCP results — the error flag included, because a
    response *shape* that varies with the outcome is as good a verdict as a response body that
    does, and ``mask_error_details`` hides the message but not the shape."""
    server = build_stream_server(stream)
    async with Client(server) as client:
        await client.call_tool("get_task", {})
        ended = await client.call_tool(SUBMIT_TOOL, {"answer": answer}, raise_on_error=False)
        following = await client.call_tool("get_task", {}, raise_on_error=False)
    return ended, following


def _observed(result: Any) -> Tuple[bool, str]:
    """Everything an agent can read off one tool call: whether it failed, and what it said."""
    return bool(result.is_error), result.content[0].text


async def test_a_malformed_summary_is_invisible_to_the_agent(tmp_path: Path) -> None:
    # The stream refuses to headline a wrong-typed summary and stops when it meets one. An env
    # can meet that condition on one verdict and not the other without meaning to — a grader that
    # stringifies its result on the path it thought was the happy one is enough. If the
    # terminating call then came back as an MCP *error* where a clean seal returns the fixed
    # payload, the agent would read its verdict off `is_error` without one byte of the response
    # differing, and the queue below plays the same index twice, so a verdict on the first
    # identifies the second. Masking the message does not touch this: the shape is the signal.
    class _MalformedWhenCorrect(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = FeedbackCollection()
            if not terminated:
                return fb
            correct = bool(evidence.verdict.get("correct")) if evidence is not None else False
            # A string on the correct path, a real bool on the other one.
            value: Any = "true" if correct else False
            fb.episode.append(EpisodeFeedback(name="correct", value=value))
            return fb

    def _run(label: str) -> TaskStream:
        return TaskStream(
            lambda _name: _MalformedWhenCorrect(tasks=TASKS),
            [TaskRef(ENV_NAME, 2), TaskRef(ENV_NAME, 2)],
            prov_dir=tmp_path / label / "prov",
        )

    wrong, right = _run("wrong"), _run("right")
    wrong_end, wrong_next = await _play_one_task(wrong, "not the answer")
    right_end, right_next = await _play_one_task(right, "10")

    # Ending the task looks the same either way, error flag and all.
    assert _observed(wrong_end) == _observed(right_end), "the outcome is readable off the call"
    assert _observed(wrong_end)[0] is False
    assert json.loads(_observed(wrong_end)[1])["terminated"] is True
    # ...and so does the call the agent makes next, which is where the same leak would surface
    # one call later: a stopped stream is the end of the queue, not an error.
    assert _observed(right_next)[0] is False and _observed(wrong_next)[0] is False
    ended = json.loads(_observed(right_next)[1])
    assert ended == await _reads_as_exhausted(tmp_path, consumed=1)

    # The harness is told everything, on the stream and out of the drain.
    assert right.stopped and not wrong.stopped
    with pytest.raises(RuntimeError, match="cannot headline.*'correct' must be true or false"):
        await right.aclose()
    (row,) = _rows(tmp_path / "right")
    assert row["score"] is None
    assert row["observed"] == [_episode_item("correct", "true")], "the evidence must be durable"
    assert "cannot headline" in (row["diagnostic"] or "")
    await wrong.aclose()
    assert [r["score"]["success"] for r in _rows(tmp_path / "wrong")] == [False, False]


async def test_an_env_that_raises_while_ending_a_task_is_redacted_and_stops_the_run(
    tmp_path: Path,
) -> None:
    # The same leak with a wider blast radius, and the one that does not stop the run on its own.
    # An episode commits its terminal and *then* serializes the feedback it is about to hand
    # over, so a value the wire refuses — a non-finite number is the reachable one — raises from
    # a call that has already ended the task. Unhandled that is three failures at once: the agent
    # reads the verdict off `is_error`; the stream carries on and dispenses the repeat it just
    # identified; and the row lands with no feedback at all, so a solved task is recorded
    # unscored while the drain reports a clean run.
    class _UnserializableWhenCorrect(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            if terminated and evidence is not None and evidence.verdict.get("correct"):
                fb.episode.append(EpisodeFeedback(name="reward", value=float("nan")))
            return fb

    def _run(label: str) -> TaskStream:
        return TaskStream(
            lambda _name: _UnserializableWhenCorrect(tasks=TASKS),
            [TaskRef(ENV_NAME, 2), TaskRef(ENV_NAME, 2)],
            prov_dir=tmp_path / label / "prov",
        )

    wrong, right = _run("wrong"), _run("right")
    wrong_end, _ = await _play_one_task(wrong, "not the answer")
    right_end, right_next = await _play_one_task(right, "10")

    assert _observed(wrong_end) == _observed(right_end), "the outcome is readable off the call"
    assert _observed(wrong_end)[0] is False
    # The run is over rather than continuing into the repeat.
    ended = json.loads(_observed(right_next)[1])
    assert ended == await _reads_as_exhausted(tmp_path, consumed=1)

    # A task was still dispensed, so exactly one row is owed and lands — unscored, with the
    # feedback that never serialized absent from it, which is what makes the loss legible.
    (row,) = _rows(tmp_path / "right")
    assert row["observed"] == [], "feedback that never serialized cannot be on the row"
    assert row["score"] == {"reward": None, "success": None, "feedback": []}
    assert right.stopped, "a run that lost an outcome reported itself complete"
    with pytest.raises(RuntimeError, match="raised while ending a task.*must be finite") as end:
        await right.aclose()
    assert isinstance(end.value.__cause__, ValueError)
    await wrong.aclose()
    assert [r["score"]["success"] for r in _rows(tmp_path / "wrong")] == [False, False]


class _UnrenderableError(RuntimeError):
    """An env failure whose own message is a second failure.

    Not exotic on purpose: a message built lazily from state the failure has already torn down
    raises exactly here, when the stream asks for it rather than when the env raised."""

    def __str__(self) -> str:
        raise RuntimeError("formatting this failure failed")


def _fails_the_terminal(exc: Callable[[], BaseException]) -> Callable[[str], _FixtureScoreEnv]:
    """A non-seal env — `verify` runs inline, past the terminal step — that fails the call that
    ends the task."""

    class _Failing(_FixtureScoreEnv):
        score_terminal_tool = None

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if terminated:
                raise exc()
            return FeedbackCollection()

    return lambda _name: _Failing(tasks=TASKS)


@pytest.mark.parametrize("failure", [RuntimeError, _UnrenderableError])
async def test_an_env_failure_is_answered_the_same_way_whether_or_not_it_formats(
    tmp_path: Path, failure: Callable[[], BaseException]
) -> None:
    # The stop above is built by *formatting* the failure it is stopping for, and formatting an
    # exception runs the env's code a second time, outside the `except` that just caught it.
    # Unguarded that costs three invariants at once, and unlike a malformed summary it is
    # silent: the message is an argument to `_stop`, so the stop never happens and the queue is
    # served on against an env that fails every task in it; `aclose()` reports a clean run
    # having lost the outcome; and the exception the *formatter* raised goes back to the agent
    # in place of the one constant every ending answers with — on exactly the call the redaction
    # exists for.
    #
    # Parametrized against a plain failure so the delta is the formatter and nothing else.
    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        _fails_the_terminal(failure),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()
    assert _terminal_text(await stream.dispatch(TERMINATE_TOOL_NAME, {})) == clean
    assert stream.stopped, "the stop its own message described was never published"
    assert len(_rows(tmp_path)) == 1, "a dispensed task still owes exactly one row"
    with pytest.raises(RuntimeError, match="raised while ending a task") as end:
        await stream.aclose()
    # The class is still named, which is the part of a failure that never needed the env's help.
    assert isinstance(end.value.__cause__, failure), "the failure itself must be kept"
    assert failure.__name__ in str(end.value)


async def test_a_stopped_stream_reports_to_the_harness_and_not_to_the_agent(
    tmp_path: Path,
) -> None:
    # The two surfaces are deliberately asymmetric. A harness driving the stream directly gets
    # the failure raised at it, with the original cause chained; an agent driving the same object
    # over MCP gets the end of the queue. The refusal text names the provenance path — harness
    # territory — which the agent's answer structurally cannot carry.
    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _tracked_factory(built),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    _unwritable_prov(tmp_path)
    await stream.get_task()
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert stream.stopped

    with pytest.raises(RuntimeError, match="could not be recorded") as refused:
        await stream.get_task()  # the direct API is unchanged
    assert isinstance(refused.value.__cause__, OSError)
    assert str(tmp_path) in str(refused.value)

    server = build_stream_server(stream)
    async with Client(server) as client:
        out = await client.call_tool("get_task", {}, raise_on_error=False)
    assert not out.is_error
    seen = out.content[0].text  # type: ignore[union-attr]
    ended = json.loads(seen)
    # The row this stream could not write left its entry to be retried by the drain, so one
    # episode is still open here. That count is reported rather than composed: it is the caller's
    # own work, and it is the number `queue_info` answers this same caller with anyway.
    assert ended == await _reads_as_exhausted(tmp_path, consumed=1, in_flight=1)
    assert ended["in_flight"] == stream.queue_info().in_flight
    assert str(tmp_path) not in seen and "record" not in seen
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()


async def test_an_ordinary_call_that_fails_is_still_the_env_s_own_answer(
    tmp_path: Path,
) -> None:
    # The boundary is the call that ends the task, not failure in general. A mid-episode call
    # that raises has ended nothing: the task is still live, the agent will call again, and the
    # exception is the env's own answer — no different in kind from the env text an ordinary
    # response returns verbatim, which the serving layer already cannot police. Redacting it
    # would only hide a broken env from the person running it.
    class _RaisesMidEpisode(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if not terminated:
                raise RuntimeError("the env fell over mid-episode")
            return super()._verify(trajectory, task, terminated=terminated, evidence=evidence)

    stream = TaskStream(
        lambda _name: _RaisesMidEpisode(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        await stream.get_task()
        with pytest.raises(RuntimeError, match="fell over mid-episode"):
            await stream.dispatch("noop", {})
        assert stream.queue_info().in_flight == 1, "the task ended when nothing ended it"
        assert not stream.stopped
        # ...and the task can still be finished, which is the whole reason it is not redacted.
        assert json.loads(
            _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        )["terminated"] is True
    assert [row["score"]["success"] for row in _rows(tmp_path)] == [True]


class _VaryingManifest(_FixtureScoreEnv):
    """Publishes a different manifest for task 1 than for task 0, so construction must raise —
    the shortest way to a constructor that fails with catalog envs already built."""

    def describe(self, task_id=None) -> TaskSpec:
        spec = super().describe(task_id)
        if task_id == "1":
            spec.tools = [
                *spec.tools,
                ToolManifest(
                    name="extra",
                    description="d",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]
        return spec


async def test_a_failed_construction_closes_the_catalog_envs(tmp_path: Path) -> None:
    # A constructor that raises hands back no stream, so nothing else can ever close the envs it
    # built — and the factory may have provisioned real resources building them.
    #
    # Where that close runs is part of the fix, not an implementation detail: `close()` is a
    # coroutine and the env contract says nothing about loop affinity, so an env built inside a
    # running loop may hold objects belonging to it. It is closed on the loop that built it and
    # on no other. A synchronous constructor cannot await that, so it is scheduled there and
    # completes at the caller's next suspension — which is what the sleep below is.
    class _Observable(_VaryingManifest):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.closed = False
            self.closed_on: Any = None

        async def close(self) -> None:
            self.closed = True
            self.closed_on = asyncio.get_running_loop()
            await super().close()

    built: List[_Observable] = []

    def factory(_name: str) -> _Observable:
        env = _Observable(tasks=TASKS)
        built.append(env)
        return env

    with pytest.raises(ValueError, match="different tool manifest") as raised:
        TaskStream(
            factory,
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
        )
    # Nothing about the cleanup is swallowed: what is still in flight is on the error itself.
    assert any("closed on the loop that built it" in note for note in raised.value.__notes__)
    await asyncio.sleep(0)
    assert built and all(env.closed for env in built)
    assert all(env.closed_on is asyncio.get_running_loop() for env in built)


def test_a_failed_construction_outside_a_loop_closes_the_catalog_envs_there(
    tmp_path: Path,
) -> None:
    # The other half of the same contract, and the reason it is not simply "schedule it": with
    # no loop running there is nothing to conflict with, so the close is complete — and finished
    # — before the caller ever sees the error.
    class _Observable(_VaryingManifest):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.closed = False

        async def close(self) -> None:
            self.closed = True
            await super().close()

    built: List[_Observable] = []

    def factory(_name: str) -> _Observable:
        env = _Observable(tasks=TASKS)
        built.append(env)
        return env

    with pytest.raises(ValueError, match="different tool manifest") as raised:
        TaskStream(
            factory,
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
        )
    assert built and all(env.closed for env in built)
    assert not getattr(raised.value, "__notes__", [])


def test_a_failed_construction_lets_every_catalog_env_go_independently(
    tmp_path: Path,
) -> None:
    # `Env.close()` is third-party code that may block for as long as it likes — an MCP session
    # that never answers, a subprocess that will not reap. Closed one at a time, the first such
    # env decided whether any env after it was closed at all: they stayed open, holding whatever
    # they hold, for exactly as long as it hung, with the constructor already committed to
    # raising and nothing else left holding them. So every close is started before any of them is
    # waited for.
    #
    # The first env below waits for the second one to *start* closing, which serially is a wait
    # that can only time out. It is not a bound on teardown — there is none — it is how a test
    # asks "did the second one get to start" without hanging when the answer is no.
    gate: Dict[str, Any] = {}

    class _WaitsForItsSibling(_FixtureScoreEnv):
        def __init__(self, key: str, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.key = key
            self.saw_sibling: Any = None
            self.closed = False

        async def close(self) -> None:
            started = gate.setdefault("started", asyncio.Event())
            if self.key == "first":
                try:
                    await asyncio.wait_for(started.wait(), 5.0)
                    self.saw_sibling = True
                except (asyncio.TimeoutError, TimeoutError):
                    self.saw_sibling = False
            else:
                started.set()
            self.closed = True
            await super().close()

    built: List[_WaitsForItsSibling] = []

    def factory(_name: str) -> _WaitsForItsSibling:
        env = _WaitsForItsSibling(
            key="first" if not built else "second", tasks=TASKS
        )
        built.append(env)
        return env

    # Refused after the catalog is built, so both envs exist and both are this cleanup's to
    # release: a stored dispense naming a task this queue does not hold at that position.
    (tmp_path / "prov").mkdir(parents=True)
    (tmp_path / "prov" / "dispenses.jsonl").write_text(
        json.dumps({"lease": "l", "seq": 1, "position": 1, "env": "a", "task_idx": 0}) + "\n"
    )
    with pytest.raises(ValueError, match="holds a dispense record"):
        TaskStream(
            factory,
            [TaskRef("a", 0), TaskRef("b", 0)],
            prov_dir=tmp_path / "prov",
            resume=True,
        )
    assert len(built) == 2 and all(env.closed for env in built)
    assert built[0].saw_sibling is True, (
        "the second env's close did not start until the first one had finished"
    )


async def test_a_catalog_env_is_never_closed_on_a_foreign_loop(tmp_path: Path) -> None:
    # The failure this replaces: closing on a private worker loop. An env whose resources belong
    # to the loop that built it then fails to close (a future attached to a different loop), or
    # deadlocks the constructor, which is blocking the very loop the close is waiting on. The
    # env contract permits such an env — `close()` is a coroutine and the factory is explicitly
    # allowed to provision resources — so the serving layer may not assume otherwise.
    owner = asyncio.get_running_loop()
    loops: List[Any] = []

    class _LoopBound(_VaryingManifest):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self.loop = asyncio.get_running_loop()

        async def close(self) -> None:
            running = asyncio.get_running_loop()
            loops.append(running)
            if running is not self.loop:
                raise RuntimeError("closed on a loop that does not own this env's resources")
            await super().close()

    with pytest.raises(ValueError, match="different tool manifest"):
        TaskStream(
            lambda _name: _LoopBound(tasks=TASKS),
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
        )
    await asyncio.sleep(0)
    assert loops and all(loop is owner for loop in loops)


def _unwritable_prov(tmp_path: Path) -> None:
    """Make the real append fail, without reaching inside the stream: a *directory* where the
    results file goes, so `open("a")` raises where a full or read-only volume would."""
    (tmp_path / "prov" / "results.jsonl").mkdir(parents=True)


async def test_a_task_whose_row_failed_is_no_longer_the_live_episode(tmp_path: Path) -> None:
    # With one slot there is no lease to name a task with, so a call resolves to whichever entry
    # is not sealed — and a seal that failed on the storage hands its claim back, which is what
    # that read was standing on. The task the stream has already force-terminated therefore
    # becomes the task every later call is routed to, and the agent is answered as though its
    # call had ended something. What a call after a finished task gets is `no_active_task`,
    # whether or not the row could be written.
    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _tracked_factory(built), [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    _unwritable_prov(tmp_path)
    await stream.get_task()
    live = next(iter(stream._live.values()))  # noqa: SLF001 - inspecting the routing
    reached: List[str] = []
    original = live.episode.call

    async def watched(tool: str, arguments: Any = None) -> Any:
        reached.append(tool)
        return await original(tool, arguments)

    live.episode.call = watched  # type: ignore[method-assign]
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert stream.stopped
    after_terminal = list(reached)

    late = json.loads(_terminal_text(await stream.dispatch("noop", {})))
    assert late.get("error") == "no_active_task", "a task the stream had ended was still live"
    assert reached == after_terminal, "a late call reached an episode the stream had ended"
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()
    assert stream.results == ()


async def test_a_failed_row_write_still_releases_the_episode(tmp_path: Path) -> None:
    # The row and the episode are independent: the episode owns MCP sessions and an env, and its
    # `_Live` entry is the only handle on them. A seal whose write fails hands its claim back so
    # the drain can retry it, so the entry outlives the failed call on purpose — but the drain is
    # the last chance, and a write still failing there must not take the release with it.
    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _tracked_factory(built), [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov"
    )
    clean = await _clean_terminal_response(tmp_path)
    _unwritable_prov(tmp_path)
    await stream.get_task()
    assert stream.queue_info().in_flight == 1
    # The caller is told the task ended and nothing else: a failure that reached it as an
    # exception would be a response shape that varies with what the seal found (see
    # `test_a_malformed_summary_is_invisible_to_the_agent`).
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean
    assert stream.queue_info().in_flight == 1  # still sealable: the claim was handed back

    # The drain reports the row the run never got — with the write failure itself still on the
    # chain, which is where it went instead of to the caller.
    assert stream.stopped
    with pytest.raises(RuntimeError, match="record is incomplete") as raised:
        await stream.aclose()
    assert isinstance(raised.value.__cause__, OSError)
    assert stream.queue_info().in_flight == 0  # the episode is released; only a row is owed
    assert built[1].closed, "the episode's env outlived the failed write"
    assert all(env.closed for env in built)


async def test_a_row_counts_as_recorded_only_once_it_is_durable(tmp_path: Path) -> None:
    # `results` is what a harness scores the run off. Publishing a row there that never reached
    # the file would show a complete set of outcomes over a record silently one short — the
    # failure mode that matters here is not the leak, it is being believed.
    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _tracked_factory(built),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    clean = await _clean_terminal_response(tmp_path)
    _unwritable_prov(tmp_path)
    await stream.get_task()
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean
    assert stream.results == ()

    # ...and the stream stops rather than serving the rest of the queue over a holed record.
    assert stream.stopped
    with pytest.raises(RuntimeError, match="could not be recorded") as refused:
        await stream.get_task()
    assert isinstance(refused.value.__cause__, OSError)
    assert stream.queue_info().consumed == 1
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()


def _fs_key(path: Path) -> Tuple[int, int]:
    """Identify a file or directory the way an fd does, so an `os.fsync(fd)` can be attributed
    to it without reaching inside the code under test."""
    info = path.stat()
    return (info.st_dev, info.st_ino)


def _unsyncable_results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the real sync of the results file fail — the counterpart of `_unwritable_prov`, and
    scoped the same way: only that file, so everything else the stream persists still works."""
    results_path = tmp_path / "prov" / "results.jsonl"
    real_fsync = os.fsync

    def _fsync(fd: int) -> None:
        info = os.fstat(fd)
        if results_path.exists() and (info.st_dev, info.st_ino) == _fs_key(results_path):
            raise OSError(errno.EIO, "Input/output error")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _fsync)


async def test_a_row_is_synced_before_it_is_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Closing the file hands the bytes to the kernel and no further: it survives this process
    # dying, not the host. The row is answered to the agent and published on `results` the
    # moment the append returns, so the append is where the record has to become durable.
    real_fsync = os.fsync
    synced: List[Dict[str, Any]] = []
    prov = tmp_path / "prov"
    results_path = prov / "results.jsonl"

    def _watch(fd: int) -> None:
        info = os.fstat(fd)
        # An inode identifies the log only while the log is holding it. A filesystem that
        # recycles inodes hands this same number to a short-lived file first, so a sync from
        # before the log existed would otherwise be attributed to it and read back empty.
        existed = results_path.exists()
        synced.append(
            {
                "key": (info.st_dev, info.st_ino),
                "existed": existed,
                "published": len(stream.results),
                "on_disk": results_path.read_text() if existed else "",
            }
        )
        real_fsync(fd)

    stream = _stream(tmp_path, [0])
    monkeypatch.setattr(os, "fsync", _watch)
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    rows = [
        entry
        for entry in synced
        if entry["existed"] and entry["key"] == _fs_key(results_path)
    ]
    assert rows, "the row's file was never synced — only handed to the page cache"
    # Synced with the row already in it, and before `results` could report the outcome.
    assert '"task_idx": 0' in rows[0]["on_disk"]
    assert rows[0]["published"] == 0
    # The first row also syncs the directory: an unsynced entry can lose the whole new file.
    assert _fs_key(prov) in [entry["key"] for entry in synced]
    assert len(stream.results) == 1 and len(_rows(tmp_path)) == 1


async def test_a_provenance_directory_the_run_created_is_synced_into_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Syncing the provenance directory persists the entries *inside* it. The entry that names
    # the directory itself lives one level up, so a first run against a fresh output path can
    # lose the entire record — rows, file and directory — with every write having returned
    # successfully. Every level on the path must be synced into the level above it.
    #
    # Every level, not only the ones this run created: `mkdir` makes a level visible immediately
    # and durable never, so a directory that already existed says nothing about whether anyone
    # ever published its entry. A harness that made the output path a moment earlier, or a
    # writer that made it and died before its own sync, leaves a run that syncs nothing at all
    # and reports every write a success over a record a crash can still take whole.
    real_fsync = os.fsync
    synced: List[Tuple[int, int]] = []

    def _watch(fd: int) -> None:
        info = os.fstat(fd)
        synced.append((info.st_dev, info.st_ino))
        real_fsync(fd)

    # Two levels that do not exist yet, under one that does — and one run whose whole path was
    # made by somebody else before it started.
    run_dir = tmp_path / "runs" / "run-1"
    prov = run_dir / "prov"
    handed_over = tmp_path / "handed" / "over" / "prov"
    handed_over.mkdir(parents=True)  # created by another writer; nothing says it was synced
    stream = TaskStream(_env_for, [TaskRef(ENV_NAME, 0)], prov_dir=prov)
    monkeypatch.setattr(os, "fsync", _watch)
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    for made in (prov, run_dir):
        assert _fs_key(made.parent) in synced, (
            f"{made.name} is on this run's path, so the entry naming it — which lives in "
            f"{made.parent.name} — is what a crash can still lose"
        )

    synced.clear()
    second = TaskStream(_env_for, [TaskRef(ENV_NAME, 1)], prov_dir=handed_over)
    async with second:
        await second.get_task()
        await second.dispatch(SUBMIT_TOOL, {"answer": "6"})
    for level in (handed_over, handed_over.parent):
        assert _fs_key(level.parent) in synced, (
            f"{level.name} existed before this run, which is not evidence anyone published it"
        )
    assert (prov / "results.jsonl").exists() and len(stream.results) == 1
    assert (handed_over / "results.jsonl").exists() and len(second.results) == 1


async def test_a_row_is_synced_before_the_byte_that_commits_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row that survives the crash may not take the rest of the file with it. Written together,
    # a row and its terminator are one `write` — a row carrying a lot of published feedback
    # still is — and a crash can persist the block holding the newline while losing one in the
    # middle, so a torn write reads back as a whole record. Syncing the row first makes the
    # terminator mean something: a terminated line was durable before it, and the only thing a
    # reader has to leave out is an unterminated last one, which is a write that never returned.
    # That is what keeps one torn record at the end of a line-delimited file from taking every
    # intact row before it down with it.
    real_fsync = os.fsync
    ends: List[bytes] = []
    results_path = tmp_path / "prov" / "results.jsonl"

    def _watch(fd: int) -> None:
        real_fsync(fd)
        if results_path.is_file():
            ends.append(results_path.read_bytes()[-1:])

    stream = _stream(tmp_path, [0])
    monkeypatch.setattr(os, "fsync", _watch)
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    written = [end for end in ends if end]
    assert written, "the row's file was never synced"
    assert written[0] != b"\n", "the row and its terminator were made durable together"
    assert written[-1] == b"\n", "the terminator that commits the row was never synced"
    assert len(_rows(tmp_path)) == 1


async def test_a_row_that_cannot_be_synced_is_a_failed_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A row the volume refuses to sync is a row the run did not record, and is treated exactly
    # like a row it refused to write: nothing published, the episode released anyway, and the
    # stream stopped rather than serving the rest of the queue over a holed record.
    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _tracked_factory(built),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    clean = await _clean_terminal_response(tmp_path)
    _unsyncable_results(tmp_path, monkeypatch)
    await stream.get_task()
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean

    assert stream.results == ()
    assert stream.queue_info().in_flight == 1  # still sealable: the claim was handed back
    with pytest.raises(RuntimeError, match="could not be recorded"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()
    assert stream.queue_info().in_flight == 0
    assert built[1].closed, "the episode's env outlived the failed sync"
    assert stream.stopped
    with pytest.raises(RuntimeError, match="could not be recorded") as refused:
        await stream.get_task()
    assert isinstance(refused.value.__cause__, OSError)
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()


async def test_a_failed_drain_still_releases_every_env(tmp_path: Path) -> None:
    # The same order-of-operations one level up: `aclose` sets `_closed` on the way in, so a
    # seal that raises mid-drain would skip the catalog release *and* every entry after it,
    # with a second call returning immediately and no way left to repair either.
    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _tracked_factory(built), [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov"
    )
    await stream.get_task()  # live and unsealed: the drain is what will seal it
    _unwritable_prov(tmp_path)

    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()
    assert len(built) == 2 and all(env.closed for env in built)
    assert stream.queue_info().in_flight == 0


async def test_wrong_answer_scores_zero_not_an_error(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "nope"})
    row = _rows(tmp_path)[0]
    assert row["closure"] == "sealed" and row["score"]["success"] is False


def _feedback_env(
    items: List[Tuple[str, str, Any]],
) -> Callable[[str], _FixtureScoreEnv]:
    """A factory whose episodes publish exactly ``items`` — ``(level, name, value)`` each — at
    the terminal, in the order given. The collection is built by appending, which is how every
    env in this repo builds one and is the reason a name can appear twice."""

    class _Published(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = FeedbackCollection()
            if terminated:
                for level, name, value in items:
                    if level == "episode":
                        fb.episode.append(EpisodeFeedback(name=name, value=value))
                    else:
                        fb.inference.append(
                            InferenceFeedback(name=name, value=value, step=1)
                        )
            return fb

    return lambda _name: _Published(tasks=TASKS)


def _summary_env(name: str, value: Any) -> Callable[[str], _FixtureScoreEnv]:
    """A factory whose episodes publish exactly one feedback item, ``name`` = ``value``."""
    return _feedback_env([("episode", name, value)])


@pytest.mark.parametrize(
    "name, value",
    [
        ("success", "false"),  # every non-empty string is truthy, whatever it says
        ("correct", "False"),
        ("correct", 0.25),  # a partial-credit number published under a boolean name
    ],
)
async def test_a_wrong_typed_success_is_neither_coerced_nor_dropped(
    tmp_path: Path, name: str, value: Any
) -> None:
    # `EpisodeFeedbackValue` is `float | bool | str` and the wire validator permits all three,
    # so an env can legally publish any of them under a summary name. `bool(...)` turns every
    # one of these into `True`: the env's own "not solved" recorded as solved, beside the raw
    # feedback that says otherwise. That is how a record comes to overstate a benchmark.
    #
    # Dropping it to a quiet `None` is not the fix either — it would read as "this env publishes
    # no success field" and count as a failure. So the row lands, carrying the offending value
    # verbatim as durable evidence, and then the stream stops: the type an env gives a field
    # belongs to the env, so the rest of the queue would be scored the same unusable way.
    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        _summary_env(name, value), [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov"
    )
    await stream.get_task()
    # Loud to the harness, silent to the agent: the caller is told the task ended, in the same
    # bytes an ordinary seal answers with, because whether the summary was readable is a fact
    # about the outcome and an exception here would be a response shape carrying it.
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "nope"})) == clean

    (row,) = _rows(tmp_path)
    assert row["score"] is None, f"{value!r} was coerced into a verdict"
    assert row["observed"] == [
        _episode_item(name, value)
    ], "the env's own output must survive verbatim"
    assert "cannot headline" in (row["diagnostic"] or ""), "the file must say why it is unscored"
    assert stream.results[0].to_wire() == row
    assert stream.queue_info().in_flight == 0, "the episode outlived its recorded row"
    assert stream.stopped
    with pytest.raises(RuntimeError, match=f"{name!r} must be true or false"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match=f"cannot headline.*{name!r} must be true or false"):
        await stream.aclose()


async def test_a_wrong_typed_reward_is_not_silently_dropped(tmp_path: Path) -> None:
    # The mirror image, and the reason the answer is not merely "accept a bool, else write
    # `None`": reading `reward` was already strict but *silent*, so a string reward became a
    # `None` indistinguishable from an env that publishes no reward — a broken env reading as a
    # run of unscored tasks. Both summary fields now fail the same way, loudly.
    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        _summary_env("reward", "0.75"), [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov"
    )
    await stream.get_task()
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean

    (row,) = _rows(tmp_path)
    assert row["score"] is None and row["observed"] == [_episode_item("reward", "0.75")]
    assert stream.stopped
    with pytest.raises(RuntimeError, match="cannot headline.*'reward' must be a number"):
        await stream.aclose()


async def test_an_env_that_publishes_no_summary_records_none_and_carries_on(
    tmp_path: Path,
) -> None:
    # The other half of the contract, and what makes `None` worth reading: it means "the env
    # published no such field" and nothing else, precisely because a malformed value stops the
    # stream rather than landing here. Text feedback under a name that is not a summary name is
    # ordinary — the strictness is about the two headline fields, not about feedback at large.
    async with TaskStream(
        _summary_env("notes", "graded offline"),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    ) as stream:
        for _ in range(2):
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    rows = _rows(tmp_path)
    assert [(r["score"]["success"], r["score"]["reward"]) for r in rows] == [(None, None)] * 2
    assert all(
        r["observed"] == [_episode_item("notes", "graded offline")] for r in rows
    )
    assert all(r["closure"] == "sealed" and r["diagnostic"] is None for r in rows)


@pytest.mark.parametrize(
    "name, first, second",
    [
        # The wrong-typed value published first and a well-formed one second. Flattened by
        # name, the second overwrote the first: the row read as an honest solve and the stop
        # above never fired — the strictness was reachable only when the malformed value
        # happened to be last.
        ("success", "false", True),
        # Both well-formed, so no type check can see this one at all. Whichever the record
        # keeps is a verdict the env did not unambiguously give it.
        ("correct", True, False),
        ("reward", 1.0, 0.0),
    ],
)
async def test_a_summary_name_published_twice_is_refused_not_resolved(
    tmp_path: Path, name: str, first: Any, second: Any
) -> None:
    # The feedback wire is an ordered *list* and nothing upstream enforces one item per name:
    # `FeedbackCollection` validates neither, and every env in this repo builds its collection
    # by appending. So "last one wins" is a rule the record would be inventing — and inventing
    # against the package, since `FeedbackCollection.get` and `EvalResult.value` both answer
    # with the first match.
    #
    # There is no silent answer that is right. Keeping either occurrence decides the benchmark
    # headline by list order and erases the other from the row that is supposed to be evidence.
    # So it is refused exactly like a wrong-typed value: the row lands first — unscored, with a
    # diagnostic, carrying both occurrences verbatim so the ambiguity is legible in the file —
    # and then the stream stops.
    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        _feedback_env([("episode", name, first), ("episode", name, second)]),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean

    (row,) = _rows(tmp_path)
    assert row["score"] is None, "an ambiguous headline was taken"
    assert row["observed"] == [
        _episode_item(name, first),
        _episode_item(name, second),
    ], "an occurrence the stream refused to trust was dropped from the record"
    assert "cannot headline" in (row["diagnostic"] or ""), "the file must say why it is unscored"
    assert row["closure"] == "sealed", "how the task ended is a separate question"
    assert stream.results[0].to_wire() == row
    assert stream.queue_info().in_flight == 0, "the episode outlived its recorded row"
    assert stream.stopped
    with pytest.raises(RuntimeError, match=f"{name!r} was published 2 times"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match=f"cannot headline.*{name!r} was published 2 times"):
        await stream.aclose()


class _Undescribable(str):
    """A feedback value whose own description is a second failure.

    A JSON scalar as far as every boundary between the env and the row is concerned — it
    serializes, and the wire's `isinstance` checks admit it — but asking what it *is* raises."""

    def __repr__(self) -> str:
        raise RuntimeError("repr exploded")


def _publishes_undescribable(items: List[Tuple[str, Any]]) -> Callable[[str], _FixtureScoreEnv]:
    """A factory whose episodes publish ``(name, value)`` at episode level, with each value
    swapped in *after* the model was constructed.

    The assignment is the point and it is not a contrivance: pydantic coerces a `str` subclass
    back to a plain `str` at construction, so a value published the ordinary way never reaches
    the record as a subclass. The models are mutable and do not validate on assignment — the
    hole `dump_item`'s own docstring names — and `dump_item` then carries the value through
    verbatim, because the wire dict is built from `item.value` and it validates a *copy*."""

    class _Published(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = FeedbackCollection()
            if terminated:
                for name, value in items:
                    item = EpisodeFeedback(name=name, value="placeholder")
                    item.value = value
                    fb.episode.append(item)
            return fb

    return lambda _name: _Published(tasks=TASKS)


@pytest.mark.parametrize(
    "items, refusal",
    [
        # Wrong-typed: the refusal names what the value must be, and the value is the decoration.
        ([("success", _Undescribable("false"))], "'success' must be true or false"),
        # Duplicated: the refusal is about the count, which no env code is needed to know.
        (
            [("success", _Undescribable("x")), ("success", True)],
            "'success' was published 2 times",
        ),
    ],
)
async def test_a_summary_value_that_cannot_be_described_still_fails_loudly(
    tmp_path: Path, items: List[Tuple[str, Any]], refusal: str
) -> None:
    # `_pick_summary` refuses a summary value by building a message *about* it, and building it
    # calls the env's own `__repr__`. Unguarded, the value decides whether its own refusal
    # happens: the raise escaped the `_MalformedSummary` handler, so the row was never composed,
    # no evidence reached the file, and the run reported "a dispensed task could not be recorded"
    # — a storage failure that never happened, sending an operator to the provenance volume for
    # something that never touched it.
    #
    # The refusal outranks its own decoration. The row lands first with the value verbatim, the
    # stream stops loudly, and the description degrades to the failure that asking raised.
    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        _publishes_undescribable(items),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()
    # Silent to the agent, exactly as a describable malformed summary is: the answer may not
    # vary with what the env published, and least of all with whether describing it worked.
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "nope"})) == clean

    (row,) = _rows(tmp_path)
    assert row["score"] is None, "an unreadable value was headlined"
    assert row["observed"] == [
        _episode_item(name, json.loads(json.dumps(value))) for name, value in items
    ], "the durable evidence must survive a value the record cannot describe"
    assert "cannot headline" in (row["diagnostic"] or ""), "the file must say why it is unscored"
    assert stream.results[0].to_wire() == row, "the durable row and the in-memory one must agree"
    assert stream.queue_info().in_flight == 0, "the episode outlived its recorded row"
    assert stream.stopped, "the stop the refusal owes was lost to the refusal's own message"
    with pytest.raises(RuntimeError, match=refusal):
        await stream.get_task()
    with pytest.raises(RuntimeError, match=f"cannot headline.*{refusal}") as end:
        await stream.aclose()
    # The env's own failure is named rather than swallowed, so the operator is pointed at the
    # env that published the value and not at the filesystem.
    assert "RuntimeError: repr exploded" in str(end.value)


async def test_a_feedback_name_that_cannot_be_compared_still_lands_the_row_it_earned(
    tmp_path: Path,
) -> None:
    # The other half of the funnel, and the one that was left open. Finding a summary item
    # *matches names*, and a name is an object the env supplied — the same post-construction
    # `str` subclass the wire admits above, here with a raising `__eq__` instead of a raising
    # `__repr__`. That raise is not a `_MalformedSummary`, so it escaped the one handler around
    # the summary and took the whole seal with it: no row was ever composed, nothing reached the
    # file, and the durable dispense went unanswered — so `reconcile` reported the run as a
    # broker crash. The task below is *correct*, and a correct, sealed, graded task was being
    # filed as an evaluator that fell over.
    #
    # A summary this record cannot read is the same finding as one it cannot honour, so it gets
    # the same answer: the row lands with the env's evidence verbatim and no headline, and then
    # the stream stops.
    class _ExplodingName(str):
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("eq exploded")

        def __ne__(self, other: object) -> bool:
            raise RuntimeError("eq exploded")

        __hash__ = str.__hash__

    class _NamesExplode(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            for item in fb.episode:
                item.name = _ExplodingName(item.name)
            return fb

    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        lambda _name: _NamesExplode(tasks=TASKS),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean

    (row,) = _rows(tmp_path)
    assert row["closure"] == "sealed", "the agent's own seal was reclassified"
    assert row["score"] is None, "a summary this record could not read was headlined anyway"
    assert row["observed"] == [
        _episode_item("correct", True)
    ], "the evidence the agent earned must survive verbatim"
    assert "cannot headline" in (row["diagnostic"] or ""), "the file must say why it is unscored"
    # Serialised before it is compared, because the in-memory row keeps the env's items exactly
    # as the env published them — the name in it is the object that raises.
    assert json.loads(json.dumps(stream.results[0].to_wire())) == row
    assert stream.queue_info().in_flight == 0, "the episode outlived its recorded row"
    # The point of the row landing: the dispense is answered, so the run is not read back as a
    # crash that never happened.
    assert reconcile(tmp_path / "prov") == []
    assert stream.stopped, "an env whose names cannot be compared may not serve the rest of a queue"
    with pytest.raises(RuntimeError, match="cannot headline.*eq exploded"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="cannot headline.*eq exploded") as end:
        await stream.aclose()
    assert "RuntimeError: eq exploded" in str(end.value)


async def test_a_verdict_this_record_cannot_read_still_lands_the_row_it_owes(
    tmp_path: Path,
) -> None:
    # The same defect one step earlier, at the sibling read: the closure is classified from the
    # core's terminal payload, which is the verdict dict *the env returned* with the core's flag
    # written over it — so reading it compares this module's key against keys the env owns, and
    # an env's `__eq__` is env code. Uncontained, that raise reached `_seal_redacted`, which
    # stops the stream and swallows the exception, so no row was composed even though `dispatch`
    # had already decided one was owed. The dispense went unanswered and `reconcile` called the
    # run a broker crash.
    #
    # A verdict this record cannot read is a task with no verdict standing behind it, which is
    # what a failed terminal already means — so it is recorded as one, and the row lands.
    class _ExplodingKey(str):
        def __eq__(self, other: object) -> bool:
            raise RuntimeError("key eq exploded")

        def __ne__(self, other: object) -> bool:
            raise RuntimeError("key eq exploded")

        __hash__ = str.__hash__

    class _VerdictKeyExplodes(_FixtureScoreEnv):
        async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
            evidence = await super().finalize(req)
            evidence.verdict = {_ExplodingKey("finalize_error"): False, "correct": True}
            return evidence

    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        lambda _name: _VerdictKeyExplodes(tasks=TASKS),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()
    assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean

    (row,) = _rows(tmp_path)
    assert row["closure"] == "finalize_error", "a task with no readable verdict is not a seal"
    assert row["score"] is None, "a task with no verdict behind it may not be scored"
    assert "key eq exploded" in (row["diagnostic"] or ""), "the file must name what failed"
    assert stream.results[0].to_wire() == row
    assert reconcile(tmp_path / "prov") == [], "the durable dispense went unanswered"
    assert stream.stopped
    with pytest.raises(RuntimeError, match="key eq exploded"):
        await stream.aclose()


async def test_a_row_keeps_each_item_at_the_level_the_env_published_it(
    tmp_path: Path,
) -> None:
    # A name-keyed row could not express this at all: an inference item is scoped to one step
    # and an episode item to the whole task, so flattened they are indistinguishable — and when
    # they share a name one of them simply disappears. The row is the wire items instead, in
    # order, with `level` and `step`, which is how the JSONL trace and `EvalResult` already
    # record feedback.
    #
    # It also settles the cross-level case without a tie-break: the headline is read from the
    # episode level, because that is what this package means by an outcome that belongs to the
    # task (`select_inband` withholds episode feedback until the terminal for exactly that
    # reason). The step's value is evidence on the row, never the verdict.
    async with TaskStream(
        _feedback_env(
            [
                ("inference", "format_reward", True),
                ("inference", "correct", False),
                ("episode", "correct", True),
            ]
        ),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    ) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = _rows(tmp_path)
    assert row["observed"] == [
        {"name": "format_reward", "value": True, "level": "inference", "step": 1},
        {"name": "correct", "value": False, "level": "inference", "step": 1},
        _episode_item("correct", True),
    ]
    assert row["score"]["success"] is True, "the episode-level item is the task's outcome"


async def test_an_inference_item_never_headlines_a_row(tmp_path: Path) -> None:
    # Dense per-step reward is a legitimate env design — `wordle` publishes one — so a stream
    # must serve it rather than refuse it. What it must not do is headline it: a `reward` that
    # meant "this task" on one row and "step 1" on the next would be unaggregatable, and
    # indistinguishable in the file. The run therefore continues, scored but with empty
    # headlines, and the step's value stays on the row as evidence for whoever wonders why.
    async with TaskStream(
        _feedback_env([("inference", "reward", 0.5), ("inference", "success", True)]),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    ) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = _rows(tmp_path)
    assert (row["score"]["reward"], row["score"]["success"]) == (None, None)
    assert [item["value"] for item in row["observed"]] == [0.5, True]
    assert all(item["level"] == "inference" for item in row["observed"])


async def test_the_run_keeps_the_row_the_file_holds(tmp_path: Path) -> None:
    # What a reader is handed being a copy is half of it; the other half is what it is a copy
    # *of*, and that half a rewrite can drop without anything noticing. What the run keeps is the
    # row re-read from the wire form it just committed, not the one it composed (see
    # `test_reading_the_recorded_rows_cannot_rewrite_them` for the reader's half). Composed, the
    # row is a handle on the env's own values — `observed` holds the
    # items the episode published and `score.feedback` is that same list — so a reader that
    # copies it runs the env's code, on a run that is already over. A feedback value is allowed
    # to be a `str` subclass (the models do not validate on assignment), and one whose
    # `__deepcopy__` raises turns reading a finished run's results into an exception.
    #
    # Re-read, the same row is plain data: the copies below are of what the file holds.
    class _Uncopyable(str):
        def __deepcopy__(self, memo: Any) -> Any:
            raise RuntimeError("this value cannot be copied")

    class _PublishesASubclass(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            if terminated:
                # Assigned rather than constructed, for the reason `_publishes_undescribable`
                # gives: pydantic coerces the subclass away at construction.
                item = EpisodeFeedback(name="note", value="placeholder")
                item.value = _Uncopyable("kept")
                fb.episode.append(item)
            return fb

    async with TaskStream(
        lambda _name: _PublishesASubclass(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    ) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    (row,) = stream.results  # a read that runs env code does not get this far
    note = next(item for item in row.observed if item["name"] == "note")
    assert type(note["value"]) is str, "the run kept the env's object rather than the wire form"
    assert row.score is not None and row.score.success is True
    assert row.observed is not row.score.feedback, "one list is both halves of the row"
    assert [row.to_wire()] == _rows(tmp_path), "memory and the file are one record"


async def test_pulling_a_new_task_seals_the_abandoned_one(tmp_path: Path) -> None:
    # Abandoning a task by pulling the next one must still land exactly one row for it —
    # scored authoritatively, not silently dropped.
    async with _stream(tmp_path, [0, 1]) as stream:
        await stream.get_task()
        await stream.get_task()
        assert stream.queue_info() == QueueInfo(remaining=0, consumed=2, in_flight=1)
    rows = _rows(tmp_path)
    assert len(rows) == 2
    assert rows[0]["task_idx"] == 0 and rows[0]["closure"] == "drained"
    assert rows[0]["score"]["success"] is False


async def test_orderly_drain_seals_the_live_episode(tmp_path: Path) -> None:
    stream = _stream(tmp_path, [0])
    async with stream:
        await stream.get_task()
        assert stream.queue_info().in_flight == 1
    assert len(_rows(tmp_path)) == 1
    assert stream.queue_info().in_flight == 0
    # Idempotent: a second drain neither re-seals nor duplicates a row.
    await stream.aclose()
    assert len(_rows(tmp_path)) == 1


async def test_a_repeated_task_index_is_dispensed_twice(tmp_path: Path) -> None:
    # The queue is a sequence, so the same index may appear twice and must play twice.
    async with _stream(tmp_path, [1, 1]) as stream:
        for _ in range(2):
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})
    rows = _rows(tmp_path)
    assert [row["task_idx"] for row in rows] == [1, 1]
    assert [row["position"] for row in rows] == [0, 1]


async def test_dispatch_without_a_task_is_a_stream_error_not_an_env_step(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0]) as stream:
        result = await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        payload = json.loads(result.content[0].text)  # type: ignore[union-attr]
        assert payload["error"] == "no_active_task" and payload["stream_error"] is True
        assert stream.queue_info().consumed == 0
    # The unstarted task was never dispensed, so it left no row.
    assert not (tmp_path / "prov" / "results.jsonl").exists()


async def test_rejects_a_queue_whose_manifest_varies_by_task(tmp_path: Path) -> None:
    # A server publishes ONE schema per tool name at startup, but `describe(task_id)` is a
    # per-task contract. A queue whose tasks disagree must fail at construction — after the
    # first dispense the contract is already public.
    class _Varying(_FixtureScoreEnv):
        def describe(self, task_id=None) -> TaskSpec:
            spec = super().describe(task_id)
            if task_id == "1":
                spec.tools = [
                    *spec.tools,
                    ToolManifest(
                        name="extra",
                        description="d",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ]
            return spec

    with pytest.raises(ValueError, match="different tool manifest"):
        TaskStream(
            lambda _name: _Varying(tasks=TASKS),
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
        )


async def test_rejects_a_queue_whose_tool_descriptions_vary_by_task(tmp_path: Path) -> None:
    # A tool's description is advertised ONCE, at registration, from whichever task was inspected
    # first — so a per-task description leaves the published contract disagreeing with the framing
    # a later task hands the agent, and is a channel for task-specific text the tool contract never
    # declared. Same rule as a varying schema: reject before anything is served.
    class _VaryingDescription(_FixtureScoreEnv):
        def describe(self, task_id=None) -> TaskSpec:
            spec = super().describe(task_id)
            if task_id == "1":
                spec.tools = [
                    ToolManifest(
                        name=m.name,
                        description=f"{m.description} (task {task_id})",
                        input_schema=m.input_schema,
                        terminal_kind=m.terminal_kind,
                    )
                    for m in spec.tools
                ]
            return spec

    with pytest.raises(ValueError, match="different tool manifest"):
        TaskStream(
            lambda _name: _VaryingDescription(tasks=TASKS),
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
        )


def _drifting_factory(
    mutate: Any, built: List[_TrackedEnv]
) -> Any:
    """A factory whose CATALOG instance is clean and whose every later instance publishes a
    mutated manifest — the shape the construction-time check cannot see, since it only ever
    inspects the first instance. Each env is recorded so its release is observable."""

    class _Drifts(_TrackedEnv):
        def __init__(self, drift: bool, **kwargs: Any) -> None:
            self._drift = drift
            super().__init__(**kwargs)

        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            if self._drift:
                mutate(spec)
            return spec

    def factory(_name: str) -> _Drifts:
        env = _Drifts(drift=bool(built), tasks=TASKS)
        built.append(env)
        return env

    return factory


async def test_an_episode_that_adds_a_tool_is_never_dispensed(tmp_path: Path) -> None:
    # The endpoint registers one schema per tool name when the stream is built, from a fresh
    # instance of the env; every episode after that is a DIFFERENT instance. An instance that
    # adds a tool would have the agent told to call something the server answers with
    # `Unknown tool` — an unearned failure, recorded as an ordinary scored row. So the task is
    # not dispensed at all.
    def add_a_tool(spec: TaskSpec) -> None:
        spec.tools = [
            *spec.tools,
            ToolManifest(
                name="hint",
                description="Ask for a hint.",
                input_schema={"type": "object", "properties": {}},
            ),
        ]

    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _drifting_factory(add_a_tool, built),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    with pytest.raises(RuntimeError, match="different tool manifest.*added \\['hint'\\]"):
        await stream.get_task()

    # Nothing was handed out: no position consumed, no row owed, and the episode's env released
    # here rather than left for a drain that would score a task nobody ever saw.
    assert stream.queue_info() == QueueInfo(remaining=2, consumed=0, in_flight=0)
    assert stream.results == ()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    # Not even a dispense record: one would read back through `reconcile` as a task that was
    # handed out and crashed, which is why the check sits above `_write_dispense`.
    assert read_dispenses(tmp_path / "prov") == []
    assert built[1].closed, "the episode's env outlived the refused dispense"

    # And the drift belongs to the factory, not to this task, so the rest of the queue would be
    # scored against the same broken contract: the stream stops instead.
    with pytest.raises(RuntimeError, match="no further task can be scored"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()
    assert all(env.closed for env in built)


async def test_an_episode_that_changes_a_schema_is_never_dispensed(tmp_path: Path) -> None:
    # The other half, and the one that fails silently. FastMCP does not validate these calls —
    # `build_tool` passes the arguments straight through — so a drifted schema is not rejected
    # at the door; the EPISODE validates the score terminal against its own schema. A call
    # shaped to the published contract is then refused as invalid, and a call shaped to the
    # drifted one is accepted and graded on arguments the finalizer does not read. Both score
    # zero for a right answer.
    def rename_the_argument(spec: TaskSpec) -> None:
        for manifest in spec.tools:
            if manifest.name == SUBMIT_TOOL:
                manifest.input_schema = {
                    "type": "object",
                    "properties": {"final_answer": {"type": "string"}},
                    "required": ["final_answer"],
                }

    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _drifting_factory(rename_the_argument, built),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    with pytest.raises(RuntimeError, match=f"changed \\['{SUBMIT_TOOL}'\\]"):
        await stream.get_task()

    # Over MCP the agent is told less still: a stopped stream serves no further task, so it reads
    # as the end of the queue — the same answer an exhausted one gives, byte for byte apart from
    # the count of tasks the caller itself played. An error here would be a response shape that
    # varies with an integrity failure, which is the channel a terminating call closes; the
    # contract the env tried to publish is nowhere in it either way.
    server = build_stream_server(stream)
    async with Client(server) as client:
        out = await client.call_tool("get_task", {}, raise_on_error=False)
        assert not out.is_error
        seen = out.content[0].text  # type: ignore[union-attr]
        ended = json.loads(seen)
        assert ended == await _reads_as_exhausted(tmp_path, consumed=0)
        assert "final_answer" not in seen
    # ...while the harness has the whole of it, on the stream and out of the drain.
    assert stream.stopped
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()
    assert stream.results == ()
    assert all(env.closed for env in built)


async def test_a_drifted_tool_name_that_cannot_be_described_still_stops_the_stream(
    tmp_path: Path,
) -> None:
    # The refusal names the tools the two manifests disagree about, and naming an env's object
    # calls the env's own `__repr__`. Unguarded, that let the env decide whether its own refusal
    # happened: the raise came from *inside* the message, which is an argument to the stop, so
    # `_stop` was never reached — the drift was refused for this task and nothing else, and the
    # rest of the queue was dispensed and scored against the very contract the call refused.
    #
    # Neither half of the comparison is the env's object any more. The published side is frozen
    # when it is read (`_frozen_manifest`) and the episode side is snapshotted in wire form
    # (`hgym.serve.episode._wire_form`), so an undescribable name is plain text by the time
    # anything here reads it: the drift is named plainly and the stop lands. The guard the name
    # used to need is still what stands behind the values that *are* the env's — the published
    # feedback a summary refusal is about (see `_pick_summary`).
    #
    # What this pins is the outcome, which is the same either way: a contract the endpoint does
    # not serve stops the run rather than being scored against.
    class _Undescribes(_TrackedEnv):
        def __init__(self, undescribable: bool, **kwargs: Any) -> None:
            self._undescribable = undescribable
            super().__init__(**kwargs)

        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            if not self._undescribable:
                return spec
            extra = ToolManifest(
                name="hint",
                description="Ask for a hint.",
                input_schema={"type": "object", "properties": {}},
            )
            # Assigned rather than constructed, for the reason `_publishes_undescribable` gives:
            # the models coerce the subclass away at construction and do not validate on
            # assignment.
            extra.name = _Undescribable("hint")
            spec.tools = [*spec.tools, extra]
            return spec

    built: List[_TrackedEnv] = []

    def factory(_name: str) -> _Undescribes:
        env = _Undescribes(undescribable=not built, tasks=TASKS)
        built.append(env)
        return env

    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    with pytest.raises(RuntimeError, match=r"different tool manifest.*removed \['hint'\]"):
        await stream.get_task()

    # The stop is the whole point: the drift belongs to the factory, so without it the rest of
    # the queue is served against the same contract the endpoint does not expose.
    assert stream.stopped, "the stop the refusal owes was lost to the refusal's own message"
    assert stream.queue_info() == QueueInfo(remaining=2, consumed=0, in_flight=0)
    with pytest.raises(RuntimeError, match="no further task can be scored"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()
    assert stream.results == ()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert all(env.closed for env in built)


async def test_an_episode_manifest_that_cannot_be_compared_still_stops_the_stream(
    tmp_path: Path,
) -> None:
    # The check reads the episode's manifest before it can refuse it, and reading it serializes
    # the schema the env advertises. A schema that cannot be serialized therefore raises from
    # *inside* the comparison that gates the stop, so the drift right beside it — the same
    # rewritten `submit` schema the test above refuses — is never found and `_stop` is never
    # reached. `stopped` stays false and `aclose()` reports a clean run, while an env that
    # drifts on only some instances has the rest of its queue dispensed and scored.
    #
    # A manifest this stream cannot compare is one it cannot confirm, which is the finding this
    # check already exists to act on: the published signature is only ever a schema that
    # serialized, so one that does not is a different manifest and is refused like any other.
    def make_the_schema_unserializable(spec: TaskSpec) -> None:
        for manifest in spec.tools:
            if manifest.name == SUBMIT_TOOL:
                manifest.input_schema = {"type": object()}

    built: List[_TrackedEnv] = []
    stream = TaskStream(
        _drifting_factory(make_the_schema_unserializable, built),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    with pytest.raises(RuntimeError, match="different tool manifest.*not be compared") as refused:
        await stream.get_task()
    # The env's own failure is named rather than swallowed, so the operator is pointed at the
    # schema that could not be read.
    assert "TypeError" in str(refused.value)

    assert stream.stopped, "the stop the refusal owes was lost to reading the manifest"
    assert stream.queue_info() == QueueInfo(remaining=2, consumed=0, in_flight=0)
    with pytest.raises(RuntimeError, match="no further task can be scored"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()
    assert stream.results == ()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert all(env.closed for env in built)


async def test_an_episode_is_checked_against_the_contract_its_own_seal_enforces(
    tmp_path: Path,
) -> None:
    # The check above reads the episode's manifest, and reading it used to be a *second* call
    # into `Env.describe` — the first having already been spent capturing the schemas the seal
    # validates a terminal call's arguments against. Nothing obliges two calls to agree. An env
    # that answers the first with a private contract and every later one with the published
    # contract therefore passed the drift check, was framed with the published tools, and then
    # refused the agent's correctly-shaped submission against a schema it had never been shown:
    # the task drained with `correct=False`, an advertised-correct action recorded as a wrong
    # answer, with nothing anywhere saying the two contracts differed.
    #
    # One description per episode is what closes it. The contract the check confirms, the
    # contract the framing is built from and the contract the seal enforces are the same object,
    # so an env that publishes two of them is caught by the check that already exists instead of
    # slipping between its two reads.
    class _PrivateFirstDescription(_TrackedEnv):
        """Publishes a contract of its own on its FIRST description and the published one after."""

        def __init__(self, private: bool, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._private = private
            self.describes = 0

        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            self.describes += 1
            if self._private and self.describes == 1:
                spec.tools = [
                    m.model_copy(
                        update={
                            "input_schema": {
                                "type": "object",
                                "properties": {"choice": {"type": "integer"}},
                                "required": ["choice"],
                                "additionalProperties": False,
                            }
                        }
                    )
                    if m.name == SUBMIT_TOOL
                    else m
                    for m in spec.tools
                ]
            return spec

    built: List[_PrivateFirstDescription] = []

    def factory(_name: str) -> _PrivateFirstDescription:
        # The catalog instance is described at construction; only the episodes drift.
        env = _PrivateFirstDescription(private=bool(built), tasks=TASKS)
        built.append(env)
        return env

    stream = TaskStream(
        factory, [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)], prov_dir=tmp_path / "prov"
    )
    with pytest.raises(RuntimeError, match="different tool manifest.*changed"):
        await stream.get_task()
    # No task was handed out at all, so there is no framing to have been wrong about and no row
    # to carry a number the agent could not have earned.
    assert stream.results == ()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert stream.stopped, "an env that publishes two contracts may not serve the rest of a queue"
    assert built[1].describes == 1, "the episode was described more than once"
    assert built[1].closed, "the refused episode's env was never released"
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()


async def test_reading_the_published_contract_cannot_rewrite_it(tmp_path: Path) -> None:
    # Both public readings of the frozen manifest used to hand back the stream's own schema
    # object: `DispensedTask.tools[i]["input_schema"]` *was* `TaskStream.tools[i].input_schema`,
    # and both were the dict the env returned. Editing either one edited the contract — the
    # framing every later task is dispensed with, and the manifest a new episode is confirmed
    # against, which is what makes it worse than a cosmetic leak: the drift check would compare
    # a fresh episode against the rewritten schema and find them in agreement.
    #
    # So each read is its own detached view, and the endpoint goes on serving what it registered.
    clean = await _clean_terminal_response(tmp_path)
    async with _stream(tmp_path, [0, 1]) as stream:
        first = await stream.get_task()
        assert first is not None
        published = {tool.name: tool.input_schema for tool in stream.tools}
        framed = {tool["name"]: tool["input_schema"] for tool in first.tools}
        assert published[SUBMIT_TOOL] == framed[SUBMIT_TOOL]
        assert published[SUBMIT_TOOL] is not framed[SUBMIT_TOOL]
        assert published[SUBMIT_TOOL] is not next(
            tool.input_schema for tool in stream.tools if tool.name == SUBMIT_TOOL
        ), "two reads of the manifest share one object"

        # Two payloads built from one task do not share a schema either.
        wire = first.to_wire()
        next(t for t in wire["tools"] if t["name"] == SUBMIT_TOOL)["input_schema"]["required"] = [
            "invented on the wire"
        ]
        again = next(t for t in first.to_wire()["tools"] if t["name"] == SUBMIT_TOOL)
        assert again["input_schema"]["required"] == ["answer"], "two payloads share one schema"

        framed[SUBMIT_TOOL]["required"] = ["invented"]
        published[SUBMIT_TOOL]["required"] = ["also invented"]
        served = {tool.name: tool.input_schema for tool in stream.tools}
        assert served[SUBMIT_TOOL]["required"] == ["answer"], "the contract was rewritten"
        # A server built after the edit registers what the stream froze, not what a reader wrote.
        async with Client(build_stream_server(stream)) as client:
            listed = {tool.name: tool.inputSchema for tool in await client.list_tools()}
        assert listed[SUBMIT_TOOL]["required"] == ["answer"]
        # The endpoint still honours it, and the next task is still framed with it.
        assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean
        second = await stream.get_task()
        assert second is not None
        assert {tool["name"]: tool["input_schema"] for tool in second.tools} == served

    first_row = stream.results[0]
    assert first_row.score is not None and first_row.score.success is True


@pytest.mark.parametrize(
    ("break_spec", "defect"),
    [
        pytest.param(
            lambda spec: setattr(spec, "instructions", object()),
            "instructions are not the text",
            id="the instructions are not text",
        ),
        pytest.param(
            lambda spec: setattr(spec, "horizon", 2.5),
            "budget is not a whole number",
            id="the budget is not a whole number of steps",
        ),
        pytest.param(
            # `bool` is an `int` subclass, so an unguarded test would advertise `True` as a
            # budget of one step.
            lambda spec: setattr(spec, "horizon", True),
            "budget is not a whole number",
            id="the budget is a bool",
        ),
        pytest.param(
            # Text to every `isinstance` check, and a `UnicodeEncodeError` the moment the
            # endpoint encodes it — a framing whose type is right and whose bytes do not exist.
            # The three above are settled by the declared types; this one is only settled by
            # actually encoding the two values, which is what the check proves rather than
            # assumes. The phrasing is the spec-side one: a failure the whole-object proof
            # caught instead would blame the framing, not the env that published it.
            lambda spec: setattr(spec, "instructions", "answer this \ud800 question"),
            "could not put it on the wire",
            id="the instructions are text the endpoint cannot encode",
        ),
    ],
)
async def test_a_task_this_endpoint_cannot_hand_over_is_never_dispensed(
    tmp_path: Path, break_spec: Any, defect: str
) -> None:
    # The framing is two values off the episode's own spec plus the frozen contract, and a model
    # validates a field when it is built rather than when it is assigned — so an env that edits
    # its spec afterwards publishes a framing nothing between here and the wire looks at.
    #
    # The dispense is durable *before* the task is handed out, so where that is found decides
    # what it costs. Found afterwards it is a committed dispense the agent was never answered
    # for: the episode is live, the drain ends it, and the row says the agent played the task out
    # and got it wrong — a wrong number standing where a missing one was the truth. So it is
    # found first, and there is no task rather than a task nobody received.
    class _Unframable(_TrackedEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            break_spec(spec)
            return spec

    built: List[_Unframable] = []

    def factory(_name: str) -> _Unframable:
        env = _Unframable(tasks=TASKS)
        built.append(env)
        return env

    exhausted = await _reads_as_exhausted(tmp_path, consumed=0)
    stream = TaskStream(factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    server = build_stream_server(stream)
    async with Client(server) as client:
        answer = json.loads((await client.call_tool("get_task", {})).content[0].text)  # type: ignore[union-attr]
    # Nothing was handed out, so nothing was spent: no durable dispense for recovery to answer,
    # no row, and the position is still owed.
    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert stream.results == ()
    assert reconcile(tmp_path / "prov") == []
    assert stream.queue_info() == QueueInfo(remaining=1, consumed=0, in_flight=0)
    assert built[-1].closed, "the refused episode's env was never released"
    # An env that publishes a framing this endpoint cannot carry publishes it again next time it
    # is asked, and a refusal does not advance the position — so this is the run's, not the
    # task's. The agent reads it as the end of the queue; the harness gets the reason.
    assert stream.stopped
    assert answer == exhausted
    with pytest.raises(RuntimeError, match="framing this stream cannot hand out") as closing:
        await stream.aclose()
    assert defect in str(closing.value.__cause__)


async def test_a_framing_is_plain_data_by_the_time_a_task_is_dispensed(
    tmp_path: Path,
) -> None:
    # The sibling of the check above, on the half of the framing that comes from the frozen
    # contract. A tool's name and description are the env's objects, and the stream used to keep
    # them: `model_copy` is shallow, so the description an env published was the one every task's
    # framing carried and `DispensedTask.to_wire` deep-copied — running the env's code long after
    # the dispense was durable. A `str` subclass whose `__deepcopy__` raises therefore left a
    # committed dispense with no task to answer it, and the drain scored the task the agent never
    # received. (Only the catalog instance carries it here: an episode's own spec is deep-copied
    # when it is described, so an episode that also held one would be refused before the dispense
    # by the path above.)
    class _Uncopyable(str):
        def __deepcopy__(self, memo: Any) -> Any:
            raise RuntimeError("this description cannot be copied")

    class _PoisonsTheCatalogDescription(_TrackedEnv):
        def __init__(self, poisoned: bool, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._poisoned = poisoned

        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            if self._poisoned:
                for manifest in spec.tools:
                    manifest.description = _Uncopyable(manifest.description)
            return spec

    built: List[_PoisonsTheCatalogDescription] = []

    def factory(_name: str) -> _PoisonsTheCatalogDescription:
        # The catalog instance is the one whose manifest this stream freezes and advertises.
        env = _PoisonsTheCatalogDescription(poisoned=not built, tasks=TASKS)
        built.append(env)
        return env

    stream = TaskStream(factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    # Nothing the env kept reaches what the endpoint advertises, so nothing it kept can decide
    # whether a dispensed task can be delivered.
    assert all(type(tool.description) is str for tool in stream.tools)
    async with stream:
        task = await stream.get_task()
        assert task is not None
        wire = task.to_wire()
        assert all(type(tool["description"]) is str for tool in wire["tools"])
        assert {t["description"] for t in wire["tools"]} == {
            t.description for t in stream.tools
        }, "freezing the description changed what the endpoint says a tool is for"
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    (row,) = stream.results
    assert row.closure == "sealed" and row.score is not None and row.score.success is True
    assert reconcile(tmp_path / "prov") == []


async def test_an_episode_is_not_opened_on_whether_the_env_can_copy_its_own_schema(
    tmp_path: Path,
) -> None:
    # An episode snapshots the contract it was opened on: normalized, so what it enforces is what
    # a client is shown, and detached, so the env that published it cannot rewrite it afterwards.
    # Which of those happens first decides whose code decides that an episode exists at all.
    #
    # Detached first, the copy is taken of the env's own objects — so a `__deepcopy__` an env
    # wrote runs on a value the wire form replaces with plain data one line later, and a raise
    # there comes back out of `get_task` as the env's own exception with the stream unstopped,
    # `aclose` reporting a clean run, and the position still owed. Normalized first, the copy is
    # of data the round trip already made plain, and that value's copy code is never reached.
    #
    # This is not decoration: the value is JSON-clean, so nothing at construction sees it — the
    # frozen manifest is a JSON round trip and strips the subclass — and the episode is the only
    # place it is copied.
    class _Uncopyable(str):
        def __deepcopy__(self, memo: Any) -> Any:
            raise RuntimeError("this schema value cannot be copied")

    class _PoisonsItsOwnSchema(_TrackedEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            for manifest in spec.tools:
                if manifest.name == SUBMIT_TOOL:
                    manifest.input_schema = {
                        **manifest.input_schema,
                        "title": _Uncopyable("submit your answer"),
                    }
            return spec

    def factory(_name: str) -> _PoisonsItsOwnSchema:
        return _PoisonsItsOwnSchema(tasks=TASKS)

    stream = TaskStream(factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    async with stream:
        task = await stream.get_task()
        assert task is not None, "the env published nothing this endpoint could not carry"
        assert stream.queue_info() == QueueInfo(remaining=0, consumed=1, in_flight=1)
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert not stream.stopped
    (row,) = stream.results
    assert row.closure == "sealed" and row.score is not None and row.score.success is True
    assert reconcile(tmp_path / "prov") == []


@pytest.mark.parametrize(
    ("break_tool", "match"),
    [
        pytest.param(
            lambda m: setattr(m, "description", {"not": "text"}),
            "description is dict, not text",
            id="a description that is not text",
        ),
        pytest.param(
            lambda m: setattr(m, "description", object()),
            "could not put on the wire.*not JSON serializable",
            id="a description the wire cannot carry",
        ),
        pytest.param(
            lambda m: setattr(m, "description", "a hint \ud800 for the model"),
            "could not put on the wire.*surrogates not allowed",
            id="a description that is text but not UTF-8",
        ),
        pytest.param(
            lambda m: m.input_schema.__setitem__("const", float("nan")),
            "could not put on the wire.*not JSON compliant",
            id="a schema value strict JSON refuses",
        ),
    ],
)
async def test_a_tool_this_endpoint_could_not_advertise_is_refused_before_it_is_served(
    tmp_path: Path, break_tool: Any, match: str
) -> None:
    # The other half of freezing the advertised manifest: what the endpoint will actually send.
    # A tool's name and description are text on the wire, and a value that is not text at all
    # cannot be made into plain data — but neither type nor `json.dumps` settles it. The stdlib
    # encoder accepts `NaN` and writes a token no JSON parser must read (FastMCP sends `null`),
    # and a `str` may hold an unpaired surrogate that is text to every check and a
    # `UnicodeEncodeError` the moment a transport encodes it. Accepted, each one advertises a
    # contract the episode then refuses the agent against — an advertised-correct action recorded
    # as a wrong answer.
    #
    # So the freeze proves the encode the endpoint does, and refuses at construction, where
    # nothing has been served and no env holds a task.
    class _CannotBeAdvertised(_FixtureScoreEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            for manifest in spec.tools:
                break_tool(manifest)
            return spec

    with pytest.raises(ValueError, match=match):
        TaskStream(
            lambda _name: _CannotBeAdvertised(tasks=TASKS),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=tmp_path / "prov",
        )
    assert not (tmp_path / "prov").exists()


async def test_the_episode_enforces_the_contract_the_endpoint_advertises(
    tmp_path: Path,
) -> None:
    # The two ends of one contract: the endpoint advertises this stream's frozen copy, and the
    # episode validates a terminal call against the schema it was opened on. A JSON scalar has
    # subclasses and the models do not validate on assignment, so a `const` can be text that
    # serialises as `"4"` and matches nothing once it is compared. Advertised, the agent is shown
    # `const: "4"`; enforced, the value it was shown is refused — and the manifest check cannot
    # see the difference, because it compares the serialised form too.
    #
    # The agent then sends exactly the advertised-correct action, the seal refuses it, and the
    # run records a task it answered wrong. So what an episode enforces is the wire form, which
    # is the same value the endpoint published.
    class _NeverEqual(str):
        def __eq__(self, other: Any) -> bool:
            return False

        def __ne__(self, other: Any) -> bool:
            return True

        __hash__ = str.__hash__

    class _ConstIsNeverEqual(_FixtureScoreEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            for manifest in spec.tools:
                if manifest.name == SUBMIT_TOOL:
                    # Assigned after construction, for the reason `_publishes_undescribable`
                    # gives: pydantic coerces the subclass away when the model is built.
                    schema = json.loads(json.dumps(manifest.input_schema))
                    schema["properties"]["answer"]["const"] = _NeverEqual("4")
                    manifest.input_schema = schema
            return spec

    clean = await _clean_terminal_response(tmp_path)
    stream = TaskStream(
        lambda _name: _ConstIsNeverEqual(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        task = await stream.get_task()
        assert task is not None
        framed = next(t for t in task.tools if t["name"] == SUBMIT_TOOL)
        advertised = framed["input_schema"]["properties"]["answer"]["const"]
        assert type(advertised) is str and advertised == "4"
        # The action the framing describes is the action the endpoint accepts.
        assert _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == clean

    (row,) = stream.results
    assert row.closure == "sealed", "an advertised-correct action was refused"
    assert row.score is not None and row.score.success is True
    assert not stream.stopped


async def test_a_queue_entry_is_the_identity_its_record_is_filed_under(tmp_path: Path) -> None:
    # `TaskRef` annotates its fields and validates neither, so a queue could hold anything —
    # and these two are identity, not payload. Several things coerce them and none of them agree:
    # the episode loads `int(task_idx)`, the durable row is appended with the caller's own value,
    # and the run's in-memory copy is re-read through `from_wire`, which coerces again. A `1.9`
    # played task 1, was recorded as `1.9`, and came back as `1` — three readings of one task,
    # with a resume refusing the run's own record as a queue it does not match.
    #
    # Refused before an env is built, because a canonical identity invented after the commit is
    # what put a number in the file that nothing else agrees with.
    for bad, match in (
        ((ENV_NAME, 1.9), "task index must be a whole number"),
        ((ENV_NAME, True), "task index must be a whole number"),
        ((ENV_NAME, "0"), "task index must be a whole number"),
        (("", 0), "env must be a non-empty string"),
        ((object(), 0), "env must be a non-empty string"),
    ):
        with pytest.raises(ValueError, match=match):
            TaskStream(_env_for, [bad], prov_dir=tmp_path / "prov")  # type: ignore[list-item]
    assert not (tmp_path / "prov").exists()

    # And the ordinary entry still plays, with one identity from the queue to the file.
    async with _stream(tmp_path, [1]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})
    (durable,) = _rows(tmp_path)
    (row,) = stream.results
    assert durable["task_idx"] == row.task_idx == 1
    assert read_results(tmp_path / "prov")[0].task_idx == 1


async def test_a_call_is_routed_to_the_episode_by_a_name_this_stream_owns(
    tmp_path: Path,
) -> None:
    # The endpoint advertises the frozen contract and the episode enforces its own wire-form
    # snapshot, so both ends of a call are this stream's plain data. The route between them was
    # not: it kept the env's own name object and passed that object into the episode. A name
    # whose `__hash__` answers differently once the run is under way therefore loses the call —
    # on a tool the endpoint published in plain text, that the agent called exactly as published.
    #
    # Nothing downstream can see it. The lost call is not an outcome, the agent goes on to end
    # the task itself, and the rule that keeps a recovered task the agent's (correctly) declines
    # to stop or unscore it — so the missing step is simply absent from the record and the score
    # it changes looks fully earned. Two correct rules, one wrong number, and only the route to
    # blame.
    class _Unhashable(str):
        """Ordinary text on the wire; a key that cannot be looked up once the run has started."""

        _armed = False

        def __hash__(self) -> int:
            if type(self)._armed:
                raise RuntimeError("this tool name cannot be hashed")
            return str.__hash__(self)

    class _NoopIsTheEnvsOwnObject(_FixtureScoreEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            for manifest in spec.tools:
                if manifest.name == "noop":
                    # Assigned rather than constructed: pydantic coerces the subclass away when
                    # the model is built, and does not validate on assignment.
                    manifest.name = _Unhashable("noop")
            return spec

    stream = TaskStream(
        lambda _name: _NoopIsTheEnvsOwnObject(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    try:
        async with Client(build_stream_server(stream)) as client:
            assert "noop" in {tool.name for tool in await client.list_tools()}
            await client.call_tool("get_task", {})
            _Unhashable._armed = True  # the contract is published; the run starts here
            step = await client.call_tool("noop", {}, raise_on_error=False)
            assert not step.is_error, "the advertised call never reached the episode"
            assert json.loads(step.content[0].text)["terminated"] is False  # type: ignore[union-attr]
            done = await client.call_tool(SUBMIT_TOOL, {"answer": "4"}, raise_on_error=False)
            assert not done.is_error
        await stream.aclose()
    finally:
        _Unhashable._armed = False

    (row,) = stream.results
    assert row.closure == "sealed"
    assert row.score is not None and row.score.success is True
    assert not stream.stopped
    # The step the agent made is in the record, which is what a score computed from it needs.
    assert [r["closure"] for r in _rows(tmp_path)] == ["sealed"]


async def test_a_framing_that_cannot_be_read_still_stops_the_stream(tmp_path: Path) -> None:
    # Confirming the framing reads the episode's own spec, and an attribute of an env's object
    # is the env's code. Unguarded, the env decides whether its own refusal happens: the raise
    # comes from inside the check, so the task is refused for this episode and nothing else —
    # `stopped` stays false, the position is never advanced, and every later dispense meets the
    # same env again while `aclose()` reports a clean run over a queue it never served.
    #
    # A framing this stream cannot read is one it cannot hand over, which is the finding this
    # check already exists to act on.
    #
    # A subclass of the spec rather than a stand-in, because the episode snapshots the contract
    # it was opened on with `model_copy(deep=True)` — a copy keeps the class, so what the check
    # reads is still an object whose own code answers for the field.
    class _UnreadableSpec(TaskSpec):
        def __getattribute__(self, name: str) -> Any:
            if name == "instructions":
                raise RuntimeError("instructions exploded")
            return super().__getattribute__(name)

    class _Unreadable(_TrackedEnv):
        def describe(self, task_id: Any = None) -> Any:
            spec = super().describe(task_id)
            return _UnreadableSpec.model_construct(**dict(spec))

    built: List[_TrackedEnv] = []

    def factory(_name: str) -> _Unreadable:
        env = _Unreadable(tasks=TASKS)
        built.append(env)
        return env

    stream = TaskStream(
        factory, [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)], prov_dir=tmp_path / "prov"
    )
    with pytest.raises(RuntimeError, match="cannot hand out.*reading it raised") as refused:
        await stream.get_task()
    # The env that published it is pointed at, rather than the failure being swallowed.
    assert "RuntimeError: instructions exploded" in str(refused.value)

    assert stream.stopped, "the stop the refusal owes was lost to reading the framing"
    assert stream.queue_info() == QueueInfo(remaining=2, consumed=0, in_flight=0)
    with pytest.raises(RuntimeError, match="could not be served past it"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()
    assert stream.results == ()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert reconcile(tmp_path / "prov") == []
    assert all(env.closed for env in built)


async def test_the_whole_framing_is_proved_before_the_dispense_is_committed(
    tmp_path: Path,
) -> None:
    # Two of the framing's fields were proved and the rest were assembled after the dispense was
    # durable. An env key is one of those: it is a caller's own string, checked to be exact
    # non-empty text and nothing more, and a Python `str` may hold an unpaired surrogate that no
    # transport can encode. The endpoint then answers `get_task` with a serialization error — the
    # agent is handed nothing — while the dispense record already says a task went out, so the
    # drain ends a task nobody received and files it as one the agent played and lost.
    #
    # What is proved is therefore the object that will be returned, in the wire form it will be
    # returned as, before anything of the record is durable. A field added to the framing later
    # is covered the day it is added rather than the day someone remembers to list it.
    key = "\ud800"
    exhausted = await _reads_as_exhausted(tmp_path, consumed=0)
    stream = TaskStream(_env_for, [TaskRef(key, 0)], prov_dir=tmp_path / "prov")
    server = build_stream_server(stream)
    async with Client(server) as client:
        out = await client.call_tool("get_task", {}, raise_on_error=False)
        assert not out.is_error, "the endpoint answered a dispense it could not encode"
        answer = json.loads(out.content[0].text)  # type: ignore[union-attr]

    assert answer == exhausted
    # Nothing was spent: no durable dispense for recovery to answer, no row, position still owed.
    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert stream.results == ()
    assert reconcile(tmp_path / "prov") == []
    assert stream.queue_info() == QueueInfo(remaining=1, consumed=0, in_flight=0)
    # An env key is the run's, not this task's, so the queue cannot be served past it.
    assert stream.stopped
    with pytest.raises(RuntimeError, match="could not be put on the wire"):
        await stream.aclose()


async def test_the_stream_drives_the_terminal_it_published(tmp_path: Path) -> None:
    # The score terminal's name is what this stream calls to end a task the agent did not, and
    # the episode finds that terminal by looking the name up. Kept as the env's own object, a
    # name whose `__hash__` answers differently once the run is under way makes the stream's own
    # terminal uncallable: the reserved abort answers instead, the row is an ordinary wrong
    # answer, and nothing anywhere says a name was involved.
    #
    # Frozen native — before the score-terminal cache, the signature or the framing are derived
    # from it — the name the stream drives is its own plain copy.
    class _Unhashable(str):
        _armed = False

        def __hash__(self) -> int:
            if type(self)._armed:
                raise RuntimeError("this tool name cannot be hashed")
            return str.__hash__(self)

    class _SubmitIsTheEnvsOwnObject(_FixtureScoreEnv):
        def describe(self, task_id: Any = None) -> TaskSpec:
            spec = super().describe(task_id)
            for manifest in spec.tools:
                if manifest.name == SUBMIT_TOOL:
                    manifest.name = _Unhashable(SUBMIT_TOOL)
                    # Nothing required, so the terminal the stream drives on the agent's behalf
                    # is one it *can* call: the fallback to the reserved abort exists for a
                    # score terminal whose arguments cannot be invented, and this test is about
                    # the other reason a forced terminal goes uncalled.
                    manifest.input_schema = {
                        **manifest.input_schema,
                        "required": [],
                    }
            return spec

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            if terminated and evidence is not None:
                # Which terminal ended this task, published where the row can carry it: the
                # score terminal the endpoint advertises, or the reserved abort.
                fb.episode.append(
                    EpisodeFeedback(name="reward", value=1.0 if evidence.source != "abort" else 0.0)
                )
            return fb

    stream = TaskStream(
        lambda _name: _SubmitIsTheEnvsOwnObject(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    try:
        await stream.get_task()
        _Unhashable._armed = True  # the contract is published; the run starts here
        await stream.aclose()  # the agent ends nothing, so the stream drives its own terminal
    finally:
        _Unhashable._armed = False

    (row,) = stream.results
    assert row.score is not None and row.score.reward == 1.0, (
        "the stream could not call the terminal it published"
    )
    assert not stream.stopped


async def test_the_framing_lists_the_tools_the_endpoint_serves(tmp_path: Path) -> None:
    # The contract the agent is framed with and the contract it can call are one object. The
    # framing is built from the published manifest — the same list the server registered — so
    # the two cannot drift apart even before the check above rejects an env that tries.
    async with _stream(tmp_path, [0]) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            task = json.loads((await client.call_tool("get_task", {})).content[0].text)  # type: ignore[union-attr]
            framed = {t["name"]: t["input_schema"] for t in task["tools"]}
            served = {t.name: t.inputSchema for t in await client.list_tools()}
        assert framed and framed == {n: s for n, s in served.items() if n in framed}
        assert set(framed) <= set(served), "the framing named a tool the server does not serve"
        assert task["env"] == ENV_NAME


async def test_rejects_unsupported_configurations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_in_flight must be at least 1"):
        _stream(tmp_path, [0], max_in_flight=0)
    with pytest.raises(ValueError, match="non-empty queue"):
        _stream(tmp_path, [])
    # An env key is joined into a tool name only once a second env is in the queue, so that is
    # also the only time its shape is anyone's business. One env, and the key stays the private
    # label it has always been — `test_serve_stream_multi_env.py` holds the other half of this.
    with pytest.raises(ValueError, match="ambiguous"):
        TaskStream(
            _env_for,
            [TaskRef("has__separator", 0), TaskRef("other", 0)],
            prov_dir=tmp_path / "prov",
        )


async def test_each_episode_gets_its_own_env(tmp_path: Path) -> None:
    # Sealing an episode closes its env, and closing a ToolUsingEnv ends EVERY session it
    # tracks — so the stream must hand each episode a fresh instance.
    built: List[_FixtureScoreEnv] = []

    def factory(_name: str) -> _FixtureScoreEnv:
        env = _FixtureScoreEnv(tasks=TASKS)
        built.append(env)
        return env

    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})
    # One catalog env for the manifest + one per dispensed task, all distinct objects.
    assert len(built) == 3
    assert len({id(env) for env in built}) == 3


# ----- over MCP -----


async def test_mcp_end_to_end(tmp_path: Path) -> None:
    # The same loop an agent runs: get_task, work it with the advertised native tools, submit.
    async with _stream(tmp_path, [0, 2]) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            names = {tool.name for tool in await client.list_tools()}
            assert {"get_task", "queue_info", SUBMIT_TOOL, "terminate"} <= names

            for expected in ("4", "10"):
                task = json.loads((await client.call_tool("get_task", {})).content[0].text)  # type: ignore[union-attr]
                assert set(task) == {"env", "instructions", "budget", "tools"}
                out = await client.call_tool(SUBMIT_TOOL, {"answer": expected})
                assert json.loads(out.content[0].text)["terminated"] is True  # type: ignore[union-attr]

            exhausted = json.loads((await client.call_tool("get_task", {})).content[0].text)  # type: ignore[union-attr]
            assert exhausted == {"done": True, "remaining": 0, "consumed": 2, "in_flight": 0}
    assert [row["score"]["success"] for row in _rows(tmp_path)] == [True, True]


async def test_server_rejects_a_control_tool_collision(tmp_path: Path) -> None:
    class _Colliding(_FixtureScoreEnv):
        def describe(self, task_id=None) -> TaskSpec:
            spec = super().describe(task_id)
            spec.tools = [
                *spec.tools,
                ToolManifest(
                    name="get_task",
                    description="d",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]
            return spec

    stream = TaskStream(
        lambda _name: _Colliding(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    try:
        with pytest.raises(ValueError, match="reserved control tool"):
            build_stream_server(stream)
    finally:
        await stream.aclose()


# ----- cancellation an env *emits* -----
#
# `asyncio.CancelledError` inherits from `BaseException`, so it walks straight through an
# `except Exception` written to contain everything. In this module that handler shape means two
# unrelated things — a boundary whose contract is that nothing escapes, and a deliberate
# passthrough where a caller's cancellation must end that caller — and the same exception object
# is legitimate at one and a defect at the other. These pin the containment side: a
# `CancelledError` raised by an env is that env failing, and it may not change what the harness
# answers, what it records, or whether it can be shut down.


class _CancelsOnClose(_FixtureScoreEnv):
    """An env whose teardown raises `CancelledError` — what an env inherits by awaiting a child
    task that was cancelled, and what it can simply raise."""

    def __init__(self, cancels: bool, summary: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cancels = cancels
        self._summary = summary
        self.closes = 0

    def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
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
    # The per-episode release runs as the entry's own tail task and its callers join it through
    # a shield, so a `CancelledError` observed inside it is not cancellation of the caller — the
    # shield already separated the two. Letting it out lands it in the middle of `_seal`, PAST
    # the durable append and BEFORE the stop that seal still owes: the agent's terminating call
    # answers with a traceback instead of the constant, the stop is never published, and
    # `aclose` then reports a clean run over an env whose headline this record has already
    # refused to read. The row is durable either way; what is at stake is everything the seal
    # does after it.
    factory, built = _cancelling_close_factory("episode", summary=summary)
    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()

    answer = _terminal_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    assert answer == await _clean_terminal_response(tmp_path), (
        "a cancelling teardown changed the agent's answer"
    )
    assert built[1].closes == 1, "the episode's env was never closed, so nothing was under test"
    (row,) = stream.results
    assert len(_rows(tmp_path)) == 1, "the durable row is owed whatever the teardown did"
    assert row.closure == "sealed", "a teardown failure is not how the task ended"

    if headlinable:
        assert row.score is not None and row.score.success is True
        assert not stream.stopped, "a teardown failure is not the run's outcome"
        await stream.aclose()
    else:
        assert row.score is None
        assert "cannot headline" in (row.diagnostic or ""), "the row must say why it is unscored"
        assert stream.stopped, "the stop the seal owed was lost with the cancelled release"
        with pytest.raises(RuntimeError, match="cannot headline"):
            await stream.get_task()
        with pytest.raises(RuntimeError, match="cannot headline"):
            await stream.aclose()


async def test_a_catalog_teardown_that_cancels_still_lets_the_stream_close(
    tmp_path: Path,
) -> None:
    # The catalog is released once, under a flag set before the loop runs, so a cancellation out
    # of one env's close ends the drain with every env after it still open and no later `aclose`
    # willing to try again. Shutdown would have no orderly exit, and it would report itself
    # cancelled for a teardown failure that is not the run's outcome.
    factory, built = _cancelling_close_factory("catalog")
    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

    await stream.aclose()
    await stream.aclose()
    assert built[0].closes == 1, "the catalog env was never closed, so nothing was under test"
    assert len(stream.results) == 1
    with pytest.raises(RuntimeError, match="this stream is closed"):
        await stream.get_task()


def _raises_on_the_terminal_step(exc: Callable[[], BaseException]) -> Any:
    """An env with no score terminal, so `_verify` runs inline on the terminating call and its
    failure reaches the stream through that call itself."""

    class _RaisesOnTheTerminalStep(_FixtureScoreEnv):
        score_terminal_tool = None

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if terminated:
                raise exc()
            return FeedbackCollection()

    return lambda _name: _RaisesOnTheTerminalStep(tasks=TASKS)


async def test_an_env_that_cancels_on_the_agents_terminal_answers_the_same_way(
    tmp_path: Path,
) -> None:
    # The terminal is committed and the env is what failed, so a row is owed and the stream must
    # stop — the same answer an env raising anything else gets, and for the same reason. Read as
    # cancellation instead it skips all of it: `dispatch` raises at the agent on exactly the call
    # the redaction exists for, the stop is never recorded so the queue serves on against an env
    # that raises for every task in it, and `aclose` returns clean having lost an outcome.
    stream = TaskStream(
        _raises_on_the_terminal_step(asyncio.CancelledError),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()

    answer = _terminal_text(await stream.dispatch(TERMINATE_TOOL_NAME, {}))
    assert answer == await _clean_terminal_response(tmp_path), (
        "an env failure changed the agent's answer"
    )
    assert stream.stopped, "the stream served on against an env that raised at the terminal"
    assert len(stream.results) == 1
    with pytest.raises(RuntimeError, match="raised while ending a task") as end:
        await stream.aclose()
    assert isinstance(end.value.__cause__, asyncio.CancelledError)
    assert len(_rows(tmp_path)) == 1


@pytest.mark.parametrize(
    "failure", [asyncio.CancelledError, RuntimeError], ids=["cancels", "raises"]
)
async def test_a_forced_terminal_that_fails_is_recorded_the_same_way_however_it_failed(
    tmp_path: Path, failure: Callable[[], BaseException]
) -> None:
    # The same env on the stream's own forced terminal, which is where cancellation costs most:
    # this runs while the drain is composing the row, so letting one out cancels whoever is
    # sealing — no row is composed for a task that was dispensed, the entry is handed back
    # unsealed, and the drain reports the run as cancelled instead of recording what the queue is
    # still owed. Contained, it is classified exactly as any other failure of a terminal the
    # stream drove: the terminal is committed and the env is what failed, so the task ended with
    # no verdict standing behind it and the row lands unscored over a stopped queue. The two
    # parameters assert that is one behaviour and not two.
    stream = TaskStream(
        _raises_on_the_terminal_step(failure),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()

    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.aclose()
    (row,) = stream.results
    assert row.closure == "finalize_error", "an env the stream could not end is not an agent seal"
    assert row.score is None, "a task with no verdict behind it may not be scored"
    assert failure.__name__ in (row.diagnostic or ""), "the failure must be on the row"
    assert [r["closure"] for r in _rows(tmp_path)] == ["finalize_error"]
    assert stream.stopped, "the queue may not be served on against an env that raises at the end"


def _raises_one_object_at_both_boundaries(exc: BaseException) -> Any:
    """A non-seal env — ``verify`` runs inline on every call — that raises the **same object** on
    a call the agent makes mid-episode and on the terminal the stream drives afterwards.

    One instance, two boundaries: which failure a `terminal_error` is cannot be read back off
    the object, because nothing stops an env raising one twice."""

    class _RaisesTheSameObject(_FixtureScoreEnv):
        score_terminal_tool = None

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            raise exc

    return lambda _name: _RaisesTheSameObject(tasks=TASKS)


async def test_one_failure_raised_on_both_boundaries_is_still_a_failed_terminal(
    tmp_path: Path,
) -> None:
    # Python lets one exception instance be raised as often as its raiser likes, so *which*
    # failure a terminal error is cannot be read back off the object. An env that raises one
    # shared error on a call the agent made and raises that same object again on the terminal the
    # stream drives would, read by identity, have a genuine terminal failure taken for the
    # promoted call error above: unscored either way, but with a diagnostic naming the wrong
    # boundary, no stop, and the rest of the queue dispensed and scored against an env that can
    # end no task of its own.
    #
    # So the entry records where the failure came from at the moment it records the failure, and
    # the promotion is the only thing that reads as one.
    shared = RuntimeError("the same object, raised twice")
    stream = TaskStream(
        _raises_one_object_at_both_boundaries(shared),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
    await stream.get_task()
    with pytest.raises(RuntimeError, match="the same object"):
        await stream.dispatch("noop", {})  # the agent's call is lost, and the entry keeps it

    # The stream now has to end the task itself, and the terminal it drives raises that same
    # object again — on its way out of a task it did end, which is the failure that stops a run.
    with pytest.raises(RuntimeError, match="no further task could be scored"):
        await stream.get_task()
    with pytest.raises(RuntimeError, match="failed while the stream ended a task"):
        await stream.aclose()

    (row,) = stream.results
    assert row.closure == "finalize_error" and row.score is None
    assert "while the stream ended the task" in (row.diagnostic or ""), (
        "a failed terminal was read as the agent's own lost call"
    )
    assert "on a call the agent made" not in (row.diagnostic or "")
    assert stream.stopped
    # The task after it was never served, which is the whole of what the stop buys.
    assert len(_rows(tmp_path)) == 1
    assert stream.queue_info().remaining == 1


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
        def describe(self, task_id=None) -> TaskSpec:
            spec = super().describe(task_id)
            if self._cancels:  # the episode instance, never the catalog one
                spec.tools = [
                    *spec.tools,
                    ToolManifest(
                        name="hint",
                        description="d",
                        input_schema={"type": "object", "properties": {}},
                    ),
                ]
            return spec

    def factory(_name: str) -> Any:
        env = _DriftsAndCancels(cancels=bool(built), tasks=TASKS)
        built.append(env)
        return env

    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
    )
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
    class _CancelsOnCloseAndVaries(_VaryingManifest):
        async def close(self) -> None:
            await super().close()
            raise asyncio.CancelledError()

    with pytest.raises(ValueError, match="different tool manifest") as refused:
        TaskStream(
            lambda _name: _CancelsOnCloseAndVaries(tasks=TASKS),
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
        )
    notes = getattr(refused.value, "__notes__", [])
    assert any("could not be closed" in note for note in notes), (
        "a cleanup that could not finish must be reported on the error being raised"
    )


# ----- the structural guard -----
#
# The same defect landed at several different handlers, which makes the handler the wrong unit
# to fix. The rule lives in one place — `_must_propagate` — and these two keep it from being
# bypassed: the first is behavioural, over every third-party surface the stream calls; the
# second refuses the handler shape that hid it, at the source.


def _stream_whose(surface: str, tmp_path: Path) -> Tuple[TaskStream, str, Dict[str, Any]]:
    """A stream in which exactly one third-party surface raises `CancelledError`, plus the tool
    that ends its task."""
    queue = [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)]
    prov = tmp_path / "prov"
    if surface == "an episode teardown":
        factory, _ = _cancelling_close_factory("episode")
        return TaskStream(factory, queue, prov_dir=prov), SUBMIT_TOOL, {"answer": "4"}
    if surface == "the catalog teardown":
        factory, _ = _cancelling_close_factory("catalog")
        return TaskStream(factory, queue, prov_dir=prov), SUBMIT_TOOL, {"answer": "4"}
    if surface == "a terminal step":
        return (
            TaskStream(
                _raises_on_the_terminal_step(asyncio.CancelledError), queue, prov_dir=prov
            ),
            TERMINATE_TOOL_NAME,
            {},
        )
    raise AssertionError(f"unknown surface {surface!r}")


@pytest.mark.parametrize("ended_by", ["the agent", "the drain"])
@pytest.mark.parametrize(
    "surface", ["an episode teardown", "the catalog teardown", "a terminal step"]
)
async def test_no_cancellation_an_env_raises_reaches_the_harness(
    tmp_path: Path, surface: str, ended_by: str
) -> None:
    # One statement over every third-party surface this module calls: whatever an env raises, the
    # harness's own API never answers with cancellation. It may raise the loud integrity error,
    # it may return normally — but a caller of `get_task`, `dispatch` or `aclose` is never
    # *cancelled* by code it was only supposed to be running.
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


def test_a_containment_boundary_may_not_catch_exception_and_leave_cancellation_to_chance() -> (
    None
):
    # The source-level half, because the behavioural half above can only cover the surfaces that
    # exist today. Any `try` that runs an env's code is a containment boundary, and
    # `except Exception` there is silently *not* a decision about `CancelledError` — which is
    # exactly how this landed at several handlers at once. So the shape is refused: such a
    # boundary catches `BaseException` and asks `_must_propagate`, or it names
    # `asyncio.CancelledError` itself. Elsewhere `except Exception` keeps its meaning, a
    # deliberate passthrough of the caller's own cancellation, and is left alone.
    #
    # A tripwire, not a proof: it keys on the names this module gives its third-party handles, so
    # renaming one would slip past. It fails at the moment the shape is written, which a comment
    # cannot do.
    handles = frozenset({"env", "episode"})
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
        "these boundaries run an env's code and catch `Exception`, so a `CancelledError` it "
        "raises escapes a handler meant to contain everything: " + "; ".join(offenders)
    )
