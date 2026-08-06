"""The feedback regime: what a terminating call tells the agent, and what the record says it told.

Two policies ship. :class:`Never` is the default and is today's behaviour byte for byte — the
fixed payload, no verdict anywhere on the agent-visible surface. :class:`Immediate` opens exactly
one channel: the sealed task's own published episode-level feedback, on the call that ended it.

The property these tests exist for is that ``Immediate`` opens *only* that channel. A response
that also said which task this was, what the queue had left, or that the stream had stopped would
be the same leak the redaction closes, arrived at through the door the training regime needs
open. So the adversarial half of this file asks, of every answer: could an agent read anything
here about a task other than the one it just ended?

:class:`EvalStream` is the other half — the default made structural, so that "this run was
evaluation-grade" is a claim about a construction and about a stamp on every row, rather than
about a value someone could have edited.

The other half of that claim is about the *record*, and it is what the last two sections here are
for. A stamp on every row is worth what the reader can conclude from it, so two things may never
happen: a stream that reveals while its rows say it did not, and two streams writing one record.
Neither is prevented by anything a row can say about itself — the first is a policy asserting its
own regime, the second is two runs each honestly stamping their own — so both are refused at the
construction site, and these tests are the adversary that goes looking for the way through.
"""

from __future__ import annotations

import asyncio
import copy
import json
import os
import pickle
import subprocess
import sys
import textwrap
import threading
import time
from abc import ABCMeta
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence

import pytest
from fastmcp import Client

from hgym.serve import stream as stream_module
from hgym.serve.stream import (
    EvalStream,
    FeedbackPolicy,
    Immediate,
    Never,
    ResultRow,
    TaskRef,
    TaskStream,
    build_stream_server,
    read_dispenses,
    read_results,
    reconcile,
)
from hgym.types import EpisodeFeedback, FeedbackCollection, InferenceFeedback
from tests._fixtures.choice_env import _FixtureChoiceEnv
from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [
    {"id": "q0", "question": "2+2?", "answer": "4"},
    {"id": "q1", "question": "3+3?", "answer": "6"},
    {"id": "q2", "question": "5+5?", "answer": "10"},
]

CHOICE_TASKS = [{"id": "c0", "choice": 7}]

ANSWERS = "answers"
CHOICES = "choices"

# The wire members a `ResultRow` carried before a regime was recorded on it. Written out rather
# than derived, because this is the contract an existing reader was written against: the regime
# has to be *additive* to it, and a test that computed this from the current row could not see a
# member being renamed or dropped underneath one.
_ROW_MEMBERS_BEFORE_THE_REGIME = {
    "seq",
    "lease",
    "position",
    "env",
    "task_idx",
    "closure",
    "score",
    "observed",
    "diagnostic",
    "extensions",
}


def _env_for(_name: str) -> _FixtureScoreEnv:
    return _FixtureScoreEnv(tasks=TASKS)


def _stream(tmp_path: Path, indices: List[int], **kwargs: Any) -> TaskStream:
    return TaskStream(
        _env_for,
        [TaskRef(ENV_NAME, i) for i in indices],
        prov_dir=tmp_path / "prov",
        **kwargs,
    )


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


def _text(result: Any) -> str:
    return result.content[0].text


def _rows(tmp_path: Path) -> List[Dict[str, Any]]:
    path = tmp_path / "prov" / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _episode_level(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(item) for item in items if item.get("level") == "episode"]


def _unwritable_prov(tmp_path: Path) -> None:
    """Make the real append fail without reaching inside the stream — the same directory trick
    the durability tests use."""
    (tmp_path / "prov" / "results.jsonl").mkdir(parents=True)


# ----- Never is today's behaviour -----


async def test_the_default_is_never_and_answers_exactly_as_it_did(tmp_path: Path) -> None:
    # The default is not merely "safe", it is the same bytes: a stream constructed without the
    # argument and one handed `Never()` answer a terminating call identically, and that answer is
    # the three-member payload every existing caller already parses. A default that had drifted
    # would silently change what every run that never mentions feedback tells its agent.
    async with _stream(tmp_path / "default", [0]) as implicit:
        await implicit.get_task()
        default = _text(await implicit.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    async with _stream(tmp_path / "explicit", [0], feedback=Never()) as named:
        await named.get_task()
        explicit = _text(await named.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert default == explicit
    assert set(json.loads(default)) == {"content", "terminated", "hint"}
    assert "correct" not in default
    # ...and the task really was graded, so this is not vacuous.
    assert implicit.results[0].score is not None
    assert implicit.results[0].score.success is True
    assert implicit.feedback == Never()


async def test_never_answers_the_same_bytes_however_the_task_went(tmp_path: Path) -> None:
    # The invariant the default exists for, restated where the policy could have broken it: a
    # right answer and a wrong one on the SAME task index come back indistinguishable.
    seen: List[str] = []
    async with _stream(tmp_path, [2, 2], feedback=Never()) as stream:
        for answer in ("not the answer", "10"):
            await stream.get_task()
            seen.append(_text(await stream.dispatch(SUBMIT_TOOL, {"answer": answer})))
    assert seen[0] == seen[1]
    assert [row.score.success for row in stream.results if row.score] == [False, True]


# ----- Immediate reveals the sealed row's own feedback, and only that -----


async def test_immediate_returns_the_feedback_the_row_records(tmp_path: Path) -> None:
    # The channel, doing its job: what comes back is the env's published episode-level items,
    # verbatim, and it is the same list the file holds under `observed`. "The same values
    # results.jsonl records" is the whole contract — not a re-derivation, not a summary.
    async with _stream(tmp_path, [2, 2], feedback=Immediate()) as stream:
        answers: List[Dict[str, Any]] = []
        for answer in ("not the answer", "10"):
            await stream.get_task()
            answers.append(_payload(await stream.dispatch(SUBMIT_TOOL, {"answer": answer})))

    wrong, right = answers
    assert wrong["feedback"] == [{"name": "correct", "value": False, "level": "episode"}]
    assert right["feedback"] == [{"name": "correct", "value": True, "level": "episode"}]
    # Byte-for-byte the record's own values, read back off the file rather than off the object.
    recorded = [_episode_level(row["observed"]) for row in _rows(tmp_path)]
    assert [answer["feedback"] for answer in answers] == recorded


async def test_immediate_keeps_the_envelope_the_redacted_answer_uses(tmp_path: Path) -> None:
    # A policy may add one member and change nothing else. The envelope is where a leak would be
    # cheapest to add and hardest to notice, so it is pinned against the redacted constant.
    async with _stream(tmp_path / "silent", [0]) as quiet:
        await quiet.get_task()
        redacted = _payload(await quiet.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    async with _stream(tmp_path / "loud", [0], feedback=Immediate()) as loud:
        await loud.get_task()
        revealed = _payload(await loud.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert set(revealed) == set(redacted) | {"feedback"}
    assert {key: revealed[key] for key in redacted} == redacted


async def test_immediate_reveals_nothing_the_stream_knows(tmp_path: Path) -> None:
    # The row the agent's feedback is read off also carries the lease, the queue position, the
    # task index, the closure and the run's diagnostic. None of them is feedback, and a response
    # composed from "the row" rather than from "the row's published items" would carry all of
    # them. Checked against the recorded row so the test names real values, not guesses.
    async with _stream(tmp_path, [1], feedback=Immediate()) as stream:
        await stream.get_task()
        seen = _text(await stream.dispatch(SUBMIT_TOOL, {"answer": "6"}))

    row = stream.results[0]
    assert row.lease and row.lease not in seen
    for leaked in ("lease", "position", "task_idx", "seq", "closure", "diagnostic", "extensions"):
        assert leaked not in seen, f"{leaked!r} reached the caller: {seen}"
    # The env key is on the row too, and the framing already named it — but the stream still does
    # not repeat it here, because this member is the env's feedback and nothing else.
    assert set(json.loads(seen)) == {"content", "terminated", "hint", "feedback"}


async def test_immediate_adds_no_summary_of_its_own(tmp_path: Path) -> None:
    # `Score.reward`/`Score.success` are this record's *reading* of the published items, not
    # values an env emitted. Putting them on the wire would send a number no env published, and
    # would make the response depend on whether that reading succeeded — a second channel, whose
    # presence would say the summary was readable.
    class _NamesNeitherSummary(_FixtureScoreEnv):
        """Publishes a verdict under a name the summary never reads."""

        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = FeedbackCollection()
            if terminated:
                fb.episode.append(EpisodeFeedback(name="graded", value=0.5))
            return fb

    stream = TaskStream(
        lambda _name: _NamesNeitherSummary(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        feedback=Immediate(),
    )
    async with stream:
        await stream.get_task()
        payload = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert payload["feedback"] == [{"name": "graded", "value": 0.5, "level": "episode"}]
    row = stream.results[0]
    assert row.score is not None and row.score.reward is None and row.score.success is None
    assert "reward" not in json.dumps(payload) and "success" not in json.dumps(payload)


async def test_immediate_never_reveals_inference_level_feedback(tmp_path: Path) -> None:
    # `wire.select_inband` already draws this line for a single served episode: episode-level
    # feedback rides out on the terminal, per-step feedback is recorded and not surfaced, because
    # silent dense shaping invalidates a comparison. A stream that revealed step feedback at its
    # terminal would be looser than the episode it is made of. The item is still on the row.
    class _AlsoShapesPerStep(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
            fb.inference.append(
                InferenceFeedback(name="step_shaping", value=0.25, step=len(trajectory))
            )
            return fb

    stream = TaskStream(
        lambda _name: _AlsoShapesPerStep(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        feedback=Immediate(),
    )
    async with stream:
        await stream.get_task()
        payload = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert payload["feedback"] == [{"name": "correct", "value": True, "level": "episode"}]
    assert "step_shaping" not in json.dumps(payload)
    # Recorded, though — the row keeps every item at the level the env published it.
    levels = {item["name"]: item["level"] for item in stream.results[0].observed}
    assert levels == {"correct": "episode", "step_shaping": "inference"}


async def test_immediate_reveals_an_empty_list_when_an_env_publishes_nothing(
    tmp_path: Path,
) -> None:
    # The member is a property of the policy, never of the task: present always, empty when there
    # is nothing to say. A member that appeared only when there was something would make its
    # absence the signal.
    class _PublishesNothing(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            return FeedbackCollection()

    stream = TaskStream(
        lambda _name: _PublishesNothing(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        feedback=Immediate(),
    )
    async with stream:
        await stream.get_task()
        payload = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert payload["feedback"] == []
    assert set(payload) == {"content", "terminated", "hint", "feedback"}


# ----- the adversarial half: only the sealed task's own feedback -----


async def test_immediate_answers_a_lease_with_its_own_tasks_feedback(tmp_path: Path) -> None:
    # The question concurrency asks: with two episodes live, is the feedback on a terminating
    # call the feedback of the task that CALL ended, or merely the last outcome the run recorded?
    # The two come apart here on purpose. The first task's row is appended and then its seal
    # blocks in the episode's teardown — the tail of the seal, after the durable write — so the
    # second task's row lands in between. A response composed from "the run's latest row" would
    # hand the first agent the second task's verdict.
    closing = asyncio.Event()
    release = asyncio.Event()

    class _SlowToLetGo(_FixtureScoreEnv):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._holds = False

        def _load_task(self, task_idx: Any) -> Dict[str, Any]:
            task = super()._load_task(task_idx)
            self._holds = task["task_idx"] == 0
            return task

        async def close(self) -> None:
            if self._holds:
                closing.set()
                await release.wait()
            await super().close()

    stream = TaskStream(
        lambda _name: _SlowToLetGo(tasks=TASKS),
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
        max_in_flight=2,
        feedback=Immediate(),
    )
    async with stream:
        first = await stream.get_task()
        second = await stream.get_task()
        assert first is not None and second is not None

        # The first agent answers correctly; its row lands and its seal then stalls letting go.
        held = asyncio.ensure_future(
            stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease})
        )
        await closing.wait()
        # The second answers wrongly, and is fully recorded while the first is still sealing.
        sibling = _payload(
            await stream.dispatch(SUBMIT_TOOL, {"answer": "not it", "lease": second.lease})
        )
        assert [row.task_idx for row in stream.results] == [0, 1]
        release.set()
        own = _payload(await held)

    assert own["feedback"] == [{"name": "correct", "value": True, "level": "episode"}], (
        "a task was answered with a sibling's verdict"
    )
    assert sibling["feedback"] == [{"name": "correct", "value": False, "level": "episode"}]


async def test_immediate_never_crosses_envs(tmp_path: Path) -> None:
    # The same question across envs, where a mix-up is legible in the names themselves: the score
    # env publishes `correct`, the choice env publishes `success` and `reward`. Two leases live at
    # once, one per env, and each terminating call is answered out of its own env's episode.
    def factory(name: str) -> Any:
        if name == ANSWERS:
            return _FixtureScoreEnv(tasks=TASKS)
        if name == CHOICES:
            return _FixtureChoiceEnv(tasks=CHOICE_TASKS)
        raise AssertionError(name)

    stream = TaskStream(
        factory,
        [TaskRef(ANSWERS, 0), TaskRef(CHOICES, 0)],
        prov_dir=tmp_path / "prov",
        max_in_flight=2,
        feedback=Immediate(),
    )
    async with stream:
        answering = await stream.get_task()
        choosing = await stream.get_task()
        assert answering is not None and choosing is not None
        chose = _payload(
            await stream.dispatch(
                f"{CHOICES}__{SUBMIT_TOOL}", {"choice": 7, "lease": choosing.lease}
            )
        )
        answered = _payload(
            await stream.dispatch(
                f"{ANSWERS}__{SUBMIT_TOOL}", {"answer": "4", "lease": answering.lease}
            )
        )

    assert [item["name"] for item in answered["feedback"]] == ["correct"]
    assert [item["name"] for item in chose["feedback"]] == ["success", "reward"]
    assert chose["feedback"] == [
        {"name": "success", "value": True, "level": "episode"},
        {"name": "reward", "value": 1.0, "level": "episode"},
    ]


async def test_immediate_says_nothing_about_the_queue(tmp_path: Path) -> None:
    # Three plays of ONE index, so the tasks are identical and only the queue differs between
    # them: first of three, second, last. The three answers are byte-identical, which is the
    # strongest available statement that no count, position or remaining-ness rides along.
    #
    # It is also, deliberately, a demonstration of what `Immediate` costs: a queue that repeats
    # an index answers the repeat the same way it answered the original, which is exactly the
    # correlation `Never` exists to deny an evaluation.
    async with _stream(tmp_path, [2, 2, 2], feedback=Immediate()) as stream:
        seen = []
        for _ in range(3):
            await stream.get_task()
            seen.append(_text(await stream.dispatch(SUBMIT_TOOL, {"answer": "10"})))
    assert seen[0] == seen[1] == seen[2]
    for absent in ("remaining", "consumed", "in_flight", "queue"):
        assert absent not in seen[0]


async def test_a_stop_is_not_readable_off_a_terminating_call(tmp_path: Path) -> None:
    # The failure this must not become: a seal whose row cannot be written stops the stream, and
    # the call that triggered it is the one call that knows. Under `Never` that answer is the
    # constant. Under `Immediate` it has to stay in the same shape — same members, no code, no
    # message — and the only honest content is nothing, because no row was recorded to reveal.
    # Pinned against a *clean* task that published nothing, so the two are one answer.
    class _PublishesNothing(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            return FeedbackCollection()

    quiet = TaskStream(
        lambda _name: _PublishesNothing(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "quiet",
        feedback=Immediate(),
    )
    async with quiet:
        await quiet.get_task()
        clean = _text(await quiet.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    stream = _stream(tmp_path, [0, 1], feedback=Immediate())
    _unwritable_prov(tmp_path)
    await stream.get_task()
    stopped = _text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    assert stream.stopped
    with pytest.raises(RuntimeError, match="record is incomplete"):
        await stream.aclose()

    assert stopped == clean, "the stop is readable off the answer to a terminating call"
    assert stream.results == ()


async def test_get_task_composes_the_same_answer_under_both_policies(tmp_path: Path) -> None:
    # The stop redaction lives on `get_task`, not on the terminal, and it may not vary with the
    # feedback policy: a stopped stream answers with the payload an exhausted queue gives, and
    # `Immediate` adds nothing to it — feedback rides the call that ended a task, and a question
    # about the queue ended none.
    async def answers(policy: Any, name: str) -> Dict[str, Any]:
        stream = _stream(tmp_path / name, [0, 1], feedback=policy)
        _unwritable_prov(tmp_path / name)
        server = build_stream_server(stream)
        async with Client(server) as client:
            await client.call_tool("get_task", {})
            await client.call_tool(SUBMIT_TOOL, {"answer": "4"})
            out = await client.call_tool("get_task", {})
        assert stream.stopped
        with pytest.raises(RuntimeError, match="record is incomplete"):
            await stream.aclose()
        return json.loads(out.content[0].text)  # type: ignore[union-attr]

    quiet = await answers(Never(), "never")
    loud = await answers(Immediate(), "immediate")
    assert quiet == loud
    assert set(loud) == {"done", "remaining", "consumed", "in_flight"}
    assert loud["done"] is True and loud["remaining"] == 0


async def test_a_task_the_stream_ended_reveals_nothing(tmp_path: Path) -> None:
    # `Immediate` answers the call that ended a task. A task the STREAM ended — displaced by the
    # next pull, drained at close, or timed out — has no such call, and its feedback may not be
    # smuggled onto the next `get_task`: that would be a verdict arriving on a question about the
    # queue, and it would be the previous task's verdict beside the next task's framing.
    async with _stream(tmp_path, [0, 1], feedback=Immediate()) as stream:
        server = build_stream_server(stream)
        async with Client(server) as client:
            first = json.loads(
                (await client.call_tool("get_task", {})).content[0].text  # type: ignore[union-attr]
            )
            # No terminal: pulling again displaces and force-scores the first task.
            second = json.loads(
                (await client.call_tool("get_task", {})).content[0].text  # type: ignore[union-attr]
            )
    assert "feedback" not in first and "feedback" not in second
    assert set(second) == {"env", "instructions", "budget", "tools"}
    # The displaced task was scored, and only the harness has it.
    assert [row.closure for row in stream.results] == ["drained", "drained"]
    assert stream.results[0].score is not None


async def test_a_call_that_ended_nothing_is_told_only_that_the_task_is_over(
    tmp_path: Path,
) -> None:
    # The other side of the same boundary, and the one a *call* can reach. The reveal belongs to
    # the call that sealed the task; every call that arrives after the seal is answered with the
    # episode's tombstone, which is `terminated` too and ended nothing. Composed from the recorded
    # row, an ordinary `noop` racing an accepted `submit` is handed the verdict its sibling
    # earned — a second recipient of a task's feedback, on a call that did not ask for it and did
    # not end anything. Under `Never` both answers are the same constant either way, which is the
    # whole reason this survived until a revealing policy existed.
    #
    # What the tombstone gets instead is the envelope with the member EMPTY, not the envelope
    # without the member. Absence would be a new signal: under a revealing policy the member is
    # always present, so a response missing it would say "you were not the call that ended this"
    # in a shape no policy chose — and it is not a shape either policy can otherwise produce. The
    # empty list says the same nothing as the three answers that already reveal nothing, which is
    # pinned below against one of them, byte for byte.
    entered = asyncio.Event()
    release = asyncio.Event()

    class _SlowFinalize(_FixtureScoreEnv):
        async def finalize(self, req: Any) -> Any:
            entered.set()
            await release.wait()
            return await super().finalize(req)

    async def raced(policy: Any, name: str) -> List[str]:
        entered.clear()
        release.clear()
        stream = TaskStream(
            lambda _name: _SlowFinalize(tasks=TASKS),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=tmp_path / name / "prov",
            feedback=policy,
            provenance_timeout=None,
        )
        # Entered by hand so the finalizer is released whatever an assertion below does: a task
        # left blocked in its env would wedge the drain rather than report anything.
        await stream.__aenter__()
        try:
            await stream.get_task()
            ending = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
            await entered.wait()  # sealed, and its finalizer is still grading
            tombstoned = asyncio.ensure_future(stream.dispatch("noop", {}))
            await asyncio.sleep(0.05)  # the `noop` is tombstoned, with no verdict to wait for
            # It waits all the same: the seal is joined whether or not its row is this call's to
            # hear about, because "the stream recorded the outcome" may not be said before the
            # outcome is durable, and a seal that failed needs every joiner to report it.
            waited = not tombstoned.done()
            release.set()
            answers = await asyncio.gather(ending, tombstoned)
        finally:
            release.set()
            await stream.aclose()
        assert waited, "a tombstoned call answered ahead of the seal it joined"
        # The tombstone changed nothing about the record: one row, sealed and scored by the call
        # that earned it.
        assert [row.closure for row in stream.results] == ["sealed"]
        assert stream.results[0].score is not None and stream.results[0].score.success is True
        return [_text(answer) for answer in answers]

    earned, unearned = await raced(Immediate(), "immediate")
    assert json.loads(earned)["feedback"] == [
        {"name": "correct", "value": True, "level": "episode"}
    ]
    assert json.loads(unearned)["feedback"] == [], (
        "a call that ended nothing was answered with the task's verdict"
    )
    assert set(json.loads(unearned)) == {"content", "terminated", "hint", "feedback"}

    # Nothing new is readable off it: those are the bytes a revealing run already sends whenever
    # there is nothing to reveal, so the tombstone adds no answer the response space did not hold.
    class _PublishesNothing(_FixtureScoreEnv):
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            return FeedbackCollection()

    quiet = TaskStream(
        lambda _name: _PublishesNothing(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "quiet" / "prov",
        feedback=Immediate(),
    )
    async with quiet:
        await quiet.get_task()
        silent = _text(await quiet.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    assert unearned == silent

    # And `Never` is untouched: both callers still get the one constant, so the pair of responses
    # a tombstoned caller can compare across the two policies says only what the policies say.
    assert await raced(Never(), "never") == [stream_module._TASK_OVER] * 2


# ----- the regime, stamped into the record -----


async def test_every_row_and_dispense_records_the_regime(tmp_path: Path) -> None:
    # "Evaluation-grade" is a claim about how a row was produced, so it is written on the row.
    # And on the dispense, before the task goes out — the only record a crash leaves behind.
    for policy, regime in ((Never(), "never"), (Immediate(), "immediate")):
        root = tmp_path / regime
        async with _stream(root, [0, 1], feedback=policy) as stream:
            for answer in ("4", "6"):
                await stream.get_task()
                await stream.dispatch(SUBMIT_TOOL, {"answer": answer})
        assert [row["feedback_regime"] for row in _rows(root)] == [regime, regime]
        assert [row.feedback_regime for row in stream.results] == [regime, regime]
        assert [
            record["feedback_regime"] for record in read_dispenses(root / "prov")
        ] == [regime, regime]
        assert [row.feedback_regime for row in read_results(root / "prov")] == [regime, regime]


async def test_an_abandoned_dispense_reconciles_under_its_own_regime(tmp_path: Path) -> None:
    # The row a crash produces is built from the dispense alone. Defaulting its regime would make
    # every abandoned task of a practice run read back as evaluation-grade — a row that looks
    # MORE trustworthy than the run that produced it, which is the direction that matters.
    stream = _stream(tmp_path, [0], feedback=Immediate())
    dispensed = await stream.get_task()
    assert dispensed is not None  # dispensed, never sealed: the process "dies" here
    abandoned = reconcile(tmp_path / "prov")
    assert [(row.closure, row.feedback_regime) for row in abandoned] == [
        ("broker_abort", "immediate")
    ]
    await stream.aclose()


def test_a_row_written_before_the_regime_existed_reads_as_never() -> None:
    # The reader idiom, from the other side: a row with no stamp came from a stream that revealed
    # nothing, because there was no other kind. Absent must therefore read as `never` and not as
    # unknown, or every archived run becomes unclassifiable.
    old = {
        "seq": 1,
        "lease": "a" * 32,
        "position": 0,
        "env": ENV_NAME,
        "task_idx": 0,
        "closure": "sealed",
        "score": None,
        "observed": [],
        "diagnostic": None,
        "extensions": {},
    }
    assert set(old) == _ROW_MEMBERS_BEFORE_THE_REGIME
    assert ResultRow.from_wire(old).feedback_regime == "never"
    assert old.get("feedback_regime", "never") == "never"  # the documented idiom


def test_the_record_type_and_the_wire_agree_about_an_unstamped_row() -> None:
    # The default on :class:`ResultRow` itself, which `from_wire` never exercises because it
    # always passes the member. The two have to be the same value or a row stops round-tripping:
    # a row built in memory without a regime would serialise as one posture and read back as the
    # other, and the object and the file would disagree about the one thing the stamp is for.
    bare = ResultRow(
        seq=1,
        lease="a" * 32,
        position=0,
        env=ENV_NAME,
        task_idx=0,
        closure="sealed",
        score=None,
    )
    assert bare.feedback_regime == "never"
    assert bare.to_wire()["feedback_regime"] == "never"
    assert ResultRow.from_wire(bare.to_wire()) == bare
    # ...and a stored row from before the member existed is that same row, not a different one.
    unstamped = {key: value for key, value in bare.to_wire().items() if key != "feedback_regime"}
    assert ResultRow.from_wire(unstamped) == bare


async def test_the_regime_is_additive_on_the_wire(tmp_path: Path) -> None:
    # Wire compatibility: a `Never` row is still the row an existing reader parses, plus one
    # member. Anything renamed or dropped here breaks a consumer that never asked for feedback.
    async with _stream(tmp_path, [0]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    wire = stream.results[0].to_wire()
    assert set(wire) == _ROW_MEMBERS_BEFORE_THE_REGIME | {"feedback_regime"}
    assert wire["feedback_regime"] == "never"
    assert _rows(tmp_path) == [wire]


async def test_a_record_may_not_mix_regimes(tmp_path: Path) -> None:
    # One record is read as one run: its rows are averaged together, and whether the agent was
    # told each verdict is what decides which claim that mean supports. So a resume that would
    # append rows of the other posture is refused at construction, before anything is spent.
    async with _stream(tmp_path, [0, 1], feedback=Immediate()) as first:
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "4"})

    with pytest.raises(ValueError, match="feedback regime 'immediate'"):
        _stream(tmp_path, [0, 1], resume=True)
    with pytest.raises(ValueError, match="feedback regime 'immediate'"):
        EvalStream(
            _env_for,
            [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
            prov_dir=tmp_path / "prov",
            resume=True,
        )
    # Resuming under the regime the record was written with is exactly what a crashed run needs.
    async with _stream(tmp_path, [0, 1], resume=True, feedback=Immediate()) as resumed:
        await resumed.get_task()
        await resumed.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert [row.position for row in resumed.results] == [1]
    assert {row["feedback_regime"] for row in _rows(tmp_path)} == {"immediate"}


async def test_a_dispense_alone_is_enough_to_refuse_a_mixed_resume(tmp_path: Path) -> None:
    # The dispense is stamped before the task is handed out, so a run that crashed before its
    # first row still cannot be continued under the other posture — the half of the record that
    # survives a kill is the half that carries the regime.
    stream = _stream(tmp_path, [0, 1], feedback=Immediate())
    assert await stream.get_task() is not None
    await stream.aclose()
    (tmp_path / "prov" / "results.jsonl").unlink()  # the row never made it to disk
    with pytest.raises(ValueError, match="a dispense record written under feedback regime"):
        _stream(tmp_path, [0, 1], resume=True)


# ----- refusals -----


@pytest.mark.parametrize("bad", [True, "immediate", None, 1, Immediate])
async def test_a_feedback_argument_that_is_not_a_policy_is_refused(
    tmp_path: Path, bad: Any
) -> None:
    # An allow-list, not a duck-type. What this argument decides is whether the agent is told its
    # verdict, and a value that merely looks usable would decide it by whatever it answers — with
    # the answer already on the wire by the time anyone noticed.
    with pytest.raises(ValueError, match="feedback must be Never\\(\\) or Immediate\\(\\)"):
        _stream(tmp_path, [0], feedback=bad)


class _Liar(FeedbackPolicy):
    """A policy that stamps the regime with no channel and then opens one."""

    regime: ClassVar[str] = "never"
    reveals: ClassVar[bool] = True

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return published


async def test_a_policy_that_would_reveal_under_another_regimes_name_is_refused(
    tmp_path: Path,
) -> None:
    # THE defect this allow-list exists for. `regime` and `reveals` are not two independent facts
    # a policy gets to assert about itself: a subclass naming itself `never` while revealing gets
    # a run whose terminals carry the verdict and whose every dispense record and result row say
    # no channel was open — a practice row that reads as evaluation-grade, with the documented
    # reader check `row.get("feedback_regime", "never")` agreeing. Nothing downstream can catch
    # it, because both halves of the record are internally consistent; the only place it can be
    # caught is the construction site, and only by refusing to take the claim on trust.
    with pytest.raises(ValueError, match="feedback must be Never\\(\\) or Immediate\\(\\)") as bad:
        _stream(tmp_path, [0], feedback=_Liar())
    # ...and the refusal says how a real new policy gets in, because the alternative reading of
    # this error is "policies are closed", which is not what it means.
    message = str(bad.value)
    assert "added to this module" in message
    assert "not by subclassing" in message
    # Refused before anything was spent: no record, and nothing claimed.
    assert not (tmp_path / "prov").exists()


async def test_a_shipped_policy_is_served_as_itself_whatever_its_instance_says(
    tmp_path: Path,
) -> None:
    # The hole an exact-type check does not close by itself, and the reason the regime is read
    # from the module rather than from the object. `Never` is a frozen dataclass whose `regime`
    # and `reveals` are ClassVars — no constructor argument, and `never.reveals = True` raises —
    # but frozen is not sealed: the instance has a `__dict__`, and `object.__setattr__` shadows
    # the class attribute on it with no subclass anywhere. A stream that admitted the exact type
    # and then trusted the instance would reveal under a record stamped `never`, exactly as the
    # subclass above would have.
    liar = Never()
    object.__setattr__(liar, "reveals", True)
    assert liar.reveals is True and type(liar) is Never  # the shadowing really took

    async with _stream(tmp_path, [0], feedback=liar) as stream:
        await stream.get_task()
        answer = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    # Served as the `Never` it is: the fixed payload, no feedback member, an honest stamp.
    assert set(answer) == {"content", "terminated", "hint"}
    assert [row["feedback_regime"] for row in _rows(tmp_path)] == ["never"]
    assert stream.results[0].score is not None  # the task really was graded


async def test_a_shipped_policy_cannot_supply_the_behaviour_its_regime_names(
    tmp_path: Path,
) -> None:
    # The same shadowing, aimed at the method instead of the class variables — and this is the
    # half that reaches the agent. A table that supplied `regime` and `reveals` but then called
    # `policy.reveal(...)` would look up a name on an object the caller owns, and an instance
    # dictionary is looked at first: an admitted `Immediate` answers the terminating call with
    # items no env published, and can add ones the row never held, while the dispense before it,
    # the row beside it and any resume after it all stamp `immediate`. Nothing in the artifact
    # records the signal the agent was actually shown, and the next run's scores were earned
    # under it. So the table supplies the behaviour the regime is a name for.
    policy = Immediate()
    object.__setattr__(
        policy,
        "reveal",
        lambda published: [
            {"name": "correct", "value": False, "level": "episode"},
            {"name": "target", "value": "10", "level": "episode"},
        ],
    )
    assert policy.reveal([])  # the shadowing really took: the lookup finds the caller's function

    async with _stream(tmp_path, [0], feedback=policy) as stream:
        await stream.get_task()
        answer = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    # `immediate` means the sealed row's own episode-level items, verbatim: the answer is what
    # the file holds, it is not what the caller's function returns, and it invents nothing.
    row = _rows(tmp_path)[0]
    assert answer["feedback"] == _episode_level(row["observed"])
    assert answer["feedback"] == [{"name": "correct", "value": True, "level": "episode"}]
    assert set(answer) == {"content", "terminated", "hint", "feedback"}
    assert row["feedback_regime"] == "immediate"
    # Ignored, not refused, and nothing about the run is disturbed by it: a stream reads no
    # attribute of the object it was handed, so there is nothing here to detect or to report.
    assert not stream.stopped
    assert stream.results[0].score is not None


class _Wearing:
    """An object that answers ``isinstance`` and ``.__class__`` with :class:`Immediate`, and is
    not one."""

    @property  # type: ignore[misc]
    def __class__(self) -> type:  # type: ignore[override]
        return Immediate

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return [{"name": "correct", "value": False, "level": "episode"}]


class _Answering(ABCMeta):
    """A metaclass that answers every class attribute lookup out of :class:`Immediate`."""

    def __getattr__(cls, name: str) -> Any:
        return getattr(Immediate, name)


class _Costume(FeedbackPolicy, metaclass=_Answering):
    """A policy whose *class* borrows `Immediate`'s regime, reveal flag and method."""

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return [{"name": "correct", "value": False, "level": "episode"}]


@pytest.mark.parametrize("costume", [_Wearing, _Costume])
async def test_an_object_that_answers_like_a_policy_is_still_refused(
    tmp_path: Path, costume: type
) -> None:
    # The two ways an object can *say* it is `Immediate` without being one, refused where every
    # other impostor is. `_Wearing.__class__` is a property, so `isinstance` says yes and a check
    # written with it would admit the object whole — its own `reveal` and all. `_Costume` hooks
    # the lookup one level up: its metaclass answers `regime`, `reveals` and `reveal` out of
    # `Immediate`, so every class-level read of it agrees with the real policy. `type(x) is C`
    # consults neither, which is why the admission is spelled that way and why reading the
    # behaviour off the admitted *class* is safe.
    impostor = costume()
    assert isinstance(impostor, FeedbackPolicy)  # both pass the check this module does not make
    with pytest.raises(ValueError, match="feedback must be Never\\(\\) or Immediate\\(\\)"):
        _stream(tmp_path, [0], feedback=impostor)
    assert not (tmp_path / "prov").exists()


async def test_the_policy_table_is_the_only_place_a_regime_is_named(tmp_path: Path) -> None:
    # The allow-list is what makes a regime name mean one policy, which is what the resume check
    # needs it to mean: two policies sharing a name would let a record be continued by a stream
    # that did not write it. Pinned here rather than in prose, because this is the invariant a
    # third entry has to preserve — and the table, not the class attribute, is what is served.
    admitted = {
        policy: (regime, reveals) for policy, regime, reveals, _reveal in stream_module._POLICIES
    }
    assert admitted == {Never: ("never", False), Immediate: ("immediate", True)}
    regimes = [regime for _policy, regime, _reveals, _reveal in stream_module._POLICIES]
    assert len(set(regimes)) == len(regimes)
    for policy, regime, reveals, reveal in stream_module._POLICIES:
        assert type(regime) is str and regime
        assert type(reveals) is bool
        # The class still declares them, and the table still agrees with the class — the table
        # exists so that an *instance* cannot disagree, not so the class can.
        assert (policy.regime, policy.reveals) == (regime, reveals)
        # The fourth member is the behaviour itself, snapshotted from the class: what a stream
        # invokes is this function, applied to whatever instance it was handed, so a policy added
        # later cannot be added as a name whose meaning is then supplied by the caller.
        assert reveal is policy.__dict__["reveal"]


def _admitting(monkeypatch: pytest.MonkeyPatch, policy: type, regime: str, reveals: bool) -> None:
    """Add a policy to the module's allow-list, which is how a real one is added too."""
    monkeypatch.setattr(
        stream_module,
        "_POLICIES",
        stream_module._POLICIES + ((policy, regime, reveals, policy.reveal),),
    )


class _Delayed(FeedbackPolicy):
    """A stand-in for the policies the design says arrive later: holds every verdict back."""

    regime: ClassVar[str] = "delayed"
    reveals: ClassVar[bool] = True

    def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
        return ()


async def test_a_new_policy_is_admitted_by_being_added_to_the_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The extension story, executed rather than promised. `Delayed(k)`, `Batched(n)` and
    # `Noisy(p)` are still policies rather than new surface — what the allow-list changes is where
    # one comes from, and this is the whole of what a third costs: a line in the table. The
    # envelope containment is unchanged by it, which is the reason the list can grow safely.
    _admitting(monkeypatch, _Delayed, "delayed", True)
    async with _stream(tmp_path, [0], feedback=_Delayed()) as stream:
        await stream.get_task()
        answer = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    # Revealing, so the member is there; holding, so it is empty — and the record says `delayed`,
    # never one of the two regimes it is not.
    assert answer["feedback"] == []
    assert set(answer) == {"content", "terminated", "hint", "feedback"}
    assert [row["feedback_regime"] for row in _rows(tmp_path)] == ["delayed"]
    assert read_dispenses(tmp_path / "prov")[0]["feedback_regime"] == "delayed"


async def test_a_policy_that_cannot_answer_stops_the_run_and_says_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Containment for the policies the allow-list will admit later: composing a revealing answer
    # runs the policy's own code, and one that raises will raise for every task — so the row
    # already stamped with its regime would be claiming a channel the agent was never told
    # through. The failure is contained (no traceback at the agent, no change of shape), and the
    # stream stops: loud to the harness, silent to the agent. Neither shipped policy can reach
    # this today, so the test admits one that can, the way a real one would be admitted.
    class _Exploding(FeedbackPolicy):
        regime: ClassVar[str] = "exploding"
        reveals: ClassVar[bool] = True

        def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
            raise RuntimeError("the policy could not be asked")

    class _Unsendable(FeedbackPolicy):
        regime: ClassVar[str] = "unsendable"
        reveals: ClassVar[bool] = True

        def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
            return [{"name": "nan", "value": float("nan"), "level": "episode"}]

    _admitting(monkeypatch, _Exploding, "exploding", True)
    _admitting(monkeypatch, _Unsendable, "unsendable", True)
    for policy, root in ((_Exploding(), "raises"), (_Unsendable(), "unsendable")):
        stream = _stream(tmp_path / root, [0, 1], feedback=policy)
        await stream.get_task()
        payload = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        assert payload["feedback"] == []
        assert set(payload) == {"content", "terminated", "hint", "feedback"}
        assert stream.stopped
        with pytest.raises(RuntimeError, match="feedback policy could not answer"):
            await stream.aclose()
        # The row the task earned is still on the record, under the regime it was served with.
        assert stream.results[0].score is not None
        assert stream.results[0].score.success is True
        assert stream.results[0].feedback_regime == policy.regime


async def test_a_policy_that_cannot_answer_is_reported_at_the_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The other half of "silent to the agent": the harness has to be able to find out, and the
    # place it finds out is the same one every integrity failure here is reported at.
    class _Exploding(FeedbackPolicy):
        regime: ClassVar[str] = "exploding"
        reveals: ClassVar[bool] = True

        def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
            raise RuntimeError("the policy could not be asked")

    _admitting(monkeypatch, _Exploding, "exploding", True)
    stream = _stream(tmp_path, [0, 1], feedback=_Exploding())
    await stream.get_task()
    await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    with pytest.raises(RuntimeError, match="feedback policy could not answer") as raised:
        await stream.aclose()
    assert "could not be asked" in str(raised.value)


# ----- EvalStream: a locked construction -----


async def test_eval_stream_refuses_a_feedback_argument(tmp_path: Path) -> None:
    # Including the right one. A value that has to be passed is a value the next edit can change,
    # and `EvalStream(..., feedback=Never())` would read as reviewed while inviting exactly that
    # edit. There is nothing to pass — that is the whole statement the class makes.
    for policy in (Immediate(), Never(), None, "never"):
        with pytest.raises(ValueError, match="takes no `feedback` policy") as raised:
            EvalStream(
                _env_for,
                [TaskRef(ENV_NAME, 0)],
                prov_dir=tmp_path / "prov",
                feedback=policy,
            )
        message = str(raised.value)
        assert "closed by construction" in message
        assert "TaskStream(..., feedback=...)" in message


async def test_eval_stream_serves_with_the_channel_closed_and_stamps_it(
    tmp_path: Path,
) -> None:
    # What it guarantees, end to end: the terminal answer is the redacted constant a `Never`
    # stream gives, and every record it writes says so — the reader's whole check.
    async with _stream(tmp_path / "control", [0]) as control:
        await control.get_task()
        redacted = _text(await control.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    stream = EvalStream(_env_for, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    async with stream:
        await stream.get_task()
        assert _text(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})) == redacted

    assert stream.feedback == Never()
    assert isinstance(stream, TaskStream)
    (row,) = _rows(tmp_path)
    assert row["feedback_regime"] == "never"
    assert row.get("feedback_regime", "never") == "never"  # the documented reader idiom
    assert read_dispenses(tmp_path / "prov")[0]["feedback_regime"] == "never"


async def test_eval_stream_passes_every_other_argument_through(tmp_path: Path) -> None:
    # The refused knob is the one that decides posture; the rest change how a queue is served,
    # not what the agent is told, and an evaluation that could not use them would be a worse
    # evaluation. Checked rather than asserted in prose: a mis-wired `super().__init__` would
    # otherwise silently drop a deadline or a concurrency setting.
    stream = EvalStream(
        _env_for,
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 1)],
        prov_dir=tmp_path / "prov",
        max_in_flight=2,
        deadline=30.0,
        provenance_timeout=5.0,
    )
    async with stream:
        first = await stream.get_task()
        second = await stream.get_task()
        assert first is not None and second is not None
        assert first.lease and second.lease and first.lease != second.lease
        assert stream.max_in_flight == 2
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4", "lease": first.lease})
    assert {row.feedback_regime for row in stream.results} == {"never"}


# ----- one record, one stream: the directory is owned, not merely unrecorded -----


def _claim(tmp_path: Path) -> Path:
    return tmp_path / "prov" / "claim.json"


async def test_two_fresh_streams_cannot_serve_one_directory(tmp_path: Path) -> None:
    # The other way a record ends up mixing regimes, and it needs no custom policy at all: the
    # freshness check reads what the directory already holds, so two streams that reach it before
    # either has written both pass it. Both then serve, and the file ends with two rows for queue
    # position 0 — one `never`, one `immediate`, both seq 1 — with no stop, no exception, and
    # nothing on either row saying the other exists. A consumer averages them as one run.
    first = _stream(tmp_path, [0], feedback=Never())
    with pytest.raises(ValueError, match="claimed by another stream") as refused:
        _stream(tmp_path, [0], feedback=Immediate())
    # The refusal is useful to the human who has to decide what to do about it: which regime the
    # holder serves under, and the one flag that overrides it.
    message = str(refused.value)
    assert "'never'" in message
    assert "resume=True" in message
    assert "fresh provenance directory" in message

    # ...and the one stream that did claim it serves normally, into a record of one regime.
    async with first:
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "4"})
    rows = _rows(tmp_path)
    assert [row["position"] for row in rows] == [0]
    assert {row["feedback_regime"] for row in rows} == {"never"}


async def test_a_run_that_finished_in_the_window_is_not_served_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The test above keeps its winner alive, so the loser meets a *claim*. The dangerous loser
    # meets a *record*: a check that reads the directory and a claim that installs ownership were
    # two operations, and a check that does not hold its exclusion across the claim it authorises
    # authorises nothing. A whole other fresh run can begin, seal and RELEASE its claim inside
    # that window, so the constructor that resumes finds no claim to lose to and installs its own
    # over a directory that now holds a complete run — an `immediate` row and a `never` row for
    # queue position 0, both seq 1, both streams closing normally.
    #
    # The window is opened at the module's own directory lock rather than at anything the fix
    # introduced, so this asks about the ordering and not about a shape: a constructor paused at
    # the door of that exclusion has done every unlocked thing it is going to do and has taken
    # nothing.
    locked = stream_module._locked
    opened: List[str] = []
    released: List[bool] = []

    def _winner() -> None:
        async def serve() -> None:
            stream = TaskStream(
                _env_for,
                [TaskRef(ENV_NAME, 0)],
                prov_dir=tmp_path / "prov",
                feedback=Immediate(),
            )
            async with stream:
                await stream.get_task()
                await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})

        asyncio.run(serve())

    def _paused(directory: Path) -> Any:
        if not opened:
            opened.append("held")
            # A whole other fresh run, start to finish, in the window: it claims the directory,
            # records position 0 under `immediate`, and lets the claim go on its way out. Its own
            # locking is the real one — the guard above fires once, for the paused constructor.
            thread = threading.Thread(target=_winner)
            thread.start()
            thread.join(60.0)
            released.append(not _claim(tmp_path).exists())
        return locked(directory)

    monkeypatch.setattr(stream_module, "_locked", _paused)
    with pytest.raises(ValueError, match="already holds records") as refused:
        EvalStream(_env_for, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    assert released == [True], "the winner still held the directory; this is the other test"
    assert "resume=True" in str(refused.value)

    # One run in the record, and the refused constructor left nothing of its own behind — no
    # claim for the next attempt to have to break, and not a row or a dispense under its regime.
    assert not _claim(tmp_path).exists()
    rows = _rows(tmp_path)
    assert [row["position"] for row in rows] == [0]
    assert {row["feedback_regime"] for row in rows} == {"immediate"}
    assert [record["feedback_regime"] for record in read_dispenses(tmp_path / "prov")] == [
        "immediate"
    ]


async def test_the_claim_is_taken_before_the_first_write_not_by_it(tmp_path: Path) -> None:
    # Where the ownership has to be decided: a claim taken at the first append would leave the
    # whole of construction — the env factory, the manifest validation, the replay — as a window
    # in which two streams both believe they own the directory. So it is taken at construction,
    # and the loser never reaches a factory call.
    stream = _stream(tmp_path, [0])
    assert _claim(tmp_path).is_file()
    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()

    factories: List[str] = []

    def _counting(name: str) -> _FixtureScoreEnv:
        factories.append(name)
        return _env_for(name)

    with pytest.raises(ValueError, match="claimed by another stream"):
        TaskStream(_counting, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    assert factories == []
    await stream.aclose()


async def test_an_orderly_close_leaves_the_directory_unclaimed(tmp_path: Path) -> None:
    # What makes `resume=True` an assertion about a *crash* rather than a routine incantation: a
    # run that finished releases the directory, so the only claim anyone is ever asked to break
    # belongs to a stream that really did stop without letting go.
    async with _stream(tmp_path, [0, 1]) as stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert not _claim(tmp_path).exists()
    # So a resume of a cleanly closed record breaks nothing: there is no claim to take over, and
    # `resume=True` goes back to meaning only what the record needs it to mean.
    async with _stream(tmp_path, [0, 1], resume=True) as resumed:
        assert _claim(tmp_path).is_file()
        await resumed.get_task()
        await resumed.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert not _claim(tmp_path).exists()
    assert [row["position"] for row in _rows(tmp_path)] == [0, 1]


async def test_a_crashed_run_is_resumed_into_its_own_directory(tmp_path: Path) -> None:
    # The constraint the ownership claim may not break. A crashed evaluation resumes into the
    # directory it crashed in, and the claim it left behind is exactly what makes that directory
    # look owned. `resume=True` is the override — the human asserting the prior stream is dead,
    # which is a fact no liveness oracle here could establish and the human already had to know
    # to be passing the flag at all.
    crashed = _stream(tmp_path, [0, 1], feedback=Immediate())
    await crashed.get_task()
    await crashed.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert _claim(tmp_path).is_file()  # no `aclose`: the claim outlives the stream

    async with _stream(tmp_path, [0, 1], resume=True, feedback=Immediate()) as resumed:
        await resumed.get_task()
        await resumed.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert [row.position for row in resumed.results] == [1]
    assert [row["position"] for row in _rows(tmp_path)] == [0, 1]
    assert {row["feedback_regime"] for row in _rows(tmp_path)} == {"immediate"}


async def test_the_claim_carries_the_regime_a_record_does_not_have_yet(tmp_path: Path) -> None:
    # The gap between the claim and the first dispense, which the record cannot cover. A run
    # killed in it wrote no dispense and no row, so the regime check — which reads records — has
    # nothing to compare against, and a resume under the other posture would be waved through
    # onto a file that starts empty and ends mixed. The claim is the only thing that names the
    # regime before any record does, so the same check reads it.
    _stream(tmp_path, [0, 1], feedback=Immediate())  # claimed, then killed
    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert not (tmp_path / "prov" / "results.jsonl").exists()

    with pytest.raises(ValueError, match="an ownership claim written under feedback regime"):
        _stream(tmp_path, [0, 1], resume=True, feedback=Never())
    # Resuming under the regime the claim names is what a crashed run actually needs.
    async with _stream(tmp_path, [0, 1], resume=True, feedback=Immediate()) as resumed:
        await resumed.get_task()
        await resumed.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert {row["feedback_regime"] for row in _rows(tmp_path)} == {"immediate"}


async def test_a_stream_that_lost_its_directory_writes_nothing_more(tmp_path: Path) -> None:
    # Ownership has to cover the run, not just its first instant. `resume=True` is a human
    # assertion and a human can be wrong about it — the crashed stream was still alive — so the
    # dispossessed stream has to find out. It finds out where a task would first become part of
    # this record, which is before the task is handed out: no dispense, no episode, no row.
    living = _stream(tmp_path, [0, 1])
    await living.get_task()
    await living.dispatch(SUBMIT_TOOL, {"answer": "4"})
    taker = _stream(tmp_path, [0, 1], resume=True)

    with pytest.raises(RuntimeError, match="no longer claimed by this stream"):
        await living.get_task()
    assert living.stopped
    with pytest.raises(RuntimeError, match="could not record a dispense") as drained:
        await living.aclose()
    assert "no longer claimed by this stream" in str(drained.value.__cause__)

    async with taker:
        await taker.get_task()
        await taker.dispatch(SUBMIT_TOOL, {"answer": "6"})
    # Exactly one row per position: the dispossessed stream added nothing after it was displaced.
    assert [row["position"] for row in _rows(tmp_path)] == [0, 1]


async def test_a_displaced_stream_cannot_seal_the_task_it_still_holds(tmp_path: Path) -> None:
    # A dispense is not the only thing a stream appends, and the *result* is the one that carries
    # a score. A takeover displaces a stream that has a task in flight: that task's position has
    # no row, so it is precisely the position the resuming stream replays, and both streams then
    # seal into one record. Checked only where a task is dispensed, the displaced stream's seal
    # went through — two scored rows for queue position 0, one wrong and one right, both honestly
    # stamped, neither stream stopped, both closes returning normally, and a consumer averaging
    # one queued task to 0.5 with nothing in the artifact saying so.
    displaced = _stream(tmp_path, [0])
    await displaced.get_task()
    taker = _stream(tmp_path, [0], resume=True)
    await taker.get_task()

    # The agent is answered, as it is for every integrity failure here — loud to the harness,
    # silent to the agent — and nothing at all is recorded for the task.
    answer = _payload(await displaced.dispatch(SUBMIT_TOOL, {"answer": "not the answer"}))
    assert set(answer) == {"content", "terminated", "hint"}
    assert displaced.stopped
    assert list(displaced.results) == []

    async with taker:
        await taker.dispatch(SUBMIT_TOOL, {"answer": "4"})
    rows = _rows(tmp_path)
    assert [row["position"] for row in rows] == [0]
    assert rows[0]["score"]["success"] is True

    # And the close reports it rather than succeeding quietly: a stream that could not record a
    # task it was handed is a run whose record is incomplete, whoever else owns the directory.
    with pytest.raises(RuntimeError, match="could not record every dispensed task") as drained:
        await displaced.aclose()
    assert "no longer claimed by this stream" in str(drained.value.__cause__)


def test_a_stream_may_not_be_shared_across_a_fork(tmp_path: Path) -> None:
    # Ownership is process-bound, and a token alone cannot make it so: the token is a value in
    # memory and `fork` copies memory, so both children hold a stream whose `owner` matches the
    # claim on disk. They also hold its queue position, its `seq` counter and its lease set — so
    # both passed every check, both served position 0, both exited cleanly, and the record ended
    # with two contradictory rows under one `seq` (and, since a record and its terminator are two
    # writes, with lines from the two processes interleaved into JSON that will not parse at all).
    # The claim records the pid that took it, and every append asks whether that pid is this one:
    # not whether the other process is alive, which is a question with no honest answer here, but
    # whether the process about to write is the one that owns the directory.
    #
    # Sync, and forking before any loop exists: a `TaskStream` is built by an ordinary
    # constructor, and forking a process with a running event loop would leave each child holding
    # the parent's loop state — a failure of its own, indistinguishable from the refusal under
    # test.
    stream = _stream(tmp_path, [0])
    verdicts = [tmp_path / "child-0", tmp_path / "child-1"]

    children = []
    for verdict, answer in zip(verdicts, ("4", "not the answer")):

        async def serve(answer: str = answer) -> None:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": answer})
            await stream.aclose()

        pid = os.fork()
        if pid == 0:  # pragma: no cover - the child never returns to the test runner
            outcome = "SERVED"
            try:
                asyncio.run(serve())
            except BaseException as exc:  # noqa: BLE001 - reported to the parent, not raised
                outcome = f"REFUSED {exc}"
            try:
                verdict.write_text(outcome, encoding="utf-8")
            finally:
                os._exit(0)
        children.append(pid)

    for pid in children:
        os.waitpid(pid, 0)
    outcomes = [verdict.read_text(encoding="utf-8") for verdict in verdicts]
    assert all(outcome.startswith("REFUSED") for outcome in outcomes), outcomes
    assert all("claimed by this stream in another process" in one for one in outcomes), outcomes
    # The refusal names what a caller has to do about it, because "you forked" is not advice.
    assert all("in the process that will serve it" in one for one in outcomes), outcomes
    # Nothing was written by either child: the refusal is before the append, not a repair after
    # it, so there is no dispense to reconcile and no row to disagree with another.
    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    # The parent still owns the directory a child could not take from it, and serves normally.
    assert json.loads(_claim(tmp_path).read_text(encoding="utf-8"))["pid"] == os.getpid()
    asyncio.run(_serve_one(stream))
    assert [row["position"] for row in _rows(tmp_path)] == [0]


async def _serve_one(stream: TaskStream) -> None:
    """One task through a stream that is already built, and an orderly close."""
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})


def _eval_stream(tmp_path: Path, indices: Sequence[int] = (0,), **kwargs: Any) -> EvalStream:
    """The exact class an evaluation is served by, not a `TaskStream` standing in for one. The
    refusals below live on the base, and "the base defines it" is not the property — "the object
    an evaluation actually holds refuses" is, and only the concrete class can say so."""
    return EvalStream(
        _env_for,
        [TaskRef(ENV_NAME, i) for i in indices],
        prov_dir=tmp_path / "prov",
        **kwargs,
    )


# A duplicate is a second stream with the first one's name. The `fork` refusal above closes the
# version that copies the *process*; these close the version that copies the *object*, which is
# strictly easier to reach — one stdlib call, no processes, nothing private touched — and lands in
# the worse place, since the pid the fork check relies on matches by construction inside one
# process. Reproduced before it was closed, on an exact `EvalStream`, with `copy.copy` and again
# by pickling: two objects, one claim, two scored rows for queue position 0 under `seq` 1, one
# `success=false` and one `success=true`, both stamped `never`, neither stream stopped and both
# closes clean — one queued task averaging to 0.5 with nothing in the artifact saying so.


async def test_a_stream_may_not_be_copied(tmp_path: Path) -> None:
    # The reproduction, refused at the line that made the second object. `copy.copy` duplicates
    # the ownership token, the pid it runs under is the same pid, and the queue position and `seq`
    # are plain integers that come across with it — so the copy passed `_holds_claim` and
    # `_append_owned` authorised its writes as honestly as it authorises the original's. Nothing
    # downstream can tell the two apart, so the refusal has to be here, before a second usable
    # object exists and before either of them has dispensed anything.
    stream = _eval_stream(tmp_path, max_in_flight=2)
    with pytest.raises(TypeError, match="copy.copy of this EvalStream is refused") as refused:
        copy.copy(stream)
    message = str(refused.value)
    assert "ownership identity, not a value" in message
    # The refusal names what a caller has to do instead, because "no" is not advice — and names
    # the sanctioned second stream, which takes a directory over deliberately and mints an
    # identity of its own rather than borrowing one.
    assert "Build the stream where it will serve" in message
    assert "resume=True" in message

    # Refused before any dispense: no task was handed out, so there is no record to reconcile and
    # nothing to tell apart. The claim is untouched — a refused copy costs the original nothing.
    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert _claim(tmp_path).is_file()

    # Holding a stream is not copying one: the shallow copy of a *container* aliases it, which is
    # the same object and the same identity, and is left alone. What is refused is making a second
    # stream, not putting one in a list.
    assert copy.copy([stream])[0] is stream
    assert copy.copy({"stream": stream})["stream"] is stream

    # And the stream the copy was taken of serves the queue it always would have, into a record
    # that holds one row for the one queued task — including across the moment a copy would do the
    # most damage, with a task dispensed and unsealed, where the duplicate would inherit the live
    # lease as well as the position.
    async with stream:
        dispensed = await stream.get_task()
        assert dispensed is not None
        with pytest.raises(TypeError, match="copy.copy of this EvalStream is refused"):
            copy.copy(stream)
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}, lease=dispensed.lease)
    rows = _rows(tmp_path)
    assert [row["position"] for row in rows] == [0]
    assert [row["seq"] for row in rows] == [1]
    # A property of the type and not of a state: a stream that has released its claim and has
    # nothing left to serve is still not a thing there may be two of.
    assert not _claim(tmp_path).exists()
    with pytest.raises(TypeError, match="copy.copy of this EvalStream is refused"):
        copy.copy(stream)


async def test_a_stream_may_not_be_deep_copied(tmp_path: Path) -> None:
    # The same refusal one level down, and not for the shallow copy's reason: a deep copy of a
    # stream would duplicate the token just as exactly, while also copying a live env catalog and
    # the episodes in flight, which are handles onto sessions and directories that no copy gets a
    # second of. Refused whole rather than left to fail partway through a copy, since a
    # half-copied serving stream is a mess with an ownership claim in it.
    stream = _eval_stream(tmp_path, max_in_flight=2)
    for duplicate in (
        lambda: copy.deepcopy(stream),
        # No `copy` call in sight at the call site: deep-copying anything that merely holds the
        # stream arrives at the same place, which is the realistic way a run would do this to
        # itself — a config dict or a run record snapshotted "safely".
        lambda: copy.deepcopy({"stream": stream, "tag": "run-1"}),
        lambda: copy.deepcopy([[stream]]),
    ):
        with pytest.raises(TypeError, match="copy.deepcopy of this EvalStream is refused"):
            duplicate()

    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert _claim(tmp_path).is_file()
    await _serve_one(stream)
    assert [row["position"] for row in _rows(tmp_path)] == [0]


async def test_a_stream_may_not_be_pickled(tmp_path: Path) -> None:
    # The duplication surface with the longest reach and the only one that *stores* the identity:
    # before this, `pickle.dumps` returned a few kilobytes with the ownership token among them,
    # and `pickle.loads` handed back an `EvalStream` that served into the original's directory
    # beside it. Nobody types this; a `spawn` process argument, a process pool or a task queue
    # does. Every protocol, because one that held at the default alone would be no refusal at all.
    stream = _eval_stream(tmp_path, max_in_flight=2)
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        with pytest.raises(TypeError, match="pickle of this EvalStream is refused"):
            pickle.dumps(stream, protocol=protocol)
        # The protocol method itself, not only what `pickle` does with it: a serialiser that asks
        # the object for its reconstructor directly gets the same answer.
        with pytest.raises(TypeError, match="pickle of this EvalStream is refused"):
            stream.__reduce_ex__(protocol)
    with pytest.raises(TypeError, match="pickle of this EvalStream is refused"):
        stream.__reduce__()

    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert _claim(tmp_path).is_file()
    await _serve_one(stream)
    assert [row["position"] for row in _rows(tmp_path)] == [0]


async def test_every_duplication_protocol_is_refused_on_the_class_that_serves(
    tmp_path: Path,
) -> None:
    # The family, enumerated, on both classes. Two things are checked of each surface and the
    # second is why this test exists: that the refusal is *defined* rather than reached through a
    # neighbour. `copy.copy` falls back to the pickle protocol when `__copy__` is missing and
    # pickling falls back between its own two methods, so a stream with a hole in it still refuses
    # a lot of things — an enumeration that only called the stdlib verbs would go green with the
    # method deleted. Called directly here, on the object, so each surface answers for itself.
    #
    # Both classes, because placement is the question this file settles: the refusals live on
    # `TaskStream`, where the ownership machinery they defend lives, and `EvalStream` inherits
    # them. Two rows for one queue position is a broken record under `immediate` exactly as it is
    # under `never` — a training run's reward signal is the thing being poisoned there — so this
    # is not an evaluation-grade guarantee bolted onto the evaluation-grade class.
    surfaces = {
        "__copy__": lambda stream: stream.__copy__(),
        "__deepcopy__": lambda stream: stream.__deepcopy__({}),
        "__reduce__": lambda stream: stream.__reduce__(),
        "__reduce_ex__": lambda stream: stream.__reduce_ex__(pickle.HIGHEST_PROTOCOL),
        # The state itself, which is the token and the queue position and the `seq`, and which
        # `object.__new__(cls).__dict__.update(state)` turns back into a serving stream. Reachable
        # through neither `copy` nor `pickle` now that both refuse above, and refused anyway: a
        # guarantee that depends on which door the caller came through is not a property of the
        # object.
        "__getstate__": lambda stream: stream.__getstate__(),
    }
    for name in surfaces:
        assert name in vars(TaskStream), f"{name} is not defined on TaskStream"

    for stream in (_stream(tmp_path / "task", [0]), _eval_stream(tmp_path / "eval")):
        for name, attempt in surfaces.items():
            with pytest.raises(TypeError, match="is refused") as refused:
                attempt(stream)
            assert type(stream).__name__ in str(refused.value), name
            assert "ownership identity, not a value" in str(refused.value), name
        # There is nothing to revive a state into either, so the halves of the protocol agree:
        # no `__setstate__` was added to accept what `__getstate__` refuses to produce.
        assert not hasattr(stream, "__setstate__")
        await stream.aclose()


async def test_a_stream_duplicated_past_the_refusals_still_cannot_write(
    tmp_path: Path,
) -> None:
    # The backstop, and the reason the ownership check asks *which object* and not only which
    # stream and which process. Both of those are values — a token is a string, a pid is an int —
    # so anything built out of a stream's state carries them and answers "yes". The refusals above
    # close every protocol a duplicate is normally made through; this closes the answer itself, at
    # the append, where the record is actually defended. Built the one way no protocol of ours is
    # consulted about, which is deliberate: this test reaches into the object exactly as a cloner
    # that bypasses `copy` and `pickle` would, and asserts that reaching in is not enough.
    stream = _eval_stream(tmp_path, max_in_flight=2)
    duplicate = object.__new__(type(stream))
    duplicate.__dict__.update(stream.__dict__)

    # It is not caught by being unrecognisable — it holds the claim's own token, under the claim's
    # own pid. It is caught because it is not the object the claim was taken for.
    held = json.loads(_claim(tmp_path).read_text(encoding="utf-8"))
    assert duplicate._owner == held["owner"] and held["pid"] == os.getpid()

    with pytest.raises(RuntimeError, match="duplicated from") as refused:
        await duplicate.get_task()
    # Not the fork message, which is what the token alone would have earned this object: the pid
    # matches, so telling this caller they forked would send them looking for a second process
    # that does not exist. The advice is the one that fits — build the stream where it serves.
    assert "inherited across a fork" not in str(refused.value)
    assert "build the stream where it will serve" in str(refused.value)
    assert duplicate.stopped
    # And the close reports it rather than succeeding quietly, naming the refusal underneath: a
    # stream that could not record the task it was handed served nothing, whoever owns the
    # directory.
    with pytest.raises(RuntimeError, match="could not record a dispense") as drained:
        await duplicate.aclose()
    assert "duplicated from" in str(drained.value.__cause__)

    # Nothing of the duplicate's reached the record — the refusal is before the append, not a
    # repair after it — and the original serves its queue and closes clean.
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    assert _claim(tmp_path).is_file()
    await _serve_one(stream)
    rows = _rows(tmp_path)
    assert [(row["position"], row["seq"]) for row in rows] == [(0, 1)]
    assert not _claim(tmp_path).exists()


async def test_a_refused_resume_puts_back_the_claim_it_displaced(tmp_path: Path) -> None:
    # A resume takes the directory over on the way in — it has to, since everything after that
    # point builds envs and reads the record it is continuing — so a construction that then
    # refuses has displaced a claim it never got to use. Leaving it gone would let a mistyped
    # queue stop a stream that was serving perfectly well: the dispossessed one can no longer
    # record the task it is holding, and the run nobody asked to stop is the one that had done
    # nothing wrong. A refusal changes nothing, and that now includes what it took over.
    serving = _stream(tmp_path, [0, 1], feedback=Immediate())
    await serving.get_task()
    held = json.loads(_claim(tmp_path).read_text(encoding="utf-8"))

    with pytest.raises(ValueError, match="dispense record"):
        _stream(tmp_path, [2], resume=True, feedback=Immediate())  # a queue that disagrees
    assert json.loads(_claim(tmp_path).read_text(encoding="utf-8")) == held

    # ...so the stream that was serving still owns its directory: it records the task it was
    # holding, serves the rest of its queue, and closes with the directory unclaimed.
    async with serving:
        await serving.dispatch(SUBMIT_TOOL, {"answer": "4"})
        await serving.get_task()
        await serving.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert [row["position"] for row in _rows(tmp_path)] == [0, 1]
    assert not _claim(tmp_path).exists()


async def test_a_refused_construction_leaves_the_directory_as_it_found_it(
    tmp_path: Path,
) -> None:
    # A claim is only ever released by the stream that took it, and a constructor that raises
    # hands back no stream — so a refusal that left its claim behind would make the next attempt,
    # the corrected one, look like a crashed run needing `resume=True`. Whatever this constructor
    # made, it unmakes. The refusal has to come from *after* the claim to be testing anything:
    # the argument checks all run before it, so this one is a factory that will not build.
    def _unbuildable(name: str) -> _FixtureScoreEnv:
        raise RuntimeError("this env cannot be provisioned")

    with pytest.raises(RuntimeError, match="cannot be provisioned"):
        TaskStream(_unbuildable, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    assert not (tmp_path / "prov").exists()
    # ...so the corrected attempt is an ordinary fresh run, needing no flag to get in.
    async with _stream(tmp_path, [0]) as retried:
        await retried.get_task()
        await retried.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert [row["position"] for row in _rows(tmp_path)] == [0]

    # ...and a directory the caller prepared is left standing, because removing one this call did
    # not create would be a refusal with a side effect.
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    with pytest.raises(RuntimeError, match="cannot be provisioned"):
        TaskStream(_unbuildable, [TaskRef(ENV_NAME, 0)], prov_dir=prepared)
    assert prepared.is_dir()
    assert list(prepared.iterdir()) == []


async def test_a_record_written_before_claims_existed_still_resumes(tmp_path: Path) -> None:
    # Backward compatibility, in the direction that matters: a provenance directory recorded by a
    # stream that never took a claim is a directory with no claim in it, and a resume must not
    # require one to be there. Absent reads as unowned, exactly as an absent regime stamp reads
    # as `never`.
    async with _stream(tmp_path, [0, 1], feedback=Immediate()) as first:
        await first.get_task()
        await first.dispatch(SUBMIT_TOOL, {"answer": "4"})
    _claim(tmp_path).unlink(missing_ok=True)  # as if written before ownership was recorded

    async with _stream(tmp_path, [0, 1], resume=True, feedback=Immediate()) as resumed:
        await resumed.get_task()
        await resumed.dispatch(SUBMIT_TOOL, {"answer": "6"})
    assert [row["position"] for row in _rows(tmp_path)] == [0, 1]


async def test_an_unreadable_claim_refuses_rather_than_grants(tmp_path: Path) -> None:
    # A claim that will not parse still says a stream was here. Every reading of it has to fail
    # towards refusal: the fresh path refuses on the file's existence (with a poorer message
    # than it would like), and an owner nobody can read is an owner no stream's token equals.
    stream = _stream(tmp_path, [0])
    _claim(tmp_path).write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="claimed by another stream"):
        _stream(tmp_path, [0])
    # The holder is dispossessed by it too, rather than reading its own name into the wreckage.
    with pytest.raises(RuntimeError, match="no longer claimed by this stream"):
        await stream.get_task()


# Run by `test_only_one_of_many_processes_can_claim_one_directory`, in its own interpreter,
# because that test is about an exclusion that has to hold between processes.
_RACER = """
import sys
import time
from pathlib import Path

from hgym.serve.stream import Immediate, Never, TaskRef, TaskStream
from tests._fixtures.score_env import ENV_NAME, _FixtureScoreEnv

prov, regime, verdict, hold = (Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
tasks = [{"id": "q0", "question": "2+2?", "answer": "4"}]
try:
    TaskStream(
        lambda _name: _FixtureScoreEnv(tasks=tasks),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=prov,
        feedback=Immediate() if regime == "immediate" else Never(),
    )
except ValueError as exc:
    verdict.write_text("REFUSED " + str(exc), encoding="utf-8")
else:
    verdict.write_text("CLAIMED", encoding="utf-8")
    # Hold the directory rather than releasing it, so that no loser can win by arriving late:
    # every refusal this test counts is a refusal against a claim that is still held.
    while hold.exists():
        time.sleep(0.01)
"""


async def test_only_one_of_many_processes_can_claim_one_directory(tmp_path: Path) -> None:
    # Across processes, which is where this has to hold: a harness that launches two runs against
    # one provenance directory does not do it from one interpreter, so an in-process registry
    # would prove nothing — and neither would a lock that is per-process rather than per-open
    # file. The exclusion is an `O_EXCL` create, one atomic operation of which exactly one caller
    # returns holding a descriptor, so there is no window in which two streams are both owners
    # and no check either of them could have won by racing.
    racer = tmp_path / "racer.py"
    racer.write_text(textwrap.dedent(_RACER), encoding="utf-8")
    hold = tmp_path / "hold"
    hold.write_text("serving", encoding="utf-8")
    environ = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    # Both regimes, because mixing them is the damage a second writer does.
    regimes = ["never", "immediate"] * 3
    verdicts = [tmp_path / f"verdict-{number}" for number in range(len(regimes))]
    running = [
        subprocess.Popen(
            [sys.executable, str(racer), str(tmp_path / "prov"), regime, str(verdict), str(hold)],
            stderr=subprocess.PIPE,
            text=True,
            env=environ,
        )
        for regime, verdict in zip(regimes, verdicts)
    ]
    try:
        # Each child records its verdict and only then holds, so waiting for the files is waiting
        # for every constructor to have finished — the winner never exits on its own.
        deadline = time.monotonic() + 60.0
        while not all(verdict.exists() for verdict in verdicts):
            assert time.monotonic() < deadline, [
                verdict.name for verdict in verdicts if not verdict.exists()
            ]
            await asyncio.sleep(0.05)
        outcomes = [verdict.read_text(encoding="utf-8") for verdict in verdicts]
    finally:
        hold.unlink(missing_ok=True)
        for process in running:
            try:
                process.communicate(timeout=60)
            except subprocess.TimeoutExpired:  # pragma: no cover - only for a wedged child
                process.kill()
                process.communicate()

    claimed = [outcome for outcome in outcomes if outcome == "CLAIMED"]
    refused = [outcome for outcome in outcomes if outcome.startswith("REFUSED")]
    assert len(claimed) == 1, outcomes
    assert len(refused) == len(running) - 1, outcomes
    assert all("claimed by another stream" in outcome for outcome in refused), refused
    # One claim on disk, and one regime in it — whichever process won.
    held = json.loads((tmp_path / "prov" / "claim.json").read_text(encoding="utf-8"))
    assert held["feedback_regime"] in {"never", "immediate"}
    assert held["pid"] in [process.pid for process in running]


# Run by `test_a_takeover_cannot_land_between_a_check_and_its_append`, in its own interpreter:
# the window it opens sits between two statements of one process, and only another process can
# enter it.
_INTERLEAVER = """
import asyncio
import sys
import time
from pathlib import Path

from hgym.serve import stream as stream_module
from hgym.serve.stream import TaskRef, TaskStream
from tests._fixtures.score_env import ENV_NAME, _FixtureScoreEnv

prov, role, verdict, inside = (
    Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
)
tasks = [{"id": "q0", "question": "2+2?", "answer": "4"}]
appending = stream_module._append_jsonl


def paused(path, record, *, durable=False):
    # The window a check-then-append leaves open, held wide: the ownership check has returned and
    # the record has not been written yet. Patched on the module's own append rather than on
    # anything the fix introduced, so this test asks about the ordering and not about a shape.
    if path.name == "dispenses.jsonl":
        inside.write_text("dispensing", encoding="utf-8")
        time.sleep(2.0)
    return appending(path, record, durable=durable)


async def main():
    if role == "holder":
        stream_module._append_jsonl = paused
    else:
        while not inside.exists():  # the takeover starts only once the holder is in the window
            await asyncio.sleep(0.01)
    try:
        stream = TaskStream(
            lambda _name: _FixtureScoreEnv(tasks=tasks),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=prov,
            resume=(role == "taker"),
        )
        await stream.get_task()
    except BaseException as exc:
        verdict.write_text("REFUSED " + str(exc), encoding="utf-8")
    else:
        verdict.write_text("DISPENSED", encoding="utf-8")


asyncio.run(main())
"""


async def test_a_takeover_cannot_land_between_a_check_and_its_append(tmp_path: Path) -> None:
    # Ownership re-read before a write is not ownership *of* the write. The check returns, the
    # takeover lands, and the append the check authorised is still to come — so it goes through,
    # on behalf of a stream that no longer owns the directory. The taking-over stream seeds its
    # numbering from a record that does not hold the dispense yet, and both processes file
    # `(seq=1, position=0)`: two dispenses under one identifier, which is exactly what the resume
    # numbering exists to prevent, with `reconcile` owing a `broker_abort` it cannot tell apart.
    #
    # So the claim is verified inside the same exclusion the append happens in. Whichever process
    # gets there first, the other is *ordered* behind it rather than interleaved with it: the
    # takeover waits for the record it is about to continue, and the displaced stream finds a
    # claim that is no longer its own the next time it writes.
    interleaver = tmp_path / "interleaver.py"
    interleaver.write_text(textwrap.dedent(_INTERLEAVER), encoding="utf-8")
    environ = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    verdicts = {role: tmp_path / f"verdict-{role}" for role in ("holder", "taker")}
    running = [
        subprocess.Popen(
            [
                sys.executable,
                str(interleaver),
                str(tmp_path / "prov"),
                role,
                str(verdict),
                str(tmp_path / "inside"),
            ],
            stderr=subprocess.PIPE,
            text=True,
            env=environ,
        )
        for role, verdict in verdicts.items()
    ]
    try:
        for process in running:
            assert process.communicate(timeout=120)[1] == "", process.args
    finally:
        for process in running:
            if process.poll() is None:  # pragma: no cover - only for a wedged child
                process.kill()
                process.communicate()

    outcomes = {role: verdict.read_text(encoding="utf-8") for role, verdict in verdicts.items()}
    # Neither is refused here: the holder's dispense is its own, and the takeover is a call the
    # human is entitled to make. What may not happen is the two of them numbering it together.
    assert outcomes == {"holder": "DISPENSED", "taker": "DISPENSED"}, outcomes
    dispenses = read_dispenses(tmp_path / "prov")
    assert [record["position"] for record in dispenses] == [0, 0]
    assert [record["seq"] for record in dispenses] == [1, 2], dispenses
    assert len({record["lease"] for record in dispenses}) == 2
    # ...and the file is one record per line. Two appends in flight at once interleave a record
    # into the middle of another's line, since the record and the terminator that commits it are
    # two writes — a record destroyed rather than merely doubled, which the same exclusion is
    # what rules out.
    lines = (tmp_path / "prov" / "dispenses.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["seq"] for line in lines] == [1, 2]
