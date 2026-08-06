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
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Sequence

import pytest
from fastmcp import Client

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
    with pytest.raises(ValueError, match="feedback must be a FeedbackPolicy"):
        _stream(tmp_path, [0], feedback=bad)


async def test_a_policy_that_cannot_name_its_regime_is_refused(tmp_path: Path) -> None:
    # The regime is written into every record and compared against the record on resume, and
    # `reveals` decides a wire shape. Both are read once, at construction, and both are exact
    # types: a truthy non-bool would open the verdict channel on a policy that never said so.
    class _Nameless(FeedbackPolicy):
        regime: ClassVar[Any] = ""
        reveals: ClassVar[bool] = False

        def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
            return ()

    class _Truthy(FeedbackPolicy):
        regime: ClassVar[str] = "truthy"
        reveals: ClassVar[Any] = 1

        def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
            return published

    with pytest.raises(ValueError, match="regime must be a non-empty string"):
        _stream(tmp_path / "a", [0], feedback=_Nameless())
    with pytest.raises(ValueError, match="`reveals` must be a bool"):
        _stream(tmp_path / "b", [0], feedback=_Truthy())


async def test_a_policy_that_cannot_answer_stops_the_run_and_says_nothing(
    tmp_path: Path,
) -> None:
    # Composing a revealing answer runs caller-supplied code, and one that raises will raise for
    # every task — so the row already stamped with its regime would be claiming a channel the
    # agent was never told through. The failure is contained (no traceback at the agent, no
    # change of shape), and the stream stops: loud to the harness, silent to the agent.
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


async def test_a_policy_that_cannot_answer_is_reported_at_the_drain(tmp_path: Path) -> None:
    # The other half of "silent to the agent": the harness has to be able to find out, and the
    # place it finds out is the same one every integrity failure here is reported at.
    class _Exploding(FeedbackPolicy):
        regime: ClassVar[str] = "exploding"
        reveals: ClassVar[bool] = True

        def reveal(self, published: Sequence[Dict[str, Any]]) -> Sequence[Dict[str, Any]]:
            raise RuntimeError("the policy could not be asked")

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
