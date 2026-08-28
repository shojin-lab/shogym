"""The paired feedback policies: one named channel each, and nothing else.

:class:`Information` and :class:`Placebo` exist for a comparison :class:`Never` and
:class:`Immediate` cannot host. ``Never`` answers with no channel at all, so an agent under a
revealing policy is handed both a verdict *and* a member its control never sees; ``Immediate``
answers with everything the row records, so the size and contents of what an agent reads vary
with how many metrics its env happens to publish. The pair answers with exactly one named item,
so two arms of one design differ in what the channel says and in nothing else.

What these tests are adversarial about is that "nothing else": that neither policy reveals the
summary numbers beside its item, that neither reveals the other's, and that a run's regime is
still a property of its record rather than of what the agent could read off the shape of an
answer.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shogym.feedback.wire import (
    CHANNEL_FEEDBACK_NAME,
    NOTICE_FEEDBACK_NAME,
    REPORT_FEEDBACK_NAME,
)
from shogym.serve.stream import (
    FeedbackPolicy,
    Immediate,
    Information,
    Never,
    Placebo,
    TaskRef,
    TaskStream,
)
from tests._fixtures.channel_env import (
    ENV_NAME,
    NOTICE_TEXT,
    REPORT_TEXT,
    _FixtureChannelEnv,
)
from tests._fixtures.score_env import SUBMIT_TOOL

TASKS = [
    {"id": "q0", "question": "2+2?", "answer": "4"},
    {"id": "q1", "question": "3+3?", "answer": "6"},
]


def _env_for(_name: str) -> _FixtureChannelEnv:
    return _FixtureChannelEnv(tasks=TASKS)


def _stream(tmp_path: Path, indices: List[int], **kwargs: Any) -> TaskStream:
    return TaskStream(
        _env_for,
        [TaskRef(ENV_NAME, i) for i in indices],
        prov_dir=tmp_path / "prov",
        **kwargs,
    )


def _payload(result: Any) -> Dict[str, Any]:
    return json.loads(result.content[0].text)


def _rows(tmp_path: Path) -> List[Dict[str, Any]]:
    path = tmp_path / "prov" / "results.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ----- each policy opens its own channel and only its own -----


async def test_information_reveals_the_report_and_nothing_beside_it(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0], feedback=Information()) as stream:
        await stream.get_task()
        answer = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert answer["feedback"] == [
        {"name": CHANNEL_FEEDBACK_NAME, "value": REPORT_TEXT, "level": "episode"}
    ]
    # The env published `correct` too, and the record kept it. The channel did not.
    assert "correct" in json.dumps(_rows(tmp_path))
    assert "correct" not in json.dumps(answer)


async def test_placebo_reveals_the_notice_and_nothing_beside_it(tmp_path: Path) -> None:
    async with _stream(tmp_path, [0], feedback=Placebo()) as stream:
        await stream.get_task()
        answer = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert answer["feedback"] == [
        {"name": CHANNEL_FEEDBACK_NAME, "value": NOTICE_TEXT, "level": "episode"}
    ]
    assert REPORT_TEXT not in json.dumps(answer)


async def test_placebo_says_the_same_thing_however_the_task_went(tmp_path: Path) -> None:
    # The property that makes it a control: a right answer and a wrong one on the same task come
    # back as the same bytes, so nothing about the outcome is readable from the channel.
    seen: List[str] = []
    async with _stream(tmp_path, [0, 0], feedback=Placebo()) as stream:
        for answer in ("not the answer", "4"):
            await stream.get_task()
            seen.append((await stream.dispatch(SUBMIT_TOOL, {"answer": answer})).content[0].text)

    assert seen[0] == seen[1]
    # ...and the two tasks really did score differently, so this is not vacuous.
    assert [row.score.success for row in stream.results if row.score] == [False, True]


async def test_the_two_arms_differ_in_the_value_and_in_nothing_else(tmp_path: Path) -> None:
    # The whole point of the pair. Same envelope, same member, one item each, same length, so an
    # agent cannot read which arm it is in off anything but the content.
    async with _stream(tmp_path / "info", [0], feedback=Information()) as treated:
        await treated.get_task()
        treatment = _payload(await treated.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    async with _stream(tmp_path / "placebo", [0], feedback=Placebo()) as control:
        await control.get_task()
        placebo = _payload(await control.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert set(treatment) == set(placebo)
    assert {key: treatment[key] for key in treatment if key != "feedback"} == {
        key: placebo[key] for key in placebo if key != "feedback"
    }
    assert len(treatment["feedback"]) == len(placebo["feedback"]) == 1
    # The strong form, and the one an equal length alone does not give: the two serialized items
    # are identical apart from the value's own bytes. An item still carrying the env's own
    # `notice` would name its arm, and the control could be told apart without a word of what it
    # was handed being read.
    treated_item, control_item = dict(treatment["feedback"][0]), dict(placebo["feedback"][0])
    assert treated_item["name"] == control_item["name"] == CHANNEL_FEEDBACK_NAME
    assert treated_item.pop("value") != control_item.pop("value")
    assert treated_item == control_item
    assert NOTICE_FEEDBACK_NAME not in json.dumps(placebo)
    assert REPORT_FEEDBACK_NAME not in json.dumps(treatment)
    assert len(json.dumps(treatment)) == len(json.dumps(placebo))


async def test_the_envelope_is_the_one_a_redacted_answer_uses(tmp_path: Path) -> None:
    # A policy may add one member and change nothing else, whichever policy it is.
    async with _stream(tmp_path / "silent", [0]) as quiet:
        await quiet.get_task()
        redacted = _payload(await quiet.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    for name, policy in (("info", Information()), ("placebo", Placebo())):
        async with _stream(tmp_path / name, [0], feedback=policy) as stream:
            await stream.get_task()
            revealed = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        assert set(revealed) == set(redacted) | {"feedback"}
        assert {key: revealed[key] for key in redacted} == redacted


async def test_a_missing_channel_answers_with_an_empty_member(tmp_path: Path) -> None:
    # An env that publishes no notice is not a shape an agent can read: the member is a property
    # of the policy, so it is present and empty rather than absent.
    from tests._fixtures.score_env import _FixtureScoreEnv

    stream = TaskStream(
        lambda _name: _FixtureScoreEnv(tasks=TASKS),
        [TaskRef("_fixture_score", 0)],
        prov_dir=tmp_path / "prov",
        feedback=Placebo(),
    )
    async with stream:
        await stream.get_task()
        answer = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
    assert answer["feedback"] == []
    assert stream.results[0].score is not None


# ----- the record still says which regime it was written under -----


async def test_each_policy_stamps_its_own_regime(tmp_path: Path) -> None:
    for name, policy, regime in (
        ("info", Information(), "information"),
        ("placebo", Placebo(), "placebo"),
    ):
        async with _stream(tmp_path / name, [0], feedback=policy) as stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
        assert [row["feedback_regime"] for row in _rows(tmp_path / name)] == [regime]


def test_the_four_regimes_are_distinct() -> None:
    # One regime, one policy: two policies sharing a name would make a resumed run's check pass
    # over a record they did not both write.
    regimes = [Never.regime, Immediate.regime, Information.regime, Placebo.regime]
    assert len(set(regimes)) == len(regimes)


# ----- the regime is the assignment, not a record of what was delivered -----


@pytest.mark.parametrize(
    "policy, regime", [(Information(), "information"), (Placebo(), "placebo")]
)
async def test_a_cancelled_terminal_is_a_scored_row_nobody_was_told(
    tmp_path: Path, policy: Any, regime: str
) -> None:
    """The gap between assignment and exposure, in both arms.

    The row is durable before the policy's answer is composed, which it has to be: the answer is
    composed from the recorded row. So a caller that goes away inside the finalizer leaves a row
    that is sealed, scored and stamped with its arm, and nothing whatever reached the agent. The
    stamp is still the right thing for an intention-to-treat estimate, because every assigned task
    has one, and it is not evidence of a delivered verdict. What was actually delivered is the
    runner's to record."""
    release = asyncio.Event()

    class _SlowFinalize(_FixtureChannelEnv):
        async def finalize(self, req: Any) -> Any:
            await release.wait()
            return await super().finalize(req)

    stream = TaskStream(
        lambda _name: _SlowFinalize(tasks=TASKS),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        feedback=policy,
    )
    await stream.__aenter__()
    try:
        await stream.get_task()
        call = asyncio.ensure_future(stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))
        await asyncio.sleep(0.05)  # inside the blocked finalizer
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        closing = asyncio.ensure_future(stream.aclose())
        await asyncio.sleep(0.05)  # the drain waits for the finalization rather than racing it
        release.set()
        await closing
    finally:
        release.set()  # never leave the evaluator blocked, however this test ends
        await stream.aclose()

    (row,) = _rows(tmp_path)
    assert row["closure"] == "sealed"
    assert row["score"] is not None and row["score"]["success"] is True
    assert row["feedback_regime"] == regime


async def test_a_task_the_stream_ended_still_carries_its_assignment(tmp_path: Path) -> None:
    # The other shape of the same absence, and the commoner one: a task the STREAM ended has no
    # terminating call to answer, so nothing was delivered through the channel it was assigned.
    # Its row still carries that arm, which is what an intention-to-treat estimate needs from it.
    async with _stream(tmp_path, [0, 1], feedback=Information()) as stream:
        await stream.get_task()
        await stream.get_task()  # displaces and force-scores the first
        await stream.dispatch(SUBMIT_TOOL, {"answer": "6"})

    rows = _rows(tmp_path)
    assert [row["closure"] for row in rows] == ["drained", "sealed"]
    assert {row["feedback_regime"] for row in rows} == {"information"}


# ----- admission is still by exact type -----


def test_a_lookalike_channel_policy_is_refused(tmp_path: Path) -> None:
    # Subclassing is not how a policy is admitted, and the pair changes nothing about that: a
    # subclass free to name its own regime could reveal a verdict while every row said `placebo`.
    class _Sneaky(Placebo):
        regime = "placebo"
        reveals = True

        def reveal(self, published: Any) -> Any:
            return published

    try:
        _stream(tmp_path, [0], feedback=_Sneaky())
    except ValueError as exc:
        assert "Information()" in str(exc) and "Placebo()" in str(exc)
    else:  # pragma: no cover - the refusal is the test
        raise AssertionError("a FeedbackPolicy subclass was admitted")


async def test_a_shadowed_reveal_on_an_admitted_policy_is_not_called(tmp_path: Path) -> None:
    # `frozen` is not sealed: an instance dictionary can shadow the method. The stream calls the
    # module's function for the admitted type, so the shadow is never found.
    policy = Information()
    object.__setattr__(policy, "reveal", lambda published: [{"name": "x", "value": "forged"}])
    stream = _stream(tmp_path, [0], feedback=policy)
    assert isinstance(stream.feedback, FeedbackPolicy)

    async with stream:
        await stream.get_task()
        answer = _payload(await stream.dispatch(SUBMIT_TOOL, {"answer": "4"}))

    assert answer["feedback"] == [
        {"name": CHANNEL_FEEDBACK_NAME, "value": REPORT_TEXT, "level": "episode"}
    ]
