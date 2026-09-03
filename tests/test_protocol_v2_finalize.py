"""Ending an attempt that nothing is going to finish.

Three ways in, one transition. A controller asks the stream to end an attempt, the generation's
own deadline ends one that stayed active too long, and the gateway ends one whose environment
step budget is spent. All of them arrive at the same place: the attempt fails finally, its
capacity comes back, the payload it was owed is resolved without being rendered, its outcome is
written at the floor, and nothing is minted for the model to read.

A fourth way in belongs to the seal itself. An accepted terminal makes the attempt the seal's,
and the exact filing sent again continues one that was interrupted, but a batch that cannot go on
is not a step any retry can take: it ends the attempt from inside the seal, with no
acknowledgement and nothing committed. Work that failed for good and work that answered with a
result the seal cannot vouch for both end it, and they are written apart. That ending is driven
twice, once at the stream and once through the transport that filed, because the record a
transport keeps for a lost acknowledgement has nothing to collect once the attempt has ended and
a transport still holding it would refuse every call it was ever asked afterwards.

What the tests below pin is mostly what finalization may not do. It may not overtake a terminal
the stream has accepted, it may not run while a result is outstanding, and it may not leave an
opening for a second outcome afterwards. The deadline is a durable timer, so it is exercised in
skipped time and the history it leaves is replayed at the end. And an ending reaches every
attempt that was waiting on it: a schedule may gate one task on another sealing, and an attempt
that ended without a filing will never seal, so the gate it holds shut is floored with it.

Two of them are about answers that never arrive. Every durable step here can have its response
lost, and the step cap is two of them in a row, so the cuts between them are driven directly:
the world must not be called twice for one step, and a spent attempt must not be left running.

Two more are about the clock, which is the one thing here that moves without a call. The timer
fires while a result is out with the harness, whose attestation was fixed before it fired, and
it fires while there is no Worker running at all.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from fastmcp import Client  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    IMMEDIATE,
    PAYLOAD_FIRST,
    RELEASE_AT_SEAL,
    EligibilityGate,
    PresentationCommit,
    PullRequest,
    ReleasePlan,
    TerminalMetadata,
)
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    PULL_TOOL,
    StreamGateway,
    build_gateway_server,
    terminal_manifest,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    ABANDONED,
    DEADLINE,
    SEAL_FAILED,
    SEAL_UNUSABLE,
    STEP_CAP,
    BlobsVerified,
    ConsumerClaim,
    EnvironmentCall,
    FinalizeRequest,
    GradeAttemptInput,
    OfferedMessage,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    VerifyBlobsInput,
    assignments_for,
    configuration_hash,
    generate_payload_bundle_activity,
    grade_attempt_activity,
    protocol_error_code,
    resume_stream,
    seal_attempt_activity,
    start_stream,
    stream_replayer,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GRADE_ATTEMPT,
    VERIFY_BLOBS,
)
from tests._fixtures.policy_rows import registering_the_receipt  # noqa: E402

TEST_ENV = "wordle_v1"
TASK_BODY = "file the report"
CLAIM_HASH = "d" * 64
TRANSCRIPT_BLOB = "e" * 64
PROVIDER_TURN_BLOB = "f" * 64
CHECKPOINT_BLOB = "9" * 64
CONSUMER = ConsumerClaim(consumer_id="harness-1", claim_hash=CLAIM_HASH)

ATTEMPT = "00000000000000000000000000000100"
DEADLINE_MS = 600_000
# The deadline for the one test that has to fire it while an Activity is still running. The
# clock is skipped rather than waited on, and skipping past an Activity's own timeout would
# fail that Activity instead of expiring the attempt, so this one is well inside it.
DEADLINE_INSIDE_A_READ_MS = 20_000


def oid(value: int) -> str:
    return f"{value:032x}"


def make_start(
    *,
    bodies: Any = (TASK_BODY,),
    attempt_deadline_ms: int = 0,
    terminal: str = "submit",
    argument_names: Any = ("answer",),
    release: Optional[ReleasePlan] = None,
    without_payload: Any = (),
    blob_root: Optional[str] = None,
) -> StreamStart:
    """A generation whose every public identifier is fixed before it serves anything."""
    tasks = [
        TaskItem(
            task_position=index,
            attempt_id=oid(0x100 + index * 4),
            task_message_id=oid(0x101 + index * 4),
            ack_message_id=oid(0x102 + index * 4),
            payload_position=index,
            payload_message_id=oid(0x103 + index * 4),
            body=body,
        )
        for index, body in enumerate(bodies)
    ]
    return registering_the_receipt(
        StreamStart(
            configuration_hash="c" * 64,
            consumer_claim_hash=CLAIM_HASH,
            initial_cursor=oid(1),
            done_message_id=oid(2),
            id_key_hex="ab" * 32,
            hidden_execution_id="execution-1",
            canonicalization_version="kernel.1",
            terminal_tool=TerminalTool(
                public_tool_name=terminal,
                native_terminal_name=terminal,
                argument_names=list(argument_names),
            ),
            tasks=tasks,
            attempt_deadline_ms=attempt_deadline_ms,
            blob_root=blob_root,
            release=IMMEDIATE if release is None else release,
            assignments=(
                []
                if release is None
                else assignments_for(tasks, release, without_payload=without_payload)
            ),
        )
    )


# The delayed leg, written as a plan: the filler waits for A's payload to be presented and B
# waits for the filler to seal. Neither the filler nor B is a position anything is delivered
# against, so neither carries an obligation of its own.
LEG_BODIES = ("A.", "B.", "The filler.")
LEG_PLAN = ReleasePlan(
    RELEASE_AT_SEAL,
    PAYLOAD_FIRST,
    BY_POSITION,
    gates=[
        EligibilityGate(oid(0x108), after_payload_position=0),
        EligibilityGate(oid(0x104), after_sealed_attempt_id=oid(0x108)),
    ],
)
LEG_WITHOUT_PAYLOAD = (oid(0x104), oid(0x108))


class Caller:
    """One authenticated consumer, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: StreamHandle, cursor: str) -> None:
        self.stream = stream
        self.cursor = cursor
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x1000 + self._counter)

    async def pull(self) -> OfferedMessage:
        return await self.stream.pull(
            PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )

    async def present(self, message: OfferedMessage) -> None:
        ack = await self.stream.present(
            message,
            attestation_id=self.next_id(),
            transcript_blob=TRANSCRIPT_BLOB,
            provider_turn_blob=PROVIDER_TURN_BLOB if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=CHECKPOINT_BLOB if message.kind == "task" else None,
        )
        self.cursor = ack.cursor

    def seal_request(self, attempt_id: str = ATTEMPT) -> SealRequest:
        return SealRequest(
            metadata=TerminalMetadata(
                request_id=self.next_id(),
                last_presented_cursor=self.cursor,
                attempt_id=attempt_id,
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
            native_arguments={"answer": "42"},
        )

    async def finalize(self, reason: str, attempt_id: str = ATTEMPT) -> Any:
        return await self.stream.finalize(
            FinalizeRequest(
                request_id=self.next_id(), attempt_id=attempt_id, reason=reason
            )
        )

    async def take(self) -> OfferedMessage:
        """Pull one message and present it, which is what the harness does all day."""
        message = await self.pull()
        await self.present(message)
        return message


async def refused(awaitable: Any) -> str:
    """Return the protocol error code a refused call carries, from either boundary.

    A refused Update carries the code in its application failure and a refused tool call carries
    the canonical record as the error's text, so a test that drives both reads both.
    """
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001 - the code is the assertion
        code = protocol_error_code(error)
        if code is not None:
            return code
        record = json.loads(str(error))
        assert record["kind"] == "protocol_error"
        return record["code"]
    raise AssertionError("the call was accepted")


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await WorkflowEnvironment.start_time_skipping()
    except Exception as error:
        pytest.skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


async def open_stream(
    environment: WorkflowEnvironment,
    start: Optional[StreamStart] = None,
    *,
    workflow_id: str,
) -> Caller:
    stream = await start_stream(
        environment.client, start or make_start(), workflow_id=workflow_id
    )
    receipt = await stream.claim_consumer(CONSUMER)
    return Caller(stream, receipt.initial_cursor)


@pytest_asyncio.fixture
async def caller(env: WorkflowEnvironment) -> AsyncIterator[Caller]:
    async with stream_worker(env.client):
        yield await open_stream(env, workflow_id="stream/finalize/1")


@pytest.mark.network
async def test_the_controller_ends_an_active_attempt_and_mints_nothing(caller: Caller) -> None:
    """Capacity back, obligation resolved, outcome at the floor, and not one byte offered."""
    await caller.take()
    before = await caller.stream.stream_state()
    assert before.attempts[ATTEMPT] == "active"

    receipt = await caller.finalize(ABANDONED)
    assert receipt.attempt_id == ATTEMPT
    assert receipt.reason == ABANDONED
    assert receipt.score == 0.0
    assert receipt.capacity_in_use == 0
    assert receipt.obligation_state == "final_failed"

    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "final_failed"
    assert state.obligations[ATTEMPT] == "final_failed"
    assert state.final_failures == {ATTEMPT: ABANDONED}
    assert state.capacity_in_use == 0
    # Nothing was offered, nothing is outstanding, no candidate was built, and the cursor is
    # where the task presentation left it. The model has no way to know this happened.
    assert state.offer_count == before.offer_count
    assert state.presentation_count == before.presentation_count
    assert state.pending_message_id is None
    assert state.cursor == before.cursor
    assert state.materialization_count == 0


@pytest.mark.network
async def test_a_finalization_never_overtakes_an_accepted_terminal(caller: Caller) -> None:
    """Once a filing has been accepted the attempt is the seal's, before and after the Ack."""
    await caller.take()
    ack = await caller.stream.seal(caller.seal_request())
    assert await refused(caller.finalize(ABANDONED)) == "conflicting_seal"

    await caller.present(ack)
    assert await refused(caller.finalize(ABANDONED)) == "conflicting_seal"
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "ack_presented"
    assert state.final_failures == {}


@pytest.mark.network
async def test_a_finalization_waits_for_an_outstanding_result(caller: Caller) -> None:
    """A result nobody has presented is a caller's turn, not a controller's."""
    await caller.take()
    wait = await caller.pull()
    assert wait.kind == "wait"
    assert await refused(caller.finalize(ABANDONED)) == "outstanding_response"
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "active"

    await caller.present(wait)
    assert (await caller.finalize(ABANDONED)).reason == ABANDONED


@pytest.mark.network
async def test_an_unknown_attempt_and_a_reason_this_generation_does_not_declare(
    caller: Caller,
) -> None:
    """The two things a controller can get wrong that are not about the attempt's state."""
    await caller.take()
    assert await refused(caller.finalize(ABANDONED, attempt_id=oid(0xBEEF))) == "invalid_attempt"
    assert (
        await refused(
            caller.stream.finalize(
                FinalizeRequest(
                    request_id=caller.next_id(), attempt_id=ATTEMPT, reason="bored"
                )
            )
        )
        == "invalid_message"
    )
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "active"


@pytest.mark.network
async def test_a_filing_for_a_finalized_attempt_conflicts_however_often_it_is_retried(
    caller: Caller,
) -> None:
    """The attempt has an outcome already, so a filing would be a second one."""
    await caller.take()
    await caller.finalize(STEP_CAP)
    request = caller.seal_request()
    assert await refused(caller.stream.seal(request)) == "conflicting_seal"

    # The exact same terminal request again, under a fresh Update so the workflow answers
    # rather than Temporal's own deduplication.
    assert (
        await refused(
            caller.stream.handle.execute_update(
                StreamWorkflow.seal_attempt,
                args=[request, caller.stream.writer],
                id="retry-of-the-filing",
            )
        )
        == "conflicting_seal"
    )
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "final_failed"
    assert state.pending_message_id is None
    assert state.offer_count == 1


@pytest.mark.network
async def test_a_deadline_ends_an_attempt_that_stayed_active(env) -> None:
    """A durable timer, fired in skipped time, on an attempt nobody was finishing."""
    async with stream_worker(env.client):
        caller = await open_stream(
            env,
            make_start(attempt_deadline_ms=DEADLINE_MS),
            workflow_id="stream/finalize/deadline",
        )
        await caller.take()
        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "active"
        assert state.final_failures == {}

        await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))
        # The filing that arrives after the deadline is a filing for an attempt that is over.
        assert await refused(caller.stream.seal(caller.seal_request())) == "conflicting_seal"
        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: DEADLINE}
        assert state.capacity_in_use == 0
        assert state.pending_message_id is None


@pytest.mark.network
async def test_a_deadline_does_not_end_an_attempt_that_was_filed_in_time(env) -> None:
    """The timer is disarmed by the accepted terminal, not by the seal committing."""
    async with stream_worker(env.client):
        caller = await open_stream(
            env,
            make_start(attempt_deadline_ms=DEADLINE_MS),
            workflow_id="stream/finalize/in-time",
        )
        await caller.take()
        ack = await caller.stream.seal(caller.seal_request())
        await caller.present(ack)

        await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))
        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "ack_presented"
        assert state.final_failures == {}


@pytest.mark.network
async def test_done_counts_a_finalized_attempt_as_resolved_and_the_history_replays(env) -> None:
    """A closed queue and one ended attempt is a generation with nothing left to do."""
    async with stream_worker(env.client):
        caller = await open_stream(
            env,
            make_start(attempt_deadline_ms=DEADLINE_MS),
            workflow_id="stream/finalize/done",
        )
        await caller.take()
        await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))
        await caller.stream.close_queue()

        done = await caller.pull()
        assert done.kind == "done"
        await caller.present(done)
        outcome = await caller.stream.handle.result()
        assert outcome.generation_state == "done"
        assert outcome.finalized == 1
        assert outcome.sealed == 0
        assert outcome.payloads_delivered == 0

    history = await caller.stream.handle.fetch_history()
    await stream_replayer().replay_workflow(history)


@pytest.mark.network
async def test_the_step_budget_ends_the_attempt_and_the_next_pull_moves_on(env) -> None:
    """The env's budget, counted where the calls that spend it actually pass.

    Nothing about the budget reaches the stream on its own: an environment call never becomes
    an Update. So the gateway counts, and the call that finds nothing left to spend is where the
    attempt ends. The call that spends the last step is not that one: an attempt out of world
    steps can still be filed, and under this protocol filing is a call to the stream rather than
    a step in the world, so an environment that promises a fixed number of moves and a terminal
    after them would otherwise lose every full-length attempt unsealed. The model reads its last
    observation, files or does not, and the next pull is what says the stream has moved on.
    """
    worlds: List[ServedEpisode] = []

    async def open_world(attempt_id: str) -> ServedEpisode:
        started = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        worlds.append(started)
        return started

    async with stream_worker(env.client):
        episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        try:
            stream = await start_stream(
                env.client,
                make_start(
                    bodies=("guess the first word", "and the second"),
                    terminal="terminate",
                    argument_names=(),
                ),
                workflow_id="stream/finalize/budget",
            )
            receipt = await stream.claim_consumer(CONSUMER)
            # The contract this gateway serves, with a budget short enough to spend.
            spec = episode.describe().model_copy(update={"horizon": 2})
            gateway = StreamGateway(
                stream,
                episode,
                spec,
                terminal_manifest(spec),
                initial_cursor=receipt.initial_cursor,
                open_episode=open_world,
            )
            first = json.loads(await gateway.pull({}))
            assert first["kind"] == "task"
            attempt = first["attempt_id"]

            for word in ("crane", "slate"):
                played = await gateway.environment("guess", _guess(attempt, word))
                # The call that spends the last of the budget is answered like any other.
                assert json.loads(played.content[0].text)["valid"] is True

            # And the attempt is still the attempt: the budget bought world calls, and what an
            # attempt with none left can still do is file.
            state = await gateway.stream_state()
            assert state.attempts[attempt] == "active"
            assert state.final_failures == {}

            # The call after that is the one with nothing to spend. It reaches no world, ends
            # the attempt, and is refused, and so is everything addressed to that attempt after.
            assert (
                await refused(gateway.environment("guess", _guess(attempt, "adieu")))
                == "invalid_attempt"
            )
            state = await gateway.stream_state()
            assert state.attempts[attempt] == "final_failed"
            assert state.final_failures == {attempt: STEP_CAP}
            assert state.capacity_in_use == 0
            # The env was not sealed and not graded: the ending is the stream's.
            assert episode.sealed is False
            assert len(episode._trajectory) == 2
            assert (
                await refused(gateway.environment("guess", _guess(attempt, "adieu")))
                == "invalid_attempt"
            )

            # The next pull is the next task.
            second = json.loads(await gateway.pull({}))
            assert second["kind"] == "task"
            assert second["attempt_id"] != attempt

            # The ended attempt's world is not the next task's. Nothing sealed it, so no
            # acknowledgement retired it, and it is retired where the next task is presented:
            # the calls that spend the new budget reach the world this task was given.
            assert len(worlds) == 1
            played = await gateway.environment("guess", _guess(second["attempt_id"], "adieu"))
            assert json.loads(played.content[0].text)["valid"] is True
            assert len(worlds[0]._trajectory) == 1
            assert len(episode._trajectory) == 2
        finally:
            for world in worlds:
                await world.close()
            await episode.close()


@pytest.mark.network
async def test_a_replacement_transport_ends_an_attempt_whose_budget_is_already_spent(
    env,
) -> None:
    """The budget belongs to the attempt, so a new transport does not hand it back.

    An attempt out of world steps stays active for the filing it is still owed, which means a
    gateway built over that generation afterwards finds work it may route to. What it must not
    find is a budget: the calls that spent it were granted by the stream and counted there, and
    a replacement reads that count rather than starting the attempt again at nothing.
    """
    async with stream_worker(env.client):
        episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        try:
            stream = await start_stream(
                env.client,
                make_start(terminal="terminate", argument_names=()),
                workflow_id="stream/finalize/budget-replacement",
            )
            receipt = await stream.claim_consumer(CONSUMER)
            spec = episode.describe().model_copy(update={"horizon": 2})
            gateway = StreamGateway(
                stream,
                episode,
                spec,
                terminal_manifest(spec),
                initial_cursor=receipt.initial_cursor,
            )
            first = json.loads(await gateway.pull({}))
            attempt = first["attempt_id"]
            for word in ("crane", "slate"):
                await gateway.environment("guess", _guess(attempt, word))
            state = await gateway.stream_state()
            assert state.attempts[attempt] == "active"
            assert state.environment_calls == {attempt: 2}

            # A second transport over the same generation and the same world, built the way a
            # replacement process builds one: it kept nothing of what the first one counted.
            replacement = StreamGateway(
                stream,
                episode,
                spec,
                terminal_manifest(spec),
                initial_cursor=state.cursor,
            )
            assert (
                await refused(replacement.environment("guess", _guess(attempt, "adieu")))
                == "invalid_attempt"
            )
            # The world was not reached and the ending is the one the budget owes.
            assert len(episode._trajectory) == 2
            state = await gateway.stream_state()
            assert state.attempts[attempt] == "final_failed"
            assert state.final_failures == {attempt: STEP_CAP}
            assert episode.sealed is False
        finally:
            await episode.close()


@pytest.mark.network
async def test_a_replacement_transport_gets_what_is_left_of_the_budget_and_no_more(
    env,
) -> None:
    """Rebuilding mid-attempt buys nothing: the horizon bounds the attempt, not the process.

    The gateway is replaced with steps still to spend, so it has an active route and a world to
    reach, and every path a model has to that world runs through the count the stream keeps.
    Rebuilding again before the last step changes nothing either: what is left is what is left,
    and the trajectory stops at the horizon however many transports served it.
    """
    async with stream_worker(env.client):
        episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        try:
            stream = await start_stream(
                env.client,
                make_start(terminal="terminate", argument_names=()),
                workflow_id="stream/finalize/budget-remainder",
            )
            receipt = await stream.claim_consumer(CONSUMER)
            spec = episode.describe().model_copy(update={"horizon": 3})

            def transport(cursor: str) -> StreamGateway:
                return StreamGateway(
                    stream, episode, spec, terminal_manifest(spec), initial_cursor=cursor
                )

            gateway = transport(receipt.initial_cursor)
            first = json.loads(await gateway.pull({}))
            attempt = first["attempt_id"]
            await gateway.environment("guess", _guess(attempt, "crane"))

            # One step spent, two left, and a transport that knows about neither.
            state = await gateway.stream_state()
            gateway = transport(state.cursor)
            for word in ("slate", "adieu"):
                played = await gateway.environment("guess", _guess(attempt, word))
                assert json.loads(played.content[0].text)["valid"] is True
                gateway = transport((await gateway.stream_state()).cursor)

            assert len(episode._trajectory) == 3
            state = await gateway.stream_state()
            assert state.environment_calls == {attempt: 3}
            assert (
                await refused(gateway.environment("guess", _guess(attempt, "bloke")))
                == "invalid_attempt"
            )
            assert len(episode._trajectory) == 3
            state = await gateway.stream_state()
            assert state.attempts[attempt] == "final_failed"
            assert state.final_failures == {attempt: STEP_CAP}
        finally:
            await episode.close()


@pytest.mark.network
async def test_an_owner_that_restores_the_world_restores_the_budget_with_it(env) -> None:
    """The budget comes back exactly where the world it was spent in does.

    A replacement that puts the attempt back at the checkpoint its task started from is saying
    the calls since then did not happen to the world it is continuing, and the generation writes
    that down: the count it holds for that attempt is what a claim restored it to. So the
    transport that serves the restored world reads a budget as new as the world is, which is the
    same rule as the one that refuses a replacement over a live world, read the other way.
    """
    restored: List[ServedEpisode] = []

    async def open_world(attempt_id: str) -> ServedEpisode:
        started = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        restored.append(started)
        return started

    async with stream_worker(env.client):
        episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        try:
            start = make_start(terminal="terminate", argument_names=())
            stream = await start_stream(
                env.client, start, workflow_id="stream/finalize/budget-restored"
            )
            receipt = await stream.claim_consumer(CONSUMER)
            spec = episode.describe().model_copy(update={"horizon": 2})
            gateway = StreamGateway(
                stream,
                episode,
                spec,
                terminal_manifest(spec),
                initial_cursor=receipt.initial_cursor,
            )
            first = json.loads(await gateway.pull({}))
            attempt = first["attempt_id"]
            for word in ("crane", "slate"):
                await gateway.environment("guess", _guess(attempt, word))
            state = await gateway.stream_state()
            assert state.environment_calls == {attempt: 2}
            # The grants are what a claim has to say it put back, so this is the attempt the
            # replacement below restores rather than simply continues.
            assert state.restoration_required == [attempt]

            replacement = await resume_stream(
                env.client,
                workflow_id=stream.handle.id,
                configuration_hash=configuration_hash(start),
                claimant_id="the-new-owner",
                restored_checkpoints={attempt: state.task_checkpoints[attempt]},
            )
            state = await replacement.stream_state()
            assert state.environment_calls == {}
            world = await open_world(attempt)
            served = StreamGateway(
                replacement,
                world,
                spec,
                terminal_manifest(spec),
                initial_cursor=state.cursor,
                open_episode=open_world,
            )
            for word in ("adieu", "bloke"):
                played = await served.environment("guess", _guess(attempt, word))
                assert json.loads(played.content[0].text)["valid"] is True
            assert len(world._trajectory) == 2
            # And the restored budget is a budget, not an exemption from one.
            assert (
                await refused(served.environment("guess", _guess(attempt, "irate")))
                == "invalid_attempt"
            )
            state = await served.stream_state()
            assert state.final_failures == {attempt: STEP_CAP}
        finally:
            for world in restored:
                await world.close()
            await episode.close()


@pytest.mark.network
@pytest.mark.parametrize("how", ["exhausted", "non_retryable"])
async def test_a_seal_whose_work_finally_failed_ends_the_attempt_it_prepared(env, how) -> None:
    """The batch behind an accepted terminal cannot fail into a state with no way out.

    An attempt is the seal's while the seal can still make progress. Once its work has finally
    failed, whether by exhausting the retries or by declaring itself non-retryable, there is no
    step left for the exact filing to take again, and the attempt is ended from inside the seal.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def grader_that_keeps_failing(request: Any) -> Any:
        raise RuntimeError("the grader is down")

    @activity.defn(name=GRADE_ATTEMPT)
    async def grader_that_will_not_be_retried(request: Any) -> Any:
        raise ApplicationError("this grader cannot score this task", non_retryable=True)

    grader = {
        "exhausted": grader_that_keeps_failing,
        "non_retryable": grader_that_will_not_be_retried,
    }[how]
    async with stream_worker(
        env.client,
        activities=[seal_attempt_activity, grader, generate_payload_bundle_activity],
    ):
        caller = await open_stream(env, workflow_id=f"stream/finalize/seal-{how}")
        await caller.take()
        with pytest.raises(Exception):
            await caller.stream.seal(caller.seal_request())

        # The ending, and nothing else: no acknowledgement was offered, no candidate was built,
        # and the cursor is where the task presentation left it.
        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: SEAL_FAILED}
        assert state.capacity_in_use == 0
        assert state.obligations[ATTEMPT] == "final_failed"
        assert state.materialization_count == 0
        assert state.pending_message_id is None
        assert state.offer_count == 1

        # And the generation can reach Done, which is the whole of what a stranded attempt cost.
        await caller.stream.close_queue()
        done = await caller.pull()
        assert done.kind == "done"


@pytest.mark.network
@pytest.mark.parametrize("how", ["non_retryable", "unusable"])
async def test_a_seal_that_ended_its_attempt_leaves_the_transport_serving(env, how) -> None:
    """The filing that ended an attempt is not one the transport waits for an answer to.

    A filing whose answer never came is kept, because the acknowledgement it may have minted is
    reachable through that request and no other, and every other call is refused meanwhile. A
    batch that ended the attempt minted nothing, so there is nothing to come back for: the
    transport that kept it would otherwise refuse every call it is ever asked, the exact filing
    included, for the rest of the generation. That is true of both endings, the Activity that
    failed for good and the one that answered with a result the seal could not vouch for.

    It is driven through a real MCP client because what the model is handed is part of the
    ending. The generation minted no result and moved no cursor, and the failed tool call is
    still something a harness writes into its transcript, so the bytes of it are pinned: one
    generic failure, no protocol code, and nothing about why this generation could not go on.
    The retry that follows is the contrast, because that one does carry a code.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def grader_that_will_not_be_retried(request: Any) -> Any:
        raise ApplicationError("this grader cannot score this task", non_retryable=True)

    @activity.defn(name=GRADE_ATTEMPT)
    async def score_of_another_seal(request: GradeAttemptInput) -> Any:
        return replace(await grade_attempt_activity(request), seal_id="0" * 64)

    # The third element is what this generation knows and the model may not be told: the words
    # the failure was raised with, the type it was raised as, and the ending it was written as.
    grader, reason, private = {
        "non_retryable": (
            grader_that_will_not_be_retried,
            SEAL_FAILED,
            ("this grader cannot score this task", "ActivityError", SEAL_FAILED),
        ),
        "unusable": (
            score_of_another_seal,
            SEAL_UNUSABLE,
            ("the score is not this seal's", "UnusableActivityResult", SEAL_UNUSABLE),
        ),
    }[how]

    worlds: List[ServedEpisode] = []
    retired: List[bool] = []

    async def open_world(attempt_id: str) -> ServedEpisode:
        started = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        worlds.append(started)
        return started

    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            grader,
            generate_payload_bundle_activity,
        ],
    ):
        episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        closing = episode.close

        async def counted_close(*, finalize: bool = True) -> None:
            retired.append(finalize)
            await closing(finalize=finalize)

        episode.close = counted_close  # type: ignore[method-assign]
        try:
            stream = await start_stream(
                env.client,
                make_start(
                    bodies=("guess the first word", "and the second"),
                    terminal="terminate",
                    argument_names=(),
                ),
                workflow_id=f"stream/finalize/seal-at-the-gateway/{how}",
            )
            receipt = await stream.claim_consumer(CONSUMER)
            spec = episode.describe().model_copy(update={"horizon": 4})
            gateway = StreamGateway(
                stream,
                episode,
                spec,
                terminal_manifest(spec),
                initial_cursor=receipt.initial_cursor,
                open_episode=open_world,
            )
            async with Client(build_gateway_server(gateway)) as client:
                first = json.loads((await client.call_tool(PULL_TOOL, {})).content[0].text)
                attempt = first["attempt_id"]
                filing = {"attempt_id": attempt, "arguments": {}}

                # The filing failed and the caller is told that much and no more. It is not a
                # refusal: the stream accepted the terminal and the batch behind it could not go
                # on, so nothing here carries a protocol code and no result was minted to read.
                # What crosses is the transport's own generic failure, which is exactly what a
                # harness that keeps its own transcript records.
                faulted = await client.call_tool("terminate", filing, raise_on_error=False)
                assert faulted.is_error
                assert faulted.structured_content is None
                assert [block.text for block in faulted.content] == [
                    "Error calling tool 'terminate': Workflow update failed"
                ]
                # And the reason this generation could not go on stayed in this generation.
                assert not [word for word in private if word in faulted.content[0].text]

                state = await gateway.stream_state()
                assert state.attempts[attempt] == "final_failed"
                assert state.final_failures == {attempt: reason}
                assert state.pending_message_id is None

                # The exact filing again is answered about an attempt that is over, rather than
                # raising the same fault for ever, and that answer is a refusal with a code.
                retried = await client.call_tool("terminate", filing, raise_on_error=False)
                assert retried.is_error
                assert json.loads(retried.content[0].text)["code"] == "invalid_attempt"

                # And the next task is served, in a world of its own, with the ended attempt's
                # world retired first and without its lifecycle claiming a second outcome for it.
                second = json.loads((await client.call_tool(PULL_TOOL, {})).content[0].text)
                assert second["kind"] == "task"
                assert second["attempt_id"] != attempt
                assert retired == [False]
                assert len(worlds) == 1
        finally:
            for opened in worlds:
                await opened.close()


@pytest.mark.network
async def test_an_ending_floors_every_attempt_that_was_waiting_on_it(env) -> None:
    """A stop before the outcome writes the floor over the whole of what that outcome covered.

    The filler waits for A's payload and B waits for the filler. Ending A makes both of those
    facts impossible, so both are floored in the same transition rather than left planned behind
    a gate that can no longer open.
    """
    async with stream_worker(env.client):
        caller = await open_stream(
            env,
            make_start(
                bodies=LEG_BODIES, release=LEG_PLAN, without_payload=LEG_WITHOUT_PAYLOAD
            ),
            workflow_id="stream/finalize/leg",
        )
        first = await caller.take()
        assert first.attempt_id == ATTEMPT

        receipt = await caller.finalize(ABANDONED)
        assert sorted(receipt.also_finalized) == [oid(0x104), oid(0x108)]
        state = await caller.stream.stream_state()
        assert state.attempts == {
            ATTEMPT: "final_failed",
            oid(0x104): "final_failed",
            oid(0x108): "final_failed",
        }
        # A's own ending is the one that was asked for; the two behind it were abandoned,
        # because what happened to them is that the fact they waited on stopped being possible.
        assert state.final_failures == {
            ATTEMPT: ABANDONED,
            oid(0x104): ABANDONED,
            oid(0x108): ABANDONED,
        }

        # Nothing is left waiting, so a closed queue is a generation with nothing left to do.
        await caller.stream.close_queue()
        done = await caller.pull()
        assert done.kind == "done"
        assert state.pending_message_id is None


@pytest.mark.network
async def test_a_deadline_that_passes_while_a_world_is_being_called_ends_it_afterwards(
    env,
) -> None:
    """The timer records the expiry where it fires, and the ending waits for the call to settle.

    An environment call holds the generation and reaches a world this stream cannot see, so an
    attempt cannot be ended while one is out: that would be deciding an effect nothing here can
    observe. The expiry is durable meanwhile, and the release is what applies it.
    """
    async with stream_worker(env.client):
        caller = await open_stream(
            env,
            make_start(attempt_deadline_ms=DEADLINE_MS),
            workflow_id="stream/finalize/held",
        )
        await caller.take()
        call = EnvironmentCall(call_id=oid(0x9001), attempt_id=ATTEMPT)
        assert (await caller.stream.begin_environment_call(call)).held is True

        for _ in range(3):
            await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))
        state = await caller.stream.stream_state()
        # The deadline fired and said so, and the attempt is still the held call's.
        assert state.deadline_expired == [ATTEMPT]
        assert state.attempts[ATTEMPT] == "active"
        assert state.final_failures == {}
        # A recorded expiry is not a licence to overtake the call it is waiting for, so a
        # controller's ending contends with that call like every other Update does.
        assert await refused(caller.finalize(ABANDONED)) == "overlapping_call"

        # Giving the grant back settles the effect, and the ending it was waiting for lands.
        await caller.stream.end_environment_call(call)
        state = await _settled(caller)
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: DEADLINE}
        assert state.deadline_expired == []

        # A filing that arrives now is a filing for an attempt that is over.
        assert await refused(caller.stream.seal(caller.seal_request())) == "conflicting_seal"

        await caller.stream.close_queue()
        await caller.present(await caller.pull())
        await caller.stream.handle.result()

    # The deferred expiry is a durable fact like every other, so the history it leaves replays.
    history = await caller.stream.handle.fetch_history()
    await stream_replayer().replay_workflow(history)


@pytest.mark.network
async def test_a_takeover_ends_the_grant_a_deferred_expiry_is_waiting_on(env) -> None:
    """A call that never comes back is ended by the owner that replaces the one holding it.

    The expiry waits for the held call rather than cancelling an effect this stream cannot
    observe, so a call that never returns leaves it waiting. What ends that wait is a takeover:
    the replacement restores the world it is continuing, claims the generation, which fences the
    writer that opened the call, and ends the grant by name. The ending lands then, under the
    new owner, with nothing minted for the model either way.
    """
    start = make_start(attempt_deadline_ms=DEADLINE_MS)
    async with stream_worker(env.client):
        caller = await open_stream(env, start, workflow_id="stream/finalize/takeover")
        await caller.take()
        call = EnvironmentCall(call_id=oid(0x9002), attempt_id=ATTEMPT)
        assert (await caller.stream.begin_environment_call(call)).held is True

        await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))
        state = await caller.stream.stream_state()
        assert state.deadline_expired == [ATTEMPT]
        # The grant is a change this generation authorized, so the attempt is one a claim has
        # to say it put back rather than one it may simply continue.
        assert state.restoration_required == [ATTEMPT]

        replacement = await resume_stream(
            env.client,
            workflow_id=caller.stream.handle.id,
            configuration_hash=configuration_hash(start),
            claimant_id="the-new-owner",
            restored_checkpoints={ATTEMPT: state.task_checkpoints[ATTEMPT]},
        )
        # The writer the claim fenced cannot end its own grant any more, and the one that
        # replaced it can.
        assert await refused(caller.stream.end_environment_call(call)) == "fenced_writer"
        assert (await replacement.end_environment_call(call)).held is True

        second = Caller(replacement, (await replacement.stream_state()).cursor)
        state = await _settled(second)
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: DEADLINE}
        assert state.deadline_expired == []
        assert state.capacity_in_use == 0
        assert state.ownership_epoch == 2
        assert state.pending_message_id is None
        assert state.offer_count == 1


@pytest.mark.network
async def test_an_attestation_built_before_the_deadline_passed_does_not_present_after_it(
    env,
) -> None:
    """A deadline passes on the generation's clock, so it can land inside a delivery.

    Everything else the attestation covers moves on a call, and a call is one thing at a time.
    The expiry is the exception: it is written where the timer fires, which can be while a
    result is out with the harness and its attestation already fixed. That attestation says
    what the stream was before the delivery, so once the expiry is a fact it is stale and the
    harness rebuilds it, rather than committing a description of a stream that has moved.
    """
    async with stream_worker(env.client):
        caller = await open_stream(
            env,
            make_start(attempt_deadline_ms=DEADLINE_MS),
            workflow_id="stream/finalize/attested",
        )
        await caller.take()
        wait = await caller.pull()
        assert wait.kind == "wait"
        before = await caller.stream.stream_state()
        stale = PresentationCommit(
            attestation_id=caller.next_id(),
            cursor_before=before.cursor,
            message_id=wait.message_id,
            visible_bytes_sha256=sha256(wait.visible_text.encode("utf-8")).hexdigest(),
            transcript_blob=TRANSCRIPT_BLOB,
            provider_turn_blob=None,
            task_start_checkpoint_blob=None,
            completed_turn=False,
            stream_state_before_sha256=before.stream_state_sha256,
        )

        await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))
        after = await caller.stream.stream_state()
        # The expiry is recorded and the ending is waiting for the result to be presented, so
        # every other fact this attestation covers is exactly where it was.
        assert after.deadline_expired == [ATTEMPT]
        assert after.attempts[ATTEMPT] == "active"
        assert after.cursor == before.cursor
        assert after.stream_state_sha256 != before.stream_state_sha256
        assert await refused(caller.stream.commit_presentation(stale)) == "invalid_message"

        # Rebuilt against the stream as it is now, the same bytes present, and the ending the
        # expiry was waiting for lands as soon as nothing is outstanding.
        await caller.present(wait)
        state = await _settled(caller)
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: DEADLINE}
        assert state.presentation_count == after.presentation_count + 1


@pytest.mark.network
async def test_an_attestation_the_deadline_overtakes_mid_commit_does_not_present(
    env, tmp_path: Path
) -> None:
    """The projection is compared again after the store has been read, and against the clock.

    A commit verifies the objects its attestation cites, and that read is the one await inside
    it. The deadline runs under that read: an attestation the stream found current on the way
    in can be describing a stream that has moved by the time the read comes back. The pull that
    offered the result and the timer that expires the attempt are not two calls the stream can
    order against each other, so the only place to catch it is the way back in, next to the
    check that the writer is still the owner.
    """
    reading = asyncio.Event()
    holding = asyncio.Event()
    let_go = asyncio.Event()

    @activity.defn(name=VERIFY_BLOBS)
    async def a_read_of_the_store_that_can_be_held(request: VerifyBlobsInput) -> BlobsVerified:
        """Answer for the store, and stay inside the read while this test holds it there."""
        if holding.is_set():
            reading.set()
            await let_go.wait()
        return BlobsVerified(verified=list(request.references), unverified=[])

    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            grade_attempt_activity,
            generate_payload_bundle_activity,
            a_read_of_the_store_that_can_be_held,
        ],
    ):
        caller = await open_stream(
            env,
            make_start(
                attempt_deadline_ms=DEADLINE_INSIDE_A_READ_MS, blob_root=str(tmp_path)
            ),
            workflow_id="stream/finalize/held-read",
        )
        await caller.take()
        wait = await caller.pull()
        assert wait.kind == "wait"
        before = await caller.stream.stream_state()
        current = PresentationCommit(
            attestation_id=caller.next_id(),
            cursor_before=before.cursor,
            message_id=wait.message_id,
            visible_bytes_sha256=sha256(wait.visible_text.encode("utf-8")).hexdigest(),
            transcript_blob=TRANSCRIPT_BLOB,
            provider_turn_blob=None,
            task_start_checkpoint_blob=None,
            completed_turn=False,
            stream_state_before_sha256=before.stream_state_sha256,
        )

        # The commit is sent while it is still current, and held inside the store read.
        holding.set()
        landing = asyncio.ensure_future(caller.stream.commit_presentation(current))
        await asyncio.wait_for(reading.wait(), timeout=30)

        await env.sleep(timedelta(milliseconds=DEADLINE_INSIDE_A_READ_MS + 1000))
        during = await caller.stream.stream_state()
        assert during.deadline_expired == [ATTEMPT]
        assert during.stream_state_sha256 != before.stream_state_sha256

        # The read comes back to a stream the expiry has moved, so the attestation is stale now
        # and nothing of it commits.
        holding.clear()
        let_go.set()
        assert await refused(landing) == "invalid_message"
        after = await caller.stream.stream_state()
        assert after.presentation_count == before.presentation_count
        assert after.cursor == before.cursor
        assert after.pending_message_id == wait.message_id

        # Rebuilt against the stream as it is, the same bytes present, and the ending lands.
        await caller.present(wait)
        state = await _settled(caller)
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: DEADLINE}
        assert state.presentation_count == before.presentation_count + 1


@pytest.mark.network
async def test_an_armed_deadline_survives_the_worker_that_armed_it(env) -> None:
    """The clock an attempt is on is the generation's, and the generation outlives a process.

    The timer is the durable kind, so it is in the history rather than in the Worker that
    started it. It is armed here by one Worker, it fires while there is no Worker at all, and
    the replacement that reads the history is what ends the attempt.
    """
    # Without a sticky cache the server does not hold the stream's tasks for a Worker that has
    # gone away, so the replacement picks the stream up at once and replays it from history.
    async with stream_worker(env.client, cached_workflows=0):
        caller = await open_stream(
            env,
            make_start(attempt_deadline_ms=DEADLINE_MS),
            workflow_id="stream/finalize/replacement",
        )
        await caller.take()
        assert (await caller.stream.stream_state()).attempts[ATTEMPT] == "active"

    await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))

    async with stream_worker(env.client, cached_workflows=0):
        state = await _ended(caller)
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: DEADLINE}
        assert state.capacity_in_use == 0
        # Nothing was minted while there was nobody to mint it, so the model has no more to
        # read now than it had before the Worker went away.
        assert state.pending_message_id is None
        assert state.offer_count == 1

        await caller.stream.close_queue()
        done = await caller.pull()
        assert done.kind == "done"
        await caller.present(done)
        await caller.stream.handle.result()

    history = await caller.stream.handle.fetch_history()
    await stream_replayer().replay_workflow(history)


@pytest.mark.network
async def test_a_negative_deadline_is_a_generation_this_kernel_will_not_serve(env) -> None:
    """Zero is the one value that turns the deadline off, and a typo must not read as zero."""
    async with stream_worker(env.client):
        with pytest.raises(Exception):
            caller = await open_stream(
                env,
                make_start(attempt_deadline_ms=-1),
                workflow_id="stream/finalize/negative",
            )
            await caller.pull()


@pytest.mark.network
async def test_one_logical_finalization_is_one_ending_however_often_it_is_retried(
    caller: Caller,
) -> None:
    """A controller that lost the answer reaches the answer, and not a refusal or a second one."""
    await caller.take()
    request = FinalizeRequest(request_id=caller.next_id(), attempt_id=ATTEMPT, reason=ABANDONED)
    receipt = await caller.stream.finalize(request)

    # The same request again, under a fresh Update so the workflow answers rather than
    # Temporal's own deduplication.
    replayed = await caller.stream.handle.execute_update(
        StreamWorkflow.finalize_attempt,
        args=[request, caller.stream.writer],
        id="a-retry-of-the-ending",
    )
    assert replayed == receipt

    # The same logical ID carrying something else is a conflict rather than a second ending.
    assert (
        await refused(
            caller.stream.finalize(
                FinalizeRequest(
                    request_id=request.request_id, attempt_id=oid(0x104), reason=ABANDONED
                )
            )
        )
        == "request_conflict"
    )
    state = await caller.stream.stream_state()
    assert state.final_failures == {ATTEMPT: ABANDONED}


@pytest.mark.network
async def test_the_ending_a_logical_request_reached_is_the_one_a_new_owner_reads(env) -> None:
    """The map from a logical request to its ending outlives the owner that made the request.

    A controller that lost the answer to an ending is often the process that then goes away,
    and what asks again is the owner that replaced it. Its question is its own Update, because
    every Update ID names the epoch it was sent under, so it reaches the stream rather than the
    answer held by the Update of a writer that has been fenced. What it reaches there is the
    receipt bound to that request when the ending was written, which is what the stream owes a
    logical request, and not a refusal about an attempt that is over. The binding is the thing
    being pinned: clearing it on a claim, or reading it after the state it describes, would
    turn one ending into a conflict for the only caller still asking about it.
    """
    start = make_start()
    async with stream_worker(env.client):
        caller = await open_stream(env, start, workflow_id="stream/finalize/replay-takeover")
        await caller.take()
        request = FinalizeRequest(
            request_id=caller.next_id(), attempt_id=ATTEMPT, reason=ABANDONED
        )
        receipt = await caller.stream.finalize(request)

        replacement = await resume_stream(
            env.client,
            workflow_id=caller.stream.handle.id,
            configuration_hash=configuration_hash(start),
            claimant_id="the-new-owner",
        )
        # The writer that made the request is fenced, and anything it asks now says so.
        fenced = FinalizeRequest(
            request_id=caller.next_id(), attempt_id=ATTEMPT, reason=ABANDONED
        )
        assert await refused(caller.stream.finalize(fenced)) == "fenced_writer"

        # The same logical request, asked by the owner that replaced it, reads the same ending.
        assert await replacement.finalize(request) == receipt
        # And the same ID carrying something else is a conflict under the new owner too.
        assert (
            await refused(
                replacement.finalize(
                    FinalizeRequest(
                        request_id=request.request_id, attempt_id=ATTEMPT, reason=STEP_CAP
                    )
                )
            )
            == "request_conflict"
        )
        state = await replacement.stream_state()
        assert state.ownership_epoch == 2
        assert state.final_failures == {ATTEMPT: ABANDONED}
        assert state.attempts[ATTEMPT] == "final_failed"


class Lossy:
    """The real stream handle, with one named call's response thrown away.

    ``lose`` says which call, and ``applied`` whether the stream got it before the answer went
    missing. Those are the two cuts a durable step has, and the caller cannot tell them apart,
    which is the whole reason the record has to survive both.
    """

    def __init__(self, inner: Any, lose: str, *, applied: bool) -> None:
        self._inner = inner
        self._lose = lose
        self._applied = applied
        self._lost = False
        self.finalize_ids: List[str] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def _maybe_lose(self, which: str, call: Any) -> Any:
        if which != self._lose or self._lost:
            return await call()
        self._lost = True
        if not self._applied:
            raise RuntimeError(f"the {which} never reached the stream")
        await call()
        raise RuntimeError(f"the {which} answer never came back")

    async def finalize(self, request: Any) -> Any:
        self.finalize_ids.append(request.request_id)
        return await self._maybe_lose("finalize", lambda: self._inner.finalize(request))

    async def end_environment_call(self, call: Any) -> Any:
        return await self._maybe_lose(
            "release", lambda: self._inner.end_environment_call(call)
        )


@pytest.mark.network
@pytest.mark.parametrize(
    "lose,applied",
    [("release", False), ("release", True), ("finalize", False), ("finalize", True)],
)
async def test_the_spent_budget_survives_a_lost_response_and_spends_no_second_step(
    env, lose: str, applied: bool
) -> None:
    """The step cap is two durable operations on two calls, and either answer can go missing.

    The call that spends the last step gives the generation back, and the call after it, which
    has nothing left to spend, ends the attempt. What must not happen is the world being called
    twice for one step, or the attempt that ran out being left running because the ending's
    answer went missing. So the observation is held until the release has landed, the ending is
    kept until it has, and the exact call again finishes whichever of the two was left open with
    the same request rather than a second one.
    """
    dispatches = 0
    async with stream_worker(env.client):
        episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        world = episode.call

        async def counted(tool_name: str, arguments: Any) -> Any:
            nonlocal dispatches
            dispatches += 1
            return await world(tool_name, arguments)

        episode.call = counted  # type: ignore[method-assign]
        try:
            stream = await start_stream(
                env.client,
                make_start(
                    bodies=("guess the first word", "and the second"),
                    terminal="terminate",
                    argument_names=(),
                ),
                workflow_id=f"stream/finalize/lost-{lose}-{applied}",
            )
            receipt = await stream.claim_consumer(CONSUMER)
            lossy = Lossy(stream, lose, applied=applied)
            spec = episode.describe().model_copy(update={"horizon": 1})
            gateway = StreamGateway(
                lossy,
                episode,
                spec,
                terminal_manifest(spec),
                initial_cursor=receipt.initial_cursor,
            )
            first = json.loads(await gateway.pull({}))
            attempt = first["attempt_id"]

            async def until_answered(word: str) -> Any:
                """One call, and the exact call again where its answer went missing."""
                try:
                    return await gateway.environment("guess", _guess(attempt, word))
                except Exception:  # noqa: BLE001 - the retry is the subject
                    return await gateway.environment("guess", _guess(attempt, word))

            # The call that spends the last step, whose release may be the lost one.
            played = await until_answered("crane")
            assert json.loads(played.content[0].text)["valid"] is True

            # And the call with nothing to spend, whose ending may be the lost one. Either way
            # it reaches no world and the attempt is over when it has been answered.
            with pytest.raises(Exception):
                await until_answered("slate")

            state = await gateway.stream_state()
            assert dispatches == 1
            assert state.attempts[attempt] == "final_failed"
            assert state.final_failures == {attempt: STEP_CAP}
            assert state.capacity_in_use == 0
            # One logical ending, however many times it had to be sent.
            assert len(set(lossy.finalize_ids)) == 1
        finally:
            await episode.close()


@pytest.mark.network
async def test_a_world_that_will_not_close_leaves_the_next_task_unoffered(env) -> None:
    """The ended attempt's world is retired before the pull that could reserve the next one.

    Nothing presents an ending, so the acknowledgement that normally retires a world never
    comes. Retiring it while preparing an already offered task is too late: a cleanup that
    fails would leave the stream holding a task and the old world still running.
    """
    worlds: List[ServedEpisode] = []
    cleanups = 0

    async def open_world(attempt_id: str) -> ServedEpisode:
        started = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        worlds.append(started)
        return started

    async with stream_worker(env.client):
        episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
        closing = episode.close

        async def close_that_fails_once(*, finalize: bool = True) -> None:
            nonlocal cleanups
            cleanups += 1
            if cleanups == 1:
                raise RuntimeError("the world would not close")
            await closing(finalize=finalize)

        try:
            stream = await start_stream(
                env.client,
                make_start(
                    bodies=("guess the first word", "and the second"),
                    terminal="terminate",
                    argument_names=(),
                ),
                workflow_id="stream/finalize/cleanup",
            )
            receipt = await stream.claim_consumer(CONSUMER)
            spec = episode.describe().model_copy(update={"horizon": 1})
            gateway = StreamGateway(
                stream,
                episode,
                spec,
                terminal_manifest(spec),
                initial_cursor=receipt.initial_cursor,
                open_episode=open_world,
            )
            first = json.loads(await gateway.pull({}))
            await gateway.environment("guess", _guess(first["attempt_id"], "crane"))
            # The budget is one call, so the second one is where the attempt ends.
            assert (
                await refused(gateway.environment("guess", _guess(first["attempt_id"], "slate")))
                == "invalid_attempt"
            )
            episode.close = close_that_fails_once  # type: ignore[method-assign]

            with pytest.raises(Exception):
                await gateway.pull({})
            # The cleanup was tried and it failed, so nothing was offered and no world opened.
            state = await gateway.stream_state()
            assert cleanups == 1
            assert state.pending_message_id is None
            assert worlds == []

            # The call that comes back asks for the cleanup again and then serves the next task.
            second = json.loads(await gateway.pull({}))
            assert second["kind"] == "task"
            assert second["attempt_id"] != first["attempt_id"]
            assert cleanups == 2
            assert len(worlds) == 1
        finally:
            for opened in worlds:
                await opened.close()


async def _settled(caller: Caller) -> Any:
    """Read the stream once the workflow has had a chance to act on what it was told."""
    for _ in range(50):
        state = await caller.stream.stream_state()
        if not state.deadline_expired:
            return state
        await asyncio.sleep(0.05)
    return await caller.stream.stream_state()


async def _ended(caller: Caller, attempt_id: str = ATTEMPT) -> Any:
    """Read the stream once it has had a chance to act on its own clock rather than on a call."""
    for _ in range(50):
        state = await caller.stream.stream_state()
        if attempt_id in state.final_failures:
            return state
        await asyncio.sleep(0.05)
    return await caller.stream.stream_state()


def _guess(attempt_id: str, word: str) -> Dict[str, Any]:
    return {"attempt_id": attempt_id, "arguments": {"word": word}}
