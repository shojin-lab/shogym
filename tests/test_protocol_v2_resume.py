"""Taking a generation over, and what the writer it replaced may still do.

Temporal replays a history, so the state comes back on its own. What these tests are about is
everything replay does not decide: which process is allowed to write next, whether it is
resuming the generation it thinks it is, and whether the bytes an event references exist.

The cuts are the contract's, and they are made two ways. The worker is stopped before an offer,
after one, after a lost response, around a presentation commit, and with an acknowledgement
offered and unpresented, and a new worker under a new epoch picks the stream up. A seal is cut
the other way, because the interesting moment is the one Temporal will not lose: the call is in
flight when the new owner claims. Either way the generation has to end in the same place: the
same message replayed, the same acknowledgement returned, one seal, one presentation, and no
identifier minted twice.

Fencing is proved with a handle that still works. The old owner reaches the generation exactly
as it did before, and the generation refuses it, which is a fence rather than a lost socket.

These are marked ``network`` because the time-skipping environment downloads a test server on
first use, and they skip rather than fail when it is not there.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.client import WorkflowFailureError, WorkflowUpdateFailedError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    NEVER,
    FilesystemBlobStore,
    PresentationCommit,
    PullRequest,
    TerminalMetadata,
    blob_ref,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    STREAM_TASK_QUEUE,
    ConsumerClaim,
    EnvironmentCall,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    OfferedMessage,
    OwnershipClaim,
    SealAttemptInput,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    VerifyBlobsInput,
    Writer,
    assignments_for,
    configuration_hash,
    generate_payload_bundle_activity,
    grade_attempt_activity,
    protocol_error_code,
    resume_run_directory,
    resume_stream,
    seal_attempt_activity,
    start_stream,
    stream_replayer,
    stream_worker,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GENERATE_PAYLOAD_BUNDLE,
    GRADE_ATTEMPT,
    SEAL_ATTEMPT,
    VERIFY_BLOBS,
)
from shogym.serve.protocol_v2.gateway import install_policies  # noqa: E402
from shogym.serve.protocol_v2.rundir import (  # noqa: E402
    ResumeRefused,
    create_run_directory,
)
from tests._fixtures.policy_rows import registering_the_receipt  # noqa: E402

CLAIM_HASH = "d" * 64
BLOB = "e" * 64
CONSUMER = ConsumerClaim(consumer_id="harness-1", claim_hash=CLAIM_HASH)

ATTEMPT = "00000000000000000000000000000100"
TASK_ID = "00000000000000000000000000000101"
ACK_ID = "00000000000000000000000000000102"
DONE_ID = "00000000000000000000000000000002"


def oid(value: int) -> str:
    return f"{value:032x}"


def make_start(*, version: int = 2) -> StreamStart:
    """One task, every public identifier fixed before anything is served."""
    return registering_the_receipt(
        StreamStart(
            configuration_hash="c" * 64,
            consumer_claim_hash=CLAIM_HASH,
            initial_cursor=oid(1),
            done_message_id=DONE_ID,
            id_key_hex="ab" * 32,
            hidden_execution_id="execution-1",
            canonicalization_version="kernel.1",
            terminal_tool=TerminalTool(
                public_tool_name="submit", native_terminal_name="submit", argument_names=["answer"]
            ),
            tasks=[
                TaskItem(
                    task_position=0,
                    attempt_id=ATTEMPT,
                    task_message_id=TASK_ID,
                    ack_message_id=ACK_ID,
                    payload_position=0,
                    payload_message_id=oid(0x103),
                    body="file the report",
                )
            ],
            protocol_version=version,
        )
    )


class Caller:
    """One owner of one generation, keeping the cursor its requests carry."""

    def __init__(self, stream: StreamHandle, cursor: str, counter: int = 0) -> None:
        self.stream = stream
        self.cursor = cursor
        self.counter = counter

    def next_id(self) -> str:
        self.counter += 1
        return oid(0x1000 + self.counter)

    def pull_request(self) -> PullRequest:
        return PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)

    async def pull(self, request: Optional[PullRequest] = None) -> OfferedMessage:
        return await self.stream.pull(request or self.pull_request())

    async def presentation_for(
        self, message: OfferedMessage, *, transcript_blob: str = BLOB
    ) -> PresentationCommit:
        """Build the attestation for ``message`` against the state it would commit from."""
        state = await self.stream.stream_state()
        completed = message.kind == "seal_ack"
        return PresentationCommit(
            attestation_id=self.next_id(),
            cursor_before=state.cursor,
            message_id=message.message_id,
            visible_bytes_sha256=sha256(message.visible_text.encode("utf-8")).hexdigest(),
            transcript_blob=transcript_blob,
            provider_turn_blob=transcript_blob if completed else None,
            task_start_checkpoint_blob=transcript_blob if message.kind == "task" else None,
            completed_turn=completed,
            stream_state_before_sha256=state.stream_state_sha256,
        )

    async def present(self, message: OfferedMessage, *, transcript_blob: str = BLOB) -> None:
        commit = await self.presentation_for(message, transcript_blob=transcript_blob)
        ack = await self.stream.commit_presentation(commit)
        self.cursor = ack.cursor

    def seal_request(self, arguments: Optional[Dict[str, Any]] = None) -> SealRequest:
        return SealRequest(
            metadata=TerminalMetadata(
                request_id=self.next_id(),
                last_presented_cursor=self.cursor,
                attempt_id=ATTEMPT,
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
            native_arguments=dict(arguments or {"answer": "42"}),
        )

    async def take(self) -> OfferedMessage:
        message = await self.pull()
        await self.present(message)
        return message


async def refused(awaitable: Any) -> str:
    """Return the protocol error code a refused call carries."""
    try:
        await awaitable
    except WorkflowUpdateFailedError as error:
        code = protocol_error_code(error)
        assert code is not None, error
        return code
    raise AssertionError("the call was accepted")


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await WorkflowEnvironment.start_time_skipping()
    except Exception as error:
        pytest.skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


def worker(environment: WorkflowEnvironment, activities: Optional[List[Any]] = None) -> Any:
    """A worker with no sticky cache, so its replacement picks the stream up at once."""
    return stream_worker(environment.client, activities=activities, cached_workflows=0)


async def open_stream(
    environment: WorkflowEnvironment,
    start: StreamStart,
    *,
    workflow_id: str = "stream/resume/1",
) -> Caller:
    stream = await start_stream(environment.client, start, workflow_id=workflow_id)
    receipt = await stream.claim_consumer(CONSUMER)
    return Caller(stream, receipt.initial_cursor)


async def take_over(
    environment: WorkflowEnvironment,
    caller: Caller,
    start: StreamStart,
    *,
    restored: Optional[Dict[str, str]] = None,
) -> Caller:
    """Claim a generation whose last writer is gone, and read the cursor from the authority."""
    stream = await resume_stream(
        environment.client,
        workflow_id=caller.stream.handle.id,
        configuration_hash=configuration_hash(start),
        claimant_id="the-new-owner",
        restored_checkpoints=restored,
    )
    state = await stream.stream_state()
    return Caller(stream, state.cursor, counter=caller.counter)


@pytest.mark.network
async def test_a_resume_fences_the_writer_it_replaced(env: WorkflowEnvironment) -> None:
    """One writer at a time. The old handle still calls, and the generation refuses it."""
    start = make_start()
    async with worker(env):
        first = await open_stream(env, start)
        task = await first.pull()
        second = await take_over(env, first, start)
        state = await second.stream.stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (2, 2)
        assert state.fencing_token_hash is not None

        assert await refused(first.pull()) == "fenced_writer"
        assert await refused(first.present(task)) == "fenced_writer"
        assert await refused(first.stream.close_queue()) == "fenced_writer"
        assert await refused(first.stream.seal(first.seal_request())) == "fenced_writer"
        # The query answers both of them, so the read that can be refused is refused too, and
        # the writer that holds the generation is still answered by it.
        assert await refused(first.stream.confirm_state()) == "fenced_writer"
        assert (await second.stream.confirm_state()).ownership_epoch == 2
        after = await second.stream.stream_state()
        assert (after.cursor, after.offer_count, after.presentation_count) == (oid(1), 1, 0)
        assert after.queue_closed is False
        assert after.attempts[ATTEMPT] == "task_offered"

        # A claimant that read a stale epoch loses the swap, and loses it without mutation.
        late = StreamHandle(second.stream.handle)
        assert (
            await refused(
                late.claim_ownership(
                    configuration_hash=configuration_hash(start),
                    previous_epoch=1,
                    claimant_id="late",
                    reason="resume",
                )
            )
            == "fenced_writer"
        )
        assert (await second.stream.stream_state()).ownership_epoch == 2
        # The offer the fenced owner was holding is still the generation's one offer.
        await second.present(task)
        assert (await second.stream.stream_state()).cursor == TASK_ID


@pytest.mark.network
async def test_a_fenced_writer_does_not_reach_the_world_either(
    env: WorkflowEnvironment,
) -> None:
    """The one call that never reaches this stream is refused by this stream all the same.

    A writer that has been replaced still holds an episode, a session, or a world of its own,
    and an ordinary environment call goes to that world rather than here. It is only fenced if
    the generation is asked before the world is touched, so it is: the permission that call
    needs is an Update like any other, and the epoch is checked before anything else about it.
    """
    start = make_start()
    async with worker(env):
        first = await open_stream(env, start)
        await first.present(await first.pull())
        state = await first.stream.stream_state()
        assert state.attempts[ATTEMPT] == "active"

        second = await take_over(env, first, start)
        call = EnvironmentCall(call_id=oid(0x3001), attempt_id=ATTEMPT)
        assert await refused(first.stream.begin_environment_call(call)) == "fenced_writer"
        # Nothing was granted, so the owner that replaced it can still take the generation.
        assert (await second.stream.begin_environment_call(call)).held is True
        assert (await second.stream.end_environment_call(call)).held is True


@pytest.mark.network
async def test_a_claim_does_not_grant_a_second_lease_over_an_unresolved_one(
    env: WorkflowEnvironment,
) -> None:
    """A grant the generation made is not released by replacing the writer that holds it.

    Every other call a claim finds in flight is refused where it comes back, so releasing the
    stream costs nothing. An environment call is the one that cannot be: it is changing a world
    this stream never sees, and the grant is the whole of what said it could. So the hold is
    carried over instead of dropped, and the generation grants no second call until the new
    owner has ended the first one by name.

    The owner that carries it is one that restored the attempt, because a grant is also the
    thing that says this attempt's world has moved past the checkpoint it would come back from.
    """
    start = make_start()
    async with worker(env):
        first = await open_stream(env, start)
        await first.present(await first.pull())
        held = EnvironmentCall(call_id=oid(0x3001), attempt_id=ATTEMPT)
        assert (await first.stream.begin_environment_call(held)).held is True

        second = await take_over(env, first, start, restored={ATTEMPT: BLOB})
        assert (await second.stream.stream_state()).environment_call == held.call_id
        another = EnvironmentCall(call_id=oid(0x3002), attempt_id=ATTEMPT)
        assert await refused(second.stream.begin_environment_call(another)) == "overlapping_call"
        assert await refused(second.pull()) == "overlapping_call"

        # The writer it replaced may not give the lease back, so the new owner ends that call.
        assert await refused(first.stream.end_environment_call(held)) == "fenced_writer"
        assert (await second.stream.end_environment_call(held)).held is True
        assert (await second.stream.stream_state()).environment_call is None
        later = EnvironmentCall(call_id=oid(0x3003), attempt_id=ATTEMPT)
        assert (await second.stream.begin_environment_call(later)).held is True


@pytest.mark.network
async def test_an_active_attempt_comes_back_only_over_a_world_nothing_happened_in(
    env: WorkflowEnvironment,
) -> None:
    """An active attempt is continued by a new owner only if it is the one it was left as.

    A crash right after the Task is the case a resume is for: the attempt is active, nothing has
    happened in its world, and the checkpoint it would come back from is still what the stream
    is holding. A crash after the generation has authorized a call to that world is the other
    case, and it looks identical from here, which is exactly why the claim has to say which of
    the two it is. So the generation counts what it authorized and holds the claim to it: an
    owner that names the attempt and the checkpoint it put back continues, and one that names
    nothing is refused with the attempt, the epoch and the world where they were.

    What restoring means is not this generation's to check, because the world is not one it can
    reach. What it can check is that the claim is about an attempt this generation is holding
    active and about the exact bytes it made that attempt's checkpoint, and it does.
    """
    start = make_start()
    async with worker(env):
        first = await open_stream(env, start)
        await first.take()
        state = await first.stream.stream_state()
        assert state.attempts[ATTEMPT] == "active"
        assert state.task_checkpoints == {ATTEMPT: BLOB}
        assert state.restoration_required == []

        # Nothing has happened since that checkpoint, so this takeover restores nothing.
        second = await take_over(env, first, start)
        assert (await second.stream.stream_state()).ownership_epoch == 2

        # Now the world moves, once, and the generation is the thing that said it could.
        call = EnvironmentCall(call_id=oid(0x3001), attempt_id=ATTEMPT)
        assert (await second.stream.begin_environment_call(call)).held is True
        assert (await second.stream.end_environment_call(call)).held is True
        state = await second.stream.stream_state()
        assert state.attempts[ATTEMPT] == "active"
        assert state.restoration_required == [ATTEMPT]

        claimant = StreamHandle(second.stream.handle)
        claim = dict(
            configuration_hash=configuration_hash(start),
            previous_epoch=2,
            claimant_id="the-third-owner",
            reason="resume",
        )
        assert await refused(claimant.claim_ownership(**claim)) == "invalid_attempt"
        # A claim about other bytes, or about an attempt this generation is not holding active,
        # is not a restoration either.
        for named in ({ATTEMPT: BLOB[:63] + "f"}, {oid(0x9999): BLOB}):
            wrong = claimant.claim_ownership(**claim, restored_checkpoints=named)
            assert await refused(wrong) == "invalid_attempt"

        # Nothing moved under any of them, and the owner that has the generation still has it.
        state = await second.stream.stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (2, 2)
        assert state.attempts[ATTEMPT] == "active"
        assert (await second.stream.begin_environment_call(call)).held is True
        assert (await second.stream.end_environment_call(call)).held is True

        # The owner that put the world back names it, and continues the attempt it restored.
        receipt = await claimant.claim_ownership(**claim, restored_checkpoints={ATTEMPT: BLOB})
        assert receipt.ownership_epoch == 3
        third = Caller(claimant, (await claimant.stream_state()).cursor, counter=second.counter)
        state = await claimant.stream_state()
        assert state.restoration_required == []
        assert state.task_checkpoints == {ATTEMPT: BLOB}
        acknowledgement = await third.stream.seal(third.seal_request())
        assert acknowledgement.kind == "seal_ack"


@pytest.mark.network
async def test_a_claim_that_believes_another_configuration_fences_nobody(
    env: WorkflowEnvironment,
) -> None:
    """A resume of something else is refused before the epoch moves, so nothing is fenced."""
    start = make_start()
    async with worker(env):
        first = await open_stream(env, start)
        changed = replace(start, capacity=2)
        claimant = StreamHandle(first.stream.handle)
        assert (
            await refused(
                claimant.claim_ownership(
                    configuration_hash=configuration_hash(changed),
                    previous_epoch=1,
                    claimant_id="a-changed-generation",
                    reason="resume",
                )
            )
            == "configuration_mismatch"
        )
        state = await first.stream.stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (1, 1)
        assert state.configuration_hash == configuration_hash(start)
        # The writer that held the stream still holds it.
        assert (await first.pull()).message_id == TASK_ID


@pytest.mark.network
async def test_a_generation_nobody_ever_claimed_is_still_taken(
    env: WorkflowEnvironment,
) -> None:
    """Creating a generation is a Workflow and then a claim, and a process can die between them.

    What it leaves is a generation that exists and has never had a writer. Nothing was served
    under it, because no call passes the ownership check while no token is installed, so the
    claim that takes it is a first claim rather than a replacement, and the resume path makes it.
    """
    start = make_start()
    async with worker(env):
        handle = await env.client.start_workflow(
            StreamWorkflow.run,
            start,
            id="stream/resume/unclaimed",
            task_queue=STREAM_TASK_QUEUE,
        )
        state = await StreamHandle(handle).stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (0, 0)
        never = Writer(ownership_epoch=0, fencing_token="a" * 64)
        assert (
            await refused(
                handle.execute_update(StreamWorkflow.claim_consumer, args=[CONSUMER, never])
            )
            == "fenced_writer"
        )

        taken = await resume_stream(
            env.client,
            workflow_id="stream/resume/unclaimed",
            configuration_hash=configuration_hash(start),
            claimant_id="the-recovering-owner",
        )
        receipt = await taken.claim_consumer(CONSUMER)
        state = await taken.stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (1, 1)
        assert (await Caller(taken, receipt.initial_cursor).pull()).message_id == TASK_ID


@pytest.mark.network
async def test_the_cuts_around_an_offer_replay_it_and_mint_nothing(
    env: WorkflowEnvironment,
) -> None:
    """Before the offer, after it, and after a lost response: the same message, once."""
    start = make_start()
    async with worker(env):
        caller = await open_stream(env, start)

    # Nothing had been offered, so the new owner offers the generation's first message.
    async with worker(env):
        caller = await take_over(env, caller, start)
        request = caller.pull_request()
        task = await caller.pull(request)
        assert task.message_id == TASK_ID

    # The offer stands and its response was lost. The exact request replays those bytes, a new
    # request does not inherit them, and neither one is a second offer.
    async with worker(env):
        caller = await take_over(env, caller, start)
        assert await caller.pull(request) == task
        assert await refused(caller.pull()) == "outstanding_response"
        state = await caller.stream.stream_state()
        assert (state.offer_count, state.presentation_count) == (1, 0)
        assert state.pending_message_id == TASK_ID
        await caller.present(task)
        polling = caller.pull_request()
        wait = await caller.pull(polling)
        assert wait.kind == "wait"

    # A Wait draws its identifier from the keyed stream, so replaying one is where a resume
    # would mint a second identifier if it minted anything at all.
    async with worker(env):
        caller = await take_over(env, caller, start)
        assert await caller.pull(polling) == wait
        state = await caller.stream.stream_state()
        assert (state.wait_count, state.offer_count) == (1, 2)
        assert state.ownership_epoch == 4


@pytest.mark.network
async def test_a_new_owner_reads_back_the_call_a_reserved_result_is_owed_to(
    env: WorkflowEnvironment,
) -> None:
    """The only retry that reaches a reserved result is the exact request that asked for it.

    That request's identity is the caller's, minted where the call was made, so a process that
    replaces the one that made it holds none of it and every fresh request it could invent is
    refused while the result stands. The generation names the call instead: what a replacement
    cannot keep, the authority already has.
    """
    start = make_start()
    async with worker(env):
        caller = await open_stream(env, start)
        request = caller.pull_request()
        task = await caller.pull(request)

    async with worker(env):
        taken = await take_over(env, caller, start)
        state = await taken.stream.stream_state()
        assert state.pending_origin == "pull"
        assert state.pending_request_id == request.request_id
        # The cursor is the one that result was offered against, because a reservation and an
        # advanced cursor cannot both be true, so the exact request rebuilds from the Query
        # alone. It is the retry, not a second one: nothing new is offered.
        rebuilt = PullRequest(
            request_id=str(state.pending_request_id), last_presented_cursor=state.cursor
        )
        assert rebuilt == request
        assert await taken.stream.pull(rebuilt) == task
        assert (await taken.stream.stream_state()).offer_count == 1

        # A reserved acknowledgement says the same about the filing that earned it.
        await taken.present(task)
        filing = taken.seal_request()
        await taken.stream.seal(filing)
        state = await taken.stream.stream_state()
        assert state.pending_origin == "terminal"
        assert state.pending_request_id == filing.metadata.request_id


@pytest.mark.network
async def test_a_committed_presentation_replays_its_acknowledgement(
    env: WorkflowEnvironment,
) -> None:
    """The commit landed and the receipt was lost. The same commit returns the same receipt."""
    start = make_start()
    async with worker(env):
        caller = await open_stream(env, start)
        task = await caller.pull()
        commit = await caller.presentation_for(task)
        acknowledgement = await caller.stream.commit_presentation(commit)

    async with worker(env):
        caller = await take_over(env, caller, start)
        assert caller.cursor == TASK_ID
        assert await caller.stream.commit_presentation(commit) == acknowledgement
        state = await caller.stream.stream_state()
        assert (state.presentation_count, state.cursor) == (1, TASK_ID)
        assert state.attempts[ATTEMPT] == "active"
        # A committed presentation is never administered again, under any attestation.
        second = await caller.presentation_for(task)
        assert await refused(caller.stream.commit_presentation(second)) == "already_presented"


@pytest.mark.network
@pytest.mark.parametrize("cut", [SEAL_ATTEMPT, GRADE_ATTEMPT, GENERATE_PAYLOAD_BUNDLE])
async def test_a_prepared_seal_outlives_the_owner_that_prepared_it(
    env: WorkflowEnvironment, cut: str
) -> None:
    """Fenced before the seal is prepared, after it, and on the edge of the batch.

    The cut is the fence landing on a call that is already in flight, which is the state a
    worker death leaves once Temporal redelivers the Update, without waiting for an Activity
    timeout to say so. Nothing commits in any of the three, and the exact terminal request
    continues the same prepared seal rather than starting a second one.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    async def paused(inner: Any, request: Any) -> Any:
        started.set()
        await release.wait()
        return await inner(request)

    @activity.defn(name=SEAL_ATTEMPT)
    async def paused_seal(request: SealAttemptInput) -> Any:
        return await paused(seal_attempt_activity, request)

    @activity.defn(name=GRADE_ATTEMPT)
    async def paused_grade(request: GradeAttemptInput) -> Any:
        return await paused(grade_attempt_activity, request)

    @activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
    async def paused_bundle(request: GeneratePayloadBundleInput) -> Any:
        return await paused(generate_payload_bundle_activity, request)

    served = {
        SEAL_ATTEMPT: seal_attempt_activity,
        GRADE_ATTEMPT: grade_attempt_activity,
        GENERATE_PAYLOAD_BUNDLE: generate_payload_bundle_activity,
    }
    served[cut] = {
        SEAL_ATTEMPT: paused_seal,
        GRADE_ATTEMPT: paused_grade,
        GENERATE_PAYLOAD_BUNDLE: paused_bundle,
    }[cut]
    activities = list(served.values()) + [verify_blobs_activity]
    start = make_start()
    async with worker(env, activities):
        first = await open_stream(env, start)
        await first.take()
        request = first.seal_request()
        sealing = asyncio.create_task(first.stream.seal(request))
        await asyncio.wait_for(started.wait(), timeout=30)

        second = await take_over(env, first, start)
        release.set()
        assert await refused(sealing) == "fenced_writer"

        # The seal was prepared and nothing about it became authoritative.
        state = await second.stream.stream_state()
        assert state.attempts[ATTEMPT] == "sealing"
        assert (state.obligations[ATTEMPT], state.materialization_count) == ("assigned", 0)
        assert (state.capacity_in_use, state.offer_count) == (1, 1)

        # The stream is not held by the call that was fenced, and the exact terminal request
        # continues the prepared seal rather than starting a second one.
        acknowledgement = await second.stream.seal(request)
        assert acknowledgement.message_id == ACK_ID
        state = await second.stream.stream_state()
        assert state.attempts[ATTEMPT] == "sealed"
        assert (state.materialization_count, state.offer_count) == (1, 2)

        await second.present(acknowledgement)
        await second.present(await second.pull())
        await second.stream.close_queue()
        await second.present(await second.pull())
        outcome = await second.stream.handle.result()
        assert (outcome.sealed, outcome.payloads_delivered) == (1, 1)

    # A history holding a claim, a fenced call, and a continued seal still replays.
    await stream_replayer().replay_workflow(await second.stream.handle.fetch_history())


@pytest.mark.network
async def test_a_new_owner_continues_a_seal_it_kept_nothing_of(env: WorkflowEnvironment) -> None:
    """A filing prepared by a process that is gone, continued by one that never saw it.

    A prepared seal reserves no result, so the pending fields say nothing about it and the
    attempt being in ``sealing`` is all the state a replacement can see. What continuing it
    takes is the identity the filing was made under, and that was minted in the process that
    died. The generation names it per attempt, which is the same answer it gives for a reserved
    result. The filing itself is not the generation's to hand back: the arguments are the model's
    own tool call, and the harness rebuilds them from the transcript it wrote them into.
    """
    started = asyncio.Event()
    release = asyncio.Event()

    @activity.defn(name=SEAL_ATTEMPT)
    async def paused_seal(request: SealAttemptInput) -> Any:
        started.set()
        await release.wait()
        return await seal_attempt_activity(request)

    activities = [
        paused_seal,
        grade_attempt_activity,
        generate_payload_bundle_activity,
        verify_blobs_activity,
    ]
    start = make_start()
    async with worker(env, activities):
        first = await open_stream(env, start)
        await first.take()
        filing = first.seal_request()
        sealing = asyncio.create_task(first.stream.seal(filing))
        await asyncio.wait_for(started.wait(), timeout=30)

        second = await take_over(env, first, start)
        release.set()
        assert await refused(sealing) == "fenced_writer"

        state = await second.stream.stream_state()
        assert state.attempts[ATTEMPT] == "sealing"
        assert (state.pending_origin, state.pending_request_id) == (None, None)
        assert state.prepared_seals == {ATTEMPT: filing.metadata.request_id}

        # Any other identity is a second filing for an attempt that already has one.
        assert await refused(second.stream.seal(second.seal_request())) == "conflicting_seal"
        assert (await second.stream.stream_state()).attempts[ATTEMPT] == "sealing"

        # The identity and the cursor come from the Query and the arguments from the transcript,
        # which is the whole of the filing, so this is the retry rather than a second seal.
        rebuilt = SealRequest(
            metadata=TerminalMetadata(
                request_id=state.prepared_seals[ATTEMPT],
                last_presented_cursor=state.cursor,
                attempt_id=ATTEMPT,
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
            native_arguments={"answer": "42"},
        )
        assert rebuilt == filing
        acknowledgement = await second.stream.seal(rebuilt)
        assert acknowledgement.message_id == ACK_ID
        after = await second.stream.stream_state()
        assert (after.attempts[ATTEMPT], after.prepared_seals) == ("sealed", {})
        assert (after.materialization_count, after.offer_count) == (1, 2)


@pytest.mark.network
async def test_an_acknowledgement_offered_before_the_cut_is_replayed_not_reissued(
    env: WorkflowEnvironment,
) -> None:
    """The seal committed and its bytes were lost. The retry replays them, and only them."""
    start = make_start()
    async with worker(env):
        caller = await open_stream(env, start)
        await caller.take()
        request = caller.seal_request()
        acknowledgement = await caller.stream.seal(request)

    async with worker(env):
        caller = await take_over(env, caller, start)
        assert await caller.stream.seal(request) == acknowledgement
        state = await caller.stream.stream_state()
        assert (state.attempts[ATTEMPT], state.offer_count) == ("sealed", 2)
        assert state.materialization_count == 1

        await caller.present(acknowledgement)
        # Under a fresh Update, so the workflow answers rather than Temporal's deduplication.
        assert (
            await refused(
                caller.stream.handle.execute_update(
                    StreamWorkflow.seal_attempt,
                    args=[request, caller.stream.writer],
                    id="the-terminal-request-once-more",
                )
            )
            == "already_presented"
        )
        assert (
            await refused(caller.stream.seal(caller.seal_request({"answer": "43"})))
            == "conflicting_seal"
        )


@pytest.mark.network
async def test_a_blob_is_verified_before_an_event_may_reference_it(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """A reference to bytes nobody installed is a reference to nothing, and is refused."""
    root = tmp_path / "run"
    start = replace(make_start(), blob_root=str(FilesystemBlobStore.under(root).root))
    install_policies(FilesystemBlobStore.under(root), start)
    run = create_run_directory(
        root,
        workflow_id="stream/resume/blobs",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash=configuration_hash(start),
    )
    async with worker(env):
        caller = await open_stream(env, start, workflow_id="stream/resume/blobs")
        assert (await caller.stream.stream_state()).blob_verification == "required"
        task = await caller.pull()

        absent = blob_ref("a transcript nobody installed").sha256
        assert await refused(caller.present(task, transcript_blob=absent)) == "invalid_message"
        state = await caller.stream.stream_state()
        assert (state.presentation_count, state.cursor) == (0, oid(1))

        installed = run.blobs.put(b"the transcript, as presented").sha256
        await caller.present(task, transcript_blob=installed)
        assert (await caller.stream.stream_state()).cursor == TASK_ID

        # An object that stopped hashing to its own name is refused like an absent one.
        run.blobs.path_for(installed).write_bytes(b"a different transcript")
        acknowledgement = await caller.stream.seal(caller.seal_request())
        assert (
            await refused(caller.present(acknowledgement, transcript_blob=installed))
            == "invalid_message"
        )

        # The directory is what the next owner resumes from, and it fences this one. The object
        # a committed event already references is put back first, because a resume reads those.
        run.blobs.path_for(installed).write_bytes(b"the transcript, as presented")
        taken = await resume_run_directory(
            env.client, root, start=start, claimant_id="the-next-owner"
        )
        assert (await taken.stream_state()).ownership_epoch == 2
        assert await refused(caller.pull()) == "fenced_writer"


@pytest.mark.network
async def test_a_resume_reads_what_the_committed_events_reference(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """A presentation's blobs are verified when it commits, and again when an owner changes.

    The first read says the objects were there then, which is not the same as now: the store is
    a directory and the bytes under a name can stop being the bytes that name promised. So the
    whole committed set is read again before the epoch moves, and a generation whose history
    cites something the store can no longer produce is not handed to a new owner.
    """
    root = tmp_path / "run"
    start = replace(make_start(), blob_root=str(FilesystemBlobStore.under(root).root))
    install_policies(FilesystemBlobStore.under(root), start)
    run = create_run_directory(
        root,
        workflow_id="stream/resume/committed",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash=configuration_hash(start),
    )
    async with worker(env):
        caller = await open_stream(env, start, workflow_id="stream/resume/committed")
        installed = run.blobs.put(b"the transcript the task was presented in").sha256
        await caller.present(await caller.pull(), transcript_blob=installed)

        run.blobs.path_for(installed).write_bytes(b"something else under that name")
        claimant = StreamHandle(caller.stream.handle)
        claim = dict(
            configuration_hash=configuration_hash(start),
            previous_epoch=1,
            claimant_id="the-next-owner",
            reason="resume",
        )
        assert await refused(claimant.claim_ownership(**claim)) == "invalid_message"
        # Nothing moved, so the owner that has it still has it.
        state = await caller.stream.stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (1, 1)
        assert (await caller.pull()).kind == "wait"

        # The same claim, once the store can produce what the history cites.
        run.blobs.path_for(installed).write_bytes(b"the transcript the task was presented in")
        receipt = await claimant.claim_ownership(**claim)
        assert receipt.ownership_epoch == 2


@pytest.mark.network
async def test_a_resume_reads_what_committed_while_it_was_reading(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The writer this claim replaces is still the writer while the store is being read.

    So a claim that read the committed set once would be checking a snapshot. A presentation
    the old owner had already begun can commit inside that read, and its references are then
    part of the history the claim is about to hand on and outside the set the claim checked.
    The read is repeated over whatever the history cites that this claim has not read yet, and
    the pass that finds nothing new is the one with no await between it and the swap.
    """
    root = tmp_path / "run"
    start = replace(make_start(), blob_root=str(FilesystemBlobStore.under(root).root))
    install_policies(FilesystemBlobStore.under(root), start)
    run = create_run_directory(
        root,
        workflow_id="stream/resume/reading",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash=configuration_hash(start),
    )
    reads: List[Tuple[str, ...]] = []
    pause_on: List[Tuple[str, ...]] = []
    reading = asyncio.Event()
    release = asyncio.Event()

    @activity.defn(name=VERIFY_BLOBS)
    async def watched_verify(request: VerifyBlobsInput) -> Any:
        references = tuple(request.references)
        reads.append(references)
        if pause_on and references == pause_on[0]:
            pause_on.pop(0)
            reading.set()
            await release.wait()
        return await verify_blobs_activity(request)

    activities = [
        seal_attempt_activity,
        grade_attempt_activity,
        generate_payload_bundle_activity,
        watched_verify,
    ]
    async with worker(env, activities):
        caller = await open_stream(env, start, workflow_id="stream/resume/reading")
        task_blob = run.blobs.put(b"the transcript the task was presented in").sha256
        wait_blob = run.blobs.put(b"the transcript the wait was presented in").sha256
        await caller.present(await caller.pull(), transcript_blob=task_blob)

        # The preimage of the policy this generation resolved to is a reference from the start,
        # so it is in every set a claim reads.
        descriptor = start.dispositions[0].policy_digest or ""
        pause_on.append((descriptor, task_blob))
        claimant = StreamHandle(caller.stream.handle)
        claim = dict(
            configuration_hash=configuration_hash(start),
            previous_epoch=1,
            claimant_id="the-next-owner",
            reason="resume",
        )
        claiming = asyncio.create_task(claimant.claim_ownership(**claim))
        await asyncio.wait_for(reading.wait(), timeout=30)

        # The old owner commits while the claim is inside the store, and the object it cites
        # is one the store then cannot produce.
        waiting = await caller.pull()
        assert waiting.kind == "wait"
        await caller.present(waiting, transcript_blob=wait_blob)
        run.blobs.path_for(wait_blob).write_bytes(b"something else under that name")
        release.set()

        # The first read is the creating claim's, which is the descriptor and nothing else: a
        # generation that cannot produce the bytes its own record names is refused before it
        # serves rather than at the resume that noticed. The second is the task presentation's,
        # the third is the resuming claim's, the fourth is the wait presentation's, and the fifth
        # is that claim reading what it had not read.
        assert await refused(claiming) == "invalid_message"
        assert reads == [
            (descriptor,),
            (task_blob,),
            (descriptor, task_blob),
            (wait_blob,),
            (wait_blob,),
        ]
        # Nothing moved, so the owner that has it still has it.
        state = await caller.stream.stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (1, 1)
        assert (await caller.pull()).kind == "wait"

        # The same claim, once the store can produce what the history cites.
        run.blobs.path_for(wait_blob).write_bytes(b"the transcript the wait was presented in")
        receipt = await claimant.claim_ownership(**claim)
        assert receipt.ownership_epoch == 2
        assert reads[-1] == (descriptor, task_blob, wait_blob)


@pytest.mark.network
async def test_a_directory_resume_is_held_to_what_its_new_owner_serves(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The directory says which generation. What it is being resumed as is the claimant's own.

    A resume that read the hash back out of the manifest and presented that to the authority
    would compare one recorded value with a copy of itself, and a replacement serving a changed
    queue, plan, capacity or environment would pass it. So the hash is derived from the
    composition the resuming process serves, and a process composed for something else does not
    get the generation.
    """
    root = tmp_path / "run"
    start = replace(make_start(), blob_root=str(FilesystemBlobStore.under(root).root))
    install_policies(FilesystemBlobStore.under(root), start)
    create_run_directory(
        root,
        workflow_id="stream/resume/configuration",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash=configuration_hash(start),
    )
    changed = replace(start, capacity=2)
    async with worker(env):
        caller = await open_stream(env, start, workflow_id="stream/resume/configuration")
        with pytest.raises(ResumeRefused) as caught:
            await resume_run_directory(
                env.client, root, start=changed, claimant_id="the-changed-runner"
            )
        assert caught.value.code == "configuration_mismatch"
        # Refused before the authority was asked, so the owner that has it still has it.
        state = await caller.stream.stream_state()
        assert (state.ownership_epoch, state.ownership_claims) == (1, 1)
        assert (await caller.pull()).kind == "task"

        # The same directory, taken by a process composed for the generation it holds.
        taken = await resume_run_directory(
            env.client, root, start=start, claimant_id="the-next-owner"
        )
        assert (await taken.stream_state()).ownership_epoch == 2


@pytest.mark.network
async def test_a_generation_that_says_two_versions_serves_nothing(
    env: WorkflowEnvironment,
) -> None:
    """Protocol one never starts, an unserved schedule never starts, and neither may call."""
    mixed = replace(make_start(), schedule_version="shogym.schedule.0")
    async with worker(env):
        for index, refusable in enumerate((make_start(version=1), mixed)):
            handle = await env.client.start_workflow(
                StreamWorkflow.run,
                refusable,
                id=f"stream/versions/{index}",
                task_queue=STREAM_TASK_QUEUE,
            )
            with pytest.raises(WorkflowFailureError) as caught:
                await handle.result()
            assert protocol_error_code(caught.value.cause) == "unsupported_version"

        # A call that says version one is refused whatever the generation says, and a claim
        # that says it is refused before the compare and swap.
        caller = await open_stream(env, make_start(), workflow_id="stream/versions/live")
        current = caller.stream.writer
        aged = Writer(
            ownership_epoch=current.ownership_epoch,
            fencing_token=current.fencing_token,
            protocol_version=1,
        )
        assert (
            await refused(
                caller.stream.handle.execute_update(
                    StreamWorkflow.pull,
                    args=[caller.pull_request(), aged],
                    id="a-version-one-pull",
                )
            )
            == "unsupported_version"
        )
        assert (
            await refused(
                caller.stream.handle.execute_update(
                    StreamWorkflow.claim_ownership,
                    OwnershipClaim(
                        claimant_id="version-one",
                        previous_epoch=1,
                        fencing_token="f" * 64,
                        configuration_hash=configuration_hash(make_start()),
                        reason="resume",
                        protocol_version=1,
                    ),
                    id="a-version-one-claim",
                )
            )
            == "unsupported_version"
        )
        assert (await caller.stream.stream_state()).ownership_epoch == 1


def test_the_configuration_hash_covers_everything_a_resume_is_held_to() -> None:
    """Change anything the generation is, and a resume of the old one no longer matches."""
    start = make_start()
    assert configuration_hash(start) == configuration_hash(make_start())
    changed = [
        replace(start, capacity=2),
        replace(start, tasks=[replace(start.tasks[0], body="another order")]),
        replace(start, release=NEVER),
        replace(start, assignments=assignments_for(start.tasks, NEVER)),
        replace(start, schedule_version="shogym.schedule.2"),
        replace(start, protocol_version=3),
        replace(start, configuration_hash="f" * 64),
        replace(start, canonicalization_version="kernel.2"),
        replace(start, id_key_hex="cd" * 32),
        replace(start, wait_retry_after_ms=2000),
        replace(start, attempt_deadline_ms=600_000),
        replace(start, budget=52),
    ]
    hashes = {configuration_hash(one) for one in changed}
    assert len(hashes) == len(changed)
    assert configuration_hash(start) not in hashes

    # Where a run keeps its bytes is deployment, and the same generation moved is the same one.
    assert configuration_hash(replace(start, blob_root="/elsewhere")) == configuration_hash(start)

    # A generation that declares no budget is what everything above is, and it hashes what it
    # hashed before there was a budget to declare. The proof of that is a history this code did
    # not write, replayed in the policy suite against the digest its own build committed.
    assert start.budget is None
