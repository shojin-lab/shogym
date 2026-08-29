"""The durable stream kernel: what one generation will and will not do.

The tests below drive a real workflow through the same Updates a gateway uses, on Temporal's
time-skipping test environment. They are marked ``network`` because that environment downloads
a test server on first use, and they skip rather than fail when it is not there.

What they are pinning is the order the protocol insists on. An offer is a reservation, so the
same request gets the same bytes back until a presentation commits and an error afterwards,
and a different request never inherits the offer. A seal is one transaction, so the
acknowledgement exists only after the score, the candidate bundle, and the released capacity
do. Done is late, so an active task or an undelivered payload keeps it away. And the whole
thing survives losing the worker in the middle, because the workflow is the only authority.

The model-visible bytes are pinned as literals. They are what the harness must insert and what
a presentation attests to, so a test that recomputed them from the same code that produced
them would pin nothing.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from dataclasses import replace
from hashlib import sha256
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.client import (  # noqa: E402
    WorkflowFailureError,
    WorkflowUpdateFailedError,
)
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    PullRequest,
    TerminalMetadata,
    visible_bytes,
)
from shogym.serve.protocol_v2 import Payload  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    ConsumerClaim,
    GeneratePayloadBundleInput,
    OfferedMessage,
    SealAttemptInput,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    generate_payload_bundle_activity,
    grade_attempt_activity,
    hidden_seal_id,
    kernel_activities,
    protocol_error_code,
    seal_attempt_activity,
    start_stream,
    stream_replayer,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GENERATE_PAYLOAD_BUNDLE,
    GRADE_ATTEMPT,
    SEAL_ATTEMPT,
)
from shogym.serve.protocol_v2.kernel.messages import GradeAttemptInput  # noqa: E402

TASK_BODY = "file the report"
FILING = {"answer": "42"}
CLAIM_HASH = "d" * 64
TRANSCRIPT_BLOB = "e" * 64
PROVIDER_TURN_BLOB = "f" * 64
CHECKPOINT_BLOB = "9" * 64
CONSUMER = ConsumerClaim(consumer_id="harness-1", claim_hash=CLAIM_HASH)

# The generation's preallocated public identifiers, and the bytes they end up in. The digest
# is SHA-256 over the length-prefixed domain tag, attempt ID, terminal tool name, and the
# canonical submission "answer='42'"; the payload body quotes its first sixteen characters.
ATTEMPT = "00000000000000000000000000000100"
TASK_ID = "00000000000000000000000000000101"
ACK_ID = "00000000000000000000000000000102"
PAYLOAD_ID = "00000000000000000000000000000103"
DONE_ID = "00000000000000000000000000000002"
DIGEST = "be479ed7d985a6fd522c999eab03a639486f7c7762d1c0479e07b78b69aa4d91"

GOLDEN_TASK = (
    '{"attempt_id":"00000000000000000000000000000100","body":"file the report",'
    '"kind":"task","message_id":"00000000000000000000000000000101","protocol_version":2}'
)
GOLDEN_ACK = (
    '{"attempt_id":"00000000000000000000000000000100","canonicalization_version":"kernel.1",'
    '"kind":"seal_ack","message_id":"00000000000000000000000000000102","protocol_version":2,'
    f'"submission_digest":"{DIGEST}"}}'
)
GOLDEN_PAYLOAD = (
    '{"attempt_id":"00000000000000000000000000000100",'
    f'"body":"receipt 0 for {DIGEST[:16]}",'
    '"kind":"payload","message_id":"00000000000000000000000000000103","protocol_version":2}'
)
GOLDEN_DONE = (
    '{"kind":"done","message_id":"00000000000000000000000000000002","protocol_version":2}'
)
GOLDEN_REJECT_BODY = "missing answer; unknown reply"


def oid(value: int) -> str:
    return f"{value:032x}"


def make_start(*, bodies: Any = (TASK_BODY,), capacity: int = 1, version: int = 2) -> StreamStart:
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
    return StreamStart(
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
        tasks=tasks,
        capacity=capacity,
        protocol_version=version,
    )


class Caller:
    """One authenticated consumer, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: StreamHandle, cursor: str) -> None:
        self.stream = stream
        self.cursor = cursor
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x1000 + self._counter)

    def pull_request(self, request_id: Optional[str] = None) -> PullRequest:
        return PullRequest(
            request_id=request_id or self.next_id(), last_presented_cursor=self.cursor
        )

    async def pull(self, request: Optional[PullRequest] = None) -> OfferedMessage:
        return await self.stream.pull(request or self.pull_request())

    async def present(self, message: OfferedMessage) -> None:
        ack = await self.stream.present(
            message,
            attestation_id=self.next_id(),
            transcript_blob=TRANSCRIPT_BLOB,
            provider_turn_blob=PROVIDER_TURN_BLOB if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=CHECKPOINT_BLOB if message.kind == "task" else None,
        )
        self.cursor = ack.cursor

    def seal_request(
        self,
        attempt_id: str,
        arguments: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> SealRequest:
        return SealRequest(
            metadata=TerminalMetadata(
                request_id=request_id or self.next_id(),
                last_presented_cursor=self.cursor,
                attempt_id=attempt_id,
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
            native_arguments=dict(FILING if arguments is None else arguments),
        )

    async def seal(self, request: Optional[SealRequest] = None) -> OfferedMessage:
        return await self.stream.seal(request or self.seal_request(ATTEMPT))

    async def take(self) -> OfferedMessage:
        """Pull one message and present it, which is what the harness does all day."""
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


async def failed(awaitable: Any) -> str:
    """Return the failure type a call that could not be completed carries.

    A failure is not a refusal. It has no protocol code, because nothing the caller sent is
    what is wrong with it.
    """
    try:
        await awaitable
    except WorkflowUpdateFailedError as error:
        cause = error.cause
        assert isinstance(cause, ApplicationError), cause
        assert protocol_error_code(error) is None, error
        return cause.type or ""
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
    workflow_id: str = "stream/test/1",
) -> Caller:
    stream = await start_stream(
        environment.client, start or make_start(), workflow_id=workflow_id
    )
    receipt = await stream.claim_consumer(CONSUMER)
    return Caller(stream, receipt.initial_cursor)


@pytest_asyncio.fixture
async def caller(env: WorkflowEnvironment) -> AsyncIterator[Caller]:
    async with stream_worker(env.client):
        yield await open_stream(env)


@pytest.mark.network
async def test_one_task_from_offer_to_done(caller: Caller) -> None:
    """Task, SealAck, Payload, Done: the whole loop, byte for byte."""
    task = await caller.pull()
    assert task.visible_text == GOLDEN_TASK
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "task_offered"
    assert state.capacity_in_use == 1

    await caller.present(task)
    assert caller.cursor == TASK_ID
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "active"

    ack = await caller.seal()
    assert ack.visible_text == GOLDEN_ACK
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "sealed"
    assert state.obligations[ATTEMPT] == "eligible"
    # Capacity is released by the seal, not by the acknowledgement reaching anyone.
    assert state.capacity_in_use == 0

    # The acknowledgement is outstanding, and the next pull is invalid until it is presented
    # as the last result of a completed provider turn.
    assert await refused(caller.pull()) == "outstanding_response"
    await caller.present(ack)
    assert caller.cursor == ACK_ID

    payload = await caller.pull()
    assert payload.visible_text == GOLDEN_PAYLOAD
    await caller.present(payload)

    await caller.stream.close_queue()
    done = await caller.pull()
    assert done.visible_text == GOLDEN_DONE
    await caller.present(done)

    outcome = await caller.stream.handle.result()
    assert outcome.generation_state == "done"
    assert outcome.sealed == 1
    assert outcome.payloads_delivered == 1
    assert outcome.cursor == DONE_ID


@pytest.mark.network
async def test_a_full_capacity_waits_and_every_early_poll_is_its_own_message(env) -> None:
    """A busy stream never drains itself, and a poll before readiness is a new Wait."""
    async with stream_worker(env.client):
        caller = await open_stream(env, make_start(bodies=(TASK_BODY, "and the second")))
        await caller.take()

        first = await caller.pull()
        assert first.kind == "wait"
        assert '"retry_after_ms":1000' in first.visible_text
        # A Wait says nothing about why. The record that does is not on the wire.
        assert "capacity" not in first.visible_text
        await caller.present(first)

        second = await caller.pull()
        assert second.kind == "wait"
        assert second.message_id != first.message_id

        state = await caller.stream.stream_state()
        assert state.wait_count == 2
        assert state.tasks_remaining == 1
        assert state.attempts[oid(0x104)] == "planned"


@pytest.mark.network
async def test_an_offer_belongs_to_its_request_until_it_is_presented(caller: Caller) -> None:
    """A retry replays the offer, a changed retry conflicts, a new request never inherits it."""
    request = caller.pull_request()
    task = await caller.pull(request)

    # The same logical request under a fresh Update, so the workflow answers rather than
    # Temporal's own deduplication.
    replayed = await caller.stream.handle.execute_update(
        StreamWorkflow.pull, request, id="replay-of-the-first-pull"
    )
    assert replayed == task

    changed = PullRequest(request_id=request.request_id, last_presented_cursor=oid(0xDEAD))
    assert await refused(caller.stream.pull(changed)) == "request_conflict"
    assert await refused(caller.pull()) == "outstanding_response"
    state = await caller.stream.stream_state()
    assert state.offer_count == 1
    assert state.pending_message_id == TASK_ID

    await caller.present(task)
    assert (
        await refused(
            caller.stream.handle.execute_update(
                StreamWorkflow.pull, request, id="replay-after-presentation"
            )
        )
        == "already_presented"
    )
    stale = PullRequest(request_id=caller.next_id(), last_presented_cursor=oid(1))
    assert await refused(caller.stream.pull(stale)) == "invalid_cursor"


@pytest.mark.network
async def test_a_presentation_is_verified_not_believed(caller: Caller) -> None:
    """Wrong bytes, a stale cursor, and a missing checkpoint are all refused."""
    task = await caller.pull()
    forged = OfferedMessage(
        message_id=task.message_id, kind=task.kind, visible_text="{}", attempt_id=task.attempt_id
    )
    assert (
        await refused(
            caller.stream.present(
                forged,
                attestation_id=caller.next_id(),
                transcript_blob=TRANSCRIPT_BLOB,
                task_start_checkpoint_blob=CHECKPOINT_BLOB,
            )
        )
        == "invalid_message"
    )
    # A Task presentation is the state a crash restores from, so it must carry one.
    assert (
        await refused(
            caller.stream.present(
                task, attestation_id=caller.next_id(), transcript_blob=TRANSCRIPT_BLOB
            )
        )
        == "invalid_message"
    )
    await caller.present(task)


@pytest.mark.network
async def test_a_terminal_request_seals_once_and_replays_until_presented(caller: Caller) -> None:
    """One seal, one score, one acknowledgement, however many times it is asked for."""
    await caller.take()
    request = caller.seal_request(ATTEMPT)
    ack = await caller.seal(request)

    replayed = await caller.stream.handle.execute_update(
        StreamWorkflow.seal_attempt, request, id="replay-of-the-terminal-request"
    )
    assert replayed == ack
    state = await caller.stream.stream_state()
    assert state.offer_count == 2

    await caller.present(ack)
    assert (
        await refused(
            caller.stream.handle.execute_update(
                StreamWorkflow.seal_attempt, request, id="replay-after-ack-presentation"
            )
        )
        == "already_presented"
    )

    # A second filing for a sealed attempt changes nothing, whether the request ID is the old
    # one carrying new arguments or a new one entirely.
    reused = SealRequest(
        metadata=request.metadata,
        public_tool_name="submit",
        native_terminal_name="submit",
        native_arguments={"answer": "43"},
    )
    assert await refused(caller.stream.seal(reused)) == "request_conflict"
    assert (
        await refused(caller.stream.seal(caller.seal_request(ATTEMPT, {"answer": "43"})))
        == "conflicting_seal"
    )
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "ack_presented"
    assert state.obligations[ATTEMPT] == "eligible"


@pytest.mark.network
async def test_malformed_arguments_are_rejected_without_touching_the_attempt(
    caller: Caller,
) -> None:
    """A SealReject is a result like any other, and the attempt stays exactly where it was."""
    await caller.take()
    reject = await caller.seal(caller.seal_request(ATTEMPT, {"reply": "42"}))
    assert reject.kind == "seal_reject"
    assert f'"body":"{GOLDEN_REJECT_BODY}"' in reject.visible_text
    assert '"code":"invalid_arguments"' in reject.visible_text
    state = await caller.stream.stream_state()
    assert state.attempts[ATTEMPT] == "active"

    # A corrected filing waits for the refusal to be presented, like every other result.
    assert await refused(caller.stream.seal(caller.seal_request(ATTEMPT))) == "outstanding_response"
    await caller.present(reject)
    ack = await caller.seal()
    assert ack.visible_text == GOLDEN_ACK


@pytest.mark.network
@pytest.mark.parametrize("corrupted", ["seal", "grade", "grade_seal", "bundle", "bundle_filing"])
async def test_a_result_the_seal_cannot_vouch_for_never_becomes_an_acknowledgement(
    env: WorkflowEnvironment, corrupted: str
) -> None:
    """A seal stops where its batch is assembled when a result does not describe what it built.

    An Activity that answers for another attempt, or another seal, or another filing, or that
    records measurements of something other than the body it carries, acknowledges nothing.
    """

    @activity.defn(name=SEAL_ATTEMPT)
    async def seal_of_another_attempt(request: SealAttemptInput) -> Any:
        return replace(await seal_attempt_activity(request), attempt_id=oid(0x999))

    @activity.defn(name=GRADE_ATTEMPT)
    async def score_of_another_attempt(request: GradeAttemptInput) -> Any:
        return replace(await grade_attempt_activity(request), attempt_id=oid(0x999))

    @activity.defn(name=GRADE_ATTEMPT)
    async def score_of_another_seal(request: GradeAttemptInput) -> Any:
        # The public attempt ID is branch-neutral, so a score that carries it is not yet a score
        # for this branch's filing. Two fork children share the ID and not the seal.
        return replace(await grade_attempt_activity(request), seal_id="0" * 64)

    @activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
    async def bundle_measuring_something_else(request: GeneratePayloadBundleInput) -> Any:
        # Every hash and count it records describes the body it replaced, which is exactly what
        # a build gate and a matched envelope read before anything is served.
        bundle = await generate_payload_bundle_activity(request)
        [candidate] = bundle.candidates
        return replace(bundle, candidates=[replace(candidate, body="a body nobody measured")])

    @activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
    async def bundle_built_for_another_filing(request: GeneratePayloadBundleInput) -> Any:
        # Internally flawless and still the wrong bundle: it is the requested attempt at the
        # requested position, and every measurement describes the body it carries, but the body
        # was rendered from a submission this obligation never filed.
        return await generate_payload_bundle_activity(replace(request, submission_digest="0" * 64))

    served = {
        "seal": [seal_of_another_attempt, grade_attempt_activity, generate_payload_bundle_activity],
        "grade": [seal_attempt_activity, score_of_another_attempt, generate_payload_bundle_activity],
        "grade_seal": [
            seal_attempt_activity,
            score_of_another_seal,
            generate_payload_bundle_activity,
        ],
        "bundle": [seal_attempt_activity, grade_attempt_activity, bundle_measuring_something_else],
        "bundle_filing": [
            seal_attempt_activity,
            grade_attempt_activity,
            bundle_built_for_another_filing,
        ],
    }[corrupted]
    async with stream_worker(env.client, activities=served):
        caller = await open_stream(env, workflow_id=f"stream/unusable/{corrupted}")
        await caller.take()
        assert await failed(caller.seal()) == "UnusableActivityResult"
        # The seal stopped before the transition it could not have completed truthfully: no
        # acknowledgement was offered, no obligation was materialized, and the attempt is still
        # the one that was being sealed.
        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "sealing"
        assert state.obligations[ATTEMPT] == "assigned"
        assert state.pending_message_id is None


@pytest.mark.network
async def test_done_waits_for_the_last_payload(env) -> None:
    """A closed and empty queue is not enough: a live task or an undelivered payload is."""
    async with stream_worker(env.client):
        caller = await open_stream(env)
        await caller.stream.close_queue()
        task = await caller.take()
        assert task.kind == "task"
        early = await caller.pull()
        assert early.kind == "wait"
        await caller.present(early)

        ack = await caller.seal()
        await caller.present(ack)
        payload = await caller.pull()
        assert payload.kind == "payload"
        await caller.present(payload)
        assert (await caller.pull()).kind == "done"


@pytest.mark.network
async def test_the_generation_has_one_consumer_and_one_call_in_flight(env) -> None:
    """A second consumer and an overlapping call are both refused before any mutation."""

    started = asyncio.Event()

    @activity.defn(name=SEAL_ATTEMPT)
    async def slow_seal(request: SealAttemptInput) -> Any:
        started.set()
        await asyncio.sleep(2)
        return await seal_attempt_activity(request)

    activities = [slow_seal, grade_attempt_activity, generate_payload_bundle_activity]
    async with stream_worker(env.client, activities=activities):
        stream = await start_stream(env.client, make_start(), workflow_id="stream/test/2")
        unclaimed = PullRequest(request_id=oid(9), last_presented_cursor=oid(1))
        assert await refused(stream.pull(unclaimed)) == "consumer_conflict"
        receipt = await stream.claim_consumer(CONSUMER)
        caller = Caller(stream, receipt.initial_cursor)
        assert (
            await refused(stream.claim_consumer(ConsumerClaim("harness-2", CLAIM_HASH)))
            == "consumer_conflict"
        )
        # The same consumer claiming twice is a lost response, not a second consumer.
        assert (await stream.claim_consumer(CONSUMER)).claim_epoch == receipt.claim_epoch

        await caller.take()
        sealing = asyncio.create_task(caller.seal())
        await asyncio.wait_for(started.wait(), timeout=30)
        assert await refused(caller.pull()) == "overlapping_call"
        assert (await sealing).kind == "seal_ack"


@pytest.mark.network
async def test_the_stream_outlives_its_worker(env) -> None:
    """Stop the worker mid-attempt, start another, and the stream carries on where it was."""
    # Without a sticky cache the server does not hold the stream's tasks for a Worker that has
    # gone away, so the replacement picks the stream up at once and replays it from history.
    async with stream_worker(env.client, cached_workflows=0):
        caller = await open_stream(env, workflow_id="stream/test/3")
        await caller.take()

    async with stream_worker(env.client, cached_workflows=0):
        ack = await caller.seal()
        assert ack.visible_text == GOLDEN_ACK
        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "sealed"
        await caller.present(ack)
        await caller.present(await caller.pull())
        await caller.stream.close_queue()
        await caller.present(await caller.pull())
        assert (await caller.stream.handle.result()).sealed == 1

    history = await caller.stream.handle.fetch_history()
    await stream_replayer().replay_workflow(history)


@pytest.mark.network
async def test_a_version_one_generation_never_starts(env) -> None:
    """The kernel serves protocol v2 and refuses to guess at anything else."""
    async with stream_worker(env.client):
        stream = await start_stream(
            env.client, make_start(version=1), workflow_id="stream/test/4"
        )
        with pytest.raises(WorkflowFailureError) as caught:
            await stream.handle.result()
        assert protocol_error_code(caught.value.cause) == "unsupported_version"


async def test_a_quickstart_install_never_imports_temporal() -> None:
    """Importing shogym must not pull in Temporal. The durable path is opt in."""
    probe = "import shogym, sys; assert 'temporalio' not in sys.modules, sorted(sys.modules)"
    subprocess.run([sys.executable, "-c", probe], check=True)


async def test_the_payload_bundle_measures_the_complete_result() -> None:
    """A family gate compares complete serialized results, so that is what is measured."""
    bundle = await generate_payload_bundle_activity(
        GeneratePayloadBundleInput(
            attempt_id=ATTEMPT,
            payload_position=0,
            payload_message_id=PAYLOAD_ID,
            submission_digest=DIGEST,
            canonical_submission_text="answer='42'",
        )
    )
    [candidate] = bundle.candidates
    serialized = visible_bytes(
        Payload(message_id=PAYLOAD_ID, attempt_id=ATTEMPT, body=candidate.body)
    )
    assert candidate.visible_sha256 == sha256(serialized).hexdigest()
    assert candidate.visible_byte_count == len(serialized)
    assert candidate.inner_sha256 != candidate.visible_sha256


async def test_the_seal_key_is_branch_local_while_the_attempt_id_is_not() -> None:
    """Two executions that share a public attempt ID must not share an environment seal."""
    keys: List[str] = [
        hidden_seal_id("execution-1", 0, ATTEMPT),
        hidden_seal_id("execution-2", 0, ATTEMPT),
        hidden_seal_id("execution-1", 1, ATTEMPT),
    ]
    assert len(set(keys)) == 3
    assert hidden_seal_id("execution-1", 0, ATTEMPT) == keys[0]
    assert all(ATTEMPT not in key for key in keys)


def test_the_worker_registers_the_stream_and_its_three_activities() -> None:
    """The names a Worker serves are the names the workflow schedules."""
    names = {activity.__temporal_activity_definition.name for activity in kernel_activities()}
    assert names == {
        "shogym.protocol_v2.SealAttemptActivity",
        "shogym.protocol_v2.GradeAttemptActivity",
        "shogym.protocol_v2.GeneratePayloadBundleActivity",
    }
