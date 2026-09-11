"""A generation that outlives one execution: what crosses the boundary and what does not.

The durable service counts the Updates one execution accepted and refuses the next one past its
cap. A roster longer than that cap therefore needs more than one execution, and the generation
gets one by continuing as new at a quiet point well before the cap, under the same workflow
identifier, carrying the whole logical projection with it.

The promise under test is that nothing about that is visible. A run that crossed several
boundaries answers every public call, publishes every state field, records every attempt and
presents every message exactly as the same run that crossed none. So most of what is here is a
twin: two generations driven through the same steps, one with the trigger lowered until it turns
over and one with it left alone, compared field by field.

The rest is the machinery that makes the twin true. The admission gate is what bounds the work
accepted between the decision to turn over and the boundary, and it is tested under continuous
traffic with the one Activity that can hold a handler open deliberately paused. The exact-outcome
cache is what answers a retry that carries an Update identifier the execution before it answered,
and it is tested per slot against the transport's own recovery records. The Activity identifiers
are what a failure recorded after a boundary publishes, and they are tested against the twin.

Most of these drive a real workflow on Temporal's time-skipping environment and are marked
``durable``, because the binary that environment runs on is prepared before the suite is. The few
that are about the service's cap itself need the local dev service, because the time-skipping
server does not enforce the cap and takes no dynamic configuration to lower it; those are marked
``dev_server`` and ``network``, the dev binary being a download nothing prepares, and they skip
unless ``SHOGYM_DEV_SERVER_TESTS`` is set.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import time
import typing
from hashlib import sha256
from types import SimpleNamespace
from contextlib import asynccontextmanager
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio.common import RetryPolicy  # noqa: E402
from temporalio.client import (  # noqa: E402
    Client,
    WorkflowUpdateFailedError,
)
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    PresentationCommit,
    canonical_json,
    IMMEDIATE,
    PAYLOAD_FIRST,
    RELEASE_AT_SEAL,
    EligibilityGate,
    PresentationAck,
    ReleasePlan,
    InfoRequest,
    PullRequest,
    TerminalMetadata,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    CARRIER_SCHEMA_VERSIONS,
    STEP_CAP,
    STREAM_TASK_QUEUE,
    BlobsVerified,
    CarriedProjection,
    ConsumerClaim,
    EnvironmentCall,
    FinalizeRequest,
    OfferedMessage,
    OwnershipClaim,
    OwnershipReceipt,
    SealRequest,
    StreamCarry,
    StreamHandle,
    StreamStart,
    StreamState,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    Writer,
    assignments_for,
    configuration_hash,
    durable_client,
    kernel_activities,
    protocol_error_code,
    refuse_a_carried_projection,
    resume_stream,
    start_stream,
    stream_replayer,
    stream_worker,
    turnover_pending,
)
from temporalio.converter import default as default_converter  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GRADE_ATTEMPT,
    VERIFY_BLOBS,
    grade_attempt_activity,
)
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    CarriedAttempt,
    GradeAttemptInput,
    GradeAttemptResult,
    VerifyBlobsInput,
    carrier_version,
    pack_carrier,
    unpack_carrier,
)
from shogym.serve.protocol_v2.kernel.runtime import temporal_home  # noqa: E402
from shogym.serve.protocol_v2.reader import _refuse_a_moved_history  # noqa: E402
from tests._fixtures.policy_rows import registering_the_receipt  # noqa: E402
from tests._fixtures.temporal_server import time_skipping_environment  # noqa: E402
from tests._fixtures.upstream_gate import environmental_skip  # noqa: E402

CLAIM_HASH = "d" * 64
BLOB = "e" * 64
CONSUMER = ConsumerClaim(consumer_id="harness-1", claim_hash=CLAIM_HASH)

DEV_SERVER_ONLY = pytest.mark.skipif(
    not os.environ.get("SHOGYM_DEV_SERVER_TESTS"),
    reason="the dev service is slow and runs in real time; set SHOGYM_DEV_SERVER_TESTS to run it",
)


def oid(value: int) -> str:
    return f"{value:032x}"


def attempt(index: int) -> str:
    return oid(0x100 + index * 4)


def make_start(*, tasks: int = 3, release=IMMEDIATE, body: str = "file the report") -> StreamStart:
    """One generation, with every public identifier fixed before it serves anything."""
    items = [
        TaskItem(
            task_position=index,
            attempt_id=attempt(index),
            task_message_id=oid(0x101 + index * 4),
            ack_message_id=oid(0x102 + index * 4),
            payload_position=index,
            payload_message_id=oid(0x103 + index * 4),
            body=f"{body} {index}",
        )
        for index in range(tasks)
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
                public_tool_name="submit", native_terminal_name="submit", argument_names=[]
            ),
            tasks=items,
            release=release,
            assignments=assignments_for(items, release),
        )
    )


def _asking_start(root: str, tasks: int = 3) -> StreamStart:
    """A generation that declares the info tool and a store, for the contract's sake.

    The tool so that asking how much queue is left is one of the operations compared. The store
    so that the generation actually reads it: a claim verifies before it swaps the epoch and every
    presentation verifies its references, and a generation given no store does neither and leaves
    the counters that record those reads at zero on both sides of every comparison.
    """
    return replace(make_start(tasks=tasks), info=True, blob_root=root)


@activity.defn(name=VERIFY_BLOBS)
async def _a_store_that_answers(payload: VerifyBlobsInput) -> BlobsVerified:
    """A store that produces whatever is asked of it, so the reads happen and none of them fail."""
    return BlobsVerified(verified=list(payload.references), unverified=[])


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await time_skipping_environment()
    except Exception as error:  # noqa: BLE001 - an unusable server is the machine's, not the test's
        environmental_skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


@pytest.fixture
def turnover_at(monkeypatch: pytest.MonkeyPatch):
    """Lower the trigger so a boundary is a handful of Updates away rather than seventeen hundred.

    The constant is a module attribute and the workflow sandbox passes this package through, so
    the workflow reads the value this sets. Lowering it is the only way to reach the decision
    point in a test: the real trigger is above anything a suite would drive.
    """

    def lower(updates: int, *, reserve: Optional[int] = None) -> None:
        monkeypatch.setattr(kernel_workflow, "TURNOVER_TRIGGER", updates)
        if reserve is not None:
            monkeypatch.setattr(kernel_workflow, "ADMISSION_RESERVE", reserve)

    return lower


def _a_carrier(start: StreamStart, **overrides: Any) -> StreamCarry:
    """A syntactically complete projection for ``start``, for the tests about the pairing."""
    projection = CarriedProjection(
        configuration_hash=configuration_hash(start),
        ownership_epoch=4,
        fencing_token_hash="a" * 64,
        ownership_claims=2,
        consumer_id="harness-1",
        claim_epoch=1,
        cursor=oid(9),
        queue_closed=True,
        hidden_ordinal=3,
        seal_ordinal=2,
        wait_count=1,
        wait_reasons={"gate": 1},
        offer_count=7,
        eligibility_count=2,
        handed_out_attempt_ids=[attempt(0)],
        activity_ordinal=5,
        verification_batches=3,
        attempts=[],
        obligations=[],
        presented=[],
        committed_blobs=[],
        pull_requests=[],
        info_requests=[],
        terminal_requests=[],
        finalize_requests=[],
        attestations=[],
    )
    if overrides:
        projection = replace(projection, **overrides)
    return pack_carrier(
        projection,
        default_converter().payload_converter,
        version=carrier_version(start, projection),
    )


def _unpacked(carry: StreamCarry) -> CarriedProjection:
    """The projection one carrier holds, for the tests that read a carrier back."""
    return unpack_carrier(carry, default_converter().payload_converter)


def _a_generation_holding_nothing() -> StreamState:
    """One state reading of a generation that is reaching for a boundary and holding nothing.

    It is the reading the wait rule is hardest on: traffic is being held back, and nothing about
    the generation says why it cannot get to its boundary. Every other case in that test is this
    one with a single field moved.
    """
    return StreamState(
        generation_state="open",
        cursor=oid(1),
        configuration_hash="c" * 64,
        stream_state_sha256="a" * 64,
        ownership_epoch=1,
        fencing_token_hash="d" * 64,
        ownership_claims=1,
        blob_verification="unchecked",
        consumer_id="harness-1",
        queue_closed=True,
        tasks_remaining=1,
        capacity=1,
        capacity_in_use=0,
        pending_message_id=None,
        pending_kind=None,
        pending_origin=None,
        pending_request_id=None,
        environment_call=None,
        prepared_seals={},
        task_checkpoints={},
        graded_evidence={},
        environment_calls={},
        restoration_required=[],
        attempts={},
        obligations={},
        release_plan_id="plan",
        release_predicate="at_seal",
        assignment_count=1,
        materialization_count=0,
        eligibility_count=0,
        offer_count=0,
        presentation_count=0,
        payload_delivery_count=0,
        wait_count=0,
        wait_reasons={},
        final_failures={},
        turnover_requested=True,
    )


async def refused(awaitable: Any) -> str:
    """Return the protocol error code a refused call carries."""
    try:
        await awaitable
    except WorkflowUpdateFailedError as error:
        code = protocol_error_code(error)
        assert code is not None, error
        return code
    raise AssertionError("the call was accepted")


async def rejected(awaitable: Any) -> bool:
    """Whether the generation rejected this Update because it is between executions.

    A rejection is not a refusal: it carries no protocol code, because nothing the caller sent
    is wrong. So it is read by its own type rather than by the closed set, and a test that
    confused the two would be asserting the agent had been shown an error.
    """
    try:
        await awaitable
    except WorkflowUpdateFailedError as error:
        assert protocol_error_code(error) is None, "a rejection is not a protocol refusal"
        return turnover_pending(error)
    raise AssertionError("the call was accepted")


class Caller:
    """One consumer of one generation, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: StreamHandle, cursor: str) -> None:
        self.stream = stream
        self.cursor = cursor
        self.acknowledgements: List[PresentationAck] = []
        self.reference: Callable[[OfferedMessage], str] = lambda message: BLOB
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x2000 + self._counter)

    async def pull(self, request: Optional[PullRequest] = None) -> OfferedMessage:
        return await self.stream.pull(
            request
            or PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )

    async def attest(self, message: OfferedMessage) -> PresentationAck:
        """Present one message and keep the acknowledgement the generation answered with.

        Every acknowledgement is kept, not only the ones a caller happens to look at. They are
        what a presentation is answered with, they carry the cursor and the projection hash the
        next request is built against, and a comparison that dropped the ones taken inside a
        helper would be comparing a chosen subset of the answers rather than the answers.

        ``self.reference`` is what a presentation cites for the objects it names. One constant is
        enough for a test about protocol steps, and it is the default. A test about what a
        generation's projection encodes to needs what a harness really cites, which is a digest
        of the bytes it stored, so that caller replaces this with one.
        """
        reference = self.reference(message)
        ack = await self.stream.present(
            message,
            attestation_id=self.next_id(),
            transcript_blob=reference,
            provider_turn_blob=reference if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=reference if message.kind == "task" else None,
        )
        self.cursor = ack.cursor
        self.acknowledgements.append(ack)
        return ack

    async def present(self, message: OfferedMessage) -> None:
        await self.attest(message)

    async def take(self) -> OfferedMessage:
        message = await self.pull()
        await self.present(message)
        return message

    async def work(self, task: OfferedMessage) -> OfferedMessage:
        acknowledgement = await self.stream.seal(
            SealRequest(
                metadata=TerminalMetadata(
                    request_id=self.next_id(),
                    last_presented_cursor=self.cursor,
                    attempt_id=task.attempt_id or "",
                ),
                public_tool_name="submit",
                native_terminal_name="submit",
            )
        )
        await self.present(acknowledgement)
        return acknowledgement

    async def take_info(self) -> OfferedMessage:
        """Ask how much of the queue there is, and present the answer."""
        answer = await self.stream.info(
            InfoRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )
        await self.present(answer)
        return answer

    async def call_the_world(self, task: OfferedMessage) -> None:
        """Take the generation for one environment call and give it straight back."""
        call = EnvironmentCall(call_id=self.next_id(), attempt_id=task.attempt_id or "")
        await self.stream.begin_environment_call(call)
        await self.stream.end_environment_call(call)


async def open_stream(
    client: Client, start: StreamStart, *, workflow_id: str
) -> Caller:
    stream = await start_stream(client, start, workflow_id=workflow_id)
    receipt = await stream.claim_consumer(CONSUMER)
    return Caller(stream, receipt.initial_cursor)


async def served(caller: Caller, *, limit: int = 400) -> List[OfferedMessage]:
    """Drive one generation to Done the way a model would, and report what it was served."""
    seen: List[OfferedMessage] = []
    for _ in range(limit):
        message = await caller.take()
        seen.append(message)
        if message.kind == "done":
            return seen
        if message.kind == "task":
            seen.append(await caller.work(message))
    raise AssertionError("the generation never reached Done")


def _instead_of_verification(stub: Any) -> List[Any]:
    """The kernel's Activities with the store read replaced by one a test can hold open."""
    from shogym.serve.protocol_v2.kernel.activities import verify_blobs_activity

    return [row for row in kernel_activities() if row is not verify_blobs_activity] + [stub]


async def _resume(
    client: Client, start: StreamStart, *, workflow_id: str, claimant: str
) -> StreamHandle:
    """Take a running generation over, the way a replacement process does.

    A replacement says which active attempts it put back and from which checkpoint, because a
    generation holding an attempt whose world it has authorized a change to refuses a claim that
    says nothing about it. The state names both, so this asks it rather than assuming a
    generation that granted nothing.
    """
    handle = StreamHandle(client.get_workflow_handle_for(StreamWorkflow.run, workflow_id))
    standing = await handle.stream_state()
    return await resume_stream(
        client,
        workflow_id=workflow_id,
        configuration_hash=configuration_hash(start),
        claimant_id=claimant,
        restored_checkpoints={
            attempt_id: standing.task_checkpoints[attempt_id]
            for attempt_id in standing.restoration_required
        },
    )


def _encoded(start: StreamStart) -> int:
    """How many bytes the configured converter turns one start into.

    The same converter the service is given, so this is the number the service would measure
    against its own limit rather than an estimate of it.
    """
    return default_converter().payload_converter.to_payloads([start])[0].ByteSize()


def _in_the_background(work: Any) -> "asyncio.Task[Any]":
    """Start work that is meant to be abandoned, and remember to collect what it raised."""
    return asyncio.ensure_future(work)


async def _let_go(task: "asyncio.Task[Any]") -> None:
    """Cancel a background task and consume whatever it ended with.

    A task cancelled without its result being read leaves the loop complaining at teardown about
    an exception nobody retrieved, which is noise in every later run of the file. These tasks are
    deliberately abandoned, so the abandoning is where the answer gets read.
    """
    task.cancel()
    try:
        await task
    except BaseException:  # noqa: BLE001 - the point is that nothing here is worth raising
        pass


async def _straight_at_the_generation(
    caller: "Caller", update: Any, *args: Any, update_id: str
) -> Any:
    """Send one Update without the transport's wait, so a rejection surfaces at once.

    The transport answers a rejection by waiting for the boundary, which is right for a caller
    and wrong for a test whose subject is what the generation admits. This is the same Update
    the transport would send, sent without that answer wrapped around it.
    """
    return await caller.stream.handle.execute_update(update, args=list(args), id=update_id)


async def _accepted_updates(client: Client, workflow_id: str) -> int:
    """How many Updates the execution this generation is in has accepted, from its history."""
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    return sum(
        1
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED
    )


async def run_ids(client: Client, workflow_id: str) -> List[str]:
    """Every execution this generation ran in, from the first, by following each close event.

    A replay of the latest execution alone can pass while a predecessor's own continuation
    command was nondeterministic, so a check that means anything walks the chain rather than
    fetching the history a bare handle reaches.
    """
    described = await client.get_workflow_handle(workflow_id).describe()
    run_id = described.raw_description.workflow_execution_info.first_run_id
    chain = [run_id]
    while True:
        history = await client.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
        closing = history.events[-1]
        closed = closing.workflow_execution_continued_as_new_event_attributes
        successor = closed.new_execution_run_id
        if not successor:
            return chain
        chain.append(successor)
        run_id = successor


# What holds without a server at all.


def test_the_trigger_and_the_reserve_fit_under_the_cap_with_the_margin_over() -> None:
    """The inequality the whole design rests on, asserted where the numbers are written.

    The generation latches at its own trigger, the service may have admitted more than the
    counter has seen, the gate admits a bounded number after the latch, and what is left is the
    margin. If those four ever added to the cap, a run would reach the refusal the turnover
    exists to avoid, so the module asserts it at import and this says the same thing out loud.
    """
    assert (
        kernel_workflow.TURNOVER_TRIGGER
        + kernel_workflow.ADMISSION_LEAD
        + kernel_workflow.ADMISSION_RESERVE
        + kernel_workflow.TURNOVER_MARGIN
        < kernel_workflow.SERVICE_UPDATE_CAP
    )


def test_the_ceiling_leaves_room_under_the_service_and_over_the_recorded_profile() -> None:
    """The size ceiling has two margins to keep, and they pull in opposite directions.

    Above it is the service's own limit on one payload: a ceiling at or over that would let the
    generation submit a continuation the service refuses, which is a fault where there was a
    plan. Below it is the profile the package supports: a ceiling under that would make the
    generation refuse its own boundary and serve on to the cap, which is the failure this whole
    change exists to remove.

    What this checks and what it cannot check are worth being exact about. The number below is a
    measurement that was taken once, of the real two hundred task roster read out of the run that
    stopped, driven to Done under the declared workload: fifteen world calls a task, a lost reply
    and the confirmation its recovery costs, identifiers minted the way the transport mints them,
    objects cited by the digest of their bytes, and every one of the eleven boundaries that
    generation decided on. It is the size at the largest quiet point that roster has, the boundary
    that falls after the last payload of the last task is presented. That roster is not in this
    repository, so this cannot re-measure it: what it checks is that the ceiling the code carries
    today still leaves that recorded measurement the room it was chosen to leave. A roster that
    grew outside this repository, or a carrier field added inside it, would not move this number.
    The live re-measurement is the two hundred task test beside it, driven the same way with
    bodies that do not compress at all, which asserts the same headroom against what it actually
    encodes.
    """
    assert (
        kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
        < kernel_workflow.SERVICE_PAYLOAD_LIMIT_BYTES
    )
    recorded = 1_122_914
    assert recorded < kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
    headroom = 1 - recorded / kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
    assert headroom > 0.25, "the recorded profile has to sit well under the ceiling, not near it"


def test_the_progress_whitelist_fits_many_times_over_inside_the_reserve() -> None:
    """What the gate admits after the latch, counted, against what it will admit at most.

    Each of these is work that can make the boundary quiet, and each succeeds at most once per
    latch: there is one message that can be pending, one grant that can be held, one seal that
    can be prepared, one claim that recovers an abandoned grant with the confirmation that
    follows it, and one queue to close. Seven successes, and the reserve is many times that.

    What the reserve is a ceiling on is attempts, not successes, and the difference is the whole
    reason it is a ceiling at all. Every one of these can fail and be sent again: a presentation
    whose references the store cannot produce is refused and the transport builds a fresh
    attestation for the same message, which is its supported repair and not misbehaviour. So the
    seven below are a floor on what has to fit, and the reserve above them is room for the
    failures in between.

    That room is finite, so it can run out, and what happens then is not a wedge. A generation
    that can no longer admit the work which would clear what it is holding recognises that,
    records why, releases the gate and serves out the execution it is in.

    It can run out two ways, because the reserve is not the only ceiling. The whole reserve can
    go, which forty repaired presentations do. Or the smaller allowance a claim draws on can go,
    which four claims that failed do: a claim reads the store before it swaps the epoch, so a
    store that cannot answer refuses it, and a claim that took nothing over still spent a slot.
    That one matters more than its size suggests, because a claim is the only call that can
    install the owner an abandoned hold is waiting for. Both are the same ending and both name
    which allowance it was.

    Three tests drive these: the repeatable calls that must not spend a named action's slot, the
    repair that spends the whole reserve and is served anyway, and the claims that fail and are
    repaired.
    """
    whitelist = {
        "the pending message's presentation commit": 1,
        "the held grant's end": 1,
        "a prepared seal's retry": 1,
        "the presentation of that seal's acknowledgement": 1,
        "the claim that recovers an abandoned grant": 1,
        "the confirmation that follows it": 1,
        "closing the queue": 1,
    }
    assert sum(whitelist.values()) == 7
    assert sum(whitelist.values()) <= kernel_workflow.ADMISSION_RESERVE
    # The allowances the repeatable calls draw on leave the named work its own room, and the
    # claim's is separate from the confirmations' so that spending one cannot lock out the other.
    assert (
        kernel_workflow.ADMISSION_REPEATS + kernel_workflow.ADMISSION_CLAIMS
        < kernel_workflow.ADMISSION_RESERVE
    )
    assert kernel_workflow.ADMISSION_CLAIMS >= 1
    # Both ceilings that can gate necessary clearing work have a reason to report when they run
    # out, and the one that cannot gate any is deliberately absent from that list.
    assert kernel_workflow.ADMISSION_EXHAUSTED != kernel_workflow.RECOVERY_CLAIMS_EXHAUSTED


def test_the_carrier_is_outside_what_a_resume_is_held_to() -> None:
    """Adding the projection to a start changes nothing about the generation's identity.

    The hash names its keys explicitly, and this is the test that keeps it that way. A resume
    derives the hash from the composition the resuming process serves, which carries nothing;
    if the carrier entered the hash, every resume of a generation that had crossed a boundary
    would be refused for a configuration that had not changed.
    """
    start = make_start()
    assert configuration_hash(replace(start, carry=_a_carrier(start))) == configuration_hash(start)


def test_every_attempt_field_either_crosses_or_is_declared_to_stay_behind() -> None:
    """A field added to an attempt is a decision about the boundary rather than a default.

    The carried row is the list of what crosses. Everything else on an attempt is named in the
    list of what deliberately does not: the task itself, which the start already carries, the
    three the seal writes and nothing reads, and the expiry, which a legal boundary has already
    applied. A new field belongs in one list or the other, and this fails until it is in one.
    """
    on_the_attempt = {row.name for row in fields(kernel_workflow._Attempt)}
    carried = {row.name for row in fields(CarriedAttempt) if row.name != "attempt_id"}
    assert carried == set(kernel_workflow._CARRIED_ATTEMPT_FIELDS)
    assert carried | set(kernel_workflow._UNCARRIED_ATTEMPT_FIELDS) == on_the_attempt
    assert not carried & set(kernel_workflow._UNCARRIED_ATTEMPT_FIELDS)


def test_every_type_the_carrier_names_can_be_resolved_where_it_is_named() -> None:
    """The converter decodes a dataclass by its annotations, so every one has to resolve.

    This module writes its annotations as text, and the converter reads them back through the
    module's own namespace. A type named in a field but never imported there fails to resolve,
    and the converter reports only that it could not convert the whole value: the successor
    would decode nothing and fail its first activation with a message naming the field rather
    than the name. So the resolution is checked directly, where the answer is legible.
    """

    from shogym.serve.protocol_v2.kernel import messages

    unresolvable = []
    for name in dir(messages):
        candidate = getattr(messages, name)
        if isinstance(candidate, type) and hasattr(candidate, "__dataclass_fields__"):
            try:
                typing.get_type_hints(candidate)
            except NameError as error:  # noqa: PERF203 - the report is the point
                unresolvable.append((name, str(error)))
    assert unresolvable == []


def test_a_generation_being_created_is_refused_a_carried_projection() -> None:
    """The two doors a caller comes in through refuse a start that already holds a projection."""
    start = make_start()
    refuse_a_carried_projection(start)
    with pytest.raises(ValueError, match="no earlier execution"):
        refuse_a_carried_projection(replace(start, carry=_a_carrier(start)))


# What one gateway keeps open, whether it can still be open when a boundary happens, and what
# answers the request it sends again if it is. The keys are the gateway's own record classes and
# the test below checks that against the union the gateway actually declares, so a record added
# later fails here rather than being quietly left out of the proof.
_WHAT_A_RECORD_RESENDS = {
    "_Idle": (
        "nothing is open",
        "crosses, and sends nothing again",
        None,
    ),
    "_RequestUncertain": (
        "a pull, info or seal request went out and no answer came back",
        "crosses only where the request was refused or failed, because a request that offered "
        "a message leaves that message pending and pending is what a boundary waits for",
        "the request tables, or the kept failure of a filing whose work failed for good",
    ),
    "_Offered": (
        "a message was offered and its attestation built",
        "cannot cross: the message is pending",
        None,
    ),
    "_PresentationUncertain": (
        "a presentation commit went out and no acknowledgement came back",
        "crosses where the commit was applied, which is what clears the pending message",
        "the attestation tables",
    ),
    "_PresentationRefused": (
        "a message is still held and its attestation was refused",
        "cannot cross: the message is still pending",
        None,
    ),
    "_HorizonOwed": (
        "a filing this gateway owes on an attempt's behalf, frozen before it is sent",
        "crosses in the window before the filing is accepted: the grant was given back first, "
        "and no message is pending and no seal is prepared until the filing lands",
        "the terminal request table, or the kept failure of a filing whose work failed for good",
    ),
    "_LeaseHeld": (
        "a grant this gateway asked for and has not given back",
        "crosses where the end was applied and its answer was lost, which is what releases the "
        "grant that would otherwise block the boundary",
        "the last environment end this generation answered",
    ),
    "_ResultOwed": (
        "a result that exists and has not reached the call that asked for it",
        "crosses: nothing about it is outstanding in the generation",
        "nothing kept, because the confirmation that hands it over mints a fresh identifier "
        "every time and an answer from the last time it asked would be the wrong answer",
    ),
}


def test_every_handler_writes_its_answer_into_the_journal() -> None:
    """The journal's inventory is the handler list, not a list somebody wrote down.

    What a repeated Update identifier is answered with used to be kept for a chosen few
    operations, and every review found another one that had been left out. It is kept for all of
    them now, and this is the check that keeps it that way: the thirteen registered Updates are
    read off the workflow, and every one of them has to name a result type the journal can decode
    an answer back into. A handler added later fails this rather than being quietly uncovered.

    The last two are the fork and a child's own preparation, which no gateway sends and no agent
    reaches. They are here for the same reason as the eleven: each exact Update returns the
    complete evidence of what it did, and the exact outcome journal is a usable recovery path for
    either only if that answer decodes back.
    """
    from temporalio import workflow as temporal_workflow

    registered = set(temporal_workflow._Definition.must_from_class(StreamWorkflow).updates)
    assert len(registered) == 13, sorted(registered)
    assert registered == set(kernel_workflow.ANSWER_TYPES), (
        registered ^ set(kernel_workflow.ANSWER_TYPES)
    )


def test_the_carrier_packs_losslessly_and_writes_the_same_bytes_every_time() -> None:
    """The carried projection round trips, and packing it twice produces one answer.

    It becomes part of a command in a history that has to replay to the same bytes, so the
    encoding is fixed rather than incidental: sorted keys, fixed separators, a fixed deflate
    level. What remains outside this test's reach is the deflate implementation itself, which
    belongs to the interpreter and is the same for every Worker of one deployment. Two things
    stand in for that: the encoding is named in the carrier and a writing this code does not know
    is refused rather than misread, and the whole of what a reader needs is the packed value plus
    that name, so a Worker that reads it back reads it the same way whatever wrote it.
    """
    start = make_start(tasks=1)
    projection = replace(
        _unpacked(_a_carrier(start)),
        journal=[
            ["present-1-a", "commit_presentation", 1, "row", "att-1", "", "", "", None, ""],
            ["pull-1-b", "pull", 1, "value", "", "", "", "", {"kind": "info"}, ""],
            ["own-0-c", "claim_ownership", 0, "protocol", "", "fenced_writer", "", "", None, ""],
            ["seal-1-d", "seal_attempt", 1, "failure", "", "", "ActivityError", "no", None, ""],
        ],
    )
    converter = default_converter().payload_converter
    version = carrier_version(start, projection)
    packed = pack_carrier(projection, converter, version=version)
    assert packed == pack_carrier(projection, converter, version=version), (
        "one projection packs to one string"
    )
    assert unpack_carrier(packed, converter) == projection
    with pytest.raises(Exception, match="reads"):
        unpack_carrier(replace(packed, encoding="something-else.v9"), converter)
    with pytest.raises(Exception, match="reads"):
        unpack_carrier(replace(packed, encoding="", data=""), converter)


def test_sharing_keeps_values_apart_that_a_reader_has_to_tell_apart() -> None:
    """Distinct scalars stay distinct through the sharing, and shared structure is not aliased.

    Sharing writes each distinct value once, and what counts as one value is the question. Python
    equality is the wrong answer for it: it makes a boolean equal to an integer and a negative
    zero equal to a positive one, and a projection carrying either pair would come back with the
    pair collapsed. And a value written once is still two values when it is read: a historical
    reading of a table and the live table it was a reading of are restored as separate objects, so
    that later work on the live one cannot rewrite the history.
    """
    from shogym.serve.protocol_v2.kernel.messages import _shared, _unshared

    table = {"cursor": "c" * 24, "count": 1}
    value = {
        "history": [dict(table), dict(table)],
        "live": dict(table),
        "numbers": [0.0, -0.0, 1, 1.0],
        "flags": [True, 1, False, 0],
    }
    written = _shared(value)
    read = _unshared(written)
    assert read == value
    assert json.dumps(read, sort_keys=True) == json.dumps(value, sort_keys=True)
    assert [repr(number) for number in read["numbers"]] == ["0.0", "-0.0", "1", "1.0"]
    assert [type(flag).__name__ for flag in read["flags"]] == ["bool", "int", "bool", "int"]
    read["live"]["count"] = 2
    assert read["history"][0]["count"] == 1, "a live table must not write through to a reading"
    assert read["history"][0] is not read["history"][1]
    assert sum(1 for node in written[1] if node[0] == "d") == 2, (
        "three occurrences of one table are written as one node, beside the root"
    )


def test_a_read_is_taken_by_the_execution_it_was_in_as_well_as_how_far_it_had_got() -> None:
    """The mark is the pair, so an equal count in another execution is a moved history.

    A generation may run in more than one execution, and the count starts again in each. A read
    bracketed by counts alone would compare a predecessor's count with a successor's and call
    them equal, or call a smaller one a history going backwards. The pair says which execution
    as well as how far, and equality of the pair is the whole of the question.
    """
    _refuse_a_moved_history("stream/one", ("run-a", 40), ("run-a", 40))
    with pytest.raises(Exception, match="moved it"):
        _refuse_a_moved_history("stream/one", ("run-a", 40), ("run-a", 41))
    with pytest.raises(Exception, match="moved it"):
        _refuse_a_moved_history("stream/one", ("run-a", 40), ("run-b", 40))
    with pytest.raises(Exception, match="moved it"):
        _refuse_a_moved_history("stream/one", ("run-a", 40), ("run-b", 3))


# What holds on a running generation.


def _comparable(state) -> Dict[str, Any]:
    """One state reading, minus the fields two separate generations always differ in.

    Everything else has to be equal between a run that crossed boundaries and one that did not.
    The five that may differ are the ones named in the contract above, and none of them is about
    what the generation served: four say what happened to the generation itself, and the fencing
    token is minted fresh by whichever process claimed it, so two separate runs never share one.
    That the token crosses a boundary unchanged is the subject of its own test, where the
    comparison is one generation against itself rather than against another.
    """
    return {
        row.name: getattr(state, row.name)
        for row in fields(state)
        if row.name not in _PERMITTED_DIFFERENCES
    }


async def _drive(client: Client, start: StreamStart, workflow_id: str) -> Dict[str, Any]:
    """Run one generation to Done and report everything a caller or a reader can see of it."""
    caller = await open_stream(client, start, workflow_id=workflow_id)
    await caller.stream.close_queue()
    seen = await served(caller)
    state = await caller.stream.stream_state()
    records = await caller.stream.handle.query(StreamWorkflow.generation_records)
    return {
        "kinds": [message.kind for message in seen],
        "message_ids": [message.message_id for message in seen],
        "visible": [message.visible_text for message in seen],
        "state": _comparable(state),
        "attempts": records.attempts,
        "presentations": records.presentations,
        "turnovers": state.turnovers,
    }


# The contract this file exists to establish, stated once so the tests below can be read against
# it rather than against a summary of them.
#
# For every operation one gateway can make against a generation, in the order a gateway can make
# them, the complete result is equal between a generation that crossed boundaries and one that
# did not. Complete means the whole returned value and not a chosen field of it: the offered
# message with its bytes, the presentation acknowledgement with its cursor and its projection
# hash, the environment lease with its held flag and cursor, the info answer, the finalization
# receipt with its capacity and its cascade, the ownership receipt a takeover is given, and the
# state reading taken between each of them.
#
# Five differences are permitted and each is named. Four are the operational fields that exist to
# say what happened to the generation rather than what it served: how many boundaries it crossed,
# whether it is holding traffic back for another, whether it gave up on one and why, and the size
# that refused one. The fifth is the fencing token hash, which whichever process claims the
# generation mints fresh, so two separate runs never share one and the difference says nothing
# about a boundary.
#
# Everything else in the state is compared, including the two counters that say what the
# generation has done with its blob store. The count of finished store reads is a count like the
# offers and the eligibilities beside it, so it crosses a boundary and has to be equal; the count
# of reads in flight cannot cross, because a legal boundary has no handler running, and is zero
# on both sides of one. The twin runs with a store configured so that both are exercised rather
# than left at zero, which is what makes this class of difference visible at all.
#
# What this contract is about is one gateway's answers. A repeated Update identifier is a
# different question, and the answer to it is not a chosen set of slots: the generation keeps
# every outcome it completed and answers any of them, whoever asks. The matrix below is where
# that is driven, handler by handler.
_PERMITTED_DIFFERENCES = (
    "turnovers",
    "turnover_requested",
    "turnover_refused",
    "turnover_refused_bytes",
    "fencing_token_hash",
)


def _without_the_token(receipt: OwnershipReceipt) -> Dict[str, Any]:
    """One ownership receipt, minus the token hash a fresh process always mints differently."""
    return {
        row.name: getattr(receipt, row.name)
        for row in fields(receipt)
        if row.name != "fencing_token_hash"
    }


async def _every_operation_a_gateway_can_make(
    client: Client, start: StreamStart, workflow_id: str, *, turnover_at=None
) -> List[Tuple[str, Any]]:
    """Drive one generation through every operation one gateway can make, keeping every answer.

    The order is a gateway's order and the answers are kept whole. Where ``turnover_at`` is
    given, the generation is made to cross a boundary between the steps a boundary can fall
    between, which is every quiet point in this sequence.
    """
    journal: List[Tuple[str, Any]] = []

    async def note(label: str, value: Any) -> None:
        journal.append((label, value))

    async def maybe_cross(label: str) -> None:
        if turnover_at is None:
            return
        await _cross_a_boundary(caller, turnover_at, workflow_id, client)
        journal.append((f"crossed before {label}", True))

    stream = await start_stream(client, start, workflow_id=workflow_id)
    receipt = await stream.claim_consumer(CONSUMER)
    caller = Caller(stream, receipt.initial_cursor)
    await note("consumer receipt", receipt)
    await note("state after binding", _comparable(await caller.stream.stream_state()))

    await note("queue closed", await caller.stream.close_queue())
    await maybe_cross("the first pull")

    task = await caller.pull()
    await note("task offered", task)
    await note("task presented", await caller.attest(task))
    await note("state after the task", _comparable(await caller.stream.stream_state()))

    call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
    await note("grant", await caller.stream.begin_environment_call(call))
    await note("grant returned", await caller.stream.end_environment_call(call))
    await maybe_cross("the filing")

    acknowledgement = await caller.stream.seal(
        SealRequest(
            metadata=TerminalMetadata(
                request_id=caller.next_id(),
                last_presented_cursor=caller.cursor,
                attempt_id=task.attempt_id or "",
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
        )
    )
    await note("acknowledgement offered", acknowledgement)
    await note("acknowledgement presented", await caller.attest(acknowledgement))
    await maybe_cross("the payload")

    payload = await caller.pull()
    await note("payload offered", payload)
    await note("payload presented", await caller.attest(payload))

    answer = await caller.take_info()
    await note("info offered", answer)
    await note("state after info", _comparable(await caller.stream.stream_state()))
    await maybe_cross("the ending")

    await note(
        "finalization receipt",
        await caller.stream.finalize(
            FinalizeRequest(
                request_id=caller.next_id(), attempt_id=attempt(2), reason=STEP_CAP
            )
        ),
    )
    await maybe_cross("the takeover")

    taken = StreamHandle(client.get_workflow_handle_for(StreamWorkflow.run, workflow_id))
    standing = await taken.stream_state()
    await note(
        "takeover receipt",
        _without_the_token(
            await taken.claim_ownership(
                configuration_hash=configuration_hash(start),
                previous_epoch=standing.ownership_epoch,
                claimant_id="the-second",
                reason="resume",
            )
        ),
    )
    caller.stream = taken
    await note("confirmed state", _comparable(await caller.stream.confirm_state()))

    seen = await served(caller)
    await note("kinds served to the end", [message.kind for message in seen])
    await note("messages served to the end", seen)
    records = await caller.stream.handle.query(StreamWorkflow.generation_records)
    await note("attempt records", records.attempts)
    await note("presentations", records.presentations)
    await note("every acknowledgement", list(caller.acknowledgements))
    await note("final state", _comparable(await caller.stream.stream_state()))
    return journal


@pytest.mark.durable
async def test_a_generation_that_crossed_boundaries_answers_as_the_one_that_did_not(
    env, turnover_at, tmp_path
) -> None:
    """The twin, against the contract stated above rather than against a summary of it.

    Two generations are driven through every operation one gateway can make, in a gateway's
    order, and every answer is kept whole: the offered messages with their bytes, the
    presentation acknowledgements with their cursors and projection hashes, the environment
    leases, the info answer, the finalization receipt, the ownership receipt of a takeover, and
    the state readings taken between them. One crosses a boundary at every quiet point in that
    sequence and the other crosses none, and the two are compared entry by entry.
    """
    verifying = _instead_of_verification(_a_store_that_answers)
    async with stream_worker(env.client, activities=verifying):
        plain = await _every_operation_a_gateway_can_make(
            env.client, _asking_start(str(tmp_path)), "stream/twin/plain"
        )
    turnover_at(10_000)
    async with stream_worker(env.client, activities=verifying):
        crossed = await _every_operation_a_gateway_can_make(
            env.client, _asking_start(str(tmp_path)), "stream/twin/crossed", turnover_at=turnover_at
        )
    turnover_at(10_000)
    # The store was read on both sides, which is what makes the counts worth comparing at all.
    final_plain = dict(plain)["final state"]
    assert final_plain["verification_batches"] > 0
    assert final_plain["verifying"] == 0

    crossings = [label for label, _ in crossed if label.startswith("crossed before")]
    assert len(crossings) == 5, crossings
    without_crossings = [row for row in crossed if not row[0].startswith("crossed before")]
    assert [label for label, _ in without_crossings] == [label for label, _ in plain]
    for (label, mine), (_, theirs) in zip(without_crossings, plain):
        assert mine == theirs, label


@pytest.mark.durable
async def test_the_projection_hash_a_presentation_attests_to_does_not_move_at_a_boundary(
    env, turnover_at
) -> None:
    """A generation that crossed boundaries publishes the same projection hash as one that did not.

    The hash is what an attestation is built against and checked against, so a value that moved
    at a boundary would refuse a presentation whose harness did nothing wrong. Nothing in it is
    per execution, and this is what says so.
    """
    async with stream_worker(env.client):
        plain = await _drive(env.client, make_start(tasks=3), "stream/hash/plain")
    turnover_at(6)
    async with stream_worker(env.client):
        crossed = await _drive(env.client, make_start(tasks=3), "stream/hash/crossed")
    assert crossed["state"]["stream_state_sha256"] == plain["state"]["stream_state_sha256"]
    assert crossed["state"]["configuration_hash"] == plain["state"]["configuration_hash"]


@pytest.mark.durable
async def test_the_generation_keeps_its_identifier_and_its_owner_across_a_boundary(
    env, turnover_at
) -> None:
    """One workflow identifier, several executions, and the writer that opened it still admitted.

    The transport that is serving the agent holds a token minted once, at the first claim. An
    execution that came back at epoch zero would fence that transport out of its own generation
    at the first call after the boundary, so the epoch and the token hash cross and the same
    handle keeps writing.
    """
    turnover_at(6)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=3), workflow_id="stream/owner/1")
        await caller.stream.close_queue()
        before = await caller.stream.stream_state()
        await served(caller)
        after = await caller.stream.stream_state()
    assert after.turnovers >= 2
    assert after.ownership_epoch == before.ownership_epoch
    assert after.fencing_token_hash == before.fencing_token_hash
    assert after.consumer_id == "harness-1"
    assert after.ownership_claims == before.ownership_claims == 1
    chain = await run_ids(env.client, "stream/owner/1")
    assert len(chain) == after.turnovers + 1


@pytest.mark.durable
async def test_the_gate_bounds_what_is_accepted_while_a_claim_holds_a_handler_open(
    env, turnover_at, monkeypatch, tmp_path
) -> None:
    """The counterexample the gate exists for, run: a claim paused mid verification.

    An ownership claim reads the store before it swaps the epoch, and it reads until the set
    stops growing. While that read is in flight the claim's handler is unfinished, so the
    boundary cannot be reached, and the writer the claim is replacing is still the writer, so
    its calls are still admitted. Without a gate the generation would accept whatever that
    writer sent for as long as the read took, which is not bounded by anything.

    With the gate, what is accepted after the latch is bounded by the reserve and by nothing
    else. Traffic that cannot make the boundary quiet is rejected before acceptance and costs
    the generation nothing; traffic that can is accepted until the reserve is spent, and then
    that is refused too. This drives both and counts the accepted Updates in the history.
    """
    pause = asyncio.Event()
    verifying = asyncio.Event()

    @activity.defn(name=VERIFY_BLOBS)
    async def held_verification(payload: VerifyBlobsInput) -> BlobsVerified:
        if pause.is_set():
            verifying.set()
            await asyncio.sleep(3600)
        return BlobsVerified(verified=list(payload.references), unverified=[])

    reserve = 12
    turnover_at(10_000, reserve=reserve)
    start = replace(make_start(tasks=2), blob_root=str(tmp_path))
    async with stream_worker(env.client, activities=_instead_of_verification(held_verification)):
        caller = await open_stream(env.client, start, workflow_id="stream/gate/1")
        await caller.stream.close_queue()
        task = await caller.take()
        await caller.work(task)

        pause.set()
        claim = _in_the_background(
            _resume(env.client, start, workflow_id="stream/gate/1", claimant="the-replacement")
        )
        await asyncio.wait_for(verifying.wait(), timeout=30)

        await _latch(caller, turnover_at, env.client, "stream/gate/1", reserve=reserve)
        accepted_before = await _accepted_updates(env.client, "stream/gate/1")

        # Traffic that cannot make the boundary quiet, sent continuously. It goes straight at the
        # generation rather than through the transport, because what is under test is what the
        # generation admits: the transport's own answer to a rejection is to wait, which is the
        # subject of its own tests.
        for _ in range(20):
            assert await rejected(
                _straight_at_the_generation(
                    caller,
                    StreamWorkflow.pull,
                    PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor),
                    caller.stream.writer,
                    update_id=caller.next_id(),
                )
            )
        assert await _accepted_updates(env.client, "stream/gate/1") == accepted_before

        # Traffic that is on the whitelist, sent until its allowance is spent and past it.
        spent = 0
        for _ in range(reserve * 3):
            try:
                await _straight_at_the_generation(
                    caller,
                    StreamWorkflow.confirm_state,
                    caller.stream.writer,
                    update_id=caller.next_id(),
                )
                spent += 1
            except WorkflowUpdateFailedError as error:
                assert turnover_pending(error)
        assert spent <= reserve
        assert await _accepted_updates(env.client, "stream/gate/1") - accepted_before <= reserve
        # The boundary never happened, because the claim's handler never finished.
        assert (await caller.stream.stream_state()).turnovers == 0
        await _let_go(claim)


async def _latch(
    caller: Caller, turnover_at, client: Client, workflow_id: str, *, reserve: Optional[int] = None
) -> None:
    """Make the next Update this generation accepts be the one that decides to turn over.

    The trigger goes one above what this execution has already accepted, so a replay of the
    history recorded so far never reaches the decision and the Update sent next does. That is
    how it happens in a run: the count rises under live traffic and crosses a trigger that has
    not moved. Lowering the trigger under a count already reached would put the decision inside
    the replay instead, where the marker gating it reads as absent and stays that way.
    """
    turnover_at(await _accepted_updates(client, workflow_id) + 1, reserve=reserve)
    await caller.stream.confirm_state()


async def _cross_a_boundary(caller: Caller, turnover_at, workflow_id: str, client: Client) -> None:
    """Make this generation cross exactly one boundary, from wherever it is now."""
    before = (await caller.stream.stream_state()).turnovers
    await _latch(caller, turnover_at, client, workflow_id)
    for _ in range(400):
        if (await caller.stream.stream_state()).turnovers > before:
            turnover_at(10_000)
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the generation never crossed a boundary")


@pytest.mark.durable
async def test_the_same_environment_end_after_a_boundary_answers_as_it_did_before(
    env, turnover_at
) -> None:
    """The lease an end returned, returned again, rather than a report that nothing was held.

    An end is answered with whether this call was the one holding the stream and with the cursor
    as it was. Across a boundary the grant is gone by construction, so the successor executing
    the same Update would say it held nothing where its predecessor said it held the stream. The
    cache is what makes the two answers the same, and a fresh identifier for the same logical
    request still gets the ordinary answer for a grant nobody is holding.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/end/1")
        await caller.stream.close_queue()
        task = await caller.take()
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        await caller.stream.begin_environment_call(call)
        held = await caller.stream.end_environment_call(call)
        assert held.held is True

        await _cross_a_boundary(caller, turnover_at, "stream/end/1", env.client)

        again = await caller.stream.end_environment_call(call)
        assert again == held
        # A different call identifier is a different logical request, and it is answered on the
        # state the generation is in rather than out of anything kept.
        other = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        assert (await caller.stream.end_environment_call(other)).held is False


@pytest.mark.durable
async def test_a_begin_sent_twice_grants_one_world_and_counts_one_call(env, turnover_at) -> None:
    """The same begin, sent again after a boundary, is answered rather than granted a second time.

    A grant is a change to a world this stream cannot see and a durable count against the
    attempt, and the count is what a later claim is held to. Executing the same begin twice
    would leave a hold nothing out there matches and a count that says a world moved twice.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/begin/1")
        await caller.stream.close_queue()
        task = await caller.take()
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        granted = await caller.stream.begin_environment_call(call)
        await caller.stream.end_environment_call(call)
        counted = (await caller.stream.stream_state()).environment_calls

        await _cross_a_boundary(caller, turnover_at, "stream/begin/1", env.client)

        assert await caller.stream.begin_environment_call(call) == granted
        assert (await caller.stream.stream_state()).environment_calls == counted
        # And nothing is holding the stream, because nothing was granted the second time.
        assert (await caller.stream.stream_state()).environment_call is None


@pytest.mark.durable
async def test_a_failure_after_a_boundary_names_the_activity_the_twin_would_have_named(
    env, turnover_at
) -> None:
    """The Activity identifier a reader gets is the generation's ordinal, not the execution's.

    The service numbers Activities per execution from one, so a run that crossed a boundary
    would publish a smaller identifier than the same run without one, for the same logical step.
    A record that carried an already-recorded failure would not show this: the difference is in
    the identifier of a failure that has not happened yet, so the failure is generated after the
    boundary in one run and at the same logical point in the other.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def refusing_grade(request: GradeAttemptInput) -> GradeAttemptResult:
        if request.attempt_id == attempt(1):
            raise ApplicationError("this grader will not score it", non_retryable=True)
        return await grade_attempt_activity(request)

    served_activities = [
        row for row in kernel_activities() if row is not grade_attempt_activity
    ] + [refusing_grade]

    async def drive(workflow_id: str, *, cross: bool) -> Dict[str, Any]:
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id=workflow_id)
        await caller.stream.close_queue()
        first = await caller.take()
        await caller.work(first)
        await caller.take()
        if cross:
            await _cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        second = await caller.take()
        with pytest.raises(WorkflowUpdateFailedError):
            await caller.work(second)
        records = await caller.stream.handle.query(StreamWorkflow.generation_records)
        return {
            "records": records.attempts,
            "presentations": records.presentations,
            "state": _comparable(await caller.stream.stream_state()),
        }

    turnover_at(10_000)
    async with stream_worker(env.client, activities=served_activities):
        plain = await drive("stream/activity/plain", cross=False)
        turnover_at(10_000)
        crossed = await drive("stream/activity/crossed", cross=True)

    # The whole row rather than the identifier alone: the ending, the five failure details the
    # Activity's failure was read into, the score, the digests and every delivery flag.
    assert crossed["records"] == plain["records"]
    assert crossed["presentations"] == plain["presentations"]
    assert crossed["state"] == plain["state"]
    # And the identifier itself, named, because it is the one the boundary would have moved.
    failed = next(row for row in plain["records"] if row.attempt_id == attempt(1))
    assert failed.final_failure == "seal_failed"
    assert failed.failure_activity_id is not None
    assert (
        next(
            row for row in crossed["records"] if row.attempt_id == attempt(1)
        ).failure_activity_id
        == failed.failure_activity_id
    )


@pytest.mark.durable
async def test_the_filing_whose_work_failed_is_answered_the_same_way_after_a_boundary(
    env, turnover_at
) -> None:
    """A filing that ended its own attempt is answered with the failure, not with a conflict.

    The work behind an accepted terminal can fail for good, and the ending it writes leaves no
    row in the table a retry is answered from. Sent again under its own identifier after a
    boundary, the filing would meet the attempt it ended and be told it conflicts with it, which
    is not what the caller was told the first time. A fresh identifier for the same logical
    filing is the case that does get the conflict, because that is a second filing.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def refusing_grade(request: GradeAttemptInput) -> GradeAttemptResult:
        raise ApplicationError("this grader will not score it", non_retryable=True)

    served_activities = [
        row for row in kernel_activities() if row is not grade_attempt_activity
    ] + [refusing_grade]

    turnover_at(10_000)
    async with stream_worker(env.client, activities=served_activities):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/failed/1")
        await caller.stream.close_queue()
        task = await caller.take()
        filing = SealRequest(
            metadata=TerminalMetadata(
                request_id=caller.next_id(),
                last_presented_cursor=caller.cursor,
                attempt_id=task.attempt_id or "",
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
        )
        with pytest.raises(WorkflowUpdateFailedError) as first:
            await caller.stream.seal(filing)
        assert protocol_error_code(first.value) is None

        await _cross_a_boundary(caller, turnover_at, "stream/failed/1", env.client)

        with pytest.raises(WorkflowUpdateFailedError) as again:
            await caller.stream.seal(filing)
        # Still a fault rather than a refusal, carrying the same words and naming the same
        # failure. What is reproduced is what a caller reads: the declared type and the message.
        # The wrapper an Activity failure arrives in the first time is not rebuilt, so the
        # second is an application failure whose type is that wrapper's name.
        assert protocol_error_code(again.value) is None
        assert again.value.cause.message == first.value.cause.message
        assert again.value.cause.type == type(first.value.cause).__name__
        # A second filing for the same attempt is what the conflict is for.
        assert await refused(caller.stream.seal(replace(filing, metadata=replace(
            filing.metadata, request_id=caller.next_id()
        )))) == "conflicting_seal"


@pytest.mark.durable
async def test_a_pending_message_makes_the_boundary_wait_and_then_lets_it_through(
    env, turnover_at
) -> None:
    """A message offered and not yet attested to is exactly what a boundary waits for.

    The attestation a transport is holding was built against the projection as it stood, and it
    is committed by a later Update. If a boundary landed between the two, that commit would have
    to be answered by an execution that was not there when the attestation was built. It cannot:
    the message is pending, and pending is the first thing the boundary predicate refuses.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/wait/1")
        await caller.stream.close_queue()
        message = await caller.pull()
        assert (await caller.stream.stream_state()).pending_message_id == message.message_id

        await _latch(caller, turnover_at, env.client, "stream/wait/1")
        for _ in range(10):
            await asyncio.sleep(0.02)
        assert (await caller.stream.stream_state()).turnovers == 0

        # The commit is on the whitelist, so it is admitted, and the boundary follows it.
        await caller.present(message)
        for _ in range(200):
            if (await caller.stream.stream_state()).turnovers == 1:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the boundary never followed the presentation")
        turnover_at(10_000)
        assert (await caller.stream.stream_state()).pending_message_id is None


@pytest.mark.durable
async def test_a_held_grant_makes_the_boundary_wait_until_it_is_given_back(
    env, turnover_at
) -> None:
    """A call to a world this stream cannot see holds the generation, and holds the boundary.

    The grant is the last thing the stream knows about that world. Handing the generation on
    while one is out would leave the successor with a hold nothing can end by name, so the
    boundary waits for the end and the end is on the whitelist.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/grant/1")
        await caller.stream.close_queue()
        task = await caller.take()
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        await caller.stream.begin_environment_call(call)

        await _latch(caller, turnover_at, env.client, "stream/grant/1")
        for _ in range(10):
            await asyncio.sleep(0.02)
        assert (await caller.stream.stream_state()).turnovers == 0

        await caller.stream.end_environment_call(call)
        for _ in range(200):
            if (await caller.stream.stream_state()).turnovers == 1:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the boundary never followed the end of the grant")
        turnover_at(10_000)
        assert (await caller.stream.stream_state()).environment_call is None


@pytest.mark.durable
async def test_a_pull_repeated_under_its_own_identifier_after_a_boundary_gets_its_offer(
    env, turnover_at
) -> None:
    """A repeated Update identifier gets what it got, on either side of a boundary.

    Inside one execution the service answers a repeated identifier from its own record, without
    the handler running: a pull sent twice under one identifier gets the same offered message
    back, bytes and all, even after that message has been presented. That record belongs to the
    execution and a boundary leaves it behind, so the generation keeps the same record itself and
    carries it, and the answer on the far side is the answer.

    A fresh identifier for the same logical request is the other contract and is unaffected: it
    reaches the handler and is judged on the state it finds, which is how a request made from a
    cursor that has moved is told so.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/table/1")
        await caller.stream.close_queue()
        request = PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor)
        offered = await caller.stream.pull(request)

        await caller.present(offered)
        await _cross_a_boundary(caller, turnover_at, "stream/table/1", env.client)

        # The same request, once the message it reserved has been presented, and after a
        # boundary. It is the same offered message, bytes and all, which is what the same
        # identifier would have been answered with had no boundary happened at all.
        assert await caller.stream.pull(request) == offered
        # A fresh request at a stale cursor is refused for the cursor, which is the ordinary
        # answer and not anything the boundary decided.
        stale = PullRequest(
            request_id=caller.next_id(),
            last_presented_cursor=request.last_presented_cursor,
        )
        assert await refused(caller.stream.pull(stale)) == "invalid_cursor"


@pytest.mark.durable
async def test_a_takeover_after_a_boundary_finds_the_generation_where_it_left_it(
    env, turnover_at
) -> None:
    """A replacement process resumes the chain by identifier and continues where it was.

    A resume addresses the generation by its workflow identifier and reads the epoch it finds,
    so it follows the chain without knowing there is one. What it needs to find is the epoch,
    the cursor, the roster's states and the checkpoints, all of which crossed.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/take/1")
        await caller.stream.close_queue()
        task = await caller.take()
        await caller.work(task)
        await _cross_a_boundary(caller, turnover_at, "stream/take/1", env.client)

        before = await caller.stream.stream_state()
        replacement = await _resume(
            env.client, make_start(tasks=2), workflow_id="stream/take/1", claimant="the-second"
        )
        after = await replacement.stream_state()
        assert after.ownership_epoch == before.ownership_epoch + 1
        assert after.ownership_claims == before.ownership_claims + 1
        assert after.cursor == before.cursor
        assert after.attempts == before.attempts
        assert after.turnovers == before.turnovers
        # And the writer that was replaced is fenced, which is what a takeover means.
        assert await refused(caller.stream.confirm_state()) == "fenced_writer"


@pytest.mark.durable
async def test_every_execution_of_a_chain_replays_on_its_own(env, turnover_at) -> None:
    """Each link, fetched by its own run identifier and replayed, not just the last one.

    A replay of the latest execution can pass while a predecessor's own continuation command was
    nondeterministic, because that command is not in the successor's history at all. So the chain
    is walked from the first execution, each history is fetched by the run identifier the link
    before it named, and each is replayed on its own.
    """
    turnover_at(6)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=3), workflow_id="stream/chain/1")
        await caller.stream.close_queue()
        await served(caller)
        chain = await run_ids(env.client, "stream/chain/1")
        assert len(chain) >= 3, "the lowered trigger has to produce several links"
        for run_id in chain:
            handle = env.client.get_workflow_handle("stream/chain/1", run_id=run_id)
            await stream_replayer().replay_workflow(await handle.fetch_history())


@pytest.mark.durable
async def test_an_execution_that_never_latched_acquires_the_boundary_when_traffic_resumes(
    env, turnover_at
) -> None:
    """An open generation from before the boundary existed can still cross one, live.

    The branch is behind a marker, and where that marker is read decides whether an execution
    already running can ever take it. Read at the top of the run method it would be read during
    every replay of a history recorded before it, answered no, and remembered as no for the rest
    of that execution: the generation would replay correctly and never acquire a turnover.

    Read at the decision point it is read for the first time when the generation decides to turn
    over, which a replay of an older history never reaches. This drives a generation that never
    latched, with the sticky cache off so every Workflow Task replays the whole history first,
    and then lowers the trigger under live traffic.
    """
    turnover_at(10_000)
    async with stream_worker(env.client, cached_workflows=0):
        caller = await open_stream(env.client, make_start(tasks=3), workflow_id="stream/old/1")
        await caller.stream.close_queue()
        task = await caller.take()
        await caller.work(task)
        await caller.take()
        assert (await caller.stream.stream_state()).turnovers == 0

        await _latch(caller, turnover_at, env.client, "stream/old/1")
        for _ in range(400):
            if (await caller.stream.stream_state()).turnovers == 1:
                break
            await asyncio.sleep(0.05)
        else:
            raise AssertionError("an open generation never acquired the boundary")
        turnover_at(10_000)
        assert [message.kind for message in await served(caller)][-1] == "done"


def test_a_start_recorded_before_the_carrier_existed_decodes_with_none() -> None:
    """The recorded fixture's own bytes, decoded by this code, carry no projection.

    The field is defaulted, so bytes that never held it decode with the default rather than
    failing. That is the whole reason the projection rides on the start rather than as a second
    argument: a second argument changes how many payloads a history has, and every recorded
    history has one.
    """
    import base64

    from temporalio.api.common.v1 import Payload
    recorded = json.loads(
        Path("tests/_fixtures/recorded_before_policies.json").read_text(encoding="utf-8")
    )
    started = recorded["events"][0]["workflowExecutionStartedEventAttributes"]
    assert len(started["input"]["payloads"]) == 1
    assert "carry" not in json.dumps(started["input"])
    # And it is put through the converter the service is given, which is what actually decides
    # whether those bytes still name a generation this code can serve.
    bytes_recorded = base64.b64decode(started["input"]["payloads"][0]["data"])
    decoded = default_converter().payload_converter.from_payload(
        Payload(metadata={"encoding": b"json/plain"}, data=bytes_recorded), StreamStart
    )
    assert isinstance(decoded, StreamStart)
    assert decoded.carry is None
    assert len(decoded.tasks) >= 1


@pytest.mark.durable
async def test_a_carrier_that_will_not_fit_is_refused_and_the_size_is_recorded(
    env, turnover_at, monkeypatch
) -> None:
    """An unsupported profile stops honestly: the generation keeps serving and says why.

    The service bounds one payload, and the replaced start is one payload. A generation whose
    projection has outgrown the ceiling cannot hand itself on, and pretending otherwise would
    turn a bounded run into a failed one at the moment it tried. So the size is measured at the
    quiet point, the boundary is refused, the number is published where a launcher reads it, and
    the generation goes on serving in the execution it is already in.
    """
    monkeypatch.setattr(kernel_workflow, "TURNOVER_PAYLOAD_CEILING_BYTES", 1)
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/size/1")
        await caller.stream.close_queue()
        task = await caller.take()
        await caller.work(task)
        await _latch(caller, turnover_at, env.client, "stream/size/1")
        for _ in range(400):
            state = await caller.stream.stream_state()
            if state.turnover_refused_bytes is not None:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the oversized carrier was never refused")

        assert state.turnovers == 0
        assert state.turnover_refused_bytes > 1
        # And it keeps serving: the generation reaches Done in the execution it is in.
        turnover_at(10_000)
        assert [message.kind for message in await served(caller)][-1] == "done"


@pytest.mark.durable
async def test_the_supported_profile_encodes_well_under_the_ceiling(env, turnover_at) -> None:
    """The measurement the ceiling is set from: a full roster, driven to the end, sized.

    The measured object is the whole replaced start as the configured converter produces it,
    which is the original roster, the projection, the payload metadata and the encoding
    overhead together. It is measured at the largest quiet point a generation of this shape has,
    which is the boundary that falls after the last payload of the last task is presented.

    That boundary is reached rather than arranged for. The roster is driven to Done through every
    boundary the generation decides on for itself, so the projection this measures is one a chain
    of executions really carried, and the last of them is aligned with the last presentation so
    that the largest projection is the one measured. The measurement itself is taken by lowering
    the ceiling to one byte there, which makes the generation publish the size it measured and go
    on serving, so the run still reaches Done afterwards.

    The traffic is the declared workload rather than the smallest one that fills a roster. Every
    task takes fifteen world calls, because a call is the commonest thing an agent does and the
    identity of each one is fresh; every task loses one reply, which is the same Update sent again
    and costs the journal nothing; and every lost reply leaves the transport holding a result it
    has to confirm the stream before handing over, which is a fresh identifier answered with a
    whole state reading. Those readings are what makes this workload the hard one: two hundred
    snapshots of one growing projection, differing in a few entries each.

    The identifiers are the transport's own shape rather than this file's. A gateway mints a
    random one for every request and every attestation, and cites objects by the digest of the
    bytes it stored. Sequential identifiers and one repeated reference would be a few hundred
    distinct strings where a real run has thousands, and what the sharing can do with them is
    exactly what this measurement is about.

    The bodies are incompressible for the same reason. A real roster's descriptions are prose and
    pack well, so packing them is not what this has to survive: what it has to survive is a roster
    whose bodies pack not at all, which is the bound rather than the expectation.
    """
    roster = 200
    calls = 15
    body = secrets.token_hex(1024)
    async with stream_worker(env.client):
        start = make_start(tasks=roster, body=body)
        start = replace(
            start,
            tasks=[replace(item, body=secrets.token_hex(1024)) for item in start.tasks],
        )
        caller = await open_stream(env.client, start, workflow_id="stream/profile/1")
        caller.next_id = lambda: secrets.token_hex(16)
        caller.reference = lambda message: sha256(
            message.visible_text.encode("utf-8")
        ).hexdigest()
        await caller.stream.close_queue()
        ceiling = kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
        try:
            for index in range(roster):
                task = await caller.take()
                assert task.kind == "task"
                for _ in range(calls):
                    call = EnvironmentCall(
                        call_id=secrets.token_hex(16), attempt_id=task.attempt_id or ""
                    )
                    await caller.stream.begin_environment_call(call)
                    await caller.stream.end_environment_call(call)
                    # The reply the transport was waiting for is lost, so it sends the same Update
                    # again. The generation answers it from what it answered the first time.
                    await caller.stream.end_environment_call(call)
                # And the waiter that lost the reply is gone, so the transport confirms the stream
                # before it hands the result over. That is a fresh identifier every time.
                await caller.stream.confirm_state()
                await caller.work(task)
                if index == roster - 1:
                    # The last legal boundary, aligned with the presentation that follows: the
                    # trigger is set so the next accepted Update latches, and the ceiling is
                    # lowered so that boundary is measured and refused rather than crossed.
                    turnover_at(await _accepted_updates(env.client, "stream/profile/1") + 2)
                    kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES = 1
                payload = await caller.take()
                assert payload.kind == "payload"

            for _ in range(2000):
                state = await caller.stream.stream_state()
                if state.turnover_refused_bytes is not None:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("the generation never measured its carrier")
        finally:
            kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES = ceiling
        turnover_at(10_000)
        measured = state.turnover_refused_bytes
        alone = _encoded(start)
        print(
            f"\nprofile: {roster} tasks driven to Done, {calls} world calls each, one lost reply"
            f" and one confirmation each, {len(body)} byte incompressible bodies,"
            f" gateway-shaped identifiers"
            f"\n  boundaries crossed:            {state.turnovers}"
            f"\n  the original start alone:      {alone} bytes"
            f"\n  the last boundary measured:    {measured} bytes"
            f"\n  what the projection adds:      {measured - alone} bytes"
            f"\n  the ceiling:                   {kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES}"
            f"\n  the service's payload limit:   {2 * 1024 * 1024} bytes"
        )
        # The chain is real: the generation crossed boundaries of its own before this one, so the
        # projection measured here is one that had already been carried across executions.
        assert state.turnovers > 0
        assert measured < kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
        # And with the room the recorded profile was given, asserted against what was actually
        # encoded rather than against a number written down once. This is the assertion that moves
        # when a carrier field is added or a row grows, because its input is measured on every run.
        headroom = 1 - measured / kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
        assert headroom > 0.25, "a roster of this shape has to sit well under the ceiling"
        # And the generation that measured the boundary it would not cross serves on to Done.
        assert (await caller.take()).kind == "done"


async def _carried(client: Client, workflow_id: str, run_id: str) -> Dict[str, Any]:
    """The projection one execution was handed, read out of the bytes it was started with.

    The bytes are the packed carrier, so this reads them back the way the workflow does and hands
    over the plain value inside. What the callers of this ask about is what a continuation was
    given, which is that value and not the string it travelled as.
    """
    history = await client.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
    started = history.events[0].workflow_execution_started_event_attributes
    carry = json.loads(started.input.payloads[0].data.decode("utf-8"))["carry"]
    return asdict(_unpacked(StreamCarry(**carry)))


@pytest.mark.durable
async def test_the_carried_projection_is_written_in_one_order_and_only_one(
    env, turnover_at
) -> None:
    """Every unordered collection sorted, and the one that means an order keeping it.

    This value becomes a command in a history that has to replay to the same bytes. A set
    serialized in whatever order a hash table happened to hold it would make the same boundary
    produce different bytes on a replay, and the replay would fail on a difference nothing in
    the generation caused. The presentations are the exception that proves it: their order is
    what a harness reconciles its own transcript against, so it is kept rather than sorted.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=3), workflow_id="stream/order/1")
        await caller.stream.close_queue()
        for _ in range(2):
            task = await caller.take()
            await caller.work(task)
            await caller.take()
        await _cross_a_boundary(caller, turnover_at, "stream/order/1", env.client)

        chain = await run_ids(env.client, "stream/order/1")
        carry = await _carried(env.client, "stream/order/1", chain[-1])
        assert carry["committed_blobs"] == sorted(carry["committed_blobs"])
        assert carry["handed_out_attempt_ids"] == sorted(carry["handed_out_attempt_ids"])
        assert list(carry["wait_reasons"]) == sorted(carry["wait_reasons"])
        for name in ("attempts", "obligations"):
            keys = [row["attempt_id"] for row in carry[name]]
            assert keys == sorted(keys), name
        for name in ("pull_requests", "info_requests", "terminal_requests", "finalize_requests"):
            keys = [row["request_id"] for row in carry[name]]
            assert keys == sorted(keys), name
        attested = [row["attestation_id"] for row in carry["attestations"]]
        assert attested == sorted(attested)
        # The presentations keep the order they were committed in, which is not their sorted one.
        orders = [row["order"] for row in carry["presented"]]
        assert orders == list(range(len(orders)))
        # And the text of a message already presented is not carried, because nothing can ask
        # for it again: the request that reserved it is told it has been presented.
        assert all(row["message"]["visible_text"] == "" for row in carry["pull_requests"])


@pytest.mark.durable
async def test_eight_attempts_being_worked_at_once_cross_a_boundary_together(
    env, turnover_at
) -> None:
    """Attempts in flight do not block a boundary, and they are all still in flight after it.

    An active attempt is state rather than something part way through: nothing about it is
    waiting on an answer this generation owes, and the world it belongs to is somewhere this
    stream cannot see either way. So a generation working eight at once crosses, and the eight
    are still eight on the other side.
    """
    turnover_at(10_000)
    start = replace(make_start(tasks=8), capacity=8)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, start, workflow_id="stream/eight/1")
        await caller.stream.close_queue()
        tasks = [await caller.take() for _ in range(8)]
        assert all(message.kind == "task" for message in tasks)
        before = await caller.stream.stream_state()
        assert before.capacity_in_use == 8

        await _cross_a_boundary(caller, turnover_at, "stream/eight/1", env.client)

        after = await caller.stream.stream_state()
        assert after.capacity_in_use == 8
        assert after.attempts == before.attempts
        assert after.task_checkpoints == before.task_checkpoints
        for task in tasks:
            await caller.work(task)
        assert [message.kind for message in await served(caller)][-1] == "done"


@pytest.mark.durable
async def test_a_payload_built_but_not_yet_offered_crosses_with_its_body(env, turnover_at) -> None:
    """An obligation that can still be offered keeps its candidate, and is offered afterwards.

    The bytes of a payload are built at the seal and offered at the next pull, so a boundary can
    land between the two. What crosses is the candidate itself, because nothing else could
    produce those exact bytes again, and the count of what was built crosses beside it because
    that count is inside the projection a presentation attests to.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/built/1")
        await caller.stream.close_queue()
        task = await caller.take()
        await caller.work(task)
        before = await caller.stream.stream_state()
        assert before.obligations[task.attempt_id or ""] == "eligible"
        assert before.materialization_count == 1

        await _cross_a_boundary(caller, turnover_at, "stream/built/1", env.client)

        after = await caller.stream.stream_state()
        assert after.obligations == before.obligations
        assert after.materialization_count == before.materialization_count
        payload = await caller.take()
        assert payload.kind == "payload"
        assert payload.message_id == oid(0x103)


@pytest.mark.durable
async def test_a_seal_in_flight_holds_the_boundary_until_its_answer_lands(
    env, turnover_at
) -> None:
    """A handler awaiting an Activity is unfinished, and an unfinished handler blocks a boundary.

    A boundary is a fresh history, and a fresh history cannot finish an Activity the one before
    it started. The answer would be lost and the caller would never learn what became of its
    filing, so the wait covers the whole batch behind a seal and not only the state it leaves.
    """
    release = asyncio.Event()

    @activity.defn(name=GRADE_ATTEMPT)
    async def slow_grade(request: GradeAttemptInput) -> GradeAttemptResult:
        await release.wait()
        return await grade_attempt_activity(request)

    served_activities = [
        row for row in kernel_activities() if row is not grade_attempt_activity
    ] + [slow_grade]

    turnover_at(10_000)
    async with stream_worker(env.client, activities=served_activities):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/seal/1")
        await caller.stream.close_queue()
        task = await caller.take()
        filing = asyncio.create_task(caller.work(task))
        for _ in range(200):
            if (await caller.stream.stream_state()).attempts[task.attempt_id or ""] == "sealing":
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the filing never reached the seal")

        await _latch(caller, turnover_at, env.client, "stream/seal/1")
        for _ in range(10):
            await asyncio.sleep(0.02)
        assert (await caller.stream.stream_state()).turnovers == 0

        release.set()
        await filing
        for _ in range(400):
            if (await caller.stream.stream_state()).turnovers == 1:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the boundary never followed the seal")
        turnover_at(10_000)
        assert [message.kind for message in await served(caller)][-1] == "done"


@pytest.mark.durable
async def test_a_deadline_armed_before_a_boundary_still_ends_its_attempt_after_one(
    env, turnover_at
) -> None:
    """The timer does not cross; the time it was set for does, and the successor arms again.

    A deadline is an absolute moment on the generation's clock rather than an interval from
    whenever a worker last picked the generation up. So what crosses is that moment, the
    successor arms a timer for what is left of it, and the attempt ends when it would have
    ended.
    """
    turnover_at(10_000)
    start = replace(make_start(tasks=2), attempt_deadline_ms=60_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, start, workflow_id="stream/clock/1")
        await caller.stream.close_queue()
        task = await caller.take()
        assert (await caller.stream.stream_state()).attempts[task.attempt_id or ""] == "active"

        await _cross_a_boundary(caller, turnover_at, "stream/clock/1", env.client)
        assert (await caller.stream.stream_state()).attempts[task.attempt_id or ""] == "active"

        await env.sleep(timedelta(milliseconds=90_000))
        for _ in range(400):
            state = await caller.stream.stream_state()
            if state.attempts[task.attempt_id or ""] == "final_failed":
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the carried deadline never ended its attempt")
        assert state.final_failures[task.attempt_id or ""] == "deadline"


@pytest.mark.durable
async def test_what_a_query_read_before_a_boundary_an_update_after_one_still_answers(
    env, turnover_at
) -> None:
    """A read on one execution and a write on the next describe one generation.

    Every call goes out with no execution named, so a Query lands wherever the generation is and
    the Update after it lands wherever the generation is then. That is safe only because a
    boundary moves no field either of them reads: the cursor a pull is made from and the
    projection an attestation is built against are the same on both sides.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/read/1")
        await caller.stream.close_queue()
        read = await caller.stream.stream_state()

        await _cross_a_boundary(caller, turnover_at, "stream/read/1", env.client)

        # The request is built from what the Query said and sent to the execution that followed.
        offered = await caller.stream.pull(
            PullRequest(request_id=caller.next_id(), last_presented_cursor=read.cursor)
        )
        assert offered.kind == "task"
        assert (await caller.stream.stream_state()).stream_state_sha256 != read.stream_state_sha256


@pytest.mark.durable
async def test_the_wait_total_and_every_reason_behind_it_cross(env, turnover_at) -> None:
    """The reasons are published and nothing a presentation records could rebuild them.

    A Wait carries no reason on the wire, and the presented row for one keeps a digest and a
    kind. So the histogram is not derivable on the far side of a boundary from anything else
    that crossed: it crosses itself, canonically encoded, or the state a harness reads stops
    saying what its Waits were for.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        # The queue is left open, so a pull with nothing eligible waits rather than reaching
        # Done, and the reason it waits for is the open queue.
        caller = await open_stream(env.client, make_start(tasks=1), workflow_id="stream/wait/2")
        task = await caller.take()
        await caller.work(task)
        await caller.take()
        waited = await caller.take()
        assert waited.kind == "wait"
        before = await caller.stream.stream_state()
        assert before.wait_count == 1
        assert before.wait_reasons == {"queue_open": 1}

        await _cross_a_boundary(caller, turnover_at, "stream/wait/2", env.client)

        after = await caller.stream.stream_state()
        assert after.wait_count == before.wait_count
        assert after.wait_reasons == before.wait_reasons
        # And what follows is counted on top of what crossed rather than starting again.
        await caller.stream.close_queue()
        assert (await caller.take()).kind == "done"
        assert (await caller.stream.stream_state()).wait_count == 1


@pytest.mark.durable
async def test_a_finalization_receipt_is_the_one_that_was_given_and_not_a_rebuilt_one(
    env, turnover_at
) -> None:
    """The receipt crosses whole, capacity in use and cascade included, as they were.

    A receipt reports how much capacity was in use and which other attempts the ending reached,
    both read at the moment the ending happened. A retry answered from a later projection would
    report a different generation: capacity moves as attempts end and the cascade is the list
    that ending produced rather than the list of everything that has ended since.
    """
    turnover_at(10_000)
    gated = replace(make_start(tasks=3), capacity=3)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, gated, workflow_id="stream/final/1")
        await caller.stream.close_queue()
        first = await caller.take()
        second = await caller.take()
        assert (await caller.stream.stream_state()).capacity_in_use == 2

        ending = FinalizeRequest(
            request_id=caller.next_id(), attempt_id=second.attempt_id or "", reason=STEP_CAP
        )
        receipt = await caller.stream.finalize(ending)
        assert receipt.capacity_in_use == 1
        assert receipt.reason == STEP_CAP

        await _cross_a_boundary(caller, turnover_at, "stream/final/1", env.client)

        # The same logical request, answered from the table it bound rather than from now, when
        # the other attempt has since been worked and the capacity has moved again.
        await caller.work(first)
        assert await caller.stream.finalize(ending) == receipt


@pytest.mark.durable
async def test_info_across_a_boundary_still_counts_an_attempt_it_handed_out(
    env, turnover_at
) -> None:
    """What was handed out stays handed out, whatever became of it afterwards.

    The count is read from the reservations the generation made rather than from the states its
    attempts are in, because an ending overwrites the state and leaves the reservation where it
    was. Those reservations cross as the set of who was handed out, so an attempt handed out
    before a boundary and finally failed after one is still counted as consumed.
    """
    turnover_at(10_000)
    asking = replace(make_start(tasks=3), info=True)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, asking, workflow_id="stream/info/1")
        await caller.stream.close_queue()
        task = await caller.take()
        before = await caller.take_info()
        assert json.loads(before.visible_text)["consumed"] == 1

        await _cross_a_boundary(caller, turnover_at, "stream/info/1", env.client)

        await caller.stream.finalize(
            FinalizeRequest(
                request_id=caller.next_id(), attempt_id=task.attempt_id or "", reason=STEP_CAP
            )
        )
        after = await caller.take_info()
        counted = json.loads(after.visible_text)
        assert counted["consumed"] == 1
        assert counted["in_flight"] == 0
        assert counted["remaining"] == 2


@pytest.mark.durable
async def test_the_same_ownership_claim_after_a_boundary_returns_the_receipt_it_returned(
    env, turnover_at
) -> None:
    """A claim replayed under its own identifier is answered, not told it fenced itself.

    A claim moves the epoch, and the compare that lets it is against the epoch it read. Sent
    again after a boundary it would read the epoch it moved and be told it was fenced by its
    own swap. A fresh identifier making the same logical claim is the case that really is
    fenced, because a second claimant reading a stale epoch is exactly what the compare is for.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/claim/1")
        await caller.stream.close_queue()
        handle = caller.stream.handle
        claim = OwnershipClaim(
            claimant_id="the-replacement",
            previous_epoch=1,
            fencing_token="f" * 64,
            configuration_hash=configuration_hash(make_start(tasks=2)),
            reason="resume",
        )
        receipt = await handle.execute_update(
            StreamWorkflow.claim_ownership, claim, id="own-1-the-replacement"
        )
        assert receipt.ownership_epoch == 2

        # The claim fenced the caller that opened the generation, so the latch is reached
        # through the writer the claim installed rather than through the one it replaced.
        writer = Writer(ownership_epoch=2, fencing_token="f" * 64)
        turnover_at(await _accepted_updates(env.client, "stream/claim/1") + 1)
        await handle.execute_update(
            StreamWorkflow.confirm_state, args=[writer], id="confirm-2-latching"
        )
        for _ in range(400):
            state = await handle.query(StreamWorkflow.stream_state)
            if state.turnovers == 1:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the generation never crossed a boundary")
        turnover_at(10_000)

        assert (
            await handle.execute_update(
                StreamWorkflow.claim_ownership, claim, id="own-1-the-replacement"
            )
            == receipt
        )
        # A fresh identifier for the same logical claim reads a stale epoch and is fenced.
        assert (
            await refused(
                handle.execute_update(
                    StreamWorkflow.claim_ownership, claim, id="own-1-the-replacement-again"
                )
            )
            == "fenced_writer"
        )
        assert state.ownership_epoch == 2


@pytest.mark.durable
async def test_the_same_consumer_claim_after_a_boundary_returns_the_receipt_it_returned(
    env, turnover_at
) -> None:
    """The consumer binding is answered from what it was answered with, refusal included.

    A binding normally rebuilds to the same receipt, so the cache is not what makes an ordinary
    retry work. What it is for is the refusal: a claim refused because another consumer held the
    generation must stay refused after a boundary, rather than being reconsidered against a
    state that has moved.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/bind/1")
        handle = caller.stream.handle
        writer = caller.stream.writer
        other = ConsumerClaim(consumer_id="someone-else", claim_hash=CLAIM_HASH)
        assert (
            await refused(
                handle.execute_update(
                    StreamWorkflow.claim_consumer, args=[other, writer], id="claim-1-someone-else"
                )
            )
            == "consumer_conflict"
        )

        await _cross_a_boundary(caller, turnover_at, "stream/bind/1", env.client)

        assert (
            await refused(
                handle.execute_update(
                    StreamWorkflow.claim_consumer, args=[other, writer], id="claim-1-someone-else"
                )
            )
            == "consumer_conflict"
        )
        # And the consumer that does hold it is still the one that holds it.
        assert (await caller.stream.stream_state()).consumer_id == "harness-1"


def test_one_worker_per_run_on_that_run_s_own_task_queue(tmp_path) -> None:
    """The deployment invariant that keeps a chain away from a Worker that predates the carrier.

    A Worker whose package is older than the projection would decode the start, silently drop
    the field it does not know, and serve the roster again from the beginning. Nothing in that
    older code could refuse it: the refusal did not exist there, and a decoder that ignores
    fields it has never heard of is what makes a defaulted field safe in the first place.

    So the exclusion is not a check inside the workflow, it is the shape of the deployment. The
    package embeds its own service, a run gets its own directory and its own task queue, and one
    Worker from one package polls it. This asserts that shape rather than a refusal: the run
    directory names one task queue, the package that created it wrote its version down beside
    the manifest, and the Worker this package builds serves that queue and one workflow type.
    """
    from shogym import __version__ as package_version
    from shogym.serve.protocol_v2.rundir import create_run_directory, serving_record

    run = create_run_directory(
        tmp_path, workflow_id="stream/deploy/1", task_queue="run-own-queue", configuration_hash="c"
    )
    recorded = serving_record(tmp_path)
    assert recorded == {
        "package": "shogym",
        "version": package_version,
        "task_queue": "run-own-queue",
    }
    assert run.manifest.task_queue == "run-own-queue"
    # And the manifest itself is unchanged: five fields, checked for exact equality, so a
    # directory this code wrote is still one that code without the record can open.
    assert set(run.manifest.to_wire()) == {
        "protocol_version",
        "schedule_version",
        "workflow_id",
        "task_queue",
        "configuration_hash",
    }


# What only the service's own cap can show.
#
# The time-skipping test server does not enforce the cap: it accepts Updates past it and takes no
# dynamic configuration to lower it. So the one thing the whole design exists to avoid, a
# generation refused at the cap, can only be driven on the local dev service with the limit
# configured down. These are that, and they run in real time.


@asynccontextmanager
async def _dev_service(*settings: str) -> AsyncIterator[WorkflowEnvironment]:
    """The local dev service, with the limits these tests are about configured down."""
    extra: List[str] = []
    for setting in settings:
        extra += ["--dynamic-config-value", setting]
    try:
        environment = await WorkflowEnvironment.start_local(
            download_dest_dir=str(temporal_home()), dev_server_extra_args=extra
        )
    except Exception as error:  # noqa: BLE001 - an absent dev service is a skip, not a failure
        pytest.skip(f"the local dev service is unavailable: {error}")
    try:
        yield environment
    finally:
        await environment.shutdown()


@DEV_SERVER_ONLY
@pytest.mark.dev_server
@pytest.mark.network
async def test_a_generation_that_would_have_reached_the_cap_reaches_done_instead(
    turnover_at,
) -> None:
    """The whole point, end to end, against the limit itself.

    The service is configured with a cap this roster costs more than twice over, and with its
    own suggestion pinned to the cap so that the only thing that can act early is the kernel's
    own counter. Both arms run against that. Left alone, the generation is refused by the
    service part way through and the transport has no answer for it: the refusal carries no
    protocol code, so it reaches the caller as a fault. Given a trigger under the cap, the same
    roster reaches Done with every call answered normally.
    """
    cap = 60
    roster = 20
    async with _dev_service(
        f"history.maxTotalUpdates={cap}",
        "history.maxTotalUpdates.suggestContinueAsNewThreshold=1.0",
    ) as env:
        # The control. One execution, and the roster costs more Updates than one may accept.
        turnover_at(10_000)
        async with stream_worker(env.client):
            stopped = await open_stream(
                env.client, make_start(tasks=roster), workflow_id="stream/cap/control"
            )
            await stopped.stream.close_queue()
            with pytest.raises(Exception) as refusal:
                await served(stopped)
            assert "updates" in str(refusal.value).lower()
            assert protocol_error_code(refusal.value) is None

        turnover_at(cap // 3)
        async with stream_worker(env.client):
            caller = await open_stream(
                env.client, make_start(tasks=roster), workflow_id="stream/cap/1"
            )
            await caller.stream.close_queue()
            seen = await served(caller)
            state = await caller.stream.stream_state()
    assert [message.kind for message in seen] == [
        "task",
        "seal_ack",
        "payload",
    ] * roster + ["done"]
    assert state.turnovers >= 3, "one execution could not have accepted this many Updates"


@DEV_SERVER_ONLY
@pytest.mark.dev_server
@pytest.mark.network
async def test_the_services_own_suggestion_is_a_second_trigger_under_update_pressure(
    turnover_at,
) -> None:
    """With the kernel's counter out of reach, the service's own hint still brings a boundary.

    The service publishes one boolean with three possible causes and the generation cannot tell
    which fired. It is taken as a second trigger for exactly this: a deployment whose cap is
    lower than the one this package supports, where a counter set for the supported cap would
    never latch in time. This pins the cap low, leaves the trigger far above anything the run
    reaches, and drives the roster to Done on the hint alone.
    """
    async with _dev_service("history.maxTotalUpdates=60") as env:
        turnover_at(10_000)
        async with stream_worker(env.client):
            caller = await open_stream(
                env.client, make_start(tasks=20), workflow_id="stream/hint/1"
            )
            await caller.stream.close_queue()
            seen = await served(caller)
            state = await caller.stream.stream_state()
    assert [message.kind for message in seen][-1] == "done"
    assert state.turnovers >= 2


@DEV_SERVER_ONLY
@pytest.mark.dev_server
@pytest.mark.network
async def test_every_update_is_answered_while_only_one_may_be_in_flight(turnover_at) -> None:
    """The boundary interleaving, under the configuration that makes it reachable.

    An Update the service admitted but the generation has not seen when the continuation is
    proposed is aborted retryably and re-aimed at the execution that follows. With only one
    Update allowed in flight at a time, a boundary reached under continuous traffic puts a
    request in exactly that position. What has to hold is that every call is answered, and that
    no caller is handed the service's own words about an execution that has finished.
    """
    async with _dev_service("history.maxInFlightUpdates=1") as env:
        turnover_at(8)
        async with stream_worker(env.client):
            caller = await open_stream(
                env.client, make_start(tasks=4), workflow_id="stream/inflight/1"
            )
            await caller.stream.close_queue()
            seen = await served(caller)
            state = await caller.stream.stream_state()
    assert [message.kind for message in seen][-1] == "done"
    assert state.turnovers >= 2


@DEV_SERVER_ONLY
@pytest.mark.dev_server
@pytest.mark.network
async def test_the_copied_run_reader_answers_for_a_chain_as_one_generation(
    turnover_at, tmp_path
) -> None:
    """A read off a copy of a run that crossed boundaries reports the whole roster.

    The reader copies the run's database, brings a private service up on it and asks one Query
    on the latest execution. That answers for the whole generation only because the whole
    projection crossed: every attempt's record and every presentation, in order. And the mark it
    brackets the read with is the pair, so a copy whose execution moved under it is refused for
    the right reason rather than for a count that went backwards.
    """
    from shogym.serve.protocol_v2.reader import read_records
    from shogym.serve.protocol_v2.rundir import create_run_directory

    start = make_start(tasks=3)
    create_run_directory(
        tmp_path,
        workflow_id="stream/reader/1",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash=configuration_hash(start),
    )
    turnover_at(6)
    async with durable_client(run_directory=tmp_path) as client:
        async with stream_worker(client):
            caller = await open_stream(client, start, workflow_id="stream/reader/1")
            await caller.stream.close_queue()
            await served(caller)
            state = await caller.stream.stream_state()
            assert state.turnovers >= 2
    turnover_at(10_000)

    records = await read_records(tmp_path)
    assert [row.attempt_id for row in records.records] == [attempt(index) for index in range(3)]
    assert all(row.state == "ack_presented" for row in records.records)
    assert all(row.payload_delivered for row in records.records)
    assert [row.order for row in records.presentations] == list(
        range(len(records.presentations))
    )


@pytest.mark.durable
async def test_a_call_waits_out_a_boundary_a_world_call_is_holding_open(
    env, turnover_at, monkeypatch
) -> None:
    """A boundary blocked by a grant is not a stalled generation, however long it takes.

    A grant is held for as long as the agent's own tool call runs, which is minutes at a stretch,
    and the boundary waits for its end. A call rejected meanwhile has to wait for that too. If
    the wait were bounded by elapsed time, a long world call would turn a healthy boundary into a
    fault for whoever else was calling, which is a smaller version of the wedge this change
    exists to remove.

    So the bound is the generation's own liveness, and this pins it: the clock that gives up is
    set to nothing at all, the grant is held for far longer than that, and the waiting call is
    answered normally once the grant comes back and the boundary follows.
    """
    monkeypatch.setattr("shogym.serve.protocol_v2.kernel.runtime._TURNOVER_STALLED_AFTER", 0.0)
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=3), workflow_id="stream/long/1")
        await caller.stream.close_queue()
        task = await caller.take()
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        await caller.stream.begin_environment_call(call)
        await _latch(caller, turnover_at, env.client, "stream/long/1")
        state = await caller.stream.stream_state()
        assert state.turnover_requested is True
        assert state.environment_call == call.call_id

        # A call that cannot make the boundary quiet. It is rejected and it waits, and the
        # clock that would give up on a stalled generation has already run out.
        ending = FinalizeRequest(
            request_id=caller.next_id(), attempt_id=attempt(2), reason=STEP_CAP
        )
        waiting = asyncio.create_task(caller.stream.finalize(ending))
        for _ in range(75):
            await asyncio.sleep(0.02)
        assert not waiting.done(), "the call gave up on a boundary a world call was holding open"
        assert (await caller.stream.stream_state()).turnovers == 0

        # The world call comes back, the boundary follows, and the waiting call is answered.
        await caller.stream.end_environment_call(call)
        receipt = await asyncio.wait_for(waiting, timeout=30)
        assert receipt.attempt_id == attempt(2)
        assert receipt.reason == STEP_CAP
        assert (await caller.stream.stream_state()).turnovers == 1


def test_the_wait_rule_reads_a_generation_reaching_for_a_boundary_from_its_state() -> None:
    """What a caller held off for a boundary keeps waiting on, case by case.

    A rejected call has to tell a generation working towards a boundary from one that has
    stopped, and it has only the state to tell them apart with. Six readings say it is still
    working and one says it is not, and every one of the six is something that clears by work
    somebody else is doing rather than by time passing. The store read is the one worth naming
    twice: a claim reading the store moves no projection and names nothing else, so without it a
    claim following its ordinary retry policy would read as a generation that had stopped.
    """
    from shogym.serve.protocol_v2.kernel.runtime import (
        _still_reaching_for_a_boundary,
        _where_it_stands,
    )

    def reading(**overrides: Any) -> StreamState:
        return replace(_a_generation_holding_nothing(), **overrides)

    standing = reading()
    mark = _where_it_stands(standing)
    # The one shape a caller stops waiting on: holding traffic back, naming nothing, reading
    # nothing, and not moving since the last look.
    assert not _still_reaching_for_a_boundary(standing, mark)
    # And every shape it does not stop on.
    assert _still_reaching_for_a_boundary(reading(turnover_requested=False), mark)
    assert _still_reaching_for_a_boundary(reading(pending_message_id=oid(7)), mark)
    assert _still_reaching_for_a_boundary(reading(environment_call="call-1"), mark)
    assert _still_reaching_for_a_boundary(reading(prepared_seals={attempt(0): "r-1"}), mark)
    assert _still_reaching_for_a_boundary(reading(verifying=1), mark)
    assert _still_reaching_for_a_boundary(reading(stream_state_sha256="b" * 64), mark)
    # A store read that finished since the last look is movement no projection can show.
    assert _still_reaching_for_a_boundary(reading(verification_batches=1), mark)
    # And a first look has nothing to compare against.
    assert _still_reaching_for_a_boundary(standing, None)


@pytest.mark.durable
async def test_what_a_caller_is_handed_when_it_stops_waiting_is_not_a_refusal(
    env, turnover_at, monkeypatch
) -> None:
    """The give-up path hands back the rejection as the fault it arrived as.

    There is no protocol code on it, so the transport raises it rather than mapping it to
    anything the agent has a name for. What there is not, on this path or any other, is a new
    code for the agent to be shown.

    Reaching this path takes a generation that is holding traffic back while naming nothing,
    reading nothing and standing still, which this workflow has no way to be: every await in
    every handler either names the state it is holding or counts itself as a store read. So the
    reading is the thing forced here, and what is under test is what the transport does with it.
    """
    monkeypatch.setattr(
        "shogym.serve.protocol_v2.kernel.runtime._still_reaching_for_a_boundary",
        lambda state, moved: False,
    )
    monkeypatch.setattr("shogym.serve.protocol_v2.kernel.runtime._TURNOVER_STALLED_AFTER", 0.0)
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/stop/1")
        await caller.stream.close_queue()
        message = await caller.pull()
        await _latch(caller, turnover_at, env.client, "stream/stop/1")

        with pytest.raises(WorkflowUpdateFailedError) as gave_up:
            await caller.stream.finalize(
                FinalizeRequest(
                    request_id=caller.next_id(), attempt_id=attempt(1), reason=STEP_CAP
                )
            )
        assert turnover_pending(gave_up.value)
        assert protocol_error_code(gave_up.value) is None, "not a code the agent has ever seen"
        # And the generation is untouched by the call that gave up: it still holds the one
        # message it owes, and its boundary is still ahead of it.
        state = await caller.stream.stream_state()
        assert state.pending_message_id == message.message_id
        assert state.turnovers == 0


@pytest.mark.durable
async def test_an_execution_the_branch_is_unavailable_to_keeps_serving_to_the_end(
    env, turnover_at
) -> None:
    """A generation recorded before this branch existed must not be gated by it.

    The branch is behind a marker, and an execution whose own replay reaches the decision is
    told the marker is absent and remembered that way. That answer is correct: such an execution
    cannot continue as new, because the command it would issue is not in the history it is
    replaying against.

    What must not follow is that it starts holding traffic back. A generation that can neither
    turn over nor serve is wedged where it stands, at whatever point the trigger happens to be
    crossed, which is worse than the cap it was avoiding. So everything the latch switches on
    reads whether the branch is actually available, and an execution it is not available to
    behaves exactly as it did before there was a boundary to reach: it keeps serving, it reports
    that it is holding nothing back, and it reaches Done in the execution it is already in.
    """
    turnover_at(10_000)
    async with stream_worker(env.client, cached_workflows=0):
        caller = await open_stream(env.client, make_start(tasks=3), workflow_id="stream/before/1")
        await caller.stream.close_queue()
        task = await caller.take()
        await caller.work(task)
        recorded = await _accepted_updates(env.client, "stream/before/1")
        assert (await caller.stream.stream_state()).turnover_requested is False

    # The trigger is now under what this execution has already accepted, so its own replay
    # reaches the decision and is told the branch is not there.
    turnover_at(max(1, recorded - 2))
    async with stream_worker(env.client, cached_workflows=0):
        assert (await caller.stream.stream_state()).turnover_requested is False
        assert [message.kind for message in await served(caller)][-1] == "done"
        finished = await caller.stream.stream_state()
    assert finished.turnovers == 0
    assert finished.turnover_requested is False
    assert finished.generation_state == "done"


@pytest.mark.durable
async def test_a_claim_reading_the_store_over_and_over_is_not_a_generation_that_stopped(
    env, turnover_at, monkeypatch
) -> None:
    """A store read that retries is progress, and the caller waiting on it has to see that.

    An ownership claim reads the store before it may swap the epoch, and it reads until the set
    stops growing, so a presentation that commits while the first batch runs adds a second one.
    Each batch is an Activity with a timeout of its own and retries of its own, and in the
    package's own policy that is three attempts a batch. Through all of it the projection stands
    still, nothing is pending, no grant is held and no seal is prepared: to a caller comparing
    projection hashes the generation looks stopped, and it is not.

    So the generation counts its store reads and the caller reads that count. This drives two
    batches, each retrying to its last attempt, and holds them open well past the clock that
    gives up. The caller waiting on them is the writer the claim installed rather than one about
    to be fenced, which is what makes it the case that matters: a caller that was going to be
    refused anyway proves nothing about waiting.
    """
    monkeypatch.setattr(kernel_workflow, "_ACTIVITY_TIMEOUT", timedelta(seconds=0.6))
    monkeypatch.setattr(
        kernel_workflow,
        "_ACTIVITY_RETRY",
        RetryPolicy(
            initial_interval=timedelta(seconds=0.01),
            backoff_coefficient=2.0,
            maximum_attempts=3,
        ),
    )
    monkeypatch.setattr("shogym.serve.protocol_v2.kernel.runtime._TURNOVER_STALLED_AFTER", 1.0)
    monkeypatch.setattr("shogym.serve.protocol_v2.kernel.runtime._TURNOVER_FIRST_WAIT", 0.01)
    monkeypatch.setattr("shogym.serve.protocol_v2.kernel.runtime._TURNOVER_LONGEST_WAIT", 0.01)

    second_transcript = "f" * 64
    reading = asyncio.Event()
    began = asyncio.Event()
    slow: Dict[str, Any] = {"first": None, "first_done": False, "batches": []}

    @activity.defn(name=VERIFY_BLOBS)
    async def slow_verification(payload: VerifyBlobsInput) -> BlobsVerified:
        info = activity.info()
        if reading.is_set() and slow["first"] is None and BLOB in payload.references:
            slow["first"] = info.activity_id
        this_one = info.activity_id == slow["first"] or (
            slow["first_done"] and list(payload.references) == [second_transcript]
        )
        if this_one:
            began.set()
            await asyncio.sleep(0.2)
            if info.attempt < 3:
                raise ApplicationError("the store did not answer")
            if info.activity_id == slow["first"]:
                slow["first_done"] = True
            slow["batches"].append(info.activity_id)
        return BlobsVerified(verified=list(payload.references), unverified=[])

    turnover_at(10_000)
    start = replace(make_start(tasks=3), capacity=2, blob_root="/nowhere")
    async with stream_worker(env.client, activities=_instead_of_verification(slow_verification)):
        caller = await open_stream(env.client, start, workflow_id="stream/reading/1")
        await caller.stream.close_queue()
        await caller.take()
        pending = await caller.pull()
        reading.set()
        claim = _in_the_background(
            _resume(env.client, start, workflow_id="stream/reading/1", claimant="the-reader")
        )
        # The presentation commits while the first batch is still retrying, which is what gives
        # the claim a second batch to read.
        await asyncio.wait_for(began.wait(), timeout=30)
        ack = await caller.stream.present(
            pending,
            attestation_id=caller.next_id(),
            transcript_blob=second_transcript,
            task_start_checkpoint_blob=BLOB,
        )
        caller.cursor = ack.cursor
        # A second claim takes the generation while the first is still reading, and it reads
        # quickly. The caller below therefore writes under the epoch that is current, not one
        # about to be replaced: a caller that was going to be refused anyway would prove nothing
        # about waiting. The first claim is still unfinished, and that is what holds the
        # boundary open with nothing named and nothing moving.
        caller.stream = await _resume(
            env.client, start, workflow_id="stream/reading/1", claimant="the-winner"
        )
        await _latch(caller, turnover_at, env.client, "stream/reading/1")
        held = await caller.stream.stream_state()
        assert held.turnover_requested is True
        assert held.pending_message_id is None and held.environment_call is None
        assert held.prepared_seals == {}

        started = time.monotonic()
        answer = await caller.stream.pull(
            PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor)
        )
        waited = time.monotonic() - started
    assert answer.kind in ("task", "payload", "wait")
    assert waited > 1.0, "the caller has to have waited past the clock that gives up"
    assert len(slow["batches"]) >= 2, "the claim has to have read the store more than once"
    await _let_go(claim)


@pytest.mark.durable
async def test_a_deadline_already_due_is_applied_before_the_boundary_rather_than_after_it(
    env, turnover_at
) -> None:
    """A clock that has run out belongs to the execution it ran out in.

    The timed wait records an expiry when it ends on the clock. It can also end on a generation
    deciding to continue as new, and it returns then without asking whether a deadline came due
    while it waited. A boundary that trusted only the recorded flag would carry an attempt across
    still active, with its ending written afterwards by an execution whose clock never reached
    it. The record would say the attempt was live at the moment the generation handed itself on,
    and it was not.

    This stages that interleaving: an attempt is left active with a deadline, no Worker is there
    to notice when it passes, the clock is advanced past it, the Update that latches is queued,
    and a Worker starts to find the timer and the latch waiting together. What the successor is
    handed has to say the attempt was already ended.
    """
    deadline_ms = 2000
    start = replace(make_start(tasks=2), attempt_deadline_ms=deadline_ms)
    turnover_at(10_000)
    async with stream_worker(env.client, cached_workflows=0):
        caller = await open_stream(env.client, start, workflow_id="stream/overdue/1")
        await caller.stream.close_queue()
        task = await caller.take()
        assert (await caller.stream.stream_state()).attempts[task.attempt_id or ""] == "active"
        accepted = await _accepted_updates(env.client, "stream/overdue/1")

    # No Worker is watching while the clock passes the deadline.
    await asyncio.sleep(deadline_ms / 1000 + 0.2)
    turnover_at(accepted + 1)
    latching = _in_the_background(caller.stream.confirm_state())
    await asyncio.sleep(0.05)

    async with stream_worker(env.client, cached_workflows=0):
        await latching
        for _ in range(400):
            state = await caller.stream.stream_state()
            if state.turnovers == 1:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the generation never crossed a boundary")
        turnover_at(10_000)

        chain = await run_ids(env.client, "stream/overdue/1")
        carried = await _carried(env.client, "stream/overdue/1", chain[-1])
        crossed = {row["attempt_id"]: row for row in carried["attempts"]}
        # The ending was written by the execution whose clock ran out, not the one after it.
        assert crossed[task.attempt_id or ""]["state"] == "final_failed"
        assert crossed[task.attempt_id or ""]["final_failure"] == "deadline"
        assert crossed[task.attempt_id or ""]["deadline_at"] is None
        assert state.final_failures[task.attempt_id or ""] == "deadline"


@pytest.mark.durable
async def test_repeatable_calls_cannot_spend_the_slot_a_held_grant_needs(
    env, turnover_at
) -> None:
    """The reserve is not one pool, because not every call on the whitelist is equally scarce.

    Several of the operations that clear a boundary can be sent again and again: a confirmation
    mints a fresh identifier every time by design, and a claim refused on a stale epoch can be
    made once more. The end of a held grant cannot. Nothing but the caller holding that grant can
    send it, and if repeats had already spent the reserve it would be refused for the want of a
    slot, leaving a generation that is holding traffic back for a boundary that its own gate has
    made unreachable.

    So the repeatable ones have an allowance of their own inside the reserve. This spends that
    allowance to its end and then requires the end of the grant to be admitted anyway.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/slots/1")
        await caller.stream.close_queue()
        task = await caller.take()
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        await caller.stream.begin_environment_call(call)
        await _latch(caller, turnover_at, env.client, "stream/slots/1")

        # Spend the allowance the repeats share, and keep going well past it.
        accepted = 0
        for _ in range(kernel_workflow.ADMISSION_RESERVE * 2):
            try:
                await _straight_at_the_generation(
                    caller,
                    StreamWorkflow.confirm_state,
                    caller.stream.writer,
                    update_id=caller.next_id(),
                )
                accepted += 1
            except WorkflowUpdateFailedError as error:
                assert turnover_pending(error)
        assert accepted <= kernel_workflow.ADMISSION_REPEATS
        assert accepted < kernel_workflow.ADMISSION_RESERVE

        # And the one call nothing else can make is still admitted.
        lease = await caller.stream.end_environment_call(call)
        assert lease.held is True
        for _ in range(400):
            if (await caller.stream.stream_state()).turnovers == 1:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the boundary never followed the end of the grant")


@pytest.mark.durable
async def test_a_filing_frozen_before_it_was_sent_is_answered_after_a_boundary(
    env, turnover_at
) -> None:
    """The window a filing this gateway owes can be open in, driven rather than reasoned about.

    A gateway that spends the last of an attempt's budget owes the filing that ends it. It gives
    the grant back first, because the generation holds the stream for the call it granted and
    would refuse a filing sent while it does, and only then writes the filing down and sends it.
    In between, the generation has no grant held, no message pending and no seal prepared: a
    boundary is legal there, and the record the gateway is holding is that owed filing.

    So the filing has to be answerable on the far side. This drives exactly that window: the
    grant comes back, the boundary happens, and the filing that was frozen before it is sent
    afterwards and sealed normally. Sent again under its own identifier it reaches the answer it
    already got, which is what the terminal table is for.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/owed/1")
        await caller.stream.close_queue()
        task = await caller.take()
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        await caller.stream.begin_environment_call(call)
        await caller.stream.end_environment_call(call)

        # The filing is fixed here, before it is sent, exactly as the gateway fixes it.
        filing = SealRequest(
            metadata=TerminalMetadata(
                request_id=caller.next_id(),
                last_presented_cursor=caller.cursor,
                attempt_id=task.attempt_id or "",
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
        )
        await _cross_a_boundary(caller, turnover_at, "stream/owed/1", env.client)

        acknowledgement = await caller.stream.seal(filing)
        assert acknowledgement.kind == "seal_ack"
        assert acknowledgement.attempt_id == task.attempt_id
        # Sent again inside this execution it is answered from the service's own record of the
        # identifier, without the handler running. That is Temporal's ordinary caching and it is
        # not what this test is about; it is asserted so the next line is read against it.
        assert await caller.stream.seal(filing) == acknowledgement

        # Across a boundary the service's record is gone and the handler decides. The attempt has
        # been sealed and its acknowledgement presented, so the answer is that it was presented,
        # which is the carried table judging the filing rather than any bytes being replayed.
        # And after the acknowledgement is presented and a boundary is crossed, the same filing
        # still reaches the answer it was given, because the answer to an identifier does not
        # change when the state behind it does.
        await caller.attest(acknowledgement)
        await _cross_a_boundary(caller, turnover_at, "stream/owed/1", env.client)
        assert await caller.stream.seal(filing) == acknowledgement


@pytest.mark.durable
async def test_a_finalization_that_ends_more_than_it_names_carries_the_whole_list(
    env, turnover_at
) -> None:
    """The cascade is part of the receipt, and a rebuilt receipt would not have it.

    An ending reaches further than the attempt named in it. A task gated on another sealing can
    never be reached once that other one has finally failed, so it is floored in the same
    transition, and the receipt says which ones went with it. That list is read at the moment the
    ending happened: rebuilt from a later projection it would name whatever had ended since, and
    the capacity beside it would be whatever capacity was in use by then rather than then.

    So this ends one attempt while a second is being worked and a third is gated behind the
    first, checks that the receipt reports a nonempty cascade and the capacity of that moment,
    and then asks for it again across a boundary.
    """
    turnover_at(10_000)
    plan = ReleasePlan(
        RELEASE_AT_SEAL,
        PAYLOAD_FIRST,
        BY_POSITION,
        gates=[EligibilityGate(attempt(2), after_sealed_attempt_id=attempt(1))],
    )
    items = make_start(tasks=3).tasks
    start = registering_the_receipt(
        replace(
            make_start(tasks=3),
            capacity=2,
            release=plan,
            assignments=assignments_for(items, plan),
        )
    )
    async with stream_worker(env.client):
        caller = await open_stream(env.client, start, workflow_id="stream/cascade/1")
        await caller.stream.close_queue()
        first = await caller.take()
        second = await caller.take()
        assert {first.attempt_id, second.attempt_id} == {attempt(0), attempt(1)}
        assert (await caller.stream.stream_state()).capacity_in_use == 2

        ending = FinalizeRequest(
            request_id=caller.next_id(), attempt_id=attempt(1), reason=STEP_CAP
        )
        receipt = await caller.stream.finalize(ending)
        # The gated task can never be reached now, so it is floored with this ending and named.
        assert receipt.also_finalized == [attempt(2)]
        # And the capacity is the capacity of that moment: the other attempt is still being
        # worked, so one of the two is still in use.
        assert receipt.capacity_in_use == 1

        await _cross_a_boundary(caller, turnover_at, "stream/cascade/1", env.client)
        # The other attempt is worked to its end, which moves both the capacity and the set of
        # attempts that have finally failed. The receipt does not move with them.
        await caller.work(first)
        assert (await caller.stream.stream_state()).capacity_in_use == 0
        assert await caller.stream.finalize(ending) == receipt


async def _the_code_a_tool_was_refused_with(call: Any) -> str:
    """The protocol code a gateway answered one tool call with, as the agent would read it.

    A refusal reaches an agent on the error channel, and its whole text is the canonical record.
    So the code is read back out of that text rather than out of anything the transport keeps.
    """
    try:
        await asyncio.wait_for(call, timeout=30)
    except Exception as error:  # noqa: BLE001 - the refusal is what is being read
        return json.loads(str(error))["code"]
    raise AssertionError("the tool call was answered rather than refused")


async def _a_gateway_over(
    caller: Caller, start: StreamStart, *, seed: Any = None
) -> Any:
    """One real gateway speaking for one generation, for the tests about its recovery paths.

    ``seed`` is the world the generation was opened on, which the first task presented claims.
    A test that never presents a task needs none, and passes none.
    """
    from shogym.serve.protocol_v2.gateway import StreamGateway
    from shogym.task import TaskSpec, ToolManifest

    terminal = ToolManifest(
        name="submit",
        description="",
        input_schema={"type": "object"},
        terminal_kind="score",
    )
    return StreamGateway(
        caller.stream,
        seed,
        TaskSpec(env_name="probe", instructions="", tools=[terminal]),
        terminal,
        initial_cursor=caller.cursor,
        generation=start,
    )


@pytest.mark.durable
async def test_a_generation_gives_up_on_a_boundary_its_own_gate_can_no_longer_reach(
    env, turnover_at, tmp_path
) -> None:
    """The gate bounds attempts, and the work that clears a boundary is allowed to fail.

    A presentation whose references the store cannot produce is refused, and the transport's
    supported repair is to build a fresh attestation for the same message and send it again. Every
    one of those is an accepted Update, so ordinary repair spends the reserve. Once the reserve is
    spent nothing more is admitted, and the message the generation is holding can only be cleared
    by an Update that will now be refused: the boundary is holding traffic back for a quiet point
    that can never arrive.

    So the generation recognises that and does what it does when a carrier will not fit. It
    records why, releases the gate and goes back to serving the execution it is in. The residual
    is real and is the reason the record is kept: this generation may now reach the service cap.

    Both arms are driven with one ordinary gateway and no traffic beside it. The control is the
    same forty refusals with no boundary decided, which repairs and is served.
    """
    broken = {"store": False}

    @activity.defn(name=VERIFY_BLOBS)
    async def unreliable_store(payload: VerifyBlobsInput) -> BlobsVerified:
        references = list(payload.references)
        if broken["store"]:
            return BlobsVerified(verified=[], unverified=references)
        return BlobsVerified(verified=references, unverified=[])

    async def arm(workflow_id: str, *, gated: bool) -> Dict[str, Any]:
        broken["store"] = False
        turnover_at(10_000)
        start = replace(make_start(tasks=2), blob_root=str(tmp_path))
        caller = await open_stream(env.client, start, workflow_id=workflow_id)
        await caller.stream.close_queue()
        await caller.take()
        gateway = await _a_gateway_over(caller, start)
        before = await _accepted_updates(env.client, workflow_id)
        if gated:
            turnover_at(before + 1)

        broken["store"] = True
        refusals = 0
        for _ in range(kernel_workflow.ADMISSION_RESERVE):
            try:
                await asyncio.wait_for(gateway.pull({}), timeout=10)
            except Exception as error:  # noqa: BLE001 - the refusal is the point
                assert "invalid_message" in str(error), error
                refusals += 1

        # The record the gateway is holding is the one the correspondence classifies as unable
        # to cross, and it is holding it because the message it names is still pending.
        held = type(gateway._recovery).__name__
        broken["store"] = False
        repaired = await asyncio.wait_for(gateway.pull({}), timeout=30)
        state = await caller.stream.stream_state()
        turnover_at(10_000)
        return {
            "record while refused": held,
            "refusals": refusals,
            "repaired": json.loads(repaired)["kind"],
            "pending": state.pending_message_id,
            "gating": state.turnover_requested,
            "refused": state.turnover_refused,
            "turnovers": state.turnovers,
        }

    async with stream_worker(env.client, activities=_instead_of_verification(unreliable_store)):
        control = await arm("stream/repair/control", gated=False)
        gated = await arm("stream/repair/gated", gated=True)

    # The control repairs and is served, which is what the gateway's recovery path is for.
    assert control["record while refused"] == "_PresentationRefused"
    assert gated["record while refused"] == "_PresentationRefused"
    assert control["refusals"] == kernel_workflow.ADMISSION_RESERVE
    assert control["repaired"] == "wait"
    assert control["pending"] is None
    assert control["refused"] is None

    # And so is the arm that decided to turn over, because the generation gave the boundary up
    # rather than holding traffic back for one it could no longer reach.
    assert gated["refusals"] == kernel_workflow.ADMISSION_RESERVE
    assert gated["repaired"] == "wait"
    assert gated["pending"] is None
    assert gated["gating"] is False
    assert gated["turnovers"] == 0
    # And it says why, which is what a launcher reports the residual by.
    assert gated["refused"] == kernel_workflow.ADMISSION_EXHAUSTED


@pytest.mark.durable
async def test_confirmations_cannot_spend_the_claim_that_recovers_an_abandoned_grant(
    env, turnover_at
) -> None:
    """The one call on the whitelist that nothing can substitute for keeps its own allowance.

    A grant held by an owner that then goes away is ended by name, and only the owner that
    replaces it can end it, so the claim that installs a replacement is the only way out. Sharing
    one allowance with confirmations meant spending it on confirmations locked that claim out: the
    world stayed held, no new owner could take the generation, and the grant that blocked the
    boundary could never be released.

    So a claim has an allowance of its own. This spends the confirmations' allowance to its end,
    well past it, and then requires a valid recovery claim to be admitted anyway.
    """
    turnover_at(10_000)
    start = make_start(tasks=2)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, start, workflow_id="stream/recover/1")
        await caller.stream.close_queue()
        task = await caller.take()
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        await caller.stream.begin_environment_call(call)
        await _latch(caller, turnover_at, env.client, "stream/recover/1")

        spent = 0
        for _ in range(kernel_workflow.ADMISSION_REPEATS * 3):
            try:
                await _straight_at_the_generation(
                    caller,
                    StreamWorkflow.confirm_state,
                    caller.stream.writer,
                    update_id=caller.next_id(),
                )
                spent += 1
            except WorkflowUpdateFailedError as error:
                assert turnover_pending(error)
        assert spent <= kernel_workflow.ADMISSION_REPEATS

        # The owner that held the grant is gone. A replacement claims, and the claim is admitted.
        replacement = await _resume(
            env.client, start, workflow_id="stream/recover/1", claimant="the-replacement"
        )
        # And having claimed, it can end the grant by name, which is the whole point of admitting
        # the claim: nothing else could have released it.
        assert (await replacement.end_environment_call(call)).held is True
        for _ in range(400):
            if (await replacement.stream_state()).turnovers == 1:
                break
            await asyncio.sleep(0.02)
        else:
            raise AssertionError("the boundary never followed the end of the grant")


@pytest.mark.durable
@pytest.mark.parametrize(
    "holding",
    ("a held grant", "a pending message"),
    ids=("grant", "pending"),
)
async def test_claims_that_failed_do_not_close_the_recovery_they_were_making(
    env, turnover_at, holding
) -> None:
    """A claim can fail without taking the generation over, and failures are not free.

    A claim reads the store before it swaps the epoch, so a store that cannot produce what it is
    asked for refuses the claim. Nothing has been taken over, but the Update was accepted, and
    four of those spend the allowance claims have. The claim that would have succeeded once the
    store came back is then refused for the want of a slot, and it is the only call that could
    have helped: an owner that went away leaves what it was holding to its replacement, and only
    a claim installs one.

    Nothing else in the generation notices. The total reserve is barely touched, so a fallback
    that looked only at the total would see a generation with thirty six slots to spare, and the
    transport reads the thing being held as work in progress and waits on it for ever.

    So the fallback asks about the allowance that actually gates the work, and gives the boundary
    up when the last route to it closes. Both arms are driven: without a boundary decided the
    repaired claim installs the owner and the hold is cleared, and with one decided the same
    happens and the generation records which allowance ran out.
    """
    broken = {"store": False}

    @activity.defn(name=VERIFY_BLOBS)
    async def unreliable_store(payload: VerifyBlobsInput) -> BlobsVerified:
        references = list(payload.references)
        if broken["store"]:
            return BlobsVerified(verified=[], unverified=references)
        return BlobsVerified(verified=references, unverified=[])

    async def arm(workflow_id: str, *, gated: bool) -> Dict[str, Any]:
        broken["store"] = False
        turnover_at(10_000)
        start = replace(make_start(tasks=2), blob_root="/verified-by-the-stub")
        caller = await open_stream(env.client, start, workflow_id=workflow_id)
        await caller.stream.close_queue()
        call = None
        if holding == "a held grant":
            task = await caller.take()
            call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
            await caller.stream.begin_environment_call(call)
        else:
            await caller.pull()
        if gated:
            await _latch(caller, turnover_at, env.client, workflow_id)

        broken["store"] = True
        failures = 0
        for index in range(kernel_workflow.ADMISSION_CLAIMS):
            try:
                await asyncio.wait_for(
                    _resume(
                        env.client,
                        start,
                        workflow_id=workflow_id,
                        claimant=f"the-unlucky-{index}",
                    ),
                    timeout=20,
                )
            except WorkflowUpdateFailedError as error:
                assert protocol_error_code(error) == "invalid_message"
                failures += 1

        broken["store"] = False
        replacement = await asyncio.wait_for(
            _resume(env.client, start, workflow_id=workflow_id, claimant="the-lucky-one"),
            timeout=30,
        )
        state = await caller.stream.stream_state()
        turnover_at(10_000)
        return {
            "failures": failures,
            "epoch": replacement.writer.ownership_epoch,
            "gating": state.turnover_requested,
            "refused": state.turnover_refused,
            "handle": replacement,
            "call": call,
        }

    async with stream_worker(env.client, activities=_instead_of_verification(unreliable_store)):
        control = await arm("stream/claims/control", gated=False)
        gated = await arm("stream/claims/gated", gated=True)

        # Every claim before the repair failed without taking the generation over.
        assert control["failures"] == kernel_workflow.ADMISSION_CLAIMS
        assert gated["failures"] == kernel_workflow.ADMISSION_CLAIMS
        # And in both arms the repaired claim installed the owner it was always going to install.
        assert control["epoch"] == gated["epoch"] == 2
        # The control was never holding traffic back, so it has nothing to say about a boundary.
        assert control["gating"] is False
        assert control["refused"] is None
        # The arm that had decided to turn over gave the boundary up rather than holding traffic
        # back for one whose last route it had closed, and it says which allowance closed it.
        assert gated["gating"] is False
        assert gated["refused"] == kernel_workflow.RECOVERY_CLAIMS_EXHAUSTED

        # And in both arms the new owner can now clear what was being held.
        for arm_result in (control, gated):
            if holding == "a held grant":
                assert (await arm_result["handle"].end_environment_call(arm_result["call"])).held


@pytest.mark.durable
async def test_an_acknowledgement_already_given_survives_a_takeover_and_a_boundary(
    env, turnover_at
) -> None:
    """A receipt is a receipt, whoever is writing by the time its caller asks for it again.

    A presentation commits and the acknowledgement is lost on the way back. That is an ordinary
    thing to happen, and the transport's answer to it is ordinary too: it kept the message and
    the exact attestation it sent, and it sends that attestation again. Inside one execution the
    service answers from its own record of what that Update completed with, without the handler
    running and therefore without asking who is writing now.

    Put a takeover and a boundary between the commit and the retry and the ordering starts to
    matter. The service's record is gone with the execution, so the handler runs, and if it asked
    who was writing first it would find the epoch moved and refuse a commit that had already
    succeeded. That refusal is not even recoverable: the transport would go back to repair a
    presentation whose message is no longer pending, and there is nothing left for it to do.

    So the answer comes before the question. Both arms are driven through a real gateway, one
    with a takeover and a boundary between the loss and the retry and one with only the takeover,
    and the two hand back the same bytes.
    """

    async def arm(workflow_id: str, *, cross: bool) -> str:
        turnover_at(10_000)
        start = make_start(tasks=2)
        caller = await open_stream(env.client, start, workflow_id=workflow_id)
        await caller.stream.close_queue()
        await caller.take()
        gateway = await _a_gateway_over(caller, start)

        # The commit lands and its acknowledgement is lost on the way back, which leaves the
        # transport holding the message and the exact attestation it sent.
        answered = caller.stream.commit_presentation

        async def lose_the_answer(commit: Any) -> None:
            await answered(commit)
            raise RuntimeError("the acknowledgement was lost on the way back")

        caller.stream.commit_presentation = lose_the_answer
        with pytest.raises(RuntimeError, match="lost on the way back"):
            await gateway.pull({})
        caller.stream.commit_presentation = answered
        assert type(gateway._recovery).__name__ == "_PresentationUncertain"

        replacement = await _resume(
            env.client, start, workflow_id=workflow_id, claimant="the-replacement"
        )
        if cross:
            crossing = Caller(replacement, (await replacement.stream_state()).cursor)
            await _cross_a_boundary(crossing, turnover_at, workflow_id, env.client)
        # The original transport retries, under the owner it still believes it is.
        return await asyncio.wait_for(gateway.pull({}), timeout=30)

    async with stream_worker(env.client):
        without = await arm("stream/receipt/plain", cross=False)
        crossed = await arm("stream/receipt/crossed", cross=True)
    assert json.loads(without)["kind"] == "wait"
    assert crossed == without


@pytest.mark.durable
async def test_a_finalization_retried_after_a_takeover_and_a_boundary_gets_its_receipt(
    env, turnover_at
) -> None:
    """The controller's retry is promised the same Update, and a takeover does not change that.

    A controller that loses the answer to an ending sends the same request again, and the same
    logical request is one ending. Inside one execution the service hands the receipt back
    without the handler running. Across a takeover and a boundary the handler does run, and the
    receipt has to be reachable before the epoch it was given under is compared with the epoch
    now, or a controller would be told it was fenced by an owner it never raced.
    """
    turnover_at(10_000)
    start = make_start(tasks=3)

    async def arm(workflow_id: str, *, cross: bool) -> Any:
        turnover_at(10_000)
        caller = await open_stream(env.client, start, workflow_id=workflow_id)
        await caller.stream.close_queue()
        await caller.take()
        ending = FinalizeRequest(
            request_id=caller.next_id(), attempt_id=attempt(2), reason=STEP_CAP
        )
        receipt = await caller.stream.finalize(ending)
        replacement = await _resume(
            env.client, start, workflow_id=workflow_id, claimant="the-replacement"
        )
        if cross:
            crossing = Caller(replacement, (await replacement.stream_state()).cursor)
            await _cross_a_boundary(crossing, turnover_at, workflow_id, env.client)
        return receipt, await caller.stream.finalize(ending)

    async with stream_worker(env.client):
        given, again = await arm("stream/ending/plain", cross=False)
        crossed_given, crossed_again = await arm("stream/ending/crossed", cross=True)
    assert again == given
    assert crossed_again == crossed_given == given


@pytest.mark.durable
async def test_a_refused_begin_stays_refused_under_its_own_identifier_and_is_judged_under_another(
    env, turnover_at
) -> None:
    """The paired case for a cached refusal: the same identifier, and a fresh one, after a change.

    A begin is refused while the generation is holding a message it owes. That refusal is the
    answer that identifier got, and inside one execution the service would hand it back
    unchanged however the world moved afterwards. The world does move: the message is presented,
    and the condition the refusal rested on stops being true.

    So the two arms answer differently on purpose. The same identifier is answered with the
    refusal it was given, because that is what it was given and a boundary must not turn a
    refusal into a grant. A fresh identifier is a request the generation has never answered, and
    it is judged on the state it finds, which by then permits the grant. Both arms are taken
    after a boundary, which is where the two could otherwise be confused: the service's own
    record of the first answer is gone, and only what the generation carried can tell them apart.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/begin/2")
        await caller.stream.close_queue()
        task = await caller.take()
        owed = await caller.pull()
        assert owed.kind == "wait"

        # Refused, because the generation is holding a message it owes.
        call = EnvironmentCall(call_id=caller.next_id(), attempt_id=task.attempt_id or "")
        assert await refused(caller.stream.begin_environment_call(call)) == "outstanding_response"

        # The condition behind that refusal stops being true, and a boundary is crossed.
        await caller.attest(owed)
        await _cross_a_boundary(caller, turnover_at, "stream/begin/2", env.client)
        assert (await caller.stream.stream_state()).pending_message_id is None

        # The same identifier is answered with the refusal it was given.
        assert await refused(caller.stream.begin_environment_call(call)) == "outstanding_response"
        state = await caller.stream.stream_state()
        assert state.environment_call is None
        assert state.environment_calls.get(task.attempt_id or "", 0) == 0

        # A fresh identifier for the same call is a request the generation has never answered,
        # and it is judged on the state it finds, which now permits the grant.
        granted = await _straight_at_the_generation(
            caller,
            StreamWorkflow.begin_environment_call,
            call,
            caller.stream.writer,
            update_id=caller.next_id(),
        )
        assert granted.held is True
        assert granted.call_id == call.call_id
        after = await caller.stream.stream_state()
        assert after.environment_call == call.call_id
        assert after.environment_calls[task.attempt_id or ""] == 1


@pytest.mark.durable
async def test_a_lost_offer_reaches_the_same_ending_whether_or_not_a_boundary_intervened(
    env, turnover_at
) -> None:
    """A superseded transport's retry, and everything after it, is the same across a boundary.

    This is the schedule a review reproduced against the complete journal, kept here as a test.
    One transport's successful pull is lost on the way back, so it holds the request as uncertain
    and will send that exact Update again. A replacement claims the generation, takes the message
    that transport was owed and presents it, and then a boundary happens. The original transport
    retries.

    What it gets is a refusal, because it is fenced, and the refusal is not the interesting part.
    What follows it is: the retry reaching the answer its Update actually got leaves the transport
    holding a refused presentation, and the next tool call it makes is answered on that, so the
    agent behind it sees an outstanding response rather than a second fencing. An arm where the
    answer came from the service's own record instead of the carried journal has to end in the
    same place, so both are driven and compared whole.
    """

    async def arm(workflow_id: str, *, cross: bool) -> Dict[str, Any]:
        turnover_at(10_000)
        start = make_start(tasks=2)
        caller = await open_stream(env.client, start, workflow_id=workflow_id)
        await caller.stream.close_queue()
        await caller.take()
        gateway = await _a_gateway_over(caller, start)

        # The pull lands and its answer is lost on the way back, which leaves the transport
        # holding an uncertain request under an identifier it will send again.
        answered = caller.stream.pull

        async def lose_the_answer(request: PullRequest) -> OfferedMessage:
            await answered(request)
            raise RuntimeError("the offer was lost on the way back")

        caller.stream.pull = lose_the_answer
        with pytest.raises(RuntimeError, match="lost on the way back"):
            await gateway.pull({})
        caller.stream.pull = answered
        held = type(gateway._recovery).__name__

        # A replacement takes the generation over, takes the message the first transport was owed
        # and presents it, and then the generation crosses a boundary or does not.
        replacement = await _resume(
            env.client, start, workflow_id=workflow_id, claimant="the-replacement"
        )
        taking = Caller(replacement, (await replacement.stream_state()).cursor)
        # The replacement inherits the request the first transport left open, which is the one
        # call that can reach the message the generation is holding: a reservation belongs to the
        # logical request that asked for it, and every other request is refused while it stands.
        owed = await taking.pull(gateway._recovery.request)
        await taking.attest(owed)
        if cross:
            await _cross_a_boundary(taking, turnover_at, workflow_id, env.client)

        retry = await _the_code_a_tool_was_refused_with(gateway.pull({}))
        after = type(gateway._recovery).__name__
        following = await _the_code_a_tool_was_refused_with(gateway.info({}))
        turnover_at(10_000)
        return {
            "the message that was lost": owed.kind,
            "held while uncertain": held,
            "the retry": retry,
            "held after the retry": after,
            "the call after that": following,
            "boundaries": (await caller.stream.stream_state()).turnovers,
        }

    async with stream_worker(env.client):
        without = await arm("stream/seed/wait/plain", cross=False)
        crossed = await arm("stream/seed/wait/crossed", cross=True)

    assert without["boundaries"] == 0 and crossed["boundaries"] == 1
    assert without == {**crossed, "boundaries": 0}
    # And what they agree on is the whole schedule, named rather than left to the comparison.
    assert crossed["the message that was lost"] == "wait"
    assert crossed["held while uncertain"] == "_RequestUncertain"
    assert crossed["the retry"] == "fenced_writer"
    assert crossed["held after the retry"] == "_PresentationRefused"
    assert crossed["the call after that"] == "outstanding_response"


@pytest.mark.durable
async def test_a_superseded_transports_retry_leaves_the_replacements_world_where_it_is(
    env, turnover_at
) -> None:
    """The other schedule a review reproduced: a fenced transport reaching a live world's route.

    A transport loses the answer to a successful Task pull, so it will send that exact Update
    again. A replacement claims the generation, takes the task that transport was owed, presents
    it and records the world it opened for that attempt in the route the environment resolves a
    seal against. Then a boundary happens, and the first transport retries.

    Its retry reaches the offer its Update was answered with, which is what the journal is for,
    and the work it does with that offer is where the danger was: it prepares a world for the
    attempt before the presentation that will fence it, and writing that world into a shared
    route would leave the replacement's seal reaching a world nobody worked in. The route now
    compares owners, so the retry is refused at the stream and leaves the pairing alone.

    Both arms are driven for the same reason as the others: the outcome the retry reaches must
    not depend on whether the answer came from the service's record or from what was carried.
    """

    async def arm(workflow_id: str, *, cross: bool) -> Dict[str, Any]:
        turnover_at(10_000)
        start = make_start(tasks=2)
        caller = await open_stream(env.client, start, workflow_id=workflow_id)
        await caller.stream.close_queue()
        gateway = await _a_gateway_over(
            caller, start, seed=SimpleNamespace(env="probe", session_id="old-world")
        )
        route = gateway._route

        answered = caller.stream.pull

        async def lose_the_answer(request: PullRequest) -> OfferedMessage:
            await answered(request)
            raise RuntimeError("the offer was lost on the way back")

        caller.stream.pull = lose_the_answer
        with pytest.raises(RuntimeError, match="lost on the way back"):
            await gateway.pull({})
        caller.stream.pull = answered

        # The replacement takes the generation over, inherits the request left open, presents the
        # task it was owed and records the world it opened for that attempt.
        replacement = await _resume(
            env.client, start, workflow_id=workflow_id, claimant="the-replacement"
        )
        taking = Caller(replacement, (await replacement.stream_state()).cursor)
        task = await taking.pull(gateway._recovery.request)
        await taking.attest(task)
        live = SimpleNamespace(env="probe", session_id="live-world")
        route.record(task.attempt_id or "", live, replacement.writer.ownership_epoch)
        if cross:
            await _cross_a_boundary(taking, turnover_at, workflow_id, env.client)

        retry = await _the_code_a_tool_was_refused_with(gateway.pull({}))
        turnover_at(10_000)
        return {
            "the task that was lost": task.attempt_id,
            "the retry": retry,
            "where the world is": route(task.attempt_id or ""),
            "boundaries": (await caller.stream.stream_state()).turnovers,
        }

    async with stream_worker(env.client):
        without = await arm("stream/seed/route/plain", cross=False)
        crossed = await arm("stream/seed/route/crossed", cross=True)

    assert without["boundaries"] == 0 and crossed["boundaries"] == 1
    assert without == {**crossed, "boundaries": 0}
    assert crossed["the retry"] == "fenced_writer"
    assert crossed["where the world is"] == ("probe", "live-world")


# The eleven Updates a generation answers. Naming them here rather than reading them off the
# workflow is deliberate: a handler added later has to be given a cell in the matrix below by
# hand, and the check that the two lists agree is what makes leaving one out a failure.
_EVERY_HANDLER = (
    "claim_ownership",
    "claim_consumer",
    "pull",
    "info",
    "seal_attempt",
    "commit_presentation",
    "finalize_attempt",
    "close_queue",
    "begin_environment_call",
    "end_environment_call",
    "confirm_state",
)

# What a generation reports about itself rather than about what it served. A boundary moves every
# one of these by definition: it is a count of boundaries, of the work a boundary does, or of what
# is in flight at the moment of the reading. None of them is inside any hash, none is part of an
# answer a caller is owed, and a comparison that included them would be asserting that a boundary
# is invisible to the code whose job is to report it.
_OPERATIONAL = (
    "turnovers",
    "turnover_requested",
    "turnover_refused",
    "turnover_refused_bytes",
    "verifying",
    "verification_batches",
    "unfinished_handlers",
)


def _without_the_secret_this_arm_minted(rows: Any, token: str) -> Any:
    """Replace one arm's own fencing token hash wherever it appears in what was answered.

    A process mints its own token when it claims, so two runs of one schedule differ in it and in
    nothing else it stands for. Comparing the two arms on it would be comparing two secrets rather
    than two behaviours, and the fact it is there for, that the token survives a boundary and the
    owner that holds it keeps writing, is checked inside each arm where the secret is one secret.
    """
    digest = sha256(token.encode("utf-8")).hexdigest()
    return json.loads(json.dumps(rows, default=str).replace(digest, "the token this arm minted"))


def _outcome_of(value: Any) -> Any:
    """One answer as a comparable value, with what a generation says about itself left out."""
    if not is_dataclass(value):
        return value
    return {
        field.name: getattr(value, field.name)
        for field in fields(value)
        if field.name not in _OPERATIONAL
    }


async def _answered_with(call: Any) -> Any:
    """What one Update was answered with: the value, or the protocol code it was refused with."""
    try:
        return ("answered", _outcome_of(await asyncio.wait_for(call, timeout=30)))
    except WorkflowUpdateFailedError as error:
        code = protocol_error_code(error)
        return ("refused", code if code is not None else str(error))




async def _sent_once(caller: Caller, row: Dict[str, Any], update_id: str) -> Any:
    """Send one cell's exact request under ``update_id`` and report how it was answered."""
    return await _answered_with(
        _straight_at_the_generation(caller, row["update"], *row["args"], update_id=update_id)
    )


async def _one_of_every_handler(
    caller: Caller, start: StreamStart, *, refused: bool
) -> List[Dict[str, Any]]:
    """Drive one completed outcome out of each of the eleven handlers, and keep every request.

    The order is the order a generation is really driven in, because most of these are only
    answerable from somewhere: a presentation needs a message pending, a release needs a grant
    held, a seal needs a live attempt. Each cell is sent straight at the generation so that the
    identifier is this test's to repeat, and the request and the identifier are both kept, which
    is what lets the same call be sent again later as itself.

    The ownership cell comes first, and everything after it speaks for the owner it installed.
    Taken last it would leave every other request naming an owner the generation had already
    replaced, and the arm that takes nobody over would be testing ten fenced calls under a label
    that says nothing was taken over. The writer each cell was built with is kept beside it, so
    the arm can require what it says it is: the same owner throughout where nothing took the
    generation over, and an older one where something did.

    The arm that wants refusals sends the same eleven under a writer the generation never
    installed. That is the one refusal every handler here has in common, it is raised inside the
    handler rather than by a validator, and so it is a completed outcome exactly as a success is.
    """
    rows: List[Dict[str, Any]] = []

    async def cell(name: str, update: Any, *args: Any) -> Any:
        row = {"handler": name, "update": update, "args": list(args), "id": caller.next_id()}
        row["epoch"] = next(
            (item.ownership_epoch for item in args if isinstance(item, Writer)), None
        )
        row["first"] = await _sent_once(caller, row, row["id"])
        rows.append(row)
        return row["first"]

    taken = await cell(
        "claim_ownership",
        StreamWorkflow.claim_ownership,
        OwnershipClaim(
            claimant_id="the-matrix",
            previous_epoch=caller.stream.writer.ownership_epoch,
            fencing_token="c" * 64,
            configuration_hash=("f" * 64) if refused else configuration_hash(start),
            reason="resume",
        ),
    )
    if taken[0] == "answered":
        # The claim installed a new owner, and this caller is it: the call went straight at the
        # generation, so the handle it went through has to be told what it now holds.
        caller.stream._writer = Writer(
            ownership_epoch=taken[1]["ownership_epoch"], fencing_token="c" * 64
        )

    writer = caller.stream.writer
    if refused:
        writer = replace(writer, fencing_token="f" * 64)
    task = await caller.take()
    attempt_id = task.attempt_id or ""

    call = EnvironmentCall(call_id=caller.next_id(), attempt_id=attempt_id)
    await cell("begin_environment_call", StreamWorkflow.begin_environment_call, call, writer)
    await cell("end_environment_call", StreamWorkflow.end_environment_call, call, writer)
    await cell("confirm_state", StreamWorkflow.confirm_state, writer)
    await cell("close_queue", StreamWorkflow.close_queue, writer)
    pulled = await cell(
        "pull",
        StreamWorkflow.pull,
        PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor),
        writer,
    )

    # The message the pull answered with is what the presentation cell presents, so the two cells
    # are one protocol step rather than two unrelated ones. An arm whose pull was refused has no
    # message to present and attests the task it already presented instead: what that cell is for
    # here is the handler, and the handler refuses it before it reads any of it.
    offered = pulled[1] if pulled[0] == "answered" else None
    state = await caller.stream.stream_state()
    await cell(
        "commit_presentation",
        StreamWorkflow.commit_presentation,
        PresentationCommit(
            attestation_id=caller.next_id(),
            cursor_before=state.cursor,
            message_id=offered["message_id"] if offered else task.message_id,
            visible_bytes_sha256=sha256(
                (offered["visible_text"] if offered else task.visible_text).encode("utf-8")
            ).hexdigest(),
            transcript_blob=BLOB,
            provider_turn_blob=None,
            task_start_checkpoint_blob=BLOB if offered and offered["kind"] == "task" else None,
            completed_turn=False,
            stream_state_before_sha256=state.stream_state_sha256,
        ),
        writer,
    )
    if offered is not None:
        caller.cursor = (await caller.stream.stream_state()).cursor

    asked = await cell(
        "info",
        StreamWorkflow.info,
        InfoRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor),
        writer,
    )
    if asked[0] == "answered":
        await caller.attest(OfferedMessage(**asked[1]))
    sealed = await cell(
        "seal_attempt",
        StreamWorkflow.seal_attempt,
        SealRequest(
            metadata=TerminalMetadata(
                request_id=caller.next_id(),
                last_presented_cursor=caller.cursor,
                attempt_id=attempt_id,
            ),
            public_tool_name="submit",
            native_terminal_name="submit",
        ),
        writer,
    )
    if sealed[0] == "answered":
        await caller.attest(OfferedMessage(**sealed[1]))
    await cell(
        "finalize_attempt",
        StreamWorkflow.finalize_attempt,
        FinalizeRequest(request_id=caller.next_id(), attempt_id=attempt(2), reason=STEP_CAP),
        writer,
    )
    await cell("claim_consumer", StreamWorkflow.claim_consumer, CONSUMER, writer)
    assert {row["handler"] for row in rows} == set(_EVERY_HANDLER)
    return rows


@pytest.mark.durable
@pytest.mark.parametrize("outcome", ("success", "refusal"), ids=("answered", "refused"))
@pytest.mark.parametrize("takeover", (False, True), ids=("one owner", "taken over"))
async def test_update_outcomes_survive_every_boundary_schedule(
    env, turnover_at, outcome, takeover
) -> None:
    """Every handler, every identifier, every schedule: a boundary changes none of them.

    This is the systematic version of the individual schedules beside it. Each of the eleven
    Updates a generation answers is driven to a completed outcome, once as a success and once as
    a refusal; then the generation is taken over or is not; then it crosses a boundary or does
    not; then every one of those exact identifiers is sent again, and every one of them is sent
    again under a fresh identifier as well.

    The whole table is compared between the arm that crossed a boundary and the arm that did
    not. That is the claim in one assertion: the exact identifiers are answered with what they
    were answered with the first time, the fresh ones are answered on the state the generation is
    in, and which side of a boundary the caller is on decides neither.

    What is left out of the comparison is what a generation says about itself, listed above: the
    count of boundaries and the work of crossing one. Those differ by construction, and a
    comparison that required them not to would be requiring the boundary not to have happened.

    The ownership axis is required to mean what it says. Every request is built under the owner
    the ownership cell installed, so in the arm that takes nobody over the saved requests name the
    owner the generation still has, and the fresh identifiers among them are judged on the state
    rather than turned away as a fenced writer's. In the arm that takes the generation over they
    name an older owner, which is the other half of the axis. Both are asserted rather than
    assumed, because an arm whose requests were all fenced would pass this test while testing one
    thing twice.
    """

    async def arm(workflow_id: str, *, cross: bool) -> List[Dict[str, Any]]:
        turnover_at(10_000)
        start = replace(make_start(tasks=3), info=True)
        caller = await open_stream(env.client, start, workflow_id=workflow_id)
        minted = caller.stream.writer.fencing_token
        rows = await _one_of_every_handler(caller, start, refused=outcome == "refusal")

        driving = caller
        if takeover:
            replacement = await _resume(
                env.client, start, workflow_id=workflow_id, claimant="the-replacement"
            )
            driving = Caller(replacement, (await replacement.stream_state()).cursor)
            driving._counter = 0x400
        if cross:
            await _cross_a_boundary(driving, turnover_at, workflow_id, env.client)

        # The ownership axis, checked rather than assumed. Ten of the eleven requests name a
        # writer, and which owner that is decides what a fresh identifier for one of them means.
        owner = (await caller.stream.stream_state()).ownership_epoch
        named = [row["epoch"] for row in rows if row["epoch"] is not None]
        assert len(named) == 10
        if takeover:
            assert all(epoch < owner for epoch in named), (named, owner)
        else:
            assert all(epoch == owner for epoch in named), (named, owner)

        for row in rows:
            row["exact"] = await _sent_once(caller, row, row["id"])
            row["fresh"] = await _sent_once(caller, row, caller.next_id())
        turnover_at(10_000)
        return _without_the_secret_this_arm_minted(
            [
                {
                    "handler": row["handler"],
                    "first": row["first"],
                    "exact": row["exact"],
                    "fresh": row["fresh"],
                }
                for row in rows
            ],
            minted,
        )

    async with stream_worker(env.client):
        without = await arm("stream/matrix/plain", cross=False)
        crossed = await arm("stream/matrix/crossed", cross=True)

    assert len(crossed) == len(_EVERY_HANDLER) == 11
    assert {row["first"][0] for row in crossed} == {
        "answered" if outcome == "success" else "refused"
    }
    # And in the arm nobody took over, a fresh identifier under the standing owner is answered on
    # the state rather than fenced, which is the half of the axis that was being claimed and not
    # driven. Which of them succeed is the state's business; that none of them is fenced is this.
    # The ownership claim is not one of the ten: it carries the epoch it read rather than a
    # writer, so sending the same claim again is a compare and swap against an epoch that has
    # moved, and being fenced is the right answer to it.
    if outcome == "success" and not takeover:
        fenced = [
            row["handler"]
            for row in crossed
            if row["handler"] != "claim_ownership" and row["fresh"][1] == "fenced_writer"
        ]
        assert fenced == [], fenced
    # Every exact identifier was answered with what it was answered with the first time, on both
    # sides of the boundary. This is the journal's whole promise, said once for all eleven.
    for row in crossed:
        assert row["exact"] == row["first"], row["handler"]
    for row in without:
        assert row["exact"] == row["first"], row["handler"]
    # And the two arms agree cell for cell, which is the difference the boundary did not make.
    assert crossed == without


def test_a_recipe_that_rebuilds_other_bytes_is_refused_rather_than_served(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The promise a recipe makes is kept at the restore, or the carrier is refused there.

    A recipe stands in for bytes a caller may still be owed, and it is written only where the
    generation proved it reproduces them. What that proof cannot cover is a carrier written by
    other code: a recipe read back has to be rebuilt and compared against the digest the offer it
    replaces committed to, and a generation that cannot make those exact bytes again must refuse
    the whole carrier rather than come up serving something else under the same identifier.
    """
    # The rebuild reads the converter the way every other value does, and outside a Worker there
    # is no workflow to read it from. The one the service is configured with is the same object.
    monkeypatch.setattr(
        kernel_workflow.workflow, "payload_converter", lambda: default_converter().payload_converter
    )
    start = make_start(tasks=2)
    generation = StreamWorkflow.__new__(StreamWorkflow)
    generation._start = start
    generation._items = {item.attempt_id: item for item in start.tasks}

    made = generation._from_recipe(f"task {attempt(0)} " + "0" * 64)
    assert made is not None and made["kind"] == "task"
    honest = f"task {attempt(0)} " + sha256(canonical_json(made)).hexdigest()
    assert generation._rebuilt("pull-1", kernel_workflow._Answer("pull", 1, "value", recipe=honest))

    for tampered in (
        f"task {attempt(0)} " + "0" * 64,
        f"task {attempt(1)} " + honest.split(" ")[-1],
        f"task {'f' * 32} " + honest.split(" ")[-1],
        "payload " + attempt(0) + " " + honest.split(" ")[-1],
    ):
        with pytest.raises(ApplicationError, match="rebuild"):
            generation._rebuilt("pull-1", kernel_workflow._Answer("pull", 1, "value", recipe=tampered))


@pytest.mark.durable
async def test_an_offered_task_crosses_as_a_recipe_and_comes_back_as_the_same_bytes(
    env, turnover_at
) -> None:
    """The offers a long generation answered cross as instructions for making them again.

    A task offer is the one answer built entirely out of things the start already carries, and
    the start rides beside the projection. Writing the bytes down a second time is what made the
    journal the largest thing in the carrier, so what crosses is the attempt the offer was for
    and the digest of what was answered, and the bytes are made again on the other side.

    This reads the carrier the successor was actually handed, requires the row to be a recipe
    rather than a body, and then sends the same pull again to require that what comes back is the
    same message down to the byte.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/recipe/1")
        await caller.stream.close_queue()
        request = PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor)
        update_id = caller.next_id()
        offered = await _straight_at_the_generation(
            caller, StreamWorkflow.pull, request, caller.stream.writer, update_id=update_id
        )
        assert offered.kind == "task"
        await caller.attest(offered)
        await _cross_a_boundary(caller, turnover_at, "stream/recipe/1", env.client)

        chain = await run_ids(env.client, "stream/recipe/1")
        carried = await _carried(env.client, "stream/recipe/1", chain[-1])
        rows = [row for row in carried["journal"] if row[0] == update_id]
        assert len(rows) == 1, carried["journal"]
        # The row carries a recipe and no body, which is the size claim in one assertion.
        assert rows[0][9].startswith(f"task {offered.attempt_id} ")
        assert rows[0][8] == ""
        assert offered.visible_text not in json.dumps(carried)

        # And the caller that was owed those bytes is given those bytes.
        again = await _straight_at_the_generation(
            caller, StreamWorkflow.pull, request, caller.stream.writer, update_id=update_id
        )
        assert again == offered


def test_a_carrier_this_code_cannot_read_is_refused_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reading a carrier is one thing, and it either works or the execution ends saying so.

    Three ways a carrier can be unreadable, and none of them may reach the generation half
    applied or fail the Workflow Task for ever on bytes that are never going to decode: a writing
    this code does not know, bytes that do not decode under a writing it does, and a value that
    decodes but whose shape is wrong, which is only found while it is being written over the
    state. The last one is why the refusal covers the application and not only the decoding.

    A refusal raised inside the reading is already the answer and passes through as itself, which
    is what keeps a carrier composed against another generation saying that rather than saying it
    could not be read.
    """
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "payload_converter",
        lambda: default_converter().payload_converter,
    )
    start = make_start(tasks=2)
    generation = StreamWorkflow.__new__(StreamWorkflow)
    generation._start = start
    generation._continued = True
    generation._configuration_hash = configuration_hash(start)
    carrier = _a_carrier(start)

    with pytest.raises(ApplicationError, match="could not be read"):
        generation._restore(replace(carrier, encoding="something-else.v9"))
    with pytest.raises(ApplicationError, match="could not be read"):
        generation._restore(replace(carrier, data="not a packed projection"))

    # A shape the decoding accepts and the application cannot use. What raises it there is an
    # ordinary Python error rather than a refusal, and it is refused all the same.
    def cannot_be_written(projection: Any) -> None:
        raise IndexError("list index out of range")

    monkeypatch.setattr(generation, "_apply", cannot_be_written)
    with pytest.raises(ApplicationError, match="could not be read"):
        generation._restore(carrier)

    # And the refusals the reading raises itself are still their own answers.
    generation._configuration_hash = "f" * 64
    with pytest.raises(ApplicationError, match="another generation"):
        generation._restore(carrier)
    generation._continued = False
    with pytest.raises(ApplicationError, match="started fresh"):
        generation._restore(carrier)
    with pytest.raises(ApplicationError, match="version"):
        generation._continued = True
        generation._restore(
            replace(carrier, carrier_schema_version=max(CARRIER_SCHEMA_VERSIONS) + 1)
        )
