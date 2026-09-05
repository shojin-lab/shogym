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
``network`` for the download it does on first use. The few that are about the service's cap
itself need the local dev service, because the time-skipping server does not enforce the cap and
takes no dynamic configuration to lower it; those are marked ``dev_server`` and skip unless
``SHOGYM_DEV_SERVER_TESTS`` is set.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from dataclasses import fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio.client import (  # noqa: E402
    Client,
    WorkflowUpdateFailedError,
)
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    IMMEDIATE,
    InfoRequest,
    PullRequest,
    TerminalMetadata,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    CARRIER_SCHEMA_VERSION,
    STEP_CAP,
    STREAM_TASK_QUEUE,
    BlobsVerified,
    ConsumerClaim,
    EnvironmentCall,
    FinalizeRequest,
    OfferedMessage,
    OwnershipClaim,
    SealRequest,
    StreamCarry,
    StreamHandle,
    StreamStart,
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
)
from shogym.serve.protocol_v2.kernel.runtime import temporal_home  # noqa: E402
from shogym.serve.protocol_v2.reader import _refuse_a_moved_history  # noqa: E402
from tests._fixtures.policy_rows import registering_the_receipt  # noqa: E402

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


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await WorkflowEnvironment.start_time_skipping()
    except Exception as error:  # noqa: BLE001 - an absent test server is a skip, not a failure
        pytest.skip(f"the Temporal test server is unavailable: {error}")
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
    carrier = StreamCarry(
        carrier_schema_version=CARRIER_SCHEMA_VERSION,
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
    return replace(carrier, **overrides) if overrides else carrier


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
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x2000 + self._counter)

    async def pull(self, request: Optional[PullRequest] = None) -> OfferedMessage:
        return await self.stream.pull(
            request
            or PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )

    async def present(self, message: OfferedMessage) -> None:
        ack = await self.stream.present(
            message,
            attestation_id=self.next_id(),
            transcript_blob=BLOB,
            provider_turn_blob=BLOB if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=BLOB if message.kind == "task" else None,
        )
        self.cursor = ack.cursor

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
    """Take a running generation over, the way a replacement process does."""
    return await resume_stream(
        client,
        workflow_id=workflow_id,
        configuration_hash=configuration_hash(start),
        claimant_id=claimant,
    )


def _encoded(start: StreamStart) -> int:
    """How many bytes the configured converter turns one start into.

    The same converter the service is given, so this is the number the service would measure
    against its own limit rather than an estimate of it.
    """
    from temporalio.converter import default as default_converter

    return default_converter().payload_converter.to_payloads([start])[0].ByteSize()


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


def test_the_ceiling_leaves_room_under_the_service_and_over_the_supported_profile() -> None:
    """The size ceiling has two margins to keep, and they pull in opposite directions.

    Above it is the service's own limit on one payload: a ceiling at or over that would let the
    generation submit a continuation the service refuses, which is a fault where there was a
    plan. Below it is the profile the package supports: a ceiling under that would make the
    generation refuse its own boundary and wedge at the cap, which is the failure this whole
    change exists to remove. The second is the one to watch as the profile grows.
    """
    assert (
        kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
        < kernel_workflow.SERVICE_PAYLOAD_LIMIT_BYTES
    )
    # The measured two hundred task AutomationBench roster, driven to its last quiet point.
    measured = 1_351_436
    assert measured < kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
    headroom = 1 - measured / kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
    assert headroom > 0.25, "the supported profile has to sit well under the ceiling, not near it"


def test_the_progress_whitelist_fits_many_times_over_inside_the_reserve() -> None:
    """What the gate admits after the latch, counted, against what it will admit at most.

    Each of these is the work that can make the boundary quiet, and each is bounded per latch:
    one presentation commit for the one message that can be pending, one end for the one grant
    that can be held, one retry and one presentation for the one seal that can be prepared, one
    claim and one confirmation to recover a grant whose owner went away, and one close of the
    queue. The reserve is a ceiling on the sum of them and on every refusal in between.
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
    import typing

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


def test_what_the_transport_can_send_again_is_what_the_generation_kept_an_answer_for() -> None:
    """The two lists, side by side: what a lost answer is retried as, and what answers it.

    Which Update identifiers can arrive twice across a boundary is read off the transport rather
    than assumed. It keeps one recovery record at a time, and each state of it names exactly what
    that state sends again. Against that goes what the generation can answer such a retry with: a
    request table for the requests that bound a message, and the exact-outcome cache for the
    handlers that bind nothing.

    Three of the transport's states cannot be at a legal boundary at all. A message offered and
    not yet attested to is a pending message, and a pending message is what a boundary waits for;
    a filing this transport owes on an attempt's behalf is a prepared seal, and so is that. What
    is left is the list below, and every entry of it has an answer.
    """
    resent_across_a_boundary = {
        "_RequestUncertain": "the pull, info or seal request whose answer was lost",
        "_PresentationUncertain": "the presentation commit whose acknowledgement was lost",
        "_LeaseHeld": "the end of a grant this transport is holding",
        "_ResultOwed": "the confirmation before a kept result is handed over",
        "the resumer": "the ownership claim and the consumer claim a takeover makes",
    }
    answered_by_a_request_table = {
        "the pull, info or seal request whose answer was lost": (
            "_pull_requests, _info_requests, _terminal_requests"
        ),
        "the presentation commit whose acknowledgement was lost": (
            "_attestations and _attestation_identities"
        ),
    }
    answered_by_the_exact_outcome_cache = {
        "the end of a grant this transport is holding": "_last_end",
        "the ownership claim and the consumer claim a takeover makes": (
            "_last_ownership and _last_consumer"
        ),
    }
    # A confirmation is deliberately a fresh identifier every time, so there is nothing to
    # answer twice: the transport asks whether the stream still admits it, and an answer from
    # the last time it asked would be the wrong answer.
    answered_by_asking_again = {"the confirmation before a kept result is handed over"}
    covered = (
        set(answered_by_a_request_table)
        | set(answered_by_the_exact_outcome_cache)
        | answered_by_asking_again
    )
    assert set(resent_across_a_boundary.values()) == covered
    # And two slots the transport does not resend into but the generation answers anyway, because
    # a begin accepted twice would grant a world twice and a filing whose work failed for good
    # left no row behind it to be answered from. Every slot named above is one the generation
    # actually keeps, which is what makes the two lists comparable rather than a pair of stories.
    kept = set(StreamWorkflow.__init__.__code__.co_names)
    assert {
        "_last_begin",
        "_last_end",
        "_last_ownership",
        "_last_consumer",
        "_failed_seals",
    } <= kept


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
    Four may differ and none of them is about what the generation served: the chain position, the
    latch, and the size that refused a boundary all exist to say what happened to the generation
    rather than what it served, and the fencing token is minted fresh by whichever process
    claimed it, so two runs never share one. That the token crosses a boundary unchanged is the
    subject of its own test, where the comparison is one generation against itself.
    """
    operational = (
        "turnovers",
        "turnover_requested",
        "turnover_refused_bytes",
        "fencing_token_hash",
    )
    return {
        row.name: getattr(state, row.name) for row in fields(state) if row.name not in operational
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


@pytest.mark.network
async def test_a_generation_that_crossed_boundaries_answers_as_the_one_that_did_not(
    env, turnover_at
) -> None:
    """The twin. Same roster, same steps, one turned over several times and one never did.

    This is the whole promise in one test: every message, every identifier, every byte, every
    state field, every attempt record and every presentation is compared between the two, and
    the only permitted difference is the count of boundaries crossed.
    """
    async with stream_worker(env.client):
        plain = await _drive(env.client, make_start(tasks=4), "stream/twin/plain")
    assert plain["turnovers"] == 0

    turnover_at(6)
    async with stream_worker(env.client):
        crossed = await _drive(env.client, make_start(tasks=4), "stream/twin/crossed")
    assert crossed["turnovers"] >= 3, "the lowered trigger has to produce several boundaries"

    assert crossed["kinds"] == plain["kinds"]
    assert crossed["message_ids"] == plain["message_ids"]
    assert crossed["visible"] == plain["visible"]
    assert crossed["state"] == plain["state"]
    assert crossed["attempts"] == plain["attempts"]
    assert crossed["presentations"] == plain["presentations"]


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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
    monkeypatch.setattr("shogym.serve.protocol_v2.kernel.runtime._TURNOVER_STALLED_AFTER", 0.0)
    pause = asyncio.Event()
    verifying = asyncio.Event()

    @activity.defn(name=VERIFY_BLOBS)
    async def held_verification(payload: VerifyBlobsInput) -> BlobsVerified:
        if pause.is_set():
            verifying.set()
            await asyncio.sleep(3600)
        return BlobsVerified(verified=list(payload.references), unverified=[])

    reserve = 5
    turnover_at(10_000, reserve=reserve)
    start = replace(make_start(tasks=2), blob_root=str(tmp_path))
    async with stream_worker(env.client, activities=_instead_of_verification(held_verification)):
        caller = await open_stream(env.client, start, workflow_id="stream/gate/1")
        await caller.stream.close_queue()
        task = await caller.take()
        await caller.work(task)

        pause.set()
        claim = asyncio.create_task(
            _resume(env.client, start, workflow_id="stream/gate/1", claimant="the-replacement")
        )
        await asyncio.wait_for(verifying.wait(), timeout=30)

        # The latch is set from here. The trigger goes under the count this execution has
        # already reached, and the next accepted Update is the one that reads it and latches.
        turnover_at(1, reserve=reserve)
        await caller.stream.confirm_state()
        accepted_before = await _accepted_updates(env.client, "stream/gate/1")

        # Traffic that cannot make the boundary quiet, sent continuously and serialized through
        # the one operation slot the transport has. Every one of these is rejected.
        for _ in range(20):
            assert await rejected(
                caller.stream.pull(
                    PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor)
                )
            )
        assert await _accepted_updates(env.client, "stream/gate/1") == accepted_before

        # Traffic that is on the whitelist, sent until the reserve is spent and past it.
        spent = 0
        for _ in range(reserve * 3):
            try:
                await caller.stream.confirm_state()
                spent += 1
            except WorkflowUpdateFailedError as error:
                assert turnover_pending(error)
        assert spent <= reserve
        assert (
            await _accepted_updates(env.client, "stream/gate/1") - accepted_before <= reserve
        )
        # The boundary never happened, because the claim's handler never finished.
        assert (await caller.stream.stream_state()).turnovers == 0
        claim.cancel()


async def _latch(caller: Caller, turnover_at, client: Client, workflow_id: str) -> None:
    """Make the next Update this generation accepts be the one that decides to turn over.

    The trigger goes one above what this execution has already accepted, so a replay of the
    history recorded so far never reaches the decision and the Update sent next does. That is
    how it happens in a run: the count rises under live traffic and crosses a trigger that has
    not moved. Lowering the trigger under a count already reached would put the decision inside
    the replay instead, where the marker gating it reads as absent and stays that way.
    """
    turnover_at(await _accepted_updates(client, workflow_id) + 1)
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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

    async def drive(workflow_id: str, *, cross: bool) -> Optional[str]:
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
        rows = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        failed = next(row for row in rows if row.attempt_id == attempt(1))
        assert failed.final_failure == "seal_failed"
        return failed.failure_activity_id

    turnover_at(10_000)
    async with stream_worker(env.client, activities=served_activities):
        plain = await drive("stream/activity/plain", cross=False)
        turnover_at(10_000)
        crossed = await drive("stream/activity/crossed", cross=True)
    assert plain is not None
    assert crossed == plain


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
async def test_a_pull_retried_under_its_own_identifier_across_a_boundary_replays_its_offer(
    env, turnover_at
) -> None:
    """The request tables cross, so a retry reaches the reservation rather than a new selection.

    The service deduplicates an Update identifier per execution, so after a boundary a retry
    reaches the handler. What answers it is the generation's own binding of that request to the
    message it reserved, and a fresh identifier for the same logical request is told the message
    has been presented rather than handed a second one.
    """
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(env.client, make_start(tasks=2), workflow_id="stream/table/1")
        await caller.stream.close_queue()
        request = PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor)
        offered = await caller.stream.pull(request)

        await caller.present(offered)
        await _cross_a_boundary(caller, turnover_at, "stream/table/1", env.client)

        # The same request, once the message it reserved has been presented.
        assert await refused(caller.stream.pull(request)) == "already_presented"
        # A fresh request at a stale cursor is refused for the cursor, which is the ordinary
        # answer and not anything the boundary decided.
        stale = PullRequest(
            request_id=caller.next_id(),
            last_presented_cursor=request.last_presented_cursor,
        )
        assert await refused(caller.stream.pull(stale)) == "invalid_cursor"


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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
    recorded = json.loads(
        Path("tests/_fixtures/recorded_before_policies.json").read_text(encoding="utf-8")
    )
    started = recorded["events"][0]["workflowExecutionStartedEventAttributes"]
    assert "carry" not in json.dumps(started["input"])
    assert len(started["input"]["payloads"]) == 1


@pytest.mark.network
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


@pytest.mark.network
async def test_the_supported_profile_encodes_well_under_the_ceiling(env, turnover_at) -> None:
    """The measurement the ceiling is set from: a full roster, driven to the end, sized.

    The measured object is the whole replaced start as the configured converter produces it,
    which is the original roster, the projection, the payload metadata and the encoding
    overhead together. It is measured where it matters, at the last quiet point of a roster
    driven almost to Done, because that is where the projection is largest.

    The measurement is taken by lowering the ceiling to one byte at that point, which makes the
    generation refuse the boundary and publish the size it measured rather than crossing it.
    """
    roster = 200
    body = "x" * 2048
    turnover_at(10_000)
    async with stream_worker(env.client):
        caller = await open_stream(
            env.client, make_start(tasks=roster, body=body), workflow_id="stream/profile/1"
        )
        await caller.stream.close_queue()
        for _ in range(roster - 1):
            task = await caller.take()
            assert task.kind == "task"
            await caller.work(task)
            payload = await caller.take()
            assert payload.kind == "payload"

        ceiling = kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
        kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES = 1
        try:
            await _latch(caller, turnover_at, env.client, "stream/profile/1")
            for _ in range(2000):
                state = await caller.stream.stream_state()
                if state.turnover_refused_bytes is not None:
                    break
                await asyncio.sleep(0.02)
            else:
                raise AssertionError("the generation never measured its carrier")
        finally:
            kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES = ceiling
        measured = state.turnover_refused_bytes
        alone = _encoded(make_start(tasks=roster, body=body))
        print(
            f"\nprofile: {roster} tasks, {len(body)} byte bodies, default converter, no codec"
            f"\n  the original start alone:      {alone} bytes"
            f"\n  the whole replaced start:      {measured} bytes"
            f"\n  what the projection adds:      {measured - alone} bytes"
            f"\n  the ceiling:                   {kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES}"
            f"\n  the service's payload limit:   {2 * 1024 * 1024} bytes"
        )
        assert measured < kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES


async def _carried(client: Client, workflow_id: str, run_id: str) -> Dict[str, Any]:
    """The projection one execution was handed, read out of the bytes it was started with."""
    history = await client.get_workflow_handle(workflow_id, run_id=run_id).fetch_history()
    started = history.events[0].workflow_execution_started_event_attributes
    return json.loads(started.input.payloads[0].data.decode("utf-8"))["carry"]


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
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


@pytest.mark.network
async def test_a_call_gives_up_on_a_generation_that_is_latched_and_not_moving(
    env, turnover_at, monkeypatch
) -> None:
    """The one shape a caller stops waiting on, and what it is handed when it does.

    A generation that says it is latched, names nothing it is waiting for and is not moving has
    stopped. What the caller gets then is the rejection as the fault it already arrived as: no
    protocol code, so the transport raises it rather than mapping it, the run is recorded
    incomplete, and the agent is shown no code it has never seen.
    """
    monkeypatch.setattr("shogym.serve.protocol_v2.kernel.runtime._TURNOVER_STALLED_AFTER", 0.0)
    reserve = 1
    turnover_at(10_000, reserve=reserve)
    pause = asyncio.Event()
    verifying = asyncio.Event()

    @activity.defn(name=VERIFY_BLOBS)
    async def held_verification(payload: VerifyBlobsInput) -> BlobsVerified:
        if pause.is_set():
            verifying.set()
            await asyncio.sleep(3600)
        return BlobsVerified(verified=list(payload.references), unverified=[])

    start = replace(make_start(tasks=2), blob_root="/nowhere")
    async with stream_worker(env.client, activities=_instead_of_verification(held_verification)):
        caller = await open_stream(env.client, start, workflow_id="stream/stalled/1")
        await caller.stream.close_queue()
        pause.set()
        claim = asyncio.create_task(
            _resume(env.client, start, workflow_id="stream/stalled/1", claimant="the-second")
        )
        await asyncio.wait_for(verifying.wait(), timeout=30)
        await _latch(caller, turnover_at, env.client, "stream/stalled/1")

        state = await caller.stream.stream_state()
        assert state.turnover_requested is True
        assert state.environment_call is None and state.pending_message_id is None
        assert state.prepared_seals == {}

        with pytest.raises(WorkflowUpdateFailedError) as gave_up:
            await caller.stream.pull(
                PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor)
            )
        assert turnover_pending(gave_up.value)
        assert protocol_error_code(gave_up.value) is None, "not a code the agent has ever seen"
        claim.cancel()
