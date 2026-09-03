"""A generation under a schedule: what it serves, and in what order.

The runs here are multi-task generations, because a schedule that only ever sees one task is
indistinguishable from no schedule at all. Under Immediate every task is followed by exactly
one payload; under Never there is no payload to follow it; and under a plan with gates the
served order is the leg automaton's rather than the queue's.

Every test drives a real workflow through the same Updates a gateway sends, on Temporal's
time-skipping environment. They are marked ``network`` because that environment downloads a
test server on first use, and they skip rather than fail when it is not there.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, AsyncIterator, List, Optional, Tuple

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio.client import WorkflowFailureError, WorkflowUpdateFailedError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    IMMEDIATE,
    NEVER,
    PAYLOAD_FIRST,
    RELEASE_AT_SEAL,
    EligibilityGate,
    PullRequest,
    ReleasePlan,
    TerminalMetadata,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    ConsumerClaim,
    OfferedMessage,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    assignments_for,
    protocol_error_code,
    start_stream,
    stream_worker,
)

DOSE = 12
CLAIM_HASH = "d" * 64
BLOB = "e" * 64
CONSUMER = ConsumerClaim(consumer_id="harness-1", claim_hash=CLAIM_HASH)


def oid(value: int) -> str:
    return f"{value:032x}"


def attempt(index: int) -> str:
    return oid(0x100 + index * 4)


def make_start(
    *,
    bodies: Tuple[str, ...],
    release: ReleasePlan = IMMEDIATE,
    evaluation_only: bool = False,
    without_payload: Tuple[str, ...] = (),
) -> StreamStart:
    """One generation, with every public identifier fixed before it serves anything."""
    items = [
        TaskItem(
            task_position=index,
            attempt_id=attempt(index),
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
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id="execution-1",
        canonicalization_version="kernel.1",
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=[]
        ),
        tasks=items,
        release=release,
        assignments=assignments_for(items, release, without_payload=without_payload),
        evaluation_only=evaluation_only,
    )


def dose_bodies() -> Tuple[str, ...]:
    return tuple(f"Round {index}." for index in range(DOSE))


# The delayed transfer leg, written as a plan: the filler waits for the first payload to be
# presented, and B waits for the filler to seal. B sits ahead of the filler in the queue, so the
# order this serves is the automaton's and not the manifest's.
LEG_BODIES = ("A.", "B.", "The filler.")
LEG_PLAN = ReleasePlan(
    RELEASE_AT_SEAL,
    PAYLOAD_FIRST,
    BY_POSITION,
    gates=[
        EligibilityGate(attempt(2), after_payload_position=0),
        EligibilityGate(attempt(1), after_sealed_attempt_id=attempt(2)),
    ],
)
# One payload in the leg, and it is A's. The filler is the delay and B is the outcome, and
# neither is a position anything is delivered against, so neither carries an obligation. The
# plan is the same plan either way: what has a payload is the roster's fact.
LEG_WITHOUT_PAYLOAD = (attempt(1), attempt(2))


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await WorkflowEnvironment.start_time_skipping()
    except Exception as error:  # noqa: BLE001 - an absent test server is a skip, not a failure
        pytest.skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


async def refused(awaitable: Any) -> str:
    """Return the protocol error code a refused call carries."""
    try:
        await awaitable
    except WorkflowUpdateFailedError as error:
        code = protocol_error_code(error)
        assert code is not None, error
        return code
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

    def pull_request(self) -> PullRequest:
        return PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)

    async def pull(self, request: Optional[PullRequest] = None) -> OfferedMessage:
        return await self.stream.pull(request or self.pull_request())

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
        """Pull one message and present it, which is what the harness does all day."""
        message = await self.pull()
        await self.present(message)
        return message

    async def work(self, task: OfferedMessage) -> OfferedMessage:
        """File against the task in ``task`` and present the acknowledgement it answers with."""
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


async def open_stream(
    environment: WorkflowEnvironment, start: StreamStart, *, workflow_id: str
) -> Caller:
    stream = await start_stream(environment.client, start, workflow_id=workflow_id)
    receipt = await stream.claim_consumer(CONSUMER)
    return Caller(stream, receipt.initial_cursor)


async def served(caller: Caller, *, limit: int = 200) -> List[OfferedMessage]:
    """Drive one generation to Done the way a model would, and report what it was served.

    The loop is the whole of the model's behavior: pull, file when the result is a task, and
    pull again. What arrives between those calls is the schedule's business, which is what
    makes the returned sequence worth asserting on.
    """
    seen: List[OfferedMessage] = []
    for _ in range(limit):
        message = await caller.take()
        seen.append(message)
        if message.kind == "done":
            return seen
        if message.kind == "task":
            seen.append(await caller.work(message))
    raise AssertionError("the generation never reached Done")


@pytest.mark.network
async def test_every_task_is_followed_by_one_payload_under_immediate(env) -> None:
    """Twelve tasks, twelve payloads, one Done, and each payload against its own attempt."""
    async with stream_worker(env.client):
        start = make_start(bodies=dose_bodies())
        caller = await open_stream(env, start, workflow_id="stream/immediate/1")
        # The manifest is closed by a controller. Nothing the model calls decides when a
        # generation stops accepting work.
        await caller.stream.close_queue()

        seen = await served(caller)
        assert [message.kind for message in seen] == [
            "task",
            "seal_ack",
            "payload",
        ] * DOSE + ["done"]
        for position in range(DOSE):
            trio = seen[position * 3 : position * 3 + 3]
            assert {message.attempt_id for message in trio} == {attempt(position)}
            assert [message.message_id for message in trio] == [
                start.tasks[position].task_message_id,
                start.tasks[position].ack_message_id,
                start.tasks[position].payload_message_id,
            ]

        state = await caller.stream.stream_state()
        assert state.release_predicate == "at_seal"
        assert state.release_plan_id == IMMEDIATE.release_plan_id
        # Six facts, not one. Every payload was assigned, built, released, offered, and
        # presented, and only the last of those is a delivery.
        assert state.assignment_count == DOSE
        assert state.materialization_count == DOSE
        assert state.eligibility_count == DOSE
        assert state.offer_count == DOSE * 3 + 1
        assert state.presentation_count == DOSE * 3 + 1
        assert state.payload_delivery_count == DOSE
        assert state.wait_count == 0


@pytest.mark.network
async def test_never_creates_no_obligation_to_deliver(env) -> None:
    """The same twelve tasks under Never: no payload is built, released, offered, or shown."""
    start = make_start(bodies=dose_bodies(), release=NEVER)
    # The roster says what the generation will do rather than what the manifest could have
    # asked for, so no row here claims a payload that nothing was ever going to release.
    assert [row.creates_payload_obligation for row in start.assignments] == [False] * DOSE
    async with stream_worker(env.client):
        caller = await open_stream(env, start, workflow_id="stream/never/1")
        await caller.stream.close_queue()

        seen = await served(caller)
        assert [message.kind for message in seen] == ["task", "seal_ack"] * DOSE + ["done"]

        state = await caller.stream.stream_state()
        assert state.release_predicate == "never"
        # The outbox is absent rather than empty: there is no obligation to be in any state.
        assert state.obligations == {}
        assert state.assignment_count == DOSE
        assert state.materialization_count == 0
        assert state.eligibility_count == 0
        assert state.payload_delivery_count == 0
        assert state.presentation_count == DOSE * 2 + 1


@pytest.mark.network
async def test_a_gate_serves_the_leg_order_rather_than_the_queue_order(env) -> None:
    """After the first payload only the filler is eligible, and B waits for the filler's seal."""
    async with stream_worker(env.client):
        caller = await open_stream(
            env,
            make_start(
                bodies=LEG_BODIES, release=LEG_PLAN, without_payload=LEG_WITHOUT_PAYLOAD
            ),
            workflow_id="stream/leg/1",
        )
        await caller.stream.close_queue()

        seen = await served(caller)
        # A, its acknowledgement and its payload; then the filler, which is worked and
        # acknowledged and delivered nothing; then the task that sits ahead of the filler in
        # the queue. Exactly one payload stands between A and B, and it is A's.
        assert [message.kind for message in seen] == [
            "task",
            "seal_ack",
            "payload",
            "task",
            "seal_ack",
            "task",
            "seal_ack",
            "done",
        ]
        assert [message.attempt_id for message in seen] == (
            [attempt(0)] * 3 + [attempt(2)] * 2 + [attempt(1)] * 2 + [None]
        )
        state = await caller.stream.stream_state()
        # The filler's and B's outboxes are absent rather than empty, so Done did not wait for
        # them and nothing was ever built for them.
        assert set(state.obligations) == {attempt(0)}
        assert state.materialization_count == 1
        assert state.payload_delivery_count == 1
        # No Wait was needed to hold that order: a gate decides eligibility, not a delay.
        assert state.wait_count == 0


@pytest.mark.network
async def test_done_waits_for_a_live_task_and_for_an_unpresented_result(env) -> None:
    """A closed and empty queue is not enough, and a payload nobody has read is not either."""
    async with stream_worker(env.client):
        caller = await open_stream(
            env, make_start(bodies=("the only one.",)), workflow_id="stream/done/1"
        )
        await caller.stream.close_queue()
        task = await caller.take()
        early = await caller.pull()
        assert early.kind == "wait"
        await caller.present(early)

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
        # An offered result nobody has presented keeps the next pull out entirely, so Done is
        # not reachable while one is outstanding.
        assert await refused(caller.pull()) == "outstanding_response"
        await caller.present(acknowledgement)

        payload = await caller.pull()
        assert payload.kind == "payload"
        state = await caller.stream.stream_state()
        assert state.obligations[attempt(0)] == "offered"
        # An offer is not a delivery: the payload is reserved and nothing has carried it.
        assert state.payload_delivery_count == 0
        await caller.present(payload)

        state = await caller.stream.stream_state()
        assert state.presentation_count == 4
        assert state.payload_delivery_count == 1
        done = await caller.take()
        assert done.kind == "done"
        assert (await caller.stream.handle.result()).generation_state == "done"


@pytest.mark.network
async def test_positions_hold_through_retries_and_waits(env) -> None:
    """A replayed request and a run of Waits change no position and no public identifier."""
    async with stream_worker(env.client):
        start = make_start(bodies=("first.", "second."))
        caller = await open_stream(env, start, workflow_id="stream/positions/1")

        request = caller.pull_request()
        task = await caller.pull(request)
        replayed = await caller.stream.handle.execute_update(
            StreamWorkflow.pull, request, id="a-second-delivery-of-one-request"
        )
        assert replayed == task

        await caller.present(task)
        seen: List[OfferedMessage] = [task, await caller.work(task), await caller.take()]
        second = await caller.take()
        seen += [second, await caller.work(second), await caller.take()]

        # The queue is still open, so a pull now is a Wait, and an early poll is another one.
        waits = [await caller.take() for _ in range(2)]
        assert [wait.kind for wait in waits] == ["wait", "wait"]
        # The reason lives in the generation's state and never in the record.
        for wait in waits:
            assert "queue_open" not in wait.visible_text
        state = await caller.stream.stream_state()
        assert state.wait_reasons == {"queue_open": 2}

        await caller.stream.close_queue()
        assert (await caller.pull()).kind == "done"
        for kind, fixed in (
            ("task", [row.task_message_id for row in start.assignments]),
            ("seal_ack", [row.ack_message_id for row in start.assignments]),
            ("payload", [row.payload_message_id for row in start.assignments]),
        ):
            assert [message.message_id for message in seen if message.kind == kind] == fixed


@pytest.mark.network
async def test_two_generations_under_one_plan_wait_alike(env) -> None:
    """A blinded comparison matches the whole Wait pattern, so one plan has to produce one."""
    async with stream_worker(env.client):
        start = make_start(bodies=("the only one.",))
        patterns = []
        for index, generation in enumerate((start, replace(start))):
            caller = await open_stream(env, generation, workflow_id=f"stream/matched/{index}")
            task = await caller.take()
            await caller.work(task)
            await caller.take()
            waits = [await caller.take() for _ in range(3)]
            state = await caller.stream.stream_state()
            patterns.append(
                ([wait.visible_text for wait in waits], state.release_plan_id, state.wait_reasons)
            )
        assert patterns[0] == patterns[1]
        assert patterns[0][2] == {"queue_open": 3}


@pytest.mark.network
async def test_a_generation_whose_schedule_does_not_hold_together_never_starts(env) -> None:
    """The manifest, the roster, and the plan describe one generation, or nothing runs."""
    async with stream_worker(env.client):
        start = make_start(bodies=("first.", "second."))
        skewed = replace(
            start,
            assignments=[
                replace(row, payload_position=1 - row.payload_position)
                for row in start.assignments
            ],
        )
        # An evaluation that scores without delivering cannot be given an outbox either.
        evaluating = make_start(bodies=("first.",), evaluation_only=True)
        for index, refusable in enumerate((skewed, evaluating)):
            stream = await start_stream(
                env.client, refusable, workflow_id=f"stream/skewed/{index}"
            )
            with pytest.raises(WorkflowFailureError) as caught:
                await stream.handle.result()
            assert protocol_error_code(caught.value.cause) == "configuration_mismatch"
