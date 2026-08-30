"""Reading a run's records back out, and what a directory with nothing in it answers.

The protocol keeps the score in the one place it can be authoritative: the generation's own
durable history. So a record is read by asking that history rather than by parsing anything a
run wrote down while it served, and the file a reader leaves behind is a view of the answer.

The live half needs a service, so it is marked ``network`` and downloads one on first use. What
the Query answers is asked of the time-skipping environment, which is quick; the two tests of
the reader itself serve onto the database a run directory keeps and read that directory back,
because a run directory holding its own history is the arrangement the reader exists for. The
offline half is the part that has no history to ask: a directory whose database is not there is
not a failure, it is a directory with nothing to read.

Three distinctions the rows exist to keep get tests of their own. An offer is a reservation and
a presentation is what the model saw, so every message here is asked about while it is offered
as well as after it has been presented. A position is the roster's and not the order a stream
happened to serve, so one generation is gated into serving backwards. And a payload that never
arrived is one fact when the generation owed one and a different fact when it never did, which
is what a filler position is.

Reading is also the thing that must change nothing. A run stopped in the middle of a seal is
read here twice, with the environment's own Activities registered and unanswerable, to show
that the reader neither seals nor grades in the environment's place. A run stopped just after
the environment answered is the harder cut: the workflow task waiting on that answer goes to
whoever polls next, so that run is refused rather than read forward. What a read does leave
behind is a file, so one is written over a forged one and the run is then resumed and finished.

The clock is the work nobody has to ask for, so it gets two cuts of its own. A run read after
its attempt deadline has passed is refused and its file is byte for byte what it was, which is
both halves of costing the run nothing. And a read held open until that deadline falls due, with
its own Worker already polling, is the case the first check cannot see: the copy it answered
from grew while it was reading, so it is refused by the marks taken either side of the Query.

A named service is the arrangement with no copy in it. One is started here with an address, a
run is served against it and stopped mid seal, and both answers are asked for: a deployment that
is serving answers the read while nothing is started to ask it, and a deployment with nobody
serving answers nothing, which is a refusal and not a reason to start a Worker on somebody
else's service.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from contextlib import asynccontextmanager, suppress
from datetime import timedelta
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Sequence, Tuple

import pytest
import pytest_asyncio
from temporalio import activity
from temporalio.api.enums.v1 import EventType
from temporalio.client import Client
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

from shogym.serve.protocol_v2 import PullRequest, TerminalMetadata, blob_ref
from shogym.serve.protocol_v2.kernel import (
    ABANDONED,
    DEADLINE,
    SEAL_UNUSABLE,
    STREAM_TASK_QUEUE,
    TEMPORAL_ADDRESS_ENV,
    AttemptRecord,
    ConsumerClaim,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    GradeAttemptResult,
    OfferedMessage,
    PayloadBundle,
    SealAttemptInput,
    SealAttemptResult,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    assignments_for,
    configuration_hash,
    durable_client,
    generate_payload_bundle_activity,
    grade_attempt_activity,
    resume_run_directory,
    seal_attempt_activity,
    start_stream,
    stream_worker,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel.activities import (
    GENERATE_PAYLOAD_BUNDLE,
    GRADE_ATTEMPT,
    SEAL_ATTEMPT,
)
from shogym.serve.protocol_v2.kernel.runtime import STREAM_DATABASE_FILE, temporal_home
from shogym.serve.protocol_v2 import reader as reader_module
from shogym.serve.protocol_v2.reader import (
    NOTE_FILE,
    RECORDS_FILE,
    NothingToRead,
    ReadRefused,
    RunRecords,
    format_records,
    read_records,
    write_records,
)
from shogym.serve.protocol_v2.rundir import create_run_directory
from shogym.serve.protocol_v2.schedule import (
    BY_POSITION,
    IMMEDIATE,
    PAYLOAD_FIRST,
    RELEASE_AT_SEAL,
    EligibilityGate,
    ReleasePlan,
)

#: How long an attempt of a generation that declares a deadline gets, and a deadline no test
#: here is meant to reach: the second one is armed so that a row can say a clock is running.
DEADLINE_MS = 5_000
A_DEADLINE_NOTHING_REACHES_MS = 3_600_000

#: A deadline a read arrives after, on purpose. The generation is stopped while its attempt is
#: still active, and the read happens once this has gone by, which is the state where a service
#: brought up to look at a run would end an attempt its owner has not finished with.
A_DEADLINE_A_READ_ARRIVES_AFTER_MS = 2_500

#: How long a Query against a service with nobody serving it is given before the read reports
#: that nobody answered. The module's own bound is generous because a real deployment replaying
#: a long history is slow; what is being tested is the answer, so this one is short.
NOBODY_IS_SERVING_MS = 2_000

CLAIM_HASH = "d" * 64
TRANSCRIPT_BLOB = "e" * 64
PROVIDER_TURN_BLOB = "f" * 64
CHECKPOINT_BLOB = "9" * 64
CANONICALIZATION = "kernel.1"
WORKFLOW_ID = "stream/records/1"


def oid(value: int) -> str:
    return f"{value:032x}"


def make_start(
    bodies: Sequence[str],
    *,
    release: ReleasePlan = IMMEDIATE,
    without_payload: Sequence[str] = (),
    attempt_deadline_ms: int = 0,
) -> StreamStart:
    """A generation whose every public identifier is fixed before it serves anything.

    ``without_payload`` names the positions this generation delivers nothing against, which is
    what a filler is: its task is served and scored, and no payload is ever owed for it.

    The payload positions run backwards against the task positions. A row carries both because
    they are two schedules and an analysis joins on each, and a generation that numbered them
    alike would let a projection answer with whichever one it had twice.
    """
    tasks = [
        TaskItem(
            task_position=index,
            attempt_id=oid(0x100 + index * 4),
            task_message_id=oid(0x101 + index * 4),
            ack_message_id=oid(0x102 + index * 4),
            payload_position=len(bodies) - 1 - index,
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
        canonicalization_version=CANONICALIZATION,
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=["answer"]
        ),
        tasks=tasks,
        release=release,
        assignments=assignments_for(tasks, release, without_payload=without_payload),
        attempt_deadline_ms=attempt_deadline_ms,
    )


class Driver:
    """One consumer taking attempts from a stream, so a test reads as protocol steps."""

    def __init__(self, stream: StreamHandle, cursor: str, *, minted: int = 0) -> None:
        """``minted`` is how many identifiers the owner before this one already spent.

        A replacement that started counting again would reach a request the generation has
        answered, under different content, which the generation refuses as the conflict it is.
        """
        self.stream = stream
        self.cursor = cursor
        self._counter = minted

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x1000 + self._counter)

    async def offer(self) -> OfferedMessage:
        """Pull the next message and stop there. An offer is a reservation, not a delivery."""
        return await self.stream.pull(
            PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )

    async def take(self) -> OfferedMessage:
        message = await self.offer()
        await self.present(message)
        return message

    async def present(self, message: OfferedMessage) -> None:
        ack = await self.stream.present(
            message,
            attestation_id=self.next_id(),
            transcript_blob=TRANSCRIPT_BLOB,
            provider_turn_blob=PROVIDER_TURN_BLOB if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=CHECKPOINT_BLOB if message.kind == "task" else None,
        )
        self.cursor = ack.cursor

    async def file(self, attempt_id: str) -> OfferedMessage:
        """File the terminal for one attempt and return the acknowledgement it offers."""
        return await self.stream.seal(
            SealRequest(
                metadata=TerminalMetadata(
                    request_id=self.next_id(),
                    last_presented_cursor=self.cursor,
                    attempt_id=attempt_id,
                ),
                public_tool_name="submit",
                native_terminal_name="submit",
                native_arguments={"answer": "42"},
            )
        )

    async def work(self) -> str:
        """Pull one task, file it, and take the acknowledgement. No payload is pulled."""
        task = await self.take()
        assert task.attempt_id is not None
        await self.present(await self.file(task.attempt_id))
        return task.attempt_id

    async def solve(self) -> str:
        """Work one task and take the payload its seal released as well."""
        attempt_id = await self.work()
        await self.take()
        return attempt_id

    async def records(self) -> List[AttemptRecord]:
        return list(await self.stream.handle.query(StreamWorkflow.attempt_records))

    async def delivered(self) -> Tuple[bool, bool, bool]:
        """Which of the one attempt's three messages have been handed to the transport."""
        record = (await self.records())[0]
        return (record.task_delivered, record.ack_delivered, record.payload_delivered)


class _Environment:
    """What the environment's own seal has reached, and what it is waiting for.

    ``reached`` is set the moment the seal is entered, so a test stops with one in flight rather
    than guessing how long scheduling one takes. ``answers`` is what the environment is waiting
    on: a test that never sets it stops with the seal still in the environment's hands, and a
    test that sets it stops with the environment's answer durable and nobody left to apply it.
    """

    def __init__(self) -> None:
        self.reached = asyncio.Event()
        self.answers = asyncio.Event()

    def again(self) -> "_Environment":
        """Two fresh events, because one belongs to the loop that first waited on it."""
        self.reached = asyncio.Event()
        self.answers = asyncio.Event()
        return self


_ENVIRONMENT = _Environment()


@activity.defn(name=SEAL_ATTEMPT)
async def _environment_seal(request: SealAttemptInput) -> SealAttemptResult:
    """The environment's own seal, which answers when the test says the environment did.

    This is what a run that stopped in the middle of a seal leaves behind. The Activity task is
    on the run's own queue, and only the environment that owns it may say what it returns.
    """
    _ENVIRONMENT.reached.set()
    await _ENVIRONMENT.answers.wait()
    submission = "the environment's own submission"
    return SealAttemptResult(
        attempt_id=request.attempt_id,
        seal_id=request.seal_id,
        canonicalization_version=request.canonicalization_version,
        canonical_submission_text=submission,
        canonical_submission=blob_ref(submission),
        environment_recovery_token="recovery-1",
    )


@activity.defn(name=GRADE_ATTEMPT)
async def _environment_grade(request: GradeAttemptInput) -> GradeAttemptResult:
    """The environment's own grader. Nothing but the environment may reach it."""
    raise AssertionError("something graded this attempt that was not the environment")


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def _environment_payload(request: GeneratePayloadBundleInput) -> PayloadBundle:
    """The environment's own payload builder. Nothing but the environment may reach it."""
    raise AssertionError("something built this payload that was not the environment")


#: What a run served by an environment that seals and grades for itself has registered under
#: the kernel's Activity names. Blob verification stays the kernel's, because it is the
#: kernel's: it reads an object and hashes it and reaches no environment.
ENVIRONMENT_ACTIVITIES = [
    _environment_seal,
    _environment_grade,
    _environment_payload,
    verify_blobs_activity,
]


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await WorkflowEnvironment.start_time_skipping()
    except Exception as error:  # noqa: BLE001 - an absent test server is a skip, not a failure
        pytest.skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


async def open_stream(client: Client, start: StreamStart) -> Driver:
    stream = await start_stream(client, start, workflow_id=WORKFLOW_ID)
    receipt = await stream.claim_consumer(
        ConsumerClaim(consumer_id="harness-1", claim_hash=CLAIM_HASH)
    )
    return Driver(stream, receipt.initial_cursor)


def a_run_directory(root: Path, start: StreamStart) -> Path:
    """A directory naming the generation these tests serve, and holding nothing else.

    The recorded hash is the one ``start`` derives, which is what a directory a later owner can
    resume holds: a resume presents its own composition and is refused by any other value.
    """
    create_run_directory(
        root,
        workflow_id=WORKFLOW_ID,
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash=configuration_hash(start),
    )
    return root


def environment_worker(client: Client) -> Worker:
    """The half of a run the environment owns: its Activities, and no workflow.

    A run is two pollers on one queue, and separating them is what lets a test stop one of them
    while the other keeps going, which is the crash a reader can arrive after.
    """
    return Worker(client, task_queue=STREAM_TASK_QUEUE, activities=ENVIRONMENT_ACTIVITIES)


@asynccontextmanager
async def a_service_somebody_else_runs(root: Path) -> AsyncIterator[Client]:
    """A durable service with an address, which is the arrangement the address variable names.

    Its database is this test's and not the run directory's, because that is the whole of the
    difference: a run served against a named service keeps no history of its own, so there is
    nothing for a read to copy and nothing a read can protect by copying.
    """
    environment = await WorkflowEnvironment.start_local(
        dev_server_database_filename=str(root / "service.sqlite"),
        download_dest_dir=str(temporal_home()),
    )
    try:
        yield environment.client
    finally:
        await environment.shutdown()


def no_worker_a_read_may_start(*args: Any, **kwargs: Any) -> Worker:
    """Stand where the reader's own Worker would be built, on a service it does not own."""
    raise AssertionError("a read of a named service started a Worker on that service")


async def a_run_stopped_with_its_clock_running(
    root: Path, start: StreamStart
) -> Tuple[List[AttemptRecord], float]:
    """Serve one task into ``root`` and stop, leaving the attempt active and its clock armed.

    The mark that comes back is when the clock started, which is the presentation of the task.
    A test waits from there rather than from wherever it happens to be, so what it waits for is
    the deadline this generation actually armed.
    """
    async with durable_client(run_directory=root) as client:
        async with stream_worker(client):
            driver = await open_stream(client, start)
            await driver.take()
            armed = time.monotonic()
            records = await driver.records()
    return records, armed


async def past(armed: float, milliseconds: int) -> None:
    """Wait until that deadline is behind us, with room for the clock the service keeps."""
    await asyncio.sleep(max(0.0, milliseconds / 1000 + 0.7 - (time.monotonic() - armed)))


async def history_events(client: Client) -> List[int]:
    """Every event of the generation's history, by type, which is what a read must not add to."""
    fetched = await client.get_workflow_handle(WORKFLOW_ID).fetch_history()
    return [int(event.event_type) for event in fetched.events]


async def durable_answer_nobody_applied(client: Client) -> List[int]:
    """Wait until the environment's answer is in the history with no workflow task run on it."""

    async def landed() -> List[int]:
        while True:
            events = await history_events(client)
            if events[-2:] == [
                int(EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED),
                int(EventType.EVENT_TYPE_WORKFLOW_TASK_SCHEDULED),
            ]:
                return events
            await asyncio.sleep(0.2)

    return await asyncio.wait_for(landed(), timeout=60)


def database_bytes(root: Path) -> Dict[str, bytes]:
    """Every file the run's own service keeps its history in, by name and by content."""
    return {path.name: path.read_bytes() for path in sorted(root.glob(f"{STREAM_DATABASE_FILE}*"))}


def field_names() -> List[str]:
    return [field.name for field in dataclasses.fields(AttemptRecord)]


def by_id(records: List[AttemptRecord]) -> Dict[str, AttemptRecord]:
    return {record.attempt_id: record for record in records}


@pytest.mark.network
async def test_a_run_says_what_each_attempt_filed_and_scored(env: WorkflowEnvironment) -> None:
    """Two sealed attempts and one nobody pulled, read back as three rows."""
    async with stream_worker(env.client):
        driver = await open_stream(env.client, make_start(("first", "second", "third")))
        first = await driver.solve()
        second = await driver.solve()
        records = await driver.records()

    assert [record.attempt_id for record in records] == [oid(0x100), oid(0x104), oid(0x108)]
    assert [record.task_position for record in records] == [0, 1, 2]
    # The other schedule, and it is not the first one relabelled: a row is joined to a task at
    # one position and to a payload at another, and both are the roster's.
    assert [record.payload_position for record in records] == [2, 1, 0]
    assert (first, second) == (oid(0x100), oid(0x104))

    sealed = by_id(records)[first]
    assert sealed.state == "ack_presented"
    assert sealed.score == 1
    assert sealed.decode_state == "decoded"
    assert sealed.seal_ordinal == 1
    assert sealed.terminal_tool == "submit"
    assert sealed.canonicalization_version == CANONICALIZATION
    assert len(sealed.submission_digest or "") == 64
    assert sealed.task_delivered and sealed.ack_delivered and sealed.payload_delivered
    assert sealed.payload_position == 2
    # All three identifiers, each in its own field. A projection that crossed two of them would
    # still be one naming only identifiers this attempt owns, and these are the joins a
    # transcript is matched on, so the mapping is asserted and not just the membership.
    assert (sealed.task_message_id, sealed.ack_message_id, sealed.payload_message_id) == (
        oid(0x101),
        oid(0x102),
        oid(0x103),
    )
    assert sealed.creates_payload_obligation
    assert sealed.payload_state == "presented"
    assert by_id(records)[second].seal_ordinal == 2

    # The attempt nobody pulled. It has a row, because the roster is what a run is counted over,
    # and its score is absent rather than nought: nothing about it has been graded.
    untouched = by_id(records)[oid(0x108)]
    assert untouched.state == "planned"
    assert untouched.score is None
    assert untouched.submission_digest is None
    assert untouched.decode_state is None
    assert untouched.seal_ordinal is None
    assert untouched.terminal_tool is None
    assert not untouched.task_delivered
    assert not untouched.ack_delivered
    assert not untouched.payload_delivered
    assert (
        untouched.task_message_id,
        untouched.ack_message_id,
        untouched.payload_message_id,
    ) == (oid(0x109), oid(0x10A), oid(0x10B))
    # A payload was owed for it and nothing has happened to that obligation yet, which is not
    # the same row as a position that was never going to have one.
    assert untouched.creates_payload_obligation
    assert untouched.payload_state == "assigned"


@pytest.mark.network
async def test_an_offer_is_not_a_presentation_for_any_of_the_three_messages(
    env: WorkflowEnvironment,
) -> None:
    """Each message is asked about while it is offered, and again once it has been presented.

    A row that read the outbox instead of the cursor would call all three of these delivered
    the moment the stream reserved them, and the deliveries an analysis counts would become the
    messages the stream chose to send rather than the ones it handed to the transport.
    """
    async with stream_worker(env.client):
        driver = await open_stream(env.client, make_start(("first",)))

        task = await driver.offer()
        assert task.kind == "task"
        assert await driver.delivered() == (False, False, False)
        assert (await driver.records())[0].payload_state == "assigned"

        await driver.present(task)
        assert await driver.delivered() == (True, False, False)

        assert task.attempt_id is not None
        acknowledgement = await driver.file(task.attempt_id)
        assert acknowledgement.kind == "seal_ack"
        assert await driver.delivered() == (True, False, False)
        assert (await driver.records())[0].payload_state == "eligible"

        await driver.present(acknowledgement)
        assert await driver.delivered() == (True, True, False)

        payload = await driver.offer()
        assert payload.kind == "payload"
        assert await driver.delivered() == (True, True, False)
        assert (await driver.records())[0].payload_state == "offered"

        await driver.present(payload)
        assert await driver.delivered() == (True, True, True)
        assert (await driver.records())[0].payload_state == "presented"


@pytest.mark.network
async def test_a_row_keeps_its_assigned_position_and_not_the_one_it_was_served_in(
    env: WorkflowEnvironment,
) -> None:
    """A generation served backwards: position one goes out first and stays the second row.

    Positions are what an analysis joins and groups on, so they have to come from the roster
    committed before anything was served. A projection that numbered its rows by the order they
    went out, or by the order they were sealed, agrees with this one on every generation served
    front to back and disagrees on this one.
    """
    # Position zero waits for position one to be sealed, which is a gate rather than the queue,
    # so the generation is served and sealed in the opposite order from the one it is counted in.
    backwards = ReleasePlan(
        predicate=RELEASE_AT_SEAL,
        priority=PAYLOAD_FIRST,
        tie_key=BY_POSITION,
        gates=[EligibilityGate(attempt_id=oid(0x100), after_sealed_attempt_id=oid(0x104))],
    )
    async with stream_worker(env.client):
        driver = await open_stream(env.client, make_start(("first", "second"), release=backwards))
        served = [await driver.solve(), await driver.solve()]
        records = await driver.records()

    assert served == [oid(0x104), oid(0x100)]
    assert [record.task_position for record in records] == [0, 1]
    assert [record.payload_position for record in records] == [1, 0]
    assert [record.attempt_id for record in records] == [oid(0x100), oid(0x104)]
    assert [record.seal_ordinal for record in records] == [2, 1]
    assert [record.task_message_id for record in records] == [oid(0x101), oid(0x105)]


@pytest.mark.network
async def test_a_row_that_was_owed_no_payload_is_not_a_row_that_missed_one(
    env: WorkflowEnvironment,
) -> None:
    """A filler position beside one whose payload was released and never pulled.

    Both were worked and acknowledged, and nothing was delivered against either, so what was
    presented calls them the same row. They are not the same row. One is what the generation
    was built to do at that position and the other is a payload that was owed and did not
    arrive, and an analysis that cannot separate them counts fillers as failed deliveries.
    """
    filler = oid(0x100)
    async with stream_worker(env.client):
        driver = await open_stream(
            env.client, make_start(("filler", "owed"), without_payload=(filler,))
        )
        assert await driver.work() == filler
        assert await driver.work() == oid(0x104)
        records = await driver.records()

    silent, owed = records
    assert (silent.attempt_id, owed.attempt_id) == (filler, oid(0x104))
    assert silent.state == owed.state == "ack_presented"
    assert not silent.payload_delivered
    assert not owed.payload_delivered

    assert not silent.creates_payload_obligation
    assert silent.payload_state is None
    assert owed.creates_payload_obligation
    assert owed.payload_state == "eligible"


@pytest.mark.network
async def test_a_run_says_why_each_attempt_that_never_filed_ended(env: WorkflowEnvironment) -> None:
    """A finished generation whose three attempts all ended without a filing, read row by row.

    The floor is one number for four different endings, so the number alone cannot be the
    record of any of them. Here the clock ends the first, the seal's own batch ends the second
    with a result it could not vouch for, and the third is the one nothing was ever going to
    reach: its gate waited on a seal that will not happen. All three carry the floor and none
    of them carries it for the same reason, which is what an analysis counting outcomes has to
    be able to separate.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def score_of_another_seal(request: GradeAttemptInput) -> GradeAttemptResult:
        return dataclasses.replace(await grade_attempt_activity(request), seal_id="0" * 64)

    # The last task waits for the middle one to seal, which is a wait that ends when that
    # attempt ends instead.
    gated = ReleasePlan(
        predicate=RELEASE_AT_SEAL,
        priority=PAYLOAD_FIRST,
        tie_key=BY_POSITION,
        gates=[EligibilityGate(attempt_id=oid(0x108), after_sealed_attempt_id=oid(0x104))],
    )
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            score_of_another_seal,
            generate_payload_bundle_activity,
            verify_blobs_activity,
        ],
    ):
        driver = await open_stream(
            env.client,
            make_start(
                ("out of time", "unusable", "never reached"),
                release=gated,
                attempt_deadline_ms=DEADLINE_MS,
            ),
        )
        await driver.take()
        await env.sleep(timedelta(milliseconds=DEADLINE_MS + 1000))

        second = await driver.take()
        assert second.attempt_id == oid(0x104)
        with pytest.raises(Exception):
            await driver.file(second.attempt_id)

        await driver.stream.close_queue()
        done = await driver.offer()
        assert done.kind == "done"
        await driver.present(done)
        outcome = await driver.stream.handle.result()
        records = await driver.records()

    assert outcome.finalized == 3
    assert outcome.sealed == 0
    expired, unusable, unreached = records
    assert [record.state for record in records] == ["final_failed"] * 3
    assert [record.final_failure for record in records] == [DEADLINE, SEAL_UNUSABLE, ABANDONED]
    # The floor, and it is a number rather than an absence: these attempts have an outcome.
    assert [record.score for record in records] == [0.0, 0.0, 0.0]

    # The expiry is the clock's own column and it is spent, not still waiting: the ending it
    # was recorded for has landed.
    assert not any(record.deadline_expired for record in records)
    assert expired.seal_ordinal is None
    assert expired.submission_digest is None

    # The batch got far enough to fix a submission and never far enough to score it.
    assert unusable.terminal_tool == "submit"
    assert len(unusable.submission_digest or "") == 64
    assert unusable.decode_state is None

    # Nothing was ever served against the third, which is what its ending says.
    assert not unreached.task_delivered
    assert unreached.terminal_tool is None
    assert [record.payload_state for record in records] == ["final_failed"] * 3


@pytest.mark.network
async def test_the_file_a_read_leaves_behind_holds_exactly_those_rows(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """One JSON object per attempt, in the record's own field order, and a note saying so."""
    async with stream_worker(env.client):
        driver = await open_stream(env.client, make_start(("first", "second", "third")))
        await driver.solve()
        await driver.solve()
        records = await driver.records()

    run = RunRecords(root=tmp_path, workflow_id=WORKFLOW_ID, records=records)
    path = write_records(run)
    assert path == tmp_path / RECORDS_FILE

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [dataclasses.asdict(record) for record in records]
    assert [list(row) for row in rows] == [field_names()] * len(records)

    note = (tmp_path / NOTE_FILE).read_text(encoding="utf-8")
    assert "derived view" in note
    assert WORKFLOW_ID in note
    assert RECORDS_FILE in note


@pytest.mark.network
async def test_a_run_directory_is_read_out_of_the_authority_the_manifest_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path the command runs: open the directory, reach its history, project it.

    This one serves onto the database the run directory keeps and then reads that directory
    back, which is the arrangement a run of the pilot leaves behind. Reading twice answers the
    same both times, which is what reading changing nothing has to mean for a run somebody
    comes back to.

    A file claiming to hold this run's records is put there first, saying things the history
    does not. The read is not entitled to believe it, and the write that follows replaces it,
    because the history is the record and the file is a view somebody may have edited. Then the
    run is resumed and finishes its second task, which is the other half of leaving a directory
    alone: a read is something a run survives, and the file it leaves behind goes in beside the
    manifest rather than over anything the next owner opens.
    """
    monkeypatch.delenv(TEMPORAL_ADDRESS_ENV, raising=False)
    start = make_start(("first", "second"))
    root = a_run_directory(tmp_path, start)
    async with durable_client(run_directory=root) as client:
        async with stream_worker(client):
            driver = await open_stream(client, start)
            await driver.solve()
            served = await driver.records()

    forged = root / "records.jsonl"
    forged.write_text(
        json.dumps({"attempt_id": oid(0x100), "score": 1.0, "state": "sealed"}) + "\n" * 4,
        encoding="utf-8",
    )

    first = await read_records(root)
    second = await read_records(root)
    assert first.root == root
    assert first.workflow_id == WORKFLOW_ID
    assert first.records == served
    assert second.records == served

    # The rows the history answered with, and none of what the file said before this.
    assert write_records(first) == forged
    rows = [json.loads(line) for line in forged.read_text(encoding="utf-8").splitlines()]
    assert rows == [dataclasses.asdict(record) for record in served]

    async with durable_client(run_directory=root) as client:
        async with stream_worker(client):
            taken = await resume_run_directory(client, root, start=start, claimant_id="the-next")
            state = await taken.stream_state()
            resumed = Driver(taken, state.cursor, minted=100)
            assert await resumed.solve() == oid(0x104)
            finished = await resumed.records()

    # The read did not cost the run its second task, and the row the read reported is the row
    # the run went on to keep.
    assert finished[0] == served[0]
    assert [record.score for record in finished] == [1, 1]


@pytest.mark.network
async def test_reading_a_run_never_runs_the_environments_half_of_a_seal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A generation stopped mid seal, read twice, and unchanged by having been read.

    An environment that seals and grades for itself installs its own implementations under the
    kernel's Activity names, so a reader that registered the kernel's stand-ins would poll this
    run's own queue and answer in the environment's place: the submission digest and the score
    would be whoever read the directory's rather than the environment's. The Query is not what
    does that. The Worker wrapped around it is.
    """
    environment = _ENVIRONMENT.again()
    monkeypatch.delenv(TEMPORAL_ADDRESS_ENV, raising=False)
    start = make_start(("first",))
    root = a_run_directory(tmp_path, start)
    async with durable_client(run_directory=root) as client:
        async with stream_worker(client, activities=ENVIRONMENT_ACTIVITIES):
            driver = await open_stream(client, start)
            task = await driver.take()
            assert task.attempt_id is not None
            filing = asyncio.create_task(driver.file(task.attempt_id))
            await asyncio.wait_for(environment.reached.wait(), timeout=60)
            filing.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await filing
            interrupted = await driver.records()

    assert interrupted[0].state == "sealing"
    assert interrupted[0].submission_digest is None
    assert interrupted[0].score is None

    # A seal the environment is still holding is an open generation, and an open generation is
    # what this reader is for. Nothing is waiting on a Worker here, so reading answers.
    for _ in range(2):
        assert (await read_records(root)).records == interrupted


@pytest.mark.network
async def test_a_seal_the_environment_answered_is_not_applied_by_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment's answer landed with nobody left to apply it, and a read is not who does.

    Registering no Activity is not the whole of changing nothing. A Query is answered by a
    Worker polling the run's own queue, and a run whose last Activity result arrived after its
    workflow half was gone leaves a workflow task waiting for whoever polls next. That task is
    the run's next step: it takes the environment's submission, records the digest of it, and
    commands the grade. A Worker started to read would run it and write those into the history
    somebody opened the directory to look at, so a generation holding one is refused instead.
    """
    environment = _ENVIRONMENT.again()
    monkeypatch.delenv(TEMPORAL_ADDRESS_ENV, raising=False)
    start = make_start(("first",))
    root = a_run_directory(tmp_path, start)
    async with durable_client(run_directory=root) as client:
        async with environment_worker(client):
            async with stream_worker(client, activities=[]):
                driver = await open_stream(client, start)
                task = await driver.take()
                assert task.attempt_id is not None
                filing = asyncio.create_task(driver.file(task.attempt_id))
                await asyncio.wait_for(environment.reached.wait(), timeout=60)
                filing.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await filing
            # The workflow half of the run has stopped and the environment answers into a
            # history nobody is polling, which is what a crash between the two leaves behind.
            environment.answers.set()
            interrupted = await durable_answer_nobody_applied(client)

    with pytest.raises(ReadRefused) as refused:
        await read_records(root)
    assert "workflow task" in str(refused.value)

    async with durable_client(run_directory=root) as client:
        assert await history_events(client) == interrupted


@pytest.mark.network
async def test_reading_a_live_generation_leaves_its_history_byte_for_byte_where_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run stopped with an attempt still running, read, and its database file unchanged.

    Registering no Activity and refusing an unapplied workflow task are both about what a
    Worker runs, and neither of them is the whole of it. A service fires the durable timers it
    finds the moment it comes up, and it writes what it is doing down as it goes, so pointing
    one at the run's own database is already spending the history somebody came to look at. An
    attempt whose deadline passed while nobody was serving would be ended by the act of being
    read, and the ending is the owner that resumes the run's to make.

    So the read gets a copy and the run keeps its file. This one leaves an attempt active with
    its clock running, which is the state that has something to lose, and asks for the bytes of
    the database on either side of the read.
    """
    monkeypatch.delenv(TEMPORAL_ADDRESS_ENV, raising=False)
    start = make_start(("first", "second"), attempt_deadline_ms=A_DEADLINE_NOTHING_REACHES_MS)
    root = a_run_directory(tmp_path, start)
    async with durable_client(run_directory=root) as client:
        async with stream_worker(client):
            driver = await open_stream(client, start)
            await driver.solve()
            await driver.take()
            live = await driver.records()

    before = database_bytes(root)
    read = await read_records(root)
    after = database_bytes(root)

    assert read.records == live
    sealed, running = read.records
    assert sealed.score == 1
    assert sealed.final_failure is None
    # The attempt whose clock is running, said as the run has it: no ending, and no expiry.
    assert running.state == "active"
    assert running.score is None
    assert running.final_failure is None
    assert not running.deadline_expired

    assert before
    assert after == before


@pytest.mark.network
async def test_a_run_whose_clock_ran_out_is_refused_rather_than_ended_by_being_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An attempt whose deadline passed while nobody served it, read after the fact.

    The copy protects the run's file and it does not protect the run's answer. A service fires
    the durable timers it finds waiting, so bringing this history up makes the generation's next
    step ready for whoever polls, and the rows a read produced by running it would say the
    attempt ended at the floor with a score of nought. The run says no such thing. It says the
    attempt is active, and the owner that resumes it is who ends it or finishes it.

    So the read is refused, and the file it declined to read is byte for byte what it was. Both
    halves are asserted, because leaving the bytes alone is not the same as leaving the answer
    alone and this reader owes the run each of them.
    """
    monkeypatch.delenv(TEMPORAL_ADDRESS_ENV, raising=False)
    start = make_start(
        ("first", "second"), attempt_deadline_ms=A_DEADLINE_A_READ_ARRIVES_AFTER_MS
    )
    root = a_run_directory(tmp_path, start)
    live, armed = await a_run_stopped_with_its_clock_running(root, start)
    assert live[0].state == "active"
    assert live[0].final_failure is None
    assert live[0].score is None

    before = database_bytes(root)
    await past(armed, A_DEADLINE_A_READ_ARRIVES_AFTER_MS)
    with pytest.raises(ReadRefused):
        await read_records(root)

    assert before
    assert database_bytes(root) == before


@pytest.mark.network
async def test_a_read_whose_worker_moved_the_copy_is_refused_by_what_it_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clock falls due after the check, while the read's own Worker is polling.

    Refusing a generation that already holds an unapplied workflow task is one round trip, and
    the window after it is not empty. A Worker is a poller for as long as it lives, so work that
    becomes ready while it is up goes to it, and a deadline is exactly the kind of work that
    becomes ready without anybody asking for it. The Query is held here until this generation's
    clock has run out, which makes that window wide enough to see rather than making it exist.

    What catches it is the history itself, marked before the Worker starts and again once it has
    stopped. A copy that grew answered with rows the read made, and the refusal says so with
    both counts in it. The run's own file is untouched either way, and that is asserted too:
    what is wrong with such a read is whose rows it returned, not damage it did.
    """
    monkeypatch.delenv(TEMPORAL_ADDRESS_ENV, raising=False)
    start = make_start(
        ("first", "second"), attempt_deadline_ms=A_DEADLINE_A_READ_ARRIVES_AFTER_MS
    )
    root = a_run_directory(tmp_path, start)
    live, armed = await a_run_stopped_with_its_clock_running(root, start)
    assert live[0].state == "active"

    asked = reader_module._query

    async def query_once_the_clock_has_run_out(
        client: Client, workflow_id: str
    ) -> List[AttemptRecord]:
        """Hold the Query, which holds the read's Worker up, until the deadline has passed."""
        await past(armed, A_DEADLINE_A_READ_ARRIVES_AFTER_MS)
        return await asked(client, workflow_id)

    before = database_bytes(root)
    monkeypatch.setattr(reader_module, "_query", query_once_the_clock_has_run_out)
    with pytest.raises(ReadRefused) as refused:
        await read_records(root)
    assert "moved it" in str(refused.value)

    assert before
    assert database_bytes(root) == before


@pytest.mark.network
async def test_a_run_on_a_named_service_is_read_through_whoever_is_serving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A service somebody else runs is read as a client of it, and never as a Worker on it.

    There is no copy to take here, so every protection a copy gave is gone. A Worker started
    against this address would poll the live generation's own queue, and what the server would
    hand it is that generation's own unapplied work, applied in the authority rather than in a
    scratch directory. The generation here is stopped the way a crash stops one: the environment
    answered its seal into a history whose workflow half had gone, so the next step is ready and
    the only thing missing is somebody to run it.

    Both answers are asked for. While the deployment is serving, the read is answered by it and
    starts nothing to be answered, which is what the stand-in Worker here is watching for. Once
    nothing is serving, the Query reaches nobody, and that is reported as a refusal rather than
    waited on for ever or made false by starting the Worker that would have applied that step.
    """
    environment = _ENVIRONMENT.again()
    service = tmp_path / "service"
    service.mkdir()
    start = make_start(("first",))
    root = a_run_directory(tmp_path / "run", start)

    async with a_service_somebody_else_runs(service) as client:
        monkeypatch.setenv(TEMPORAL_ADDRESS_ENV, client.service_client.config.target_host)
        monkeypatch.setattr(reader_module, "stream_worker", no_worker_a_read_may_start)
        monkeypatch.setattr(
            reader_module,
            "_A_SERVING_WORKER_ANSWERS_WITHIN",
            timedelta(milliseconds=NOBODY_IS_SERVING_MS),
        )
        async with environment_worker(client):
            async with stream_worker(client, activities=[]):
                driver = await open_stream(client, start)
                task = await driver.take()
                assert task.attempt_id is not None
                served = await driver.records()
                # The deployment is up, so it answers, and the read started nothing to ask it.
                assert (await read_records(root)).records == served

                filing = asyncio.create_task(driver.file(task.attempt_id))
                await asyncio.wait_for(environment.reached.wait(), timeout=60)
                filing.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await filing
            environment.answers.set()
            interrupted = await durable_answer_nobody_applied(client)

        with pytest.raises(ReadRefused) as refused:
            await read_records(root)
        assert "no Worker serving" in str(refused.value)

        # The step nobody has applied is still nobody's, and the read did not become the one
        # that applied it.
        assert await history_events(client) == interrupted


async def test_a_run_directory_with_no_history_has_nothing_to_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest and blobs are not a record. Reading one says which half is missing.

    Nothing is started to find that out, which is the point: a directory served without keeping
    its history has nothing that could answer, and asking a service about it would only take
    longer to say so.
    """
    monkeypatch.delenv(TEMPORAL_ADDRESS_ENV, raising=False)
    a_run_directory(tmp_path, make_start(("first",)))
    with pytest.raises(NothingToRead) as refused:
        await read_records(tmp_path)
    assert "stream.sqlite" in str(refused.value)
    assert not (tmp_path / RECORDS_FILE).exists()


def test_the_table_prints_an_unsealed_attempt_as_absent_and_never_as_nought() -> None:
    """The score column is what a reader came for, so it must not invent one.

    The floor is the case that column cannot carry alone. An attempt that ended without a
    filing scores nothing, and so does one the environment graded at nothing, so the ending is
    printed beside the number and the two rows read differently.
    """
    table = format_records(
        [_record(0, "sealed"), _record(1, "planned"), _record(2, "ended")]
    ).splitlines()
    assert table[0].split() == [
        "task",
        "attempt",
        "state",
        "seal",
        "score",
        "ending",
        "decode",
        "delivered",
    ]
    assert table[1].split() == [
        "0",
        oid(0x100),
        "ack_presented",
        "1",
        "1.000",
        "-",
        "decoded",
        "task",
        "ack",
        "payload",
    ]
    assert table[2].split() == ["1", oid(0x104), "planned", "-", "-", "-", "-", "-"]
    assert table[3].split() == [
        "2",
        oid(0x108),
        "final_failed",
        "-",
        "0.000",
        "deadline",
        "-",
        "task",
    ]
    assert format_records([]) == "no attempts"


def _record(position: int, how: str) -> AttemptRecord:
    """One record of each kind, built here rather than served, so the table is what is tested."""
    sealed = how == "sealed"
    ended = how == "ended"
    return AttemptRecord(
        attempt_id=oid(0x100 + position * 4),
        task_position=position,
        payload_position=position,
        state={"sealed": "ack_presented", "planned": "planned", "ended": "final_failed"}[how],
        terminal_tool="submit" if sealed else None,
        canonicalization_version=CANONICALIZATION,
        submission_digest="a" * 64 if sealed else None,
        score=1.0 if sealed else (0.0 if ended else None),
        decode_state="decoded" if sealed else None,
        seal_ordinal=1 if sealed else None,
        final_failure="deadline" if ended else None,
        deadline_expired=False,
        task_message_id=oid(0x101 + position * 4),
        task_delivered=sealed or ended,
        ack_message_id=oid(0x102 + position * 4),
        ack_delivered=sealed,
        payload_message_id=oid(0x103 + position * 4),
        payload_delivered=sealed,
        creates_payload_obligation=True,
        payload_state={"sealed": "presented", "planned": "assigned", "ended": "final_failed"}[how],
    )
