"""What a receipt run comes to, read back out of the history that holds it.

Everything here is driven over the real ledger world and the real durable stream, and the answer
is asked of the generation rather than of anything a run wrote down as it went. Three claims are
under test.

A run answers after it has outlived the machinery that produced it. The world is closed, the
store its cells were installed in is deleted, the generation has crossed a boundary, and the
Worker that answers registers Activities that are fatal to call: the source, the selection, the
presentation and the grade or the reason coded failure all come back anyway, because the
descriptor and the failure rows are recorded state rather than objects to resolve.

Every row reads as exactly one lifecycle with an availability beside it. One generation holds a
delivered row, a commitment nobody received, a row whose evidence went missing afterwards, a
withheld row, a withheld row that lost its source, a row still waiting to seal, a position that
was never reached and a pre-artifact row that ended in a seal failure, and each is one value.

And one committed source serves one cell to the position that filed it while the next position
keeps its own source and its own grade and delivers nothing, once under each of the two policies,
with both eligible cells derivable from the one source either time.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio.client import Client, WorkflowUpdateFailedError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.service import RPCError, RPCStatusCode  # noqa: E402

from shogym.envs.receipts.env_v1 import ReceiptsV1Env, sibling  # noqa: E402
from shogym.envs.receipts.generators.ledger import GENERATOR  # noqa: E402
from shogym.envs.receipts.protocol_v2 import (  # noqa: E402
    CANONICALIZATION_VERSION,
    RECEIPTS_GRADE,
    receipt_contract,
)
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import (  # noqa: E402
    IMMEDIATE,
    PullRequest,
    TerminalMetadata,
)
from shogym.serve.protocol_v2.artifact import (  # noqa: E402
    GRADED_CELL,
    ORACLE_CELL,
    PLACEBO_CELL,
    mask_spans,
    masked_body,
    read_source_artifact,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.gateway import install_policies  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    ConsumerClaim,
    OfferedMessage,
    SealRequest,
    SourceOriginContext,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    assignments_for,
    configuration_hash,
    derived_selection,
    protocol_error_code,
    resume_stream,
    start_stream,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GENERATE_PAYLOAD_BUNDLE,
    GRADE_ATTEMPT,
    SEAL_ATTEMPT,
    VERIFY_BLOBS,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    BLINDED_RECEIPT_V1_DIGEST,
    DELIVER,
    EXPERIMENT,
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    REGISTERED,
    WITHHOLD,
    PayloadDisposition,
    PolicyProvenance,
    roster_digest,
)
from shogym.serve.protocol_v2.reader import (  # noqa: E402
    OPERATIONS_FILE,
    RECEIPT_ASSIGNED_BUT_UNSEALED,
    RECEIPT_AVAILABLE,
    RECEIPT_COMMITTED,
    RECEIPT_LEGACY,
    RECEIPT_UNAVAILABLE,
    RECEIPT_WITHHELD,
    RunRecords,
    observed_receipt,
    receipt_availability,
    receipt_lifecycle,
    write_records,
)
from tests._fixtures.receipts_bundle import private_bundle  # noqa: E402
from tests._fixtures.temporal_server import time_skipping_environment  # noqa: E402
from tests._fixtures.upstream_gate import environmental_skip  # noqa: E402

TERMINAL = "submit_filing"
CONTRACT = "ledger_receipt"
EXECUTION = "execution-1"
BODY_SIZE = 2657


def oid(value: int) -> str:
    return f"{value:032x}"


#: The eight rows of the lifecycle generation, in the order they are served.
PREARTIFACT = oid(0xA0)
DELIVERED = oid(0xA1)
FENCED = oid(0xA2)
LOST = oid(0xA3)
WITHHELD = oid(0xA4)
WITHHELD_LOST = oid(0xA5)
UNSEALED = oid(0xA6)
QUEUED = oid(0xA7)

#: The two rows of the route generation: the one that is served a cell, and the one that is not.
FILER = oid(0xB1)
SILENT = oid(0xB2)


@pytest.fixture(scope="module")
def frozen_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One admission bundle that actually verifies, shared by the module.

    This module's own copy of the process's verified template, under a room no other module
    writes into: what an environment opened on it forks and seals goes beside it.
    """
    return private_bundle(tmp_path_factory.mktemp("bundles"))


@pytest_asyncio.fixture
async def env() -> Any:
    try:
        environment = await time_skipping_environment()
    except Exception as error:  # noqa: BLE001 - an unusable server is the machine's, not the test's
        environmental_skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


@pytest_asyncio.fixture
async def world(frozen_bundle: Path, tmp_path: Path) -> Any:
    """One served ledger world, which a test may close before this fixture does."""
    episode = await ServedEpisode.start(
        "receipts_v1",
        task=0,
        env_config={"bundle": str(frozen_bundle)},
        ends_on_horizon=False,
        trace_path=tmp_path / "run.jsonl",
    )
    try:
        yield episode
    finally:
        with suppress(Exception):
            await episode.close()


@pytest.fixture
def turnover_at(monkeypatch: pytest.MonkeyPatch):
    """Lower the boundary trigger, which is the only way a test reaches one."""

    def lower(updates: int) -> None:
        monkeypatch.setattr(kernel_workflow, "TURNOVER_TRIGGER", updates)

    return lower


def contract_of(env_v1: ReceiptsV1Env, position: int = 0) -> Any:
    """The contract this family's cells are admitted and published under."""
    ordinal, index = env_v1._position(position)
    return receipt_contract(
        env_v1.instance(ordinal),
        sibling(index),
        contract_id=CONTRACT,
        cells=(
            (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
            (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
        ),
    )


def filing_of(env_v1: ReceiptsV1Env, position: int = 0) -> str:
    """The filing of an agent that applied the drawn convention exactly."""
    ordinal, index = env_v1._position(position)
    task = env_v1.instance(ordinal).side(sibling(index))
    return "\n".join(
        f"{identifier},{value}"
        for identifier, value in zip(GENERATOR.row_identifiers(task.table), task.key)
    )


def activities_of(episode: ServedEpisode, worldless: tuple = ()) -> List[Any]:
    """This environment's own Activities, over a route that reaches its one world.

    An attempt named in ``worldless`` resolves to no world at all, which is what a seal refuses
    rather than scoring a filing against an instance nobody looked up. It is how a row that ends
    in a seal failure is made without touching the environment's own code.
    """
    _version, activities, _digest = episode.env.protocol_v2_terminal(
        lambda attempt: None if attempt in worldless else (episode.env, episode.session_id)
    )
    return list(activities)


@activity.defn(name=SEAL_ATTEMPT)
async def _fatal_seal(request: Any) -> Any:
    raise ApplicationError("a read sealed an attempt", non_retryable=True)


@activity.defn(name=GRADE_ATTEMPT)
async def _fatal_grade(request: Any) -> Any:
    raise ApplicationError("a read graded an attempt", non_retryable=True)


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def _fatal_payload(request: Any) -> Any:
    raise ApplicationError("a read built a payload", non_retryable=True)


@activity.defn(name=VERIFY_BLOBS)
async def _fatal_verify(request: Any) -> Any:
    raise ApplicationError("a read went to the store", non_retryable=True)


#: What a Worker that may answer a Query and nothing else registers. They are fatal rather than
#: absent so that a read reaching for one fails loudly instead of waiting on an empty queue.
FATAL_ACTIVITIES = [_fatal_seal, _fatal_grade, _fatal_payload, _fatal_verify]


class Caller:
    """One consumer of a generation, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: Any, cursor: str, blobs: FilesystemBlobStore) -> None:
        self.stream = stream
        self.cursor = cursor
        self._counter = 0
        self.transcript = blobs.put(b"the transcript", media_type="text/plain").sha256
        self.turn = blobs.put(b"the provider turn", media_type="text/plain").sha256
        self.checkpoint = blobs.put(b"the checkpoint", media_type="text/plain").sha256
        # A hold on the acknowledgement, for the one case that needs this owner not to have its
        # answer yet. A presentation is committed before the transport has a result to give
        # anybody, and ``answered`` is set at exactly that point: the generation has moved its
        # cursor and this owner is still holding out its hand. Nothing after it runs until the
        # hold is released, so a test can put a fence between the commitment and the handoff.
        self.answered = asyncio.Event()
        self.hold: Optional[asyncio.Event] = None

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
            transcript_blob=self.transcript,
            provider_turn_blob=self.turn if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=self.checkpoint if message.kind == "task" else None,
        )
        hold = self.hold
        if hold is not None:
            self.answered.set()
            await hold.wait()
        self.cursor = ack.cursor

    async def seal(self, filing: str, attempt_id: str) -> OfferedMessage:
        return await self.stream.seal(
            SealRequest(
                metadata=TerminalMetadata(
                    request_id=self.next_id(),
                    last_presented_cursor=self.cursor,
                    attempt_id=attempt_id,
                ),
                public_tool_name=TERMINAL,
                native_terminal_name=TERMINAL,
                native_arguments={"filing": filing},
            )
        )


def a_task(attempt_id: str, position: int, body: str) -> TaskItem:
    """One roster row, with every identifier fixed from its position."""
    return TaskItem(
        task_position=position,
        attempt_id=attempt_id,
        task_message_id=oid(0x100 + position * 4),
        ack_message_id=oid(0x101 + position * 4),
        payload_position=position,
        payload_message_id=oid(0x102 + position * 4),
        body=body,
    )


def a_start(
    episode: ServedEpisode,
    contract: Any,
    blobs: Path,
    *,
    items: List[TaskItem],
    rows: List[PayloadDisposition],
    silent: List[str],
    capacity: int = 1,
) -> StreamStart:
    """One experiment generation over this world, declaring one receipt contract."""
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash="d" * 64,
        initial_cursor=oid(1),
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id=EXECUTION,
        canonicalization_version=CANONICALIZATION_VERSION,
        terminal_tool=TerminalTool(
            public_tool_name=TERMINAL,
            native_terminal_name=TERMINAL,
            argument_names=["filing"],
        ),
        tasks=items,
        capacity=capacity,
        assignments=assignments_for(items, IMMEDIATE, without_payload=silent),
        blob_root=str(blobs),
        profile=EXPERIMENT,
        grade=RECEIPTS_GRADE,
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_subject_of_this_run",
        ),
        receipt_contracts=[contract],
        receipt_source=episode.env._served.digest,
    )


def delivering(attempt_id: str, position: int, policy_digest: str, cell: str) -> PayloadDisposition:
    """A row that delivers one committed cell under the contract its capture is bound to."""
    return PayloadDisposition(
        attempt_id=attempt_id,
        payload_position=position,
        kind=DELIVER,
        policy_digest=policy_digest,
        cell=cell,
        resolution_source=REGISTERED,
        family_id=CONTRACT,
    )


def withholding(attempt_id: str, position: int) -> PayloadDisposition:
    """A row that captures under the contract and delivers nothing."""
    return PayloadDisposition(
        attempt_id=attempt_id,
        payload_position=position,
        kind=WITHHOLD,
        reason="this position is the filler of the arm",
        resolution_source=REGISTERED,
        family_id=CONTRACT,
    )


async def opened(environment: Any, start: StreamStart, blobs: Path, workflow_id: str) -> Caller:
    """Install what this generation's history will cite, then start it and claim it."""
    store = FilesystemBlobStore(blobs)
    install_policies(store, start)
    stream = await start_stream(environment.client, start, workflow_id=workflow_id)
    receipt = await stream.claim_consumer(
        ConsumerClaim(consumer_id="harness-1", claim_hash="d" * 64)
    )
    return Caller(stream, receipt.initial_cursor, store)


async def accepted_updates(client: Client, workflow_id: str) -> int:
    """How many Updates the execution this generation is in has accepted, from its history."""
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    return sum(
        1
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_WORKFLOW_EXECUTION_UPDATE_ACCEPTED
    )


async def cross_a_boundary(
    caller: Caller, turnover_at: Any, workflow_id: str, client: Client
) -> None:
    """Make this generation continue as new exactly once, from wherever it is now."""
    before = (await caller.stream.stream_state()).turnovers
    turnover_at(await accepted_updates(client, workflow_id) + 1)
    await caller.stream.confirm_state()
    for _ in range(400):
        if (await caller.stream.stream_state()).turnovers > before:
            turnover_at(10_000)
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the generation never crossed a boundary")


async def read_back(client: Client, workflow_id: str, root: Path) -> RunRecords:
    """Ask the generation for its rows, its commitments and its failures, and nothing else.

    The Worker that answers registers Activities that are fatal to call, so a read that reached
    for a seal, a grade, a renderer or the store would fail rather than quietly answer with what
    it produced itself.
    """
    async with stream_worker(client, activities=FATAL_ACTIVITIES, cached_workflows=0):
        answer = await asked(client, workflow_id)
    return RunRecords(
        root=root,
        workflow_id=workflow_id,
        records=list(answer.attempts),
        presentations=list(answer.presentations),
        operation_failures=list(answer.operation_failures),
    )


async def asked(client: Client, workflow_id: str) -> Any:
    """Ask this generation for its records, giving the Worker that just started time to poll.

    A Query needs a Worker to replay against, and this one has just been started, so a first
    attempt can reach the server before it is polling. That is a fact about starting a Worker
    rather than about the read, and it is waited out here rather than allowed to look like a
    generation that cannot answer.
    """
    for _ in range(10):
        try:
            return await client.get_workflow_handle(workflow_id).query(
                StreamWorkflow.generation_records, rpc_timeout=timedelta(seconds=5)
            )
        except RPCError as error:
            if error.status not in (RPCStatusCode.DEADLINE_EXCEEDED, RPCStatusCode.CANCELLED):
                raise
    raise AssertionError(f"no Worker answered for {workflow_id}")


def rows_of(run: RunRecords) -> Dict[str, Any]:
    """The exported lines by attempt, which is the file a reader outside this process holds."""
    path = write_records(run)
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return {line["attempt_id"]: line for line in lines}


async def worked(caller: Caller, filing: str, attempt_id: str, *, payload: bool) -> Optional[str]:
    """Take one attempt from its task to its acknowledgement, and collect its payload or not."""
    task = await caller.pull()
    assert task.kind == "task" and task.attempt_id == attempt_id
    await caller.present(task)
    await caller.present(await caller.seal(filing, attempt_id))
    if not payload:
        return None
    offered = await caller.pull()
    assert offered.kind == "payload" and offered.attempt_id == attempt_id
    await caller.present(offered)
    return offered.visible_text


def manifest_in(store: FilesystemBlobStore, commitment: str) -> Any:
    """The descriptor one commitment names, read back out of the store that installed it."""
    manifest = read_source_artifact(json.loads(store.read(commitment).decode("utf-8")))
    assert source_commitment(manifest) == commitment
    return manifest


async def test_a_receipt_run_is_read_and_exported_after_a_boundary_with_no_world_and_no_store(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The whole answer survives the machinery that produced it.

    The generation continues as new, the world it was worked in is closed, the store its cells
    and its descriptors were installed in is deleted, and the Worker that answers registers
    Activities that are fatal to call. What comes back is the source, the selection, the
    presentation and the grade, unchanged from what the same query said while all three were
    still there, because the descriptor is recorded state rather than a blob to resolve.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    filing = filing_of(world.env)
    body = world.env.describe("0").instructions
    items = [a_task(FILER, 0, body), a_task(SILENT, 1, body)]
    rows = [
        delivering(FILER, 0, GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
        withholding(SILENT, 1),
    ]
    start = a_start(world, contract, blobs, items=items, rows=rows, silent=[SILENT])
    workflow_id = "stream/read-back-continued/1"
    turnover_at(10_000)
    async with stream_worker(
        env.client, activities=activities_of(world), cached_workflows=0
    ):
        caller = await opened(env, start, blobs, workflow_id)
        visible = await worked(caller, filing, FILER, payload=True)
        await worked(caller, filing, SILENT, payload=False)
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)

    exports = tmp_path / "records"
    exports.mkdir()
    served = await read_back(env.client, workflow_id, exports)
    [filer] = [row for row in served.records if row.attempt_id == FILER]
    source = filer.source_provenance
    assert source is not None

    # What the store still holds says the answer is about the objects that were installed.
    store = FilesystemBlobStore(blobs)
    manifest = manifest_in(store, source.source_commitment)
    assert source.selected_body_reference == manifest.cells[GRADED_CELL].sha256
    assert store.read(source.selected_body_reference).decode("ascii") == json.loads(visible)["body"]

    # The filler's own descriptor goes missing on the far side of the boundary, and the claim
    # that reads it records why, so this read has a grade to report on one row and a reason
    # coded failure on the other.
    [silent_row] = [row for row in served.records if row.attempt_id == SILENT]
    assert silent_row.source_provenance is not None
    async with stream_worker(
        env.client, activities=activities_of(world), cached_workflows=0
    ):
        store.path_for(silent_row.source_provenance.source_commitment).unlink()
        with pytest.raises(WorkflowUpdateFailedError):
            await resume_stream(
                env.client,
                workflow_id=workflow_id,
                configuration_hash=configuration_hash(start),
                claimant_id="a-replacement",
            )

    # And now neither the world nor the rest of the store is there.
    await world.close()
    for digest in (source.source_commitment, source.selected_body_reference):
        store.path_for(digest).unlink()
    assert not store.holds(source.source_commitment)

    after = await read_back(env.client, workflow_id, exports)
    assert after.records == served.records
    assert after.presentations == served.presentations
    assert [(row.attempt_id, row.outcome, row.reason) for row in after.operation_failures] == [
        (SILENT, "refused", "unavailable_evidence")
    ]

    lines = rows_of(after)
    delivered, withheld = lines[FILER], lines[SILENT]
    assert delivered["receipt_lifecycle"] == RECEIPT_COMMITTED
    assert delivered["receipt_availability"] == RECEIPT_AVAILABLE
    assert delivered["source_provenance"]["source_commitment"] == source.source_commitment
    assert delivered["source_provenance"]["selected_cell"] == GRADED_CELL
    assert delivered["source_provenance"]["selected_body_reference"] == (
        manifest.cells[GRADED_CELL].sha256
    )
    assert delivered["source_provenance"]["selected_policy_digest"] == (
        GRADED_RECEIPT_ARTIFACT_V1_DIGEST
    )
    assert delivered["receipt_contract_id"] == CONTRACT
    assert delivered["score"] is not None and delivered["final_failure"] is None
    # The presentation joins by the payload's own message identity and its digest.
    assert delivered["payload_visible_sha256"] == sha256(visible.encode("utf-8")).hexdigest()
    [committed] = [
        row for row in after.presentations if row.message_id == delivered["payload_message_id"]
    ]
    assert committed.visible_bytes_sha256 == delivered["payload_visible_sha256"]
    # And the position that delivered nothing kept its source and its grade all the same.
    assert withheld["receipt_lifecycle"] == RECEIPT_WITHHELD
    assert withheld["receipt_availability"] == RECEIPT_UNAVAILABLE
    assert withheld["source_provenance"]["selected_cell"] is None
    assert withheld["score"] is not None

    # The episode behind that one word is exported beside the rows, member by member. The word
    # says the evidence is gone and this says which operation went looking, for what, and under
    # which epoch, which is what a controller reading this run afterwards acts on.
    [episode] = [
        json.loads(line)
        for line in (exports / OPERATIONS_FILE).read_text(encoding="utf-8").splitlines()
    ]
    [failure] = after.operation_failures
    assert episode == {
        "operation": failure.operation,
        "phase": "ownership_claim",
        "reason": "unavailable_evidence",
        "outcome": "refused",
        "generation": workflow_id,
        "refused_epoch": failure.refused_epoch,
        "attempt_id": SILENT,
        "payload_position": None,
        "references": [silent_row.source_provenance.source_commitment],
        "request_id": None,
        "recovered_epoch": None,
        "protocol_version": 2,
    }


def a_lifecycle_start(episode: ServedEpisode, contract: Any, blobs: Path) -> StreamStart:
    """One generation holding a row in every state a receipt run can leave one in.

    The pre-artifact row is the reason the roster is mixed: it names no contract and delivers a
    scalar receipt, which is what a generation built before any of this served, and its capture
    is refused because the seal resolves it to no world. Everything after it is under the
    contract, and the last two are never sealed at all.
    """
    body = episode.env.describe("0").instructions
    order = [
        PREARTIFACT,
        DELIVERED,
        FENCED,
        LOST,
        WITHHELD,
        WITHHELD_LOST,
        UNSEALED,
        QUEUED,
    ]
    items = [a_task(attempt_id, index, body) for index, attempt_id in enumerate(order)]
    silent = [WITHHELD, WITHHELD_LOST, QUEUED]
    rows = [
        PayloadDisposition(
            attempt_id=PREARTIFACT,
            payload_position=0,
            kind=DELIVER,
            policy_digest=BLINDED_RECEIPT_V1_DIGEST,
            cell=GRADED_CELL,
            resolution_source=REGISTERED,
        ),
        delivering(DELIVERED, 1, GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
        delivering(FENCED, 2, GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
        delivering(LOST, 3, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
        withholding(WITHHELD, 4),
        withholding(WITHHELD_LOST, 5),
        delivering(UNSEALED, 6, GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
        withholding(QUEUED, 7),
    ]
    return a_start(episode, contract, blobs, items=items, rows=rows, silent=silent)


async def test_every_row_reads_as_one_lifecycle_with_its_availability_beside_it(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """Eight rows, eight states, and two axes rather than one list.

    One list could carry neither question. A committed row whose evidence later went missing is
    committed and unavailable at once, a withheld row that lost a body has no selection for an
    unavailable predicate to be about, and a queued position before capture is neither. So each
    row here is asked for exactly one lifecycle value, and its availability is asked for beside
    that answer and never in place of it.

    The two evidence losses are the descriptors of one delivering row and one withholding row.
    They are that attempt's own object rather than a shared one, so the claim that goes looking
    afterwards names those two rows and no others.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    filing = filing_of(world.env)
    start = a_lifecycle_start(world, contract, blobs)
    workflow_id = "stream/read-back-lifecycles/1"
    exports = tmp_path / "records"
    exports.mkdir()
    async with stream_worker(
        env.client,
        activities=activities_of(world, worldless=(PREARTIFACT,)),
        cached_workflows=0,
    ):
        caller = await opened(env, start, blobs, workflow_id)
        # The pre-artifact row, whose seal reaches no world and ends the attempt.
        task = await caller.pull()
        assert task.attempt_id == PREARTIFACT
        await caller.present(task)
        with pytest.raises(WorkflowUpdateFailedError):
            await caller.seal(filing, PREARTIFACT)

        delivered = await worked(caller, filing, DELIVERED, payload=True)
        fenced = await worked(caller, filing, FENCED, payload=True)
        await worked(caller, filing, LOST, payload=True)
        await worked(caller, filing, WITHHELD, payload=False)
        await worked(caller, filing, WITHHELD_LOST, payload=False)
        # And one row that has been served its task and has not filed, which is what holds the
        # last position in the queue rather than letting it be reached.
        waiting = await caller.pull()
        assert waiting.attempt_id == UNSEALED
        await caller.present(waiting)

        state = await caller.stream.stream_state()
        served = await asked(env.client, workflow_id)

        # Another owner takes the generation over, so the transport that presented the fenced
        # row never gets to hand those bytes on: they were committed and nobody received them.
        await resume_stream(
            env.client,
            workflow_id=workflow_id,
            configuration_hash=configuration_hash(start),
            claimant_id="a-replacement",
            restored_checkpoints={
                attempt_id: state.task_checkpoints[attempt_id]
                for attempt_id in state.restoration_required
            },
        )

        # Then two descriptors go missing, and the claim that reads them says so.
        for attempt_id in (LOST, WITHHELD_LOST):
            [row] = [entry for entry in served.attempts if entry.attempt_id == attempt_id]
            assert row.source_provenance is not None
            store.path_for(row.source_provenance.source_commitment).unlink()
        with pytest.raises(WorkflowUpdateFailedError):
            await resume_stream(
                env.client,
                workflow_id=workflow_id,
                configuration_hash=configuration_hash(start),
                claimant_id="another-replacement",
            )

    run = await read_back(env.client, workflow_id, exports)
    lines = rows_of(run)
    assert [line["receipt_lifecycle"] for line in lines.values()] == [
        RECEIPT_LEGACY,
        RECEIPT_COMMITTED,
        RECEIPT_COMMITTED,
        RECEIPT_COMMITTED,
        RECEIPT_WITHHELD,
        RECEIPT_WITHHELD,
        RECEIPT_ASSIGNED_BUT_UNSEALED,
        RECEIPT_ASSIGNED_BUT_UNSEALED,
    ]
    assert [line["receipt_availability"] for line in lines.values()] == [
        RECEIPT_AVAILABLE,
        RECEIPT_AVAILABLE,
        RECEIPT_AVAILABLE,
        RECEIPT_UNAVAILABLE,
        RECEIPT_AVAILABLE,
        RECEIPT_UNAVAILABLE,
        RECEIPT_AVAILABLE,
        RECEIPT_AVAILABLE,
    ]

    # The pre-artifact row ended in a seal failure and reads as legacy rather than as a receipt
    # that failed: absence there predates the question rather than answering it.
    assert lines[PREARTIFACT]["final_failure"] == "seal_failed"
    assert lines[PREARTIFACT]["source_provenance"] is None
    assert lines[PREARTIFACT]["receipt_contract_id"] is None
    # The two that never sealed are one value between them: a delivering row and a withheld one
    # read alike before either has captured a source.
    for attempt_id in (UNSEALED, QUEUED):
        assert lines[attempt_id]["source_provenance"] is None
        assert lines[attempt_id]["receipt_contract_id"] == CONTRACT
        assert lines[attempt_id]["score"] is None
    assert lines[UNSEALED]["creates_payload_obligation"]
    assert not lines[QUEUED]["creates_payload_obligation"]
    # The withheld row that lost its source keeps its grade, its source and its lifecycle, which
    # is a state the old selection-required predicate could not reach at all.
    assert lines[WITHHELD_LOST]["score"] is not None
    assert lines[WITHHELD_LOST]["source_provenance"]["selected_cell"] is None
    # And the committed row that lost its descriptor is still committed: what was committed
    # happened, and no later loss unhappens it.
    assert lines[LOST]["payload_delivered"]
    assert lines[LOST]["source_provenance"]["selected_cell"] == PLACEBO_CELL

    # The rows the claim wrote name those two attempts and say why, in commit order.
    assert [(row.attempt_id, row.outcome, row.reason) for row in run.operation_failures] == [
        (LOST, "refused", "unavailable_evidence"),
        (WITHHELD_LOST, "refused", "unavailable_evidence"),
    ]

    # Both commitments are read the same way and only one of them was observed, because only one
    # of them reached a transport that wrote anything down.
    by_id = {row.attempt_id: row for row in run.records}
    witness = {by_id[DELIVERED].payload_message_id: sha256(delivered.encode("utf-8")).hexdigest()}
    assert observed_receipt(by_id[DELIVERED], witness)
    assert not observed_receipt(by_id[FENCED], witness)
    assert receipt_lifecycle(by_id[FENCED]) == RECEIPT_COMMITTED
    assert by_id[FENCED].payload_visible_sha256 == sha256(fenced.encode("utf-8")).hexdigest()
    assert by_id[FENCED].source_provenance.selected_cell == GRADED_CELL
    assert receipt_availability(by_id[FENCED], run.operation_failures) == RECEIPT_AVAILABLE


@pytest.mark.parametrize(
    "policy_digest, cell",
    [
        (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
        (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
    ],
)
async def test_one_source_serves_a_cell_and_leaves_the_next_position_its_own_source(
    env: Any, world: ServedEpisode, tmp_path: Path, policy_digest: str, cell: str
) -> None:
    """The small route, once under each policy: one filing, one source, one cell, one filler.

    The position that files is served exactly one committed cell of the source its own seal
    published, and the position after it captures a source of its own, records its grade and
    delivers nothing. Capture is never conditional on exposure, so the second row is a receipt
    run's row and not an absence, which is what a fork needs before any child policy exists.

    Both eligible cells come out of the one committed source either way. The derivation takes the
    descriptor, the identity it was sealed under, an allowed policy and a cell, and the store, and
    the two references it returns are the descriptor's own entries: which one an arm was served is
    the controller's choice and not a property of the source.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    filing = filing_of(world.env)
    body = world.env.describe("0").instructions
    items = [a_task(FILER, 0, body), a_task(SILENT, 1, body)]
    rows = [delivering(FILER, 0, policy_digest, cell), withholding(SILENT, 1)]
    start = a_start(world, contract, blobs, items=items, rows=rows, silent=[SILENT])
    workflow_id = f"stream/read-back-route-{cell}/1"
    exports = tmp_path / "records"
    exports.mkdir()
    async with stream_worker(
        env.client, activities=activities_of(world), cached_workflows=0
    ):
        caller = await opened(env, start, blobs, workflow_id)
        visible = await worked(caller, filing, FILER, payload=True)
        await worked(caller, filing, SILENT, payload=False)

    run = await read_back(env.client, workflow_id, exports)
    lines = rows_of(run)
    filer, silent = lines[FILER], lines[SILENT]
    assert filer["receipt_lifecycle"] == RECEIPT_COMMITTED
    assert filer["source_provenance"]["selected_cell"] == cell
    assert filer["source_provenance"]["selected_policy_digest"] == policy_digest
    assert filer["receipt_contract_id"] == CONTRACT
    assert filer["payload_policy"] in (
        "graded-receipt-artifact-v1",
        "placebo-receipt-artifact-v1",
    )
    # The filler keeps its own source and its own grade, and names no selection at all.
    assert silent["receipt_lifecycle"] == RECEIPT_WITHHELD
    assert silent["receipt_contract_id"] == CONTRACT
    assert silent["score"] is not None
    assert silent["source_provenance"]["selected_cell"] is None
    assert silent["source_provenance"]["source_commitment"] != (
        filer["source_provenance"]["source_commitment"]
    )

    # One committed source, and both eligible cells derived from it.
    commitment = filer["source_provenance"]["source_commitment"]
    manifest = manifest_in(store, commitment)
    origin = SourceOriginContext(hidden_execution_id=EXECUTION, execution_ordinal=0)
    derived = {
        each: derived_selection(
            source=manifest,
            origin=origin,
            commitment=commitment,
            cell=each,
            policy_digest=digest,
            blob_root=str(blobs),
        )
        for each, digest in (
            (GRADED_CELL, GRADED_RECEIPT_ARTIFACT_V1_DIGEST),
            (PLACEBO_CELL, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST),
        )
    }
    bodies = {each: store.read(reference.body_sha256) for each, reference in derived.items()}
    assert [reference.body_sha256 for reference in derived.values()] == [
        manifest.cells[GRADED_CELL].sha256,
        manifest.cells[PLACEBO_CELL].sha256,
    ]
    assert bodies[GRADED_CELL] != bodies[PLACEBO_CELL]
    assert {len(each) for each in bodies.values()} == {BODY_SIZE}
    spans = mask_spans(contract)
    assert masked_body(bodies[GRADED_CELL], spans) == masked_body(bodies[PLACEBO_CELL], spans)

    # And what the agent was handed is the one cell this row's policy declares.
    assert json.loads(visible)["body"] == bodies[cell].decode("ascii")
    assert filer["source_provenance"]["selected_body_reference"] == derived[cell].body_sha256
    assert filer["payload_visible_sha256"] == sha256(visible.encode("utf-8")).hexdigest()
    # The oracle is named by the source, retained beside the rest, and derivable by nobody.
    assert ORACLE_CELL in manifest.cells
    with pytest.raises(Exception, match="an arm is served"):
        derived_selection(
            source=manifest,
            origin=origin,
            commitment=commitment,
            cell=ORACLE_CELL,
            policy_digest=policy_digest,
            blob_root=str(blobs),
        )


async def test_a_commitment_the_owner_was_fenced_before_handing_on_is_committed_and_unobserved(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The bytes were committed and nobody received them, which is two facts and not one.

    A presentation is a commitment rather than a handoff. It is accepted before the transport has
    a result to give anybody, so an acknowledgement can be lost and the owner replaced before the
    model sees anything. This cuts exactly there, and the cut is a fence rather than an omission:
    the payload goes out through the same hand-on this run's other two messages go through, so an
    acknowledgement that got back to this owner would be written into its record like any other.

    What stops it is the hold. The generation commits the presentation, moves its cursor and
    answers, this owner is held with that answer still in its hand, and the replacement claims
    while it is held. Releasing the hold before the claim instead would complete the handoff, put
    the entry in the record and make the same row read observed, which is what makes the answer
    below a consequence of the fence rather than of how the test was written.

    What the reader says afterwards is that the receipt is committed, with the cell it was served
    and the digest the generation stands behind. What it does not say is that the receipt was
    observed, and the reason is the record itself: the harness's transcript snapshot is what this
    transport actually wrote as it went, and there is no entry in it for that message.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    filing = filing_of(world.env)
    body = world.env.describe("0").instructions
    items = [a_task(DELIVERED, 0, body)]
    rows = [delivering(DELIVERED, 0, GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL)]
    start = a_start(world, contract, blobs, items=items, rows=rows, silent=[])
    workflow_id = "stream/read-back-fenced/1"
    exports = tmp_path / "records"
    exports.mkdir()
    # What the harness wrote down, entry by entry, as each answer came back to it.
    written: Dict[str, str] = {}

    async with stream_worker(
        env.client, activities=activities_of(world), cached_workflows=0
    ):
        caller = await opened(env, start, blobs, workflow_id)

        async def hand_on(message: OfferedMessage) -> None:
            """Present one message and write it into the harness's own record, as a run does."""
            await caller.present(message)
            written[message.message_id] = sha256(
                message.visible_text.encode("utf-8")
            ).hexdigest()

        await hand_on(await caller.pull())
        await hand_on(await caller.seal(filing, DELIVERED))

        offered = await caller.pull()
        assert offered.kind == "payload"
        # The same hand-on the other two went through, with its acknowledgement held. If the
        # answer reached this owner the entry would be written, so the hold is what puts the
        # fence between the commitment and the handoff rather than the shape of this test.
        caller.hold = asyncio.Event()
        handing_on = asyncio.ensure_future(hand_on(offered))
        await asyncio.wait_for(caller.answered.wait(), timeout=30)
        assert (await caller.stream.stream_state()).cursor == offered.message_id
        assert offered.message_id not in written

        # The commitment stands and this owner is replaced while its answer is still in its hand.
        await resume_stream(
            env.client,
            workflow_id=workflow_id,
            configuration_hash=configuration_hash(start),
            claimant_id="a-replacement",
        )
        # And it goes away holding it, so the entry is one nothing ever wrote.
        handing_on.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await handing_on
        assert offered.message_id not in written
        # Nothing this owner sends now is accepted, which is what makes the missing entry final.
        with pytest.raises(WorkflowUpdateFailedError) as fenced:
            await caller.pull()
        assert protocol_error_code(fenced.value) == "fenced_writer"

    run = await read_back(env.client, workflow_id, exports)
    [record] = [row for row in run.records if row.attempt_id == DELIVERED]
    assert receipt_lifecycle(record) == RECEIPT_COMMITTED
    assert receipt_availability(record, run.operation_failures) == RECEIPT_AVAILABLE
    assert record.source_provenance is not None
    assert record.source_provenance.selected_cell == GRADED_CELL
    assert record.payload_visible_sha256 == sha256(
        offered.visible_text.encode("utf-8")
    ).hexdigest()
    [committed] = [
        row for row in run.presentations if row.message_id == record.payload_message_id
    ]
    assert committed.visible_bytes_sha256 == record.payload_visible_sha256

    # The snapshot holds what this transport handed on and nothing about the message it did not,
    # so the receipt is committed and not observed.
    assert record.payload_message_id not in written
    assert len(written) == 2
    assert not observed_receipt(record, written)
    # And with the entry the handoff would have written, the same row is observed: what the
    # answer turns on is the harness's own record rather than anything the generation knows.
    assert observed_receipt(
        record, {**written, record.payload_message_id: record.payload_visible_sha256}
    )
