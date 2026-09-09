"""A repair, end to end: the refusal, the rows it leaves, the claim, and the bytes going out.

Every test here drives the real ledger world and the real durable stream. A delivery whose
committed body has gone is refused with the bare token; the refusal is a row in this generation's
own record and the refused pull is retained by the transport; a controller puts the bytes back,
claims on the handle the paused gateway holds, and asks that gateway to send the same request
again; and the agent's own next pull collects the cell that was always selected.

What is under test on the durable side is the record: which operation failed, in which phase, for
which reason, attributed to what, and what became of it. Those rows are read out of the same query
as the attempts and the presentations, they cross a boundary in commit order, and a refusal and
the recovery that answers it are joined by an identity built from the operation rather than from
the transport call that carried it.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio.client import Client, WorkflowUpdateFailedError  # noqa: E402
from temporalio.converter import default as default_converter  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.envs.receipts.env_v1 import ReceiptsV1Env, sibling  # noqa: E402
from shogym.envs.receipts.generators.ledger import GENERATOR  # noqa: E402
from shogym.envs.receipts.protocol_v2 import (  # noqa: E402
    CANONICALIZATION_VERSION,
    RECEIPTS_GRADE,
    receipt_contract,
)
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    IMMEDIATE,
    RELEASE_AT_SEAL,
    TASK_FIRST,
    PullRequest,
    ReleasePlan,
    TerminalMetadata,
    pull_request_identity,
)
from shogym.serve.protocol_v2.artifact import (  # noqa: E402
    GRADED_CELL,
    PLACEBO_CELL,
    read_source_artifact,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    StreamGateway,
    install_policies,
    terminal_manifest,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    LEGACY_CARRIER_SCHEMA_VERSION,
    BlobsVerified,
    ConsumerClaim,
    GeneratePayloadBundleInput,
    OfferedMessage,
    PayloadBundle,
    SealRequest,
    StreamHandle,
    StreamStart,
    TaskItem,
    TerminalTool,
    assignments_for,
    configuration_hash,
    ownership_claim_operation_identity,
    protocol_error_code,
    VerifyBlobsInput,
    resume_stream,
    start_stream,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GENERATE_PAYLOAD_BUNDLE,
    VERIFY_BLOBS,
    generate_payload_bundle_activity,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel.messages import unpack_carrier  # noqa: E402
from shogym.serve.protocol_v2.kernel.workflow import StreamWorkflow  # noqa: E402
from shogym.serve.protocol_v2.reader import (  # noqa: E402
    receipt_availability,
    receipt_lifecycle,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    DELIVER,
    EXPERIMENT,
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    REGISTERED,
    PayloadDisposition,
    PolicyProvenance,
    descriptor_digests,
    roster_digest,
)
from tests._fixtures.policy_rows import registering_the_receipt  # noqa: E402
from tests._fixtures.receipts_bundle import private_bundle  # noqa: E402
from tests._fixtures.temporal_server import time_skipping_environment  # noqa: E402
from tests._fixtures.upstream_gate import environmental_skip  # noqa: E402

ATTEMPT = "b" * 32
SECOND = "e" * 32
TERMINAL = "submit_filing"
CONTRACT = "ledger_receipt"
EXECUTION = "execution-1"
CONVERTER = default_converter().payload_converter


def oid(value: int) -> str:
    return f"{value:032x}"


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
    """One served ledger world, with the store this run keeps its objects in."""
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


def activities_of(episode: ServedEpisode) -> List[Any]:
    """This environment's own Activities, over a route that reaches its one world."""
    _version, activities, _digest = episode.env.protocol_v2_terminal(
        lambda _attempt: (episode.env, episode.session_id)
    )
    return list(activities)


def start_for(
    episode: ServedEpisode,
    contract: Any,
    blobs: Optional[Path],
    *,
    second: bool = False,
) -> StreamStart:
    """One generation delivering a committed cell at position zero, and maybe at position one.

    The second position is a second receipt obligation of the same generation, which is what a
    continued execution owing two deliveries is made of: each owes its own check, and the first
    one passing says nothing about the second. A generation with two of them serves its tasks
    first, so that both positions are sealed and both deliveries owed at the same time, which is
    the state a boundary has to be crossed in for the case to be the case at all.
    """
    plan = (
        ReleasePlan(predicate=RELEASE_AT_SEAL, priority=TASK_FIRST, tie_key=BY_POSITION)
        if second
        else IMMEDIATE
    )
    items = [
        TaskItem(
            task_position=0,
            attempt_id=ATTEMPT,
            task_message_id=oid(0x101),
            ack_message_id=oid(0x102),
            payload_position=0,
            payload_message_id=oid(0x103),
            body=episode.env.describe("0").instructions,
        )
    ]
    rows = [
        PayloadDisposition(
            attempt_id=ATTEMPT,
            payload_position=0,
            kind=DELIVER,
            policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
            cell=GRADED_CELL,
            resolution_source=REGISTERED,
            family_id=CONTRACT,
        )
    ]
    if second:
        items.append(
            TaskItem(
                task_position=1,
                attempt_id=SECOND,
                task_message_id=oid(0x105),
                ack_message_id=oid(0x106),
                payload_position=1,
                payload_message_id=oid(0x107),
                body=episode.env.describe("0").instructions,
            )
        )
        rows.append(
            PayloadDisposition(
                attempt_id=SECOND,
                payload_position=1,
                kind=DELIVER,
                policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                cell=GRADED_CELL,
                resolution_source=REGISTERED,
                family_id=CONTRACT,
            )
        )
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
        capacity=2 if second else 1,
        release=plan,
        assignments=assignments_for(items, plan),
        blob_root=None if blobs is None else str(blobs),
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


def a_legacy_start(episode: ServedEpisode, contract: Any, blobs: Path) -> StreamStart:
    """The same generation with none of the receipt configuration on it.

    It declares no contract and no source, and the body it owes is the scalar receipt a
    pre-artifact generation registers. What it keeps is the one thing this case needs: a policy
    descriptor among the objects its history cites, which is what a claim reads back.
    """
    return registering_the_receipt(
        replace(
            start_for(episode, contract, blobs),
            receipt_contracts=[],
            receipt_source="",
        ),
        grade=RECEIPTS_GRADE,
    )


class Caller:
    """One consumer of a generation, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: Any, cursor: str, blobs: FilesystemBlobStore) -> None:
        self.stream = stream
        self.cursor = cursor
        self._counter = 0
        self.transcript = blobs.put(b"the transcript", media_type="text/plain").sha256
        self.turn = blobs.put(b"the provider turn", media_type="text/plain").sha256
        self.checkpoint = blobs.put(b"the checkpoint", media_type="text/plain").sha256

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
        self.cursor = ack.cursor

    async def seal(self, filing: str, attempt_id: str = ATTEMPT) -> OfferedMessage:
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


async def opened(
    environment: Any, start: StreamStart, blobs: Path, workflow_id: str
) -> Caller:
    """Install what this generation's history will cite, then start it and claim it."""
    store = FilesystemBlobStore(blobs)
    if start.blob_root is not None:
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


def body_of(payload: OfferedMessage) -> str:
    """The body an offered payload carries, out of the bytes it would be presented as."""
    return json.loads(payload.visible_text)["body"]


async def rows_of(caller: Caller) -> List[Any]:
    """Every operation failure this generation has recorded, in commit order."""
    records = await caller.stream.handle.query(StreamWorkflow.generation_records)
    return list(records.operation_failures)


async def worked(
    environment: Any,
    start: StreamStart,
    blobs: Path,
    workflow_id: str,
    filing: str,
) -> Caller:
    """Take one attempt to its acknowledgement and present it, leaving the payload owed."""
    caller = await opened(environment, start, blobs, workflow_id)
    await caller.present(await caller.pull())
    await caller.present(await caller.seal(filing))
    return caller


async def handed_on(client: Client, workflow_id: str) -> StreamStart:
    """The start the running execution was handed, out of the command that started it."""
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    started = history.events[0].workflow_execution_started_event_attributes
    assert started.continued_execution_run_id, "this execution continued another"
    return CONVERTER.from_payload(started.input.payloads[0], StreamStart)


async def carried_row(client: Client, workflow_id: str, attempt_id: str = ATTEMPT) -> Any:
    """One carried attempt of the execution now running, with its source and its selection."""
    carried = await handed_on(client, workflow_id)
    assert carried.carry is not None
    projection = unpack_carrier(carried.carry, CONVERTER)
    [row] = [entry for entry in projection.attempts if entry.attempt_id == attempt_id]
    return row


def a_gateway(caller: Caller, episode: ServedEpisode, blobs: Path, start: StreamStart):
    """One transport over the stream this caller holds, serving that same generation."""
    spec = episode.describe()
    return StreamGateway(
        caller.stream,
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor=caller.cursor,
        generation=start,
        blobs=FilesystemBlobStore(blobs),
    )


async def refused_record(awaitable: Any) -> Any:
    """The protocol error record a refused transport call carries, as its whole text."""
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001 - the record is the assertion
        return json.loads(str(error))
    raise AssertionError("the call was accepted")


async def claimed(caller: Caller, start: StreamStart, claimant: str = "the-controller") -> Any:
    """Claim the generation on the handle this caller holds, the way a controller does."""
    state = await caller.stream.stream_state()
    return await caller.stream.claim_ownership(
        configuration_hash=configuration_hash(start),
        previous_epoch=state.ownership_epoch,
        claimant_id=claimant,
        reason="resume",
        restored_checkpoints={
            attempt_id: state.task_checkpoints[attempt_id]
            for attempt_id in state.restoration_required
        },
    )


async def test_a_refused_delivery_is_recovered_through_the_gateway_the_agent_is_calling(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The whole public channel of a repair, on the one route the bytes can take.

    The pull that recovers is this gateway's own rather than the controller's. A controller
    pulling on its own handle would reserve the offer for a request this transport does not hold,
    and this transport's next call, idle with a message pending, would be refused and could never
    collect it: those bytes would never reach the agent and would be committed anyway.

    The repair is of the selected body alone. The unselected eligible cell of the same source
    stays missing throughout, because derivability is a fork-time requirement and a delivery
    blocked on a body it will never serve is what keeping the cells out of a delivery's required
    set exists to prevent.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-gateway/1"
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(env, start, blobs, workflow_id, filing_of(world.env))
        before = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        row = await carried_row(env.client, workflow_id)
        selected = row.selected_body_reference
        # Read the way a restore reads it: the descriptor crosses as the mapping its own
        # canonical bytes are taken from, so this asks the boundary's own reader for it.
        unselected = read_source_artifact(row.source_artifact).cells[PLACEBO_CELL].sha256

        kept = store.read(selected)
        store.path_for(selected).unlink()
        store.path_for(unselected).unlink()
        refused_epoch = (await caller.stream.stream_state()).ownership_epoch

        gateway = a_gateway(caller, world, blobs, start)
        assert (await refused_record(gateway.pull({})))["code"] == "evidence_unavailable"
        [identity] = gateway.refused_pulls

        refusal_rows = await rows_of(caller)
        [refusal] = refusal_rows
        assert refusal.operation == identity
        assert (refusal.phase, refusal.reason, refusal.outcome) == (
            "continued_first_delivery",
            "unavailable_evidence",
            "refused",
        )
        assert (refusal.attempt_id, refusal.payload_position) == (ATTEMPT, 0)
        assert refusal.references == [selected]
        assert refusal.refused_epoch == refused_epoch
        assert refusal.generation == workflow_id
        assert refusal.recovered_epoch is None

        # The repair, the claim on the handle this gateway holds, and the recovery.
        store.put(kept, media_type="text/plain")
        receipt = await claimed(caller, start)
        assert receipt.ownership_epoch == refused_epoch + 1
        assert await gateway.recover_refused_pull(identity) is True

        # Nothing was presented by the recovery: the stream is still holding what it reserved.
        held = await caller.stream.stream_state()
        assert held.pending_message_id == oid(0x103)

        delivered = json.loads(await gateway.pull({}))
        after = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        recovered_rows = await rows_of(caller)
        settled = await caller.stream.stream_state()

    # The agent was handed the cell that was always selected, and the presentation is committed.
    assert delivered["kind"] == "payload"
    assert delivered["message_id"] == oid(0x103)
    assert sha256(delivered["body"].encode("ascii")).hexdigest() == selected
    assert settled.cursor != held.cursor
    assert settled.obligations[ATTEMPT] == "presented"
    # And the unselected eligible body is still missing, which never mattered to this delivery.
    assert not store.path_for(unselected).exists()

    # The recovery is separately recorded, against the identity the refusal was recorded under.
    assert [row.outcome for row in recovered_rows] == ["refused", "recovered"]
    recovery = recovered_rows[1]
    assert recovery.operation == identity
    assert recovery.phase == "continued_first_delivery"
    assert recovery.reason == "unavailable_evidence"
    assert (recovery.attempt_id, recovery.payload_position) == (ATTEMPT, 0)
    assert recovery.refused_epoch == refused_epoch
    assert recovery.recovered_epoch == refused_epoch + 1
    assert recovered_rows[0] == refusal

    # And the source, the grade, the disposition and the payload identity are where they were.
    [was] = [record for record in before if record.attempt_id == ATTEMPT]
    [now] = [record for record in after if record.attempt_id == ATTEMPT]
    assert (was.score, was.submission_digest, was.seal_ordinal) == (
        now.score,
        now.submission_digest,
        now.seal_ordinal,
    )
    assert (was.payload_policy, was.payload_disposition) == (
        now.payload_policy,
        now.payload_disposition,
    )
    assert was.payload_message_id == now.payload_message_id == oid(0x103)
    assert now.payload_delivered and not was.payload_delivered


async def test_the_original_exact_update_keeps_returning_the_refusal_it_was_answered_with(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """A recovery answers the operation again and never rewrites the answer it already gave.

    The refused Update stands for ever, carried across every later boundary. What recovers the
    operation is the same request under a stepped epoch reaching a different Update, and what is
    not that operation is a later request reusing its id under a moved cursor: the identity
    covers the whole envelope, so that one is a conflict rather than the pull that was refused.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-update/1"
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(env, start, blobs, workflow_id, filing_of(world.env))
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        row = await carried_row(env.client, workflow_id)
        kept = store.read(row.selected_body_reference)
        store.path_for(row.selected_body_reference).unlink()

        request = PullRequest(
            request_id=caller.next_id(), last_presented_cursor=caller.cursor
        )
        fenced_writer = caller.stream.writer
        with pytest.raises(WorkflowUpdateFailedError) as refusal:
            await caller.stream.pull(request)
        assert protocol_error_code(refusal.value) == "evidence_unavailable"

        store.put(kept, media_type="text/plain")
        await claimed(caller, start)
        payload = await caller.stream.pull(request)
        await caller.present(payload)

        # The Update the refused pull was made under, asked again exactly as it was asked.
        stale = StreamHandle(caller.stream.handle, fenced_writer)
        with pytest.raises(WorkflowUpdateFailedError) as again:
            await stale.pull(request)
        assert protocol_error_code(again.value) == "evidence_unavailable"

        # And the same logical id, under the cursor the delivery moved, is not that operation.
        moved = PullRequest(
            request_id=request.request_id, last_presented_cursor=caller.cursor
        )
        with pytest.raises(WorkflowUpdateFailedError) as conflict:
            await caller.stream.pull(moved)
        assert protocol_error_code(conflict.value) == "request_conflict"
        rows = await rows_of(caller)

    assert sha256(body_of(payload).encode("ascii")).hexdigest() == row.selected_body_reference
    # Two rows and no more: the refusal, and the recovery of that same operation. Neither the
    # replayed refusal nor the conflicting request is an operation of its own.
    assert [entry.outcome for entry in rows] == ["refused", "recovered"]
    assert rows[0].operation == rows[1].operation == pull_request_identity(request)
    assert rows[1].operation != pull_request_identity(moved)


async def resumed(client: Client, workflow_id: str, start: StreamStart, claimant: str) -> Any:
    """Take the generation over the way a replacement owner does, from outside its handle."""
    return await resume_stream(
        client,
        workflow_id=workflow_id,
        configuration_hash=configuration_hash(start),
        claimant_id=claimant,
    )


async def test_a_claim_refused_twice_is_one_operation_and_recovers_under_that_identity(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The claim identity, doing the job it exists for.

    The fencing token is outside it, so a claim repeated after a repair under a fresh token is
    the operation it repeats rather than a second one nothing joins to the first. Each refusal
    appends its own row, and the swap that finally happens appends the recovery against that same
    identity.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-claim/1"
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(env, start, blobs, workflow_id, filing_of(world.env))
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        row = await carried_row(env.client, workflow_id)
        assert await rows_of(caller) == []

        # The descriptor of a sealed receipt attempt, which a claim reads and a delivery does not.
        source = row.source_commitment
        kept = store.read(source)
        store.path_for(source).unlink()
        witnessed = (await caller.stream.stream_state()).ownership_epoch

        for _ in range(2):
            with pytest.raises(WorkflowUpdateFailedError) as refusal:
                await resumed(env.client, workflow_id, start, "a-replacement")
            assert protocol_error_code(refusal.value) == "invalid_message"
        refused_rows = await rows_of(caller)

        store.put(kept, media_type="text/plain")
        replacement = await resumed(env.client, workflow_id, start, "a-replacement")
        recovered_rows = await rows_of(caller)

    identity = ownership_claim_operation_identity("a-replacement", witnessed)
    assert [entry.outcome for entry in refused_rows] == ["refused", "refused"]
    for entry in refused_rows:
        assert entry.operation == identity
        assert entry.phase == "ownership_claim"
        assert entry.reason == "unavailable_evidence"
        assert entry.attempt_id == ATTEMPT
        assert entry.references == [source]
        # A claim carries no logical request of its own, and none is invented for it.
        assert entry.request_id is None and entry.payload_position is None
        assert entry.refused_epoch == witnessed
        assert entry.recovered_epoch is None

    assert [entry.outcome for entry in recovered_rows] == ["refused", "refused", "recovered"]
    recovery = recovered_rows[2]
    assert recovery.operation == identity
    assert recovery.attempt_id == ATTEMPT
    assert recovery.refused_epoch == witnessed
    assert recovery.recovered_epoch == replacement.writer.ownership_epoch == witnessed + 1


async def test_a_claim_refused_on_a_shared_descriptor_names_every_attempt_that_needs_it(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """A failure on an object no one attempt owns is attributed rather than assigned.

    The counterpart policy's descriptor is shared by every attempt under that contract: its
    preimage is what lets a reader say what the other cell was allowed to say. So the generation
    here seals two receipt attempts under one contract and both are served the graded cell, which
    makes the placebo descriptor an object neither of them owns and both of them require.

    A claim that cannot produce it therefore has to name both, and one row naming whichever
    attempt came first is the answer this case exists to refuse. What is left over is nothing:
    every reference lost was required by a sealed attempt, so no row is written to the operation
    alone.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs, second=True)
    workflow_id = "stream/recovery-shared/1"
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await opened(env, start, blobs, workflow_id)
        await caller.present(await caller.pull())
        await caller.present(await caller.seal(filing_of(world.env), ATTEMPT))
        await caller.present(await caller.pull())
        await caller.present(await caller.seal("no,rows,here", SECOND))
        witnessed = (await caller.stream.stream_state()).ownership_epoch
        shared = descriptor_digests([], [], [contract])

        counterpart = store.read(PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST)
        store.path_for(PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST).unlink()

        with pytest.raises(WorkflowUpdateFailedError) as refusal:
            await claimed(caller, start, "a-replacement")
        assert protocol_error_code(refusal.value) == "invalid_message"
        rows = await rows_of(caller)
        store.put(counterpart, media_type="text/plain")

    # The object is the contract's, not either attempt's: both were served the graded cell.
    assert PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST in shared
    assert GRADED_RECEIPT_ARTIFACT_V1_DIGEST in shared
    # One row per sealed attempt whose required set holds it, and nothing left over for the
    # operation alone.
    assert [entry.attempt_id for entry in rows] == [ATTEMPT, SECOND]
    for entry in rows:
        assert entry.references == [PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST]
        assert entry.phase == "ownership_claim"
        assert entry.outcome == "refused"
        assert entry.reason == "unavailable_evidence"
        assert entry.refused_epoch == witnessed
        assert entry.operation == ownership_claim_operation_identity(
            "a-replacement", witnessed
        )


async def test_a_claim_refused_before_any_receipt_has_sealed_is_the_operations_alone(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """Nothing has sealed, so there is no attempt whose required set could name the object.

    The row is the operation's own: it names no attempt rather than picking one, which is the
    same rule the shared-descriptor case follows with nothing left to attribute it to.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-unsealed/1"
    async with stream_worker(env.client, activities=activities_of(world)):
        install_policies(store, start)
        store.path_for(GRADED_RECEIPT_ARTIFACT_V1_DIGEST).unlink()
        with pytest.raises(WorkflowUpdateFailedError) as refusal:
            await start_stream(env.client, start, workflow_id=workflow_id)
        assert protocol_error_code(refusal.value) == "invalid_message"
        records = await env.client.get_workflow_handle(workflow_id).query(
            StreamWorkflow.generation_records
        )

    [entry] = records.operation_failures
    assert entry.attempt_id is None
    assert entry.payload_position is None
    assert entry.request_id is None
    assert entry.references == [GRADED_RECEIPT_ARTIFACT_V1_DIGEST]
    assert entry.phase == "ownership_claim"
    assert entry.outcome == "refused"
    # The first claim of a generation witnesses the epoch nobody has yet, which is zero.
    assert entry.refused_epoch == 0
    assert entry.operation == ownership_claim_operation_identity(workflow_id, 0)


async def test_a_generation_with_no_receipt_to_lose_records_none_of_this_and_keeps_its_carrier(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """A pre-artifact generation's refusal is the one it always was, and writes no row.

    Its history cites a policy descriptor like any other generation's, so the claim that cannot
    produce it is refused here too. What must not happen is a row: these are a receipt's own
    record of its evidence, and a generation with no receipt to lose has none to record.

    The consequence is the carrier. A single row would move this generation to the later number
    for the rest of its life, and the promise is that a generation declaring no receipt
    configuration hands on the earlier one at every boundary it crosses.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    start = a_legacy_start(world, contract_of(world.env), blobs)
    workflow_id = "stream/recovery-legacy/1"
    turnover_at(10_000)
    descriptor = start.dispositions[0].policy_digest or ""
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await opened(env, start, blobs, workflow_id)
        await caller.present(await caller.pull())

        kept = store.read(descriptor)
        store.path_for(descriptor).unlink()
        with pytest.raises(WorkflowUpdateFailedError) as refusal:
            await claimed(caller, start, "a-replacement")
        assert protocol_error_code(refusal.value) == "invalid_message"
        refused_rows = await rows_of(caller)

        store.put(kept, media_type="application/json")
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        carried = await handed_on(env.client, workflow_id)
        crossed_rows = await rows_of(caller)

    # The generation is a pre-artifact one, and the object it could not produce is a real one.
    assert start.receipt_contracts == [] and start.receipt_source == ""
    assert descriptor and sha256(kept).hexdigest() == descriptor
    # The refusal happened and left nothing behind, here or on the far side of the boundary.
    assert refused_rows == []
    assert crossed_rows == []
    assert carried.carry is not None
    assert carried.carry.carrier_schema_version == LEGACY_CARRIER_SCHEMA_VERSION
    assert unpack_carrier(carried.carry, CONVERTER).operation_failures == []


async def test_a_capture_that_could_not_publish_its_source_is_recorded_as_unrecoverable(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """Evidence loss with nothing left to repair from, which is the ending rather than a refusal.

    A generation that delivers a committed cell and was given nowhere to keep one is refused at
    its first seal, before a source is captured under it. The attempt is over: no acknowledgement,
    no delivery, and no filing that could be sent again, so the row says unrecoverable rather than
    holding a refusal open for a recovery that can never come.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    start = start_for(world, contract, None)
    workflow_id = "stream/recovery-unpublishable/1"
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await opened(env, start, blobs, workflow_id)
        await caller.present(await caller.pull())
        with pytest.raises(WorkflowUpdateFailedError):
            await caller.seal(filing_of(world.env))
        rows = await rows_of(caller)
        state = await caller.stream.stream_state()
        [record] = [
            row
            for row in await caller.stream.handle.query(StreamWorkflow.attempt_records)
            if row.attempt_id == ATTEMPT
        ]

    assert state.attempts[ATTEMPT] == "final_failed"
    assert record.final_failure == "seal_failed"
    assert record.failure_kind == "UnavailableEvidence"
    [entry] = rows
    assert entry.phase == "source_publication"
    assert entry.reason == "unavailable_evidence"
    assert entry.outcome == "unrecoverable"
    assert entry.attempt_id == ATTEMPT
    # A source publication has no payload position: what failed is the capture rather than a
    # delivery, and no candidate was ever asked for.
    assert entry.payload_position is None
    assert entry.request_id is not None
    assert entry.recovered_epoch is None


async def test_every_refusal_and_every_recovery_survives_a_boundary_in_commit_order(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The rows are a list and not a field, and the order is the order they were committed in.

    Successive refusals and recoveries of one operation each append rather than overwrite, so the
    sequence says what happened rather than only what is true now, and nothing is dropped when the
    generation hands itself to the next execution.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-order/1"
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(env, start, blobs, workflow_id, filing_of(world.env))
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        row = await carried_row(env.client, workflow_id)

        # A claim refused over the descriptor, and the claim that repeats it after the repair.
        witnessed = (await caller.stream.stream_state()).ownership_epoch
        descriptor = store.read(row.source_commitment)
        store.path_for(row.source_commitment).unlink()
        with pytest.raises(WorkflowUpdateFailedError):
            await claimed(caller, start)
        store.put(descriptor, media_type="text/plain")
        await claimed(caller, start)

        # Then a delivery refused over the body, and the same pull under a stepped epoch.
        body = store.read(row.selected_body_reference)
        store.path_for(row.selected_body_reference).unlink()
        request = PullRequest(
            request_id=caller.next_id(), last_presented_cursor=caller.cursor
        )
        with pytest.raises(WorkflowUpdateFailedError):
            await caller.stream.pull(request)
        store.put(body, media_type="text/plain")
        await claimed(caller, start)
        await caller.present(await caller.stream.pull(request))

        before = await rows_of(caller)
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        after = await rows_of(caller)
        carried = unpack_carrier(
            (await handed_on(env.client, workflow_id)).carry, CONVERTER
        )

    assert [(entry.phase, entry.outcome) for entry in before] == [
        ("ownership_claim", "refused"),
        ("ownership_claim", "recovered"),
        ("continued_first_delivery", "refused"),
        ("continued_first_delivery", "recovered"),
    ]
    assert before[0].operation == before[1].operation
    assert before[0].operation == ownership_claim_operation_identity(
        "the-controller", witnessed
    )
    assert before[2].operation == before[3].operation == pull_request_identity(request)
    # And the boundary carries them: the same rows, in the same order, out of the carrier and
    # out of the query the next execution answers.
    assert after == before
    assert carried.operation_failures == before


async def test_two_obligations_in_one_continued_execution_each_owe_their_own_check(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """One receipt passing its check waves nothing else through, which is why the set is indexed.

    A single flag would be set by the first receipt of a continued execution and would let the
    second past unchecked. Here the first delivery reads the store and is answered, the second's
    body is gone, and the second is refused on its own and recovered on its own.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs, second=True)
    workflow_id = "stream/recovery-two/1"
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await opened(env, start, blobs, workflow_id)
        await caller.present(await caller.pull())
        await caller.present(await caller.seal(filing_of(world.env), ATTEMPT))
        await caller.present(await caller.pull())
        await caller.present(await caller.seal("no,rows,here", SECOND))
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        first = await carried_row(env.client, workflow_id, ATTEMPT)
        second = await carried_row(env.client, workflow_id, SECOND)
        assert first.selected_body_reference != second.selected_body_reference

        kept = store.read(second.selected_body_reference)
        store.path_for(second.selected_body_reference).unlink()

        delivered = await caller.pull()
        await caller.present(delivered)
        request = PullRequest(
            request_id=caller.next_id(), last_presented_cursor=caller.cursor
        )
        with pytest.raises(WorkflowUpdateFailedError) as refusal:
            await caller.stream.pull(request)
        assert protocol_error_code(refusal.value) == "evidence_unavailable"

        store.put(kept, media_type="text/plain")
        await claimed(caller, start)
        recovered = await caller.stream.pull(request)
        await caller.present(recovered)
        rows = await rows_of(caller)

    assert delivered.attempt_id == ATTEMPT
    assert sha256(body_of(delivered).encode("ascii")).hexdigest() == (
        first.selected_body_reference
    )
    assert recovered.attempt_id == SECOND
    assert sha256(body_of(recovered).encode("ascii")).hexdigest() == (
        second.selected_body_reference
    )
    assert [(entry.attempt_id, entry.payload_position, entry.outcome) for entry in rows] == [
        (SECOND, 1, "refused"),
        (SECOND, 1, "recovered"),
    ]


async def test_a_delivery_whose_owner_was_replaced_while_it_read_records_nothing(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """A stale owner's late failure is not the last word on an attempt that has since delivered.

    The interleaving is the one an ownership transition makes possible. One epoch's pull asks the
    store for the body it is about to offer and the answer is slow. While it is out, a controller
    puts the bytes back, claims, and releases the old owner's lock; the new epoch's pull reads the
    repaired body and offers the receipt. Then the first pull's answer arrives, saying the object
    was missing when it looked.

    That call is fenced and its answer is about a state this generation has left. Recording it
    would leave a refusal as the last row naming an attempt whose delivery has already been
    verified and offered, and a reader would report the receipt unavailable on the strength of a
    reading nobody acted on. So the owner is checked before either outcome is written down.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-overtaken/1"
    turnover_at(10_000)

    reading = asyncio.Event()
    release = asyncio.Event()
    held: Dict[str, bool] = {"once": False}

    @activity.defn(name=VERIFY_BLOBS)
    async def slow_verification(payload: VerifyBlobsInput) -> BlobsVerified:
        """The kernel's own read, held open the first time it cannot produce something."""
        asked = list(payload.references)
        missing = FilesystemBlobStore(Path(payload.blob_root)).unverified(asked)
        if missing and not held["once"]:
            held["once"] = True
            reading.set()
            await release.wait()
        return BlobsVerified(
            verified=[name for name in asked if name not in missing], unverified=missing
        )

    activities = [row for row in activities_of(world) if row is not verify_blobs_activity]
    async with stream_worker(env.client, activities=[*activities, slow_verification]):
        caller = await worked(env, start, blobs, workflow_id, filing_of(world.env))
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        row = await carried_row(env.client, workflow_id)
        selected = row.selected_body_reference
        kept = store.read(selected)
        store.path_for(selected).unlink()

        stale = asyncio.ensure_future(
            caller.stream.pull(
                PullRequest(request_id=caller.next_id(), last_presented_cursor=caller.cursor)
            )
        )
        await asyncio.wait_for(reading.wait(), timeout=30)

        # The repair and the claim, which swaps the epoch and releases the lock the stale pull
        # is still holding.
        store.put(kept, media_type="text/plain")
        await claimed(caller, start, "a-replacement")
        delivered = await caller.pull()

        release.set()
        with pytest.raises(WorkflowUpdateFailedError) as fenced:
            await stale
        rows = await rows_of(caller)
        records = await caller.stream.handle.query(StreamWorkflow.generation_records)

    assert protocol_error_code(fenced.value) == "fenced_writer"
    assert delivered.kind == "payload"
    assert sha256(body_of(delivered).encode("ascii")).hexdigest() == selected
    # No row at all: the reading the fenced call came back with is about a state that is gone.
    assert rows == []
    [record] = [entry for entry in records.attempts if entry.attempt_id == ATTEMPT]
    assert receipt_availability(record, records.operation_failures) == "available"


async def test_a_claim_refused_on_a_presentation_reference_names_the_receipt_that_cited_it(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The objects an attempt's own presentations cited are objects that attempt requires.

    A claim reads every reference this generation's history cites, and the inventory it reads
    them out of is flat: it says an object is required and cannot say by whom. So a transcript
    blob that has gone would refuse the claim and be attributed to nobody, and the receipt whose
    own presentation cited it would go on reading available while the claim that needed it was
    being refused.

    The association is kept on the attempt for that reason, and this is what it buys: the row
    names the attempt, and the reader answers unavailable beside a lifecycle that still says what
    was committed.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-presentation/1"
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(env, start, blobs, workflow_id, filing_of(world.env))
        await caller.present(await caller.pull())
        witnessed = (await caller.stream.stream_state()).ownership_epoch

        kept = store.read(caller.transcript)
        store.path_for(caller.transcript).unlink()
        with pytest.raises(WorkflowUpdateFailedError) as refusal:
            await claimed(caller, start, "a-replacement")
        assert protocol_error_code(refusal.value) == "invalid_message"
        rows = await rows_of(caller)
        records = await caller.stream.handle.query(StreamWorkflow.generation_records)
        store.put(kept, media_type="text/plain")

    [entry] = rows
    assert entry.attempt_id == ATTEMPT
    assert entry.references == [caller.transcript]
    assert (entry.phase, entry.reason, entry.outcome) == (
        "ownership_claim",
        "unavailable_evidence",
        "refused",
    )
    assert entry.refused_epoch == witnessed
    assert entry.operation == ownership_claim_operation_identity("a-replacement", witnessed)
    # The commitment happened and no later loss unhappens it, so the two axes disagree.
    [record] = [row for row in records.attempts if row.attempt_id == ATTEMPT]
    assert receipt_lifecycle(record) == "committed"
    assert receipt_availability(record, records.operation_failures) == "unavailable"


async def test_a_resolver_that_could_not_produce_the_selected_body_names_it_in_the_row(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The row for a resolver's own phase names the object the operation could not produce.

    The source is published and verified before the payload Activity is asked for anything, so
    what fails here is the resolution of the one selected reference. That reference is what a
    controller has to go and repair, and it exists only inside the seal's own batch, so it is
    written down as the batch reaches the resolver rather than recovered afterwards from a
    transition that never ran.

    The ending is unrecoverable and that is what it is: there is no acknowledgement, no delivery
    and no filing that could be sent again.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = "stream/recovery-resolver/1"
    asked: List[str] = []

    @activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
    async def unavailable_body(payload: GeneratePayloadBundleInput) -> PayloadBundle:
        """A resolver that cannot produce the body it was handed the reference for."""
        assert payload.selected is not None
        asked.append(payload.selected.body_sha256)
        raise ApplicationError(
            "the store cannot produce the selected body",
            type="UnavailableEvidence",
            non_retryable=True,
        )

    activities = [
        row for row in activities_of(world) if row is not generate_payload_bundle_activity
    ]
    async with stream_worker(env.client, activities=[*activities, unavailable_body]):
        caller = await opened(env, start, blobs, workflow_id)
        await caller.present(await caller.pull())
        refused_epoch = (await caller.stream.stream_state()).ownership_epoch
        with pytest.raises(WorkflowUpdateFailedError):
            await caller.seal(filing_of(world.env))
        rows = await rows_of(caller)

    [selected] = asked
    [entry] = rows
    assert entry.attempt_id == ATTEMPT
    assert entry.payload_position == 0
    assert (entry.phase, entry.reason, entry.outcome) == (
        "payload_offer",
        "unavailable_evidence",
        "unrecoverable",
    )
    assert entry.references == [selected]
    assert entry.refused_epoch == refused_epoch
