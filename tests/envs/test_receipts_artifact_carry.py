"""One committed source, carried across a boundary, and derived from long after the world is gone.

The route tests show a source published and one cell delivered inside one execution. What is
under test here is what happens when the generation outlives that execution. The descriptor, its
commitment, the identity it was sealed under and the selection made from it become attempt state,
cross in the carrier and are checked again on the far side, and the delivery that follows reads
the store once before it hands anything over.

Two things this file is careful about. Everything is driven over the real ledger world, so the
bodies are the environment's own bytes and the numbers are that bank's. And the derivation at the
end runs with the world closed and every call that could rebuild a body made fatal, because the
claim being tested is that a candidate for the opposite eligible cell needs the committed source
and nothing else.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.api.common.v1 import Payload  # noqa: E402
from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio.client import Client, WorkflowUpdateFailedError  # noqa: E402
from temporalio.converter import (  # noqa: E402
    DataConverter,
    DefaultPayloadConverter,
    default as default_converter,
)
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.service import RPCError  # noqa: E402
from temporalio.testing import ActivityEnvironment  # noqa: E402

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
    PresentationAck,
    PullRequest,
    TerminalMetadata,
    assignment_id_for,
)
from shogym.serve.protocol_v2.artifact import (  # noqa: E402
    GRADED_CELL,
    ORACLE_CELL,
    PLACEBO_CELL,
    SourceArtifactManifest,
    mask_spans,
    masked_body,
    payload_wire_count,
    read_source_artifact,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    EnvironmentTerminal,
    WorldRoute,
    attach_gateway,
    install_policies,
)
from shogym.serve.protocol_v2.rundir import (  # noqa: E402
    open_run_directory,
    run_directory_of,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    CONFIRMED_EXISTING,
    EXISTENCE_UNCONFIRMED,
    FORK_COMPLETE,
    FORK_CONFLICTED,    FORK_CARRIER_SCHEMA_VERSION,
    FORK_EXPIRED_AUTHORITY,
    FORK_NOT_QUIET,
    FORK_PREPARED,
    FORK_REPAIRABLE_ABSENCE,
    FORK_REQUEST_CONFLICT,
    FORK_WITNESS_MISMATCH,
    NEVER_ATTEMPTED,
    RECEIPT_CARRIER_SCHEMA_VERSION,
    BlobsVerified,
    ConsumerClaim,
    ChildReady,
    ForkAvailability,
    ForkAvailabilityInput,
    ForkChildPlan,
    ForkChildStarted,
    ForkPreparation,
    ForkPreparationInput,
    ForkRequest,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    GradeAttemptResult,
    OfferedMessage,
    PayloadBundle,
    SealAttemptInput,
    SealAttemptResult,
    SealRequest,
    SourceOriginContext,
    StartForkChildInput,
    StreamStart,
    TaskItem,
    TerminalTool,
    VerifyBlobsInput,
    assignments_for,
    configuration_hash,
    child_workflow_id,
    complete_start_digest,
    derived_selection,
    fork_availability_activity,
    fork_barrier_stands,
    fork_preparation_activity,
    fork_preparation_operation_identity,
    fork_can_be_retried,
    fork_refusal,
    fork_status,
    fork_stream,
    generate_payload_bundle_activity,
    origin_digest,
    origin_unverified,
    protocol_error_code,
    resume_stream,
    start_fork_child_activity,
    start_stream,
    stream_worker,
    verified_set_digest,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    FORK_AVAILABILITY,
    GENERATE_PAYLOAD_BUNDLE,
    GRADE_ATTEMPT,
    PREPARE_FORK_CHILD,
    SEAL_ATTEMPT,
    START_FORK_CHILD,
    STREAM_WORKFLOW_TYPE,
    VERIFY_BLOBS,
)
from shogym.serve.protocol_v2.kernel import activities as kernel_activities  # noqa: E402
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.workflow import (  # noqa: E402
    TURNOVER_PAYLOAD_CEILING_BYTES,
    StreamWorkflow,
)
from shogym.serve.protocol_v2.reader import (  # noqa: E402
    RECEIPT_AVAILABLE,
    RECEIPT_UNAVAILABLE,
    receipt_availability,
)
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    FORK_ORIGIN_DISAGREEMENT,    CONTRACT_DRIFT,
    FORK_CONTRACT_DRIFT,
    FORK_PREPARATION,
    FORK_WRONG_SOURCE,
    RECOVERED_OPERATION,
    REFUSED_OPERATION,
    UNAVAILABLE_EVIDENCE,
    UNRECOVERABLE_OPERATION,
    WRONG_SOURCE,
    CarriedAttempt,
    read_source_origin,
    unpack_carrier,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
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
from tests._fixtures.receipts_bundle import private_bundle  # noqa: E402
from tests._fixtures.temporal_server import time_skipping_environment  # noqa: E402
from tests._fixtures.upstream_gate import environmental_skip  # noqa: E402

ATTEMPT = "b" * 32
SILENT = "c" * 32
TERMINAL = "submit_filing"
CONTRACT = "ledger_receipt"
EXECUTION = "execution-1"
CONVERTER = default_converter().payload_converter

BODY_SIZE = 2657


def oid(value: int) -> str:
    return f"{value:032x}"


def carried_source(row: CarriedAttempt) -> SourceArtifactManifest:
    """The descriptor one carried row holds, read the way a restore reads it.

    It crosses as the mapping its own canonical bytes are taken from rather than as a typed
    field, so that a member this build does not declare is refused instead of being dropped by
    the decoder. A test asking what crossed therefore asks the same strict reader the far side
    asks, and gets the same answer or the same refusal.
    """
    assert row.source_artifact is not None
    return read_source_artifact(row.source_artifact)


def carried_origin(row: CarriedAttempt) -> SourceOriginContext:
    """The identity one carried row says its seal was minted under, read the same way."""
    assert row.source_origin is not None
    return read_source_origin(row.source_origin)


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
    blobs: Path,
    *,
    policy_digest: str = GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    cell: str = GRADED_CELL,
    silent: bool = False,
) -> StreamStart:
    """One generation delivering one committed cell, with every identifier fixed here."""
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
            policy_digest=policy_digest,
            cell=cell,
            resolution_source=REGISTERED,
            family_id=CONTRACT,
        )
    ]
    if silent:
        items.append(
            TaskItem(
                task_position=1,
                attempt_id=SILENT,
                task_message_id=oid(0x105),
                ack_message_id=oid(0x106),
                payload_position=1,
                payload_message_id=oid(0x107),
                body=episode.env.describe("0").instructions,
            )
        )
        rows.append(
            PayloadDisposition(
                attempt_id=SILENT,
                payload_position=1,
                kind=WITHHOLD,
                reason="this position is the filler of the arm",
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
        assignments=assignments_for(
            items, IMMEDIATE, without_payload=[SILENT] if silent else []
        ),
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


class Caller:
    """One consumer of a generation, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: Any, cursor: str, blobs: FilesystemBlobStore) -> None:
        self.stream = stream
        self.cursor = cursor
        self._counter = 0
        self.transcript = blobs.put(b"the transcript", media_type="text/plain").sha256
        self.turn = blobs.put(b"the provider turn", media_type="text/plain").sha256
        self.checkpoint = blobs.put(b"the checkpoint", media_type="text/plain").sha256
        # The last presentation and what it was answered with, which is the stream's half of a
        # freeze: the message that went, the attestation that committed it, and the cursor and
        # projection digest after it.
        self.presented: Optional[OfferedMessage] = None
        self.attested = ""
        self.acknowledged: Optional[PresentationAck] = None

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x1000 + self._counter)

    async def pull(self) -> OfferedMessage:
        return await self.stream.pull(
            PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )

    async def present(self, message: OfferedMessage) -> None:
        attestation = self.next_id()
        ack = await self.stream.present(
            message,
            attestation_id=attestation,
            transcript_blob=self.transcript,
            provider_turn_blob=self.turn if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=self.checkpoint if message.kind == "task" else None,
        )
        self.cursor = ack.cursor
        self.presented, self.attested, self.acknowledged = message, attestation, ack

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
    await hand_it_on(caller.stream, turnover_at, workflow_id, client)


async def hand_it_on(
    stream: Any, turnover_at: Any, workflow_id: str, client: Client
) -> None:
    """The same crossing for a generation nobody is pulling from yet, driven by its owner.

    A child that has claimed and has not prepared is quiet by every clause a boundary reads, so it
    crosses one like any other generation. That is the entry the gate its parent wrote is designed
    to survive, and it is where a preparation refused on the near side has to be recovered from on
    the far one.
    """
    before = (await stream.stream_state()).turnovers
    turnover_at(await accepted_updates(client, workflow_id) + 1)
    await stream.confirm_state()
    for _ in range(400):
        if (await stream.stream_state()).turnovers > before:
            turnover_at(10_000)
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the generation never crossed a boundary")


async def handed_on(client: Client, workflow_id: str) -> StreamStart:
    """The start the running execution was handed, out of the command that started it."""
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    started = history.events[0].workflow_execution_started_event_attributes
    assert started.continued_execution_run_id, "this execution continued another"
    return CONVERTER.from_payload(started.input.payloads[0], StreamStart)


def body_of(payload: OfferedMessage) -> str:
    """The body an offered payload carries, out of the bytes it would be presented as."""
    return json.loads(payload.visible_text)["body"]


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


async def test_a_boundary_between_the_acknowledgement_and_the_payload_delivers_the_same_body(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The candidate, the reference, the body and the source are what an uninterrupted run had.

    The boundary falls exactly where it is most awkward: the seal has committed, the obligation
    holds bytes nobody has been offered, and the next thing the agent does is ask for them. What
    crosses is the candidate and the selection that says what it is a cell of, and the twin here
    is the same generation driven straight through without a boundary.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    filing = filing_of(world.env)
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        straight = await worked(
            env, start_for(world, contract, blobs), blobs, "stream/carry-straight/1", filing
        )
        uninterrupted = await straight.pull()

        interrupted = await worked(
            env, start_for(world, contract, blobs), blobs, "stream/carry-crossed/1", filing
        )
        await cross_a_boundary(interrupted, turnover_at, "stream/carry-crossed/1", env.client)
        carried = await handed_on(env.client, "stream/carry-crossed/1")
        after = await interrupted.pull()

    assert uninterrupted.kind == after.kind == "payload"
    assert body_of(after) == body_of(uninterrupted)
    assert after.visible_text == uninterrupted.visible_text
    assert after.message_id == uninterrupted.message_id == oid(0x103)

    # And the record that crossed says what those bytes are: one source, one selected cell, one
    # reference, and the candidate carried with them.
    assert carried.carry is not None
    assert carried.carry.carrier_schema_version == RECEIPT_CARRIER_SCHEMA_VERSION
    projection = unpack_carrier(carried.carry, CONVERTER)
    [row] = [attempt for attempt in projection.attempts if attempt.attempt_id == ATTEMPT]
    manifest = carried_source(row)
    assert row.source_commitment == source_commitment(manifest)
    assert carried_origin(row) == SourceOriginContext(
        hidden_execution_id=EXECUTION, execution_ordinal=0
    )
    assert row.receipt_contract_id == CONTRACT
    assert row.selected_cell == GRADED_CELL
    assert row.selected_body_reference == manifest.cells[GRADED_CELL].sha256
    assert row.selected_policy_digest == GRADED_RECEIPT_ARTIFACT_V1_DIGEST
    [owed] = [entry for entry in projection.obligations if entry.attempt_id == ATTEMPT]
    assert owed.candidate is not None
    assert owed.candidate.body == body_of(after)
    assert owed.candidate.inner_sha256 == row.selected_body_reference
    # The descriptor and the canonical submission are what a later claim reads the store for; the
    # bodies are not, because one attempt's cell must never block another attempt's delivery.
    assert row.source_commitment in projection.committed_blobs
    assert manifest.canonical_submission.sha256 in projection.committed_blobs
    assert manifest.cells[GRADED_CELL].sha256 not in projection.committed_blobs
    assert manifest.cells[ORACLE_CELL].sha256 not in projection.committed_blobs


async def test_a_source_and_its_selection_cross_before_and_after_the_candidate_is_retired(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The bytes stop crossing once they can never be offered again, and the record does not.

    This is why the selection is attempt state rather than something on the obligation: an
    obligation drops its candidate at the first boundary after a presentation, and a reference
    hung on the candidate would go with it. What is left afterwards still says which cell of
    which source this position was served.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env,
            start_for(world, contract, blobs),
            blobs,
            "stream/carry-retired/1",
            filing_of(world.env),
        )
        await cross_a_boundary(caller, turnover_at, "stream/carry-retired/1", env.client)
        holding = unpack_carrier(
            (await handed_on(env.client, "stream/carry-retired/1")).carry, CONVERTER
        )
        payload = await caller.pull()
        await caller.present(payload)
        await cross_a_boundary(caller, turnover_at, "stream/carry-retired/1", env.client)
        retired = unpack_carrier(
            (await handed_on(env.client, "stream/carry-retired/1")).carry, CONVERTER
        )

    [before] = [row for row in holding.attempts if row.attempt_id == ATTEMPT]
    [after] = [row for row in retired.attempts if row.attempt_id == ATTEMPT]
    assert before.source_artifact == after.source_artifact
    assert (before.source_commitment, before.selected_cell, before.selected_body_reference) == (
        after.source_commitment,
        after.selected_cell,
        after.selected_body_reference,
    )
    [holding_owed] = [row for row in holding.obligations if row.attempt_id == ATTEMPT]
    [retired_owed] = [row for row in retired.obligations if row.attempt_id == ATTEMPT]
    assert holding_owed.candidate is not None and holding_owed.state == "eligible"
    assert retired_owed.candidate is None and retired_owed.state == "presented"
    assert retired_owed.materialized


async def test_a_position_that_delivers_nothing_crosses_with_its_source_and_no_selection(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """Capture is never conditional on exposure, and a boundary does not make it so.

    The second position is worked, sealed and scored, delivers nothing and names the contract its
    capture was validated against. What it carries is a descriptor with no selection beside it,
    which is a state the restore has to admit rather than repair.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    filing = filing_of(world.env)
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await opened(
            env,
            start_for(world, contract, blobs, silent=True),
            blobs,
            "stream/carry-withheld/1",
            )
        await caller.present(await caller.pull())
        await caller.present(await caller.seal(filing))
        await caller.present(await caller.pull())
        task = await caller.pull()
        assert task.kind == "task" and task.attempt_id == SILENT
        await caller.present(task)
        await caller.present(await caller.seal(filing, attempt_id=SILENT))
        await cross_a_boundary(caller, turnover_at, "stream/carry-withheld/1", env.client)
        projection = unpack_carrier(
            (await handed_on(env.client, "stream/carry-withheld/1")).carry, CONVERTER
        )

    [row] = [attempt for attempt in projection.attempts if attempt.attempt_id == SILENT]
    manifest = carried_source(row)
    assert row.receipt_contract_id == CONTRACT
    assert row.source_commitment == source_commitment(manifest)
    assert sorted(manifest.cells) == sorted([GRADED_CELL, ORACLE_CELL, PLACEBO_CELL])
    assert (row.selected_cell, row.selected_body_reference, row.selected_policy_digest) == (
        None,
        None,
        None,
    )
    assert row.score is not None
    # And the position that did deliver kept its own selection beside it.
    [served] = [attempt for attempt in projection.attempts if attempt.attempt_id == ATTEMPT]
    assert served.selected_cell == GRADED_CELL


async def test_the_opposite_eligible_cell_is_derivable_after_the_generation_has_continued(
    env: Any,
    world: ServedEpisode,
    tmp_path: Path,
    turnover_at: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conformance requirement: one committed source, two derivations, no world left.

    The generation served the graded cell and crossed a boundary. The world is then closed, and
    every call that could produce a body some other way is made fatal: the environment's seal, its
    grade and the renderer any scalar policy would go through. What is left is the carried source
    and the run's own store, and that is enough to build the placebo candidate for the same
    position: the exact committed bytes, identical to the delivered body outside the registered
    slots, hashing to its own committed entry, and the same bytes again on a second call.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env,
            start_for(world, contract, blobs),
            blobs,
            "stream/carry-derive/1",
            filing_of(world.env),
        )
        await cross_a_boundary(caller, turnover_at, "stream/carry-derive/1", env.client)
        served = await caller.pull()
        await caller.present(served)
        carried = await handed_on(env.client, "stream/carry-derive/1")

    projection = unpack_carrier(carried.carry, CONVERTER)
    [row] = [attempt for attempt in projection.attempts if attempt.attempt_id == ATTEMPT]
    source = carried_source(row)

    # The world goes, and so does every route back to a rendered body.
    await world.close()
    from shogym.envs.receipts import protocol_v2 as receipts_protocol
    from shogym.serve.protocol_v2.kernel import activities as kernel_activities

    def fatal(*_arguments: Any, **_keywords: Any) -> Any:
        raise AssertionError("a derivation reaches no seal, no grade and no renderer")

    # The seal's own capture, its publication and the grade's verdict, plus the renderer every
    # scalar policy goes through. Nothing below may reach any of them.
    monkeypatch.setattr(receipts_protocol, "_filed", fatal)
    monkeypatch.setattr(receipts_protocol, "_published", fatal)
    monkeypatch.setattr(receipts_protocol, "_verdict", fatal)
    monkeypatch.setattr(kernel_activities, "render_body", fatal)

    async def derive(cell: str, policy: str) -> Any:
        reference = derived_selection(
            source=source,
            origin=carried_origin(row),
            commitment=row.source_commitment or "",
            cell=cell,
            policy_digest=policy,
            blob_root=str(blobs),
        )
        bundle = await generate_payload_bundle_activity(
            GeneratePayloadBundleInput(
                attempt_id=ATTEMPT,
                payload_position=0,
                payload_message_id=oid(0x103),
                submission_digest=source.kernel_submission_digest,
                policy_digest=policy,
                cell=cell,
                selected=reference,
            )
        )
        [candidate] = bundle.candidates
        return candidate

    placebo = await derive(PLACEBO_CELL, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST)
    assert placebo.inner_sha256 == source.cells[PLACEBO_CELL].sha256
    assert len(placebo.body.encode("ascii")) == BODY_SIZE == source.receipt_contract.body_size

    # It is the counterpart of the body that was delivered: identical outside the registered
    # slots, different inside them, and neither of them is the oracle.
    spans = mask_spans(source.receipt_contract)
    delivered = body_of(served).encode("ascii")
    derived = placebo.body.encode("ascii")
    assert derived != delivered
    assert masked_body(derived, spans) == masked_body(delivered, spans)
    assert sha256(masked_body(derived, spans)).hexdigest() == (
        source.pair_parity.masked_body_sha256
    )
    oracle = FilesystemBlobStore(blobs).read(source.cells[ORACLE_CELL].sha256).decode("ascii")
    assert placebo.body != oracle and body_of(served) != oracle

    # And it is stable: the same derivation asked again returns the same bytes.
    assert (await derive(PLACEBO_CELL, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST)) == placebo
    # The cell that was delivered is derivable from the same source too, with the same result.
    graded = await derive(GRADED_CELL, GRADED_RECEIPT_ARTIFACT_V1_DIGEST)
    assert graded.body == body_of(served)


@pytest.mark.parametrize("lost", ["manifest", "body"])
async def test_a_delivery_after_a_boundary_refuses_when_the_object_it_needs_is_gone(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any, lost: str
) -> None:
    """The barrier, under the same owner that captured the source.

    Nothing has been fenced and nothing has claimed: the generation simply continued, so the set
    that remembers which obligations have been verified in this execution is empty and the first
    offer of this one reads the store. The refusal carries the bare token and leaves the
    obligation unmarked, which is what makes the repair recoverable: the bytes go back and the
    agent's next pull is answered with the cell that was always selected.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env,
            start_for(world, contract, blobs),
            blobs,
            f"stream/carry-lost-{lost}/1",
            filing_of(world.env),
        )
        before = await caller.stream.stream_state()
        await cross_a_boundary(caller, turnover_at, f"stream/carry-lost-{lost}/1", env.client)
        projection = unpack_carrier(
            (await handed_on(env.client, f"stream/carry-lost-{lost}/1")).carry, CONVERTER
        )
        [row] = [attempt for attempt in projection.attempts if attempt.attempt_id == ATTEMPT]
        digest = (
            row.source_commitment if lost == "manifest" else row.selected_body_reference
        ) or ""
        kept = store.read(digest)
        store.path_for(digest).unlink()

        with pytest.raises(WorkflowUpdateFailedError) as raised:
            await caller.pull()
        assert protocol_error_code(raised.value) == "evidence_unavailable"

        # The grade, the source and the selection are all where they were.
        after = await caller.stream.stream_state()
        assert after.attempts == before.attempts
        assert after.obligations == before.obligations

        # And a repair is all it takes: the check runs again, because the refusal marked nothing.
        store.put(kept, media_type="text/plain")
        request = PullRequest(
            request_id=caller.next_id(), last_presented_cursor=caller.cursor
        )
        payload = await caller.stream.pull(request)
        answered = await caller.stream.stream_state()
        # One read per obligation per execution: the same request sent again is answered from
        # what it reserved, and nothing about it reaches the store a second time.
        again = await caller.stream.pull(request)
        settled = await caller.stream.stream_state()

    assert payload.kind == "payload"
    assert payload.message_id == oid(0x103)
    assert sha256(body_of(payload).encode("ascii")).hexdigest() == row.selected_body_reference
    assert again == payload
    assert settled.verification_batches == answered.verification_batches
    assert answered.verification_batches > after.verification_batches


@pytest.mark.parametrize("lost", ["manifest", "submission"])
async def test_an_ownership_claim_reads_the_descriptor_and_the_submission_of_a_sealed_receipt(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any, lost: str
) -> None:
    """Two objects join the inventory a claim reads, and the three cells do not.

    A claim already reads every reference this history cites, and a sealed receipt attempt adds
    its descriptor and its canonical submission to that list: a generation whose record cannot
    say what its cells were is not one to hand a new owner. The cells are deliberately absent from
    it, because requiring them would let one attempt's unselected body block another attempt's
    delivery, and the body a delivery depends on is read by the delivery that depends on it.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    start = start_for(world, contract, blobs)
    workflow_id = f"stream/carry-claim-{lost}/1"
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(env, start, blobs, workflow_id, filing_of(world.env))
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
        projection = unpack_carrier((await handed_on(env.client, workflow_id)).carry, CONVERTER)
        [row] = [attempt for attempt in projection.attempts if attempt.attempt_id == ATTEMPT]
        digest = (
            row.source_commitment
            if lost == "manifest"
            else carried_source(row).canonical_submission.sha256
        ) or ""
        assert digest in projection.committed_blobs
        store.path_for(digest).unlink()

        state = await caller.stream.stream_state()
        with pytest.raises(WorkflowUpdateFailedError) as raised:
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
        assert protocol_error_code(raised.value) == "invalid_message"
        # A refused claim leaves the epoch where it was, so the owner that sealed still holds the
        # generation and the delivery it owes is still there to be made once the object is back.
        assert (await caller.stream.stream_state()).ownership_epoch == state.ownership_epoch


@pytest.mark.parametrize(
    "cell,policy",
    [
        (GRADED_CELL, GRADED_RECEIPT_ARTIFACT_V1_DIGEST),
        (PLACEBO_CELL, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST),
    ],
)
async def test_losing_only_the_oracle_never_stops_a_graded_or_a_placebo_delivery(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any, cell: str, policy: str
) -> None:
    """The oracle is required by publication and by nothing else, which is the point of saying so.

    It is retained, it is named by the descriptor, and no live arm verifies it. So a store that
    has lost it is a store a delivery proceeds over, whichever eligible cell the arm is serving:
    the alternative would be an arm blocked on a body it may never serve.
    """
    blobs = tmp_path / "blobs"
    store = FilesystemBlobStore(blobs)
    contract = contract_of(world.env)
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env,
            start_for(world, contract, blobs, policy_digest=policy, cell=cell),
            blobs,
            f"stream/carry-oracle-{cell}/1",
            filing_of(world.env),
        )
        await cross_a_boundary(caller, turnover_at, f"stream/carry-oracle-{cell}/1", env.client)
        projection = unpack_carrier(
            (await handed_on(env.client, f"stream/carry-oracle-{cell}/1")).carry, CONVERTER
        )
        [row] = [attempt for attempt in projection.attempts if attempt.attempt_id == ATTEMPT]
        store.path_for(carried_source(row).cells[ORACLE_CELL].sha256).unlink()
        payload = await caller.pull()
        await caller.present(payload)

    assert payload.kind == "payload"
    assert row.selected_cell == cell
    assert sha256(body_of(payload).encode("ascii")).hexdigest() == row.selected_body_reference


async def test_a_receipt_generation_continues_before_its_first_capture_and_seals_afterwards(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The number follows the declaration rather than the evidence, and the start survives it.

    A boundary forbids a sealing attempt, not an unsealed active one, so a receipt generation can
    reach one with its contract declared and nothing captured. The carrier it writes there is the
    later version, the start it hands on still declares the contract and the bank source, and the
    configuration identity rebuilt from that start is the one the carrier was composed against.
    Then it seals and delivers on the far side, which is what makes the continuation lawful rather
    than merely accepted.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await opened(
            env, start_for(world, contract, blobs), blobs, "stream/carry-early/1"
        )
        task = await caller.pull()
        await caller.present(task)
        await cross_a_boundary(caller, turnover_at, "stream/carry-early/1", env.client)
        carried = await handed_on(env.client, "stream/carry-early/1")

        assert carried.carry is not None
        assert carried.carry.carrier_schema_version == RECEIPT_CARRIER_SCHEMA_VERSION
        assert carried.receipt_source == world.env._served.digest
        assert [row.contract_id for row in carried.receipt_contracts] == [CONTRACT]
        assert carried.receipt_contracts[0] == contract
        projection = unpack_carrier(carried.carry, CONVERTER)
        assert configuration_hash(carried) == projection.configuration_hash
        assert all(row.source_artifact is None for row in projection.attempts)

        await caller.present(await caller.seal(filing_of(world.env)))
        payload = await caller.pull()

    assert payload.kind == "payload"
    assert len(body_of(payload).encode("ascii")) == BODY_SIZE


# The fork itself, driven as an Update against a real service at a real quiet boundary.
#
# A pure predicate test cannot see the ledger's own problem at all: the SDK inserts an Update into
# its in-progress map before that Update's validator runs and removes it in the handler's finally,
# so a request that waited for the public predicate would wait for itself for ever. That only
# happens when a real service accepts a real Update, which is why these are here.

FIRST_SLOT = "first"
SECOND_SLOT = "second"
FORK = "fork-1"
CELL_POLICIES = {
    GRADED_CELL: GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_CELL: PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
}


def fork_capable(start: StreamStart) -> StreamStart:
    """The same generation, declaring the two branches its fork may create."""
    return replace(start, forkable_slots=[FIRST_SLOT, SECOND_SLOT])


def plan_for(start: StreamStart, slot: str, cell: str, directory: Path) -> ForkChildPlan:
    """One child a controller asks for: its branch, its rows, its cell, its own identities."""
    rows = [
        replace(row, branch_slot=slot, policy_digest=CELL_POLICIES[cell], cell=cell)
        if row.kind == DELIVER
        else replace(row, branch_slot=slot)
        for row in start.dispositions
    ]
    return ForkChildPlan(
        branch_slot=slot,
        dispositions=rows,
        target_cell=cell,
        run_directory=str(directory),
        consumer_claim_hash=sha256(f"the consumer of {slot}".encode()).hexdigest(),
        hidden_execution_id=f"execution-{slot}",
        frozen_plan_digest=sha256(b"the plan both children are parked under").hexdigest(),
    )


async def a_fork_request(
    client: Client,
    caller: Caller,
    start: StreamStart,
    root: Path,
    *,
    fork_id: str = FORK,
    plans: Optional[List[ForkChildPlan]] = None,
) -> ForkRequest:
    """The typed fork for the generation this caller has just taken to its acknowledgement."""
    described = await client.get_workflow_handle(caller.stream.handle.id).describe()
    acknowledged = caller.presented
    assert acknowledged is not None and caller.acknowledged is not None
    return ForkRequest(
        parent_workflow_id=caller.stream.handle.id,
        parent_run_id=described.run_id,
        parent_execution_ordinal=start.execution_ordinal,
        parent_configuration_hash=configuration_hash(start),
        source_attempt_id=ATTEMPT,
        attestation_id=caller.attested,
        acknowledgement_message_id=acknowledged.message_id,
        acknowledged_visible_sha256=sha256(
            acknowledged.visible_text.encode("utf-8")
        ).hexdigest(),
        acknowledged_cursor=caller.acknowledged.cursor,
        projection_digest=caller.acknowledged.stream_state_sha256,
        checkpoint_manifest_reference=sha256(b"the checkpoint manifest").hexdigest(),
        fork_id=fork_id,
        child_plans=(
            plans
            if plans is not None
            else [
                plan_for(start, FIRST_SLOT, GRADED_CELL, root / "child-1"),
                plan_for(start, SECOND_SLOT, PLACEBO_CELL, root / "child-2"),
            ]
        ),
    )


async def scheduled_activities(client: Client, workflow_id: str) -> List[str]:
    """Every Activity identifier this execution scheduled, in the order it scheduled them."""
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    return [
        event.activity_task_scheduled_event_attributes.activity_id
        for event in history.events
        if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
    ]


async def completed_activities(client: Client, workflow_id: str) -> List[str]:
    """Every Activity this execution has finished, waited for so a race is not a flake."""
    for _ in range(500):
        history = await client.get_workflow_handle(workflow_id).fetch_history()
        started = {
            event.event_id: event.activity_task_scheduled_event_attributes.activity_id
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_SCHEDULED
        }
        done = [
            started[event.activity_task_completed_event_attributes.scheduled_event_id]
            for event in history.events
            if event.event_type == EventType.EVENT_TYPE_ACTIVITY_TASK_COMPLETED
        ]
        if done:
            return done
        await asyncio.sleep(0.02)
    raise AssertionError(f"{workflow_id} finished no Activity")


async def started_with(client: Client, workflow_id: str) -> StreamStart:
    """The start one execution was created with, out of the command that created it."""
    history = await client.get_workflow_handle(workflow_id).fetch_history()
    started = history.events[0].workflow_execution_started_event_attributes
    return CONVERTER.from_payload(started.input.payloads[0], StreamStart)


async def test_a_quiet_fork_prepares_two_children_over_one_committed_source(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The whole parent side, at a boundary the generation actually reached.

    The generation is taken to its acknowledgement and left there, which is the quiet point a fork
    is cut at: the payload is owed, nothing is pending, and the ledger of accepted handlers is
    empty by the time the fork's own validator runs. What comes back is the complete receipt, and
    what stands afterwards is a fenced parent that serves nothing and two children that exist,
    each carrying its own branch, its own cell and its own identities.

    The two children agree on the task neither has worked yet, and they agree by identity rather
    than by comparison: neither may change the manifest, the assignments or the release plan, so
    the receipt reads that task out of each child's own start and a reader can check the two.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs, silent=True))
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env, composed, blobs, "stream/fork-quiet/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        receipt = await fork_stream(env.client, request)

        assert receipt.fork_id == FORK
        assert receipt.children == 2
        assert [child.branch_slot for child in receipt.child_receipts] == [
            FIRST_SLOT,
            SECOND_SLOT,
        ]
        assert [child.target_cell for child in receipt.child_receipts] == [
            GRADED_CELL,
            PLACEBO_CELL,
        ]
        assert [child.child_ordinal for child in receipt.child_receipts] == [1, 2]
        assert all(child.child_run_id for child in receipt.child_receipts)
        identities = [child.child_workflow_id for child in receipt.child_receipts]
        assert identities == [
            child_workflow_id(
                identity_namespace="default",
                parent_workflow_id="stream/fork-quiet/1",
                fork_id=FORK,
                child_ordinal=ordinal,
            )
            for ordinal in (1, 2)
        ]
        # Same B, read out of each child rather than asserted about them, and named by the roster
        # row a reader joins this receipt to that child's own schedule on.
        [first, second] = receipt.child_receipts
        assert (
            first.next_assignment_id
            == second.next_assignment_id
            == assignment_id_for(SILENT)
        )
        assert first.next_assignment_id != SILENT
        assert first.next_task_body_sha256 == second.next_task_body_sha256
        assert first.next_task_body_sha256
        assert first.configuration_hash != second.configuration_hash

        # Both children exist, each holding its own lineage and the prefix its parent observed.
        starts = [await started_with(env.client, one) for one in identities]
        manifest = await a_committed_manifest(env.client, "stream/fork-quiet/1", blobs)
        for start, child in zip(starts, receipt.child_receipts):
            assert start.fork_origin is not None
            assert start.fork_origin.fork_id == FORK
            assert start.fork_origin.parent_workflow_id == "stream/fork-quiet/1"
            assert start.fork_origin.source_attempt_id == ATTEMPT
            # The lineage names the source and what the environment committed for that seal,
            # which is every cell of the descriptor and never the objects the presentations cited.
            assert start.fork_origin.source_commitment == source_commitment(manifest)
            assert start.fork_origin.source_artifact_references == sorted(
                reference.sha256 for reference in manifest.cells.values()
            )
            assert start.served_slot == child.branch_slot
            assert start.blob_root != composed.blob_root
            assert complete_start_digest(start) == child.complete_start_digest
        assert starts[0].tasks == starts[1].tasks == composed.tasks
        assert starts[0].assignments == starts[1].assignments == composed.assignments
        assert starts[0].release == starts[1].release == composed.release
        # And each one's inherited selection is the cell its own plan named.
        for start, cell in zip(starts, (GRADED_CELL, PLACEBO_CELL)):
            assert start.carry is not None
            projection = unpack_carrier(start.carry, CONVERTER)
            [row] = [one for one in projection.attempts if one.attempt_id == ATTEMPT]
            assert row.selected_cell == cell
            assert row.selected_body_reference == carried_source(row).cells[cell].sha256
            [owed] = [one for one in projection.obligations if one.attempt_id == ATTEMPT]
            assert owed.candidate is None
            assert owed.pending_preparation

        # The parent is fenced and parked, and it never advances again. It returns once every
        # child start is confirmed and the fork's own outcome is settled, so what a pull meets is
        # a closed execution that accepts no Update at all rather than a refusal from a handler.
        state = await caller.stream.stream_state()
        assert state.generation_state == "forked"
        with pytest.raises(Exception) as raised:
            await caller.pull()
        assert protocol_error_code(raised.value) is None
        assert fork_barrier_stands(raised.value) or isinstance(raised.value, RPCError)

        # Each child's first work is its own: it reads the parent's row for it through a
        # recorded Activity, from the fork's own namespace, before it owns or serves anything.
        # A replay therefore performs no fresh client I/O and reaches the same answer.
        for identity in identities:
            assert await completed_activities(env.client, identity) == [
                f"fork.{FORK}.origin.1"
            ]
            gated = await env.client.get_workflow_handle(identity).query(
                StreamWorkflow.stream_state
            )
            assert gated.generation_state == "open"
            assert gated.ownership_epoch == 0
            assert gated.cursor == request.acknowledged_cursor

        # Every fork-only Activity took its identifier from the fork's own namespace, so the
        # generation's ordinary numbering is where its inherited prefix left it.
        scheduled = await scheduled_activities(env.client, "stream/fork-quiet/1")
        assert [one for one in scheduled if one.startswith("fork.")] == [
            f"fork.{FORK}.availability.1",
            f"fork.{FORK}.start.1",
            f"fork.{FORK}.start.2",
        ]
        assert all(one.isdigit() for one in scheduled if not one.startswith("fork."))


async def test_a_fork_over_an_eligible_body_the_store_lost_creates_no_child_at_all(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The prebarrier read is a stage of its own, and it is what a barrier is not committed over.

    An ordinary claim reads the manifest and deliberately verifies neither eligible body, so
    without this Activity the fork would fence the parent over two references whose bytes nobody
    had read and hand one child a dependency it could never resolve. The body removed here is the
    one the opposite child would deliver, which no claim and no delivery of this parent ever
    touched.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env, composed, blobs, "stream/fork-lost/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        manifest = await a_committed_manifest(env.client, "stream/fork-lost/1", blobs)
        FilesystemBlobStore(blobs).path_for(manifest.cells[PLACEBO_CELL].sha256).unlink()

        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, request)
        assert fork_refusal(raised.value) == FORK_REPAIRABLE_ABSENCE
        assert fork_can_be_retried(raised.value)

        # No barrier stands, no child exists, and the generation is still serving.
        answer = await fork_status(env.client, request)
        assert answer.found is False
        assert answer.parent_state == "open"
        for ordinal in (1, 2):
            with pytest.raises(RPCError):
                await env.client.get_workflow_handle(
                    child_workflow_id(
                        identity_namespace="default",
                        parent_workflow_id="stream/fork-lost/1",
                        fork_id=FORK,
                        child_ordinal=ordinal,
                    )
                ).describe()
        payload = await caller.pull()
        assert payload.kind == "payload"


async def test_a_fork_that_meets_another_execution_under_a_childs_identity_ends_on_it(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The public route carries the decision, and the parent's record and journal keep it.

    Something else running under a child's derived identity is authenticated: no later reading
    reaches another answer, and creating a replacement under that identity is the one move a fork
    never makes. So what a controller is told is this fork's own permanent refusal naming that
    reason rather than the Activity failure it arrived as, and the experiment reads it as the
    decision it is instead of retrying infrastructure.

    The ending is kept where a fork's answers are kept: the exact identifier keeps it in the
    application's own journal, the record keeps it beside the children the fork already made, and
    a fresh logical retry is answered from that record. The barrier stands and both children stay
    exactly as they were, because an ending is not a claim that anything was rolled back.
    """
    blobs = tmp_path / "blobs"
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract_of(world.env), blobs))
    parent = "stream/fork-taken/1"
    taken = child_workflow_id(
        identity_namespace="default",
        parent_workflow_id=parent,
        fork_id=FORK,
        child_ordinal=1,
    )
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(env, composed, blobs, parent, filing_of(world.env))
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        await env.client.start_workflow(
            "SomethingElseEntirely",
            composed,
            id=taken,
            task_queue="another-queue-entirely",
        )

        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, request)
        assert fork_refusal(raised.value) == FORK_ORIGIN_DISAGREEMENT
        assert not fork_can_be_retried(raised.value)

        # The record keeps the ending and every child the fork made, and the parent stops here.
        answer = await fork_status(env.client, request)
        assert answer.found and answer.record is not None
        assert answer.record.status == FORK_CONFLICTED
        assert answer.record.conflict_reason == FORK_ORIGIN_DISAGREEMENT
        assert "another execution" in answer.record.conflict_clause
        assert [row.existence for row in answer.record.child_records] == [
            EXISTENCE_UNCONFIRMED,
            NEVER_ATTEMPTED,
        ]

        # A fresh logical retry is answered with the decision rather than making it again, and so
        # is the exact identifier that met it, out of the journal that kept it.
        with pytest.raises(Exception) as again:
            await fork_stream(env.client, request, attempt=2)
        assert fork_refusal(again.value) == FORK_ORIGIN_DISAGREEMENT
        with pytest.raises(Exception) as replayed:
            await fork_stream(env.client, request, attempt=1)
        assert fork_refusal(replayed.value) == FORK_ORIGIN_DISAGREEMENT

        # And what was already under that identity is untouched: nothing replaced it.
        described = await env.client.get_workflow_handle(taken).describe()
        assert described.workflow_type == "SomethingElseEntirely"


async def a_committed_manifest(
    client: Client, workflow_id: str, blobs: Path
) -> SourceArtifactManifest:
    """The descriptor one generation committed, fetched by the commitment its record names.

    The commitment is the plain digest of the descriptor's canonical bytes and those bytes are the
    object, so a reader holding it fetches what was committed rather than being told something was
    hashed.
    """
    records = await client.get_workflow_handle(workflow_id).query(
        StreamWorkflow.attempt_records
    )
    [row] = [one for one in records if one.attempt_id == ATTEMPT]
    assert row.source_provenance is not None
    return read_source_artifact(
        json.loads(FilesystemBlobStore(blobs).read(row.source_provenance.source_commitment))
    )


async def test_a_fork_refused_over_a_lost_body_goes_through_under_a_fresh_identifier(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The retry contract, driven where a repairable absence actually needs it.

    A refusal raised in the handler completes the Update as a failure, and the service answers that
    exact identifier out of its own record for ever, however thoroughly the object it was refused
    over is repaired. So the supported next move is the same fork id over the same checkpoint under
    an identifier of its own, and what makes it the same logical fork rather than a second one is
    that the id, the checkpoint and the plans are unchanged.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env, composed, blobs, "stream/fork-repair/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        manifest = await a_committed_manifest(env.client, "stream/fork-repair/1", blobs)
        store = FilesystemBlobStore(blobs)
        lost = manifest.cells[PLACEBO_CELL].sha256
        body = store.read(lost)
        store.path_for(lost).unlink()

        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, request)
        assert fork_refusal(raised.value) == FORK_REPAIRABLE_ABSENCE

        # The exact bytes are reinstalled, and the identifier that met the loss answers with it.
        assert store.put(body).sha256 == lost
        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, request)
        assert fork_refusal(raised.value) == FORK_REPAIRABLE_ABSENCE
        assert fork_can_be_retried(raised.value)

        # And the same fork under a fresh identifier is the one that goes through.
        receipt = await fork_stream(env.client, request, attempt=2)
        assert receipt.fork_id == FORK
        assert receipt.children == 2
        answer = await fork_status(env.client, request)
        assert answer.found and answer.record is not None
        assert answer.record.status == FORK_COMPLETE


async def test_a_forked_parent_answers_the_same_children_and_names_a_changed_request_a_conflict(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """Recovery, exercised on the record rather than on a description of it.

    A retried fork returns the same children: the starts are rebuilt from the same request over a
    parent that has not moved since, each rebuilt start is held to the digest the record committed
    for it, and the children that exist are preserved rather than replaced. The same fork id over
    other plans is a conflict rather than a match, an unknown fork id is answered as nothing, and
    the exact identifier that completed keeps returning what it returned.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await worked(
            env, composed, blobs, "stream/fork-again/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        receipt = await fork_stream(env.client, request)
        runs = [child.child_run_id for child in receipt.child_receipts]

        await env.client.get_workflow_handle("stream/fork-again/1").result(
            rpc_timeout=timedelta(seconds=30)
        )

        # The parent returns once its fork is settled, so the same request sent again is not sent
        # at all: a closed execution accepts no Update, and the answer is read out of the outcome
        # journal the parent still holds. Both routes return the original children.
        assert await fork_stream(env.client, request) == receipt
        answer = await fork_status(env.client, request)
        assert answer.found and answer.conflict is False and answer.record is not None
        assert answer.parent_state == "forked"
        assert answer.record.status == FORK_COMPLETE
        assert [row.existence for row in answer.record.child_records] == [
            CONFIRMED_EXISTING,
            CONFIRMED_EXISTING,
        ]
        assert [row.child_run_id for row in answer.record.child_records] == runs
        assert answer.receipt == receipt

        # A fresh identifier for the same logical fork is answered with the same children too. The
        # parent has closed, so the Update reaches no handler at all and the outcome journal has
        # never seen this identifier: the recorded evidence is what the status route returns, and a
        # completed fork reported as still in flight would send a controller back to a parent that
        # can accept nothing.
        assert await fork_stream(env.client, request, attempt=2) == receipt

        # The same fork id over other plans is a conflict rather than a match, and a fork this
        # parent never accepted is answered as nothing rather than as a conflict.
        conflicting = replace(
            request,
            child_plans=[
                plan_for(composed, FIRST_SLOT, PLACEBO_CELL, tmp_path / "child-1"),
                plan_for(composed, SECOND_SLOT, GRADED_CELL, tmp_path / "child-2"),
            ],
        )
        assert (await fork_status(env.client, conflicting)).conflict is True
        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, conflicting)
        assert fork_refusal(raised.value) == FORK_REQUEST_CONFLICT
        assert not fork_can_be_retried(raised.value)

        unknown = await fork_status(env.client, replace(request, fork_id="fork-2"))
        assert unknown.found is False
        assert unknown.conflict is False
        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, replace(request, fork_id="fork-2"))
        assert fork_refusal(raised.value) == FORK_EXPIRED_AUTHORITY

        # A history the service cannot produce at all is that same answer rather than a temporary
        # inability to read: the question was asked outside the window this parent's answers stand
        # in, and a controller records the run incomplete instead of retrying a parent that will
        # never answer.
        with pytest.raises(Exception) as raised:
            await fork_status(
                env.client, replace(request, parent_workflow_id="stream/no-such-parent/1")
            )
        assert fork_refusal(raised.value) == FORK_EXPIRED_AUTHORITY
        assert not fork_can_be_retried(raised.value)


#: The ordinals whose first start is to fail. The Worker registers this one instead of the real
#: start, because two Activities of one name are refused, and it creates every other child for
#: real: what is under test is the parent's recovery from a fork it created half of.
_FAILING_STARTS: set = set()


@activity.defn(name=START_FORK_CHILD)
async def _a_failing_start(request: StartForkChildInput) -> ForkChildStarted:
    """Fail the named child's first start once, and start every child for real after that."""
    if request.child_ordinal in _FAILING_STARTS:
        _FAILING_STARTS.discard(request.child_ordinal)
        raise ApplicationError(
            f"the service said nothing about child {request.child_ordinal}",
            non_retryable=True,
        )
    return await start_fork_child_activity(request)


async def test_a_fork_that_created_one_child_starts_the_missing_one_and_replaces_neither(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """Recovery from the state where one child exists and the other was never created.

    The children are started one at a time, so a fork can end with the first intact and the second
    unstarted, and the prepared record is what says which children this fork is. The parent stays
    fenced and parked over that record, and the same fork id under a fresh identifier starts the
    missing child alone: the one that exists is preserved by its own recorded existence and never
    replaced, which is the only thing that keeps a child's identity meaning one execution.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    _FAILING_STARTS.clear()
    _FAILING_STARTS.add(2)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(
        env.client, activities=[*activities_of(world), _a_failing_start]
    ):
        caller = await worked(
            env, composed, blobs, "stream/fork-partial/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, request)
        # It is a fault and not a decision: nothing about the request was refused.
        assert fork_refusal(raised.value) is None

        # The barrier stands over both children, one confirmed and one attempted and unknown. An
        # Activity that failed is never proof that no child exists, so the second is not in the
        # class a replacement could be created under.
        answer = await fork_status(env.client, request)
        assert answer.found and answer.record is not None
        assert answer.record.status == FORK_PREPARED
        assert [row.existence for row in answer.record.child_records] == [
            CONFIRMED_EXISTING,
            EXISTENCE_UNCONFIRMED,
        ]
        [first, second] = answer.record.child_records
        assert first.child_run_id
        assert second.child_run_id is None
        with pytest.raises(RPCError):
            await env.client.get_workflow_handle(second.child_workflow_id).describe()

        # The same fork under a fresh identifier starts the missing child and nothing else.
        receipt = await fork_stream(env.client, request, attempt=2)
        assert receipt.children == 2
        assert [child.child_ordinal for child in receipt.child_receipts] == [1, 2]
        assert receipt.child_receipts[0].child_run_id == first.child_run_id
        assert receipt.child_receipts[1].child_run_id
        assert (
            await started_with(env.client, second.child_workflow_id)
        ).fork_origin is not None
        scheduled = await scheduled_activities(env.client, "stream/fork-partial/1")
        assert [one for one in scheduled if one.startswith("fork.")] == [
            f"fork.{FORK}.availability.1",
            f"fork.{FORK}.start.1",
            f"fork.{FORK}.start.2",
            f"fork.{FORK}.start.3",
        ]


#: The prebarrier read, held open so a test can move the stream while it is running. The Worker
#: registers this one instead of the real availability Activity, because two Activities of one
#: name are refused, and it does the real read once the test lets it go.
_HELD: dict = {}


@activity.defn(name=FORK_AVAILABILITY)
async def _a_held_availability(request: ForkAvailabilityInput) -> ForkAvailability:
    """Say that the fork has reached its one await, then wait to be let go."""
    _HELD["reached"].set()
    await _HELD["release"].wait()
    return await fork_availability_activity(request)


def held_open() -> None:
    """Arm the two events the held read is driven by."""
    _HELD["reached"] = asyncio.Event()
    _HELD["release"] = asyncio.Event()


async def test_a_second_fork_arriving_while_one_is_in_flight_is_refused_as_not_quiet(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The ledger, seen the only way it can be: with a real handler accepted and unfinished.

    The SDK inserts an Update into its own in-progress map before that Update's validator runs and
    removes it in the handler's finally, so a fork that waited for the public predicate would wait
    for itself for ever. What this drives is the other half of that: a second fork arriving under
    another identifier while the first is still in flight sees the first in the ledger and is
    refused, and the first goes on to answer with the complete receipt.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    held_open()
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(
        env.client, activities=[*activities_of(world), _a_held_availability]
    ):
        caller = await worked(
            env, composed, blobs, "stream/fork-busy/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        first = asyncio.ensure_future(fork_stream(env.client, request))
        await asyncio.wait_for(_HELD["reached"].wait(), timeout=30)

        handle = env.client.get_workflow_handle_for(StreamWorkflow.run, "stream/fork-busy/1")
        with pytest.raises(Exception) as raised:
            await handle.execute_update(
                StreamWorkflow.fork_generation,
                replace(request, fork_id="fork-2"),
                id="a-second-fork",
            )
        assert fork_refusal(raised.value) == FORK_NOT_QUIET
        assert fork_can_be_retried(raised.value)

        _HELD["release"].set()
        receipt = await first
        assert receipt.fork_id == FORK
        assert receipt.children == 2
        # The refused second fork installed nothing, and the record names the first alone.
        answer = await fork_status(env.client, replace(request, fork_id="fork-2"))
        assert answer.found is False


async def test_a_boundary_that_moved_while_the_prebarrier_read_ran_installs_no_barrier(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The final recheck, driven by moving the stream in the one window where it can move.

    The barrier does not stand while the prebarrier read runs, so the generation can serve, and
    the payload offered here leaves a message pending that the boundary refuses. The recheck after
    the read is what catches it, and there is no await between that recheck and the commit, so a
    fork refused here fences nothing and creates nothing.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    held_open()
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(
        env.client, activities=[*activities_of(world), _a_held_availability]
    ):
        caller = await worked(
            env, composed, blobs, "stream/fork-moved/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        forking = asyncio.ensure_future(fork_stream(env.client, request))
        await asyncio.wait_for(_HELD["reached"].wait(), timeout=30)

        payload = await caller.pull()
        assert payload.kind == "payload"
        _HELD["release"].set()
        with pytest.raises(Exception) as raised:
            await forking
        assert fork_refusal(raised.value) == FORK_NOT_QUIET
        assert fork_can_be_retried(raised.value)

        # Nothing was fenced and nothing was created, and the generation is still serving.
        answer = await fork_status(env.client, request)
        assert answer.found is False
        assert answer.parent_state == "open"
        for ordinal in (1, 2):
            with pytest.raises(RPCError):
                await env.client.get_workflow_handle(
                    child_workflow_id(
                        identity_namespace="default",
                        parent_workflow_id="stream/fork-moved/1",
                        fork_id=FORK,
                        child_ordinal=ordinal,
                    )
                ).describe()
        await caller.present(payload)
        assert (await caller.stream.stream_state()).generation_state == "open"


#: An ownership claim's own read of the store, held open so a fork can arrive while a handler
#: that holds no stream lock is in flight. It replaces the real verification in the Worker,
#: because two Activities of one name are refused, and it reads for real once it is let go.
_HELD_CLAIM: dict = {}


@activity.defn(name=VERIFY_BLOBS)
async def _a_held_verification(request: VerifyBlobsInput) -> BlobsVerified:
    """Pause the one claim a test is holding, then read the store as the real one does."""
    if _HELD_CLAIM:
        held = dict(_HELD_CLAIM)
        _HELD_CLAIM.clear()
        held["reached"].set()
        await held["release"].wait()
    return await verify_blobs_activity(request)


def a_held_claim() -> Any:
    """Arm the pause the next ownership claim's read of the store waits inside."""
    _HELD_CLAIM.update(reached=asyncio.Event(), release=asyncio.Event())
    return _HELD_CLAIM["reached"], _HELD_CLAIM["release"]


def without_the_verification(activities: List[Any]) -> List[Any]:
    """This environment's Activities except its read of the store, which is held instead."""
    return [
        one
        for one in activities
        if getattr(one, "__temporal_activity_definition").name != VERIFY_BLOBS
    ]


async def test_a_claim_still_reading_the_store_is_what_refuses_a_fork_at_a_quiet_boundary(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The ledger's own case: an accepted handler holding nothing a boundary can see.

    An ownership claim reads the store before it swaps the epoch, and while that read runs it
    holds no operation, no grant, no pending message and no ticket, so every clause of the
    boundary reads quiet and the ledger is the only thing that says a handler is in flight. A
    fork admitted there would be cut across a claim that has still to install its new owner.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(
        env.client,
        activities=[*without_the_verification(activities_of(world)), _a_held_verification],
    ):
        caller = await worked(
            env, composed, blobs, "stream/fork-claimed/1", filing_of(world.env)
        )
        request = await a_fork_request(env.client, caller, composed, tmp_path)
        reached, release = a_held_claim()
        claiming = asyncio.ensure_future(
            resume_stream(
                env.client,
                workflow_id="stream/fork-claimed/1",
                configuration_hash=configuration_hash(composed),
                claimant_id="harness-2",
            )
        )
        await asyncio.wait_for(reached.wait(), timeout=30)

        # The generation is quiet by every clause a boundary reads, and the fork is still refused.
        state = await env.client.get_workflow_handle_for(
            StreamWorkflow.run, "stream/fork-claimed/1"
        ).query(StreamWorkflow.stream_state)
        assert state.generation_state == "open"
        assert state.pending_message_id is None
        assert state.environment_call is None
        assert state.verifying == 1
        assert state.unfinished_handlers == 1
        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, request)
        assert fork_refusal(raised.value) == FORK_NOT_QUIET
        assert fork_can_be_retried(raised.value)
        assert "has been accepted and has not finished" in str(raised.value.__cause__)
        assert "own-1-harness-2" in str(raised.value.__cause__)

        # The ledger empties as that handler finishes, and what stops the fork then is the witness
        # the claim moved rather than the ledger: the ownership epoch is inside the projection
        # digest, so the controller reads the checkpoint evidence again and submits the same plans
        # against what this generation now stands at.
        release.set()
        await claiming
        state = await env.client.get_workflow_handle_for(
            StreamWorkflow.run, "stream/fork-claimed/1"
        ).query(StreamWorkflow.stream_state)
        assert state.unfinished_handlers == 0
        assert state.ownership_epoch == 2
        with pytest.raises(Exception) as raised:
            await fork_stream(env.client, request, attempt=2)
        assert fork_refusal(raised.value) == FORK_WITNESS_MISMATCH
        receipt = await fork_stream(
            env.client, replace(request, projection_digest=state.stream_state_sha256)
        )
        assert receipt.fork_id == FORK
        assert receipt.children == 2


async def test_a_child_already_started_is_adopted_only_where_it_is_that_child(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """A duplicate is an existing child and never permission to create a replacement.

    What the adoption compares is the original execution rather than whatever runs under the
    identity now, because a child that has continued as new carries a different carrier under the
    same id. A workflow type, a task queue or a start argument that disagrees is authenticated
    and permanent: the fork keeps the identity and replaces nothing.
    """
    blobs = tmp_path / "blobs"
    composed = fork_capable(start_for(world, contract_of(world.env), blobs))
    queue = "the-queue-of-this-run"
    activities = ActivityEnvironment(client=env.client)
    asked = StartForkChildInput(
        fork_id=FORK,
        child_ordinal=1,
        child_workflow_id="stream/adopted/child-1",
        task_queue=queue,
        start=composed,
    )
    started = await activities.run(start_fork_child_activity, asked)
    assert started.created is True
    assert started.child_run_id

    # The same child asked for again is resolved to its original execution rather than started.
    again = await activities.run(start_fork_child_activity, asked)
    assert again.created is False
    assert again.child_run_id == started.child_run_id

    # What is already there under another type is not this child.
    await env.client.start_workflow(
        "SomethingElseEntirely",
        composed,
        id="stream/adopted/child-2",
        task_queue=queue,
    )
    other_type = replace(asked, child_workflow_id="stream/adopted/child-2")
    other_queue = replace(asked, task_queue="another-queue")
    other_start = replace(
        asked, start=replace(composed, hidden_execution_id="another-execution")
    )
    for disagreeing in (other_type, other_queue, other_start):
        with pytest.raises(ApplicationError) as raised:
            await activities.run(start_fork_child_activity, disagreeing)
        assert raised.value.type == "OriginDisagreement"
        assert raised.value.non_retryable is True


#: The member this converter keeps beside the body rather than inside it.
MOVED_MEMBER = "the-hidden-execution-this-converter-keeps-beside-the-body"


class AConverterThatKeepsPartOfAStartBesideIt(DefaultPayloadConverter):
    """A configured converter that puts part of a start's value in the payload's metadata.

    It delegates for everything else and restores what it moved when it decodes, so the value that
    comes back is the value that went in and the carrier stays in the format it was admitted in.
    A deployment configures its own converter, and what a start is worth is what its converter
    made of it rather than what one of a payload's two halves holds.
    """

    def to_payloads(self, values: Sequence[Any]) -> List[Payload]:
        encoded = list(super().to_payloads(values))
        for value, payload in zip(values, encoded):
            if not isinstance(value, StreamStart):
                continue
            body = json.loads(payload.data.decode("utf-8"))
            payload.metadata[MOVED_MEMBER] = str(
                body.pop("hidden_execution_id", "")
            ).encode("utf-8")
            payload.data = json.dumps(body).encode("utf-8")
        return encoded

    def from_payloads(
        self, payloads: Sequence[Payload], type_hints: Optional[List] = None
    ) -> List[Any]:
        restored = []
        for payload in payloads:
            if MOVED_MEMBER not in payload.metadata:
                restored.append(payload)
                continue
            body = json.loads(payload.data.decode("utf-8"))
            body["hidden_execution_id"] = payload.metadata[MOVED_MEMBER].decode("utf-8")
            copy = Payload()
            copy.CopyFrom(payload)
            copy.data = json.dumps(body).encode("utf-8")
            del copy.metadata[MOVED_MEMBER]
            restored.append(copy)
        return super().from_payloads(restored, type_hints)


async def test_a_duplicate_is_compared_as_the_whole_of_what_its_converter_made_of_it(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The comparison is over the complete payload, because the body is only half of one.

    A converter is free to keep part of what it encodes beside the body, and the value it encoded
    is both halves: two starts differing in a member this converter keeps in the metadata have
    equal bodies and are different starts. Adopting an execution on the body alone would confirm a
    child created from a start this fork never built, and the child's own later gate cannot repair
    that, because the parent has already said this is its child.

    The recovery a duplicate exists for is exercised in the same shape: the start this fork did
    build is adopted under the same converter, so what closes the hole is a comparison over more
    of the value rather than a comparison that refuses everything.
    """
    blobs = tmp_path / "blobs"
    composed = fork_capable(start_for(world, contract_of(world.env), blobs))
    queue = "the-queue-of-this-run"
    configured = env.client.config()
    configured["data_converter"] = DataConverter(
        payload_converter_class=AConverterThatKeepsPartOfAStartBesideIt
    )
    client = Client(**configured)
    converter = client.data_converter.payload_converter
    original = replace(composed, hidden_execution_id="the-execution-already-there")
    asked = StartForkChildInput(
        fork_id=FORK,
        child_ordinal=1,
        child_workflow_id="stream/beside-the-body/child-1",
        task_queue=queue,
        start=replace(composed, hidden_execution_id="the-execution-this-fork-built"),
    )
    # The two starts are different values, and this converter puts the difference in the metadata.
    assert converter.to_payloads([original])[0].data == (
        converter.to_payloads([asked.start])[0].data
    )
    assert dict(converter.to_payloads([original])[0].metadata) != dict(
        converter.to_payloads([asked.start])[0].metadata
    )

    await client.start_workflow(
        STREAM_WORKFLOW_TYPE, original, id=asked.child_workflow_id, task_queue=queue
    )
    activities = ActivityEnvironment(client=client)
    with pytest.raises(ApplicationError) as raised:
        await activities.run(start_fork_child_activity, asked)
    assert raised.value.type == "OriginDisagreement"
    assert raised.value.non_retryable is True

    # And the child this fork did build is still adopted, under the same converter, with the run
    # its original execution was created under.
    mine = replace(asked, child_workflow_id="stream/beside-the-body/child-2")
    started = await activities.run(start_fork_child_activity, mine)
    assert started.created is True
    again = await activities.run(start_fork_child_activity, mine)
    assert again.created is False
    assert again.child_run_id == started.child_run_id

# The child side: a gated child's own closure, the body it prepares, and what it publishes.
#
# Everything below the fork runs under a Worker that cannot seal, cannot grade and cannot render.
# The claim under test is that a child derives its first payload from committed evidence and
# nothing else, and the only way to show that is to take everything else away.


#: This environment's own capture, and whether this run may still ask for it. One Worker serves a
#: whole run, so the calls that could rebuild A are closed by name when the fork returns rather
#: than being taken away with the Worker that answered them.
_CAPTURE: dict = {}


def a_closing_worker(episode: ServedEpisode) -> List[Any]:
    """This environment's own Activities under a switch, and the read a claim needs."""
    named = {
        getattr(one, "__temporal_activity_definition").name: one
        for one in activities_of(episode)
    }
    _CAPTURE.clear()
    _CAPTURE.update({"closed": False, **named})
    return [_a_closing_seal, _a_closing_grade, _a_closing_render, verify_blobs_activity]


def close_the_capture() -> None:
    """Make every call that could rebuild A fatal, which is what a child has to work without."""
    _CAPTURE["closed"] = True


def _still_open(name: str) -> Any:
    """The environment's own Activity, while this run may still ask for A's own capture."""
    if _CAPTURE["closed"]:
        raise ApplicationError(
            f"a child derives its body from committed evidence and this run asked {name} again",
            non_retryable=True,
        )
    return _CAPTURE[name]


@activity.defn(name=SEAL_ATTEMPT)
async def _a_closing_seal(request: SealAttemptInput) -> SealAttemptResult:
    """Seal while A's own capture is open, and refuse once the fork has closed it."""
    return await _still_open(SEAL_ATTEMPT)(request)


@activity.defn(name=GRADE_ATTEMPT)
async def _a_closing_grade(request: GradeAttemptInput) -> GradeAttemptResult:
    """Grade on the same terms, so no child can be scored a second time."""
    return await _still_open(GRADE_ATTEMPT)(request)


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def _a_closing_render(request: GeneratePayloadBundleInput) -> PayloadBundle:
    """Render on the same terms, so a child's body can only have come from its preparation."""
    return await _still_open(GENERATE_PAYLOAD_BUNDLE)(request)


def carried(start: StreamStart) -> Any:
    """The projection one start hands over, read the way the far side reads it."""
    assert start.carry is not None
    return unpack_carrier(start.carry, CONVERTER)


def selected_reference(start: StreamStart) -> str:
    """The committed entry one child was selected for, out of its own carrier."""
    [row] = [one for one in carried(start).attempts if one.attempt_id == ATTEMPT]
    return row.selected_body_reference or ""


def child_closure(start: StreamStart) -> List[str]:
    """Every object one child needs: what its own claim reads back, and the body it prepares.

    Neither eligible body is in the claim-required half and the oracle is in neither, so a child
    whose counterpart body or oracle body never arrived still claims and still prepares.
    """
    return [*carried(start).committed_blobs, selected_reference(start)]


def copy_closure(source: Path, destination: str, references: List[str]) -> None:
    """Install those exact objects in a child's own store, out of the store its parent used."""
    origin = FilesystemBlobStore(source)
    into = FilesystemBlobStore(Path(destination))
    for reference in references:
        into.put(origin.read(reference))


async def a_claimed_child(client: Client, start: StreamStart, identity: str) -> Any:
    """Claim one child, once its own history has authorized the lineage it carries.

    A gated child owns nothing and serves nothing until it has read the parent's record of it, and
    it says that rather than refusing: the same claim under the same identifier reaches a child
    that has done its own first work, so waiting it out is what a controller does.
    """
    for _ in range(500):
        try:
            return await resume_stream(
                client, workflow_id=identity, configuration_hash=configuration_hash(start)
            )
        except Exception as error:  # noqa: BLE001 - the gate is waited out and nothing else is
            if not origin_unverified(error):
                raise
            await asyncio.sleep(0.02)
    raise AssertionError(f"{identity} never authorized the lineage it carries")


async def a_ready_child(
    client: Client, start: StreamStart, identity: str, blobs: Path
) -> Tuple[Any, ChildReady]:
    """Copy one child's closure into its own store, claim it, and prepare the body it owes."""
    copy_closure(blobs, start.blob_root or "", child_closure(start))
    stream = await a_claimed_child(client, start, identity)
    return stream, await stream.prepare_child(fork_id=FORK)


async def a_child_caller(stream: Any, start: StreamStart) -> Caller:
    """Bind one child's own consumer, and read from where its inherited prefix left the cursor."""
    await stream.claim_consumer(
        ConsumerClaim(
            consumer_id=f"harness-{start.served_slot}",
            claim_hash=start.consumer_claim_hash,
        )
    )
    state = await stream.stream_state()
    return Caller(stream, state.cursor, FilesystemBlobStore(Path(start.blob_root or "")))


async def child_records(client: Client, identity: str) -> Any:
    """One child's own attempts and the operation rows beside them, read off one moment."""
    return await client.get_workflow_handle(identity).query(StreamWorkflow.generation_records)


async def forked(
    env: Any,
    world: ServedEpisode,
    composed: StreamStart,
    blobs: Path,
    tmp_path: Path,
    workflow_id: str,
    *,
    crossed: bool = False,
    turnover_at: Any = None,
) -> Tuple[ForkRequest, Any, List[StreamStart]]:
    """Take one generation to its acknowledgement, fork it, and read out both child starts.

    ``crossed`` takes the parent over a continuation boundary before the fork, which is where a
    child's preparation has to work from: the world behind A is gone with the execution that held
    it and the committed evidence is all there is.

    One Worker serves the whole lineage, which is the deployment invariant, so this runs inside
    the caller's Worker rather than owning one of its own.
    """
    caller = await worked(env, composed, blobs, workflow_id, filing_of(world.env))
    if crossed:
        await cross_a_boundary(caller, turnover_at, workflow_id, env.client)
    request = await a_fork_request(env.client, caller, composed, tmp_path)
    receipt = await fork_stream(env.client, request)
    starts = [
        await started_with(env.client, child.child_workflow_id)
        for child in receipt.child_receipts
    ]
    return request, receipt, starts


async def test_a_gated_child_prepares_the_cell_it_was_selected_for_and_says_it_is_ready(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The whole child side, over a parent that has already continued as new.

    Nothing that could rebuild A is available: the world behind the filing is gone with the
    execution that held it, and the Worker serving the children refuses to seal, to grade or to
    render. What each child has is the descriptor its parent committed, the entry in it for the
    cell it was selected for, and the bytes under that entry, and that is what it delivers.

    The two children deliver different cells of one source and their payloads come to one
    measurement, which is what the pair evidence is for: publication verified both bodies against
    one registered mask and one encoded body count, so equal wire measurement and separately
    correct per-cell hashes are what each of them proves here.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs, silent=True))
    delivered = {}
    async with stream_worker(env.client, activities=a_closing_worker(world)):
        request, receipt, starts = await forked(
            env,
            world,
            composed,
            blobs,
            tmp_path,
            "stream/child-ready/1",
            crossed=True,
            turnover_at=turnover_at,
        )
        manifest = carried_source(
            [one for one in carried(starts[0]).attempts if one.attempt_id == ATTEMPT][0]
        )
        close_the_capture()
        for start, child in zip(starts, receipt.child_receipts):
            identity = child.child_workflow_id
            store = FilesystemBlobStore(Path(start.blob_root or ""))
            stream, ready = await a_ready_child(env.client, start, identity, blobs)

            # Neither the counterpart body nor the oracle body ever reached this child, and
            # neither blocked its claim or its preparation.
            assert store.unverified(
                [manifest.cells[ORACLE_CELL].sha256]
            ) == [manifest.cells[ORACLE_CELL].sha256]
            counterpart = manifest.cells[
                PLACEBO_CELL if child.target_cell == GRADED_CELL else GRADED_CELL
            ].sha256
            assert store.unverified([counterpart]) == [counterpart]

            # The readiness certifies one child at one checkpoint with one candidate and one
            # ownership state, and every value in it is one this child holds.
            assert ready.fork_id == FORK
            assert ready.child_workflow_id == identity
            assert ready.child_run_id
            assert ready.checkpoint_manifest_reference == (
                request.checkpoint_manifest_reference
            )
            assert start.fork_origin is not None
            assert ready.origin_digest == origin_digest(start.fork_origin)
            assert ready.verified_set_digest == verified_set_digest(
                carried(start).committed_blobs
            )
            assert ready.source_attempt_id == ATTEMPT
            assert ready.selected_cell == child.target_cell
            assert ready.selected_body_reference == manifest.cells[child.target_cell].sha256
            assert ready.selected_policy_digest == CELL_POLICIES[child.target_cell]
            assert ready.receipt_contract_id == CONTRACT
            assert ready.ownership_epoch == 1
            assert ready.consumer_id == ""
            assert ready.preparation_operation == fork_preparation_operation_identity(
                FORK, identity, 1
            )

            # The candidate is installed exactly once and counts as neither a materialization nor
            # a release: the obligation it fills was materialized and released by the generation
            # this child inherited. The preparation reads its objects through an Activity of its
            # own, so it adds no verification batch to the one this child's claim took.
            state = await stream.stream_state()
            assert state.materialization_count == 1
            assert state.eligibility_count == 1
            assert state.verification_batches == carried(start).verification_batches + 1
            assert state.obligations[ATTEMPT] == "eligible"

            caller = await a_child_caller(stream, start)
            payload = await caller.pull()
            assert payload.kind == "payload"
            assert payload.message_id == oid(0x103)
            delivered[child.target_cell] = payload

            # The bytes are the environment's own for this cell, and every measurement the
            # contract fixed holds over them.
            body = body_of(payload)
            raw = body.encode("ascii")
            assert sha256(raw).hexdigest() == manifest.cells[child.target_cell].sha256
            assert len(raw) == BODY_SIZE == contract.body_size
            assert sha256(masked_body(raw, mask_spans(contract))).hexdigest() == (
                manifest.pair_parity.masked_body_sha256
            )
            assert len(payload.visible_text.encode("utf-8")) == payload_wire_count(
                payload_message_id=oid(0x103),
                attempt_id=ATTEMPT,
                encoded_body_bytes=manifest.pair_parity.encoded_body_bytes,
            )
            # And no oracle: the cell nobody may be served is not the body either child got.
            assert sha256(raw).hexdigest() != manifest.cells[ORACLE_CELL].sha256

            # Every call this child made is one no unforked twin makes, its own first claim
            # among them, and every one of them took its identifier from the fork's own
            # namespace. So the child consumed no ordinary ordinal at all and its numbering is
            # exactly the number its inherited prefix left, while its verification batch count
            # is the one batch its claim took ahead of the twin's.
            scheduled = await scheduled_activities(env.client, identity)
            assert scheduled == [
                f"fork.{FORK}.origin.1",
                f"fork.{FORK}.claim.1",
                f"fork.{FORK}.preparation.1",
            ]

    # Two cells of one source, one measurement, and nothing filed twice.
    [graded, placebo] = [delivered[GRADED_CELL], delivered[PLACEBO_CELL]]
    assert body_of(graded) != body_of(placebo)
    assert len(graded.visible_text.encode("utf-8")) == len(
        placebo.visible_text.encode("utf-8")
    )


async def test_a_child_that_lost_its_body_after_claiming_refuses_and_recovers_as_one_operation(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """Valid committed bytes lost in the window between a successful claim and a preparation.

    The claim, the grade, the source and the selection are all intact and only the object is gone,
    so this is repairable rather than a decision: the exact bytes go back and the same logical
    preparation runs again. What makes it the same operation is the identity frozen at the first
    attempt, which is recomputed after the epoch steps and comes out the same, and what makes it
    reach a handler at all is the derived Update identifier, which is the one thing the epoch
    moves. The original refusal stands for ever under the identifier that met it.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(env.client, activities=a_closing_worker(world)):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-repair/1"
        )
        close_the_capture()
        start, child = starts[0], receipt.child_receipts[0]
        identity = child.child_workflow_id
        store = FilesystemBlobStore(Path(start.blob_root or ""))
        frozen = fork_preparation_operation_identity(FORK, identity, 1)
        copy_closure(blobs, start.blob_root or "", child_closure(start))
        stream = await a_claimed_child(env.client, start, identity)
        reference = selected_reference(start)
        store.path_for(reference).unlink()

        with pytest.raises(Exception) as raised:
            await stream.prepare_child(fork_id=FORK)
        assert fork_refusal(raised.value) == FORK_REPAIRABLE_ABSENCE
        assert fork_can_be_retried(raised.value)

        # The row names the object the operation could not produce, under the identity the
        # recovery will be joined to it by, and the child reads unavailable while it stands.
        records = await child_records(env.client, identity)
        [row] = records.operation_failures
        assert (row.phase, row.reason, row.outcome) == (
            FORK_PREPARATION,
            UNAVAILABLE_EVIDENCE,
            REFUSED_OPERATION,
        )
        assert row.operation == frozen
        assert row.generation == identity
        assert row.attempt_id == ATTEMPT
        assert row.payload_position == 0
        assert row.references == [reference]
        assert (row.refused_epoch, row.recovered_epoch) == (1, None)
        [attempt] = [one for one in records.attempts if one.attempt_id == ATTEMPT]
        assert receipt_availability(attempt, records.operation_failures) == (
            RECEIPT_UNAVAILABLE
        )
        assert (await stream.stream_state()).obligations[ATTEMPT] == "eligible"

        # Putting the bytes back appends nothing, and the identifier that met the loss keeps
        # returning what it was answered with.
        copy_closure(blobs, start.blob_root or "", [reference])
        with pytest.raises(Exception) as again:
            await stream.prepare_child(fork_id=FORK)
        assert fork_refusal(again.value) == FORK_REPAIRABLE_ABSENCE
        assert len((await child_records(env.client, identity)).operation_failures) == 1

        # And a boundary crossed while this child is still unprepared, which is lawful and is the
        # entry the gate its parent wrote exists to survive: the obligation carries that gate over
        # and the execution on the far side is the one the repair has to be answered by. It
        # carries the episode the first attempt opened over with it, so what keeps the repair one
        # operation is the child's own record rather than anything the far side reconstructs.
        await hand_it_on(stream, turnover_at, identity, env.client)
        crossed = await handed_on(env.client, identity)
        assert crossed.fork_origin == start.fork_origin
        assert crossed.carry is not None
        continued = unpack_carrier(crossed.carry, CONVERTER)
        [owed] = [one for one in continued.obligations if one.attempt_id == ATTEMPT]
        assert owed.pending_preparation is True
        assert owed.candidate is None
        assert owed.preparation_epoch == 1
        assert [one.operation for one in continued.operation_failures] == [frozen]

        # Progress is the same preparation under the owner that follows: a fresh derived Update
        # identifier reaches a handler, and the identity it is recorded under is the frozen one.
        second = await a_claimed_child(env.client, start, identity)
        ready = await second.prepare_child(fork_id=FORK)
        assert ready.ownership_epoch == 2
        assert ready.preparation_operation == frozen

        records = await child_records(env.client, identity)
        assert [
            (one.operation, one.outcome, one.refused_epoch, one.recovered_epoch)
            for one in records.operation_failures
        ] == [(frozen, REFUSED_OPERATION, 1, None), (frozen, RECOVERED_OPERATION, 1, 2)]
        [attempt] = [one for one in records.attempts if one.attempt_id == ATTEMPT]
        assert receipt_availability(attempt, records.operation_failures) == RECEIPT_AVAILABLE

        # A crash between the installation and the answer is a republished readiness rather than
        # a second candidate and a second episode.
        third = await a_claimed_child(env.client, start, identity)
        republished = await third.prepare_child(fork_id=FORK)
        assert republished == replace(ready, ownership_epoch=3)
        assert len((await child_records(env.client, identity)).operation_failures) == 2

        caller = await a_child_caller(third, start)
        payload = await caller.pull()
        assert sha256(body_of(payload).encode("ascii")).hexdigest() == reference


#: Whether the preparation below is holding its answer back, and whether it has reached the point
#: it holds at. The pair is how a preparation is caught in flight: one owner asks, the Activity
#: says it has read the objects, a second owner takes the generation over, and the first owner is
#: fenced on the way back in with its episode already open.
_HELD: dict = {}


@activity.defn(name=PREPARE_FORK_CHILD)
async def _a_held_preparation(request: ForkPreparationInput) -> ForkPreparation:
    """Read the objects for real, and hold the answer for as long as the test holds it."""
    prepared = await fork_preparation_activity(request)
    _HELD["running"] = True
    while _HELD.get("held"):
        await asyncio.sleep(0.02)
    return prepared


async def test_a_childs_preparation_episode_is_the_one_it_opened_whatever_its_first_attempt_did(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The episode is frozen at the first attempt, and a first attempt need not leave a row.

    A preparation writes an operation failure row when it is refused and writes none when it
    succeeds, and the receipt route's own ordering says a superseded owner is fenced without a
    stale row as well. So an identity reconstructed from rows is an identity two of the three
    first attempts cannot supply, and the record the episode is kept in is the child's own
    obligation instead.

    Both halves are driven here over a real child, each across the boundary that destroys anything
    an execution was holding in memory. The first child prepares, crosses and is reattached, and
    republishes its readiness under the episode it opened. The second is fenced after its Activity
    answered, leaves nothing behind but the episode, crosses, and the owner after that prepares
    under that same episode rather than opening a second one.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    _HELD.clear()
    async with stream_worker(
        env.client, activities=[*a_closing_worker(world), _a_held_preparation]
    ):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-episode/1"
        )
        close_the_capture()

        # The first attempt succeeds, so there is no failure row anywhere to read an epoch out of.
        start, child = starts[0], receipt.child_receipts[0]
        identity = child.child_workflow_id
        opened = fork_preparation_operation_identity(FORK, identity, 1)
        stream, ready = await a_ready_child(env.client, start, identity, blobs)
        assert (ready.ownership_epoch, ready.preparation_operation) == (1, opened)
        assert (await child_records(env.client, identity)).operation_failures == []

        # The boundary takes this execution's memory with it, and the episode crosses in the
        # record the obligation keeps rather than in the rows the attempt did not write.
        await hand_it_on(stream, turnover_at, identity, env.client)
        [owed] = [
            one
            for one in carried(await handed_on(env.client, identity)).obligations
            if one.attempt_id == ATTEMPT
        ]
        assert (owed.pending_preparation, owed.preparation_epoch) == (False, 1)

        second = await a_claimed_child(env.client, start, identity)
        again = await second.prepare_child(fork_id=FORK)
        assert again.child_run_id != ready.child_run_id
        assert again == replace(
            ready, ownership_epoch=2, child_run_id=again.child_run_id
        )
        assert (await child_records(env.client, identity)).operation_failures == []

        # And the other first attempt that leaves no row: one fenced on the way back in, after
        # its Activity answered and before anything it read could be recorded.
        held, sibling_child = starts[1], receipt.child_receipts[1]
        theirs = sibling_child.child_workflow_id
        their_episode = fork_preparation_operation_identity(FORK, theirs, 1)
        copy_closure(blobs, held.blob_root or "", child_closure(held))
        first_owner = await a_claimed_child(env.client, held, theirs)
        _HELD.update({"held": True, "running": False})
        preparing = asyncio.ensure_future(first_owner.prepare_child(fork_id=FORK))
        for _ in range(500):
            if _HELD.get("running"):
                break
            await asyncio.sleep(0.02)
        assert _HELD.get("running"), "the preparation never reached the point it holds at"
        replacement = await a_claimed_child(env.client, held, theirs)
        _HELD["held"] = False
        with pytest.raises(Exception) as fenced:
            await preparing
        assert protocol_error_code(fenced.value) == "fenced_writer"
        assert (await child_records(env.client, theirs)).operation_failures == []

        await hand_it_on(replacement, turnover_at, theirs, env.client)
        [still_owed] = [
            one
            for one in carried(await handed_on(env.client, theirs)).obligations
            if one.attempt_id == ATTEMPT
        ]
        assert (still_owed.pending_preparation, still_owed.preparation_epoch) == (True, 1)

        third = await a_claimed_child(env.client, held, theirs)
        theirs_ready = await third.prepare_child(fork_id=FORK)
        assert theirs_ready.ownership_epoch == 3
        assert theirs_ready.preparation_operation == their_episode
        assert (await child_records(env.client, theirs)).operation_failures == []
    _HELD.clear()


async def test_a_child_that_lost_its_manifest_names_the_object_it_could_not_produce(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The manifest is a second object lost in the same window, and it fails on its own terms.

    This is why a preparation reads the manifest again rather than calling the resolver bare: the
    resolver opens the store for the selected body alone, so a manifest lost after the claim would
    let a child publish readiness over a source object nobody could produce. The descriptor
    carried in state stays perfectly valid while that is true, which is the whole point of reading
    the store: a value in state is not a claim about a store.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(env.client, activities=a_closing_worker(world)):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-manifest/1"
        )
        close_the_capture()
        start, child = starts[0], receipt.child_receipts[0]
        identity = child.child_workflow_id
        store = FilesystemBlobStore(Path(start.blob_root or ""))
        frozen = fork_preparation_operation_identity(FORK, identity, 1)
        copy_closure(blobs, start.blob_root or "", child_closure(start))
        stream = await a_claimed_child(env.client, start, identity)
        [row] = [one for one in carried(start).attempts if one.attempt_id == ATTEMPT]
        commitment = row.source_commitment or ""
        store.path_for(commitment).unlink()

        with pytest.raises(Exception) as raised:
            await stream.prepare_child(fork_id=FORK)
        assert fork_refusal(raised.value) == FORK_REPAIRABLE_ABSENCE

        # Nothing was installed, nothing was marked and nothing was published: the child is where
        # it was, holding an eligible obligation and no body for it.
        records = await child_records(env.client, identity)
        [failure] = records.operation_failures
        assert failure.operation == frozen
        assert failure.references == [commitment]
        assert failure.outcome == REFUSED_OPERATION
        state = await stream.stream_state()
        assert state.obligations[ATTEMPT] == "eligible"
        assert state.payload_delivery_count == 0
        [attempt] = [one for one in records.attempts if one.attempt_id == ATTEMPT]
        assert receipt_availability(attempt, records.operation_failures) == (
            RECEIPT_UNAVAILABLE
        )

        # And the descriptor this child carries is exactly as valid as it was, because what went
        # missing is the object under its digest rather than anything about the value.
        manifest = carried_source(row)
        assert source_commitment(manifest) == commitment
        assert store.unverified([commitment]) == [commitment]

        # Repairing those exact bytes recovers under the identity the first attempt froze.
        copy_closure(blobs, start.blob_root or "", [commitment])
        second = await a_claimed_child(env.client, start, identity)
        ready = await second.prepare_child(fork_id=FORK)
        assert ready.preparation_operation == frozen
        assert ready.selected_body_reference == manifest.cells[child.target_cell].sha256
        records = await child_records(env.client, identity)
        assert [one.outcome for one in records.operation_failures] == [
            REFUSED_OPERATION,
            RECOVERED_OPERATION,
        ]


#: What one value of a preparation's result is replaced with before it comes back. The Worker
#: registers the substitute below instead of the real preparation, because two Activities of one
#: name are refused, and what it stands in for is a resolver this generation did not ask for.
_SUBSTITUTED: dict = {}

#: Which children this Worker actually ran a preparation for, in order. A preparation that never
#: reached an Activity is the point of a terminal episode, and a count is how that is observed
#: rather than inferred from the refusal that came back.
_PREPARATIONS: List[str] = []


@activity.defn(name=PREPARE_FORK_CHILD)
async def _a_substituted_preparation(request: ForkPreparationInput) -> ForkPreparation:
    """Read the objects for real, then move the one value under test on the way back."""
    _PREPARATIONS.append(request.payload.attempt_id)
    prepared = await fork_preparation_activity(request)
    if prepared.bundle is None or not _SUBSTITUTED:
        return prepared
    [candidate] = prepared.bundle.candidates
    return replace(
        prepared,
        bundle=replace(
            prepared.bundle, candidates=[replace(candidate, **_SUBSTITUTED)]
        ),
    )


async def test_a_child_handed_a_result_it_cannot_vouch_for_is_refused_for_good(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The two permanent halves of a preparation, and the repair neither of them becomes.

    A result naming a source other than the one this obligation was selected from is the wrong
    source; one naming this source and failing a check of what it is, is contract drift. Both are
    endings: the row is unrecoverable, no candidate is installed, the exact identifier keeps
    returning the refusal, and an owner that follows meets the same answer rather than a repair.

    An ending is a fact about the episode and not about the attempt that met it, so the owner that
    follows is answered from the record rather than allowed to make the decision again. That is
    the case a resolver put back in order would otherwise pass: the substitution is cleared before
    the retry here, so a preparation that ran would come back with a candidate every check admits,
    and what the child does instead is refuse without asking for one at all.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    substitutions = [
        ({"source_commitment": "a" * 64}, FORK_WRONG_SOURCE, WRONG_SOURCE),
        ({"resolver_id": "a resolver nobody registered"}, FORK_CONTRACT_DRIFT, CONTRACT_DRIFT),
    ]
    _SUBSTITUTED.clear()
    _PREPARATIONS.clear()
    async with stream_worker(
        env.client, activities=[*a_closing_worker(world), _a_substituted_preparation]
    ):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-refused/1"
        )
        close_the_capture()
        for start, child, (moved, refusal, reason) in zip(
            starts, receipt.child_receipts, substitutions
        ):
            identity = child.child_workflow_id
            frozen = fork_preparation_operation_identity(FORK, identity, 1)
            copy_closure(blobs, start.blob_root or "", child_closure(start))
            stream = await a_claimed_child(env.client, start, identity)
            _SUBSTITUTED.clear()
            _SUBSTITUTED.update(moved)

            with pytest.raises(Exception) as raised:
                await stream.prepare_child(fork_id=FORK)
            assert fork_refusal(raised.value) == refusal
            assert not fork_can_be_retried(raised.value)

            records = await child_records(env.client, identity)
            [row] = records.operation_failures
            assert (row.operation, row.phase, row.reason, row.outcome) == (
                frozen,
                FORK_PREPARATION,
                reason,
                UNRECOVERABLE_OPERATION,
            )
            assert (await stream.stream_state()).obligations[ATTEMPT] == "eligible"

            # It is an ending and not an absence, so an owner that follows meets it again rather
            # than repairing anything. The substitution is gone by then, so a preparation that ran
            # would be handed a candidate every check admits; the episode is over, so none runs,
            # and the row the ending wrote is still the only row there is.
            _SUBSTITUTED.clear()
            ran = len(_PREPARATIONS)
            second = await a_claimed_child(env.client, start, identity)
            with pytest.raises(Exception) as again:
                await second.prepare_child(fork_id=FORK)
            assert fork_refusal(again.value) == refusal
            assert not fork_can_be_retried(again.value)
            assert len(_PREPARATIONS) == ran
            outcomes = (await child_records(env.client, identity)).operation_failures
            assert [one.outcome for one in outcomes] == [UNRECOVERABLE_OPERATION]
            assert [one.refused_epoch for one in outcomes] == [1]
            assert {one.operation for one in outcomes} == {frozen}
            assert (await second.stream_state()).obligations[ATTEMPT] == "eligible"
    _SUBSTITUTED.clear()
    _PREPARATIONS.clear()


#: The size a preparation's result is grown to before it comes back, and the hold that catches one
#: owner on the way in. An empty mapping is the ordinary result, untouched.
_OVERSIZED: dict = {}

#: Which children this Worker actually ran a preparation for, so an episode that ended can be
#: observed to have run none rather than inferred from the refusal that came back.
_OVERSIZED_RUNS: List[str] = []


@activity.defn(name=PREPARE_FORK_CHILD)
async def _an_oversized_preparation(request: ForkPreparationInput) -> ForkPreparation:
    """Read the objects for real, and hand back a result too big for this child to carry."""
    _OVERSIZED_RUNS.append(request.payload.attempt_id)
    prepared = await fork_preparation_activity(request)
    if prepared.bundle is None or not _OVERSIZED:
        return prepared
    _OVERSIZED["running"] = True
    while _OVERSIZED.get("held"):
        await asyncio.sleep(0.02)
    [candidate] = prepared.bundle.candidates
    return replace(
        prepared,
        bundle=replace(
            prepared.bundle,
            candidates=[replace(candidate, body="x" * int(_OVERSIZED["size"]))],
        ),
    )


async def test_a_preparation_result_this_child_cannot_carry_ends_the_episode_it_opened(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The bytes a result comes back in are a completion like any other, classified in one place.

    A result too big for this child to carry is a decision about that result and never a fault of
    the call, so it is measured where the owner has just been checked again: the episode it ends
    is the one the child froze, the row it writes is the permanent one the declared failure model
    has for it, and an owner that follows meets the ending rather than making the decision again.

    The owner check is what comes first, and the sibling here is why. A writer a resume replaced
    while the store was being read is told that it was replaced, not that the result it never gets
    to see was too big, and it leaves nothing behind for the owner that replaced it.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    oversize = TURNOVER_PAYLOAD_CEILING_BYTES + 10_000
    _OVERSIZED.clear()
    _OVERSIZED_RUNS.clear()
    async with stream_worker(
        env.client, activities=[*a_closing_worker(world), _an_oversized_preparation]
    ):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-oversize/1"
        )
        close_the_capture()

        start, child = starts[0], receipt.child_receipts[0]
        identity = child.child_workflow_id
        frozen = fork_preparation_operation_identity(FORK, identity, 1)
        copy_closure(blobs, start.blob_root or "", child_closure(start))
        stream = await a_claimed_child(env.client, start, identity)
        _OVERSIZED.update({"size": oversize})

        with pytest.raises(Exception) as raised:
            await stream.prepare_child(fork_id=FORK)
        assert fork_refusal(raised.value) == FORK_CONTRACT_DRIFT
        assert not fork_can_be_retried(raised.value)
        assert "encodes to" in str(raised.value.cause)

        records = await child_records(env.client, identity)
        [row] = records.operation_failures
        assert (row.operation, row.phase, row.reason, row.outcome) == (
            frozen,
            FORK_PREPARATION,
            CONTRACT_DRIFT,
            UNRECOVERABLE_OPERATION,
        )
        assert row.refused_epoch == 1
        state = await stream.stream_state()
        assert state.obligations[ATTEMPT] == "eligible"
        assert state.payload_delivery_count == 0

        # It is an ending, so the owner that follows is answered from the record. The result is
        # ordinary again by then, and no preparation runs at all.
        _OVERSIZED.clear()
        ran = len(_OVERSIZED_RUNS)
        second = await a_claimed_child(env.client, start, identity)
        with pytest.raises(Exception) as again:
            await second.prepare_child(fork_id=FORK)
        assert fork_refusal(again.value) == FORK_CONTRACT_DRIFT
        assert len(_OVERSIZED_RUNS) == ran
        outcomes = (await child_records(env.client, identity)).operation_failures
        assert [one.outcome for one in outcomes] == [UNRECOVERABLE_OPERATION]

        # The sibling: an owner replaced while its Activity ran is told it was replaced, and the
        # oversize it never saw leaves no row of its own behind.
        held, theirs_child = starts[1], receipt.child_receipts[1]
        theirs = theirs_child.child_workflow_id
        their_episode = fork_preparation_operation_identity(FORK, theirs, 1)
        copy_closure(blobs, held.blob_root or "", child_closure(held))
        first_owner = await a_claimed_child(env.client, held, theirs)
        _OVERSIZED.update({"size": oversize, "held": True, "running": False})
        preparing = asyncio.ensure_future(first_owner.prepare_child(fork_id=FORK))
        for _ in range(500):
            if _OVERSIZED.get("running"):
                break
            await asyncio.sleep(0.02)
        assert _OVERSIZED.get("running"), "the preparation never reached the point it holds at"
        replacement = await a_claimed_child(env.client, held, theirs)
        _OVERSIZED["held"] = False
        with pytest.raises(Exception) as fenced:
            await preparing
        assert protocol_error_code(fenced.value) == "fenced_writer"
        assert (await child_records(env.client, theirs)).operation_failures == []

        # And the owner that replaced it is the one the same oversize is recorded under, in the
        # episode the fenced attempt opened rather than in one of its own.
        with pytest.raises(Exception) as ended:
            await replacement.prepare_child(fork_id=FORK)
        assert fork_refusal(ended.value) == FORK_CONTRACT_DRIFT
        [their_row] = (await child_records(env.client, theirs)).operation_failures
        assert their_row.operation == their_episode
        assert (their_row.refused_epoch, their_row.outcome) == (2, UNRECOVERABLE_OPERATION)
    _OVERSIZED.clear()
    _OVERSIZED_RUNS.clear()


#: The object a store loses in the window between a preparation's two reads, or nothing.
_LOST: dict = {}


class ALosingStore(FilesystemBlobStore):
    """A store that loses one object between the read that checks it and the read that uses it.

    A preparation checks both objects and then resolves the selected body, which opens the store a
    second time. The interval between those two reads is a real one, and this is how it is driven:
    the check answers truthfully that nothing is missing, and the object is gone by the time the
    resolver asks for it.
    """

    def unverified(self, digests: Any) -> List[str]:
        answer = super().unverified(digests)
        losing = _LOST.get("reference")
        if not answer and losing:
            self.path_for(str(losing)).unlink(missing_ok=True)
        return answer


async def test_a_body_lost_between_a_preparations_two_reads_is_the_absence_a_repair_answers(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any, monkeypatch: Any
) -> None:
    """Valid committed bytes lost after the claim and before derivation reads them.

    The claim, the grade, the source and the selection are all intact and only the object is gone,
    which is the repairable case: the exact bytes are reinstalled from a verified copy and the
    same logical preparation runs again under the owner that follows. Where in the operation the
    loss falls does not change what it is, so a body that disappeared between the check and the
    resolver comes back as the name that could not be produced rather than as a failure of the
    call, which nothing would classify and no row would name.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    _LOST.clear()
    monkeypatch.setattr(kernel_activities, "FilesystemBlobStore", ALosingStore)
    async with stream_worker(env.client, activities=a_closing_worker(world)):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-lost-body/1"
        )
        close_the_capture()
        start, child = starts[0], receipt.child_receipts[0]
        identity = child.child_workflow_id
        frozen = fork_preparation_operation_identity(FORK, identity, 1)
        copy_closure(blobs, start.blob_root or "", child_closure(start))
        stream = await a_claimed_child(env.client, start, identity)

        # The claim read its own set back with the body present, and the body goes missing in the
        # window inside the preparation rather than before it.
        body = selected_reference(start)
        store = FilesystemBlobStore(Path(start.blob_root or ""))
        assert store.unverified([body]) == []
        _LOST["reference"] = body

        with pytest.raises(Exception) as raised:
            await stream.prepare_child(fork_id=FORK)
        assert fork_refusal(raised.value) == FORK_REPAIRABLE_ABSENCE
        assert fork_can_be_retried(raised.value)
        assert body in str(raised.value.cause)
        _LOST.clear()
        assert store.unverified([body]) == [body]

        records = await child_records(env.client, identity)
        [failure] = records.operation_failures
        assert (failure.operation, failure.reason, failure.outcome) == (
            frozen,
            UNAVAILABLE_EVIDENCE,
            REFUSED_OPERATION,
        )
        assert failure.references == [body]
        assert (await stream.stream_state()).obligations[ATTEMPT] == "eligible"
        [attempt] = [one for one in records.attempts if one.attempt_id == ATTEMPT]
        assert receipt_availability(attempt, records.operation_failures) == (
            RECEIPT_UNAVAILABLE
        )

        # Those exact bytes put back, under the owner that follows, is the whole repair.
        copy_closure(blobs, start.blob_root or "", [body])
        second = await a_claimed_child(env.client, start, identity)
        ready = await second.prepare_child(fork_id=FORK)
        assert ready.preparation_operation == frozen
        assert ready.selected_body_reference == body
        assert [
            one.outcome
            for one in (await child_records(env.client, identity)).operation_failures
        ] == [REFUSED_OPERATION, RECOVERED_OPERATION]
    _LOST.clear()


async def test_a_prepared_child_crosses_its_own_boundary_and_serves_the_body_it_built(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """A child's own later continuation, which is a different entry from the one it was cut at.

    The lineage crosses as immutable provenance and authorizes nothing about the carry on the far
    side: what authorizes that is the service's own continuation fact, and the identity inside the
    carrier is the child's own by then rather than the parent's. Nothing asks the parent again,
    because the comparison of a child against the parent it was cut from is made once, at the
    entry that was cut.

    The body the child built crosses with it, the gate its preparation cleared stays cleared, and
    what it serves afterwards is the same bytes it would have served before.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(env.client, activities=a_closing_worker(world)):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-crossed/1"
        )
        close_the_capture()
        start, child = starts[0], receipt.child_receipts[0]
        identity = child.child_workflow_id
        stream, ready = await a_ready_child(env.client, start, identity, blobs)
        caller = await a_child_caller(stream, start)
        await cross_a_boundary(caller, turnover_at, identity, env.client)
        crossed = await handed_on(env.client, identity)

        # The lineage is retained exactly, and the carry on the far side is the child's own.
        assert crossed.fork_origin == start.fork_origin
        assert crossed.carry is not None
        assert crossed.carry.carrier_schema_version == FORK_CARRIER_SCHEMA_VERSION
        projection = unpack_carrier(crossed.carry, CONVERTER)
        assert projection.configuration_hash == configuration_hash(crossed)
        assert projection.configuration_hash != carried(start).configuration_hash
        assert projection.configuration_hash != start.fork_origin.parent_configuration_hash

        # The body it built crossed with it, and the gate its preparation cleared stays cleared.
        [owed] = [one for one in projection.obligations if one.attempt_id == ATTEMPT]
        assert owed.candidate is not None
        assert owed.pending_preparation is False
        assert owed.candidate.inner_sha256 == ready.selected_body_reference

        # And the continued execution asks the parent nothing at all. The ordinal it carries over
        # is the one it was cut at, because everything the child did on the near side was work no
        # unforked twin does and every one of those reads took a fork identifier.
        assert not [
            one
            for one in await scheduled_activities(env.client, identity)
            if one.startswith("fork.")
        ]
        assert projection.activity_ordinal == carried(start).activity_ordinal

        payload = await caller.pull()
        assert payload.kind == "payload"
        assert sha256(body_of(payload).encode("ascii")).hexdigest() == (
            ready.selected_body_reference
        )
        # The mark that says this execution read the objects behind that delivery is this
        # execution's own, so the first dependent delivery after a boundary reads them again.
        state = await stream.stream_state()
        assert state.verification_batches == projection.verification_batches + 1


async def test_a_child_claims_over_the_closure_it_was_given_and_never_over_less(
    env: Any, world: ServedEpisode, tmp_path: Path, turnover_at: Any
) -> None:
    """The child's first claim is the acceptance test for the copy that was made for it.

    The verification path refuses a claim over references the store cannot produce, so a closure
    that arrived short is caught before the child owns anything, and the row says whose evidence
    went missing. Reinstalling those exact bytes recovers the same claim rather than creating
    anything: no child is deleted here and none is replaced.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    async with stream_worker(env.client, activities=a_closing_worker(world)):
        _request, receipt, starts = await forked(
            env, world, composed, blobs, tmp_path, "stream/child-closure/1"
        )
        close_the_capture()
        start, child = starts[0], receipt.child_receipts[0]
        identity = child.child_workflow_id
        [row] = [one for one in carried(start).attempts if one.attempt_id == ATTEMPT]
        evidence = row.graded_evidence or ""
        copy_closure(
            blobs,
            start.blob_root or "",
            [one for one in child_closure(start) if one != evidence],
        )
        with pytest.raises(Exception) as raised:
            await a_claimed_child(env.client, start, identity)
        assert protocol_error_code(raised.value) == "invalid_message"

        records = await child_records(env.client, identity)
        [refused] = records.operation_failures
        assert refused.reason == UNAVAILABLE_EVIDENCE
        assert refused.outcome == REFUSED_OPERATION
        assert refused.attempt_id == ATTEMPT
        assert refused.references == [evidence]
        assert refused.generation == identity

        copy_closure(blobs, start.blob_root or "", [evidence])
        stream, ready = await a_ready_child(env.client, start, identity, blobs)
        assert ready.child_workflow_id == identity
        records = await child_records(env.client, identity)
        assert [one.outcome for one in records.operation_failures] == [
            REFUSED_OPERATION,
            RECOVERED_OPERATION,
        ]
        [attempt] = [one for one in records.attempts if one.attempt_id == ATTEMPT]
        assert receipt_availability(attempt, records.operation_failures) == RECEIPT_AVAILABLE


# Two children of one fork, working the second task at the same time, in worlds of their own.
#
# One Worker serves a whole run and an environment registers each Activity once under a fixed
# name, so the two children reach one seal implementation. What tells their worlds apart is the
# generation the call was scheduled for, and everything below is about that.


def a_routed_worker(world: ServedEpisode, route: Any) -> List[Any]:
    """This environment's own Activities over a route keyed by generation as well as attempt."""
    _version, activities, _digest = world.env.protocol_v2_terminal(route)
    return list(activities)


async def a_world_at(bundle: Path, root: Path, position: int, name: str) -> ServedEpisode:
    """One more served world of this environment, opened on the position it names."""
    return await ServedEpisode.start(
        "receipts_v1",
        task=position,
        env_config={"bundle": str(bundle)},
        ends_on_horizon=False,
        trace_path=root / f"{name}.jsonl",
    )


async def test_two_sibling_worlds_run_under_one_worker_and_capture_their_own_filings(
    env: Any, world: ServedEpisode, frozen_bundle: Path, tmp_path: Path, turnover_at: Any
) -> None:
    """Both children work the second task at once, and each seals in the world it worked in.

    The two children inherit the whole prefix, so they carry one public identifier for that task
    between them and both are working it at the same time. They also take a first claim each, so
    the owners are at one epoch and neither is older than the other. Routed by the attempt alone
    the second world recorded would replace the first and both seals would read whichever world
    was written last; routed by the generation as well, each seal reads the world its own child
    opened.

    The two worlds are opened on different positions of this environment on purpose. What a seal
    captures is read off the world it resolves to, so worlds that hold the same instance capture
    the same thing however badly they were routed, and the number is only evidence when the two
    worlds are different. The filing is the one that answers the first of them exactly, so the
    child working that world scores the whole of it and the child working the other does not.

    One of them is taken over on the way, because ownership is the half of the key that stays
    inside a generation: the replacement's world is newer than the world it replaced and older
    than nothing at all in its sibling.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs, silent=True))
    route = WorldRoute()
    parent = "stream/sibling-worlds/1"
    route.record(parent, ATTEMPT, world, 1)
    opened: List[ServedEpisode] = []
    try:
        async with stream_worker(env.client, activities=a_routed_worker(world, route)):
            _request, receipt, starts = await forked(
                env, world, composed, blobs, tmp_path, parent
            )
            callers: List[Tuple[str, Caller]] = []
            for index, (start, child) in enumerate(zip(starts, receipt.child_receipts)):
                identity = child.child_workflow_id
                stream, _ready = await a_ready_child(env.client, start, identity, blobs)
                caller = await a_child_caller(stream, start)
                await caller.present(await caller.pull())
                task = await caller.pull()
                assert task.attempt_id == SILENT
                await caller.present(task)
                own = await a_world_at(frozen_bundle, tmp_path, index, f"sibling-{index}")
                opened.append(own)
                route.record(identity, SILENT, own, 1)
                callers.append((identity, caller))

            # Two worlds, live at once, one attempt identifier, and each child resolves its own.
            assert len({one.session_id for one in opened}) == 2
            for (identity, _caller), own in zip(callers, opened):
                assert route.resolve(identity, SILENT) == (own.env, own.session_id)

            # A transport of one child coming back late with the world it was working in is
            # ordered against that child's own entry and never against its sibling's.
            route.record(callers[0][0], SILENT, opened[1], 0)
            assert route.resolve(callers[0][0], SILENT) == (opened[0].env, opened[0].session_id)

            # One child's transport is replaced, and the replacement opens a world of its own for
            # the attempt it inherited. The epoch orders the two of them inside that child, the
            # transport that was replaced cannot write its old world back, and the sibling working
            # the same attempt identifier is where it was throughout.
            taken = callers[1][0]
            again = await a_world_at(frozen_bundle, tmp_path, 1, "sibling-1-again")
            opened.append(again)
            route.record(taken, SILENT, again, 2)
            route.record(taken, SILENT, opened[1], 1)
            assert route.resolve(taken, SILENT) == (again.env, again.session_id)
            assert route.resolve(callers[0][0], SILENT) == (opened[0].env, opened[0].session_id)

            answer = filing_of(world.env)
            for identity, caller in callers:
                acknowledged = await caller.seal(answer, attempt_id=SILENT)
                assert acknowledged.kind == "seal_ack"
                await caller.present(acknowledged)

            scored = {}
            for identity, _caller in callers:
                records = await child_records(env.client, identity)
                scored[identity] = {row.attempt_id: row for row in records.attempts}

                # And the rows say which generation each of them belongs to. The first task is
                # the parent's work in both children, and the second is each child's own.
                assert records.origin is not None
                assert records.origin.parent_workflow_id == parent
                assert records.origin.fork_id == FORK
                assert scored[identity][ATTEMPT].source_generation == parent
                assert scored[identity][SILENT].source_generation == identity
                assert scored[identity][ATTEMPT].payload_delivered

            first, second = (identity for identity, _caller in callers)
            assert scored[first][SILENT].score == 1.0
            assert scored[second][SILENT].score != scored[first][SILENT].score

            # The first task was sealed once, by the parent, and both children carry that one
            # seal rather than a filing of their own.
            assert (
                scored[first][ATTEMPT].submission_digest
                == scored[second][ATTEMPT].submission_digest
            )
            assert scored[first][SILENT].seal_ordinal == scored[second][SILENT].seal_ordinal

            # A lineage totals its work by the generation each row's work belongs to. The
            # first task was worked once and both children hold it; the second was worked twice
            # and each child holds its own. Summing what each generation reports would count the
            # shared prefix once per generation instead.
            work: Dict[str, set] = {}
            for identity, _caller in callers:
                for row in scored[identity].values():
                    assert row.source_generation is not None
                    work.setdefault(row.source_generation, set()).add(row.attempt_id)
            assert work[parent] == {ATTEMPT}
            assert [work[identity] for identity, _caller in callers] == [{SILENT}, {SILENT}]

            # And what each child delivered against the inherited work is its own delivery.
            assert (
                scored[first][ATTEMPT].payload_message_id
                == scored[second][ATTEMPT].payload_message_id
            )
            assert (
                scored[first][ATTEMPT].payload_visible_sha256
                != scored[second][ATTEMPT].payload_visible_sha256
            )

            # Letting one child's world go leaves its sibling's where it is.
            route.forget(callers[0][0], SILENT, opened[0])
            assert route.resolve(callers[0][0], SILENT) is None
            assert route.resolve(callers[1][0], SILENT) is not None
    finally:
        for one in opened:
            await one.close()


async def test_a_child_is_attached_to_a_directory_of_its_own_under_one_shared_runtime(
    env: Any, world: ServedEpisode, frozen_bundle: Path, tmp_path: Path, turnover_at: Any
) -> None:
    """The sibling of the door a generation is opened by, for one that already exists.

    A fork creates its children before anything serves them, so there is no stream to start and
    no composition to complete: what a controller does is prepare the directory, copy the objects
    in, and attach. The directory is adopted rather than made fresh, because publishing a
    manifest, installing a closure and claiming ownership are three steps a crash can be between
    and the repair for any of them is to do it again.

    What the manifest then says is what a reader of that directory alone has to go on: which
    generation lives here, where the history it shares is, and which generation it was cut from.
    And closing this transport is a transport closing: the run's service and its Worker are the
    run's, so the sibling that has not been attached to yet comes up afterwards.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    turnover_at(10_000)
    composed = fork_capable(start_for(world, contract, blobs))
    route = WorldRoute()
    parent = "stream/attached-child/1"
    route.record(parent, ATTEMPT, world, 1)
    version, activities, digest = world.env.protocol_v2_terminal(route)
    environment = EnvironmentTerminal(version, list(activities), digest, route, RECEIPTS_GRADE)
    attached: Optional[ServedEpisode] = None
    try:
        async with stream_worker(env.client, activities=list(activities)):
            _request, receipt, starts = await forked(
                env, world, composed, blobs, tmp_path, parent
            )
            start, child = starts[0], receipt.child_receipts[0]
            identity = child.child_workflow_id
            await a_ready_child(env.client, start, identity, blobs)

            attached = await a_world_at(frozen_bundle, tmp_path, 0, "attached")
            # The directory is derived from the store the authenticated start names rather than
            # passed beside it, because a directory handed in would be a second opinion about
            # where this generation lives.
            directory = run_directory_of(start.blob_root or "")
            gateway = await attach_gateway(
                env.client,
                attached,
                workflow_id=identity,
                start=start,
                run_directory=directory,
                database_root="..",
                consumer_id="the-transport-of-this-child",
                environment=environment,
            )

            # The directory now says the three things a reader of it needs.
            held = open_run_directory(directory)
            assert held.manifest.workflow_id == identity
            assert held.manifest.database_root == ".."
            assert held.manifest.origin is not None
            assert held.manifest.origin.parent_workflow_id == parent
            assert held.manifest.origin.fork_id == FORK
            assert held.manifest.origin.branch_slot == start.served_slot
            assert held.database_root == tmp_path.resolve()

            # And the objects installed for this child are the ones the reopened directory's own
            # accessor produces, rather than ones it reads past: the store the authenticated start
            # names and the store a reader resolves from the directory are one place.
            body = held.blobs.read(child.selected_body_reference)
            assert sha256(body).hexdigest() == child.selected_body_reference
            assert held.blobs.root == Path(start.blob_root or "")

            # Its first act is an ordinary pull, and what it answers is the body its parent
            # selected for it.
            payload = json.loads(await gateway.pull({}))
            assert payload["kind"] == "payload"
            assert payload["attempt_id"] == ATTEMPT

            # Attaching again adopts what is there rather than refusing it or making a second.
            again = await attach_gateway(
                env.client,
                attached,
                workflow_id=identity,
                start=start,
                run_directory=directory,
                database_root="..",
                consumer_id="the-transport-of-this-child",
                environment=environment,
            )
            assert open_run_directory(directory).manifest == held.manifest
            await again.aclose()

            # And a repeat that names no consumer adopts the one this generation holds rather
            # than minting a second. The documented default is what a caller recovering a lost
            # reply sends, and a fresh logical consumer is the one thing the generation will not
            # take: minting one would fence the transport this call had just installed.
            standing = (await gateway.stream_state()).consumer_id
            assert standing == "the-transport-of-this-child"
            recovered = await attach_gateway(
                env.client,
                attached,
                workflow_id=identity,
                start=start,
                run_directory=directory,
                database_root="..",
                environment=environment,
            )
            assert (await recovered.stream_state()).consumer_id == standing
            await recovered.aclose()

            # A whole lineage copied somewhere else reads its objects back through the copy, which
            # is the read relocation promises and the one the reader resolves the same way.
            elsewhere = tmp_path / "archive"
            shutil.copytree(directory, elsewhere / directory.name)
            moved = open_run_directory(elsewhere / directory.name)
            assert moved.blobs.read(child.selected_body_reference) == body

            # A generation moved somewhere else is one this build reads and never serves.
            with pytest.raises(ValueError):
                await attach_gateway(
                    env.client,
                    attached,
                    workflow_id=identity,
                    start=start,
                    run_directory=tmp_path / "moved",
                    environment=environment,
                )

            # And closing a transport of one generation is a transport closing: the sibling is
            # brought up afterwards on the same service and the same Worker.
            other = starts[1]
            _stream, ready = await a_ready_child(
                env.client, other, receipt.child_receipts[1].child_workflow_id, blobs
            )
            assert ready.fork_id == FORK
    finally:
        if attached is not None:
            await attached.close()
