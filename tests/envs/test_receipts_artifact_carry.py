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
from dataclasses import replace
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Optional, Sequence

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
    read_source_artifact,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.gateway import install_policies  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    CONFIRMED_EXISTING,
    EXISTENCE_UNCONFIRMED,
    FORK_COMPLETE,
    FORK_CONFLICTED,
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
    ForkAvailability,
    ForkAvailabilityInput,
    ForkChildPlan,
    ForkChildStarted,
    ForkRequest,
    GeneratePayloadBundleInput,
    OfferedMessage,
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
    fork_can_be_retried,
    fork_refusal,
    fork_status,
    fork_stream,
    generate_payload_bundle_activity,
    protocol_error_code,
    resume_stream,
    start_fork_child_activity,
    start_stream,
    stream_worker,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    FORK_AVAILABILITY,
    START_FORK_CHILD,
    STREAM_WORKFLOW_TYPE,
    VERIFY_BLOBS,
)
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.workflow import StreamWorkflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    FORK_ORIGIN_DISAGREEMENT,
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
    before = (await caller.stream.stream_state()).turnovers
    turnover_at(await accepted_updates(client, workflow_id) + 1)
    await caller.stream.confirm_state()
    for _ in range(400):
        if (await caller.stream.stream_state()).turnovers > before:
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
