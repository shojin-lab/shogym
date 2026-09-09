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
from hashlib import sha256
from pathlib import Path
from typing import Any, List

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio.api.enums.v1 import EventType  # noqa: E402
from temporalio.client import Client, WorkflowUpdateFailedError  # noqa: E402
from temporalio.converter import default as default_converter  # noqa: E402

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
    SourceArtifactManifest,
    mask_spans,
    masked_body,
    read_source_artifact,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.gateway import install_policies  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    RECEIPT_CARRIER_SCHEMA_VERSION,
    ConsumerClaim,
    GeneratePayloadBundleInput,
    OfferedMessage,
    SealRequest,
    SourceOriginContext,
    StreamStart,
    TaskItem,
    TerminalTool,
    assignments_for,
    configuration_hash,
    derived_selection,
    generate_payload_bundle_activity,
    protocol_error_code,
    resume_stream,
    start_stream,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
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
