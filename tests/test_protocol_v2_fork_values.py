"""The pure values one fork is made of: what is written, what is digested, what is refused.

A fork is a controller operation over two generations, and everything it commits to before it
fences anything is a value. This file is about those values and nothing else: the checkpoint the
harness attests its own freeze with, the typed request a controller submits, the record the parent
keeps, the receipt it answers with, the digests each of them is named by, the identities the
children are derived under, the bounded difference a child's configuration may be from its
parent's, the counters a child starts from, and the measurements taken before a barrier could
commit. Nothing here opens a store, starts a service or reaches a world.

Two rules hold the digests together and both are tested rather than described. Every preimage is
one domain tag, a zero byte and the canonical encoding of what is covered, so bytes built for one
purpose are never a preimage for another. And the canonical encoding refuses a number that is not
whole, while these records hold real ones, so each covered position is named in a versioned table
and written as its exact text: a position the table does not reach is denied rather than rounded.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping

import pytest

pytest.importorskip("temporalio")

from temporalio.converter import default as default_converter  # noqa: E402

from shogym.envs.receipts.protocol_v2 import RECEIPTS_GRADE  # noqa: E402
from shogym.serve.protocol_v2 import IMMEDIATE, TASK_FIRST, PresentationAck  # noqa: E402
from shogym.serve.protocol_v2.artifact import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    BODY_MEDIA_TYPE,
    CELL_KINDS,
    GRADED_CELL,
    ORACLE_CELL,
    PLACEBO_CELL,
    PairParity,
    ReceiptContract,
    SourceArtifactManifest,
    manifest_fields,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import BlobRef  # noqa: E402
from shogym.serve.protocol_v2.errors import WireFormatError  # noqa: E402
from shogym.serve.protocol_v2.kernel import messages as kernel_messages  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    ABANDONED_REASONS,
    CARRIER_ENCODING,
    CHECKPOINT_MANIFEST_ENCODING,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    CHILD_READY_SCHEMA_VERSION,
    CHILD_START_FIELDS,
    CONFIRMED_EXISTING,
    DIFFERENCE_TAG,
    EQUAL_IN_A_CHILD,
    EXISTENCE_UNCONFIRMED,
    FORK_ABANDONED,
    FORK_CARRIER_SCHEMA_VERSION,
    FORK_CHILD_COUNT,
    FORK_COMPLETE,
    FORK_ID_TOKEN,
    FORK_NUMERIC_FIELDS,
    FORK_ORIGIN_SCHEMA_VERSION,
    FORK_PREPARED,
    FORK_RECEIPT_SCHEMA_VERSION,
    FORK_REQUEST_SCHEMA_VERSION,
    ORIGIN_TAG,
    PREPARED_FORK_SCHEMA_VERSION,
    RUN_ID_CEILING_BYTES,
    SPENT_RECOVERY_RESERVE,
    START_TAG,
    AttemptFinalized,
    CarriedAttempt,
    CarriedAttestation,
    CarriedBinding,
    CarriedFinalization,
    CarriedObligation,
    CarriedProjection,
    CheckpointComponent,
    CheckpointManifest,
    ChildReady,
    ForkChildPlan,
    ForkChildReceipt,
    ForkOrigin,
    ForkReceipt,
    ForkRequest,
    OfferedMessage,
    OperationFailure,
    PayloadCandidate,
    PreparedChild,
    PreparedFork,
    PresentedMessage,
    SettledHarness,
    SourceOriginContext,
    StartDifference,
    StreamCarry,
    StreamStart,
    TaskItem,
    TerminalTool,
    check_checkpoint_manifest,
    check_child_configuration,
    check_child_ready,
    check_complete_start_authorization,
    check_fork_origin,
    check_fork_receipt,
    check_fork_request,
    check_prepared_fork,
    check_receipt_within_bound,
    check_start_classification,
    assignments_for,
    check_transmitted_size,
    checkpoint_manifest_reference,
    child_workflow_id,
    complete_start_digest,
    configuration_hash,
    encoded_size,
    fork_preimage,
    fork_receipt_bound,
    fork_request_digest,
    origin_digest,
    origin_fields,
    start_difference_projection,
)
from shogym.serve.protocol_v2.kernel.workflow import (  # noqa: E402
    TURNOVER_PAYLOAD_CEILING_BYTES,
    ChildSelection,
    child_carrier,
    transformed_child_counters,
    transformed_child_selection,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    DELIVER,
    EXPERIMENT,
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    PLATFORM_DEFAULT,
    REGISTERED,
    SINGLETON_SLOT,
    MatchedFamily,
    PayloadDisposition,
    PolicyProvenance,
    roster_digest,
)

ATTEMPT = "b" * 32
PARENT_ID = "stream/run-1/gen-1"
PARENT_RUN = "11111111-1111-4111-8111-111111111111"
PARENT_EXECUTION = "execution-parent"
NAMESPACE = "default"
FORK = "the-first-fork"
CONTRACT = "ledger_receipt"
SOURCE = "a" * 64
RENDERER = "receipts-render-v1"
# The branches the parent declares its fork may create. Both are internal names bound to the two
# child plans, and neither is a word this generation serves.
FIRST_BRANCH = "branch-one"
SECOND_BRANCH = "branch-two"
CONVERTER = default_converter().payload_converter

CONTRACT_SHAPE = ReceiptContract(
    contract_id=CONTRACT,
    match_group="kernel",
    cells=(
        (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
        (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
    ),
    body_size=40,
    body_encoding="ascii",
    media_type=BODY_MEDIA_TYPE,
    rows_start=0,
    row_line_width=10,
    row_count=4,
    slot_spans=((2, 5),),
    manifest_schema_versions=(ARTIFACT_SCHEMA_VERSION,),
    renderer_configuration=RENDERER,
)


def oid(value: int) -> str:
    return f"{value:032x}"


#: A matched arm and a written-out roster, so the two fields that default to empty can be moved.
A_FAMILY = MatchedFamily(
    family_id=CONTRACT,
    match_group="kernel",
    cells=((GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),),
    visible_byte_count=40,
)


@contextmanager
def _classified_as(table: Mapping[str, str]) -> Iterator[None]:
    """Run under another classification, so the completeness check has something to fail on."""
    original = kernel_messages.CHILD_START_FIELDS
    kernel_messages.CHILD_START_FIELDS = dict(table)
    try:
        yield
    finally:
        kernel_messages.CHILD_START_FIELDS = original


def a_row(*, slot: str, policy: str, cell: str) -> PayloadDisposition:
    """The one obligation this generation owes, resolved on one branch."""
    return PayloadDisposition(
        attempt_id=ATTEMPT,
        payload_position=0,
        branch_slot=slot,
        kind=DELIVER,
        policy_digest=policy,
        cell=cell,
        resolution_source=REGISTERED,
        family_id=CONTRACT,
    )


def a_parent() -> StreamStart:
    """One fork-capable generation: it serves the singleton branch and declares two more."""
    rows = [
        a_row(
            slot=SINGLETON_SLOT,
            policy=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
            cell=GRADED_CELL,
        )
    ]
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash="d" * 64,
        initial_cursor=oid(1),
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id=PARENT_EXECUTION,
        canonicalization_version="kernel.1",
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=[]
        ),
        tasks=[
            TaskItem(
                task_position=0,
                attempt_id=ATTEMPT,
                task_message_id=oid(0x101),
                ack_message_id=oid(0x102),
                payload_position=0,
                payload_message_id=oid(0x103),
                body="file the report",
            )
        ],
        release=IMMEDIATE,
        blob_root="/runs/one/parent/blobs",
        profile=EXPERIMENT,
        grade=RECEIPTS_GRADE,
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_subject_of_this_run",
        ),
        receipt_contracts=[CONTRACT_SHAPE],
        receipt_source=SOURCE,
        forkable_slots=[FIRST_BRANCH, SECOND_BRANCH],
    )


# What each child of the one fork here delivers: its branch, the cell it is served from the
# source both inherit, and the policy its own row resolves to.
CHILD_PLANS = (
    (FIRST_BRANCH, GRADED_CELL, GRADED_RECEIPT_ARTIFACT_V1_DIGEST),
    (SECOND_BRANCH, PLACEBO_CELL, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST),
)


def a_carry() -> StreamCarry:
    """A carrier at the fork's own version. What is inside it is another file's subject."""
    return StreamCarry(
        carrier_schema_version=FORK_CARRIER_SCHEMA_VERSION,
        encoding=CARRIER_ENCODING,
        data="AAECAwQ",
    )


def a_checkpoint(**changes: Any) -> CheckpointManifest:
    """The harness's witness that it froze: one transcript, one entry, two components."""
    transcript = "transcript/parent/000042"
    declared = CheckpointManifest(
        transcript_reference=transcript,
        acknowledgement_locator=f"{transcript}#17",
        acknowledgement_entry_sha256="8" * 64,
        components=[
            CheckpointComponent(
                component_id="work-tree",
                sha256="9" * 64,
                size=4096,
                media_type="application/x-tar",
                restores_transcript=transcript,
            ),
            CheckpointComponent(
                component_id="home",
                sha256="a" * 64,
                size=2048,
                media_type="application/x-tar",
                restores_transcript=transcript,
            ),
        ],
        adapter_version="harness-adapter-3",
        container_image_digest="b" * 64,
        harness_configuration="c" * 64,
        settled=SettledHarness(
            transport=True, recovery=True, provider=True, model=True, compaction=True
        ),
        frozen_plan_digest="d" * 64,
        acknowledgement_message_id=oid(0x102),
        acknowledged_visible_sha256="1" * 64,
        acknowledged_cursor=oid(0x102),
        projection_digest="e" * 64,
    )
    return replace(declared, **changes) if changes else declared


def a_child(ordinal: int = 0, **changes: Any) -> StreamStart:
    """One child of that parent, complete: its own identity, branch, rows, carry and origin.

    It is built in two steps because the origin records the child's own configuration hash and
    rides outside it: the start is composed first, hashed, and then handed the lineage record.
    """
    branch, cell, policy = CHILD_PLANS[ordinal]
    rows = [a_row(slot=branch, policy=policy, cell=cell)]
    composed = replace(
        a_parent(),
        consumer_claim_hash=sha256(f"claims {branch}".encode("utf-8")).hexdigest(),
        hidden_execution_id=f"execution-child-{ordinal}",
        blob_root=f"/runs/one/child-{ordinal}/blobs",
        served_slot=branch,
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_subject_of_this_run",
        ),
        carry=a_carry(),
    )
    complete = replace(composed, fork_origin=an_origin(child=composed, ordinal=ordinal))
    return replace(complete, **changes) if changes else complete


def an_origin(*, child: StreamStart, ordinal: int, **changes: Any) -> ForkOrigin:
    """Where that child was cut from, with the source seal's own identity carried whole."""
    declared = ForkOrigin(
        parent_workflow_id=PARENT_ID,
        parent_run_id=PARENT_RUN,
        parent_configuration_hash=configuration_hash(a_parent()),
        child_configuration_hash=configuration_hash(child),
        parent_execution_ordinal=0,
        parent_hidden_execution_id=PARENT_EXECUTION,
        acknowledged_cursor=oid(0x102),
        projection_digest="e" * 64,
        attestation_id="f" * 64,
        acknowledged_visible_sha256="1" * 64,
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_checkpoint()),
        source_attempt_id=ATTEMPT,
        source_seal_id="2" * 64,
        source_submission_digest="3" * 64,
        source_canonicalization_version="kernel.1",
        source_score=0.8154321,
        source_seal_ordinal=1,
        source_graded_evidence="4" * 64,
        source_commitment="5" * 64,
        source_artifact_references=["6" * 64, "7" * 64],
        branch_slot=child.served_slot,
        dispositions_digest=roster_digest(child.dispositions),
        start_differences=start_difference_projection(a_parent(), child),
        parent_turnovers=2,
        fork_id=FORK,
        child_ordinal=ordinal,
        children=FORK_CHILD_COUNT,
    )
    return replace(declared, **changes) if changes else declared


def a_plan(ordinal: int, **changes: Any) -> ForkChildPlan:
    """The child plan a controller submits for that child, and nothing about the experiment."""
    branch, cell, _ = CHILD_PLANS[ordinal]
    child = a_child(ordinal)
    declared = ForkChildPlan(
        branch_slot=branch,
        dispositions=list(child.dispositions),
        target_cell=cell,
        run_directory=f"/runs/one/child-{ordinal}",
        consumer_claim_hash=child.consumer_claim_hash,
        hidden_execution_id=child.hidden_execution_id,
        frozen_plan_digest="d" * 64,
    )
    return replace(declared, **changes) if changes else declared


def a_request(**changes: Any) -> ForkRequest:
    """One typed fork, carrying both witnesses, the source attempt and the ordered plans."""
    declared = ForkRequest(
        parent_workflow_id=PARENT_ID,
        parent_run_id=PARENT_RUN,
        parent_execution_ordinal=0,
        parent_configuration_hash=configuration_hash(a_parent()),
        source_attempt_id=ATTEMPT,
        attestation_id="f" * 64,
        acknowledgement_message_id=oid(0x102),
        acknowledged_visible_sha256="1" * 64,
        acknowledged_cursor=oid(0x102),
        projection_digest="e" * 64,
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_checkpoint()),
        fork_id=FORK,
        child_plans=[a_plan(0), a_plan(1)],
    )
    return replace(declared, **changes) if changes else declared


def a_body_reference(cell: str) -> str:
    """The digest the source's descriptor names one cell's bytes by."""
    return sha256(f"body {cell}".encode("utf-8")).hexdigest()


def a_source_manifest() -> SourceArtifactManifest:
    """The source the parent committed at its seal, which both children read by reference.

    It is carried whole rather than by digest, which is what lets a child name the opposite cell's
    bytes with no store open: the entry for that cell is already in the value.
    """
    return SourceArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        bundle_digest=SOURCE,
        environment_task_id="03005e274867c52a",
        source_attempt_id=ATTEMPT,
        source_seal_id="2" * 64,
        execution_ordinal=0,
        canonicalization_version="kernel.1",
        canonical_submission=BlobRef(sha256="3" * 64, size=8, media_type=BODY_MEDIA_TYPE),
        kernel_submission_digest="e" * 64,
        renderer_configuration=RENDERER,
        receipt_contract=CONTRACT_SHAPE,
        grade_identity=RECEIPTS_GRADE,
        cells={
            kind: BlobRef(sha256=a_body_reference(kind), size=40, media_type=BODY_MEDIA_TYPE)
            for kind in CELL_KINDS
        },
        bank_filing_digest="f" * 64,
        bank_cell_digests={
            kind: sha256(f"bank {kind}".encode("utf-8")).hexdigest() for kind in CELL_KINDS
        },
        pair_parity=PairParity(
            body_size=40, encoded_body_bytes=44, masked_body_sha256="7" * 64
        ),
    )


def a_sealed_attempt(**changes: Any) -> CarriedAttempt:
    """The inherited attempt at the cut: its seal, its committed source and its own selection.

    The descriptor and the origin leave here as the mappings a carrier holds them as, because that
    is the shape the fork preserves and the shape a restore reads back through its strict readers.
    """
    declared = CarriedAttempt(
        attempt_id=ATTEMPT,
        state="ack_presented",
        seal_id="2" * 64,
        submission_digest="3" * 64,
        score=0.8154321,
        decode_state="decoded",
        graded_evidence="4" * 64,
        seal_ordinal=1,
        source_artifact=manifest_fields(a_source_manifest()),
        source_commitment=source_commitment(a_source_manifest()),
        source_origin=origin_fields(
            SourceOriginContext(hidden_execution_id=PARENT_EXECUTION, execution_ordinal=0)
        ),
        selected_cell=GRADED_CELL,
        selected_body_reference=a_body_reference(GRADED_CELL),
        selected_policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        receipt_contract_id=CONTRACT,
        presentation_references=["6" * 64],
    )
    return replace(declared, **changes) if changes else declared


def a_candidate() -> PayloadCandidate:
    """The body the parent selected and built, which is the one thing no child inherits."""
    return PayloadCandidate(
        cell=GRADED_CELL,
        renderer_id="kernel-receipt-artifact-1",
        match_group="kernel",
        body="the body its parent was served",
        inner_sha256=a_body_reference(GRADED_CELL),
        visible_sha256="8" * 64,
        visible_byte_count=120,
        renderer_version="1",
        policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        source_commitment=source_commitment(a_source_manifest()),
        body_reference=a_body_reference(GRADED_CELL),
        resolver_id="blob-store-artifact-1",
        resolver_version="1",
    )


def a_child_id(ordinal: int) -> str:
    """The identity that child is derived under, which nothing mints."""
    return child_workflow_id(
        identity_namespace=NAMESPACE,
        parent_workflow_id=PARENT_ID,
        fork_id=FORK,
        child_ordinal=ordinal,
    )


def a_prepared_child(ordinal: int, **changes: Any) -> PreparedChild:
    """The parent's immutable row for one child, and the authority that child asks against."""
    branch, cell, _ = CHILD_PLANS[ordinal]
    child = a_child(ordinal)
    origin = child.fork_origin
    assert origin is not None
    declared = PreparedChild(
        child_ordinal=ordinal,
        child_workflow_id=a_child_id(ordinal),
        complete_start_digest=complete_start_digest(child),
        origin_digest=origin_digest(origin),
        branch_slot=branch,
        target_cell=cell,
        selected_body_reference=a_body_reference(cell),
        consumer_claim_hash=child.consumer_claim_hash,
        hidden_execution_id=child.hidden_execution_id,
        start_differences=list(origin.start_differences),
        existence=CONFIRMED_EXISTING,
        child_run_id=f"22222222-2222-4222-8222-00000000000{ordinal}",
    )
    return replace(declared, **changes) if changes else declared


def a_prepared(**changes: Any) -> PreparedFork:
    """One fork as the parent committed it, in the transition that fenced the parent."""
    declared = PreparedFork(
        fork_id=FORK,
        request_digest=fork_request_digest(a_request()),
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_checkpoint()),
        parent_workflow_id=PARENT_ID,
        parent_run_id=PARENT_RUN,
        children=FORK_CHILD_COUNT,
        child_records=[a_prepared_child(0), a_prepared_child(1)],
    )
    return replace(declared, **changes) if changes else declared


def a_receipt(**changes: Any) -> ForkReceipt:
    """The complete evidence one fork returns, stable across every replay of it."""
    rows = []
    for ordinal in (0, 1):
        branch, cell, _ = CHILD_PLANS[ordinal]
        child = a_child(ordinal)
        row = a_prepared_child(ordinal)
        rows.append(
            ForkChildReceipt(
                child_ordinal=ordinal,
                child_workflow_id=row.child_workflow_id,
                child_run_id=row.child_run_id or "",
                configuration_hash=configuration_hash(child),
                complete_start_digest=row.complete_start_digest,
                origin_digest=row.origin_digest,
                branch_slot=branch,
                target_cell=cell,
                selected_body_reference=row.selected_body_reference,
                acknowledged_cursor=oid(0x102),
                projection_digest="e" * 64,
                start_differences=list(row.start_differences),
                next_task_body_sha256=sha256(b"file the report").hexdigest(),
                next_assignment_id="8" * 32,
            )
        )
    declared = ForkReceipt(
        fork_id=FORK,
        parent_workflow_id=PARENT_ID,
        parent_run_id=PARENT_RUN,
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_checkpoint()),
        children=FORK_CHILD_COUNT,
        child_receipts=rows,
        boundary_evidence=["quiet", "acknowledgement presented", "capacity in use is zero"],
    )
    return replace(declared, **changes) if changes else declared


def a_readiness(**changes: Any) -> ChildReady:
    """What one child publishes when it is prepared, and what releases its container."""
    row = a_prepared_child(0)
    declared = ChildReady(
        fork_id=FORK,
        child_workflow_id=row.child_workflow_id,
        child_run_id=row.child_run_id or "",
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_checkpoint()),
        origin_digest=row.origin_digest,
        verified_set_digest="9" * 64,
        source_attempt_id=ATTEMPT,
        selected_cell=row.target_cell,
        selected_body_reference=row.selected_body_reference,
        selected_policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        receipt_contract_id=CONTRACT,
        ownership_epoch=1,
        consumer_id="the child's own gateway",
        preparation_operation="a" * 64,
    )
    return replace(declared, **changes) if changes else declared


def fork_digests_now() -> Dict[str, Any]:
    """Every digest this file's fixed values are named by, as one mapping.

    It is called in this process and again in another with a different hash seed, and the two are
    compared byte for byte. A digest that depended on the order a dictionary happened to hold its
    keys in would agree with itself here and disagree there.
    """
    children = [a_child(ordinal) for ordinal in (0, 1)]
    origins = []
    for child in children:
        assert child.fork_origin is not None
        origins.append(origin_digest(child.fork_origin))
    return {
        "checkpoint": checkpoint_manifest_reference(a_checkpoint()),
        "request": fork_request_digest(a_request()),
        "identities": [a_child_id(ordinal) for ordinal in (0, 1)],
        "origins": origins,
        "starts": [complete_start_digest(child) for child in children],
        "differences": [
            [
                [row.field_name, row.value_digest]
                for row in start_difference_projection(a_parent(), child)
            ]
            for child in children
        ],
    }


def test_a_checkpoint_binds_every_component_to_the_transcript_it_restores() -> None:
    """A digest validates identified bytes and never proves one object restores another.

    A snapshot taken before the acknowledgement and a transcript written after it both hash
    correctly and both name the right identifiers, and restoring that snapshot still loses the
    acknowledgement. So the manifest is not a list of independent digests: each component asserts
    which transcript a restore of it reproduces, and one asserting another transcript is refused.
    """
    check_checkpoint_manifest(a_checkpoint())
    manifest = a_checkpoint()
    astray = replace(manifest.components[1], restores_transcript="transcript/parent/000041")
    with pytest.raises(WireFormatError) as caught:
        check_checkpoint_manifest(replace(manifest, components=[manifest.components[0], astray]))
    assert "home" in str(caught.value)


def test_a_checkpoint_over_a_harness_that_still_owes_something_is_refused() -> None:
    """The harness settles and then the platform fences, so a checkpoint says it has settled.

    A committed presentation leaves the gateway holding a result owed, which no stream-side
    quiescence check can see, and the call that hands it over is a writing call a fenced parent
    would refuse. What proves it is finished is this, and nothing on the stream side.
    """
    manifest = a_checkpoint()
    for owed in ("transport", "recovery", "provider", "model", "compaction"):
        unsettled = replace(manifest, settled=replace(manifest.settled, **{owed: False}))
        with pytest.raises(WireFormatError) as caught:
            check_checkpoint_manifest(unsettled)
        assert owed in str(caught.value)


def test_a_checkpoint_written_in_a_shape_this_build_does_not_read_is_refused() -> None:
    """Both the schema and the encoding, because a reference names one encoding's bytes."""
    with pytest.raises(WireFormatError):
        check_checkpoint_manifest(a_checkpoint(schema_version="something.else.2"))
    with pytest.raises(WireFormatError):
        check_checkpoint_manifest(a_checkpoint(encoding="whatever-json"))
    assert CHECKPOINT_MANIFEST_SCHEMA_VERSION and CHECKPOINT_MANIFEST_ENCODING


def test_a_checkpoint_naming_no_component_or_no_transcript_is_refused() -> None:
    """A manifest that names nothing a restore reads is not the harness's witness."""
    with pytest.raises(WireFormatError):
        check_checkpoint_manifest(a_checkpoint(components=[]))
    with pytest.raises(WireFormatError):
        check_checkpoint_manifest(a_checkpoint(transcript_reference=""))
    with pytest.raises(WireFormatError):
        check_checkpoint_manifest(a_checkpoint(projection_digest="not a digest"))


def test_two_checkpoints_that_differ_anywhere_are_named_by_different_references() -> None:
    """The reference is the plain digest of the manifest's own canonical bytes."""
    reference = checkpoint_manifest_reference(a_checkpoint())
    assert reference == checkpoint_manifest_reference(a_checkpoint())
    assert reference != checkpoint_manifest_reference(a_checkpoint(adapter_version="another"))
    assert reference != checkpoint_manifest_reference(
        a_checkpoint(acknowledged_cursor=oid(0x999))
    )


def test_no_domain_tag_holds_the_byte_that_separates_it_from_its_value() -> None:
    """The separator is a separator only while no tag can contain one."""
    for tag in FORK_NUMERIC_FIELDS:
        assert "\x00" not in tag
        assert fork_preimage(tag, {}).startswith(tag.encode("ascii") + b"\x00")


def test_one_value_hashes_to_five_different_things_under_five_domains() -> None:
    """Bytes built for one purpose are never a preimage for another."""
    covered = {"a": 1, "b": "two"}
    digests = {
        sha256(fork_preimage(tag, covered)).hexdigest() for tag in FORK_NUMERIC_FIELDS
    }
    assert len(digests) == len(FORK_NUMERIC_FIELDS)


def test_a_zero_to_one_grade_and_a_nonintegral_score_both_encode() -> None:
    """The two real values these records actually hold, rather than integers chosen to pass.

    The published bounds are the environment's own zero and one, and the source score is what a
    receipts component score comes to. The canonical encoding refuses a float, so both reach it
    as the exact text this kernel writes numbers in.
    """
    child = a_child()
    origin = child.fork_origin
    assert origin is not None and origin.source_score == 0.8154321
    assert RECEIPTS_GRADE.public_components[0].minimum == 0.0
    assert child.grade == RECEIPTS_GRADE
    assert len(origin_digest(origin)) == 64
    assert len(complete_start_digest(child)) == 64
    assert b'"0.8154321"' in fork_preimage(ORIGIN_TAG, origin)
    assert b'"minimum":"0"' in fork_preimage(START_TAG, {"grade": {"public_components": [
        {"minimum": 0.0, "maximum": 1.0}
    ]}})


def test_every_covered_number_moves_the_digest_it_is_covered_by() -> None:
    """Each named position in turn, because a conversion that dropped one would still encode."""
    child = a_child()
    origin = child.fork_origin
    assert origin is not None
    assert origin_digest(origin) != origin_digest(replace(origin, source_score=0.8154322))
    assert origin_digest(origin) != origin_digest(replace(origin, source_score=None))
    grade = RECEIPTS_GRADE
    for moved in (replace(grade.public_components[0], minimum=0.5),
                  replace(grade.public_components[0], maximum=0.5)):
        other = replace(child, grade=replace(grade, public_components=(moved,)))
        assert complete_start_digest(child) != complete_start_digest(other)


def test_a_bound_of_zero_and_a_whole_bound_convert_to_one_text() -> None:
    """The rule is equality of numbers rather than of representations."""
    written = {"grade": {"public_components": [{"minimum": 0, "maximum": 1}]}}
    real = {"grade": {"public_components": [{"minimum": 0.0, "maximum": 1.0}]}}
    assert fork_preimage(START_TAG, written) == fork_preimage(START_TAG, real)


def test_a_number_the_table_does_not_name_is_denied_before_the_encoder() -> None:
    """Default denial: a covered value holding a number no entry reaches fails the check.

    It fails as a conversion that names no such field rather than as an encoder that met a
    float, which is the difference between a table somebody has to extend and a value somebody
    has to round.
    """
    with pytest.raises(WireFormatError) as caught:
        fork_preimage(DIFFERENCE_TAG, {"a_new_measure": 0.5})
    assert "conversion" in str(caught.value) and "a_new_measure" in str(caught.value)
    with pytest.raises(WireFormatError) as caught:
        fork_preimage(ORIGIN_TAG, {"source_score": 0.5, "a_second_score": 0.25})
    assert "a_second_score" in str(caught.value)


def test_a_number_that_is_not_finite_is_a_refusal_rather_than_a_text() -> None:
    """Nothing is rounded and nothing is coerced, so an infinity has no digest at all."""
    child = a_child()
    origin = child.fork_origin
    assert origin is not None
    with pytest.raises(WireFormatError) as caught:
        origin_digest(replace(origin, source_score=float("inf")))
    assert "finite" in str(caught.value)
    with pytest.raises(WireFormatError):
        origin_digest(replace(origin, source_score=float("nan")))


def test_a_digest_taken_under_a_domain_this_build_does_not_declare_is_refused() -> None:
    """The table is what says a domain exists, so a tag outside it has no conversion."""
    with pytest.raises(WireFormatError):
        fork_preimage("stream-fork-something-v1", {"a": 1})


def test_a_starts_digest_covers_its_origin_by_reference_and_its_carrier_by_value() -> None:
    """The start covers the origin by reference and the origin covers nothing of the start.

    Nothing is recursive: the origin holds no digest of the start it rides in, and the digest of
    the start lives in the parent's prepared record rather than inside the start it covers. The
    carrier is inside it by value, which is what authenticates the rows that ride in the carrier
    without a second commitment to them.
    """
    child = a_child()
    origin = child.fork_origin
    assert origin is not None
    assert complete_start_digest(child) == complete_start_digest(a_child())
    substituted = replace(child, fork_origin=replace(origin, source_seal_ordinal=99))
    assert complete_start_digest(child) != complete_start_digest(substituted)
    altered = replace(child, carry=replace(a_carry(), data="BBECAwQ"))
    assert complete_start_digest(child) != complete_start_digest(altered)
    assert origin_digest(origin) == origin_digest(
        an_origin(child=replace(child, fork_origin=None), ordinal=0)
    )


def test_a_start_holding_no_origin_has_no_complete_start_digest() -> None:
    """A complete start digest is a child's, and a generation with no lineage is not one."""
    with pytest.raises(WireFormatError):
        complete_start_digest(a_parent())


def test_a_start_whose_origin_was_substituted_fails_the_prepared_record_comparison() -> None:
    """Which is the comparison a child makes against the row its parent committed for it.

    A signature inside the start would prove nothing, because the key that verifies it travels in
    the same start. The parent's own record is the authority, and this is the check that reads it.
    """
    row = a_prepared_child(0)
    assert complete_start_digest(a_child(0)) == row.complete_start_digest
    sibling = a_child(1)
    assert sibling.fork_origin is not None
    exchanged = replace(a_child(0), fork_origin=sibling.fork_origin)
    assert complete_start_digest(exchanged) != row.complete_start_digest


def test_every_fork_digest_is_the_same_value_in_a_second_process() -> None:
    """Under a different hash seed, byte compared, because a digest is a function of a value.

    Canonical ordering is what makes that true: an object serialized in the order a dictionary
    happened to hold its keys would agree with itself in one process and disagree in the next.
    """
    here = fork_digests_now()
    probe = (
        "import json, runpy;"
        f" module = runpy.run_path({str(Path(__file__))!r});"
        " print(json.dumps(module['fork_digests_now']()))"
    )
    environment = dict(os.environ, PYTHONHASHSEED="12345")
    finished = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parents[1]),
        env=environment,
    )
    assert json.loads(finished.stdout) == here
    assert here["identities"] == [a_child_id(0), a_child_id(1)]


def test_a_childs_identity_is_derived_from_its_parents_and_never_minted() -> None:
    """The parent identity, the token for a fork, the ordinal, and half the identity digest."""
    derived = a_child_id(0)
    parent, token, ordinal, tail = derived.rsplit(".", 3)
    assert parent == PARENT_ID
    assert token == FORK_ID_TOKEN
    assert ordinal == "0"
    assert len(tail) == 32 and set(tail) <= set("0123456789abcdef")
    assert a_child_id(0) != a_child_id(1)


def test_a_recovery_run_of_the_same_fork_derives_the_same_children() -> None:
    """Every value in the preimage is fixed when the request is admitted, and none is a run id.

    So a recovery deriving from that admitted request alone reaches the identities the parent's
    prepared record already holds, and a resubmission against a moved execution, which carries
    another exact run id and the same fork, reaches the same two.
    """

    def derived(request: ForkRequest) -> List[str]:
        return [
            child_workflow_id(
                identity_namespace=NAMESPACE,
                parent_workflow_id=request.parent_workflow_id,
                fork_id=request.fork_id,
                child_ordinal=ordinal,
            )
            for ordinal, _ in enumerate(request.child_plans)
        ]

    recorded = [a_prepared_child(ordinal).child_workflow_id for ordinal in (0, 1)]
    assert derived(a_request()) == recorded
    moved = a_request(parent_run_id="33333333-3333-4333-8333-333333333333")
    assert derived(moved) == recorded


def test_two_parents_using_one_local_fork_id_derive_different_children() -> None:
    """Uniqueness follows from the parent identity, so a fork id is unique under one parent.

    The alternative is to require globally unique fork ids and derive from the fork id alone,
    which moves the obligation onto the caller and is not what this derives.
    """
    other = child_workflow_id(
        identity_namespace=NAMESPACE,
        parent_workflow_id="stream/run-1/gen-2",
        fork_id=FORK,
        child_ordinal=0,
    )
    assert other != a_child_id(0)
    assert other.startswith("stream/run-1/gen-2.")


def test_two_namespaces_never_derive_one_child_identity() -> None:
    """The run's identity namespace is in the preimage, so two runs cannot collide."""
    elsewhere = child_workflow_id(
        identity_namespace="another",
        parent_workflow_id=PARENT_ID,
        fork_id=FORK,
        child_ordinal=0,
    )
    assert elsewhere != a_child_id(0)


def test_one_fork_id_over_a_different_checkpoint_is_a_different_request() -> None:
    """The id binds the complete checkpoint and the ordered plans together, not the plans alone."""
    bound = fork_request_digest(a_request())
    assert bound == fork_request_digest(a_request())
    assert bound != fork_request_digest(
        a_request(checkpoint_manifest_reference=checkpoint_manifest_reference(
            a_checkpoint(adapter_version="another")
        ))
    )
    assert bound != fork_request_digest(
        a_request(child_plans=[a_plan(0, target_cell=PLACEBO_CELL), a_plan(1)])
    )
    assert bound != fork_request_digest(a_request(child_plans=[a_plan(1), a_plan(0)]))


def test_a_fork_resubmitted_against_a_moved_execution_is_the_same_logical_fork() -> None:
    """A turnover between reading the checkpoint evidence and submitting moves the run id.

    The controller reads the evidence again and resubmits the same fork under the same checkpoint
    and the same plans, and a continuation preserves the cursor and the projection digest byte
    identical either side of its boundary, so the resubmission is the fork it already asked for.
    """
    assert fork_request_digest(a_request()) == fork_request_digest(
        a_request(parent_run_id="33333333-3333-4333-8333-333333333333")
    )
    assert fork_request_digest(a_request()) != fork_request_digest(
        a_request(parent_workflow_id="stream/run-1/gen-2")
    )


def test_a_request_this_build_cannot_serve_is_refused_against_itself() -> None:
    """The number of children, the branches, and the cells, before any state is read."""
    check_fork_request(a_request())
    with pytest.raises(WireFormatError) as caught:
        check_fork_request(a_request(child_plans=[a_plan(0)]))
    assert str(FORK_CHILD_COUNT) in str(caught.value)
    with pytest.raises(WireFormatError):
        check_fork_request(a_request(child_plans=[a_plan(0), a_plan(1, branch_slot=FIRST_BRANCH)]))
    with pytest.raises(WireFormatError) as caught:
        check_fork_request(a_request(child_plans=[a_plan(0, target_cell=ORACLE_CELL), a_plan(1)]))
    assert ORACLE_CELL in str(caught.value)


def test_two_children_of_one_request_never_share_an_identity_or_a_store() -> None:
    """Two executions under one hidden execution id mint one seal id in two places for one attempt.

    The plans are read together here and nowhere else: every check after this one compares one
    child against its parent, and a request naming one hidden execution twice passes all of them.
    The consumer and the run directory are its neighbours, because children sharing either would
    share a first claim or the objects one of them verifies.
    """
    check_fork_request(a_request())
    for shared in ("hidden_execution_id", "consumer_claim_hash", "run_directory"):
        held = getattr(a_plan(0), shared)
        with pytest.raises(WireFormatError) as caught:
            check_fork_request(a_request(child_plans=[a_plan(0), a_plan(1, **{shared: held})]))
        assert shared in str(caught.value)


def test_every_fork_shape_is_refused_at_a_version_this_build_does_not_admit() -> None:
    """Six shapes, each versioned at one, each refused rather than read half understood."""
    child = a_child()
    assert child.fork_origin is not None
    shapes = (
        (child.fork_origin, check_fork_origin, FORK_ORIGIN_SCHEMA_VERSION),
        (a_request(), check_fork_request, FORK_REQUEST_SCHEMA_VERSION),
        (a_prepared(), check_prepared_fork, PREPARED_FORK_SCHEMA_VERSION),
        (a_receipt(), check_fork_receipt, FORK_RECEIPT_SCHEMA_VERSION),
        (a_readiness(), check_child_ready, CHILD_READY_SCHEMA_VERSION),
        (a_checkpoint(), check_checkpoint_manifest, CHECKPOINT_MANIFEST_SCHEMA_VERSION),
    )
    for value, check, admitted in shapes:
        assert value.schema_version == admitted
        check(value)
        with pytest.raises(WireFormatError):
            check(replace(value, schema_version=2))


def test_a_prepared_fork_that_says_two_things_about_its_ending_is_refused() -> None:
    """A status outside the closed set, an abandonment with no reason, and a reason with none."""
    check_prepared_fork(a_prepared())
    check_prepared_fork(
        a_prepared(status=FORK_ABANDONED, abandoned_reason=SPENT_RECOVERY_RESERVE)
    )
    with pytest.raises(WireFormatError):
        check_prepared_fork(a_prepared(status="halfway"))
    with pytest.raises(WireFormatError):
        check_prepared_fork(a_prepared(status=FORK_ABANDONED))
    with pytest.raises(WireFormatError):
        check_prepared_fork(a_prepared(abandoned_reason=SPENT_RECOVERY_RESERVE))
    check_prepared_fork(a_prepared(status=FORK_COMPLETE))
    assert a_prepared().status == FORK_PREPARED
    assert set(ABANDONED_REASONS) == {SPENT_RECOVERY_RESERVE, "expired_preparation"}


def test_a_prepared_fork_says_which_class_every_child_it_could_not_confirm_is_in() -> None:
    """Never attempted, confirmed existing, existence unconfirmed and proven absent, and no more.

    An Activity failure, a durably journalled failure and an empty confirmation slot each
    establish nothing, so the class is a value the record carries rather than a shorter list a
    reader infers absence from.
    """
    record = a_prepared(
        child_records=[
            a_prepared_child(0),
            a_prepared_child(1, existence=EXISTENCE_UNCONFIRMED, child_run_id=None),
        ]
    )
    check_prepared_fork(record)
    assert record.children == FORK_CHILD_COUNT and len(record.child_records) == 2
    assert record.child_records[1].child_workflow_id == a_child_id(1)
    with pytest.raises(WireFormatError):
        check_prepared_fork(a_prepared(child_records=[a_prepared_child(0, existence="maybe")]))


def test_every_field_of_a_start_is_classified_and_a_new_one_is_denied() -> None:
    """The transformation is a closed enumeration with default denial rather than a prose list.

    A field added to a start is neither silently equal nor silently free: it is denied until it
    is classified, and this is the check that makes that true.
    """
    check_start_classification()
    declared = {member for member in CHILD_START_FIELDS}
    assert declared == set(StreamStart.__dataclass_fields__)
    unclassified = dict(CHILD_START_FIELDS)
    unclassified.pop("budget")
    with pytest.raises(WireFormatError) as caught:
        with _classified_as(unclassified):
            check_start_classification()
    assert "budget" in str(caught.value)
    with pytest.raises(WireFormatError) as caught:
        with _classified_as({**CHILD_START_FIELDS, "a_field_no_start_has": EQUAL_IN_A_CHILD}):
            check_start_classification()
    assert "a_field_no_start_has" in str(caught.value)


#: What a child would look like having changed each field its class fixes, one value per name.
#: The cases below are the equal class itself rather than a list beside it, so a field that joins
#: that class later is a name with no value here rather than a case nobody wrote.
A_CHANGED_VALUE: Dict[str, Any] = {
    "receipt_source": "b" * 64,
    "receipt_contracts": [],
    "forkable_slots": [FIRST_BRANCH],
    "attempt_deadline_ms": 5000,
    "wait_retry_after_ms": 250,
    "profile": "another_profile",
    "capacity": 2,
    "grade": None,
    "families": [A_FAMILY],
    "evaluation_only": True,
    "execution_ordinal": 3,
    "initial_cursor": oid(9),
    "done_message_id": oid(8),
    "id_key_hex": "cd" * 32,
    "canonicalization_version": "kernel.2",
    "configuration_hash": "e" * 64,
    "budget": 12,
    "info": True,
    "tasks": [],
    "assignments": assignments_for(a_parent().tasks, IMMEDIATE),
    "terminal_tool": TerminalTool(
        public_tool_name="file", native_terminal_name="file", argument_names=[]
    ),
    "release": replace(IMMEDIATE, priority=TASK_FIRST),
    "schedule_version": "another schedule",
    "protocol_version": 99,
}


@pytest.mark.parametrize(
    "field_name",
    [name for name, held in CHILD_START_FIELDS.items() if held == EQUAL_IN_A_CHILD],
)
def test_a_child_that_changed_what_its_class_fixes_is_refused(field_name: str) -> None:
    """Every field the classification calls equal, moved one at a time.

    A child that changed one of these would be a different generation rather than a different
    arm, and the two the receipt route added are in that class beside the rest: a child that
    changed its bank source or its declared contracts is not the generation its parent forked.
    A child that moved its release plan to task first would be another one of them, since it
    would serve the next task ahead of the receipt it was forked to deliver.
    """
    assert field_name in A_CHANGED_VALUE, "name what a child changing this field would look like"
    check_child_configuration(a_parent(), a_child())
    with pytest.raises(WireFormatError) as caught:
        check_child_configuration(
            a_parent(), a_child(0, **{field_name: A_CHANGED_VALUE[field_name]})
        )
    assert field_name in str(caught.value)


def test_a_child_serving_a_branch_its_parent_never_declared_is_refused() -> None:
    """The declared set is what the parent committed to before it forked."""
    with pytest.raises(WireFormatError) as caught:
        check_child_configuration(a_parent(), a_child(0, served_slot="branch-three"))
    assert "branch-three" in str(caught.value)
    with pytest.raises(WireFormatError):
        check_child_configuration(
            replace(a_parent(), forkable_slots=[]), a_child()
        )


def test_a_child_serving_its_parents_own_branch_is_refused() -> None:
    """Two children's rows are told apart by their branch, so a child serves one of its own."""
    parent = replace(a_parent(), served_slot=FIRST_BRANCH)
    with pytest.raises(WireFormatError):
        check_child_configuration(parent, a_child(0))


def test_a_child_that_kept_an_identity_of_its_parents_is_refused() -> None:
    """The claim hash and the hidden execution are the two values a child is given of its own.

    The second is what stops one seal id being minted in two places for one public attempt, and
    the first is what makes the child's own gateway a different consumer.
    """
    parent = a_parent()
    for kept in ("consumer_claim_hash", "hidden_execution_id"):
        with pytest.raises(WireFormatError) as caught:
            check_child_configuration(
                parent, a_child(0, **{kept: getattr(parent, kept)})
            )
        assert kept in str(caught.value)


def test_a_child_without_a_durable_store_of_its_own_is_refused() -> None:
    """A child whose blob root is absent verifies nothing, and a shared one is a shared path."""
    with pytest.raises(WireFormatError):
        check_child_configuration(a_parent(), a_child(0, blob_root=None))
    with pytest.raises(WireFormatError):
        check_child_configuration(a_parent(), a_child(0, blob_root=a_parent().blob_root))


def test_a_childs_rows_carry_the_branch_it_serves_under_its_own_provenance_digest() -> None:
    """Policy provenance changes in exactly one derived field, and the rows change with it.

    The roster digest covers the branch slot, the policy digest and the cell of every row, so a
    child that recomputed nothing would present a digest that is not the digest of its own rows.
    The authority, the experiment and the descriptor identity stay fixed.
    """
    parent, child = a_parent(), a_child()
    check_child_configuration(parent, child)
    assert child.provenance is not None and parent.provenance is not None
    assert child.provenance.roster_digest == roster_digest(child.dispositions)
    assert child.provenance.roster_digest != parent.provenance.roster_digest
    assert child.provenance.authority == parent.provenance.authority
    assert child.provenance.experiment_id == parent.provenance.experiment_id
    stale = replace(child, provenance=parent.provenance)
    with pytest.raises(WireFormatError) as caught:
        check_child_configuration(parent, stale)
    assert "own rows" in str(caught.value)
    astray = replace(child, dispositions=[a_row(
        slot=SECOND_BRANCH, policy=GRADED_RECEIPT_ARTIFACT_V1_DIGEST, cell=GRADED_CELL
    )])
    with pytest.raises(WireFormatError):
        check_child_configuration(parent, astray)


def test_a_child_that_moved_one_of_the_three_fixed_provenance_fields_is_refused_by_name(
) -> None:
    """The authority, the experiment and the descriptor identity, moved one at a time.

    A child recomputes the digest of its own rows and keeps everything else those rows were
    registered under: who answered for them, which experiment they belong to and which descriptor
    set they were resolved against. Each mutation below leaves the roster digest correct, because
    a child whose rows and digest disagree is refused before any of these three is read and that
    refusal would say nothing about the field that moved.

    Both generations declare a descriptor identity here, since the default is the empty string and
    a comparison between two of those is a comparison with no operand in it.
    """
    stamped, born = a_parent(), a_child()
    assert stamped.provenance is not None and born.provenance is not None
    resolved = "b" * 64
    parent = replace(
        stamped, provenance=replace(stamped.provenance, descriptor_digest=resolved)
    )
    child = replace(born, provenance=replace(born.provenance, descriptor_digest=resolved))
    check_child_configuration(parent, child)
    for name, moved in (
        ("authority", PLATFORM_DEFAULT),
        ("experiment_id", "a_run_that_registered_something_else"),
        ("descriptor_digest", "c" * 64),
    ):
        assert child.provenance is not None
        astray = replace(child, provenance=replace(child.provenance, **{name: moved}))
        assert astray.provenance is not None
        assert astray.provenance.roster_digest == roster_digest(astray.dispositions)
        with pytest.raises(WireFormatError) as caught:
            check_child_configuration(parent, astray)
        assert name in str(caught.value), name


def test_a_child_resolves_the_obligations_its_parent_resolved_and_no_others() -> None:
    """The rows may be re-slotted and re-policied, and the set of obligations does not move."""
    parent = a_parent()
    extra = replace(a_row(slot=FIRST_BRANCH, policy=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                          cell=GRADED_CELL), payload_position=1)
    rows = [*a_child().dispositions, extra]
    child = a_child(0, dispositions=rows, provenance=PolicyProvenance(
        authority=REGISTERED,
        roster_digest=roster_digest(rows),
        experiment_id="the_subject_of_this_run",
    ))
    with pytest.raises(WireFormatError) as caught:
        check_child_configuration(parent, child)
    assert "resolves" in str(caught.value)


def test_both_children_carry_the_task_their_parent_queued() -> None:
    """Same B is an identity rather than a comparison: neither child may change the queue.

    The manifest, the assignments and the release plan are equal in both, so the task each child
    works next is the same item rather than two items a reader has to check the equality of.
    """
    parent, first, second = a_parent(), a_child(0), a_child(1)
    check_child_configuration(parent, first)
    check_child_configuration(parent, second)
    assert first.tasks == second.tasks == parent.tasks
    assert first.assignments == second.assignments == parent.assignments
    assert first.release == second.release == parent.release
    assert first.served_slot != second.served_slot
    assert first.dispositions[0].cell != second.dispositions[0].cell


def test_the_difference_projection_names_the_fields_a_child_actually_changed() -> None:
    """In the order a start declares them, each with the digest of the child's own value.

    The digest is of the child's value rather than of the difference, because no parent start is
    transmitted to a child: the child checks its own members against these.
    """
    child = a_child()
    projection = start_difference_projection(a_parent(), child)
    assert [row.field_name for row in projection] == [
        "consumer_claim_hash",
        "hidden_execution_id",
        "blob_root",
        "dispositions",
        "provenance",
        "served_slot",
    ]
    assert projection == [
        StartDifference(
            field_name=row.field_name,
            value_digest=sha256(
                fork_preimage(DIFFERENCE_TAG, getattr(child, row.field_name))
            ).hexdigest(),
        )
        for row in projection
    ]
    assert projection != start_difference_projection(a_parent(), a_child(1))


def test_the_difference_projection_excludes_the_carry_and_the_origin_by_construction() -> None:
    """They are lineage rather than configuration, and the restore matrix is what reads them."""
    named = {row.field_name for row in start_difference_projection(a_parent(), a_child())}
    assert a_child().carry is not None and a_parent().carry is None
    assert a_child().fork_origin is not None and a_parent().fork_origin is None
    assert "carry" not in named and "fork_origin" not in named
    assert named == {
        "consumer_claim_hash",
        "hidden_execution_id",
        "blob_root",
        "dispositions",
        "provenance",
        "served_slot",
    }


def a_projection(**changes: Any) -> CarriedProjection:
    """One parent's projection at the cut: counters worked up, and every table it owes from."""
    declared = CarriedProjection(
        configuration_hash=configuration_hash(a_parent()),
        ownership_epoch=4,
        fencing_token_hash="7" * 64,
        ownership_claims=3,
        consumer_id="the parent's own gateway",
        claim_epoch=4,
        cursor=oid(0x102),
        queue_closed=True,
        hidden_ordinal=7,
        seal_ordinal=1,
        wait_count=2,
        wait_reasons={"no_task": 2},
        offer_count=3,
        eligibility_count=1,
        handed_out_attempt_ids=[ATTEMPT],
        activity_ordinal=11,
        verification_batches=2,
        attempts=[],
        obligations=[],
        presented=[
            PresentedMessage(
                order=0,
                kind="task",
                message_id=oid(0x101),
                attempt_id=ATTEMPT,
                visible_bytes_sha256="3" * 64,
            )
        ],
        committed_blobs=["6" * 64],
        pull_requests=[
            CarriedBinding(
                request_id="a pull",
                identity="4" * 64,
                message=OfferedMessage(
                    message_id=oid(0x101), kind="task", visible_text="", attempt_id=ATTEMPT
                ),
            )
        ],
        info_requests=[],
        terminal_requests=[],
        finalize_requests=[
            CarriedFinalization(
                request_id="a finalization",
                identity="5" * 64,
                receipt=AttemptFinalized(
                    attempt_id=ATTEMPT, reason="step_cap", score=0.5, capacity_in_use=0
                ),
            )
        ],
        attestations=[
            CarriedAttestation(
                attestation_id=oid(0xF),
                identity="6" * 64,
                ack=PresentationAck(
                    attestation_id=oid(0xF), cursor=oid(0x102), stream_state_sha256="e" * 64
                ),
            )
        ],
        journal=[["an update", "seal", 4, "value", None, "", "", "", "", None]],
        turnovers=3,
        operation_failures=[
            OperationFailure(
                operation="8" * 64,
                phase="payload_offer",
                reason="unavailable_evidence",
                outcome="recovered",
                generation=PARENT_ID,
                refused_epoch=3,
                attempt_id=ATTEMPT,
            )
        ],
    )
    return replace(declared, **changes) if changes else declared


def test_a_childs_carrier_starts_from_a_first_claim_and_owes_nothing_it_cannot_answer() -> None:
    """Ownership resets and the journal and the four tables go, each named rather than absent.

    An identifier the parent answered is owed to the parent's caller, and a child answering it
    would hand a fenced transport bytes another generation produced. None of the dropped tables
    is inside the projection hash, so dropping them moves no digest the prefix committed to.
    """
    child = transformed_child_counters(a_projection())
    assert (child.ownership_epoch, child.claim_epoch, child.ownership_claims) == (0, 0, 0)
    assert child.fencing_token_hash is None and child.consumer_id is None
    assert child.journal == []
    assert child.pull_requests == child.info_requests == child.terminal_requests == []
    assert child.finalize_requests == [] and child.attestations == []
    assert child.turnovers == 0


def test_a_childs_carrier_keeps_every_count_that_describes_the_prefix_it_inherited() -> None:
    """The cursor and the counts are the prefix both children were cut from.

    The Activity ordinal is one of them, so a child's ordinary numbering equals the number its
    inherited prefix left and can be mapped to an unforked twin's, and the fork's own calls take
    their ids from a namespace of their own instead of running that number ahead.
    """
    parent = a_projection()
    child = transformed_child_counters(parent)
    for kept in (
        "configuration_hash",
        "cursor",
        "queue_closed",
        "hidden_ordinal",
        "seal_ordinal",
        "wait_count",
        "wait_reasons",
        "offer_count",
        "eligibility_count",
        "handed_out_attempt_ids",
        "activity_ordinal",
        "verification_batches",
        "attempts",
        "obligations",
        "presented",
        "committed_blobs",
        "operation_failures",
    ):
        assert getattr(child, kept) == getattr(parent, kept), kept
    assert child.operation_failures[0].generation == PARENT_ID


def test_a_child_carrier_is_the_two_transformations_and_no_third() -> None:
    """The selected delivery evidence and the counters, and everything else as the parent wrote.

    Both halves are driven here, over an attempt holding the source its parent committed and an
    obligation still offerable, because a builder that dropped the selection half would agree with
    the counters alone: a placebo child would restore holding the graded selection its parent
    made, and its own restore is what would refuse it.
    """
    parent = a_projection(
        attempts=[a_sealed_attempt()],
        obligations=[
            CarriedObligation(
                attempt_id=ATTEMPT, state="eligible", materialized=True, candidate=a_candidate()
            )
        ],
    )
    selections = [
        ChildSelection(
            attempt_id=ATTEMPT,
            cell=PLACEBO_CELL,
            policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
        )
    ]
    built = child_carrier(parent, selections=selections)
    assert built == transformed_child_counters(
        transformed_child_selection(parent, selections=selections)
    )
    assert built != transformed_child_counters(parent)
    assert built != transformed_child_selection(parent, selections=selections)
    row = built.attempts[0]
    assert (row.selected_cell, row.selected_policy_digest) == (
        PLACEBO_CELL,
        PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    )
    assert row.selected_body_reference == a_body_reference(PLACEBO_CELL)
    assert (row.seal_id, row.score, row.source_artifact, row.source_origin) == (
        parent.attempts[0].seal_id,
        parent.attempts[0].score,
        parent.attempts[0].source_artifact,
        parent.attempts[0].source_origin,
    )
    owed = built.obligations[0]
    assert owed.candidate is None and owed.pending_preparation and owed.materialized
    assert built.ownership_epoch == 0 and built.journal == []


def test_every_shape_one_fork_transmits_is_measured_before_a_barrier_could_commit() -> None:
    """The converter's actual output for each shape, rather than the size of its largest member.

    A wrapper can exceed the limit while everything it wraps fits, so what is measured is what
    crosses: the request, each plan, each complete child start, the prepared record and the
    receipt. It is the same discipline and the same ceiling the continuation measures under.
    """
    ceiling = TURNOVER_PAYLOAD_CEILING_BYTES
    shapes: Dict[str, Any] = {
        "the fork request": a_request(),
        "the first child plan": a_plan(0),
        "the second child plan": a_plan(1),
        "the first child start": a_child(0),
        "the second child start": a_child(1),
        "the prepared record": a_prepared(),
        "the fork receipt": a_receipt(),
    }
    measured = {
        name: check_transmitted_size(name, value, CONVERTER, ceiling=ceiling)
        for name, value in shapes.items()
    }
    assert all(0 < size <= ceiling for size in measured.values())
    assert measured["the fork request"] == encoded_size(a_request(), CONVERTER)


def test_a_shape_over_the_ceiling_is_refused_with_the_measurement_in_it() -> None:
    """A refusal that says how far over it was is what a controller can act on."""
    with pytest.raises(WireFormatError) as caught:
        check_transmitted_size("the fork request", a_request(), CONVERTER, ceiling=16)
    refusal = str(caught.value)
    assert "the fork request" in refusal
    assert str(encoded_size(a_request(), CONVERTER)) in refusal and "16" in refusal


def test_a_receipts_run_ids_are_bounded_before_the_service_has_minted_them() -> None:
    """One class of field cannot be measured exactly before the barrier and is bounded instead.

    A child's initial exact run id comes back in the start response and does not exist while the
    parent is building the starts, so the known parts are measured exactly, each run id is
    allowed its declared ceiling, and the response is measured against that bound when it lands.
    """
    receipt = a_receipt()
    bound = fork_receipt_bound(receipt, CONVERTER)
    assert bound >= encoded_size(receipt, CONVERTER)
    assert check_receipt_within_bound(receipt, bound, CONVERTER) == encoded_size(
        receipt, CONVERTER
    )
    longest = replace(
        receipt,
        child_receipts=[
            replace(row, child_run_id="9" * (RUN_ID_CEILING_BYTES - 8))
            for row in receipt.child_receipts
        ],
    )
    assert check_receipt_within_bound(longest, bound, CONVERTER) <= bound


def test_a_receipt_over_the_bound_proved_for_it_is_refused_with_its_measurement() -> None:
    """A response that exceeds its bound is a failure carrying the number, not a silent oversize."""
    receipt = a_receipt()
    with pytest.raises(WireFormatError) as caught:
        check_receipt_within_bound(receipt, 32, CONVERTER)
    refusal = str(caught.value)
    assert str(encoded_size(receipt, CONVERTER)) in refusal and "32" in refusal


def test_a_child_start_the_parent_recorded_is_authorized_against_its_own_row() -> None:
    """The comparison a child makes before it owns or serves anything.

    Its identity, its lineage, its complete start and each member the record says it changed,
    against the row the parent committed for its ordinal. Each child answers for its own row and
    for no other, which is what stops one sibling's evidence from authorizing the other.
    """
    for ordinal in (0, 1):
        check_complete_start_authorization(
            a_child(ordinal),
            workflow_id=a_child_id(ordinal),
            record=a_prepared_child(ordinal),
        )
    for ordinal, sibling in ((0, 1), (1, 0)):
        with pytest.raises(WireFormatError, match="child"):
            check_complete_start_authorization(
                a_child(ordinal),
                workflow_id=a_child_id(ordinal),
                record=a_prepared_child(sibling),
            )


def test_an_authentic_start_under_an_identity_the_parent_did_not_record_is_refused() -> None:
    """The one thing hashes, cursors and rows cannot catch, which is why this comparison exists.

    A start carries no workflow id and the configuration hash folds none in, so an authentic
    child start submitted under another identity passes every one of those comparisons and every
    parent lookup, and two executions under one hidden execution id mint one seal id in two
    places for the same public attempt.
    """
    child = a_child(0)
    record = a_prepared_child(0)
    check_complete_start_authorization(child, workflow_id=a_child_id(0), record=record)
    for wrong in (a_child_id(1), f"{PARENT_ID}.{FORK_ID_TOKEN}.0.{'f' * 32}", PARENT_ID):
        with pytest.raises(WireFormatError, match="runs as"):
            check_complete_start_authorization(child, workflow_id=wrong, record=record)


def test_two_siblings_that_exchanged_their_lineage_are_authorized_by_neither_row() -> None:
    """Each start is authentic and each lineage is authentic, and the pairing is not.

    The row for the ordinal a start claims is the row it is compared against, so a start wearing
    its sibling's lineage is asked for that sibling's complete start and cannot produce it, and
    the row for its own ordinal is not the row it claims to be.
    """
    exchanged = replace(a_child(0), fork_origin=an_origin(child=a_child(1), ordinal=1))
    with pytest.raises(WireFormatError, match="child 1 of its fork"):
        check_complete_start_authorization(
            exchanged, workflow_id=a_child_id(0), record=a_prepared_child(0)
        )
    with pytest.raises(WireFormatError, match="complete start"):
        check_complete_start_authorization(
            exchanged, workflow_id=a_child_id(1), record=a_prepared_child(1)
        )
    # And a lineage record that is neither sibling's fails before the start is asked about at
    # all: the record names the lineage the parent wrote, and this start carries another.
    child = a_child(0)
    origin = child.fork_origin
    assert origin is not None
    with pytest.raises(WireFormatError, match="lineage"):
        check_complete_start_authorization(
            replace(child, fork_origin=replace(origin, projection_digest="0" * 64)),
            workflow_id=a_child_id(0),
            record=a_prepared_child(0),
        )


def test_a_member_the_record_says_a_child_changed_is_compared_against_its_own_value() -> None:
    """No parent start is transmitted to a child, so the record carries the digests instead.

    Each is the digest of the child's own value under the difference domain, so a child checks
    its own members against them and a refusal names the member that moved.
    """
    child = a_child(0)
    row = a_prepared_child(0)
    assert [difference.field_name for difference in row.start_differences]
    for index, difference in enumerate(row.start_differences):
        moved = list(row.start_differences)
        moved[index] = replace(difference, value_digest="0" * 64)
        with pytest.raises(WireFormatError, match=difference.field_name):
            check_complete_start_authorization(
                child,
                workflow_id=a_child_id(0),
                record=replace(row, start_differences=moved),
            )
    unknown = replace(
        row,
        start_differences=[StartDifference(field_name="an_invention", value_digest="0" * 64)],
    )
    with pytest.raises(WireFormatError, match="not a field a start declares"):
        check_complete_start_authorization(child, workflow_id=a_child_id(0), record=unknown)


def test_a_start_carrying_no_lineage_is_authorized_by_nothing() -> None:
    """A complete start is authorized against the parent that recorded it, and this one has none."""
    with pytest.raises(WireFormatError, match="no fork origin"):
        check_complete_start_authorization(
            replace(a_child(0), fork_origin=None),
            workflow_id=a_child_id(0),
            record=a_prepared_child(0),
        )
