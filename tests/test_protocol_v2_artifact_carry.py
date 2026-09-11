"""The source evidence crossing a boundary: what is written, what is read, what is refused.

A generation that captured a source has to hand it on. The descriptor, its commitment, the
identity it was sealed under and the selection made from it become attempt state at the seal,
cross in the carrier, and are checked again on the far side against the start that arrived with
them. That checking is what these tests are about, and it is pure: nothing here opens a store,
reaches a world or runs an Activity, because a restore may do none of those either.

Three numbers name three dispositions. A generation declaring a fork writes the highest carrier
from its first boundary, before any child exists; one declaring a receipt contract writes the
middle one from its first boundary, before it has captured anything; one declaring neither writes
the earliest, and writes exactly the bytes the code before it wrote. That last promise is held by
recorded bytes rather than by a round trip: a build agreeing with itself is not evidence that it
agrees with the build it replaced.

The last section is the fork's own pure slice, which is the same subject read forwards: what a
parent's committed source becomes in a child that delivers the opposite cell of it, and what a
child's restore accepts and refuses about the state it arrives in.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytest.importorskip("temporalio")

from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.converter import default as default_converter  # noqa: E402
from temporalio.exceptions import ActivityError, ApplicationError  # noqa: E402
from temporalio.service import RPCError, RPCStatusCode  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    IMMEDIATE,
    RELEASE_AT_SEAL,
    TASK_FIRST,
    EligibilityGate,
    Payload,
    PresentationAck,
    PullRequest,
    ReleasePlan,
    SealAck,
    Task,
    canonical_json,
    visible_bytes,
)
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
    encoded_body_bytes,
    manifest_fields,
    manifest_preimage,
    mask_spans,
    masked_body,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import BlobRef  # noqa: E402
from shogym.serve.protocol_v2.errors import WireFormatError  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    EXPIRED_PREPARATION,
    FORK_ABANDONED,
    FORK_ANSWER_WINDOW_MS,
    FORK_AUTHORITY_HORIZON_MS,
    FORK_AVAILABILITY_STEP,
    FORK_CARRIER_SCHEMA_VERSION,
    FORK_CHILDREN_CONFIRMED,
    FORK_CONFLICTED,
    FORK_PREPARATION_BOUND_MS,
    FORK_PREPARED,
    FORK_RESERVE,
    FORK_EXPIRED_AUTHORITY,
    FORK_REFUSALS,
    FORK_START_STEP,
    FORK_UNREADABLE_PARENT,
    LEGACY_CARRIER_SCHEMA_VERSION,
    ORIGIN_UNVERIFIED,
    PERMANENT_FORK_REFUSALS,
    RECEIPT_CARRIER_SCHEMA_VERSION,
    RETRYABLE_FORK_REFUSALS,
    SPENT_RECOVERY_RESERVE,
    TURNOVER_PAYLOAD_CEILING_BYTES,
    AnsweredUpdate,
    CarriedProjection,
    ChildSelection,
    ForkChildPlan,
    ForkChildReceipt,
    ForkChildStarted,
    ForkOrigin,
    ForkOriginVerified,
    ForkReceipt,
    ForkRequest,
    ForkStatusAnswer,
    ForkStatusQuestion,
    OfferedMessage,
    OwnershipClaim,
    PayloadCandidate,
    PreparedChild,
    PresentedMessage,
    SourceOriginContext,
    StartDifference,
    StreamStart,
    TaskItem,
    TerminalTool,
    Writer,
    assignments_for,
    carrier_version,
    child_carrier,
    child_workflow_id,
    complete_start_digest,
    configuration_hash,
    fork_activities,
    fork_activity_id,
    fork_availability_activity,
    fork_request_digest,
    continuation_argument,
    derived_selection,
    fork_configuration,
    hidden_seal_id,
    origin_digest,
    seal_attempt_activity,
    start_difference_projection,
    stream_replayer,
    transformed_child_selection,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel import runtime as kernel_runtime  # noqa: E402
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    OperationFailure,
    FORK_CONFIGURATION_VIOLATION,
    FORK_IN_FLIGHT,
    FORK_MOVED_EXECUTION,
    FORK_NOT_QUIET,
    FORK_REPAIRABLE_ABSENCE,
    FORK_ORIGIN_DISAGREEMENT,
    FORK_UNDECLARED_BRANCH,
    FORK_UNRECOVERABLE_EVIDENCE,
    FORK_WITNESS_MISMATCH,
    CarriedAttempt,
    CarriedAttestation,
    CarriedBinding,
    CarriedObligation,
    CHILD_EXISTENCE,
    FORK_STATUSES,
    RUN_ID_CEILING_BYTES,
    check_prepared_fork,
    child_blob_root,
    encoded_size,
    fork_origin_bound,
    fork_receipt_bound,
    fork_start_bound,
    legacy_projection_members,
    legacy_start_members,
    origin_fields,
    pack_carrier,
    receipt_projection_members,
    receipt_start_members,
    resolved_echo,
    unpack_carrier,
)
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    ORIGIN_DISAGREEMENT_FAILURE,
)
from shogym.serve.protocol_v2.kernel.runtime import (  # noqa: E402
    refuse_a_carried_projection,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    BLINDED_RECEIPT_V1_DIGEST,
    DELIVER,
    EXPERIMENT,
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    KERNEL_STAND_IN_GRADE,
    LEGACY,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    PLATFORM_DEFAULT,
    POLICIES,
    REGISTERED,
    SINGLETON_SLOT,
    WITHHOLD,
    PayloadDisposition,
    PolicyProvenance,
    render_body,
    roster_digest,
)
from shogym.serve.protocol_v2.reader import (  # noqa: E402
    INHERITED_WORK,
    LOCAL_WORK,
    LineageOrigin,
    RunRecords,
    inherited_work,
    local_work,
    write_records,
)

ATTEMPT = "b" * 32
SILENT = "c" * 32
#: The moment a constructed generation reads its clock at, so a deadline comparison is a value.
A_MOMENT = datetime(2026, 9, 8, 12, 0, tzinfo=timezone.utc)
EXECUTION = "execution-1"
CONTRACT = "ledger_receipt"
SOURCE = "a" * 64
RENDERER = "receipts-render-v1"
RECORDED_CONTINUATION = "stream/recorded-continued-carrier/1"

#: A supported plan that releases at the seal and offers a task before a payload. A run under it
#: works the roster through to the last acknowledgement with every obligation still eligible,
#: which is the state the carrier is at its largest in.
TASK_AHEAD_OF_PAYLOAD = ReleasePlan(
    predicate=RELEASE_AT_SEAL, priority=TASK_FIRST, tie_key=BY_POSITION
)

CONVERTER = default_converter().payload_converter


def oid(value: int) -> str:
    return f"{value:032x}"


# One registered shape, small enough that the bodies below can be written out and still be the
# real thing: four printed rows of ten bytes each, one slot per row, and a body that is exactly
# the rows. Every digest in these tests is the hash of bytes this file holds, so a check that a
# reference names its own body is a check rather than a coincidence of stand-ins.
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

#: A second registered shape, identical in geometry and different in name. It is what makes the
#: capture association a thing a row has to say rather than a thing a restore can infer.
SECOND_CONTRACT = "second_receipt"
SECOND_SHAPE = replace(CONTRACT_SHAPE, contract_id=SECOND_CONTRACT)

BODIES = {
    GRADED_CELL: "".join(f"AB{index}{index}{index}GHIJ\n" for index in range(4)),
    PLACEBO_CELL: "".join(f"AB{index}x{index}GHIJ\n" for index in range(4)),
    ORACLE_CELL: "ABzzzGHIJ\n" * 4,
}
SUBMISSION = "a,1\nb,2\n"


def digest_of(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def a_manifest(**changes: Any) -> SourceArtifactManifest:
    """One committed source over that contract, with whatever a test wants moved."""
    declared = SourceArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        bundle_digest=SOURCE,
        environment_task_id="03005e274867c52a",
        source_attempt_id=ATTEMPT,
        source_seal_id=hidden_seal_id(EXECUTION, 0, ATTEMPT),
        execution_ordinal=0,
        canonicalization_version="kernel.1",
        canonical_submission=BlobRef(
            sha256=digest_of(SUBMISSION), size=len(SUBMISSION), media_type=BODY_MEDIA_TYPE
        ),
        kernel_submission_digest="e" * 64,
        renderer_configuration=RENDERER,
        receipt_contract=CONTRACT_SHAPE,
        grade_identity=KERNEL_STAND_IN_GRADE,
        cells={
            kind: BlobRef(sha256=digest_of(BODIES[kind]), size=40, media_type=BODY_MEDIA_TYPE)
            for kind in CELL_KINDS
        },
        bank_filing_digest="f" * 64,
        bank_cell_digests={kind: digest_of(f"bank {kind}") for kind in CELL_KINDS},
        pair_parity=PairParity(
            body_size=40,
            encoded_body_bytes=encoded_body_bytes(BODIES[GRADED_CELL]),
            masked_body_sha256=sha256(
                masked_body(BODIES[GRADED_CELL].encode("ascii"), mask_spans(CONTRACT_SHAPE))
            ).hexdigest(),
        ),
    )
    return replace(declared, **changes) if changes else declared


def a_start(
    *, contract: bool = True, silent: bool = False, second: bool = False
) -> StreamStart:
    """One generation that delivers the graded cell of that contract at position zero.

    ``second`` declares a second contract and binds the withholding to it, which is the case
    inferring the only contract stops working at: a start may declare several, and a position
    that captures and delivers nothing still says which one its capture was made under.
    """
    items = [
        TaskItem(
            task_position=0,
            attempt_id=ATTEMPT,
            task_message_id=oid(0x101),
            ack_message_id=oid(0x102),
            payload_position=0,
            payload_message_id=oid(0x103),
            body="file the report",
        )
    ]
    rows = [
        PayloadDisposition(
            attempt_id=ATTEMPT,
            payload_position=0,
            kind=DELIVER,
            policy_digest=(
                GRADED_RECEIPT_ARTIFACT_V1_DIGEST if contract else BLINDED_RECEIPT_V1_DIGEST
            ),
            cell=GRADED_CELL,
            resolution_source=REGISTERED,
            family_id=CONTRACT if contract else "",
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
                body="file the second report",
            )
        )
        rows.append(
            PayloadDisposition(
                attempt_id=SILENT,
                payload_position=1,
                kind=WITHHOLD,
                reason="this position delivers nothing",
                resolution_source=REGISTERED,
                family_id=(SECOND_CONTRACT if second else CONTRACT) if contract else "",
            )
        )
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash="d" * 64,
        initial_cursor=oid(1),
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id=EXECUTION,
        canonicalization_version="kernel.1",
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=[]
        ),
        tasks=items,
        release=IMMEDIATE,
        profile=EXPERIMENT,
        grade=KERNEL_STAND_IN_GRADE,
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_subject_of_this_run",
        ),
        receipt_contracts=(
            ([CONTRACT_SHAPE, SECOND_SHAPE] if second else [CONTRACT_SHAPE])
            if contract
            else []
        ),
        receipt_source=SOURCE if contract else "",
    )


def a_legacy_start() -> StreamStart:
    """One generation from before any of this: no rows, no families, no contract, no source.

    It is the profile a pre-policy history resumes under, and the one route where a candidate
    reaches the checks with no disposition and so no policy to be held to at all.
    """
    return replace(
        a_start(contract=False),
        profile=LEGACY,
        dispositions=[],
        provenance=None,
        families=[],
    )


def a_candidate(
    body: str = BODIES[GRADED_CELL],
    *,
    message_id: str = oid(0x103),
    attempt_id: str = ATTEMPT,
    commitment: Optional[str] = None,
    **changes: Any,
) -> PayloadCandidate:
    """The candidate a resolved graded delivery carries, measured the way one is."""
    serialized = visible_bytes(
        Payload(message_id=message_id, attempt_id=attempt_id, body=body)
    )
    declared = PayloadCandidate(
        cell=GRADED_CELL,
        renderer_id="kernel-receipt-artifact-1",
        match_group="kernel",
        body=body,
        inner_sha256=digest_of(body),
        visible_sha256=sha256(serialized).hexdigest(),
        visible_byte_count=len(serialized),
        renderer_version="1",
        policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        source_commitment=(
            source_commitment(a_manifest()) if commitment is None else commitment
        ),
        body_reference=digest_of(BODIES[GRADED_CELL]),
        resolver_id="blob-store-artifact-1",
        resolver_version="1",
    )
    return replace(declared, **changes) if changes else declared


#: The four values a resolved candidate echoes about the source it read, with a value for each
#: that a route resolving nothing has no way to have come by.
INVENTED = (
    ("source_commitment", "3" * 64),
    ("body_reference", "4" * 64),
    ("resolver_id", "blob-store-artifact-1"),
    ("resolver_version", "1"),
)


def a_scalar_candidate(
    body: str = "a body a legacy build recorded", **changes: Any
) -> PayloadCandidate:
    """The candidate a route that resolved nothing returns, the four new values empty.

    Where a policy was resolved the renderer identities are that policy's, and where none was
    they are whatever the build that recorded them wrote. Either way the values below are the
    empty strings a history with no source in it says they are.
    """
    fields: Dict[str, Any] = {
        "cell": "",
        "renderer_id": "",
        "renderer_version": "",
        "policy_digest": "",
        "source_commitment": "",
        "body_reference": "",
        "resolver_id": "",
        "resolver_version": "",
    }
    fields.update(changes)
    return a_candidate(body=body, **fields)


def a_blinded_candidate(**changes: Any) -> PayloadCandidate:
    """The candidate the shipped blinded receipt renders, which resolves nothing either."""
    policy = POLICIES[BLINDED_RECEIPT_V1_DIGEST]
    return a_scalar_candidate(
        body=render_body(
            policy, grade=None, payload_position=0, submission_digest="e" * 64
        ),
        **{
            "cell": GRADED_CELL,
            "renderer_id": policy.renderer_id,
            "renderer_version": policy.renderer_version,
            "policy_digest": BLINDED_RECEIPT_V1_DIGEST,
            **changes,
        },
    )


def a_row(**changes: Any) -> CarriedAttempt:
    """One carried attempt holding a committed source and the selection made from it.

    The descriptor and the origin leave here as the mappings the codec writes them as, which is
    what a carrier holds and what restore reads back through its own strict reader. A caller may
    hand either one a typed value, written out for it here, or a mapping of its own, passed
    through exactly as it was written so that a shape can be put on the wire on purpose.
    """
    declared = CarriedAttempt(
        attempt_id=ATTEMPT,
        state="ack_presented",
        seal_id=hidden_seal_id(EXECUTION, 0, ATTEMPT),
        submission_digest="e" * 64,
        score=1.0,
        decode_state="decoded",
        graded_evidence="9" * 64,
        seal_ordinal=1,
        source_artifact=a_manifest(),
        source_commitment=source_commitment(a_manifest()),
        source_origin=SourceOriginContext(hidden_execution_id=EXECUTION, execution_ordinal=0),
        selected_cell=GRADED_CELL,
        selected_body_reference=digest_of(BODIES[GRADED_CELL]),
        selected_policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        receipt_contract_id=CONTRACT,
    )
    row = replace(declared, **changes) if changes else declared
    return replace(
        row,
        source_artifact=(
            manifest_fields(row.source_artifact)
            if isinstance(row.source_artifact, SourceArtifactManifest)
            else row.source_artifact
        ),
        source_origin=(
            origin_fields(row.source_origin)
            if isinstance(row.source_origin, SourceOriginContext)
            else row.source_origin
        ),
    )


#: The descriptors a generation declaring this contract has to be able to produce. They are in
#: its inventory from the moment it is composed, before it has captured anything at all.
DESCRIPTORS = (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST)


def an_inventory(*objects: str) -> List[str]:
    """The names a complete carried inventory holds: the declared descriptors, and these."""
    return [*DESCRIPTORS, *objects]


def a_projection(
    start: StreamStart,
    *,
    attempts: Optional[List[CarriedAttempt]] = None,
    obligations: Optional[List[CarriedObligation]] = None,
    committed_blobs: Optional[List[str]] = None,
) -> CarriedProjection:
    """One syntactically complete projection for ``start``, with the receipt rows a test wants."""
    rows = [a_row()] if attempts is None else attempts
    inventory = (
        an_inventory(source_commitment(a_manifest()), digest_of(SUBMISSION), "9" * 64)
        if committed_blobs is None
        else committed_blobs
    )
    return CarriedProjection(
        configuration_hash=configuration_hash(start),
        ownership_epoch=1,
        fencing_token_hash="a" * 64,
        ownership_claims=1,
        consumer_id="harness-1",
        claim_epoch=1,
        cursor=oid(9),
        queue_closed=True,
        hidden_ordinal=0,
        seal_ordinal=1,
        wait_count=0,
        wait_reasons={},
        offer_count=2,
        eligibility_count=1,
        handed_out_attempt_ids=[ATTEMPT],
        activity_ordinal=5,
        verification_batches=1,
        attempts=rows,
        obligations=(
            [CarriedObligation(attempt_id=ATTEMPT, state="eligible", materialized=True,
                               candidate=a_candidate())]
            if obligations is None
            else obligations
        ),
        presented=[],
        committed_blobs=inventory,
        pull_requests=[],
        info_requests=[],
        terminal_requests=[],
        finalize_requests=[],
        attestations=[],
    )


def refused(start: StreamStart, projection: CarriedProjection) -> str:
    """Restore this projection against this start, and return what refused it."""
    with pytest.raises(ApplicationError) as raised:
        kernel_workflow._check_carried_receipts(start, projection)
    assert raised.value.type == "CarrierRefused"
    return str(raised.value)


# What version a generation writes, and what that version is a promise about.


def test_a_generation_writes_the_later_carrier_from_the_moment_it_declares_a_contract() -> None:
    """Configuration decides the number before any evidence exists, and evidence decides it after.

    A receipt generation reaches its first boundary with a contract declared and nothing captured,
    which is a lawful place to be: a boundary forbids a sealing attempt, not an unsealed active
    one. The start it hands on holds that contract and the bank source it is admitted over, so the
    number follows the declaration rather than waiting for a capture to justify it.
    """
    declaring = a_start()
    empty = a_projection(declaring, attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="active")],
                         obligations=[], committed_blobs=[])
    assert carrier_version(declaring, empty) == RECEIPT_CARRIER_SCHEMA_VERSION

    plain = a_start(contract=False)
    assert carrier_version(plain, a_projection(
        plain,
        attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="active")],
        obligations=[],
        committed_blobs=[],
    )) == LEGACY_CARRIER_SCHEMA_VERSION

    # And evidence decides it where configuration did not, so a projection holding a descriptor,
    # a selection or a resolved candidate can never be written as a record that had no room for it.
    assert carrier_version(plain, a_projection(plain)) == RECEIPT_CARRIER_SCHEMA_VERSION
    assert carrier_version(
        plain,
        a_projection(
            plain,
            attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="sealed")],
            obligations=[
                CarriedObligation(
                    attempt_id=ATTEMPT,
                    state="eligible",
                    materialized=True,
                    candidate=a_candidate(),
                )
            ],
        ),
    ) == RECEIPT_CARRIER_SCHEMA_VERSION


@pytest.mark.parametrize("field,value", INVENTED)
def test_a_result_from_a_route_that_resolved_nothing_may_not_describe_a_source(
    field: str, value: str
) -> None:
    """The version rule reads a filled value as evidence, so the route refuses to produce one.

    Two routes resolve no source. A generation that resolved no obligation at all declared
    nothing to resolve one from, and a policy that renders from a projection has nothing to
    resolve. Neither has any way to have come by a commitment, a reference or a resolver name, so
    a result carrying one is a Worker's claim about a source nobody captured, and admitting it
    would put provenance on an attempt whose record is that it captured none and then move its
    whole generation to the later carrier on the strength of it. The reading is the same reading
    the fields get everywhere else: what came back is text, and then it is refused for what it
    says.
    """
    item = a_start().tasks[0]
    filing = "e" * 64
    blinded = a_start(contract=False).dispositions[0]
    routes = (
        (None, None, a_scalar_candidate()),
        (blinded, POLICIES[BLINDED_RECEIPT_V1_DIGEST], a_blinded_candidate()),
    )
    for disposition, policy, candidate in routes:
        # The candidate each route really produces, with these four as empty as its history says.
        kernel_workflow._check_candidate(item, candidate)
        kernel_workflow._check_echo(candidate, disposition, policy, item, filing, None)

        invented = replace(candidate, **{field: value})
        with pytest.raises(ApplicationError, match="describing a source it resolved") as raised:
            kernel_workflow._check_echo(invented, disposition, policy, item, filing, None)
        refusal = raised.value
        assert isinstance(refusal, kernel_workflow._UnusableResult)
        assert (refusal.type, refusal.final_failure) == (
            "RendererDescriptorMismatch",
            kernel_workflow.SEAL_RENDERER,
        )


def test_the_legacy_carrier_drops_every_member_this_build_added_and_the_later_one_keeps_them(
) -> None:
    """The earlier number is a serialization adapter as well as a label.

    A field defaulting to absent is an explicit member of the JSON the converter writes, so an
    attempt carrying one packs to a different string from the same attempt without it while both
    carriers would report the same number. The adapter is what makes the number true: it emits the
    member set that version already had, on the carried attempt, on any nested candidate and on
    the start the continuation rides in.
    """
    start = a_start()
    projection = a_projection(start)
    written = json.loads(CONVERTER.to_payloads([projection])[0].data.decode("utf-8"))
    assert "source_artifact" in written["attempts"][0]
    assert "resolver_id" in written["obligations"][0]["candidate"]

    adapted = legacy_projection_members(written)
    # One member of the projection itself goes with them: the operation failure rows, which are a
    # list the earlier version had no name for at all.
    assert set(adapted) == set(written) - {"operation_failures"}
    for name in ("source_artifact", "source_commitment", "source_origin", "selected_cell",
                 "selected_body_reference", "selected_policy_digest", "receipt_contract_id"):
        assert name not in adapted["attempts"][0]
    for name in ("source_commitment", "body_reference", "resolver_id", "resolver_version"):
        assert name not in adapted["obligations"][0]["candidate"]
    # Everything else is passed through exactly, rather than rebuilt to look like it.
    assert adapted["attempts"][0]["seal_id"] == written["attempts"][0]["seal_id"]
    assert adapted["obligations"][0]["candidate"]["body"] == BODIES[GRADED_CELL]

    encoded_start = json.loads(CONVERTER.to_payloads([start])[0].data.decode("utf-8"))
    assert legacy_start_members(encoded_start).keys() == encoded_start.keys() - {
        "receipt_contracts",
        "receipt_source",
        "served_slot",
        "forkable_slots",
        "fork_origin",
    }

    # And the later version keeps all of it, losslessly.
    packed = pack_carrier(projection, CONVERTER, version=RECEIPT_CARRIER_SCHEMA_VERSION)
    assert packed.carrier_schema_version == RECEIPT_CARRIER_SCHEMA_VERSION
    assert unpack_carrier(packed, CONVERTER) == projection


async def test_a_history_continued_before_this_build_replays_and_is_written_the_same_way_again(
) -> None:
    """The compatibility promise, held by bytes this build did not write.

    The fixture is a whole generation that crossed a boundary, recorded by the build from before
    a source artifact existed, kept as the bytes that server wrote. Two things are asked of it.
    It replays under this code, which is what a deployment does to every open stream. And the
    continuation command it recorded, the outer start included, is exactly what this code's
    adapter writes for that same projection: a round trip through new code would only show that
    this build agrees with itself.
    """
    recorded = json.loads(
        (Path(__file__).parent / "_fixtures" / "recorded_continued_carrier.json").read_text(
            encoding="utf-8"
        )
    )
    history = WorkflowHistory.from_json(RECORDED_CONTINUATION, recorded)
    started = history.events[0].workflow_execution_started_event_attributes
    assert started.continued_execution_run_id, "the fixture is a continued execution"
    original = started.input.payloads[0].data

    start = CONVERTER.from_payload(started.input.payloads[0], StreamStart)
    assert start.carry is not None
    assert start.carry.carrier_schema_version == LEGACY_CARRIER_SCHEMA_VERSION
    assert (start.receipt_contracts, start.receipt_source) == ([], "")
    projection = unpack_carrier(start.carry, CONVERTER)
    assert all(row.source_artifact is None for row in projection.attempts)

    # The number this code would choose for that state, and then the bytes it would write.
    assert carrier_version(start, projection) == LEGACY_CARRIER_SCHEMA_VERSION
    again = replace(
        start,
        carry=pack_carrier(projection, CONVERTER, version=LEGACY_CARRIER_SCHEMA_VERSION),
    )
    argument = continuation_argument(again, CONVERTER, LEGACY_CARRIER_SCHEMA_VERSION)
    assert CONVERTER.to_payloads([argument])[0].data == original

    await stream_replayer().replay_workflow(history)


def test_a_receipt_generation_hands_the_contract_and_the_source_on_and_nothing_it_never_declared(
) -> None:
    """Its own number keeps the contract and the source and drops the members added after them.

    The receipt number is a serialization adapter too, now that a later one exists: it emits the
    member set that version had, which is this generation's whole start apart from the three names
    the fork added. A generation on the current number hands its own start over untouched.
    """
    start = a_start()
    written = json.loads(
        CONVERTER.to_payloads(
            [continuation_argument(start, CONVERTER, RECEIPT_CARRIER_SCHEMA_VERSION)]
        )[0].data.decode("utf-8")
    )
    assert written["receipt_source"] == SOURCE
    assert written["receipt_contracts"][0]["contract_id"] == CONTRACT
    assert not {"served_slot", "forkable_slots", "fork_origin"} & set(written)
    assert continuation_argument(start, CONVERTER, FORK_CARRIER_SCHEMA_VERSION) is start


# What a restore accepts, and what it refuses whole.


def test_a_carried_source_and_its_selection_are_admitted_where_every_binding_holds() -> None:
    """The shape all the refusals below are one field away from."""
    start = a_start()
    kernel_workflow._check_carried_receipts(start, a_projection(start))


@pytest.mark.parametrize(
    "name,changes,complaint",
    [
        ("attempt", {"source_attempt_id": "d" * 32}, "attempt"),
        ("seal", {"source_seal_id": hidden_seal_id("another-execution", 0, ATTEMPT)}, "seal"),
        ("ordinal", {"execution_ordinal": 1}, "execution ordinal"),
        ("filing digest", {"kernel_submission_digest": "1" * 64}, "filing digest"),
        ("canonicalization", {"canonicalization_version": "kernel.2"}, "canonicalization"),
        ("bank source", {"bundle_digest": "2" * 64}, "bank source"),
        (
            "grader",
            {"grade_identity": replace(KERNEL_STAND_IN_GRADE, grader_version="9")},
            "grader",
        ),
        (
            "contract",
            {"receipt_contract": replace(CONTRACT_SHAPE, renderer_configuration="somebody else")},
            "contract",
        ),
    ],
)
def test_each_corrupted_binding_in_a_carried_source_is_refused_whole_at_restore(
    name: str, changes: Dict[str, Any], complaint: str
) -> None:
    """One field moved is one carrier refused, and the message says which field.

    Each of these is a descriptor that is internally consistent and belongs to something else:
    another attempt, another hidden execution, another filing, another bank source, another
    grader, another renderer. The commitment is recomputed with the change, so nothing inside the
    record disagrees with anything else inside it. What refuses them is that every binding is
    compared against a value this start already holds.
    """
    start = a_start()
    moved = a_manifest(**changes)
    row = a_row(source_artifact=moved, source_commitment=source_commitment(moved))
    projection = a_projection(
        start,
        attempts=[row],
        obligations=[],
        committed_blobs=an_inventory(source_commitment(moved), digest_of(SUBMISSION), "9" * 64),
    )
    assert complaint in refused(start, projection)


@pytest.mark.parametrize(
    "name,written_source",
    [
        (
            "an unknown name",
            lambda: {**manifest_fields(a_manifest()), "hidden_execution_id": EXECUTION},
        ),
        (
            "a declared name left out",
            lambda: {
                key: value
                for key, value in manifest_fields(a_manifest()).items()
                if key != "cells"
            },
        ),
        (
            "a count written as a fraction of itself",
            lambda: _sized(manifest_fields(a_manifest()), float(len(BODIES[GRADED_CELL]))),
        ),
    ],
)
def test_a_carried_descriptor_in_a_shape_this_build_does_not_read_is_refused_at_restore(
    name: str, written_source: Any
) -> None:
    """The carrier crosses the same converter the seal result does, and is decided after it.

    Each of these is a mapping the converter has an answer for. It drops a name this build does
    not declare, it makes a whole number out of one that is not, and a declared name left out
    leaves a field this build would have to invent. A typed carrier field would take all three
    answers before any code of this generation ran, and what restore then checked would be the
    record the decoder repaired: the commitment would still recompute to itself, every binding
    would agree, and the carrier this generation was actually handed would be gone.

    So the descriptor crosses as the mapping its own canonical bytes are taken from, and the
    strict reader is what makes a descriptor of it, at the restore that refuses the whole carrier.
    """
    start = a_start()
    projection = a_projection(start, attempts=[a_row(source_artifact=written_source())])
    assert "not one this build reads" in refused(start, projection)


def _sized(written_source: Dict[str, Any], size: Any) -> Dict[str, Any]:
    """The same descriptor with the graded cell's size written as something else."""
    cells = {
        kind: {**entry, "size": size} if kind == GRADED_CELL else entry
        for kind, entry in written_source["cells"].items()
    }
    return {**written_source, "cells": cells}


def test_a_carried_origin_in_a_shape_this_build_does_not_read_is_refused_at_restore() -> None:
    """The identity a seal was minted under crosses the same way, and for the same reason.

    It is what a carried seal id is recomputed from, so a member the decoder dropped or an
    ordinal it rounded would leave restore comparing a value it repaired against a value this
    start holds and finding them equal.
    """
    start = a_start()
    for written_origin in (
        {"hidden_execution_id": EXECUTION, "execution_ordinal": 0, "fork_of": "elsewhere"},
        {"hidden_execution_id": EXECUTION, "execution_ordinal": 0.0},
        {"hidden_execution_id": "", "execution_ordinal": 0},
        {"execution_ordinal": 0},
    ):
        projection = a_projection(start, attempts=[a_row(source_origin=written_origin)])
        assert "not one this build reads" in refused(start, projection)


def test_a_carried_source_published_under_another_declared_contract_is_refused() -> None:
    """A row names one registered shape and its descriptor was published under the other.

    Both are shapes this generation declares and each is internally consistent, so nothing about
    either on its own is wrong. What is wrong is the pairing: a capture validated at publication
    against one registered shape would be restored as a capture under another, with a different
    geometry to expand its mask from. The comparison is against the contract the start holds
    rather than against another field of the same carrier, which is what makes it a check.
    """
    start = a_start(second=True)
    published = a_manifest(receipt_contract=SECOND_SHAPE)
    projection = a_projection(
        start,
        attempts=[
            a_row(source_artifact=published, source_commitment=source_commitment(published))
        ],
        obligations=[],
        committed_blobs=an_inventory(
            source_commitment(published), digest_of(SUBMISSION), "9" * 64
        ),
    )
    assert "published under a contract" in refused(start, projection)


def test_a_carrier_naming_a_commitment_the_descriptor_does_not_hash_to_is_refused() -> None:
    """The commitment is recomputed from the descriptor's own canonical bytes, every restore."""
    start = a_start()
    row = a_row(source_commitment="3" * 64)
    projection = a_projection(
        start,
        attempts=[row],
        obligations=[],
        committed_blobs=an_inventory("3" * 64, digest_of(SUBMISSION), "9" * 64),
    )
    assert "commitment" in refused(start, projection)
    # And the bytes it does hash to are the ones the descriptor's preimage names.
    assert source_commitment(a_manifest()) == sha256(manifest_preimage(a_manifest())).hexdigest()


def test_a_carried_source_from_another_origin_is_refused_because_no_other_origin_is_admitted(
) -> None:
    """The origin is a field rather than an assumption, and a generation cut from no fork admits
    one value for it.

    A continuation hands the same start on, so an ordinary one carries this generation's own
    identity. Reading the seal id back out of the carried origin rather than out of the start is
    what lets a fork child admit an inherited source without the check quietly rebinding to
    whichever generation the source arrived at.
    """
    start = a_start()
    elsewhere = SourceOriginContext(hidden_execution_id="another-execution", execution_ordinal=0)
    projection = a_projection(
        start, attempts=[a_row(source_origin=elsewhere)], obligations=[]
    )
    assert "origin" in refused(start, projection)
    assert "origin" in refused(
        start, a_projection(start, attempts=[a_row(source_origin=None)], obligations=[])
    )


@pytest.mark.parametrize(
    "missing",
    ["manifest", "submission", "grade evidence", "served policy", "counterpart policy",
     "presentation"],
)
def test_a_required_object_missing_from_the_carried_inventory_is_refused_rather_than_skipped(
    missing: str,
) -> None:
    """Retention is a claim about objects, and the inventory is where the claim is written down.

    A sealed receipt attempt requires its descriptor, its canonical submission, the evidence its
    score was taken from, the objects its own presentations cited and both policy descriptors of
    its contract at every later ownership claim. A carrier that dropped any of those names would
    hand on a generation whose next claim could not know to look for it, and a claim that read a
    list one name short would certify evidence nobody checked. The cells are deliberately not on
    that list.
    """
    start = a_start()
    transcript = "7" * 64
    row = a_row(presentation_references=[transcript])
    dropped = {
        "manifest": source_commitment(a_manifest()),
        "submission": digest_of(SUBMISSION),
        "grade evidence": "9" * 64,
        "served policy": GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        "counterpart policy": PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
        "presentation": transcript,
    }[missing]
    inventory = [
        reference
        for reference in an_inventory(
            source_commitment(a_manifest()), digest_of(SUBMISSION), "9" * 64, transcript
        )
        if reference != dropped
    ]
    projection = a_projection(
        start, attempts=[row], obligations=[], committed_blobs=inventory
    )
    assert "inventory" in refused(start, projection)
    # And neither eligible body is required there: a claim certifies what it verified.
    assert digest_of(BODIES[GRADED_CELL]) not in inventory


def test_a_generation_declaring_a_contract_carries_its_descriptors_before_its_first_capture(
) -> None:
    """The declared descriptors are required from the first boundary, capture or no capture.

    A generation reaches a boundary with its contract declared and nothing sealed, and what its
    next claim reads the store for is the carried list. So a carrier that dropped a descriptor
    before anything was captured would hand on a generation that could never discover the loss,
    and it is refused here rather than believed.
    """
    start = a_start()
    unsealed = [CarriedAttempt(attempt_id=ATTEMPT, state="active")]
    kernel_workflow._check_carried_receipts(
        start,
        a_projection(start, attempts=unsealed, obligations=[], committed_blobs=an_inventory()),
    )
    assert "inventory" in refused(
        start,
        a_projection(
            start,
            attempts=unsealed,
            obligations=[],
            committed_blobs=[GRADED_RECEIPT_ARTIFACT_V1_DIGEST],
        ),
    )


def test_a_row_that_has_presented_and_not_captured_carries_what_its_presentations_cited(
) -> None:
    """An attempt names the objects its own presentations cited from its first presentation on.

    A receipt generation reaches a boundary with a task delivered and nothing sealed, which is a
    lawful place to be, and that row already names a checkpoint the transport wrote and cited.
    The names it holds are among what its next ownership claim reads the store for, so a carrier
    that dropped one of them on this side of the seal would hand on a generation that could never
    discover the loss, exactly as it would on the far side.
    """
    start = a_start()
    checkpoint = "7" * 64
    presented = CarriedAttempt(
        attempt_id=ATTEMPT,
        state="active",
        task_start_checkpoint=checkpoint,
        presentation_references=[checkpoint],
    )
    kernel_workflow._check_carried_receipts(
        start,
        a_projection(
            start,
            attempts=[presented],
            obligations=[],
            committed_blobs=an_inventory(checkpoint),
        ),
    )
    assert "inventory" in refused(
        start,
        a_projection(
            start,
            attempts=[presented],
            obligations=[],
            committed_blobs=an_inventory(),
        ),
    )


@pytest.mark.parametrize("state", ["active", "ack_presented"])
@pytest.mark.parametrize("kept", ["neither", "the association", "the inventory"])
def test_a_checkpoint_a_row_still_names_is_required_in_both_places_it_was_written(
    state: str, kept: str
) -> None:
    """One transition wrote the checkpoint to both structures, so both are asked for it.

    The task presentation that made an attempt active cited the checkpoint, which put it in the
    attempt's own association and in the generation's inventory together. Take it out of the
    inventory and it is an object no claim will read the store for. Take it out of the
    association and the loss is one no refusal can attribute, while the row goes on naming the
    bytes a resume would be asked to restore from. Take it out of both and neither list has
    anything to say about a checkpoint the attempt is still holding, which is the double omission
    a flat inventory alone cannot see.
    """
    start = a_start()
    checkpoint = "7" * 64
    holds_source = state == "ack_presented"
    association = [checkpoint] if kept == "the association" else []
    row = (
        a_row(task_start_checkpoint=checkpoint, presentation_references=association)
        if holds_source
        else CarriedAttempt(
            attempt_id=ATTEMPT,
            state=state,
            task_start_checkpoint=checkpoint,
            presentation_references=association,
        )
    )
    named = (
        [source_commitment(a_manifest()), digest_of(SUBMISSION), "9" * 64]
        if holds_source
        else []
    )
    projection = a_projection(
        start,
        attempts=[row],
        obligations=None if holds_source else [],
        committed_blobs=an_inventory(
            *named, *([checkpoint] if kept == "the inventory" else [])
        ),
    )
    complaint = refused(start, projection)
    assert ("presentations do not name it" if kept != "the association" else "inventory") in (
        complaint
    )
    # And a generation declaring no receipt contract retains none of this, so its own row keeps
    # crossing with the checkpoint it always carried and neither list is asked about it.
    legacy = a_start(contract=False)
    kernel_workflow._check_carried_receipts(
        legacy,
        a_projection(
            legacy,
            attempts=[
                CarriedAttempt(attempt_id=ATTEMPT, state=state, task_start_checkpoint=checkpoint)
            ],
            obligations=[],
            committed_blobs=[],
        ),
    )


def test_a_row_holding_a_committed_source_and_no_grade_evidence_is_refused_whole() -> None:
    """A source and the object its score was taken from commit in one transition.

    So a carried row holding a descriptor and naming no evidence is a row this build never wrote.
    Reading the name as optional would take the object a score stands on out of what the next
    claim reads the store for, and the loss would be one nothing afterwards could find.
    """
    start = a_start()
    assert "no grade evidence" in refused(
        start,
        a_projection(
            start,
            attempts=[a_row(graded_evidence=None)],
            obligations=[],
            committed_blobs=an_inventory(
                source_commitment(a_manifest()), digest_of(SUBMISSION)
            ),
        ),
    )


def test_a_row_holding_a_selection_and_no_descriptor_is_refused_whole() -> None:
    """A selection is a cell of something, and a row with no source names no such thing."""
    start = a_start()
    stripped = CarriedAttempt(
        attempt_id=ATTEMPT,
        state="ack_presented",
        seal_id=hidden_seal_id(EXECUTION, 0, ATTEMPT),
        selected_cell=GRADED_CELL,
        selected_body_reference=digest_of(BODIES[GRADED_CELL]),
        selected_policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    )
    assert "no source" in refused(
        start, a_projection(start, attempts=[stripped], obligations=[])
    )


def test_a_sealed_row_under_a_receipt_contract_holding_no_descriptor_is_refused_whole() -> None:
    """Capture is not conditional on exposure, so a sealed receipt row carries a source or none."""
    start = a_start()
    empty = CarriedAttempt(
        attempt_id=ATTEMPT,
        state="sealed",
        seal_id=hidden_seal_id(EXECUTION, 0, ATTEMPT),
        seal_ordinal=1,
    )
    assert "carries no source" in refused(
        start, a_projection(start, attempts=[empty], obligations=[])
    )
    # And a row that never sealed is not asked for one.
    kernel_workflow._check_carried_receipts(
        start,
        a_projection(
            start,
            attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="active")],
            obligations=[],
            committed_blobs=an_inventory(),
        ),
    )


def test_a_presented_artifact_obligation_whose_attempt_holds_no_selection_is_refused_whole(
) -> None:
    """A presented obligation has dropped its candidate, so the selection is all that says which
    cell was served, and a run that cannot say that served bodies its record does not describe."""
    start = a_start()
    stripped = a_row(
        state="ack_presented",
        selected_cell=None,
        selected_body_reference=None,
        selected_policy_digest=None,
    )
    projection = a_projection(
        start,
        attempts=[stripped],
        obligations=[
            CarriedObligation(attempt_id=ATTEMPT, state="presented", materialized=True)
        ],
    )
    assert "no selection" in refused(start, projection)


def a_withheld_row(**changes: Any) -> CarriedAttempt:
    """The carried attempt for a position that captures its source and delivers nothing."""
    manifest = a_manifest(
        source_attempt_id=SILENT,
        source_seal_id=hidden_seal_id(EXECUTION, 0, SILENT),
        receipt_contract=SECOND_SHAPE,
    )
    declared = CarriedAttempt(
        attempt_id=SILENT,
        state="ack_presented",
        seal_id=hidden_seal_id(EXECUTION, 0, SILENT),
        submission_digest="e" * 64,
        score=1.0,
        decode_state="decoded",
        graded_evidence="9" * 64,
        seal_ordinal=2,
        source_artifact=manifest,
        source_commitment=source_commitment(manifest),
        source_origin=SourceOriginContext(hidden_execution_id=EXECUTION, execution_ordinal=0),
        receipt_contract_id=SECOND_CONTRACT,
    )
    row = replace(declared, **changes) if changes else declared
    return replace(
        row,
        source_artifact=(
            manifest_fields(row.source_artifact)
            if isinstance(row.source_artifact, SourceArtifactManifest)
            else row.source_artifact
        ),
        source_origin=(
            origin_fields(row.source_origin)
            if isinstance(row.source_origin, SourceOriginContext)
            else row.source_origin
        ),
    )


def test_a_withheld_row_restores_only_the_source_the_contract_its_own_row_names_captured(
) -> None:
    """A withholding names its capture contract, and the carried source has to be under it.

    A start may declare several contracts and a withheld position captures under one of them, so
    inferring which stops working at the second declaration. Fresh capture reads the row's own
    contract and validates against it; a restore that only compared a delivering row would admit
    a withheld source captured under the other declared shape, which is a consumer of carried
    state using a weaker rule than the producer.
    """
    start = a_start(silent=True, second=True)
    inventory = an_inventory(
        source_commitment(a_manifest()),
        digest_of(SUBMISSION),
        "9" * 64,
        a_withheld_row().source_commitment or "",
    )
    kept = a_projection(
        start,
        attempts=[a_row(), a_withheld_row()],
        obligations=[],
        committed_blobs=inventory,
    )
    kernel_workflow._check_carried_receipts(start, kept)

    # The same row carrying the other declared contract's source, with nothing else moved: the
    # descriptor is internally consistent and the generation declares the shape it names.
    elsewhere = a_withheld_row(
        source_artifact=a_manifest(
            source_attempt_id=SILENT, source_seal_id=hidden_seal_id(EXECUTION, 0, SILENT)
        ),
        source_commitment=source_commitment(
            a_manifest(
                source_attempt_id=SILENT, source_seal_id=hidden_seal_id(EXECUTION, 0, SILENT)
            )
        ),
        receipt_contract_id=CONTRACT,
    )
    moved = a_projection(
        start,
        attempts=[a_row(), elsewhere],
        obligations=[],
        committed_blobs=[*inventory, elsewhere.source_commitment or ""],
    )
    complaint = refused(start, moved)
    assert SECOND_CONTRACT in complaint and CONTRACT in complaint


def test_a_legacy_presented_obligation_crosses_a_restore_that_asks_a_receipt_row_for_a_selection(
) -> None:
    """The four whole refusals are scoped to artifact rows, and this is why they have to be.

    A generation that delivers the blinded receipt has no selection by design, and its presented
    obligations have to stay readable for ever. So the same restore that refuses a presented
    artifact row with no selection admits this one without asking.
    """
    start = a_start(contract=False)
    projection = a_projection(
        start,
        attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="ack_presented", seal_ordinal=1)],
        obligations=[
            CarriedObligation(attempt_id=ATTEMPT, state="presented", materialized=True)
        ],
        committed_blobs=[],
    )
    kernel_workflow._check_carried_receipts(start, projection)


def a_scalar_projection(
    start: StreamStart, candidate: PayloadCandidate
) -> CarriedProjection:
    """One sealed generation with no source in it, carrying that candidate against its position."""
    return a_projection(
        start,
        attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="ack_presented", seal_ordinal=1)],
        obligations=[
            CarriedObligation(
                attempt_id=ATTEMPT, state="eligible", materialized=True, candidate=candidate
            )
        ],
        committed_blobs=[],
    )


@pytest.mark.parametrize("field,value", INVENTED)
def test_a_carried_candidate_describing_a_source_nothing_resolved_is_refused_whole(
    monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    """The row above the candidate is held to this already, and the candidate is the other half.

    A carried attempt naming a commitment or a selection with no descriptor is refused whole, and
    a legacy row naming a capture contract is refused with it. The candidate hanging off the same
    obligation was not asked anything, so a carrier could hand a generation that declared no
    contract a body with a source commitment on it, and the number chosen for the next carrier
    would read that as evidence and write the later one. Both routes that resolve nothing are
    asked here, the generation with no disposition at all and the one delivering a policy that
    renders from a projection, and the refusal is the restore's own rather than a later surprise.
    """
    monkeypatch.setattr(kernel_workflow.workflow, "payload_converter", lambda: CONVERTER)
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "info",
        lambda: SimpleNamespace(
            workflow_id="stream/carried-scalar-candidate/1", continued_run_id="run-before"
        ),
    )
    for start, candidate in (
        (a_legacy_start(), a_scalar_candidate()),
        (a_start(contract=False), a_blinded_candidate()),
    ):
        invented = a_scalar_projection(start, replace(candidate, **{field: value}))
        # It really is what the version rule would read as evidence, and it really does survive
        # the codec: the refusal is the restore's, not the encoder's.
        assert carrier_version(start, invented) == RECEIPT_CARRIER_SCHEMA_VERSION
        packed = pack_carrier(invented, CONVERTER, version=RECEIPT_CARRIER_SCHEMA_VERSION)
        assert unpack_carrier(packed, CONVERTER) == invented
        assert "describes a source it resolved" in refused(
            start, unpack_carrier(packed, CONVERTER)
        )
        with pytest.raises(ApplicationError) as raised:
            kernel_workflow.StreamWorkflow(replace(start, carry=packed))
        assert raised.value.type == "CarrierRefused"


def test_an_active_legacy_continuation_stays_at_the_earlier_number_through_a_full_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """And the generation the refusal above is protecting crosses exactly as it always did.

    Its candidate carries the empty defaults, so nothing in its projection is evidence, the
    number is the earlier one, the adapter writes the member set that number already had, and the
    constructor restores the whole of it. The two numbers name two dispositions: this one stays
    where it is for the life of the generation.
    """
    monkeypatch.setattr(kernel_workflow.workflow, "payload_converter", lambda: CONVERTER)
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "info",
        lambda: SimpleNamespace(
            workflow_id="stream/legacy-continuation/1", continued_run_id="run-before"
        ),
    )
    for start, candidate in (
        (a_legacy_start(), a_scalar_candidate()),
        (a_start(contract=False), a_blinded_candidate()),
    ):
        projection = a_scalar_projection(start, candidate)
        assert carrier_version(start, projection) == LEGACY_CARRIER_SCHEMA_VERSION
        packed = pack_carrier(projection, CONVERTER, version=LEGACY_CARRIER_SCHEMA_VERSION)
        assert unpack_carrier(packed, CONVERTER) == projection
        restored = kernel_workflow.StreamWorkflow(replace(start, carry=packed))
        carried = restored._obligations[ATTEMPT].candidate
        assert carried is not None
        assert carried.body == candidate.body
        assert not resolved_echo(carried)


@pytest.mark.parametrize("state", ["materialized", "eligible", "offered"])
def test_an_obligation_an_offer_could_still_be_made_from_carries_both_halves_of_it(
    state: str,
) -> None:
    """A selection says which committed cell an offer carries and the candidate holds the bytes.

    The offer path reads the candidate without asking whether there is one, so an obligation that
    arrived at one of these states with either half missing restores into a generation whose next
    pull for that position has nothing to answer with and no reason to give. Each half is refused
    on its own, because either can go missing without the other.

    The one state where a selection stands with no candidate is a gated fork child's obligation,
    selected against the parent's committed source and unbuilt until the child builds it. It is
    admitted by the preparation gate the parent wrote into the child's start, and this generation
    was cut from no fork and carries no gate, so every one of these stays refused.
    """
    start = a_start()

    # The candidate alone, gone.
    assert "carries no candidate" in refused(
        start,
        a_projection(
            start,
            obligations=[
                CarriedObligation(
                    attempt_id=ATTEMPT, state=state, materialized=True, candidate=None
                )
            ],
        ),
    )

    # The selection alone, gone, with the candidate still there.
    assert "holds no selection" in refused(
        start,
        a_projection(
            start,
            attempts=[
                a_row(
                    selected_cell=None,
                    selected_body_reference=None,
                    selected_policy_digest=None,
                )
            ],
            obligations=[
                CarriedObligation(
                    attempt_id=ATTEMPT,
                    state=state,
                    materialized=True,
                    candidate=a_candidate(),
                )
            ],
        ),
    )

    # And both, which is the state an unbuilt fork child's obligation would arrive in.
    assert "holds no selection" in refused(
        start,
        a_projection(
            start,
            attempts=[
                a_row(
                    selected_cell=None,
                    selected_body_reference=None,
                    selected_policy_digest=None,
                )
            ],
            obligations=[
                CarriedObligation(
                    attempt_id=ATTEMPT, state=state, materialized=True, candidate=None
                )
            ],
        ),
    )


def test_a_presented_obligation_keeps_its_selection_and_is_not_asked_for_a_candidate() -> None:
    """An obligation that has been presented will never be offered again, so it drops the bytes.

    That is why the selection lives on the attempt rather than on the candidate: the reference
    hung on the candidate alone would go with it. What restore asks of a presented row is the
    selection, and what it does not ask for is a body nothing will ever send.
    """
    start = a_start()
    kernel_workflow._check_carried_receipts(
        start,
        a_projection(
            start,
            obligations=[
                CarriedObligation(
                    attempt_id=ATTEMPT, state="presented", materialized=True, candidate=None
                )
            ],
        ),
    )


@pytest.mark.parametrize(
    "name,changes",
    [
        ("renderer", {"renderer_id": "some-other-renderer"}),
        ("renderer version", {"renderer_version": "2"}),
        ("resolver", {"resolver_id": "some-other-resolver"}),
        ("resolver version", {"resolver_version": "2"}),
    ],
)
def test_a_carried_candidate_from_an_implementation_its_policy_does_not_declare_is_refused(
    name: str, changes: Dict[str, Any]
) -> None:
    """A carried candidate is held to the identities a fresh one is held to, and by the same rule.

    The renderer and the resolver are what a reader is told produced these bytes. A fresh
    candidate carrying either of another build's is refused where it comes back from the
    Activity, so a carried one has to be refused here: otherwise a boundary would be a place
    provenance could be moved, with the body, the reference and the commitment all still
    agreeing with the record and nothing left to disagree with.
    """
    start = a_start()
    projection = a_projection(
        start,
        obligations=[
            CarriedObligation(
                attempt_id=ATTEMPT,
                state="eligible",
                materialized=True,
                candidate=a_candidate(**changes),
            )
        ],
    )
    complaint = refused(start, projection)
    assert f"came back from a {name}" in complaint


@pytest.mark.parametrize(
    "name,changes",
    [
        ("cell", {"cell": PLACEBO_CELL}),
        ("source", {"source_commitment": "4" * 64}),
        ("reference", {"body_reference": digest_of(BODIES[PLACEBO_CELL])}),
        ("policy", {"policy_digest": PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST}),
    ],
)
def test_a_carried_candidate_that_does_not_agree_with_its_attempts_selection_is_refused(
    name: str, changes: Dict[str, Any]
) -> None:
    """A candidate crossing a boundary is held to the selection its attempt carries.

    The bytes are the largest thing in a carrier and they are the only thing that could still be
    offered, so what they are has to be settled against the record rather than against themselves.
    """
    start = a_start()
    projection = a_projection(
        start,
        obligations=[
            CarriedObligation(
                attempt_id=ATTEMPT,
                state="eligible",
                materialized=True,
                candidate=a_candidate(**changes),
            )
        ],
    )
    assert name in refused(start, projection)


def test_a_carried_candidate_is_measured_again_rather_than_believed() -> None:
    """The hashes and the byte count are recomputed from the body the carrier holds."""
    start = a_start()
    for changes in ({"visible_byte_count": 9}, {"visible_sha256": "5" * 64}):
        projection = a_projection(
            start,
            obligations=[
                CarriedObligation(
                    attempt_id=ATTEMPT,
                    state="eligible",
                    materialized=True,
                    candidate=a_candidate(**changes),
                )
            ],
        )
        assert "measure" in refused(start, projection)


def test_the_whole_receipt_carrier_round_trips_and_restores_after_it_is_packed() -> None:
    """The packing is what a boundary really does, so the checks are run over what comes back.

    The descriptor comes back as the mapping it crossed as, member for member, and becomes a
    descriptor again where restore reads it. That is the whole point of its crossing untyped: a
    field with a type would be one the decoder made that type of, and what the checks below ran
    over would be a record the decoder had repaired rather than the record that crossed.
    """
    start = a_start()
    projection = a_projection(start)
    version = carrier_version(start, projection)
    packed = pack_carrier(projection, CONVERTER, version=version)
    read_back = unpack_carrier(packed, CONVERTER)
    assert read_back == projection
    assert read_back.attempts[0].source_artifact == manifest_fields(a_manifest())
    assert read_back.attempts[0].source_origin == origin_fields(
        SourceOriginContext(hidden_execution_id=EXECUTION, execution_ordinal=0)
    )
    read = kernel_workflow._check_carried_receipts(start, read_back)
    assert read[ATTEMPT].manifest == a_manifest()
    assert read[ATTEMPT].origin == SourceOriginContext(
        hidden_execution_id=EXECUTION, execution_ordinal=0
    )


# The narrow derivation, and the identity it is checked against.


def test_a_committed_source_derives_either_eligible_cell_and_never_the_oracle() -> None:
    """What a derivation takes is the source, its origin, a target policy and cell, and a store.

    No live world, no canonical answer text, no recovery token, no second seal, no grade and no
    render. The oracle is refused at the first check: it is named by the descriptor, retained with
    everything else, and is not a cell any derivation resolves.
    """
    source = a_manifest()
    origin = SourceOriginContext(hidden_execution_id=EXECUTION, execution_ordinal=0)
    for cell, policy in (
        (GRADED_CELL, GRADED_RECEIPT_ARTIFACT_V1_DIGEST),
        (PLACEBO_CELL, PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST),
    ):
        reference = derived_selection(
            source=source,
            origin=origin,
            commitment=source_commitment(source),
            cell=cell,
            policy_digest=policy,
            blob_root="/somewhere",
        )
        assert reference.body_sha256 == digest_of(BODIES[cell])
        assert reference.contract_id == CONTRACT
        assert reference.slot_spans == mask_spans(CONTRACT_SHAPE)
    with pytest.raises(WireFormatError, match="graded"):
        derived_selection(
            source=source,
            origin=origin,
            commitment=source_commitment(source),
            cell=ORACLE_CELL,
            policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
            blob_root="/somewhere",
        )
    # A policy this contract does not admit is refused too, whichever cell it names.
    with pytest.raises(WireFormatError):
        derived_selection(
            source=source,
            origin=origin,
            commitment=source_commitment(source),
            cell=GRADED_CELL,
            policy_digest=BLINDED_RECEIPT_V1_DIGEST,
            blob_root="/somewhere",
        )


def test_a_derivation_is_checked_against_the_identity_that_sealed_the_source() -> None:
    """The source's origin and the start a derivation runs under are two different identities.

    Here they really are different: the source was sealed under one hidden execution and the
    derivation is being asked for by a generation under another. The origin is a parameter, so the
    seal id is recomputed from the identity that produced the source; handing the derivation the
    calling generation's identity instead refuses, which is what stops a later build's child from
    silently accepting a parent's source under its own context.
    """
    source = a_manifest(source_seal_id=hidden_seal_id("source-execution", 3, ATTEMPT),
                        execution_ordinal=3)
    its_own = SourceOriginContext(hidden_execution_id="source-execution", execution_ordinal=3)
    reference = derived_selection(
        source=source,
        origin=its_own,
        commitment=source_commitment(source),
        cell=PLACEBO_CELL,
        policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
        blob_root="/somewhere",
    )
    assert reference.body_sha256 == digest_of(BODIES[PLACEBO_CELL])
    target = SourceOriginContext(hidden_execution_id="target-execution", execution_ordinal=3)
    with pytest.raises(WireFormatError, match="hidden execution"):
        derived_selection(
            source=source,
            origin=target,
            commitment=source_commitment(source),
            cell=PLACEBO_CELL,
            policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
            blob_root="/somewhere",
        )
    with pytest.raises(WireFormatError, match="commitment"):
        derived_selection(
            source=source,
            origin=its_own,
            commitment="6" * 64,
            cell=PLACEBO_CELL,
            policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
            blob_root="/somewhere",
        )


def a_transport_id(name: str) -> str:
    """One identifier of the shape a transport mints: unrelated to the one before it.

    A gateway draws a fresh random identifier for every request and every attestation. Numbering
    them instead would give a few hundred strings that all share a prefix, and the packing would
    do something with them that it cannot do with a real run's, so the measurement below would be
    of a workload nobody has.
    """
    return sha256(name.encode("utf-8")).hexdigest()[:32]


def _committed_messages(items: List[TaskItem]) -> List[PresentedMessage]:
    """The presentations an attempt that has filed and been acknowledged leaves, in commit order.

    Two per attempt: the task it was handed and the acknowledgement its seal returned. The digest
    is the one the presentation was verified against, so it is taken over the bytes that message
    actually is rather than written down as a stand-in.
    """
    rows: List[PresentedMessage] = []
    for item in items:
        for kind, message_id, visible in (
            (
                "task",
                item.task_message_id,
                visible_bytes(
                    Task(
                        message_id=item.task_message_id,
                        attempt_id=item.attempt_id,
                        body=item.body,
                        budget=None,
                    )
                ),
            ),
            (
                "seal_ack",
                item.ack_message_id,
                visible_bytes(
                    SealAck(
                        message_id=item.ack_message_id,
                        attempt_id=item.attempt_id,
                        submission_digest=digest_of(f"filing {item.attempt_id}"),
                        canonicalization_version="kernel.1",
                    )
                ),
            ),
        ):
            rows.append(
                PresentedMessage(
                    order=len(rows),
                    kind=kind,
                    message_id=message_id,
                    attempt_id=item.attempt_id,
                    visible_bytes_sha256=sha256(visible).hexdigest(),
                )
            )
    return rows


def _acknowledgements(committed: List[PresentedMessage]) -> List[CarriedAttestation]:
    """One attestation per presentation, with the acknowledgement it was answered with."""
    return [
        CarriedAttestation(
            attestation_id=a_transport_id(f"attestation {row.message_id}"),
            identity=a_transport_id(f"identity {row.message_id}"),
            ack=PresentationAck(
                attestation_id=a_transport_id(f"attestation {row.message_id}"),
                cursor=row.message_id,
                stream_state_sha256=digest_of(f"state after {row.message_id}"),
            ),
        )
        for row in committed
    ]


def _reservations(items: List[TaskItem], handler: str) -> List[CarriedBinding]:
    """One binding per answered request, with the bytes of a presented message dropped.

    At a legal boundary every bound message has been presented, so what a retry is owed is the
    answer that it was presented rather than the bytes, and the table carries the identity and
    the identifier alone.
    """
    return [
        CarriedBinding(
            request_id=a_transport_id(f"{handler} {item.attempt_id}"),
            identity=digest_of(f"{handler} identity {item.attempt_id}"),
            message=OfferedMessage(
                message_id=(
                    item.task_message_id if handler == kernel_workflow.PULL
                    else item.ack_message_id
                ),
                kind="task" if handler == kernel_workflow.PULL else "seal_ack",
                visible_text="",
                attempt_id=item.attempt_id,
            ),
        )
        for item in items
    ]


def _recipes(start: StreamStart) -> Dict[str, str]:
    """The recipe the codec really writes for each offered task of ``start``.

    The digest is over the whole offered message the pull was answered with, wrapper included,
    and it is taken here from the same function restore rebuilds it with rather than from a
    second reading of what a task looks like. A fixture that wrote its own would measure a
    carrier this build refuses, which is the opposite of what a measurement is for.
    """
    generation = kernel_workflow.StreamWorkflow.__new__(kernel_workflow.StreamWorkflow)
    generation._start = start
    generation._items = {item.attempt_id: item for item in start.tasks}
    recipes: Dict[str, str] = {}
    for item in start.tasks:
        offered = generation._from_recipe(f"task {item.attempt_id} unproved")
        assert offered is not None
        recipes[item.attempt_id] = (
            f"task {item.attempt_id} {sha256(canonical_json(offered)).hexdigest()}"
        )
    return recipes


def _journal(
    items: List[TaskItem], committed: List[PresentedMessage], recipes: Dict[str, str]
) -> List[List[Any]]:
    """The four Updates each attempt was answered under, written the way the codec writes them.

    An offered task is kept as the recipe it can be rebuilt from, which is why its value is empty
    here; an acknowledgement offer keeps its bytes, because nothing rebuilds those; and a
    presentation points at the attestation row that already holds its answer.

    They are in the order the generation answered them: the task is offered, the task is
    presented, the seal returns the acknowledgement, and the acknowledgement is presented. The
    attempt is only sealable once its task has been presented, so the seal cannot come first.
    """
    acknowledged = {row.message_id: row for row in committed}
    rows: List[List[Any]] = []

    def presentation(message_id: str) -> List[Any]:
        return [
            a_transport_id(f"update present {message_id}"),
            kernel_workflow.PRESENT,
            1,
            kernel_workflow.BY_ROW,
            a_transport_id(f"attestation {acknowledged[message_id].message_id}"),
            "",
            "",
            "",
            None,
            "",
        ]

    for item in items:
        ack = visible_bytes(
            SealAck(
                message_id=item.ack_message_id,
                attempt_id=item.attempt_id,
                submission_digest=digest_of(f"filing {item.attempt_id}"),
                canonicalization_version="kernel.1",
            )
        )
        rows.append(
            [
                a_transport_id(f"update pull {item.attempt_id}"),
                kernel_workflow.PULL,
                1,
                kernel_workflow.BY_VALUE,
                "",
                "",
                "",
                "",
                "",
                recipes[item.attempt_id],
            ]
        )
        rows.append(presentation(item.task_message_id))
        rows.append(
            [
                a_transport_id(f"update seal {item.attempt_id}"),
                kernel_workflow.SEAL,
                1,
                kernel_workflow.BY_VALUE,
                "",
                "",
                "",
                "",
                {
                    "message_id": item.ack_message_id,
                    "kind": "seal_ack",
                    "visible_text": ack.decode("utf-8"),
                    "attempt_id": item.attempt_id,
                    "protocol_version": 2,
                },
                "",
            ]
        )
        rows.append(presentation(item.ack_message_id))
    return rows


def test_a_receipt_bearing_carrier_at_the_rosters_cap_fits_under_the_payload_ceiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No body crosses, so what a source adds to a carrier is a record rather than a payload.

    The measurement is the supported roster's size with a committed source, its commitment, its
    origin, its selection, its retained presentation references and, where an obligation could
    still be offered, its candidate, packed and encoded the way a continuation is. The ceiling is
    the one the generation refuses its own boundary at, so this is the question a receipt run has
    to be able to answer before it is worth serving one.

    What is measured is a projection a run of this length really leaves, so the run it is a
    projection of has to be one the declared schedule could produce. The plan here puts a task
    ahead of a payload, which is what lets every attempt reach its acknowledgement while every
    obligation is still eligible and still holding its candidate: under a payload-first plan the
    first release would outrank the second task and the whole roster could never be sealed with
    the whole outbox undelivered. That is the fullest a carrier gets, and it is the state this
    measures, with the two presentations each attempt committed in commit order, the attestations
    that acknowledged them, the request bindings that reserved them, the checkpoint each attempt
    would be restored from named by its own presentations, seal ordinals running one to the
    roster, every attempt handed out, and the four Updates each attempt was answered under in the
    order it answered them.

    Every value that differs per attempt in a real run differs here, and the identifiers a
    transport mints are the shape a transport mints them in. Sequential identifiers and one
    repeated reference would be a few hundred distinct strings where a real run has thousands,
    and what the writing does with them is exactly what this measures.

    The validity is established by the restore a boundary really runs, not by the receipt checks
    alone: the carrier is packed, the start it rides on is handed to the workflow's constructor,
    and the whole of the reading happens, the application over rebuilt state and the journal's
    recipes among it. A carrier missing the obligations, the inventory or the candidates would be
    smaller and would also be refused there, so it would answer a question nobody is asking. The
    number is read off the carrier that was accepted.
    """
    # The rebuilding and the packing read the converter the way every other value does, and
    # outside a Worker there is no workflow to read one from. The one the service is configured
    # with is the same object, and the identity is the one a continued execution arrives with.
    monkeypatch.setattr(
        kernel_workflow.workflow, "payload_converter", lambda: CONVERTER
    )
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "info",
        lambda: SimpleNamespace(
            workflow_id="stream/receipt-carrier-at-the-cap/1", continued_run_id="run-before"
        ),
    )
    roster = 200
    items = [
        TaskItem(
            task_position=index,
            attempt_id=f"{index:032x}",
            task_message_id=oid(0x1000 + index * 4),
            ack_message_id=oid(0x1001 + index * 4),
            payload_position=index,
            payload_message_id=oid(0x1002 + index * 4),
            body=f"file the report {index}",
        )
        for index in range(roster)
    ]
    rows = [
        PayloadDisposition(
            attempt_id=item.attempt_id,
            payload_position=item.payload_position,
            kind=DELIVER,
            policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
            cell=GRADED_CELL,
            resolution_source=REGISTERED,
            family_id=CONTRACT,
        )
        for item in items
    ]
    start = replace(a_start(), tasks=items, dispositions=rows,
                    release=TASK_AHEAD_OF_PAYLOAD,
                    provenance=PolicyProvenance(
                        authority=REGISTERED,
                        roster_digest=roster_digest(rows),
                        experiment_id="the_subject_of_this_run",
                    ))
    sources = {
        item.attempt_id: a_manifest(
            source_attempt_id=item.attempt_id,
            source_seal_id=hidden_seal_id(EXECUTION, 0, item.attempt_id),
            kernel_submission_digest=digest_of(f"filing {item.attempt_id}"),
            bank_filing_digest=digest_of(f"bank filing {item.attempt_id}"),
            canonical_submission=BlobRef(
                sha256=digest_of(f"{SUBMISSION}{item.attempt_id}"),
                size=len(f"{SUBMISSION}{item.attempt_id}"),
                media_type=BODY_MEDIA_TYPE,
            ),
        )
        for item in items
    }
    # What each attempt's two presentations cited: a transcript and the checkpoint with the task,
    # then a transcript and the completed provider turn with the acknowledgement. The checkpoint
    # is the one the attempt itself names, which is how the producer wrote it and what restore
    # requires of a row that names one.
    checkpoints = {item.attempt_id: digest_of(f"checkpoint {item.attempt_id}") for item in items}
    presentations = {
        item.attempt_id: [
            digest_of(f"task transcript {item.attempt_id}"),
            checkpoints[item.attempt_id],
            digest_of(f"ack transcript {item.attempt_id}"),
            digest_of(f"provider turn {item.attempt_id}"),
        ]
        for item in items
    }
    attempts = [
        a_row(
            attempt_id=item.attempt_id,
            state="ack_presented",
            task_start_checkpoint=checkpoints[item.attempt_id],
            seal_id=hidden_seal_id(EXECUTION, 0, item.attempt_id),
            submission_digest=sources[item.attempt_id].kernel_submission_digest,
            graded_evidence=digest_of(f"evidence {item.attempt_id}"),
            seal_ordinal=index + 1,
            source_artifact=sources[item.attempt_id],
            source_commitment=source_commitment(sources[item.attempt_id]),
            presentation_references=presentations[item.attempt_id],
        )
        for index, item in enumerate(items)
    ]
    obligations = [
        CarriedObligation(
            attempt_id=item.attempt_id,
            state="eligible",
            materialized=True,
            candidate=a_candidate(
                message_id=item.payload_message_id,
                attempt_id=item.attempt_id,
                commitment=source_commitment(sources[item.attempt_id]),
            ),
        )
        for item in items
    ]
    inventory = an_inventory(
        *[source_commitment(source) for source in sources.values()],
        *[source.canonical_submission.sha256 for source in sources.values()],
        *[row.graded_evidence or "" for row in attempts],
        *[name for names in presentations.values() for name in names],
    )
    committed = _committed_messages(items)
    projection = replace(
        a_projection(
            start, attempts=attempts, obligations=obligations, committed_blobs=inventory
        ),
        cursor=committed[-1].message_id,
        seal_ordinal=roster,
        handed_out_attempt_ids=[item.attempt_id for item in items],
        offer_count=2 * roster,
        eligibility_count=roster,
        activity_ordinal=3 * roster,
        presented=committed,
        attestations=_acknowledgements(committed),
        pull_requests=_reservations(items, kernel_workflow.PULL),
        terminal_requests=_reservations(items, kernel_workflow.SEAL),
        journal=_journal(items, committed, _recipes(start)),
    )
    # The facts being measured are a run's facts. Every attempt has been handed out, has sealed
    # under an ordinal of its own, has committed the two presentations an acknowledged attempt
    # commits, and was answered under the four Updates that took it there.
    assert sorted(row.seal_ordinal or 0 for row in attempts) == list(range(1, roster + 1))
    assert len(projection.handed_out_attempt_ids) == roster
    assert len(projection.presented) == 2 * roster
    assert len(projection.attestations) == 2 * roster
    assert len(projection.journal) == 4 * roster
    assert projection.cursor == items[-1].ack_message_id
    # And the schedule this history is under is the one that could have produced it: a task
    # ahead of a payload is what leaves every obligation eligible at the end of the roster.
    assert start.release.priority == TASK_FIRST
    version = carrier_version(start, projection)
    assert version == RECEIPT_CARRIER_SCHEMA_VERSION
    replaced = replace(start, carry=pack_carrier(projection, CONVERTER, version=version))
    # The size is read off a carrier the boundary's own restore accepts, and it says so first:
    # the packed carrier goes through the constructor, which unpacks it, checks the receipts,
    # writes it over the state the start rebuilt and rebuilds every recipe in the journal.
    restored = kernel_workflow.StreamWorkflow(replaced)
    assert restored._seal_ordinal == roster
    assert restored._cursor == items[-1].ack_message_id
    assert len(restored._committed_blobs) == len(inventory)
    argument = continuation_argument(replaced, CONVERTER, version)
    encoded = CONVERTER.to_payloads([argument])[0].ByteSize()
    print(
        f"\nreceipt carrier at a roster of {roster}: {encoded} bytes against the "
        f"{kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES} byte ceiling"
    )
    assert encoded < kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES
    assert 1 - encoded / kernel_workflow.TURNOVER_PAYLOAD_CEILING_BYTES > 0.25
    # And no cell body crosses with a source, which is what keeps that true as bodies grow. The
    # one body in here is the candidate's own, which is what an offer that could still happen
    # would carry, and the source beside it is references and measurements.
    written = json.dumps(
        json.loads(CONVERTER.to_payloads([projection])[0].data.decode("utf-8"))
    )
    assert json.dumps(BODIES[GRADED_CELL])[1:-1] in written
    assert json.dumps(BODIES[PLACEBO_CELL])[1:-1] not in written
    assert json.dumps(BODIES[ORACLE_CELL])[1:-1] not in written


def test_the_module_this_runs_under_needs_no_event_loop_of_its_own() -> None:
    """Every check above is pure, which is the property a restore depends on.

    A restore opens nothing: no store, no world, no Activity, no clock. So the whole of the
    validation runs here with no loop running and no worker anywhere, which is the same thing a
    successor execution's first activation has.
    """
    with pytest.raises(RuntimeError):
        asyncio.get_running_loop()
    kernel_workflow._check_carried_receipts(a_start(), a_projection(a_start()))


# The fork's pure slice: what a parent's committed source becomes in a child, what number a
# generation that declares a fork writes, and what a child's restore accepts and refuses.

#: The two slots one parent's fork may create, and the one its child below serves. They are
#: internal names for branches and say nothing about what either branch is for.
FIRST_SLOT = "first"
SECOND_SLOT = "second"
CHILD_EXECUTION = "execution-child-1"
FORK = "fork-1"

#: Which registered policy each eligible cell is delivered under.
CELL_POLICIES = {
    GRADED_CELL: GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_CELL: PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
}


def a_fork_parent(*, silent: bool = False, **changes: Any) -> StreamStart:
    """The generation A is sealed in, declaring the slots its fork may create.

    It serves the one slot every generation serves and has forked nothing yet, which is the state
    a fork-capable parent is in from its first boundary.
    """
    declared = replace(a_start(silent=silent), forkable_slots=[FIRST_SLOT, SECOND_SLOT])
    return replace(declared, **changes) if changes else declared


def a_fork_origin(child_hash: str, *, slot: str = SECOND_SLOT, **changes: Any) -> ForkOrigin:
    """The lineage one child carries: which parent, cut where, over which source."""
    manifest = a_manifest()
    declared = ForkOrigin(
        parent_workflow_id="stream/the-parent/1",
        parent_run_id="the-parent-run",
        parent_configuration_hash=configuration_hash(a_fork_parent()),
        child_configuration_hash=child_hash,
        parent_execution_ordinal=0,
        parent_hidden_execution_id=EXECUTION,
        acknowledged_cursor=oid(9),
        projection_digest=digest_of("the projection at the cut"),
        attestation_id=oid(0x201),
        acknowledged_visible_sha256=digest_of("the acknowledgement as it was presented"),
        checkpoint_manifest_reference=digest_of("the checkpoint manifest"),
        source_attempt_id=ATTEMPT,
        source_seal_id=hidden_seal_id(EXECUTION, 0, ATTEMPT),
        source_submission_digest="e" * 64,
        source_canonicalization_version="kernel.1",
        source_score=1.0,
        source_seal_ordinal=1,
        source_graded_evidence="9" * 64,
        source_commitment=source_commitment(manifest),
        source_artifact_references=sorted(
            reference.sha256 for reference in manifest.cells.values()
        ),
        branch_slot=slot,
        dispositions_digest=digest_of(f"the rows of {slot}"),
        start_differences=[
            StartDifference(field_name="served_slot", value_digest=digest_of(slot)),
            StartDifference(
                field_name="hidden_execution_id", value_digest=digest_of(CHILD_EXECUTION)
            ),
        ],
        parent_turnovers=0,
        fork_id=FORK,
        child_ordinal=1,
        children=2,
    )
    return replace(declared, **changes) if changes else declared


def a_child_start(
    *,
    cell: str = PLACEBO_CELL,
    slot: str = SECOND_SLOT,
    origin: bool = True,
    silent: bool = False,
    **changes: Any,
) -> StreamStart:
    """One child of that parent: its own slot, its own row, its own identity, its parent's lineage.

    Everything the transformation is allowed to move is moved here and nothing else is: the child
    resolves the same obligation on a branch of its own, under the policy for the cell it
    delivers, with a hidden execution id of its own and its parent's origin beside its carry.
    """
    parent = a_fork_parent(silent=silent)
    rows = [
        replace(row, branch_slot=slot, policy_digest=CELL_POLICIES[cell], cell=cell)
        if row.kind == DELIVER
        else replace(row, branch_slot=slot)
        for row in parent.dispositions
    ]
    without = replace(
        parent,
        hidden_execution_id=CHILD_EXECUTION,
        consumer_claim_hash="f" * 64,
        # A child keeps a durable store of its own, because a generation given none verifies its
        # committed objects by reading nothing at all.
        blob_root="/the-store-of-this-child",
        served_slot=slot,
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_subject_of_this_run",
        ),
    )
    declared = replace(
        without,
        fork_origin=(
            a_fork_origin(configuration_hash(without), slot=slot) if origin else None
        ),
    )
    return replace(declared, **changes) if changes else declared


def a_child_projection(
    *, cell: str = PLACEBO_CELL, parent: Optional[StreamStart] = None
) -> CarriedProjection:
    """The projection one parent hands a child that delivers this cell.

    It is the parent's carry with the delivery transformed, so it holds the parent's own
    configuration identity and says nothing about which child is about to read it.
    """
    source = a_fork_parent() if parent is None else parent
    return transformed_child_selection(
        a_projection(source),
        selections=[
            ChildSelection(attempt_id=ATTEMPT, cell=cell, policy_digest=CELL_POLICIES[cell])
        ],
    )


def test_a_fork_capable_parent_writes_the_highest_carrier_before_any_child_exists() -> None:
    """The number follows the declaration, exactly as the receipt number does.

    A parent that declares the slots its fork may create writes the highest carrier from its first
    boundary, before any fork exists and whether or not it has captured anything, and a child
    writes it because it holds an origin. Both hand their own start on untouched, so the
    configuration identity the next execution rebuilds from that start is the one the carrier
    repeats.
    """
    parent = a_fork_parent()
    nothing_yet = a_projection(
        parent,
        attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="active")],
        obligations=[],
        committed_blobs=[],
    )
    assert carrier_version(parent, nothing_yet) == FORK_CARRIER_SCHEMA_VERSION
    assert fork_configuration(parent)

    child = a_child_start()
    assert not fork_configuration(replace(child, served_slot=SINGLETON_SLOT, forkable_slots=[]))
    assert (
        carrier_version(replace(child, served_slot=SINGLETON_SLOT, forkable_slots=[]), nothing_yet)
        == FORK_CARRIER_SCHEMA_VERSION
    )

    for start in (parent, child):
        argument = continuation_argument(start, CONVERTER, FORK_CARRIER_SCHEMA_VERSION)
        assert argument is start
        written = json.loads(CONVERTER.to_payloads([argument])[0].data.decode("utf-8"))
        assert written["served_slot"] == start.served_slot
        assert written["forkable_slots"] == start.forkable_slots
        assert (written["fork_origin"] is None) == (start.fork_origin is None)
        rebuilt = CONVERTER.from_payload(CONVERTER.to_payloads([argument])[0], StreamStart)
        assert rebuilt == start
        assert configuration_hash(rebuilt) == configuration_hash(start)


def test_a_defaulted_served_slot_and_an_empty_forkable_set_promote_no_generation() -> None:
    """The default is the slot every row carries today, so reading it as a declaration would
    promote every legacy and receipt-only start to a codec that had never read it.

    Both lower numbers stay eligible: a generation declaring a receipt contract writes the middle
    one, and a generation declaring neither route writes the earliest.
    """
    for start in (a_start(), a_start(contract=False), a_legacy_start()):
        assert (start.served_slot, start.forkable_slots) == (SINGLETON_SLOT, [])
        assert not fork_configuration(start)
    receipt = a_start()
    assert carrier_version(receipt, a_projection(receipt)) == RECEIPT_CARRIER_SCHEMA_VERSION
    plain = a_start(contract=False)
    empty = a_projection(
        plain,
        attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="active")],
        obligations=[],
        committed_blobs=[],
    )
    assert carrier_version(plain, empty) == LEGACY_CARRIER_SCHEMA_VERSION


def test_a_lower_carrier_would_strip_the_fields_a_forked_identity_is_rebuilt_from() -> None:
    """Which is why the number is selected from what a start declares and not from its evidence.

    An adapter run over a fork-capable start drops the two slots from the one copy the next
    execution rebuilds its configuration identity from, restore compares that rebuilt identity
    against the carried one, and the parent would refuse its own lawful continuation.

    Both slots are inside that identity, which is what makes the loss one. The two starts here
    differ in the branch they serve and in nothing else, so a generation serving a branch of its
    own is a different generation from one serving the default.
    """
    parent = a_fork_parent()
    serving_its_own = a_fork_parent(served_slot=FIRST_SLOT)
    assert configuration_hash(serving_its_own) != configuration_hash(parent)
    for start in (parent, serving_its_own):
        written = json.loads(CONVERTER.to_payloads([start])[0].data.decode("utf-8"))
        for adapted in (receipt_start_members(written), legacy_start_members(written)):
            assert "served_slot" not in adapted
            assert "forkable_slots" not in adapted
            stripped = CONVERTER.from_payload(CONVERTER.to_payloads([adapted])[0], StreamStart)
            assert (stripped.served_slot, stripped.forkable_slots) == (SINGLETON_SLOT, [])
            assert configuration_hash(stripped) != configuration_hash(start)


def test_a_receipt_continuation_is_written_as_the_bytes_the_build_before_the_fork_wrote() -> None:
    """The middle number is an adapter now, and this is the bytes it has to keep writing.

    The fixture was recorded from the build that shipped the receipt route, before a fork member
    existed. A round trip through this build's own code would only show that it agrees with
    itself, so what is compared is the recorded outer start and the recorded carrier string.
    """
    recorded = json.loads(
        (Path(__file__).parent / "_fixtures" / "recorded_receipt_carrier.json").read_text(
            encoding="utf-8"
        )
    )
    assert recorded["carrier_schema_version"] == RECEIPT_CARRIER_SCHEMA_VERSION
    start = a_start()
    projection = a_projection(start)
    assert carrier_version(start, projection) == RECEIPT_CARRIER_SCHEMA_VERSION
    packed = pack_carrier(projection, CONVERTER, version=RECEIPT_CARRIER_SCHEMA_VERSION)
    assert (packed.encoding, packed.data) == (recorded["encoding"], recorded["carrier"])
    argument = continuation_argument(
        replace(start, carry=packed), CONVERTER, RECEIPT_CARRIER_SCHEMA_VERSION
    )
    assert CONVERTER.to_payloads([argument])[0].data.decode("utf-8") == recorded["outer_start"]
    # And the recorded carrier still reads back as the projection this build builds, which is what
    # a deployment does to a generation that crossed a boundary under the build before it.
    fixed = replace(packed, data=recorded["carrier"])
    assert unpack_carrier(fixed, CONVERTER) == projection


def test_the_preparation_gate_crosses_the_highest_carrier_and_no_lower_one() -> None:
    """A member the lower numbers had no name for is dropped by their adapters and nothing else.

    The gate is the one member a child's carried obligation holds, so a projection holding one can
    never be written as a record that had no room for it, and the two lower adapters emit the
    obligation the member set they had.
    """
    child = a_child_start()
    projection = a_child_projection()
    assert carrier_version(child, projection) == FORK_CARRIER_SCHEMA_VERSION
    # The gate alone is enough, with nothing about the start declaring anything.
    declaring_nothing = replace(
        child, served_slot=SINGLETON_SLOT, forkable_slots=[], fork_origin=None
    )
    assert not fork_configuration(declaring_nothing)
    assert carrier_version(declaring_nothing, projection) == FORK_CARRIER_SCHEMA_VERSION
    packed = pack_carrier(projection, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION)
    assert unpack_carrier(packed, CONVERTER) == projection

    written = json.loads(CONVERTER.to_payloads([projection])[0].data.decode("utf-8"))
    assert written["obligations"][0]["pending_preparation"] is True
    for adapted in (receipt_projection_members(written), legacy_projection_members(written)):
        assert "pending_preparation" not in adapted["obligations"][0]
        assert adapted["obligations"][0]["state"] == written["obligations"][0]["state"]


def test_the_child_selection_moves_three_values_and_preserves_every_other_thing_the_source_says(
) -> None:
    """Source evidence is immutable and selected delivery evidence is transformed once.

    The cell becomes the child's, the policy digest becomes its own row's and the body reference
    becomes the descriptor's own entry for that cell. The contract id does not move, because both
    cells are the pair one contract declares, and neither does anything the seal committed. The
    descriptor and the origin cross as the mappings they were written as, so the value the child's
    restore reads through the strict readers is the value the parent committed to.
    """
    # The inherited row cites a presentation, so the eighth receipt field is compared against a
    # value the parent wrote rather than against a default both sides would hold anyway, and the
    # inventory names the object it cited, which is what makes this a projection a carrier admits.
    transcript = "7" * 64
    parents = a_projection(
        a_fork_parent(),
        attempts=[a_row(presentation_references=[transcript])],
        committed_blobs=an_inventory(
            source_commitment(a_manifest()), digest_of(SUBMISSION), "9" * 64, transcript
        ),
    )
    kernel_workflow._check_carried_receipts(a_fork_parent(), parents)
    childs = transformed_child_selection(
        parents,
        selections=[
            ChildSelection(
                attempt_id=ATTEMPT,
                cell=PLACEBO_CELL,
                policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
            )
        ],
    )
    before, after = parents.attempts[0], childs.attempts[0]

    assert (after.selected_cell, after.selected_policy_digest) == (
        PLACEBO_CELL,
        PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    )
    assert after.selected_body_reference == digest_of(BODIES[PLACEBO_CELL])
    assert before.selected_body_reference == digest_of(BODIES[GRADED_CELL])
    assert after.receipt_contract_id == before.receipt_contract_id
    # The two mappings are the ones that crossed, not a second writing of them.
    assert after.source_artifact is before.source_artifact
    assert after.source_origin is before.source_origin
    # And every other value the seal committed is the parent's, byte for byte.
    for name in ("source_commitment", "seal_id", "submission_digest", "score", "decode_state",
                 "graded_evidence", "seal_ordinal", "presentation_references", "state"):
        assert getattr(after, name) == getattr(before, name)

    owed = childs.obligations[0]
    assert (owed.candidate, owed.pending_preparation, owed.materialized) == (None, True, True)
    assert owed.state == parents.obligations[0].state
    assert parents.obligations[0].candidate is not None
    # The identity in the carrier is still the parent's, which is what a fresh child carries.
    assert childs.configuration_hash == parents.configuration_hash
    assert childs.configuration_hash == configuration_hash(a_fork_parent())
    # The parent's own projection is untouched, which is what makes two children of one value.
    assert parents.attempts[0] is before


def test_a_child_projection_this_build_composed_restores_into_a_gated_child() -> None:
    """The whole of this step, in one shape: composed by the transformation, read by the restore.

    Both children are the same construction over one parent's projection: the one that delivers
    the cell its parent delivered and the one that delivers the other. Each restores against its
    own start, through the carrier it would actually cross in, with a transformed selection, no
    candidate and the gate its own preparation clears.

    What the carrier says it was composed against is the parent, because a fresh child holds its
    parent's carry, and that is the value a fresh child's own door compares against the hash its
    origin records for its parent.
    """
    for cell, slot in ((GRADED_CELL, FIRST_SLOT), (PLACEBO_CELL, SECOND_SLOT)):
        child = a_child_start(cell=cell, slot=slot)
        projection = a_child_projection(cell=cell)
        kernel_workflow._check_carried_receipts(child, projection)
        packed = pack_carrier(projection, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION)
        read_back = unpack_carrier(packed, CONVERTER)
        origin = child.fork_origin
        assert origin is not None
        assert read_back.configuration_hash == origin.parent_configuration_hash
        assert read_back.configuration_hash != configuration_hash(child)
        kernel_workflow._check_carried_receipts(child, read_back)


def test_the_public_doors_still_refuse_a_caller_supplied_carry_to_a_child() -> None:
    """A child's carry is the platform's to compose, and a caller's is refused as it always was.

    The origin changes nothing about that: the two doors a caller comes in through refuse a start
    that already holds a projection, and a constructor started fresh refuses one too.
    """
    child = a_child_start()
    refuse_a_carried_projection(child)
    carried = replace(
        child,
        carry=pack_carrier(a_child_projection(), CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION),
    )
    with pytest.raises(ValueError, match="no earlier execution"):
        refuse_a_carried_projection(carried)


def test_a_placebo_child_that_kept_the_graded_parents_selection_is_refused_at_restore(
) -> None:
    """Which is why dropping the candidate is not the transformation.

    A child built by dropping the body and leaving the selection alone holds its parent's graded
    cell under its own placebo row, and the refusal comes at restore rather than at delivery: the
    selection a child carries has to be the one its own row resolves.
    """
    child = a_child_start()
    parent = a_projection(a_fork_parent())
    kept = replace(
        parent,
        obligations=[
            replace(parent.obligations[0], candidate=None, pending_preparation=True)
        ],
    )
    assert "does not resolve" in refused(child, kept)


def test_a_child_that_kept_the_parents_candidate_is_refused_either_way() -> None:
    """The body a parent rendered is the parent's, and a child renders its own or serves nothing.

    With the gate standing it is refused as a preparation that already has what it is waiting for;
    with the gate cleared it is refused as a candidate naming a cell its attempt's selection does
    not, which is the ordinary binding check doing its ordinary work.
    """
    child = a_child_start()
    projection = a_child_projection()
    kept = replace(
        projection,
        obligations=[replace(projection.obligations[0], candidate=a_candidate())],
    )
    assert "still waiting on" in refused(child, kept)
    assert "does not" in refused(
        child,
        replace(
            kept, obligations=[replace(kept.obligations[0], pending_preparation=False)]
        ),
    )


def a_child_candidate() -> PayloadCandidate:
    """The body a child's own preparation installs for the cell that child delivers."""
    return a_candidate(
        body=BODIES[PLACEBO_CELL],
        cell=PLACEBO_CELL,
        policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
        body_reference=digest_of(BODIES[PLACEBO_CELL]),
    )


def test_a_child_whose_obligation_lost_its_gate_is_refused_like_any_damaged_carry() -> None:
    """The gate is what admits an unbuilt obligation, and its absence is the ordinary refusal.

    A child missing it is indistinguishable from a carrier that dropped a body, which is exactly
    the state the ordinary rule exists to refuse, so the ordinary rule refuses it.
    """
    child = a_child_start()
    projection = a_child_projection()
    ungated = replace(
        projection,
        obligations=[replace(projection.obligations[0], pending_preparation=False)],
    )
    assert "carries no candidate" in refused(child, ungated)


def test_a_prepared_child_stands_without_its_gate_and_is_refused_once_its_body_is_gone() -> None:
    """The gate is read before installation and after it, and it says different things.

    Preparation installs the candidate and clears the gate in one transition, so the prepared
    child carries a body and no gate. Take the body away afterwards and nothing excuses it: a
    retained origin is not a licence for a missing candidate.
    """
    child = a_child_start()
    projection = a_child_projection()
    prepared = replace(
        projection,
        obligations=[
            replace(
                projection.obligations[0],
                candidate=a_child_candidate(),
                pending_preparation=False,
            )
        ],
    )
    kernel_workflow._check_carried_receipts(child, prepared)
    assert "carries no candidate" in refused(
        child,
        replace(prepared, obligations=[replace(prepared.obligations[0], candidate=None)]),
    )


def test_a_preparation_gate_is_refused_wherever_nothing_could_have_written_one() -> None:
    """Three places: no fork behind it, no committed cell to prepare, and no offer left to owe.

    The gate is a value a parent committed to rather than a shape a carrier can assert, so every
    generation that could not have been handed one refuses it whole.
    """
    # A generation cut from no fork, which is every generation this build serves today. Its own
    # source is its own, so the gate is the only thing wrong with what it is handed.
    parent = a_fork_parent()
    parents = a_projection(parent)
    assert "cut from no fork" in refused(
        parent,
        replace(
            parents,
            obligations=[
                replace(parents.obligations[0], candidate=None, pending_preparation=True)
            ],
        ),
    )

    # A child whose row resolves the position to no committed cell at all.
    scalar = a_start(contract=False)
    forked = replace(scalar, fork_origin=a_fork_origin(configuration_hash(scalar)))
    blinded = a_scalar_projection(scalar, a_blinded_candidate())
    assert "no committed cell to prepare" in refused(
        forked,
        replace(
            blinded,
            obligations=[replace(blinded.obligations[0], pending_preparation=True)],
        ),
    )

    # And an obligation that has been presented, which no offer is owed from any more.
    child = a_child_start()
    projection = a_child_projection()
    presented = replace(
        projection,
        obligations=[
            replace(projection.obligations[0], state="presented", pending_preparation=True)
        ],
    )
    assert "only an obligation an offer could still be made from waits" in refused(
        child, presented
    )


def a_local_row(**changes: Any) -> CarriedAttempt:
    """One attempt a child sealed itself, under its own identity rather than its parent's."""
    manifest = a_manifest(
        source_attempt_id=SILENT,
        source_seal_id=hidden_seal_id(CHILD_EXECUTION, 0, SILENT),
        kernel_submission_digest="d" * 64,
    )
    declared: Dict[str, Any] = {
        "attempt_id": SILENT,
        "seal_id": hidden_seal_id(CHILD_EXECUTION, 0, SILENT),
        "submission_digest": "d" * 64,
        "score": 0.5,
        "graded_evidence": "8" * 64,
        "seal_ordinal": 2,
        "source_artifact": manifest,
        "source_commitment": source_commitment(manifest),
        "source_origin": SourceOriginContext(
            hidden_execution_id=CHILD_EXECUTION, execution_ordinal=0
        ),
        "selected_cell": None,
        "selected_body_reference": None,
        "selected_policy_digest": None,
    }
    declared.update(changes)
    return a_row(**declared)


def a_mixed_projection(rows: List[CarriedAttempt]) -> CarriedProjection:
    """One child's projection holding an inherited attempt and one it captured itself."""
    local = source_commitment(
        a_manifest(
            source_attempt_id=SILENT,
            source_seal_id=hidden_seal_id(CHILD_EXECUTION, 0, SILENT),
            kernel_submission_digest="d" * 64,
        )
    )
    return transformed_child_selection(
        a_projection(
            a_fork_parent(silent=True),
            attempts=rows,
            committed_blobs=an_inventory(
                source_commitment(a_manifest()),
                digest_of(SUBMISSION),
                "9" * 64,
                local,
                "8" * 64,
            ),
        ),
        selections=[
            ChildSelection(
                attempt_id=ATTEMPT,
                cell=PLACEBO_CELL,
                policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
            )
        ],
    )


def test_a_child_carrying_an_inherited_source_and_one_it_captured_itself_is_accepted() -> None:
    """Source authority is a fact about the attempt rather than about the generation.

    The inherited attempt was sealed under the parent's hidden execution identity, which the
    child's own origin records, and the attempt the child captured was sealed under its own. Each
    seal id is recomputed from the context that actually produced it rather than from whichever
    start it arrived at, so one carrier holds both and neither is checked against the other's
    identity.
    """
    child = a_child_start(silent=True)
    kernel_workflow._check_carried_receipts(child, a_mixed_projection([a_row(), a_local_row()]))


def test_each_carried_origin_is_substituted_independently_and_refused() -> None:
    """Two authorized identities are not one identity, and neither vouches for the other's rows.

    A third identity is refused as an origin nothing vouches for. The two authorized ones are
    refused where they are on the wrong row, by the seal id recomputed from the context the row
    now names, which is the check the origin field exists for.
    """
    child = a_child_start(silent=True)
    elsewhere = SourceOriginContext(hidden_execution_id="another-execution", execution_ordinal=0)
    inherited = SourceOriginContext(hidden_execution_id=EXECUTION, execution_ordinal=0)
    local = SourceOriginContext(hidden_execution_id=CHILD_EXECUTION, execution_ordinal=0)

    assert "vouches for" in refused(
        child, a_mixed_projection([a_row(source_origin=elsewhere), a_local_row()])
    )
    assert "vouches for" in refused(
        child, a_mixed_projection([a_row(), a_local_row(source_origin=elsewhere)])
    )
    assert "recomputed seal" in refused(
        child, a_mixed_projection([a_row(source_origin=local), a_local_row()])
    )
    assert "recomputed seal" in refused(
        child, a_mixed_projection([a_row(), a_local_row(source_origin=inherited)])
    )


def test_a_generation_with_no_fork_origin_admits_its_own_identity_and_nothing_else() -> None:
    """Which is the rule this build already had, kept exactly where no origin stands.

    The same inherited attempt that a child's own origin vouches for is refused on sight in a
    generation that was cut from nothing, because the identity that sealed it is not one that
    generation can have been.
    """
    child = a_child_start(silent=True)
    projection = a_mixed_projection([a_row(), a_local_row()])
    assert "vouches for" in refused(replace(child, fork_origin=None), projection)


def test_the_child_transformation_refuses_every_selection_a_parent_could_not_have_made() -> None:
    """It composes a child or it refuses, and it never composes one a restore would turn away.

    The oracle is refused here rather than at the far end, which is what keeps it unselectable
    from the plan rather than from a search of the bytes: all three cells share wrapper bytes, so
    nothing downstream could tell one from another by shape.
    """
    parent = a_projection(a_fork_parent())

    def composed(*selections: ChildSelection, source: Optional[CarriedProjection] = None) -> None:
        transformed_child_selection(
            parent if source is None else source,
            selections=list(selections),
        )

    placebo = ChildSelection(
        attempt_id=ATTEMPT, cell=PLACEBO_CELL, policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST
    )
    composed(placebo)
    with pytest.raises(ValueError, match="a child is served"):
        composed(replace(placebo, cell=ORACLE_CELL))
    with pytest.raises(ValueError, match="selected twice"):
        composed(placebo, placebo)
    with pytest.raises(ValueError, match="holds no attempt"):
        composed(replace(placebo, attempt_id=SILENT))
    with pytest.raises(ValueError, match="no committed selection"):
        composed(
            placebo,
            source=replace(
                parent,
                attempts=[
                    a_row(
                        source_artifact=None,
                        source_commitment=None,
                        source_origin=None,
                        selected_cell=None,
                        selected_body_reference=None,
                        selected_policy_digest=None,
                        receipt_contract_id=None,
                    )
                ],
            ),
        )
    with pytest.raises(ValueError, match="nothing has delivered"):
        composed(
            placebo,
            source=replace(
                parent,
                obligations=[
                    replace(parent.obligations[0], state="presented", candidate=None)
                ],
            ),
        )
    with pytest.raises(ValueError, match="owes no payload"):
        composed(placebo, source=replace(parent, obligations=[]))


def test_a_gated_child_projection_is_written_over_the_state_its_start_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Admitting the gate is half of it: the obligation has to come back holding it.

    The door a fresh child comes in through is the one this build still closes, a start handed a
    carrier being legal only where the service says an execution continued another, and the
    composed carry holds the identity of the parent it was cut from rather than the child's. So
    the whole constructor is driven over that same composed projection under the identity a
    child's own turnover writes, which is the door this build has. The obligation is materialized,
    unbuilt and gated afterwards, its attempt holds the child's own selection, and the gate leaves
    again in the carrier this child would hand on.
    """
    monkeypatch.setattr(kernel_workflow.workflow, "payload_converter", lambda: CONVERTER)
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "info",
        lambda: SimpleNamespace(
            workflow_id="stream/the-child/1", continued_run_id="the-parent-run"
        ),
    )
    child = a_child_start(slot=SINGLETON_SLOT)
    projection = replace(a_child_projection(), configuration_hash=configuration_hash(child))
    packed = pack_carrier(projection, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION)
    restored = kernel_workflow.StreamWorkflow(replace(child, carry=packed))

    owed = restored._obligations[ATTEMPT]
    assert (owed.pending_preparation, owed.candidate, owed.materialized) == (True, None, True)
    attempt = restored._attempts[ATTEMPT]
    assert attempt.selected_cell == PLACEBO_CELL
    assert attempt.selected_body_reference == digest_of(BODIES[PLACEBO_CELL])
    assert attempt.selected_policy_digest == PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST
    # And the source it inherited is the parent's, read back through the strict readers.
    assert attempt.source_origin == SourceOriginContext(
        hidden_execution_id=EXECUTION, execution_ordinal=0
    )
    assert attempt.seal_id == hidden_seal_id(EXECUTION, 0, ATTEMPT)
    # And the gate is written back out. A child that reaches a boundary before its preparation
    # lands hands its successor an obligation still holding it, rather than one that reads as a
    # carry that lost its candidate.
    handed_on = restored._projection().obligations
    assert [(owed.attempt_id, owed.pending_preparation) for owed in handed_on] == [(ATTEMPT, True)]


# The door a fresh child comes in through, and what it may do before its lineage is authorized.

#: The namespace the run's identities are derived under, which is what makes a child's own
#: identity a value nothing mints.
NAMESPACE = "default"


def a_child_id(ordinal: int) -> str:
    """The identity that child of that parent is derived under."""
    return child_workflow_id(
        identity_namespace=NAMESPACE,
        parent_workflow_id="stream/the-parent/1",
        fork_id=FORK,
        child_ordinal=ordinal,
    )


def a_fresh_child(
    *, cell: str = PLACEBO_CELL, presented: bool = False, **changes: Any
) -> StreamStart:
    """One child exactly as its parent hands it over: its own start, its parent's whole carry.

    The carrier is the two transformations a parent makes and no third, packed at the version a
    fork writes, so this is the value a fresh child's own constructor reads rather than a
    projection composed for a check. The lineage carries the difference projection the parent
    computed over the two starts, which is what the child compares its own members against.
    """
    parent = a_fork_parent()
    child = a_child_start(cell=cell, slot=SINGLETON_SLOT)
    origin = child.fork_origin
    assert origin is not None
    composed = a_projection(parent)
    carried = child_carrier(
        replace(
            composed,
            presented=_committed_messages(parent.tasks) if presented else composed.presented,
        ),
        selections=[
            ChildSelection(attempt_id=ATTEMPT, cell=cell, policy_digest=CELL_POLICIES[cell])
        ],
    )
    handed = replace(
        child,
        fork_origin=replace(
            origin, start_differences=start_difference_projection(parent, child)
        ),
        carry=pack_carrier(carried, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION),
    )
    return replace(handed, **changes) if changes else handed


def a_recorded_row(child: StreamStart, **changes: Any) -> PreparedChild:
    """The parent's immutable row for that child, which is the authority the child asks against."""
    origin = child.fork_origin
    assert origin is not None
    delivered = next(row for row in child.dispositions if row.kind == DELIVER)
    declared = PreparedChild(
        child_ordinal=origin.child_ordinal,
        child_workflow_id=a_child_id(origin.child_ordinal),
        complete_start_digest=complete_start_digest(child),
        origin_digest=origin_digest(origin),
        branch_slot=child.served_slot,
        target_cell=delivered.cell,
        selected_body_reference=digest_of(BODIES[delivered.cell]),
        consumer_claim_hash=child.consumer_claim_hash,
        hidden_execution_id=child.hidden_execution_id,
        start_differences=list(origin.start_differences),
    )
    return replace(declared, **changes) if changes else declared


def serving_as(
    monkeypatch: pytest.MonkeyPatch, *, workflow_id: str, continued: Optional[str] = None
) -> None:
    """Run the constructor as the service would: under this identity, continuing that run.

    The clock and the rest of the execution's identity are here because a fork reads them: the
    boundary compares deadlines against the generation's own clock, and the request names the exact
    execution it was prepared against.
    """
    monkeypatch.setattr(kernel_workflow.workflow, "payload_converter", lambda: CONVERTER)
    monkeypatch.setattr(kernel_workflow.workflow, "now", lambda: A_MOMENT)
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "info",
        lambda: SimpleNamespace(
            workflow_id=workflow_id,
            continued_run_id=continued,
            run_id=PARENT_RUN,
            namespace="default",
            task_queue="the task queue of this run",
        ),
    )


def refused_start(start: StreamStart) -> str:
    """Construct this generation, and return what its own constructor refused it with."""
    with pytest.raises(ApplicationError) as raised:
        kernel_workflow.StreamWorkflow(start)
    assert raised.value.type == "CarrierRefused"
    return str(raised.value)


def a_claim(child: StreamStart, **changes: Any) -> OwnershipClaim:
    """The first claim a child's own gateway makes, which is a claim like any other."""
    declared = OwnershipClaim(
        claimant_id="the child's own gateway",
        previous_epoch=0,
        fencing_token="c" * 64,
        configuration_hash=configuration_hash(child),
        reason="fresh",
    )
    return replace(declared, **changes) if changes else declared


def test_a_fresh_child_restores_through_the_door_its_own_lineage_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The entry this build did not have: a start the service created fresh, holding a carry.

    A carry is legal in four entries. This is the second of them: the child was created rather
    than continued, it carries the projection its parent composed, and what admits that carry is
    the parent hash its own lineage records. It comes up holding the whole inherited prefix, with
    its own selection, no candidate, the gate its preparation clears, and a first claim to make.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    child = a_fresh_child()
    restored = kernel_workflow.StreamWorkflow(child)

    origin = child.fork_origin
    assert origin is not None
    assert restored._origin_unverified
    assert restored._cursor == origin.acknowledged_cursor
    assert restored._seal_ordinal == origin.source_seal_ordinal
    # Ownership is a first claim's to install, and the parent's token can never write here.
    assert (restored._ownership_epoch, restored._fencing_token_hash) == (0, None)
    owed = restored._obligations[ATTEMPT]
    assert (owed.pending_preparation, owed.candidate, owed.materialized) == (True, None, True)
    # The source evidence crosses the entry exactly as its parent committed it, which is what
    # makes the seal the child inherits the parent's filing rather than a copy of one. Only the
    # selected delivery moved, and it moved where the parent built the start.
    attempt = restored._attempts[ATTEMPT]
    inherited = a_row()
    assert attempt.source_artifact == a_manifest()
    assert attempt.source_origin == SourceOriginContext(
        hidden_execution_id=EXECUTION, execution_ordinal=0
    )
    assert attempt.seal_id == hidden_seal_id(EXECUTION, 0, ATTEMPT)
    for name in (
        "state",
        "seal_id",
        "submission_digest",
        "score",
        "decode_state",
        "graded_evidence",
        "seal_ordinal",
        "source_commitment",
        "receipt_contract_id",
        "presentation_references",
    ):
        assert getattr(attempt, name) == getattr(inherited, name), name
    assert attempt.selected_cell == PLACEBO_CELL != inherited.selected_cell
    assert attempt.selected_body_reference == digest_of(BODIES[PLACEBO_CELL])
    assert attempt.selected_policy_digest == PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST

    # And the door stays shut for the generation that carries no lineage at all.
    assert "started fresh" in refused_start(replace(child, fork_origin=None))


def test_a_generation_cut_from_a_fork_and_handed_no_projection_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child is the prefix it inherited, so a lineage with nothing behind it is not one.

    Both ways round: created with a lineage and no carry, and continued with neither.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    alone = replace(a_fresh_child(), carry=None)
    assert "no carried projection" in refused_start(alone)

    serving_as(monkeypatch, workflow_id=a_child_id(1), continued="the childs earlier run")
    assert "no carried projection" in refused_start(alone)


def test_a_fresh_childs_carrier_is_admitted_by_the_hash_its_lineage_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Which hash the comparison is against is the whole of this, and it is the parent's.

    A fresh child holds its parent's carry, so the identity inside it is the parent's. Comparing
    it against the child's own would admit a carrier composed for this child by anybody, and
    refuse the one its parent actually handed over.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    child = a_fresh_child()
    for hash_of in (configuration_hash(child), "0" * 64):
        composed = replace(
            child_carrier(
                a_projection(a_fork_parent()),
                selections=[
                    ChildSelection(
                        attempt_id=ATTEMPT,
                        cell=PLACEBO_CELL,
                        policy_digest=CELL_POLICIES[PLACEBO_CELL],
                    )
                ],
            ),
            configuration_hash=hash_of,
        )
        elsewhere = replace(
            child,
            carry=pack_carrier(composed, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION),
        )
        assert "other than the parent its lineage names" in refused_start(elsewhere)


def test_a_fresh_child_cut_somewhere_its_carrier_does_not_stand_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cursor and the filing count the lineage names are the ones the carrier holds.

    Both are fresh-entry comparisons: they say where this child was cut, and a carrier standing
    anywhere else is a projection from another moment of the parent's life.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    child = a_fresh_child()
    origin = child.fork_origin
    assert origin is not None
    moved = replace(
        child, fork_origin=replace(origin, acknowledged_cursor=oid(0x999))
    )
    assert "cut at the cursor" in refused_start(moved)
    counted = replace(child, fork_origin=replace(origin, source_seal_ordinal=2))
    assert "cut over 2 filings" in refused_start(counted)


def test_a_lineage_this_build_cannot_read_or_that_names_another_child_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two every-entry checks, so a child that fails either refuses whole at any entry.

    A lineage record is immutable provenance rather than an authorization of anything, and a
    child that has continued twice still has to be the child that record was written about.
    """
    child = a_fresh_child()
    origin = child.fork_origin
    assert origin is not None
    unreadable = replace(
        child, fork_origin=replace(origin, schema_version=origin.schema_version + 1)
    )
    another = replace(child, fork_origin=replace(origin, child_configuration_hash="0" * 64))
    continued = a_continued_child()
    onwards = continued.fork_origin
    assert onwards is not None
    for entry, unread, mistaken in (
        (None, unreadable, another),
        (
            "the childs earlier run",
            replace(
                continued,
                fork_origin=replace(onwards, schema_version=onwards.schema_version + 1),
            ),
            replace(continued, fork_origin=replace(onwards, child_configuration_hash="0" * 64)),
        ),
    ):
        serving_as(monkeypatch, workflow_id=a_child_id(1), continued=entry)
        assert "not one this build reads" in refused_start(unread)
        assert "records another generation's identity" in refused_start(mistaken)


def test_a_child_cut_from_a_fork_and_given_no_store_is_refused_at_either_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child inherits a closure of objects and reads its own body out of one.

    A generation given no store verifies its committed objects by reading nothing at all, so a
    child without one would claim over a closure nobody checked and prepare a body out of a store
    that is not there. It is refused where the rest of the lineage is checked, which is at every
    entry rather than only at the one it was cut at.
    """
    for entry, child in (
        (None, a_fresh_child()),
        ("the childs earlier run", a_continued_child()),
    ):
        serving_as(monkeypatch, workflow_id=a_child_id(1), continued=entry)
        assert "given no store" in refused_start(replace(child, blob_root=None))


def a_continued_child(**changes: Any) -> StreamStart:
    """That same child at its own later boundary: its own carry, its lineage retained.

    The carrier holds this child's own identity, because the child composed it, and it stands
    past the cursor and the filing count the lineage records, because a continued child has
    advanced past them lawfully.
    """
    child = a_fresh_child(presented=True)
    parent = a_fork_parent()
    composed = child_carrier(
        a_projection(parent),
        selections=[
            ChildSelection(
                attempt_id=ATTEMPT,
                cell=PLACEBO_CELL,
                policy_digest=CELL_POLICIES[PLACEBO_CELL],
            )
        ],
    )
    onwards = replace(
        composed,
        configuration_hash=configuration_hash(child),
        cursor=oid(0x103),
        seal_ordinal=2,
        presented=_committed_messages(parent.tasks),
        ownership_epoch=1,
        fencing_token_hash="a" * 64,
        consumer_id="the child's own gateway",
        claim_epoch=1,
    )
    carried = replace(
        child, carry=pack_carrier(onwards, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION)
    )
    return replace(carried, **changes) if changes else carried


def test_a_child_continuation_is_authorized_by_the_service_and_by_its_own_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fourth legal entry, and the half of the matrix that is not reapplied to it.

    A continued child's carry is its own, authorized by the service's continuation fact and
    compared against this start's own identity rather than against the parent hash its lineage
    records. The cursor and the filing count that lineage names are not compared at all: the
    child has advanced past them lawfully, and the rows it presented since are its own work
    rather than a mutation of the prefix it inherited. Its lineage is retained through all of it,
    authorizing nothing about the carry beside it.
    """
    serving_as(
        monkeypatch, workflow_id=a_child_id(1), continued="the childs earlier run"
    )
    child = a_continued_child()
    restored = kernel_workflow.StreamWorkflow(child)

    origin = child.fork_origin
    assert origin is not None
    assert restored._cursor == oid(0x103) != origin.acknowledged_cursor
    assert restored._seal_ordinal == 2 != origin.source_seal_ordinal
    assert restored._start.fork_origin == origin
    # The rows it added after the cut are its own lawful work and are kept whole.
    assert list(restored._presented) == [
        row.message_id for row in _committed_messages(a_fork_parent().tasks)
    ]
    # And it is not gated again: the comparison against the parent it was cut from was made at
    # the entry that was cut, and this execution is not that entry.
    assert not restored._origin_unverified

    # A continued child holding the carry its parent composed is refused, which is the other
    # direction of the same rule: that carrier's identity is not this generation's.
    parents = replace(child, carry=a_fresh_child().carry)
    assert "another generation" in refused_start(parents)

    # And a continuation that dropped its lineage is not an ordinary generation that never had
    # one. The source it inherited was sealed under the parent's identity, and with the record
    # gone nothing in the start vouches for that, so the carrier is refused whole.
    assert "no fork origin of its own vouches for" in refused_start(
        replace(child, fork_origin=None)
    )


def test_a_gated_child_owns_nothing_and_serves_nothing_until_its_lineage_is_authorized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate, in the three places a child could otherwise spend what it was made for.

    Every accepted Update is rejected before it is admitted, which costs the generation nothing
    and leaves the same request under the same identifier reaching a child that has done its own
    first work. Ownership is what a claim installs and serving is what a writer does, and the
    handler-side reading of both says so too: the writer path says the generation has not
    authorized its lineage rather than saying an owner that never existed fenced somebody.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    child = a_fresh_child()
    restored = kernel_workflow.StreamWorkflow(child)
    writer = Writer(ownership_epoch=0, fencing_token="c" * 64)

    gated = (
        lambda: restored._claim_ownership_admitted(a_claim(child)),
        lambda: restored._pull_admitted(
            PullRequest(request_id=oid(0x711), last_presented_cursor=oid(9)), writer
        ),
        lambda: restored._confirm_state_admitted(writer),
        lambda: restored._check_claim(a_claim(child)),
        lambda: restored._require_writer(writer),
    )
    for refused_call in gated:
        with pytest.raises(ApplicationError) as raised:
            refused_call()
        assert raised.value.type == ORIGIN_UNVERIFIED

    restored._admit_lineage(a_recorded_row(child))

    assert not restored._origin_unverified
    # The claim the gate was refusing is admitted now, and the writer path has gone back to
    # answering the question it is there to answer: nobody owns this generation yet.
    restored._claim_ownership_admitted(a_claim(child))
    restored._check_claim(a_claim(child))
    with pytest.raises(ApplicationError) as fenced:
        restored._require_writer(writer)
    assert fenced.value.type == "ProtocolError" and "fenced_writer" in str(fenced.value)

    # And a generation cut from no fork is never gated at all.
    serving_as(monkeypatch, workflow_id="stream/an-ordinary-one/1")
    ordinary = kernel_workflow.StreamWorkflow(a_fork_parent())
    assert not ordinary._origin_unverified
    ordinary._check_claim(a_claim(a_fork_parent()))


def test_a_child_the_parents_record_does_not_name_never_leaves_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authentic start under the wrong identity passes every hash and cursor comparison.

    So the identity this execution is actually running as is compared against the row the parent
    committed, and a disagreement is permanent: the row came from the parent's own record, so
    asking again returns the same two values. The gate stands afterwards.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(0))
    child = a_fresh_child()
    restored = kernel_workflow.StreamWorkflow(child)
    with pytest.raises(ApplicationError) as raised:
        restored._admit_lineage(a_recorded_row(child))
    assert raised.value.type == "OriginRefused"
    assert "runs as" in str(raised.value)
    assert restored._origin_unverified
    with pytest.raises(ApplicationError) as still:
        restored._check_claim(a_claim(child))
    assert still.value.type == ORIGIN_UNVERIFIED


def test_a_carried_row_altered_off_the_boundary_fails_the_complete_start_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the withdrawn row comparison was for, done by the digest that actually covers them.

    A presented row's visible digest and an inherited score are both outside the projection hash,
    which covers the presented message identifiers rather than the rows, so a carrier altered in
    either place restores into a child whose cursor and projection digest are exactly the
    parent's. The complete start digest covers the carrier those rows ride in, so each one moves
    it, and the parent's record is what refuses them.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    child = a_fresh_child(presented=True)
    restored = kernel_workflow.StreamWorkflow(child)
    restored._admit_lineage(a_recorded_row(child))
    authorized = restored._projection_hash()

    carried = unpack_carrier(child.carry, CONVERTER) if child.carry is not None else None
    assert carried is not None
    altered = (
        replace(
            carried,
            presented=[
                replace(carried.presented[0], visible_bytes_sha256="0" * 64),
                *carried.presented[1:],
            ],
        ),
        replace(carried, attempts=[replace(carried.attempts[0], score=0.5)]),
        replace(
            carried,
            attempts=[
                replace(carried.attempts[0], selected_body_reference=digest_of(BODIES[GRADED_CELL]))
            ],
        ),
    )
    for moved in altered[:2]:
        touched = replace(
            child, carry=pack_carrier(moved, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION)
        )
        changed = kernel_workflow.StreamWorkflow(touched)
        # Nothing a presentation attests against has moved.
        assert changed._projection_hash() == authorized
        assert changed._cursor == restored._cursor
        with pytest.raises(ApplicationError) as raised:
            changed._admit_lineage(a_recorded_row(child))
        assert raised.value.type == "OriginRefused"
        assert "complete start" in str(raised.value)
        assert changed._origin_unverified

    # An altered artifact reference moves that same digest, and it is refused a step earlier too:
    # the checks that hold a selection to the source it is a cell of never let it restore at all.
    reference = replace(
        child, carry=pack_carrier(altered[2], CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION)
    )
    assert complete_start_digest(reference) != a_recorded_row(child).complete_start_digest
    assert "other than its source's own entry" in refused_start(reference)


# The parent side of a fork, as the pure predicate a validator and a handler both read.
#
# What is driven here is a constructed generation rather than a running one, because every clause
# below is a comparison against state and none of it awaits anything. The Update itself is driven
# on a real service, where the ledger's own self-counting problem is visible; nothing that runs in
# process can see that at all.

PARENT_ID = "stream/the-parent/1"
PARENT_RUN = "the-parent-run"


def a_quiet_parent(
    monkeypatch: pytest.MonkeyPatch, *, start: Optional[StreamStart] = None, **moved: Any
) -> Any:
    """One fork-capable generation standing exactly where a fork may be taken.

    The attempt is acknowledged, its payload is eligible and undelivered, the acknowledgement is in
    the presented rows with the attestation that committed it, and nothing is part way through.
    """
    declared = a_fork_parent() if start is None else start
    committed = PresentedMessage(
        order=0,
        kind="seal_ack",
        message_id=oid(0x102),
        attempt_id=ATTEMPT,
        visible_bytes_sha256=digest_of("the acknowledgement as it was presented"),
    )
    inventory = an_inventory(
        source_commitment(a_manifest()),
        digest_of(SUBMISSION),
        "9" * 64,
        source_commitment(
            a_manifest(
                source_attempt_id=SILENT, source_seal_id=hidden_seal_id(EXECUTION, 0, SILENT)
            )
        ),
    )
    projection = replace(
        a_projection(declared, committed_blobs=inventory),
        cursor=oid(0x102),
        presented=[committed],
        attestations=[
            CarriedAttestation(
                attestation_id=oid(0x201),
                identity=oid(0x202),
                ack=PresentationAck(
                    attestation_id=oid(0x201),
                    cursor=oid(0x102),
                    stream_state_sha256=digest_of("the state that was attested to"),
                ),
            )
        ],
        **moved,
    )
    serving_as(monkeypatch, workflow_id=PARENT_ID, continued="the execution before")
    return kernel_workflow.StreamWorkflow(
        replace(
            declared,
            carry=pack_carrier(projection, CONVERTER, version=FORK_CARRIER_SCHEMA_VERSION),
        )
    )


def a_two_position_parent(**changes: Any) -> StreamStart:
    """The same parent with a second position, so a prefix can owe more than one payload."""
    base = a_fork_parent()
    assert base.provenance is not None
    items = [
        *base.tasks,
        TaskItem(
            task_position=1,
            attempt_id=SILENT,
            task_message_id=oid(0x105),
            ack_message_id=oid(0x106),
            payload_position=1,
            payload_message_id=oid(0x107),
            body="file the second report",
        ),
    ]
    rows = [
        *base.dispositions,
        replace(base.dispositions[0], attempt_id=SILENT, payload_position=1),
    ]
    declared = replace(
        base,
        tasks=items,
        dispositions=rows,
        provenance=replace(base.provenance, roster_digest=roster_digest(rows)),
    )
    return replace(declared, **changes) if changes else declared


def a_second_row() -> CarriedAttempt:
    """The second position's own sealed attempt, over a source committed for that attempt."""
    manifest = a_manifest(
        source_attempt_id=SILENT, source_seal_id=hidden_seal_id(EXECUTION, 0, SILENT)
    )
    return a_row(
        attempt_id=SILENT,
        seal_id=hidden_seal_id(EXECUTION, 0, SILENT),
        seal_ordinal=2,
        source_artifact=manifest,
        source_commitment=source_commitment(manifest),
    )


def a_plan(slot: str, cell: str, **changes: Any) -> ForkChildPlan:
    """One child a controller asks for: its branch, its rows, its cell, its own identities."""
    rows = [
        replace(row, branch_slot=slot, policy_digest=CELL_POLICIES[cell], cell=cell)
        if row.kind == DELIVER
        else replace(row, branch_slot=slot)
        for row in a_fork_parent().dispositions
    ]
    declared = ForkChildPlan(
        branch_slot=slot,
        dispositions=rows,
        target_cell=cell,
        run_directory=f"/runs/{slot}",
        consumer_claim_hash=digest_of(f"the consumer of {slot}"),
        hidden_execution_id=f"execution-{slot}",
        frozen_plan_digest=digest_of("the frozen plan both children are parked under"),
    )
    return replace(declared, **changes) if changes else declared


def a_plan_over(parent: Any, slot: str, cell: str) -> ForkChildPlan:
    """One child plan resolving the rows the generation being forked actually holds."""
    rows = [
        replace(row, branch_slot=slot, policy_digest=CELL_POLICIES[cell], cell=cell)
        if row.kind == DELIVER
        else replace(row, branch_slot=slot)
        for row in parent._start.dispositions
    ]
    return replace(a_plan(slot, cell), dispositions=rows)


def a_fork_request(parent: Any, **changes: Any) -> ForkRequest:
    """The typed fork a controller submits, carrying both witnesses and the ordered plans."""
    declared = ForkRequest(
        parent_workflow_id=PARENT_ID,
        parent_run_id=PARENT_RUN,
        parent_execution_ordinal=parent._start.execution_ordinal,
        parent_configuration_hash=configuration_hash(parent._start),
        source_attempt_id=ATTEMPT,
        attestation_id=oid(0x201),
        acknowledgement_message_id=oid(0x102),
        acknowledged_visible_sha256=digest_of("the acknowledgement as it was presented"),
        acknowledged_cursor=oid(0x102),
        projection_digest=parent._projection_hash(),
        checkpoint_manifest_reference=digest_of("the checkpoint manifest"),
        fork_id=FORK,
        child_plans=[a_plan(FIRST_SLOT, GRADED_CELL), a_plan(SECOND_SLOT, PLACEBO_CELL)],
    )
    return replace(declared, **changes) if changes else declared


def fork_refusal(parent: Any, request: ForkRequest, **arguments: Any) -> Any:
    """Read this fork against this generation, and return the refusal it raised."""
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._refuse_a_fork(request, **arguments)
    return raised.value


def test_a_quiet_fork_capable_parent_refuses_nothing_about_a_well_formed_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predicate holds where every clause of it holds, which is what the refusals move off."""
    parent = a_quiet_parent(monkeypatch)
    parent._refuse_a_fork(a_fork_request(parent))


def test_a_fork_is_refused_by_the_one_clause_of_a_quiet_boundary_that_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One refusal per clause, each naming the condition rather than saying it is not quiet.

    The boundary a fork is taken at is the continuation's own: the generation is open, not
    draining, has not presented Done, holds no pending message, has no operation in flight, holds
    no environment grant, has no attempt sealing and has no deadline due. A fork arriving when any
    of them fails is refused rather than queued, because a fork that waited would hold the
    generation open against an agent still working.
    """
    moved = {
        "generation_state": lambda parent: setattr(parent, "_generation_state", "done"),
        "draining": lambda parent: setattr(parent, "_draining", True),
        "done_presented": lambda parent: setattr(parent, "_done_presented", True),
        # A message offered and not attested to is the clause a fork meets most often, because it
        # is what a generation holds for the whole of the moment an agent is being answered.
        "pending_message": lambda parent: setattr(
            parent,
            "_pending",
            kernel_workflow._Pending(
                message=OfferedMessage(
                    message_id=oid(0x103),
                    kind="task",
                    visible_text="the task nobody has attested to yet",
                    attempt_id=SILENT,
                ),
                origin=kernel_workflow.PULL,
                request_id=a_transport_id("a pull still being answered"),
            ),
        ),
        "operation_in_flight": lambda parent: setattr(parent, "_operation_in_flight", True),
        "environment_grant": lambda parent: setattr(parent, "_environment_call", "call-1"),
        "attempt_sealing": lambda parent: setattr(
            parent._attempts[ATTEMPT], "state", "sealing"
        ),
        "deadline_due": lambda parent: setattr(
            parent._attempts[ATTEMPT], "deadline_expired", True
        ),
    }
    for clause, break_it in moved.items():
        parent = a_quiet_parent(monkeypatch)
        break_it(parent)
        refusal = fork_refusal(parent, a_fork_request(parent))
        assert refusal.reason == FORK_NOT_QUIET, clause
        assert refusal.clause == clause, clause
    # And the ninth, which is the one clause a generation reaches through its own machinery rather
    # than through a field: a latch is a decision to hand this generation on, and a fork of a
    # generation that has decided that is refused until its successor exists.
    parent = a_quiet_parent(monkeypatch)
    parent._turnover_requested = True
    parent._turnover_available = True
    refusal = fork_refusal(parent, a_fork_request(parent))
    assert refusal.reason == FORK_NOT_QUIET
    assert "latched for a turnover" in refusal.clause


def test_a_handler_that_was_accepted_and_has_not_finished_refuses_a_fork_but_never_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ledger, and the exclusion that keeps a fork from waiting for itself for ever.

    The SDK inserts an Update into its own map before that Update's validator runs and removes it
    in the handler's finally, so a request that waited for the public predicate in its own
    validator or its own handler would wait for itself for ever. The ledger is this generation's
    own: a validator sees exactly the handlers that were accepted and have not finished, and a
    handler names itself and is left out.
    """
    parent = a_quiet_parent(monkeypatch)
    parent._handlers = {"present-4": "commit_presentation"}
    refusal = fork_refusal(parent, a_fork_request(parent))
    assert refusal.reason == FORK_NOT_QUIET
    assert "present-4" in refusal.clause

    parent._handlers = {"fork-1": kernel_workflow.FORK}
    parent._refuse_a_fork(a_fork_request(parent), excluding=("fork-1",))
    # And a second fork under another identity, arriving while the first is still in flight, is
    # exactly the case the exclusion must not widen to.
    refusal = fork_refusal(
        parent, a_fork_request(parent), excluding=("fork-2",)
    )
    assert refusal.reason == FORK_NOT_QUIET
    assert "fork-1" in refusal.clause


def test_a_matching_retry_never_runs_beside_the_handler_already_finishing_that_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A barrier that stands answers a retry from its record, and never beside a live handler.

    A matching request arriving after the barrier is the recovery path, so it is not judged against
    a boundary the parent has already left. That is not permission to run two of them at once: two
    handlers admitted here would schedule the same child's start work and spend the same reserve on
    it, so a fresh identifier arriving while one is in flight is refused as retryable and comes
    back once the fork it would have raced has settled.

    The clause is about another fork handler and about nothing else, so a ledger row for any other
    handler does not hold a retry off. What this generation admits after its barrier is narrower
    still: the fork's own class and nothing more, so such a row is a value this clause is read
    against rather than a state a parked parent reaches.
    """
    parent, request = a_barrier(monkeypatch)
    parent._handlers = {"fork-1": kernel_workflow.FORK}
    # The handler that is finishing this fork reads the boundary again and never sees itself.
    parent._refuse_a_fork(request, excluding=("fork-1",))
    refusal = fork_refusal(parent, request, excluding=("fork-2",))
    assert refusal.reason == FORK_IN_FLIGHT
    assert not refusal.non_retryable
    assert "fork-1" in refusal.clause
    assert "finishing this fork" in refusal.clause

    # A row for a handler that is not this fork, read against the clause rather than reached: a
    # parked parent admits the fork's own class alone, so nothing else is accepted after a barrier.
    parent._handlers = {"confirm-9": "confirm_state"}
    parent._refuse_a_fork(request, excluding=("fork-2",))
    with pytest.raises(kernel_workflow.ForkBarrier):
        parent._admit_after_a_barrier(completion=False)


def test_a_fork_is_refused_by_the_one_witness_that_did_not_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One refusal per compared field, and one per join between the two records.

    The stream's witness is the acknowledged cursor and the projection digest at it, and the
    identifiers around them are what say which acknowledgement is meant. The platform validates
    this half and the adapter attests the other; neither side proves the other's, and a mismatch in
    any of them installs no barrier and creates no child.
    """
    parent = a_quiet_parent(monkeypatch)
    for name, value, said in (
        ("acknowledgement_message_id", oid(0x999), "the acknowledgement of"),
        ("acknowledged_visible_sha256", digest_of("other bytes"), "went as other bytes"),
        ("attestation_id", oid(0x998), "holds no attestation"),
        ("acknowledged_cursor", oid(0x997), "stands at"),
        ("projection_digest", digest_of("another projection"), "other than this generation's"),
    ):
        refusal = fork_refusal(parent, a_fork_request(parent, **{name: value}))
        assert refusal.reason == FORK_WITNESS_MISMATCH, name
        assert said in refusal.clause, name

    # A message this generation never presented, named as the acknowledgement it does owe. The
    # first comparison passes and the row is what is missing, which is a different refusal from a
    # request naming somebody else's message.
    unpresented = a_quiet_parent(monkeypatch)
    unpresented._presented.pop(oid(0x102))
    refusal = fork_refusal(unpresented, a_fork_request(unpresented))
    assert refusal.reason == FORK_WITNESS_MISMATCH
    assert "presented no message" in refusal.clause

    # And the join, which is what a checkpoint naming a record that exists and is about something
    # else meets. No collision and no altered history is needed: the attestation below is a real
    # record of this generation, and it is not a record of the presentation this fork is cut at.
    elsewhere = a_quiet_parent(monkeypatch)
    elsewhere._attestation_identities[oid(0x301)] = oid(0x302)
    elsewhere._attestations[oid(0x301)] = PresentationAck(
        attestation_id=oid(0x301),
        cursor=oid(0x101),
        stream_state_sha256=digest_of("the state at some other presentation"),
    )
    refusal = fork_refusal(
        elsewhere, a_fork_request(elsewhere, attestation_id=oid(0x301))
    )
    assert refusal.reason == FORK_WITNESS_MISMATCH
    assert "committed the presentation of" in refusal.clause


def test_a_prefix_that_owes_more_than_the_inherited_payload_is_never_forked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quiet at capacity one does not mean the inherited payload is what a child pulls first.

    The schedule ranks eligible tasks and eligible obligations together under one declared key, so
    a plan that puts tasks first would serve the next task before the inherited receipt. The rule
    is therefore stated over the obligations rather than inferred from the capacity: the inherited
    one is the sole unresolved obligation, it is eligible and undelivered, and the schedule's next
    selection is precisely that payload.
    """
    older = a_quiet_parent(
        monkeypatch,
        start=a_two_position_parent(),
        obligations=[
            CarriedObligation(
                attempt_id=ATTEMPT, state="eligible", materialized=True, candidate=a_candidate()
            ),
            CarriedObligation(
                attempt_id=SILENT,
                state="materialized",
                materialized=True,
                candidate=a_candidate(
                    message_id=oid(0x107),
                    attempt_id=SILENT,
                    commitment=source_commitment(
                        a_manifest(
                            source_attempt_id=SILENT,
                            source_seal_id=hidden_seal_id(EXECUTION, 0, SILENT),
                        )
                    ),
                ),
            ),
        ],
        attempts=[a_row(), a_second_row()],
    )
    refusal = fork_refusal(older, a_fork_request(older))
    assert refusal.reason == FORK_NOT_QUIET
    assert "inherits one unresolved payload" in refusal.clause

    # An obligation nothing has released is the same refusal at the next clause: the row exists,
    # because the start creates one per assigned position, and what is missing is the release.
    missing = a_quiet_parent(monkeypatch, obligations=[])
    refusal = fork_refusal(missing, a_fork_request(missing))
    assert refusal.reason == FORK_NOT_QUIET
    assert "is assigned and a child is cut over one that is eligible" in refusal.clause

    unready = a_quiet_parent(
        monkeypatch,
        obligations=[
            CarriedObligation(
                attempt_id=ATTEMPT,
                state="materialized",
                materialized=True,
                candidate=a_candidate(),
            )
        ],
    )
    refusal = fork_refusal(unready, a_fork_request(unready))
    assert refusal.reason == FORK_NOT_QUIET
    assert "is materialized and a child is cut over one that is eligible" in refusal.clause

    # And the clause the rule exists for, which the three above never reach: the inherited payload
    # is the sole unresolved obligation and the schedule still puts something else first. The
    # second position here delivers nothing, so it creates no obligation at all and owes nothing,
    # while its task is eligible and a task-first plan ranks it ahead of the receipt.
    ahead = a_fork_parent(silent=True)
    later = a_quiet_parent(
        monkeypatch,
        start=replace(
            ahead,
            release=TASK_AHEAD_OF_PAYLOAD,
            assignments=assignments_for(
                ahead.tasks, TASK_AHEAD_OF_PAYLOAD, without_payload=[SILENT]
            ),
        ),
    )
    assert sorted(later._obligations) == [ATTEMPT]
    assert later._first_eligible() == (kernel_workflow.TASK, SILENT)
    refusal = fork_refusal(later, a_fork_request(later))
    assert refusal.reason == FORK_NOT_QUIET
    assert "next selection of this generation's schedule" in refusal.clause
    assert SILENT in refusal.clause
    # The same generation under the plan its parent actually declared serves the receipt first,
    # so what refuses the fork is the schedule rather than the second position.
    served = a_quiet_parent(
        monkeypatch,
        start=replace(
            ahead,
            assignments=assignments_for(ahead.tasks, IMMEDIATE, without_payload=[SILENT]),
        ),
    )
    assert served._first_eligible() == (kernel_workflow.PAYLOAD, ATTEMPT)
    served._refuse_a_fork(a_fork_request(served))


def test_a_prefix_that_already_delivered_a_payload_is_outside_what_this_build_forks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admissible prefix, restricted so the slot provenance question stays closed.

    A child that changed anything the prefix already delivered would be claiming its inherited
    transcript was produced under rules it was not, so a fork over a prefix that presented a
    payload is refused rather than reconciled against a new branch slot.
    """
    delivered = a_quiet_parent(
        monkeypatch,
        start=a_two_position_parent(),
        obligations=[
            CarriedObligation(
                attempt_id=ATTEMPT, state="eligible", materialized=True, candidate=a_candidate()
            ),
            CarriedObligation(attempt_id=SILENT, state="presented", materialized=True),
        ],
        attempts=[a_row(), a_second_row()],
    )
    refusal = fork_refusal(delivered, a_fork_request(delivered))
    assert refusal.reason == FORK_CONFIGURATION_VIOLATION
    assert "already delivered the payloads" in refusal.clause


def test_a_successor_that_has_been_admitted_or_a_world_that_is_open_refuses_a_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No attempt is offered or active, so no successor is admitted and no live world is open."""
    for state in ("task_offered", "active"):
        parent = a_quiet_parent(
            monkeypatch,
            start=a_two_position_parent(),
            attempts=[a_row(), CarriedAttempt(attempt_id=SILENT, state=state)],
            obligations=[
                CarriedObligation(
                    attempt_id=ATTEMPT,
                    state="eligible",
                    materialized=True,
                    candidate=a_candidate(),
                ),
                CarriedObligation(attempt_id=SILENT, state="assigned"),
            ],
        )
        refusal = fork_refusal(parent, a_fork_request(parent))
        assert refusal.reason == FORK_NOT_QUIET, state
        assert "a successor has been admitted or a world is open" in refusal.clause, state


def test_an_inherited_attempt_whose_evidence_reads_unavailable_is_never_forked_over(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A fork over evidence the parent could not produce hands both children a dependency neither
    can resolve, and the availability axis is the last row naming the attempt rather than any row.
    """
    refused = OperationFailure(
        operation="the delivery that could not read its evidence",
        phase="continued_first_delivery",
        reason="unavailable_evidence",
        outcome="refused",
        generation=PARENT_ID,
        refused_epoch=1,
        attempt_id=ATTEMPT,
    )
    standing = a_quiet_parent(monkeypatch, operation_failures=[refused])
    refusal = fork_refusal(standing, a_fork_request(standing))
    assert refusal.reason == FORK_UNRECOVERABLE_EVIDENCE
    assert "reads unavailable" in refusal.clause

    recovered = a_quiet_parent(
        monkeypatch,
        operation_failures=[refused, replace(refused, outcome="recovered", recovered_epoch=2)],
    )
    recovered._refuse_a_fork(a_fork_request(recovered))

    # And the rule is over every inherited receipt attempt rather than over the one a fork is cut
    # at. An earlier withheld receipt keeps its own committed source, both children inherit it
    # whole, and the prebarrier read opens the source of the selected cell alone, so a claim that
    # could not produce that earlier source leaves a refusal nothing this fork discharges.
    ahead = replace(
        a_fork_parent(silent=True),
        assignments=assignments_for(
            a_fork_parent(silent=True).tasks, IMMEDIATE, without_payload=[SILENT]
        ),
    )
    kept = replace(
        a_second_row(),
        selected_cell=None,
        selected_body_reference=None,
        selected_policy_digest=None,
    )
    predecessor = a_quiet_parent(monkeypatch, start=ahead, attempts=[a_row(), kept])
    assert sorted(predecessor._obligations) == [ATTEMPT]
    predecessor._refuse_a_fork(a_fork_request(predecessor))

    withheld = a_quiet_parent(
        monkeypatch,
        start=ahead,
        attempts=[a_row(), kept],
        operation_failures=[
            replace(
                refused,
                operation="the claim that could not read the source it required",
                phase="ownership_claim",
                attempt_id=SILENT,
            )
        ],
    )
    refusal = fork_refusal(withheld, a_fork_request(withheld))
    assert refusal.reason == FORK_UNRECOVERABLE_EVIDENCE
    assert f"the inherited attempt {SILENT} reads unavailable" in refusal.clause


def test_a_child_plan_naming_a_branch_its_parent_never_declared_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The slots a fork may create are the parent's declaration, inside its configuration hash.

    A child plan naming an undeclared slot is refused against a value the parent committed to
    before it forked, and a plan naming the branch the parent itself serves is refused too: two
    generations serving one slot is two rows for one exposure.
    """
    parent = a_quiet_parent(monkeypatch)
    undeclared = a_fork_request(
        parent, child_plans=[a_plan("third", GRADED_CELL), a_plan(SECOND_SLOT, PLACEBO_CELL)]
    )
    refusal = fork_refusal(parent, undeclared)
    assert refusal.reason == FORK_UNDECLARED_BRANCH
    assert "third" in refusal.clause

    served = a_quiet_parent(
        monkeypatch, start=a_fork_parent(forkable_slots=[SINGLETON_SLOT, SECOND_SLOT])
    )
    refusal = fork_refusal(
        served,
        a_fork_request(
            served,
            child_plans=[
                a_plan(SINGLETON_SLOT, GRADED_CELL),
                a_plan(SECOND_SLOT, PLACEBO_CELL),
            ],
        ),
    )
    assert refusal.reason == FORK_UNDECLARED_BRANCH
    assert "the branch its parent serves" in refusal.clause


def test_a_request_prepared_against_an_execution_that_moved_is_retried_rather_than_decided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A turnover between reading the checkpoint evidence and submitting is an ordinary event.

    The answer is to read the evidence again and resubmit the same fork id under the same
    checkpoint and plans, which is the same logical fork, so this refusal is infrastructure and not
    a decision the experiment records and stops on.
    """
    parent = a_quiet_parent(monkeypatch)
    for name, value in (
        ("parent_workflow_id", "stream/somebody-else/1"),
        ("parent_run_id", "another-run"),
        ("parent_execution_ordinal", 7),
    ):
        refusal = fork_refusal(parent, a_fork_request(parent, **{name: value}))
        assert refusal.reason == FORK_MOVED_EXECUTION, name
        assert refusal.non_retryable is False, name
    configuration = fork_refusal(
        parent, a_fork_request(parent, parent_configuration_hash="c" * 64)
    )
    assert configuration.reason == FORK_CONFIGURATION_VIOLATION
    assert configuration.non_retryable is True


def test_the_two_kinds_of_fork_refusal_are_kept_apart_by_the_reason_they_carry() -> None:
    """The experiment retries infrastructure and never retries a decision, and this is that line."""
    for reason in RETRYABLE_FORK_REFUSALS:
        assert kernel_workflow.ForkRefused(reason, "a clause").non_retryable is False, reason
    for reason in PERMANENT_FORK_REFUSALS:
        assert kernel_workflow.ForkRefused(reason, "a clause").non_retryable is True, reason
    assert set(FORK_REFUSALS) == {
        *RETRYABLE_FORK_REFUSALS,
        *PERMANENT_FORK_REFUSALS,
        FORK_EXPIRED_AUTHORITY,
    }
    # And each of them reads back as the reason it is, because the reason is the type the failure
    # carries: a reason this set gained that the reader could not name would be a refusal a
    # controller met as a fault.
    for reason in FORK_REFUSALS:
        assert kernel_runtime.fork_refusal(kernel_workflow.ForkRefused(reason, "a clause")) == (
            reason
        )


async def test_a_fork_refusal_kept_in_the_outcome_journal_still_names_its_clause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refusal has to say the same thing wherever the failure that carries it is kept.

    What the journal keeps of a failed Update is a type and a message, and nothing else, so an
    exact identifier answered out of it after a boundary or from a closed parent is raised again
    from those two alone. A reason kept beside them as a detail would be gone by then, and a
    controller would read a fork that was refused for an infrastructure reason as neither a
    decision nor something to try again.
    """
    parent = a_quiet_parent(monkeypatch)
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "current_update_info",
        lambda: SimpleNamespace(id="fork-9"),
    )

    async def refuse() -> None:
        raise kernel_workflow.ForkRefused(
            FORK_REPAIRABLE_ABSENCE, "the store could not produce the placebo body"
        )

    with pytest.raises(kernel_workflow.ForkRefused):
        await parent._answering(kernel_workflow.FORK, 1, refuse)

    kept = parent.answered_update("fork-9")
    assert kept.found is True
    assert kept.kind == "failure"
    raised = ApplicationError(kept.message, type=kept.type_name, non_retryable=True)
    assert kernel_runtime.fork_refusal(raised) == FORK_REPAIRABLE_ABSENCE
    assert kernel_runtime.fork_can_be_retried(raised) is True
    assert "the placebo body" in kept.message


class AnsweringHandle:
    """A parent handle that answers one fork Update with the receipt it was given."""

    def __init__(self, receipt: Any) -> None:
        self.receipt = receipt

    async def execute_update(self, *_arguments: Any, **_named: Any) -> Any:
        return self.receipt


class AnsweringClient:
    """A client whose parent answers with one receipt, so the receiving check can be read."""

    def __init__(self, receipt: Any) -> None:
        self.data_converter = default_converter()
        self.receipt = receipt

    def get_workflow_handle_for(self, *_arguments: Any, **_named: Any) -> AnsweringHandle:
        return AnsweringHandle(self.receipt)


def a_receipt(**changes: Any) -> ForkReceipt:
    """One complete fork receipt of the shape a parent answers with."""
    declared = ForkReceipt(
        fork_id=FORK,
        parent_workflow_id=PARENT_ID,
        parent_run_id=PARENT_RUN,
        checkpoint_manifest_reference=digest_of("the checkpoint manifest"),
        children=1,
        child_receipts=[
            ForkChildReceipt(
                child_ordinal=1,
                child_workflow_id=a_child_id(1),
                child_run_id="r" * 36,
                configuration_hash="c" * 64,
                complete_start_digest="d" * 64,
                origin_digest="e" * 64,
                branch_slot=FIRST_SLOT,
                target_cell=GRADED_CELL,
                selected_body_reference=digest_of(BODIES[GRADED_CELL]),
                acknowledged_cursor=oid(0x102),
                projection_digest=digest_of("the projection at the cut"),
                start_differences=[],
                next_task_body_sha256="f" * 64,
                next_assignment_id=SILENT,
            )
        ],
        boundary_evidence=[],
    )
    return replace(declared, **changes) if changes else declared


async def test_a_receipt_at_a_version_this_build_does_not_read_is_refused_where_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The version a receipt declares is checked at the boundary the receipt is received at.

    A receipt is evidence a consumer decides over, so the one thing that says whether this build
    may read it at all is checked before any of it is read. The check belongs here rather than
    inside whichever operation goes on to consume the receipt, because a route that skipped it
    would hand a consumer a shape nothing admitted.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    assert (
        await kernel_runtime.fork_stream(AnsweringClient(a_receipt()), request) == a_receipt()
    )

    with pytest.raises(WireFormatError) as raised:
        await kernel_runtime.fork_stream(AnsweringClient(a_receipt(schema_version=999)), request)
    assert "999" in str(raised.value)


def test_a_worker_serves_the_environments_own_activities_and_the_forks_own_beside_them() -> None:
    """The registration a fork depends on, which nothing else would have supplied.

    A Worker takes whatever Activity list it is handed wholesale, and an environment that brings
    its own terminal hands over only its seal, its grade, the payload bundle Activity and the blob
    verification Activity. The fork's four are none of those, so they are registered explicitly
    beside whatever an environment supplied.
    """
    supplied = [seal_attempt_activity, verify_blobs_activity]
    served = [
        getattr(one, "__temporal_activity_definition").name
        for one in kernel_runtime._registered(supplied)
    ]
    assert served[: len(supplied)] == [
        "shogym.protocol_v2.SealAttemptActivity",
        "shogym.protocol_v2.VerifyBlobsActivity",
    ]
    assert served[len(supplied) :] == [
        getattr(one, "__temporal_activity_definition").name for one in fork_activities()
    ]
    assert len(set(served)) == len(served)
    # A caller that supplied one of them itself keeps its own, because a Worker refuses two
    # Activities of one name.
    both = kernel_runtime._registered([*supplied, fork_availability_activity])
    assert len(both) == len(served)


def test_a_worker_a_read_stands_up_to_answer_a_query_serves_no_activity() -> None:
    """A caller handing no Activity at all is not composing a Worker that serves a generation.

    A read replays the generation to answer a Query and registers nothing, beside the Worker
    already serving the run. An Activity registered there would make it a second Worker offering
    to work the run's task queue, which the SDK refuses.
    """
    assert kernel_runtime._registered([]) == []


def test_every_fork_only_activity_takes_its_identifier_from_the_forks_own_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child's ordinary Activity numbering has to equal the number its inherited prefix left.

    So a fork-only invocation never consumes the generation's own ordinal, and the identifier it
    takes names the fork, the step and which attempt at that step it is. The ordinary counter is
    untouched by any of it, which is what lets a child be compared with an unforked twin.
    """
    parent = a_quiet_parent(monkeypatch)
    ordinal = parent._activity_ordinal
    taken = [
        parent._fork_activity(FORK, step)
        for step in (FORK_AVAILABILITY_STEP, FORK_START_STEP, FORK_START_STEP)
    ]
    assert taken == [
        f"fork.{FORK}.availability.1",
        f"fork.{FORK}.start.1",
        f"fork.{FORK}.start.2",
    ]
    assert parent._activity_ordinal == ordinal
    assert parent._next_activity_id() == str(ordinal)
    # A generation no fork cut reads under the ordinary ordinal, its own first claim included.
    assert parent._first_claim_activity() is None
    with pytest.raises(WireFormatError):
        fork_activity_id(fork_id=FORK, step="a step this build does not perform", ordinal=1)

    # A child's own first claim is the one read of the existing blob verification Activity that
    # no twin makes, so it takes a fork identifier too and a repair of it is the next attempt at
    # that step. Once the epoch has moved, a claim is an ordinary resume the twin makes as well.
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    child = kernel_workflow.StreamWorkflow(a_fresh_child())
    inherited = child._activity_ordinal
    assert child._first_claim_activity() == f"fork.{FORK}.claim.1"
    assert child._first_claim_activity() == f"fork.{FORK}.claim.2"
    assert child._activity_ordinal == inherited
    child._ownership_epoch = 1
    assert child._first_claim_activity() is None


def test_a_request_the_service_would_never_carry_is_refused_where_admission_is_decided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A request the service would never admit is a decision, and decisions are made here.

    Admission is settled synchronously and without writing, and what the request says about itself
    includes how many bytes of it there are: a generation asked to fork over a shape nothing could
    carry can answer without reading a single one of its own conditions. The refusal is permanent,
    it carries the measurement it was refused over, and no barrier stands behind it.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    parent._refuse_a_fork(request)
    [first, second] = request.child_plans
    oversized = replace(
        request,
        child_plans=[
            replace(first, run_directory="/runs/one/" + "d" * TURNOVER_PAYLOAD_CEILING_BYTES),
            second,
        ],
    )
    refusal = fork_refusal(parent, oversized)
    assert refusal.reason == FORK_CONFIGURATION_VIOLATION
    assert refusal.non_retryable
    assert "the fork request" in refusal.clause
    assert str(TURNOVER_PAYLOAD_CEILING_BYTES) in refusal.clause
    assert parent._fork is None


def test_a_child_start_this_generation_could_not_hand_on_refuses_the_fork_before_its_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A shape one fork transmits that will not fit is a refusal, never a fault of the parent.

    The cap here is the parent's own complete start, measured, so this is a generation sitting at
    the exact edge of what the service will carry for it. A child start is that start plus the
    lineage record, which is why building one crosses a limit its parent did not, and it is also
    why the refusal has to arrive as a decision: an oversize let out of an accepted call as the
    plain wire failure it is raised as would fail the Workflow Task rather than the call, and the
    generation would replay it for ever with no barrier installed and no refusal to read.

    The plan is measured first and fits, so what is named is the shape that did not, and nothing
    was committed: no barrier, no prepared record and no child.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    edge = encoded_size(parent._start, CONVERTER)
    monkeypatch.setattr(kernel_workflow, "TURNOVER_PAYLOAD_CEILING_BYTES", edge)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    refusal = raised.value
    assert refusal.reason == FORK_CONFIGURATION_VIOLATION
    assert refusal.non_retryable
    assert "the start of child 1" in refusal.clause
    assert str(edge) in refusal.clause
    assert parent._fork is None
    assert parent._generation_state == "open"
    assert parent._fencing_token_hash is not None
    # The parent's own start is what the cap was taken from, so the shape that crossed it is the
    # child's alone, and the plan the child was built from is well inside it.
    assert encoded_size(request.child_plans[0], CONVERTER) < edge


def test_a_generation_declaring_more_than_one_live_attempt_is_never_forked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Capacity one is a declaration, and slots nobody is standing in are not that declaration.

    A fork above capacity one would clone attempts with live worlds, and cloning a world needs a
    world snapshot contract this build does not write. A generation declaring two and serving none
    is quiet, has no offered or active attempt and occupies no slot, so nothing about what it is
    doing says it may be forked; what says so is the number it declared.
    """
    roomy = a_quiet_parent(monkeypatch, start=a_fork_parent(capacity=2))
    assert roomy._capacity_in_use() == 0
    refusal = fork_refusal(roomy, a_fork_request(roomy))
    assert refusal.reason == FORK_CONFIGURATION_VIOLATION
    assert refusal.non_retryable
    assert "declaring one live attempt and this one declares 2" in refusal.clause
    assert roomy._fork is None


def test_a_child_whose_plan_its_own_restore_would_refuse_never_reaches_a_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The classification says what a child may change; it does not say the result is a generation.

    A plan whose target cell and whose own delivering row name different cells is a well formed
    request over a well classified transformation, and the start it produces is one the child's own
    constructor refuses. A parent that committed a barrier over it would have fenced itself for
    ever in order to create a generation nothing can start, so the child's own pure checks are run
    over both completed starts here, before anything is committed.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan(FIRST_SLOT, GRADED_CELL, target_cell=PLACEBO_CELL),
            a_plan(SECOND_SLOT, PLACEBO_CELL),
        ],
    )
    # Nothing about the request, the boundary, the witnesses or the declared branches is wrong.
    parent._refuse_a_fork(request)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    refusal = raised.value
    assert refusal.reason == FORK_CONFIGURATION_VIOLATION
    assert refusal.non_retryable
    assert "child 1 is one that child's own restore refuses" in refusal.clause
    assert "does not resolve" in refusal.clause
    assert parent._fork is None
    assert parent._generation_state == "open"
    assert parent._fencing_token_hash is not None
    # And the second plan, which is the lawful one, is what the same build accepts.
    parent._built_children(a_fork_request(parent))


def test_a_child_start_no_ordinary_constructor_would_admit_never_reaches_a_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The admission every generation is created through runs over both completed starts.

    A plan can be well formed, well classified and refused by no clause the fork itself checks, and
    still compose a start nothing can be created from. A row saying the platform stamped it inside
    an experiment is such a plan: the branch, the cell, the policy, the contract and the payload
    position are all the parent's own, the child transformation admits it, and the schedule and
    policy admission every constructor runs refuses it. So both starts are held to that admission
    here, where a refusal installs no barrier and creates no child.
    """
    parent = a_quiet_parent(monkeypatch)
    stamped = a_plan(FIRST_SLOT, GRADED_CELL)
    request = a_fork_request(
        parent,
        child_plans=[
            replace(
                stamped,
                dispositions=[
                    replace(row, resolution_source=PLATFORM_DEFAULT)
                    for row in stamped.dispositions
                ],
            ),
            a_plan(SECOND_SLOT, PLACEBO_CELL),
        ],
    )
    # Nothing about the request, the boundary, the witnesses or the declared branches is wrong.
    parent._refuse_a_fork(request)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    refusal = raised.value
    assert refusal.reason == FORK_CONFIGURATION_VIOLATION
    assert refusal.non_retryable
    assert "child 1 is one no generation is created from" in refusal.clause
    assert "configuration_mismatch" in refusal.clause
    assert parent._fork is None
    assert parent._generation_state == "open"
    assert parent._fencing_token_hash is not None
    # And the same generation under the rows a child may lawfully carry builds both starts.
    parent._built_children(a_fork_request(parent))


def test_the_receipt_a_fork_will_answer_with_is_bounded_before_its_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reply is measured before the barrier, and the arriving one against what was proved.

    A receipt cannot be measured exactly before the children exist, because a child's initial exact
    run id comes back in its start response. So the known parts are measured with the run ids
    empty, each run id is allowed its declared ceiling, the converter's own wrapper is measured
    empty, and that bound is what has to fit. Comparing the receipt against a bound computed from
    that same receipt would compare a value with itself and admit any size at all.

    The origin verification's question and its answer are measured here too, because they are the
    other two shapes this operation transmits that are replies rather than requests.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    known = parent._the_receipt(request, built)
    assert all(row.child_run_id == "" for row in known.child_receipts)
    bounds = parent._measured_replies(request, built)
    assert bounds.receipt == fork_receipt_bound(known, CONVERTER)
    assert bounds.receipt > encoded_size(known, CONVERTER)

    # A bound that does not fit is a decision about the request, taken before anything is fenced.
    monkeypatch.setattr(kernel_workflow, "TURNOVER_PAYLOAD_CEILING_BYTES", bounds.receipt - 1)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._measured_replies(request, built)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "the fork receipt is bounded at" in raised.value.clause
    assert parent._fork is None
    monkeypatch.undo()


#: The task nobody was ever offered, and the one both children go on to work.
FINISHED = "e" * 32
NEXT = "d" * 32


def a_parent_holding_a_finished_task() -> StreamStart:
    """A roster whose next runnable task stands behind one that ended before it was offered.

    The middle task is finalized where it was planned, which a controller may ask for and the
    boundary admits: the fork clauses read live attempts and unresolved payloads, and an attempt
    that ended without ever being handed out is neither. The last task is the one both children
    work, held by the gate this generation's plan declares until the payload the fork hands them
    has been presented, which is the gate a child keeps.
    """
    base = a_fork_parent()
    assert base.provenance is not None
    items = [
        *base.tasks,
        TaskItem(
            task_position=1,
            attempt_id=FINISHED,
            task_message_id=oid(0x105),
            ack_message_id=oid(0x106),
            payload_position=1,
            payload_message_id=oid(0x107),
            body="file the report nobody was asked for",
        ),
        TaskItem(
            task_position=2,
            attempt_id=NEXT,
            task_message_id=oid(0x108),
            ack_message_id=oid(0x109),
            payload_position=2,
            payload_message_id=oid(0x10A),
            body="file the second report",
        ),
    ]
    rows = [
        *base.dispositions,
        *(
            PayloadDisposition(
                attempt_id=attempt_id,
                payload_position=position,
                kind=WITHHOLD,
                reason="this position delivers nothing",
                resolution_source=REGISTERED,
                family_id=CONTRACT,
            )
            for attempt_id, position in ((FINISHED, 1), (NEXT, 2))
        ),
    ]
    release = replace(
        IMMEDIATE, gates=[EligibilityGate(attempt_id=NEXT, after_payload_position=0)]
    )
    return replace(
        base,
        tasks=items,
        dispositions=rows,
        release=release,
        assignments=assignments_for(items, release, without_payload=[FINISHED, NEXT]),
        provenance=replace(base.provenance, roster_digest=roster_digest(rows)),
    )


def test_the_receipt_names_the_task_a_child_serves_and_not_one_that_already_ended(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same B is read out of each child's schedule rather than off the front of its roster.

    A roster is not a queue of what is left: an attempt may end where it was planned, never
    offered and never handed out, and the schedule then passes over it exactly as it passes over
    one that was worked. So the receipt asks the child's own plan what it would serve after the
    inherited payload, gate and declared order included, rather than naming the first task no
    pull has taken. A reader comparing the two children is comparing the task they will work.

    What the receipt names it by is the roster row the schedule selected, which is the identity
    the join is made on: the attempt that row stands for is derived from it and not the other way
    round, so a receipt naming the attempt would leave a reader holding neither end of the join.
    """
    parent = a_quiet_parent(
        monkeypatch,
        start=a_parent_holding_a_finished_task(),
        attempts=[
            a_row(),
            CarriedAttempt(attempt_id=FINISHED, state="final_failed", final_failure="abandoned"),
            CarriedAttempt(attempt_id=NEXT, state="planned"),
        ],
    )
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan_over(parent, FIRST_SLOT, GRADED_CELL),
            a_plan_over(parent, SECOND_SLOT, PLACEBO_CELL),
        ],
    )
    # The generation is one a fork may be cut from: the ending is behind it, and nothing about it
    # is live, offered or owed beyond the one payload a child inherits.
    parent._refuse_a_fork(request)
    assert FINISHED not in parent._handed_out()
    assert parent._attempts[FINISHED].state == "final_failed"

    built = parent._built_children(request)
    receipt = parent._the_receipt(request, built)

    bodies = {item.attempt_id: item.body for item in parent._start.tasks}
    assert [row.next_task_body_sha256 for row in receipt.child_receipts] == [
        sha256(bodies[NEXT].encode("utf-8")).hexdigest()
    ] * 2

    # And that is the task each child actually serves, joined to the child's own roster by the
    # identity the receipt names and to its schedule by the attempt that row stands for. Each
    # child offers the inherited payload first, and the task its own gate opens for once that
    # payload has been presented.
    for (row, start), receipt_row in zip(built, receipt.child_receipts):
        [roster_row] = [
            one for one in start.assignments if one.assignment_id == receipt_row.next_assignment_id
        ]
        assert roster_row.attempt_id == NEXT
        assert receipt_row.next_assignment_id != NEXT
        serving_as(monkeypatch, workflow_id=row.child_workflow_id)
        child = kernel_workflow.StreamWorkflow(start)
        assert child._first_eligible() == ("payload", ATTEMPT)
        child._obligations[ATTEMPT].state = "presented"
        assert child._first_eligible() == ("task", roster_row.attempt_id)


def test_an_inherited_ending_stays_its_parents_work_in_a_child_and_across_its_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Whose work a row is is carried, and it is carried for rows that committed no source.

    An attempt that ended before it filed anything has no source, no seal and no descriptor, and
    it is still work the parent did: the ending, the floor and the reason are the parent's own
    outcome, and both children hold that row because both inherit the whole prefix. Reading such
    a row as local work would total one ending once per generation of the lineage.

    So the parent writes whose work each row is when it builds the children, over what the row
    had done rather than over what it committed, and the child carries that forward through its
    own boundary. A row still planned at the cut is nobody's work yet: the child that goes on to
    work it is the generation it belongs to, which is what makes the inherited task the parent's
    and the next task each child's own in the same table.
    """
    parent = a_quiet_parent(
        monkeypatch,
        start=a_parent_holding_a_finished_task(),
        attempts=[
            a_row(),
            CarriedAttempt(attempt_id=FINISHED, state="final_failed", final_failure="abandoned"),
            CarriedAttempt(attempt_id=NEXT, state="planned"),
        ],
    )
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan_over(parent, FIRST_SLOT, GRADED_CELL),
            a_plan_over(parent, SECOND_SLOT, PLACEBO_CELL),
        ],
    )
    parent._refuse_a_fork(request)
    built = parent._built_children(request)

    # The ending is the parent's own, and it has no receipt evidence under it to say so.
    kept = {row.attempt_id: row for row in parent.attempt_records()}
    assert kept[FINISHED].source_generation == PARENT_ID
    assert kept[FINISHED].source_provenance is None
    assert kept[FINISHED].final_failure == "abandoned"

    for row, start in built:
        serving_as(monkeypatch, workflow_id=row.child_workflow_id)
        child = kernel_workflow.StreamWorkflow(start)
        held = {one.attempt_id: one for one in child.generation_records().attempts}
        assert held[ATTEMPT].source_generation == PARENT_ID
        assert held[FINISHED].source_generation == PARENT_ID
        assert held[FINISHED].final_failure == "abandoned"
        # And the task neither generation has worked belongs to whichever works it.
        assert held[NEXT].source_generation == row.child_workflow_id

        root = tmp_path / row.branch_slot
        root.mkdir()
        run = RunRecords(
            root=root,
            workflow_id=row.child_workflow_id,
            records=list(held.values()),
            origin=LineageOrigin(
                parent_workflow_id=PARENT_ID,
                parent_run_id=PARENT_RUN,
                acknowledged_cursor=parent._cursor,
                branch_slot=row.branch_slot,
                fork_id=FORK,
            ),
        )
        assert {one.attempt_id for one in inherited_work(run)} == {ATTEMPT, FINISHED}
        assert {one.attempt_id for one in local_work(run)} == {NEXT}
        exported = [
            json.loads(line)
            for line in write_records(run).read_text(encoding="utf-8").splitlines()
        ]
        assert [one["work_provenance"] for one in exported] == [
            INHERITED_WORK,
            INHERITED_WORK,
            LOCAL_WORK,
        ]
        assert [one["source_generation"] for one in exported] == [
            PARENT_ID,
            PARENT_ID,
            row.child_workflow_id,
        ]

        # And a child that crosses a boundary of its own carries the attribution over it.
        carried = child._projection()
        serving_as(
            monkeypatch,
            workflow_id=row.child_workflow_id,
            continued="the child's earlier execution",
        )
        again = kernel_workflow.StreamWorkflow(
            replace(
                start,
                carry=pack_carrier(
                    carried, CONVERTER, version=carrier_version(start, carried)
                ),
            )
        )
        crossed = {one.attempt_id: one for one in again.attempt_records()}
        assert crossed[ATTEMPT].source_generation == PARENT_ID
        assert crossed[FINISHED].source_generation == PARENT_ID
        assert crossed[NEXT].source_generation == row.child_workflow_id


def test_the_origin_proof_is_bounded_over_every_state_its_parent_can_answer_it_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other reply this fork transmits, measured over what it becomes and not over what it is.

    The row a parent measures before its barrier names no run id and stands in the class a child is
    in before it exists, and the fork it names has not moved through any of its own statuses. Every
    one of those changes in the proof the parent goes on answering with, and none of them is
    measured by a preflight over the row as it stands. So the bound is over the widest legal reply,
    the run id at its declared ceiling among them, and the row that arrives with a minted run id is
    measured against the retained value.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    bounds = parent._measured_replies(request, built)
    assert bounds.origin == max(
        fork_origin_bound(request.fork_id, row, CONVERTER) for row, _start in built
    )
    # The state the preflight would have measured had it measured the row as it stood.
    prestart = encoded_size(
        ForkOriginVerified(fork_id=request.fork_id, fork_status=FORK_PREPARED, record=built[0][0]),
        CONVERTER,
    )
    assert bounds.origin > prestart

    parent._commit_the_barrier(request, built, bounds)
    assert parent._fork is not None
    assert parent._fork.origin_bound == bounds.origin
    parent._note_child(1, "confirmed_existing", run_id="r" * 36)
    answered = parent.fork_child(request.fork_id, 1)
    assert answered.record.child_run_id == "r" * 36
    assert encoded_size(answered, CONVERTER) > prestart
    assert encoded_size(answered, CONVERTER) <= bounds.origin

    # A ceiling that admits the row as it stood and not the reply it becomes refuses the fork
    # before its barrier, because what has to fit is the widest reply and not the narrowest.
    narrow = a_quiet_parent(monkeypatch)
    theirs = narrow._built_children(request)
    monkeypatch.setattr(kernel_workflow, "TURNOVER_PAYLOAD_CEILING_BYTES", prestart)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        narrow._measured_replies(request, theirs)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "what the origin verification answers for child 1 is bounded at" in raised.value.clause
    assert narrow._fork is None
    monkeypatch.undo()


def test_an_origin_proof_over_the_bound_the_barrier_retained_is_refused_where_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run id is the service's, so the reply carrying one is measured against the kept bound."""
    parent, request = a_barrier(monkeypatch)
    assert parent._fork is not None
    assert parent._fork.origin_bound == max(
        fork_origin_bound(request.fork_id, row, CONVERTER)
        for row in parent._fork.child_records
    )
    parent._fork = replace(parent._fork, origin_bound=32)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._note_child(1, "confirmed_existing", run_id="r" * 36)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "the bound proved for it before the barrier was 32" in raised.value.clause
    # Nothing was moved by the refusal: the row is the one the barrier committed.
    assert parent._fork.child_records[0].child_run_id is None


class AConverterHeavyInOneState:
    """A configured converter whose encoded size does not follow the length of its strings.

    Every measurement this build takes is taken through the converter a deployment configured,
    because what the service admits is what that converter produces. Nothing says the mapping from
    a value to its bytes is monotone in the words inside it: a converter may attach whatever it
    likes to whatever it recognizes, and this one recognizes an origin proof in the state a
    barrier commits in.

    It is a delegate rather than a reimplementation, so everything else this fork encodes crosses
    exactly as it would have.
    """

    def __init__(self, padding: int) -> None:
        self.padding = padding

    def to_payloads(self, values: Any) -> Any:
        encoded = CONVERTER.to_payloads(values)
        for value, payload in zip(values, encoded):
            if isinstance(value, ForkOriginVerified) and value.fork_status == FORK_PREPARED:
                payload.metadata["the state this converter is heavy in"] = b"0" * self.padding
        return encoded

    def from_payloads(self, payloads: Any, type_hints: Any = None) -> Any:
        return CONVERTER.from_payloads(payloads, type_hints)


def test_the_origin_bound_is_measured_through_the_converter_that_encodes_the_reply(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The widest reply is the widest as the configured converter encodes it, not as text.

    A bound taken over the longest status and existence words assumes the encoded size of a reply
    follows the length of the strings in it, and a configured converter owes nobody that. So every
    legal reply is encoded and the largest is kept, which is a measurement rather than an
    assumption about one, and the reply the parent actually answers with is held to that value
    where it answers.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    row = built[0][0]
    heavy = AConverterHeavyInOneState(1 << 13)

    # The reply with the longest words in it is not the reply this converter makes the most bytes
    # of, and the one it does is the state a barrier commits in.
    longest = encoded_size(
        ForkOriginVerified(
            fork_id=request.fork_id,
            fork_status=max(FORK_STATUSES, key=len),
            record=replace(
                row,
                existence=max(CHILD_EXISTENCE, key=len),
                child_run_id="0" * RUN_ID_CEILING_BYTES,
            ),
        ),
        heavy,
    )
    prepared = ForkOriginVerified(
        fork_id=request.fork_id,
        fork_status=FORK_PREPARED,
        record=replace(row, existence="confirmed_existing", child_run_id="r" * 36),
    )
    assert encoded_size(prepared, heavy) > longest
    assert fork_origin_bound(request.fork_id, row, heavy) >= encoded_size(prepared, heavy)

    # And through the fork: the bound the barrier retains is the one the reply it goes on
    # answering with fits inside, measured through the converter that will carry it.
    monkeypatch.setattr(kernel_workflow.workflow, "payload_converter", lambda: heavy)
    bounds = parent._measured_replies(request, built)
    parent._commit_the_barrier(request, built, bounds)
    assert parent._fork is not None
    parent._note_child(1, "confirmed_existing", run_id="r" * 36)
    answered = parent.fork_child(request.fork_id, 1)
    assert answered.fork_status == FORK_PREPARED
    assert encoded_size(answered, heavy) > longest
    assert encoded_size(answered, heavy) <= bounds.origin

    # A reply that outgrew the bound its barrier proved says so where it is answered, carrying
    # the measurement, rather than crossing the wire as a silent oversize.
    parent._fork = replace(parent._fork, origin_bound=32)
    with pytest.raises(WireFormatError) as oversize:
        parent.fork_child(request.fork_id, 1)
    assert "the bound proved for it before the barrier was 32" in str(oversize.value)
    monkeypatch.undo()


class AConverterHeavyOnAnAbsentRunId:
    """A configured converter that is heavy exactly where a prepared row starts.

    A record's run id is absent until a start reply is recorded, and the row is answered from all
    the same, so this is the state of a reply rather than a state before them.
    """

    def __init__(self, padding: int) -> None:
        self.padding = padding

    def to_payloads(self, values: Any) -> Any:
        encoded = CONVERTER.to_payloads(values)
        for value, payload in zip(values, encoded):
            if (
                isinstance(value, ForkOriginVerified)
                and value.record.child_run_id is None
            ):
                payload.metadata["the state this converter is heavy in"] = b"0" * self.padding
        return encoded

    def from_payloads(self, payloads: Any, type_hints: Any = None) -> Any:
        return CONVERTER.from_payloads(payloads, type_hints)


def test_the_reply_a_child_that_asks_before_its_start_gets_is_bounded_before_the_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child may ask its origin question before any start reply is recorded.

    The row it is answered with is the row as it stands, which is the class the record was written
    in and no run id at all. Starting a child does not wait for the parent to finish and a child's
    activation does not wait for the fork to complete, so that reply is one this parent gives
    first rather than one it never gives, and a bound that allowed only for an identifier the
    service had already minted would be a bound over the wrong set of replies.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    row = built[0][0]
    heavy = AConverterHeavyOnAnAbsentRunId(1 << 13)

    unrecorded = ForkOriginVerified(
        fork_id=request.fork_id,
        fork_status=FORK_PREPARED,
        record=replace(row, existence="existence_unconfirmed", child_run_id=None),
    )
    assert fork_origin_bound(request.fork_id, row, heavy) >= encoded_size(unrecorded, heavy)

    # And through the fork: the barrier is committed over a bound that covers it, and the parent
    # answers the question that arrives before the start reply from inside that bound.
    monkeypatch.setattr(kernel_workflow.workflow, "payload_converter", lambda: heavy)
    bounds = parent._measured_replies(request, built)
    parent._commit_the_barrier(request, built, bounds)
    assert parent._fork is not None
    parent._note_child(1, "existence_unconfirmed")
    asked_early = parent.fork_child(request.fork_id, 1)
    assert asked_early.record.child_run_id is None
    assert encoded_size(asked_early, heavy) <= bounds.origin
    monkeypatch.undo()


class AConverterHeavyOnAStartResult:
    """A configured converter that is heavy around the result one child's start answers with.

    A wrapper can exceed a limit while everything it wraps fits, so what a preflight of the
    identifier inside it measures is a different shape from the one that crosses.
    """

    def __init__(self, padding: int) -> None:
        self.padding = padding

    def to_payloads(self, values: Any) -> Any:
        encoded = CONVERTER.to_payloads(values)
        for value, payload in zip(values, encoded):
            if isinstance(value, ForkChildStarted):
                payload.metadata["the result this converter is heavy in"] = b"0" * self.padding
        return encoded

    def from_payloads(self, payloads: Any, type_hints: Any = None) -> Any:
        return CONVERTER.from_payloads(payloads, type_hints)


def test_what_a_childs_start_answers_with_is_bounded_before_the_barrier_and_held_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The result a start comes back with is a shape this operation transmits like any other.

    It cannot be measured exactly before the barrier either: the run id comes back inside it, and
    whether this call created the child or found it already created is not known until it answers.
    Both are allowed for before anything is fenced, the bound is retained with the fork, and the
    result that actually arrives is measured against the retained value rather than against one
    derived from itself.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    row = built[0][0]
    heavy = AConverterHeavyOnAStartResult(1 << 13)
    both = [
        encoded_size(
            ForkChildStarted(
                child_ordinal=row.child_ordinal,
                child_workflow_id=row.child_workflow_id,
                child_run_id="r" * 36,
                created=created,
            ),
            heavy,
        )
        for created in (True, False)
    ]
    assert fork_start_bound(row, heavy) >= max(both)

    # A bound that does not fit is a decision about the request, taken before anything is fenced.
    monkeypatch.setattr(kernel_workflow.workflow, "payload_converter", lambda: heavy)
    bounds = parent._measured_replies(request, built)
    assert bounds.start == max(
        fork_start_bound(one, heavy) for one, _start in built
    )
    monkeypatch.setattr(kernel_workflow, "TURNOVER_PAYLOAD_CEILING_BYTES", bounds.start - 1)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._measured_replies(request, built)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "what the start of child 1 answers with is bounded at" in raised.value.clause
    assert parent._fork is None
    monkeypatch.undo()

    # And the result that arrives is held to the value the barrier retained.
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    parent._commit_the_barrier(request, built, parent._measured_replies(request, built))
    assert parent._fork is not None
    assert parent._fork.start_bound == max(
        fork_start_bound(one, CONVERTER) for one, _start in built
    )
    parent._fork = replace(parent._fork, start_bound=32)

    async def answered(*_arguments: Any, **named: Any) -> Any:
        asked = named["arg"] if "arg" in named else _arguments[1]
        return ForkChildStarted(
            child_ordinal=asked.child_ordinal,
            child_workflow_id=asked.child_workflow_id,
            child_run_id="r" * 36,
            created=True,
        )

    monkeypatch.setattr(kernel_workflow.workflow, "execute_activity", answered)
    with pytest.raises(kernel_workflow.ForkRefused) as oversize:
        asyncio.run(parent._start_the_children(request, built))
    assert oversize.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "the bound proved for it before the barrier was 32" in oversize.value.clause
    # Nothing was moved by the refusal: the child stays attempted with its existence unconfirmed
    # rather than confirmed over a reply this parent would not carry.
    assert parent._fork.child_records[0].existence == "existence_unconfirmed"
    assert parent._fork.child_records[0].child_run_id is None
    monkeypatch.undo()


def test_two_child_directories_that_name_one_store_never_reach_a_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each child keeps its objects and its manifest in a place of its own, and this is that check.

    A directory is a string in the plans and a store is a value a child is created with, and the
    derivation from the first to the second is not injective over strings: the store sits under a
    fixed name inside the directory and a trailing separator is dropped on the way. So two
    directories that differ as text can name one store, and every check made of the request as a
    value passes.

    A parent that committed a barrier over that pair would have fenced itself for good to create
    two children promised separate objects, and the second child's attachment would find the
    first's manifest already published where its own belongs.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan(FIRST_SLOT, GRADED_CELL, run_directory="/runs/one/shared"),
            a_plan(SECOND_SLOT, PLACEBO_CELL, run_directory="/runs/one/shared/"),
        ],
    )
    # Nothing about the request as a value is wrong: the two directories are distinct strings, and
    # nothing about the boundary, the witnesses or the declared branches is wrong either.
    assert len({plan.run_directory for plan in request.child_plans}) == 2
    parent._refuse_a_fork(request)

    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    refusal = raised.value
    assert refusal.reason == FORK_CONFIGURATION_VIOLATION
    assert refusal.non_retryable
    assert "the children 1 and 2 were given directories that name one store" in refusal.clause
    assert child_blob_root("/runs/one/shared") in refusal.clause
    assert parent._fork is None
    assert parent._generation_state == "open"
    assert parent._fencing_token_hash is not None
    # And the two directories a controller ordinarily gives still build both starts.
    parent._built_children(a_fork_request(parent))


@pytest.mark.parametrize(
    "alias",
    ["/runs/one/./shared", "/runs/one//shared", "/runs/one/shared/./"],
)
def test_two_child_directories_spelled_apart_and_opened_alike_reach_no_barrier_either(
    monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    """A separator at the end is one spelling of one place and it is not the only one.

    What opens a store takes a path, and a path drops an empty component and a component naming
    the directory it already stands in exactly as it drops a separator at the end. So the pair is
    compared as the place each one is, and every spelling that arrives at one place is refused
    where the trailing separator is, before anything is fenced and before a child exists.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan(FIRST_SLOT, GRADED_CELL, run_directory="/runs/one/shared"),
            a_plan(SECOND_SLOT, PLACEBO_CELL, run_directory=alias),
        ],
    )
    assert len({plan.run_directory for plan in request.child_plans}) == 2
    parent._refuse_a_fork(request)

    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "name one store" in raised.value.clause
    assert parent._fork is None
    assert parent._generation_state == "open"


def test_a_child_directory_that_names_its_parents_own_store_reaches_no_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child is given a store of its own, and one spelling of its parent's is not one.

    The parent's own objects and manifest are already in that place. A child created there would
    keep its bytes where its parent keeps its, and its attachment would meet the manifest its
    parent published where its own belongs, so the comparison is of the place rather than of the
    text and the refusal lands where a plan naming the same directory outright lands.
    """
    parent = a_quiet_parent(
        monkeypatch, start=a_fork_parent(blob_root="/runs/one/blobs")
    )
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan(FIRST_SLOT, GRADED_CELL, run_directory="/runs/one/."),
            a_plan(SECOND_SLOT, PLACEBO_CELL, run_directory="/runs/one/two"),
        ],
    )
    assert parent._start.blob_root not in {
        child_blob_root(plan.run_directory) for plan in request.child_plans
    }
    parent._refuse_a_fork(request)

    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "a child keeps its bytes in a durable blob_root of its own" in raised.value.clause
    assert parent._fork is None
    assert parent._generation_state == "open"

    # A directory naming the place above one of its own components is refused rather than
    # collapsed, because where that leads is not settled without reading a filesystem.
    above = a_fork_request(
        parent,
        child_plans=[
            a_plan(FIRST_SLOT, GRADED_CELL, run_directory="/runs/one/two/../three"),
            a_plan(SECOND_SLOT, PLACEBO_CELL, run_directory="/runs/one/four"),
        ],
    )
    with pytest.raises(kernel_workflow.ForkRefused) as named:
        parent._built_children(above)
    assert named.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "names the directory above" in named.value.clause
    assert parent._fork is None


def test_two_child_directories_one_rooted_and_one_not_reach_no_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A location that says where it starts from is the only one two of them are compared as.

    A location written from wherever a process happens to stand names one place in that process
    and another in the next, and the transition that would have to settle it is a replayed one
    with no directory of its own to read. So the pair below is the sibling alias every other
    spelling in this section is: two texts that arrive at one store the moment anything opens
    them. It is refused where the others are, before the parent is fenced and before a child
    exists, rather than admitted because the two texts are not equal.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan(FIRST_SLOT, GRADED_CELL, run_directory="/runs/one/shared"),
            a_plan(SECOND_SLOT, PLACEBO_CELL, run_directory="runs/one/shared"),
        ],
    )
    # The request is admitted: nothing about the boundary, the witnesses or the declared branches
    # is wrong, and the two directories are distinct strings.
    assert len({plan.run_directory for plan in request.child_plans}) == 2
    parent._refuse_a_fork(request)

    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert raised.value.non_retryable
    assert "runs/one/shared" in raised.value.clause
    assert "says where it starts from" in raised.value.clause
    assert parent._fork is None
    assert parent._generation_state == "open"
    assert parent._fencing_token_hash is not None
    # And the comparison the pair would have reached refuses the same text for the same reason,
    # so neither gate is the only one holding this pair apart.
    with pytest.raises(kernel_workflow.ForkRefused) as compared:
        parent._one_store(child_blob_root("runs/one/shared"))
    assert compared.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "says where it starts from" in compared.value.clause


def test_a_child_directory_that_says_nothing_about_where_it_starts_reaches_no_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same rule against the parent's own store, which is the other place two become one.

    A child given its parent's store under a spelling that starts from wherever the reader stands
    would keep its objects and its manifest where its parent keeps its, and the comparison that
    would catch it has one rooted operand and one that is not. Neither operand is resolved here,
    because resolving one would read the directory the process happens to stand in and this runs
    where a replay has to reach the same answer. So it is refused for what it does not say.
    """
    parent = a_quiet_parent(monkeypatch, start=a_fork_parent(blob_root="/runs/one/blobs"))
    request = a_fork_request(
        parent,
        child_plans=[
            a_plan(FIRST_SLOT, GRADED_CELL, run_directory="runs/one"),
            a_plan(SECOND_SLOT, PLACEBO_CELL, run_directory="/runs/one/two"),
        ],
    )
    # As text the child's store and its parent's are two, which is why the comparison alone admits
    # this pair: the parent's is rooted and the child's is not.
    assert child_blob_root("runs/one") != parent._start.blob_root
    parent._refuse_a_fork(request)

    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._built_children(request)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "says where it starts from" in raised.value.clause
    assert parent._fork is None
    assert parent._generation_state == "open"
    assert parent._fencing_token_hash is not None


class AStatusHandle:
    """A parent handle that answers the status Query with one recorded answer."""

    def __init__(self, answer: Any) -> None:
        self.answer = answer

    async def query(self, *_arguments: Any, **_named: Any) -> Any:
        return self.answer


class AStatusClient:
    """A client whose parent says where its fork stands, so the receiving check can be read."""

    def __init__(self, answer: Any) -> None:
        self.data_converter = default_converter()
        self.answer = answer

    def get_workflow_handle_for(self, *_arguments: Any, **_named: Any) -> AStatusHandle:
        return AStatusHandle(self.answer)


async def test_a_status_answer_at_a_version_this_build_does_not_read_is_refused_where_it_arrives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both declared shapes a status answer carries are admitted where the answer is received.

    The status route is a documented controller operation and not a step inside another one, so a
    caller reading it directly gets the same admission a caller that went on to the stream route
    would. The record and the receipt are two declared types and neither is admitted by the
    other's check: one says which children a fork prepared and where it stands, and the other is
    the fork's own outcome.
    """
    parent, request = a_barrier(monkeypatch)
    assert parent._fork is not None
    standing = parent._fork
    answered = ForkStatusAnswer(
        found=True,
        conflict=False,
        parent_state=parent._generation_state,
        record=standing,
        receipt=a_receipt(),
    )
    assert await kernel_runtime.fork_status(AStatusClient(answered), request) == answered

    with pytest.raises(WireFormatError) as record:
        await kernel_runtime.fork_status(
            AStatusClient(replace(answered, record=replace(standing, schema_version=999))),
            request,
        )
    assert "999" in str(record.value)

    with pytest.raises(WireFormatError) as receipt:
        await kernel_runtime.fork_status(
            AStatusClient(replace(answered, receipt=a_receipt(schema_version=999))), request
        )
    assert "999" in str(receipt.value)


class AnOutcomeHandle:
    """A parent handle asked what one exact identifier was answered with.

    It answers with what it was given or raises it, and it counts the asking, so a read this
    build tried as often as it tries one can be told from a read it tried once.
    """

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.reads = 0

    async def query(self, *_arguments: Any, **_named: Any) -> Any:
        self.reads += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


async def test_an_outcome_that_could_not_be_read_is_not_an_identifier_that_was_never_answered(
) -> None:
    """Nothing recorded and nothing readable are two answers, and only one of them is a receipt.

    The parent saying it holds no outcome under an identifier is what lets a fresh identifier be
    given the fork's own receipt: a completed fork has an answer, and a caller that never reached
    a handler is owed it. A read that could not be served says nothing about that identifier at
    all, and taking it for nothing recorded is how one try's caller ends up holding another try's
    success once a failure has crossed a boundary in the carried journal.

    So a read that failed comes back as the typed refusal every other read of this parent is
    classified under, and an outcome whose bytes this build cannot read comes back the same way:
    what arrived was an answer, and reporting the identifier as unanswered because of it would
    put the caller on the route kept for one the parent never answered.
    """
    identifier = "fork-1-over-a-body-that-was-lost-1"

    absent = AnOutcomeHandle(AnsweredUpdate(found=False))
    assert await kernel_runtime._answered_fork(absent, identifier, FORK) is None
    assert absent.reads == 1

    unreadable = AnOutcomeHandle(
        RPCError("the parent could not be reached", RPCStatusCode.UNAVAILABLE, b"")
    )
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        await kernel_runtime._answered_fork(unreadable, identifier, FORK)
    assert raised.value.reason == FORK_UNREADABLE_PARENT
    assert kernel_runtime.fork_can_be_retried(raised.value) is True
    assert unreadable.reads == 3

    gone = AnOutcomeHandle(
        RPCError("no execution under that identity", RPCStatusCode.NOT_FOUND, b"")
    )
    with pytest.raises(kernel_workflow.ForkRefused) as expired:
        await kernel_runtime._answered_fork(gone, identifier, FORK)
    assert expired.value.reason == FORK_EXPIRED_AUTHORITY
    assert kernel_runtime.fork_can_be_retried(expired.value) is False

    undecodable = AnOutcomeHandle(
        AnsweredUpdate(found=True, kind="value", value={"fork_id": 5})
    )
    with pytest.raises(kernel_workflow.ForkRefused) as unread:
        await kernel_runtime._answered_fork(undecodable, identifier, FORK)
    assert unread.value.reason == FORK_UNREADABLE_PARENT
    assert kernel_runtime.fork_can_be_retried(unread.value) is True

    # And the refusal a parent did record under that identifier is raised as the refusal it
    # recorded, which is the answer the whole distinction exists to keep reachable.
    refused = AnOutcomeHandle(
        AnsweredUpdate(
            found=True,
            kind="failure",
            type_name=FORK_REPAIRABLE_ABSENCE,
            message="the placebo body is not in the store this fork reads",
        )
    )
    with pytest.raises(ApplicationError) as answered_with:
        await kernel_runtime._answered_fork(refused, identifier, FORK)
    assert kernel_runtime.fork_refusal(answered_with.value) == FORK_REPAIRABLE_ABSENCE


async def test_an_origin_window_that_has_closed_is_recorded_rather_than_asked_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A closed answer window is its own state, and the child keeps everything but the gate.

    An unreachable parent is infrastructure and the reading is tried again while the child waits.
    A window that has closed is not that: the horizon a parent's answers stand behind is absolute,
    so no later reading reaches an answer and asking again for ever would spend a Worker on a
    question with no answer left. So the child records the state, keeps its identity and its
    objects, owns and serves nothing, and answers every call that would own or serve it with the
    reason rather than with the wait.
    """
    serving_as(monkeypatch, workflow_id=a_child_id(1))
    child = a_fresh_child()
    gated = kernel_workflow.StreamWorkflow(child)
    asked: List[Any] = []

    async def closed(*arguments: Any, **named: Any) -> Any:
        asked.append(named.get("activity_id"))
        raise ActivityError(
            "the origin reading",
            scheduled_event_id=1,
            started_event_id=2,
            identity="the worker that asked",
            activity_type="shogym.protocol_v2.VerifyForkOriginActivity",
            activity_id="the reading",
            retry_state=None,
        ) from ApplicationError(
            "the execution that prepared this child can no longer be read, so its answer window "
            "has closed",
            type="ExpiredAuthority",
            non_retryable=True,
        )

    monkeypatch.setattr(kernel_workflow.workflow, "execute_activity", closed)
    await gated._verify_the_lineage()
    assert len(asked) == 1
    assert gated._origin_expired is True
    # The gate is still shut and the child is still the child it was: nothing about it moved.
    assert gated._origin_unverified is True
    assert gated._start.fork_origin is not None

    writer = Writer(ownership_epoch=0, fencing_token="c" * 64)
    for refused_call in (
        lambda: gated._claim_ownership_admitted(a_claim(child)),
        lambda: gated._check_claim(a_claim(child)),
        lambda: gated._require_writer(writer),
    ):
        with pytest.raises(kernel_workflow.ForkRefused) as raised:
            refused_call()
        assert raised.value.reason == FORK_EXPIRED_AUTHORITY
        assert kernel_runtime.fork_can_be_retried(raised.value) is False
        assert a_child_id(1) in raised.value.clause

    # An unreachable parent is the other case and is untouched: it is raised, and the retry
    # policy the reading is made under is what asks again.
    other = kernel_workflow.StreamWorkflow(child)

    async def unreachable(*_arguments: Any, **_named: Any) -> Any:
        raise ActivityError(
            "the origin reading",
            scheduled_event_id=1,
            started_event_id=2,
            identity="the worker that asked",
            activity_type="shogym.protocol_v2.VerifyForkOriginActivity",
            activity_id="the reading",
            retry_state=None,
        ) from ApplicationError("the parent could not be reached", type="RPCError")

    monkeypatch.setattr(kernel_workflow.workflow, "execute_activity", unreachable)
    with pytest.raises(ActivityError):
        await other._verify_the_lineage()
    assert other._origin_expired is False
    with pytest.raises(ApplicationError) as still:
        other._check_claim(a_claim(child))
    assert still.value.type == ORIGIN_UNVERIFIED
    monkeypatch.undo()


def test_a_receipt_over_the_bound_the_barrier_retained_is_refused_rather_than_answered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The record keeps the bound, and the reply that arrives is held to the kept value."""
    parent, request = a_barrier(monkeypatch)
    assert parent._fork is not None
    built = parent._built_children(request)
    assert parent._fork.receipt_bound == fork_receipt_bound(
        parent._the_receipt(request, built), CONVERTER
    )
    parent._fork = replace(parent._fork, receipt_bound=32)
    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        parent._fork_answer(request, built)
    assert raised.value.reason == FORK_CONFIGURATION_VIOLATION
    assert "the bound proved for it before the barrier was 32" in raised.value.clause
    assert parent._fork_receipt is None


def a_barrier(monkeypatch: pytest.MonkeyPatch) -> Any:
    """One barriered generation and the exact request its barrier was committed under.

    The request is returned rather than rebuilt, because the barrier moves the generation into its
    terminal forked state and the projection digest a rebuilt request would carry is the one it
    stands at now rather than the one it was cut at.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    parent._commit_the_barrier(request, built, parent._measured_replies(request, built))
    return parent, request


def a_barriered_parent(monkeypatch: pytest.MonkeyPatch) -> Any:
    """One generation with a fork barrier standing, which is where its reserve starts counting."""
    return a_barrier(monkeypatch)[0]


def test_a_barrier_fences_the_parent_and_records_the_horizon_its_answers_stand_until(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One transition: the record, the terminal state, the fence and the two clocks.

    The horizon is recorded at the commit because the close cannot be bounded and the deletion
    clock is not the close clock either: the service schedules each execution's history for
    deletion from that execution's own close time, so a child that failed an hour after the
    barrier is deleted on its own clock while the parent is still prepared. The deadline beside it
    bounds the parent's own work and never says when it actually closes.
    """
    parent = a_barriered_parent(monkeypatch)
    assert parent._fork is not None
    assert parent._fork.status == "prepared"
    assert parent._fork.children == 2
    assert [row.existence for row in parent._fork.child_records] == [
        "never_attempted",
        "never_attempted",
    ]
    assert parent._generation_state == "forked"
    assert parent._fencing_token_hash is None
    with pytest.raises(ApplicationError) as raised:
        parent._require_writer(Writer(ownership_epoch=1, fencing_token="a" * 64))
    assert raised.value.message == "fenced_writer"

    at = int(A_MOMENT.timestamp() * 1000)
    assert parent._fork_deadline_at == at + FORK_PREPARATION_BOUND_MS
    assert parent._fork_horizon_at == at + FORK_AUTHORITY_HORIZON_MS
    assert FORK_AUTHORITY_HORIZON_MS == FORK_PREPARATION_BOUND_MS + FORK_ANSWER_WINDOW_MS
    # A boundary is out of reach of a generation that has taken one, so nothing latches after it.
    assert parent._boundary_clause() == "generation_state"


def test_a_parked_parent_bounds_what_it_accepts_and_keeps_the_reserve_for_the_fork(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reserve, charged where the service charges and read where admission is decided.

    Everything a controller wants to read is a Query, which reaches none of this and charges
    nothing, so a permanently unavailable child service cannot turn polling into exhaustion. The
    only class that reaches the counter at all is the fork's own, which is the work the reserve is
    kept for: a fifth logical retry of one fork is admitted like the fourth, and what ends the
    reserve is the twenty fourth accepted call rather than the class it belonged to.
    """
    parent = a_barriered_parent(monkeypatch)
    with pytest.raises(kernel_workflow.ForkBarrier):
        parent._admit(True)
    for spent in range(FORK_RESERVE):
        parent._admit(True, completion=True)
        parent._count_update()
        assert parent._post_barrier == spent + 1
    with pytest.raises(kernel_workflow.ForkBarrier) as raised:
        parent._admit(True, completion=True)
    assert "spent its recovery reserve" in str(raised.value)

    # And once the parent has stopped admitting fork work, expiry and exhaustion are one rule.
    stopped = a_barriered_parent(monkeypatch)
    stopped._fork_admits_nothing = True
    with pytest.raises(kernel_workflow.ForkBarrier) as raised:
        stopped._admit(True, completion=True)
    assert "admits no further work" in str(raised.value)


def parked(
    monkeypatch: pytest.MonkeyPatch, parent: Any, *, unfinished: int
) -> List[Tuple[str, bool]]:
    """Run this parent's own parking loop here, and return what it stood at while it drained.

    The two exits the loop can take need no service at all: the deadline is compared against the
    generation's own clock and the reserve is a counter it charges itself. What a Worker supplies
    is the settling of the handler ledger, and that is what the list returned here reads: the
    fork's status and whether it was still admitting fork work, recorded once for every time the
    parent asked whether the ledger had settled. An ending committed before the last outcome
    settled would show up here as a status no controller could still have been answered under,
    and a gate still open while it drained would show up as a parent taking on work it has
    already decided it will never finish.
    """
    seen: List[Tuple[str, bool]] = []
    settling = unfinished

    def all_handlers_finished() -> bool:
        nonlocal settling
        assert parent._fork is not None
        seen.append((parent._fork.status, parent._fork_admits_nothing))
        if settling:
            settling -= 1
            return False
        return True

    async def wait_condition(condition: Any, timeout: Any = None) -> None:
        while not condition():
            continue

    monkeypatch.setattr(
        kernel_workflow.workflow, "all_handlers_finished", all_handlers_finished
    )
    monkeypatch.setattr(kernel_workflow.workflow, "wait_condition", wait_condition)
    asyncio.run(parent._park_for_the_fork())
    return seen


def test_a_parent_ends_a_fork_its_reserve_or_its_deadline_ran_out_on_and_admits_nothing_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exhaustion and expiry are one ending, committed by the parent rather than waited out.

    A fork that failed part way leaves the parent parked and the children it made intact, and the
    supported next move is the same fork under the same checkpoint until the reserve is spent or
    the preparation bound expires. After that the abandoned status stands: it carries which of the
    two ended it, it is terminal, and the parent admits no further fork work and schedules no
    further start, so no Update can answer with a receipt naming a child nobody created.

    The ending waits for the ledger. An abandonment committed while a start Activity was still
    running would record an outcome the parent has not seen, so the status is still the one a
    controller could be answered under every time the ledger is asked, and moves only after.
    """
    for reason, run_it_out in (
        (
            EXPIRED_PREPARATION,
            lambda parent: setattr(parent, "_fork_deadline_at", parent._now_ms()),
        ),
        (
            SPENT_RECOVERY_RESERVE,
            lambda parent: setattr(parent, "_post_barrier", FORK_RESERVE),
        ),
    ):
        # Built the way the handler builds, so the starts the record was committed over are the
        # ones the ending is then asked to schedule.
        parent = a_quiet_parent(monkeypatch)
        request = a_fork_request(parent)
        built = parent._built_children(request)
        parent._commit_the_barrier(
            request, built, parent._measured_replies(request, built)
        )
        run_it_out(parent)
        assert parent._fork is not None and parent._fork.status == FORK_PREPARED
        assert parent._fork.abandoned_reason is None

        seen = parked(monkeypatch, parent, unfinished=2)
        assert seen == [(FORK_PREPARED, True)] * 3, reason
        assert parent._fork.status == FORK_ABANDONED
        assert parent._fork.abandoned_reason == reason
        # Both children stay exactly as the record left them: an abandonment is an ending, not a
        # claim that they never existed.
        assert [row.existence for row in parent._fork.child_records] == [
            "never_attempted",
            "never_attempted",
        ]

        # And the gate the ending installs, read where the next call would take effect.
        assert parent._fork_admits_nothing
        with pytest.raises(kernel_workflow.ForkBarrier) as refused:
            parent._admit(True, completion=True)
        assert "admits no further work" in str(refused.value)
        with pytest.raises(kernel_workflow.ForkBarrier) as unstarted:
            asyncio.run(parent._start_the_children(request, built))
        assert "child 1 is unstarted" in str(unstarted.value)
        assert [row.existence for row in parent._fork.child_records] == [
            "never_attempted",
            "never_attempted",
        ]
        # The status is terminal, so parking again is a parent that has already finished.
        assert parked(monkeypatch, parent, unfinished=0) == []


def a_start_that_disagrees(calls: List[str]) -> Any:
    """A start Activity that finds another execution under this child's derived identity."""

    async def started(*_arguments: Any, **named: Any) -> Any:
        calls.append(named.get("activity_id", ""))
        raise ActivityError(
            "the start of a child",
            scheduled_event_id=1,
            started_event_id=2,
            identity="the worker that asked",
            activity_type="shogym.protocol_v2.StartForkChildActivity",
            activity_id="the start",
            retry_state=None,
        ) from ApplicationError(
            "'stream/the-parent/1.fork.1.abcd' was created as 'SomethingElseEntirely'",
            type=ORIGIN_DISAGREEMENT_FAILURE,
            non_retryable=True,
        )

    return started


def answering_as(
    monkeypatch: pytest.MonkeyPatch, parent: Any, request: ForkRequest, update_id: str
) -> Any:
    """Run one fork under one exact Update identifier, the way an accepted handler runs."""
    monkeypatch.setattr(
        kernel_workflow.workflow,
        "current_update_info",
        lambda: SimpleNamespace(id=update_id),
    )
    return asyncio.run(
        parent._answering(
            kernel_workflow.FORK,
            parent._ownership_epoch,
            lambda: parent._fork_the_generation(request),
        )
    )


def test_a_child_identity_another_execution_holds_ends_this_fork_with_its_own_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An authenticated disagreement is this fork's ending and keeps its reason everywhere.

    Something else running under a child's derived identity is read once and never differently:
    no later reading reaches another answer, and replacing what is there is the one move a fork
    never makes. So it is the fork's own permanent refusal rather than an Activity's failure, and
    it is that before the outcome is journalled, because the journal keeps a type and a message
    and no details and a controller reads the reason out of them.

    What the ending is worth is that it is not repeated. A fresh identifier carrying the same fork
    is answered from the record without scheduling the start again, the status a controller reads
    says the same thing, and the parent stops waiting for a fork that has finished.
    """
    parent, request = a_barrier(monkeypatch)
    calls: List[str] = []
    monkeypatch.setattr(
        kernel_workflow.workflow, "execute_activity", a_start_that_disagrees(calls)
    )

    with pytest.raises(kernel_workflow.ForkRefused) as raised:
        answering_as(monkeypatch, parent, request, "fork-1")

    assert raised.value.reason == FORK_ORIGIN_DISAGREEMENT
    assert raised.value.non_retryable
    assert kernel_runtime.fork_refusal(raised.value) == FORK_ORIGIN_DISAGREEMENT
    assert not kernel_runtime.fork_can_be_retried(raised.value)
    assert "the identity of child 1 is held by another execution" in raised.value.clause
    assert len(calls) == 1

    # The ending is retained with the fork, and every child it made is left exactly as it is.
    assert parent._fork is not None
    assert parent._fork.status == FORK_CONFLICTED
    assert parent._fork.conflict_reason == FORK_ORIGIN_DISAGREEMENT
    assert "another execution" in parent._fork.conflict_clause
    check_prepared_fork(parent._fork)
    assert [row.existence for row in parent._fork.child_records] == [
        "existence_unconfirmed",
        "never_attempted",
    ]

    # The exact identifier keeps that decision, with its reason, out of the application's journal.
    journalled = parent.answered_update("fork-1")
    assert journalled.found and journalled.kind == "failure"
    assert journalled.type_name == FORK_ORIGIN_DISAGREEMENT
    assert "another execution" in journalled.message

    # A fresh logical retry is answered from the record rather than by asking again.
    with pytest.raises(kernel_workflow.ForkRefused) as again:
        answering_as(monkeypatch, parent, request, "fork-2")
    assert again.value.reason == FORK_ORIGIN_DISAGREEMENT
    assert "another execution" in again.value.clause
    assert len(calls) == 1

    # And so is a controller that reads the status of a parent it can no longer write to.
    answer = parent.fork_status(
        ForkStatusQuestion(fork_id=FORK, request_digest=fork_request_digest(request))
    )
    assert answer.found and not answer.conflict
    assert answer.record is not None and answer.record.status == FORK_CONFLICTED
    told = kernel_runtime._what_the_record_says(answer, request)
    assert isinstance(told, kernel_workflow.ForkRefused)
    assert told.reason == FORK_ORIGIN_DISAGREEMENT
    assert "another execution" in told.clause

    # The fork has ended, so the parent waits for nothing further and never rewrites the ending.
    assert parked(monkeypatch, parent, unfinished=0) == []
    parent._move_the_fork(FORK_CHILDREN_CONFIRMED)
    assert parent._fork.status == FORK_CONFLICTED


def test_a_start_that_could_not_be_made_stays_the_failure_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A start nobody could make is infrastructure, and infrastructure is not an ending.

    The fork stays prepared, the child stays attempted with its existence unconfirmed, and the
    same fork under a fresh identifier is the supported next move: turning that into a decision
    would record an ending over work that never happened.
    """
    parent, request = a_barrier(monkeypatch)

    async def unreachable(*_arguments: Any, **_named: Any) -> Any:
        raise ActivityError(
            "the start of a child",
            scheduled_event_id=1,
            started_event_id=2,
            identity="the worker that asked",
            activity_type="shogym.protocol_v2.StartForkChildActivity",
            activity_id="the start",
            retry_state=None,
        ) from ApplicationError("the service could not be reached", type="RPCError")

    monkeypatch.setattr(kernel_workflow.workflow, "execute_activity", unreachable)
    with pytest.raises(ActivityError):
        answering_as(monkeypatch, parent, request, "fork-1")

    assert parent._fork is not None
    assert parent._fork.status == FORK_PREPARED
    assert parent._fork.conflict_reason is None
    assert parent._fork.child_records[0].existence == "existence_unconfirmed"
    parent._refuse_a_fork(request, excluding=("fork-1",))


def test_a_parents_record_answers_a_child_that_asks_and_a_controller_that_asks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The two Queries a closed execution still answers, and the answer neither of them refuses.

    A refused Query would be indistinguishable from a parent nobody could read, and the two are
    different facts: one is authenticated and permanent, the other is infrastructure a child waits
    out while it stays gated. So a question this parent has no answer to comes back as a row
    naming no child, and the caller compares.

    Neither of them charges the reserve. Everything a controller wants to read is one of these
    two, so a permanently unavailable child service cannot turn polling into exhaustion of the
    slots the fork's own completion is kept in.
    """
    parent, request = a_barrier(monkeypatch)
    digest = fork_request_digest(request)

    answer = parent.fork_status(ForkStatusQuestion(fork_id=FORK, request_digest=digest))
    assert answer.found and not answer.conflict
    assert answer.parent_state == "forked"
    assert answer.record is not None and answer.record.fork_id == FORK
    changed = parent.fork_status(
        ForkStatusQuestion(fork_id=FORK, request_digest="0" * 64)
    )
    assert changed.found and changed.conflict
    unknown = parent.fork_status(
        ForkStatusQuestion(fork_id="fork-2", request_digest=digest)
    )
    assert not unknown.found and not unknown.conflict

    told = parent.fork_child(FORK, 1)
    assert told.fork_id == FORK
    assert told.fork_status == "prepared"
    assert told.record.child_ordinal == 1
    assert told.record.child_workflow_id == child_workflow_id(
        identity_namespace="default",
        parent_workflow_id=PARENT_ID,
        fork_id=FORK,
        child_ordinal=1,
    )
    for question in ((FORK, 3), ("fork-2", 1)):
        nothing = parent.fork_child(*question)
        assert nothing.record.child_workflow_id == ""
        assert nothing.record.complete_start_digest == ""
    # Seven reads of a barriered parent, and the reserve is where the barrier left it.
    assert parent._post_barrier == 0

    # A fork that has finished answers with what it finished with, because a retry under a fresh
    # identifier cannot reach the outcome journal the original Update's answer is kept in and a
    # closed parent accepts no Update at all. Until it finishes there is nothing to hand over, and
    # a request bound to something else is never handed this one's evidence.
    assert answer.receipt is None
    built = parent._built_children(request)
    for row, _start in built:
        parent._note_child(row.child_ordinal, "confirmed_existing", run_id="the-child-run")
    receipt = parent._fork_answer(request, built)
    finished = parent.fork_status(ForkStatusQuestion(fork_id=FORK, request_digest=digest))
    assert finished.receipt == receipt
    assert finished.record is not None and finished.record.status == "complete"
    other = parent.fork_status(ForkStatusQuestion(fork_id=FORK, request_digest="0" * 64))
    assert other.conflict and other.receipt is None


def test_a_child_this_parent_prepared_authorizes_its_own_complete_start_against_that_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the parent builds is what the child's own restore and gate are held to.

    The row maps the exact child workflow id and the child ordinal to that child's complete start
    digest, carrier included, and the child compares its actual service identity and its own start
    against it. This is the parent's half of that comparison, built here and read there.
    """
    parent = a_quiet_parent(monkeypatch)
    request = a_fork_request(parent)
    built = parent._built_children(request)
    assert len(built) == 2
    for (row, start), plan in zip(built, request.child_plans):
        assert start.fork_origin is not None
        assert start.served_slot == plan.branch_slot
        # The store is the run directory's own, under the fixed name every generation's reader
        # resolves through, so an object installed for this child is one the reopened directory's
        # own accessor reads rather than one it reads past.
        assert start.blob_root == child_blob_root(plan.run_directory)
        assert start.blob_root == f"{plan.run_directory}/blobs"
        assert start.hidden_execution_id == plan.hidden_execution_id
        assert start.consumer_claim_hash == plan.consumer_claim_hash
        assert row.complete_start_digest == complete_start_digest(start)
        assert row.origin_digest == origin_digest(start.fork_origin)
        assert row.branch_slot == plan.branch_slot
        assert row.target_cell == plan.target_cell
        serving_as(monkeypatch, workflow_id=row.child_workflow_id)
        kernel_workflow.check_complete_start_authorization(
            start, workflow_id=row.child_workflow_id, record=row
        )
    # Both children carry their parent's whole observed prefix and differ where the plans say.
    [first, second] = [start for _, start in built]
    assert first.tasks == second.tasks == parent._start.tasks
    assert first.carry is not None and second.carry is not None
    assert first.carry.carrier_schema_version == FORK_CARRIER_SCHEMA_VERSION
    assert configuration_hash(first) != configuration_hash(second)
    # And the build is pure, so a recovery run of the same fork derives the same two starts.
    again = a_quiet_parent(monkeypatch)._built_children(a_fork_request(parent))
    assert [row for row, _ in again] == [row for row, _ in built]
    assert [start for _, start in again] == [start for _, start in built]
