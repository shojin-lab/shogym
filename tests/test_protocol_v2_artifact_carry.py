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
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

from temporalio.client import WorkflowHistory  # noqa: E402
from temporalio.converter import default as default_converter  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    IMMEDIATE,
    RELEASE_AT_SEAL,
    TASK_FIRST,
    Payload,
    PresentationAck,
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
    FORK_CARRIER_SCHEMA_VERSION,
    LEGACY_CARRIER_SCHEMA_VERSION,
    RECEIPT_CARRIER_SCHEMA_VERSION,
    CarriedProjection,
    ChildSelection,
    ForkOrigin,
    OfferedMessage,
    PayloadCandidate,
    PresentedMessage,
    SourceOriginContext,
    StartDifference,
    StreamStart,
    TaskItem,
    TerminalTool,
    carrier_version,
    configuration_hash,
    continuation_argument,
    derived_selection,
    fork_configuration,
    hidden_seal_id,
    stream_replayer,
    transformed_child_selection,
)
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    CarriedAttempt,
    CarriedAttestation,
    CarriedBinding,
    CarriedObligation,
    legacy_projection_members,
    legacy_start_members,
    origin_fields,
    pack_carrier,
    receipt_projection_members,
    receipt_start_members,
    resolved_echo,
    unpack_carrier,
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
    POLICIES,
    REGISTERED,
    SINGLETON_SLOT,
    WITHHOLD,
    PayloadDisposition,
    PolicyProvenance,
    render_body,
    roster_digest,
)

ATTEMPT = "b" * 32
SILENT = "c" * 32
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
