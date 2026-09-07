"""The source evidence crossing a boundary: what is written, what is read, what is refused.

A generation that captured a source has to hand it on. The descriptor, its commitment, the
identity it was sealed under and the selection made from it become attempt state at the seal,
cross in the carrier, and are checked again on the far side against the start that arrived with
them. That checking is what these tests are about, and it is pure: nothing here opens a store,
reaches a world or runs an Activity, because a restore may do none of those either.

Two numbers name two dispositions. A generation declaring a receipt contract writes the later
carrier from its first boundary, before it has captured anything; one declaring none of it writes
the earlier one, and writes exactly the bytes the code before this one wrote. That second promise
is held by a recorded history rather than by a round trip: a build agreeing with itself is not
evidence that it agrees with the build it replaced.
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
    LEGACY_CARRIER_SCHEMA_VERSION,
    RECEIPT_CARRIER_SCHEMA_VERSION,
    CarriedProjection,
    OfferedMessage,
    PayloadCandidate,
    PresentedMessage,
    SourceOriginContext,
    StreamStart,
    TaskItem,
    TerminalTool,
    carrier_version,
    configuration_hash,
    continuation_argument,
    derived_selection,
    hidden_seal_id,
    stream_replayer,
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
    resolved_echo,
    unpack_carrier,
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


def test_a_receipt_generation_hands_its_own_start_on_with_the_contract_and_the_source_in_it(
) -> None:
    """The current version changes nothing about the start: it is handed on as it is."""
    start = a_start()
    assert (
        continuation_argument(start, CONVERTER, RECEIPT_CARRIER_SCHEMA_VERSION) is start
    )
    written = json.loads(
        CONVERTER.to_payloads(
            [continuation_argument(start, CONVERTER, RECEIPT_CARRIER_SCHEMA_VERSION)]
        )[0].data.decode("utf-8")
    )
    assert written["receipt_source"] == SOURCE
    assert written["receipt_contracts"][0]["contract_id"] == CONTRACT


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
    """The origin is a field rather than an assumption, and this build admits one value for it.

    A continuation hands the same start on, so an ordinary one carries this generation's own
    identity. Reading the seal id back out of the carried origin rather than out of the start is
    what lets a later build admit an inherited source without the check quietly rebinding to
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
    selected against the parent's committed source and unbuilt until the child builds it. No start
    this build admits declares a fork origin, so nothing here produces that state and it is
    refused rather than admitted on the strength of a state that cannot have been reached.
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
