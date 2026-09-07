"""The rows an operation that could not get its evidence leaves behind, and what joins them.

A refusal and the recovery that answers it are two rows about one operation, and what makes them
one operation is an identity built from the operation itself rather than from the transport that
carried it. That is the whole subject here: how those identities are built, what a row holds, and
that the rows cross a boundary in commit order and are refused whole when they are written in a
vocabulary this code does not declare.

Everything in this file is pure. No store, no world, no Activity and no loop: a restore may open
none of those, and the rows are read out of a projection rather than resolved from anything.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from typing import Any, List, Optional

import pytest

pytest.importorskip("temporalio")

from temporalio.converter import default as default_converter  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.serve.protocol_v2 import IMMEDIATE, length_prefixed  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    LEGACY_CARRIER_SCHEMA_VERSION,
    OPERATION_OUTCOMES,
    OPERATION_PHASES,
    OPERATION_REASONS,
    RECEIPT_CARRIER_SCHEMA_VERSION,
    CarriedProjection,
    GenerationRecords,
    OperationFailure,
    StreamStart,
    TaskItem,
    TerminalTool,
    carrier_version,
    configuration_hash,
    fork_preparation_operation_identity,
    ownership_claim_operation_identity,
)
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    CarriedAttempt,
    legacy_projection_members,
    pack_carrier,
    unpack_carrier,
)

ATTEMPT = "b" * 32
CONVERTER = default_converter().payload_converter
GENERATION = "stream/a-generation/1"


def oid(value: int) -> str:
    return f"{value:032x}"


def a_start() -> StreamStart:
    """One generation with a single position, declaring no receipt contract and no source.

    The rows are not receipt evidence about an attempt: they say what an operation could not do,
    and a generation that declares no contract can still have a claim refused over an object its
    history cites. So the fixture is the plainest start there is.
    """
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash="d" * 64,
        initial_cursor=oid(1),
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id="execution-1",
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
    )


def a_failure(**changes: Any) -> OperationFailure:
    """One refused delivery, which is the row every other one here is a variation of."""
    declared = OperationFailure(
        operation="f" * 64,
        phase="continued_first_delivery",
        reason="unavailable_evidence",
        outcome="refused",
        generation=GENERATION,
        refused_epoch=2,
        attempt_id=ATTEMPT,
        payload_position=0,
        references=["a" * 64],
        request_id=oid(0x1001),
    )
    return replace(declared, **changes) if changes else declared


def a_projection(
    start: StreamStart, failures: Optional[List[OperationFailure]] = None
) -> CarriedProjection:
    """One syntactically complete projection, holding the rows a test is about."""
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
        offer_count=1,
        eligibility_count=1,
        handed_out_attempt_ids=[ATTEMPT],
        activity_ordinal=5,
        verification_batches=1,
        attempts=[CarriedAttempt(attempt_id=ATTEMPT, state="sealed")],
        obligations=[],
        presented=[],
        committed_blobs=[],
        pull_requests=[],
        info_requests=[],
        terminal_requests=[],
        finalize_requests=[],
        attestations=[],
        operation_failures=list(failures or []),
    )


def test_a_claim_is_one_operation_however_many_tokens_it_takes_to_make() -> None:
    """The claim operation identity, and the two things deliberately outside it.

    A retry under a fresh fencing token is the same operation as the claim it repeats, so the
    token and its hash are not in the preimage: what is, is the claimant and the epoch the claim
    witnessed, length prefixed in that order under a tag of their own.
    """
    identity = ownership_claim_operation_identity("a-replacement", 3)
    assert identity == sha256(
        length_prefixed(b"ownership-claim-operation-v2")
        + length_prefixed(b"a-replacement")
        + length_prefixed(b"3")
    ).hexdigest()

    # The same claimant witnessing the same epoch is the same operation, however many times it
    # is attempted and whatever token each attempt mints.
    assert ownership_claim_operation_identity("a-replacement", 3) == identity
    # The witnessed epoch is inside it and the won one is not, so the next swap's attempts are a
    # different operation from this one's.
    assert ownership_claim_operation_identity("a-replacement", 4) != identity
    assert ownership_claim_operation_identity("another-owner", 3) != identity


def test_a_fork_preparation_is_one_operation_across_a_repair_that_stepped_the_epoch() -> None:
    """The preparation identity, built the claim identity's way over the creation epoch.

    The epoch inside it is the one the child's first preparation was created under, so a repair
    that stepped the epoch repeats one logical preparation rather than opening a second operation
    nothing joins to the first.
    """
    identity = fork_preparation_operation_identity("fork-1", "stream/child/1", 2)
    assert identity == sha256(
        length_prefixed(b"fork-preparation-operation-v1")
        + length_prefixed(b"fork-1")
        + length_prefixed(b"stream/child/1")
        + length_prefixed(b"2")
    ).hexdigest()
    assert fork_preparation_operation_identity("fork-1", "stream/child/1", 3) != identity
    assert fork_preparation_operation_identity("fork-2", "stream/child/1", 2) != identity
    assert fork_preparation_operation_identity("fork-1", "stream/other/1", 2) != identity
    # And it is a different value from a claim's, because the tags are different preimages.
    assert identity != ownership_claim_operation_identity("fork-1", 2)


def test_a_row_says_which_operation_failed_where_and_what_became_of_it() -> None:
    """The closed vocabularies, and the fields a claim writes absent rather than inventing."""
    assert OPERATION_PHASES == (
        "source_publication",
        "ownership_claim",
        "continued_first_delivery",
        "payload_offer",
        "fork_preparation",
    )
    assert OPERATION_REASONS == (
        "unavailable_evidence",
        "corrupt_evidence",
        "contract_drift",
        "wrong_source",
    )
    assert OPERATION_OUTCOMES == ("refused", "recovered", "unrecoverable")

    claim = a_failure(
        phase="ownership_claim",
        attempt_id=None,
        payload_position=None,
        request_id=None,
        operation=ownership_claim_operation_identity("a-replacement", 1),
        refused_epoch=1,
    )
    assert claim.request_id is None and claim.payload_position is None
    assert claim.attempt_id is None
    assert claim.recovered_epoch is None
    assert claim.generation == GENERATION


def test_the_rows_cross_a_boundary_in_commit_order_and_are_read_back_as_they_were() -> None:
    """They are durable projection state, and the order is what the list says.

    Successive refusals and recoveries of one operation each append, so a sequence that came back
    sorted or deduplicated would have lost which refusal a recovery answered.
    """
    start = a_start()
    rows = [
        a_failure(refused_epoch=2),
        a_failure(refused_epoch=2, request_id=oid(0x1002)),
        a_failure(outcome="recovered", refused_epoch=2, recovered_epoch=3),
    ]
    projection = a_projection(start, rows)
    carried = pack_carrier(
        projection, CONVERTER, version=carrier_version(start, projection)
    )
    assert carried.carrier_schema_version == RECEIPT_CARRIER_SCHEMA_VERSION
    assert unpack_carrier(carried, CONVERTER).operation_failures == rows


def test_a_generation_that_has_recorded_one_writes_the_later_carrier() -> None:
    """Evidence decides the number where configuration did not.

    A generation declaring no contract of its own can still have a claim refused over an object
    its history cites, and a record with no room for that row is not a record it may be written
    into.
    """
    start = a_start()
    assert carrier_version(start, a_projection(start)) == LEGACY_CARRIER_SCHEMA_VERSION
    assert (
        carrier_version(start, a_projection(start, [a_failure()]))
        == RECEIPT_CARRIER_SCHEMA_VERSION
    )


def test_the_legacy_carrier_has_no_name_for_a_row_and_drops_the_whole_list() -> None:
    """The earlier number is an exact serialization adapter, and this is one more member of it."""
    written = CONVERTER.to_payloads([a_projection(a_start(), [a_failure()])])[0].data
    document = json.loads(written.decode("utf-8"))
    assert document["operation_failures"]
    assert "operation_failures" not in legacy_projection_members(document)


@pytest.mark.parametrize(
    "written",
    [
        {"phase": "a_phase_nobody_declared"},
        {"reason": "the_evidence_was_odd"},
        {"outcome": "half_recovered"},
        {"operation": ""},
        {"generation": ""},
        {"attempt_id": "e" * 32},
    ],
)
def test_a_carried_row_written_in_words_this_code_does_not_declare_is_refused(
    written: dict,
) -> None:
    """A row is read by a controller rather than acted on here, which is why it is checked.

    A phase, a reason or an outcome outside the closed sets is a record nobody can classify, and
    an attempt named by a row that is not this generation's is a row about another run. Each is
    refused whole at restore rather than repaired or skipped.
    """
    start = a_start()
    projection = a_projection(start, [a_failure(**written)])
    with pytest.raises(ApplicationError) as raised:
        kernel_workflow._check_carried_receipts(start, projection)
    assert raised.value.type == "CarrierRefused"


def test_a_projection_of_well_formed_rows_restores() -> None:
    """The same check passes what it is meant to pass, one row of each phase and outcome."""
    start = a_start()
    rows = [
        a_failure(phase=phase, reason=reason, outcome=outcome, attempt_id=attempt)
        for phase, reason, outcome, attempt in [
            ("source_publication", "wrong_source", "unrecoverable", ATTEMPT),
            ("ownership_claim", "unavailable_evidence", "refused", None),
            ("continued_first_delivery", "unavailable_evidence", "recovered", ATTEMPT),
            ("payload_offer", "corrupt_evidence", "unrecoverable", ATTEMPT),
            ("fork_preparation", "contract_drift", "refused", ATTEMPT),
        ]
    ]
    kernel_workflow._check_carried_receipts(start, a_projection(start, rows))


def test_a_reader_asks_for_the_rows_in_the_call_that_asks_for_the_attempts() -> None:
    """They are the third list of that one answer, read off the same moment as the other two.

    An attempt whose evidence went missing reads as delivered in one answer and as refused in
    another, and which of the two is true is a question about one point in the run.
    """
    records = GenerationRecords(
        attempts=[], presentations=[], operation_failures=[a_failure()]
    )
    assert records.operation_failures == [a_failure()]
    # And a generation that has recorded none carries an empty list rather than nothing.
    assert GenerationRecords(attempts=[], presentations=[]).operation_failures == []


def test_a_generation_with_no_receipt_to_lose_records_none_of_this() -> None:
    """The rows are a receipt's own record, and a generation with none writes none.

    A pre-artifact generation can have a claim refused over an object its history cites, and that
    refusal is the one it always was. Recording it here would synthesize provenance for a
    generation that has none and would move the carrier it writes.
    """
    start = a_start()
    projection = a_projection(start)
    assert start.receipt_contracts == []
    assert projection.operation_failures == []
    assert carrier_version(start, projection) == LEGACY_CARRIER_SCHEMA_VERSION
