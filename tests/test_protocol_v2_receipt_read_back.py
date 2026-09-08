"""The two axes a receipt run is read on, and the file they are written into.

What became of a receipt and whether its evidence is still there are two questions. Read as one
list they collide: a committed row whose objects later go missing answers two of the values at
once, a withheld row that loses a body answers none of them because it never had a selection to
be about, and a queued position waiting for its turn answers none either. So the lifecycle is one
axis of seven values and the availability is another of two, and this file holds them to that.

Everything here is pure. The rows are built by hand, which is what lets one file hold every state
a generation can project including the ones a single run cannot reach at once, and the end to end
half, where the rows are the ones a real generation wrote, is beside it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from shogym.serve.protocol_v2.kernel.messages import (
    AttemptRecord,
    OperationFailure,
    PresentedMessage,
    SourceProvenance,
)
from shogym.serve.protocol_v2.reader import (
    RECEIPT_ASSIGNED_BUT_UNSEALED,
    RECEIPT_AVAILABILITIES,
    RECEIPT_AVAILABLE,
    RECEIPT_BUILT,
    RECEIPT_COMMITTED,
    RECEIPT_FAILED_BEFORE_CAPTURE,
    RECEIPT_LEGACY,
    RECEIPT_LIFECYCLES,
    RECEIPT_OFFERED,
    RECEIPT_UNAVAILABLE,
    RECEIPT_WITHHELD,
    OPERATIONS_FILE,
    RECORDS_FILE,
    RunRecords,
    _LIFECYCLE_TESTS,
    observed_receipt,
    receipt_availability,
    receipt_lifecycle,
    write_records,
)

import pytest

CONTRACT = "ledger_receipt"
COMMITMENT = "a" * 64
BODY = "b" * 64
VISIBLE = "c" * 64
POLICY = "d" * 64


def oid(value: int) -> str:
    return f"{value:032x}"


def a_source(**changed: Any) -> SourceProvenance:
    """One committed source as a row carries it, with the selection made from it."""
    fields: Dict[str, Any] = {
        "source_commitment": COMMITMENT,
        "bundle_digest": "e" * 64,
        "environment_task_id": "ledger/0",
        "source_attempt_id": oid(0x100),
        "source_seal_id": "f" * 64,
        "execution_ordinal": 0,
        "canonicalization_version": "receipts.1",
        "renderer_configuration": "ledger-renderer-1",
        "canonical_submission_sha256": "1" * 64,
        "kernel_submission_digest": "2" * 64,
        "selected_cell": "graded",
        "selected_body_reference": BODY,
        "selected_policy_digest": POLICY,
    }
    fields.update(changed)
    return SourceProvenance(**fields)


def a_row(**changed: Any) -> AttemptRecord:
    """One attempt's record, with every field this file does not care about fixed."""
    fields: Dict[str, Any] = {
        "attempt_id": oid(0x100),
        "task_position": 0,
        "payload_position": 0,
        "state": "ack_presented",
        "terminal_tool": "submit_filing",
        "terminal_source": "agent",
        "canonicalization_version": "receipts.1",
        "submission_digest": "3" * 64,
        "score": 0.5,
        "decode_state": "decoded",
        "seal_ordinal": 1,
        "final_failure": None,
        "deadline_expired": False,
        "task_message_id": oid(0x101),
        "task_delivered": True,
        "ack_message_id": oid(0x102),
        "ack_delivered": True,
        "payload_message_id": oid(0x103),
        "payload_delivered": True,
        "creates_payload_obligation": True,
        "payload_state": "presented",
        "payload_policy": "graded-receipt-artifact-v1",
        "payload_disposition": "deliver:graded-receipt-artifact-v1:graded:registered",
        "profile": "experiment",
        "payload_resolution_source": "registered",
        "receipt_contract_id": CONTRACT,
        "source_provenance": a_source(),
        "payload_visible_sha256": VISIBLE,
    }
    fields.update(changed)
    return AttemptRecord(**fields)


def a_failure(**changed: Any) -> OperationFailure:
    """One operation failure naming this attempt, refused unless a test says otherwise."""
    fields: Dict[str, Any] = {
        "operation": "4" * 64,
        "phase": "continued_first_delivery",
        "reason": "unavailable_evidence",
        "outcome": "refused",
        "generation": "stream/read-back/1",
        "refused_epoch": 1,
        "attempt_id": oid(0x100),
    }
    fields.update(changed)
    return OperationFailure(**fields)


#: One row per lifecycle value, each in the state that value names.
ROWS: Dict[str, AttemptRecord] = {
    RECEIPT_LEGACY: a_row(
        receipt_contract_id=None,
        source_provenance=None,
        final_failure="seal_failed",
        state="final_failed",
        payload_state="final_failed",
        payload_delivered=False,
        payload_visible_sha256=None,
        payload_policy="blinded-receipt-v1",
    ),
    RECEIPT_FAILED_BEFORE_CAPTURE: a_row(
        source_provenance=None,
        final_failure="seal_failed",
        state="final_failed",
        payload_state="final_failed",
        payload_delivered=False,
        payload_visible_sha256=None,
    ),
    RECEIPT_ASSIGNED_BUT_UNSEALED: a_row(
        source_provenance=None,
        score=None,
        seal_ordinal=None,
        state="active",
        payload_state="assigned",
        payload_delivered=False,
        payload_visible_sha256=None,
    ),
    RECEIPT_WITHHELD: a_row(
        source_provenance=a_source(
            selected_cell=None, selected_body_reference=None, selected_policy_digest=None
        ),
        creates_payload_obligation=False,
        payload_state=None,
        payload_delivered=False,
        payload_visible_sha256=None,
        payload_policy=None,
    ),
    RECEIPT_BUILT: a_row(
        payload_state="eligible", payload_delivered=False, payload_visible_sha256=None
    ),
    RECEIPT_OFFERED: a_row(
        payload_state="offered", payload_delivered=False, payload_visible_sha256=None
    ),
    RECEIPT_COMMITTED: a_row(),
}


def matching(record: AttemptRecord) -> List[str]:
    """Every lifecycle value whose own definition describes this row."""
    return [value for value, matches in _LIFECYCLE_TESTS if matches(record)]


def a_run(
    root: Path,
    records: List[AttemptRecord],
    failures: Optional[List[OperationFailure]] = None,
) -> RunRecords:
    return RunRecords(
        root=root,
        workflow_id="stream/read-back/1",
        records=records,
        presentations=[
            PresentedMessage(
                order=0,
                kind="payload",
                message_id=oid(0x103),
                attempt_id=oid(0x100),
                visible_bytes_sha256=VISIBLE,
            )
        ],
        operation_failures=list(failures or []),
    )


def exported(run: RunRecords) -> List[Dict[str, Any]]:
    path = write_records(run)
    assert path == run.root / RECORDS_FILE
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def episodes(run: RunRecords) -> List[Dict[str, Any]]:
    """The failed operations one export writes beside the attempts, in the order it wrote them."""
    write_records(run)
    text = (run.root / OPERATIONS_FILE).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_exactly_one_lifecycle_value_describes_every_row_a_generation_projects() -> None:
    """Seven values, and each row answers one of them by its own definition and not by order.

    Both halves are checked here: every value has a row, and every row is described by exactly
    one value rather than by the first one it happens to reach.
    """
    assert sorted(ROWS) == sorted(RECEIPT_LIFECYCLES)
    assert [value for value, _ in _LIFECYCLE_TESTS] == list(RECEIPT_LIFECYCLES)
    for value, record in ROWS.items():
        assert matching(record) == [value]
        assert receipt_lifecycle(record) == value


def test_a_pre_artifact_row_that_ended_in_a_seal_failure_reads_as_legacy() -> None:
    """Absence that predates the question is not an answer to it.

    An attempt from a generation that declared no contract can end in a seal failure like any
    other, and it holds no descriptor because none was ever asked of it. Read as a receipt that
    failed before capture it would be a run that lost evidence it never had.
    """
    ended = ROWS[RECEIPT_LEGACY]
    assert ended.final_failure == "seal_failed"
    assert receipt_lifecycle(ended) == RECEIPT_LEGACY
    # And the same ending under a declared contract is the other answer, which is what makes the
    # order of the two load bearing rather than incidental.
    assert receipt_lifecycle(ROWS[RECEIPT_FAILED_BEFORE_CAPTURE]) == RECEIPT_FAILED_BEFORE_CAPTURE


def test_a_queued_position_before_capture_is_assigned_rather_than_nothing() -> None:
    """A delivering row and a withheld one read alike before either has captured a source."""
    delivering = ROWS[RECEIPT_ASSIGNED_BUT_UNSEALED]
    queued = a_row(
        source_provenance=None,
        score=None,
        seal_ordinal=None,
        state="planned",
        creates_payload_obligation=False,
        payload_state=None,
        payload_delivered=False,
        payload_visible_sha256=None,
        payload_policy=None,
    )
    assert receipt_lifecycle(delivering) == RECEIPT_ASSIGNED_BUT_UNSEALED
    assert receipt_lifecycle(queued) == RECEIPT_ASSIGNED_BUT_UNSEALED
    assert matching(queued) == [RECEIPT_ASSIGNED_BUT_UNSEALED]


def test_availability_reads_beside_the_lifecycle_and_never_instead_of_it(tmp_path: Path) -> None:
    """A committed row that loses an object is committed and unavailable, and both are true.

    What was committed happened, and no later loss unhappens it. The withheld case is why the two
    are separate axes: a row with no selection has nothing an unavailable predicate about a
    selection could be about, and it can still lose the body it captured.
    """
    committed = ROWS[RECEIPT_COMMITTED]
    withheld = ROWS[RECEIPT_WITHHELD]
    lost = [a_failure()]
    assert (receipt_lifecycle(committed), receipt_availability(committed, lost)) == (
        RECEIPT_COMMITTED,
        RECEIPT_UNAVAILABLE,
    )
    assert (receipt_lifecycle(withheld), receipt_availability(withheld, lost)) == (
        RECEIPT_WITHHELD,
        RECEIPT_UNAVAILABLE,
    )
    assert receipt_availability(committed, []) == RECEIPT_AVAILABLE
    assert set(RECEIPT_AVAILABILITIES) == {RECEIPT_AVAILABLE, RECEIPT_UNAVAILABLE}
    # And the file says both, in that order, beside the row's own fields.
    [row] = exported(a_run(tmp_path, [committed], lost))
    assert row["receipt_lifecycle"] == RECEIPT_COMMITTED
    assert row["receipt_availability"] == RECEIPT_UNAVAILABLE


def test_the_last_row_naming_an_attempt_is_the_word_on_its_availability() -> None:
    """A refusal and the recovery that answered it both stand, and the latest one answers."""
    record = ROWS[RECEIPT_COMMITTED]
    refused = a_failure()
    recovered = a_failure(outcome="recovered", recovered_epoch=2)
    assert receipt_availability(record, [refused]) == RECEIPT_UNAVAILABLE
    assert receipt_availability(record, [refused, recovered]) == RECEIPT_AVAILABLE
    assert receipt_availability(record, [refused, recovered, refused]) == RECEIPT_UNAVAILABLE
    # An ending nothing can answer reads as unavailable for as long as the record stands.
    assert receipt_availability(record, [a_failure(outcome="unrecoverable")]) == (
        RECEIPT_UNAVAILABLE
    )


def test_a_failure_naming_another_attempt_says_nothing_about_this_one() -> None:
    """The rows are joined by attempt, so one position's loss is not another's."""
    record = ROWS[RECEIPT_COMMITTED]
    elsewhere = a_failure(attempt_id=oid(0x200))
    nobodys = a_failure(attempt_id=None)
    assert receipt_availability(record, [elsewhere, nobodys]) == RECEIPT_AVAILABLE


def test_a_row_in_no_state_this_generation_projects_is_reported_rather_than_labelled() -> None:
    """The point of two axes was to stop a row matching nothing being filed under the nearest.

    Nothing this build projects reaches here: an obligation is only ended while its attempt holds
    no source, and a captured attempt with an obligation always carries the selection made from
    it. A row that got here anyway is a claim about the projection, so it is said rather than
    dressed as one of the seven.
    """
    impossible = a_row(payload_state="final_failed", payload_delivered=False)
    assert matching(impossible) == []
    with pytest.raises(ValueError, match="no state this generation projects"):
        receipt_lifecycle(impossible)


def test_the_export_names_the_source_member_by_member(tmp_path: Path) -> None:
    """The nested value is converted rather than handed to the encoder as an object.

    The row builder hands raw field values over, and a source is a record and not a string, so a
    file that relied on attribute lookup would either fail to encode or write whatever a future
    field happened to be. Naming the members is what makes the file's shape a decision.
    """
    [committed, withheld] = exported(
        a_run(tmp_path, [ROWS[RECEIPT_COMMITTED], ROWS[RECEIPT_WITHHELD]])
    )
    source = committed["source_provenance"]
    assert source == {
        "source_commitment": COMMITMENT,
        "bundle_digest": "e" * 64,
        "environment_task_id": "ledger/0",
        "source_attempt_id": oid(0x100),
        "source_seal_id": "f" * 64,
        "execution_ordinal": 0,
        "canonicalization_version": "receipts.1",
        "renderer_configuration": "ledger-renderer-1",
        "canonical_submission_sha256": "1" * 64,
        "kernel_submission_digest": "2" * 64,
        "selected_cell": "graded",
        "selected_body_reference": BODY,
        "selected_policy_digest": POLICY,
        "protocol_version": 2,
    }
    assert committed["receipt_contract_id"] == CONTRACT
    assert committed["payload_visible_sha256"] == VISIBLE
    assert committed["score"] == 0.5
    # A position that delivers nothing keeps its source and its grade and names no selection, and
    # the file carries the same members either way so a reader joins on an absence.
    assert list(withheld["source_provenance"]) == list(source)
    assert withheld["source_provenance"]["selected_cell"] is None
    assert withheld["score"] == 0.5


def test_a_row_that_never_carried_a_source_writes_the_absence(tmp_path: Path) -> None:
    """A legacy row exports the same keys and no answers, so the file's shape does not move."""
    [row] = exported(a_run(tmp_path, [ROWS[RECEIPT_LEGACY]]))
    assert row["source_provenance"] is None
    assert row["receipt_contract_id"] is None
    assert row["receipt_lifecycle"] == RECEIPT_LEGACY
    assert row["receipt_availability"] == RECEIPT_AVAILABLE


def test_a_receipt_is_observed_only_where_the_harness_wrote_those_exact_bytes_down() -> None:
    """Committed is this generation's claim, and observation is the harness's.

    A commitment is accepted before the transport has a result to hand anybody, so a row can be
    committed with nobody having received the bytes. What settles it is the harness's own durable
    entry for that message, matched on the identity and on the digest of what it wrote, and both
    have to hold: an entry under another identity is another message, and an entry whose digest
    is not this one is not a witness for these bytes.
    """
    committed = ROWS[RECEIPT_COMMITTED]
    assert observed_receipt(committed, {committed.payload_message_id: VISIBLE})
    assert not observed_receipt(committed, {})
    assert not observed_receipt(committed, {committed.payload_message_id: "9" * 64})
    assert not observed_receipt(committed, {oid(0x999): VISIBLE})


def test_a_commitment_that_never_happened_is_never_observed() -> None:
    """No witness upgrades a row this generation never committed, however it was written down."""
    for value in (RECEIPT_BUILT, RECEIPT_OFFERED, RECEIPT_WITHHELD, RECEIPT_LEGACY):
        record = ROWS[value]
        assert not observed_receipt(record, {record.payload_message_id: VISIBLE})


def test_the_export_carries_the_failed_operations_member_by_member(tmp_path: Path) -> None:
    """One word on an attempt's line is not the episode, so the episode is exported beside it.

    The availability axis says whether the evidence behind a receipt is still there, and that is
    all a row can say. Which operation failed, at which phase, for which reason, under which
    epoch, which objects it could not produce and whether anything answered it afterwards are the
    facts a controller acts on, and a file that dropped them would hand an external reader an
    unavailable row and no way to find out what to repair.

    The members are named rather than read off the record, and the order is the commit order the
    generation kept them in, because a refusal and the recovery that answered it are two rows
    about one operation and the latest word is the one at the end.
    """
    refused = a_failure(references=[BODY], request_id=oid(0x1001), payload_position=0)
    recovered = a_failure(
        outcome="recovered",
        references=[BODY],
        request_id=oid(0x1001),
        payload_position=0,
        recovered_epoch=2,
    )
    run = a_run(tmp_path, [ROWS[RECEIPT_COMMITTED]], [refused, recovered])
    rows = episodes(run)
    assert rows == [
        {
            "operation": "4" * 64,
            "phase": "continued_first_delivery",
            "reason": "unavailable_evidence",
            "outcome": outcome,
            "generation": "stream/read-back/1",
            "refused_epoch": 1,
            "attempt_id": oid(0x100),
            "payload_position": 0,
            "references": [BODY],
            "request_id": oid(0x1001),
            "recovered_epoch": epoch,
            "protocol_version": 2,
        }
        for outcome, epoch in (("refused", None), ("recovered", 2))
    ]


def test_an_operation_that_named_no_attempt_is_exported_where_no_attempt_row_could_hold_it(
    tmp_path: Path,
) -> None:
    """A claim refused before any receipt sealed is the operation's alone, and it still exports.

    There is no attempt for that row to be a column of, so a file that folded these into the
    attempts would lose it entirely, and it is exactly the row a controller reading an
    unrecoverable run needs.
    """
    alone = a_failure(
        phase="ownership_claim",
        attempt_id=None,
        references=["9" * 64],
        outcome="unrecoverable",
    )
    rows = episodes(a_run(tmp_path, [ROWS[RECEIPT_LEGACY]], [alone]))
    assert [row["attempt_id"] for row in rows] == [None]
    assert rows[0]["phase"] == "ownership_claim"
    assert rows[0]["outcome"] == "unrecoverable"
    assert rows[0]["references"] == ["9" * 64]
    # And a run with nothing to report writes the file anyway, so a reader finding it empty is
    # reading a run that had no failed operation rather than a build that forgot to write one.
    assert episodes(a_run(tmp_path, [ROWS[RECEIPT_COMMITTED]])) == []
