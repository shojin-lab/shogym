"""What a generation whose subject is not its payload policy registers in order to serve at all.

A generation says what each of its payloads may contain, and there is no default under the
experiment profile. Most of the kernel's tests are about something else: the schedule, the seal,
a resume, a reader, an ending. Their subject is the receipt that says a filing was answered and
nothing about the work in it, which is a body an experiment registers rather than a body anything
falls back to.

So this is that registration, once, rather than a policy decision repeated in each of those
files. What it declares is the stream's own stand-in grade, the blinded receipt against every
position that owes a payload, and the reason against every position that owes none.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List

from shogym.serve.protocol_v2.kernel.messages import StreamStart, assignments_for
from shogym.serve.protocol_v2.policy import (
    BLINDED_RECEIPT_V1,
    DELIVER,
    EXPERIMENT,
    KERNEL_STAND_IN_GRADE,
    REGISTERED,
    WITHHOLD,
    GradeIdentity,
    PayloadDisposition,
    PolicyProvenance,
    policy_digest,
    roster_digest,
)

__all__ = ["registering_the_receipt"]


def registering_the_receipt(
    start: StreamStart, *, grade: GradeIdentity = KERNEL_STAND_IN_GRADE
) -> StreamStart:
    """Return ``start`` as an experiment that registered the blinded receipt everywhere.

    With the registration that entitles it to say so, because a profile with nothing behind it
    is a word rather than a fact about the run.
    """
    roster = list(start.assignments) or assignments_for(start.tasks, start.release)
    owed = start.release.creates_obligations
    rows: List[PayloadDisposition] = [
        PayloadDisposition(
            attempt_id=row.attempt_id,
            payload_position=row.payload_position,
            kind=DELIVER,
            policy_digest=policy_digest(BLINDED_RECEIPT_V1),
            cell=BLINDED_RECEIPT_V1.cells[0],
        )
        if owed and row.creates_payload_obligation
        else PayloadDisposition(
            attempt_id=row.attempt_id,
            payload_position=row.payload_position,
            kind=WITHHOLD,
            reason="this generation is not about what a body says",
        )
        for row in roster
    ]
    return replace(
        start,
        profile=EXPERIMENT,
        grade=grade,
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_receipt_is_the_subject",
        ),
    )
