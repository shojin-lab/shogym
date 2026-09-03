"""The schedule model on its own: what a plan may declare, and what follows from it.

Nothing here starts a stream. A release plan, an assignment roster, and the order they imply
are pure values, and testing them as values is what keeps the rule that a release changes
readiness and never content checkable by reading rather than by running. These tests import no
Temporal, because the schedule model does not.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional

import pytest

from shogym.serve.protocol_v2 import (
    BY_ATTEMPT_ID,
    BY_POSITION,
    IMMEDIATE,
    NEVER,
    PAYLOAD_FIRST,
    RELEASE_AT_SEAL,
    TASK_FIRST,
    Assignment,
    EligibilityGate,
    ReleasePlan,
    ScheduleView,
    WireFormatError,
    assignment_id_for,
    check_release,
    eligible_tasks,
    order_key,
)
from shogym.serve.protocol_v2.schedule import PAYLOAD, TASK


def oid(value: int) -> str:
    return f"{value:032x}"


def attempt(index: int) -> str:
    return oid(0x100 + index * 4)


def roster(count: int, plan: ReleasePlan = IMMEDIATE) -> List[Assignment]:
    """One row per task, the way a closed manifest of ``count`` tasks implies it."""
    return [
        Assignment(
            assignment_id=assignment_id_for(attempt(index)),
            attempt_id=attempt(index),
            task_position=index,
            payload_position=index,
            task_message_id=oid(0x101 + index * 4),
            ack_message_id=oid(0x102 + index * 4),
            payload_message_id=oid(0x103 + index * 4),
            release_plan_id=plan.release_plan_id,
            creates_payload_obligation=plan.creates_obligations,
        )
        for index in range(count)
    ]


def view(
    offered: Optional[List[str]] = None,
    sealed: Optional[List[str]] = None,
    presented: Optional[List[int]] = None,
) -> ScheduleView:
    return ScheduleView(
        offered_attempts=frozenset(offered or []),
        sealed_attempts=frozenset(sealed or []),
        presented_payload_positions=frozenset(presented or []),
    )


def test_a_plan_is_identified_by_everything_it_declares() -> None:
    """Two arms declaring one plan carry one plan ID, and a changed rule is a changed plan."""
    restated = ReleasePlan(RELEASE_AT_SEAL, PAYLOAD_FIRST, BY_POSITION)
    assert IMMEDIATE.release_plan_id == restated.release_plan_id
    assert IMMEDIATE.release_plan_id != NEVER.release_plan_id
    for changed in (
        ReleasePlan(RELEASE_AT_SEAL, TASK_FIRST, BY_POSITION),
        ReleasePlan(RELEASE_AT_SEAL, PAYLOAD_FIRST, BY_ATTEMPT_ID),
        ReleasePlan(RELEASE_AT_SEAL, PAYLOAD_FIRST, BY_POSITION, predicate_version="other.1"),
        ReleasePlan(
            RELEASE_AT_SEAL,
            PAYLOAD_FIRST,
            BY_POSITION,
            gates=[EligibilityGate(attempt(1), after_payload_position=0)],
        ),
    ):
        assert changed.release_plan_id != IMMEDIATE.release_plan_id
    # Never is the plan with no outbox, and that is a property of the plan rather than of a run.
    assert IMMEDIATE.creates_obligations and not NEVER.creates_obligations
    # A roster row is derived from the attempt it assigns, so one manifest gives one roster.
    assert assignment_id_for(attempt(0)) == roster(1)[0].assignment_id


def test_a_plan_declares_only_rules_this_layer_has() -> None:
    """An undeclared predicate, priority, tie key, or gate is refused, not approximated."""
    with pytest.raises(WireFormatError, match="predicate"):
        ReleasePlan("after_a_while", PAYLOAD_FIRST, BY_POSITION)
    with pytest.raises(WireFormatError, match="priority"):
        ReleasePlan(RELEASE_AT_SEAL, "whatever_is_ready", BY_POSITION)
    with pytest.raises(WireFormatError, match="tie_key"):
        ReleasePlan(RELEASE_AT_SEAL, PAYLOAD_FIRST, "arrival")
    with pytest.raises(WireFormatError, match="exactly one condition"):
        EligibilityGate(attempt(1))
    with pytest.raises(WireFormatError, match="exactly one condition"):
        EligibilityGate(attempt(1), after_payload_position=0, after_sealed_attempt_id=attempt(0))
    with pytest.raises(WireFormatError, match="its own seal"):
        EligibilityGate(attempt(1), after_sealed_attempt_id=attempt(1))
    with pytest.raises(WireFormatError, match="at most one gate"):
        ReleasePlan(
            RELEASE_AT_SEAL,
            PAYLOAD_FIRST,
            BY_POSITION,
            gates=[
                EligibilityGate(attempt(1), after_payload_position=0),
                EligibilityGate(attempt(1), after_sealed_attempt_id=attempt(0)),
            ],
        )


def test_a_roster_is_checked_against_the_generation_it_claims_to_assign() -> None:
    """One row per attempt, unique positions, and every row naming this generation's plan."""
    check_release(IMMEDIATE, roster(3), evaluation_only=False)
    rows = roster(3)
    other = ReleasePlan(RELEASE_AT_SEAL, TASK_FIRST, BY_POSITION)
    with pytest.raises(WireFormatError, match="own release plan"):
        check_release(other, rows, evaluation_only=False)
    doubled = rows + [rows[0]]
    with pytest.raises(WireFormatError, match="exactly once"):
        check_release(IMMEDIATE, doubled, evaluation_only=False)
    collided = rows[:2] + [
        Assignment(
            assignment_id=assignment_id_for(attempt(9)),
            attempt_id=attempt(9),
            task_position=2,
            payload_position=1,
            task_message_id=oid(0x901),
            ack_message_id=oid(0x902),
            payload_message_id=oid(0x903),
            release_plan_id=IMMEDIATE.release_plan_id,
        )
    ]
    with pytest.raises(WireFormatError, match="payload_position is unique"):
        check_release(IMMEDIATE, collided, evaluation_only=False)


def test_an_evaluation_only_generation_pins_never() -> None:
    """A generation that scores without delivering has no outbox, and cannot be given one."""
    rows = roster(2, NEVER)
    check_release(NEVER, rows, evaluation_only=True)
    assert [row.creates_payload_obligation for row in rows] == [False, False]
    with pytest.raises(WireFormatError, match="must be Never"):
        check_release(IMMEDIATE, roster(2), evaluation_only=True)

    # A row cannot claim a payload this plan will never release. Those are one fact rather than
    # two, and a roster that said otherwise would read afterwards as a delivery gone missing.
    claiming = [replace(row, creates_payload_obligation=True) for row in rows]
    with pytest.raises(WireFormatError, match="no assignment creates one"):
        check_release(NEVER, claiming, evaluation_only=True)


def test_a_gate_names_facts_this_generation_will_have() -> None:
    """A gate waiting on something the generation does not contain is refused at construction."""
    stranger = ReleasePlan(
        RELEASE_AT_SEAL,
        PAYLOAD_FIRST,
        BY_POSITION,
        gates=[EligibilityGate(oid(0xABC), after_payload_position=0)],
    )
    with pytest.raises(WireFormatError, match="a task this generation was assigned"):
        check_release(stranger, roster(2, stranger), evaluation_only=False)

    late = ReleasePlan(
        RELEASE_AT_SEAL,
        PAYLOAD_FIRST,
        BY_POSITION,
        gates=[EligibilityGate(attempt(1), after_payload_position=7)],
    )
    with pytest.raises(WireFormatError, match="a payload position this generation assigned"):
        check_release(late, roster(2, late), evaluation_only=False)

    # A gate cannot wait for a payload under a plan that releases none.
    unreachable = ReleasePlan(
        "never",
        PAYLOAD_FIRST,
        BY_POSITION,
        gates=[EligibilityGate(attempt(1), after_payload_position=0)],
    )
    with pytest.raises(WireFormatError, match="never releases"):
        check_release(unreachable, roster(2, unreachable), evaluation_only=False)

    # Nor for a payload the roster gave that position none of. A row that creates no payload
    # obligation is a task with nothing delivered against it, which the leg's filler is: the
    # roster is accepted, and a gate waiting on the payload it does not have is not.
    silent = ReleasePlan(
        RELEASE_AT_SEAL,
        PAYLOAD_FIRST,
        BY_POSITION,
        gates=[EligibilityGate(attempt(1), after_payload_position=0)],
    )
    rows = roster(2, silent)
    check_release(
        silent,
        [rows[0], replace(rows[1], creates_payload_obligation=False)],
        evaluation_only=False,
    )
    with pytest.raises(WireFormatError, match="no assignment creates"):
        check_release(
            silent,
            [replace(rows[0], creates_payload_obligation=False), rows[1]],
            evaluation_only=False,
        )

    circular = ReleasePlan(
        RELEASE_AT_SEAL,
        PAYLOAD_FIRST,
        BY_POSITION,
        gates=[
            EligibilityGate(attempt(0), after_sealed_attempt_id=attempt(1)),
            EligibilityGate(attempt(1), after_sealed_attempt_id=attempt(0)),
        ],
    )
    with pytest.raises(WireFormatError, match="wait on each other"):
        check_release(circular, roster(2, circular), evaluation_only=False)


def test_the_order_is_declared_rather_than_discovered() -> None:
    """Priority ranks the two kinds and the tie key orders within one. Arrival enters neither."""
    rows = roster(3)
    payload_first = [order_key(IMMEDIATE, PAYLOAD, row)[0] for row in rows]
    task_first_plan = ReleasePlan(RELEASE_AT_SEAL, TASK_FIRST, BY_POSITION)
    assert payload_first == [0, 0, 0]
    assert [order_key(IMMEDIATE, TASK, row)[0] for row in rows] == [1, 1, 1]
    assert [order_key(task_first_plan, TASK, row)[0] for row in rows] == [0, 0, 0]
    assert [order_key(task_first_plan, PAYLOAD, row)[0] for row in rows] == [1, 1, 1]

    # The two tie keys are only told apart by a roster that disagrees with itself, so this one
    # runs the attempt IDs backwards against the positions: by position the lowest declared
    # position leads, and by attempt ID the position is not read at all.
    crossed = [
        replace(row, task_position=index, payload_position=index)
        for index, row in enumerate(reversed(rows))
    ]
    shuffled = list(reversed(crossed))
    by_position = ReleasePlan(RELEASE_AT_SEAL, PAYLOAD_FIRST, BY_POSITION)
    by_id = ReleasePlan(RELEASE_AT_SEAL, PAYLOAD_FIRST, BY_ATTEMPT_ID)
    ordered = sorted(shuffled, key=lambda row: order_key(by_position, PAYLOAD, row))
    assert [row.payload_position for row in ordered] == [0, 1, 2]
    assert [row.attempt_id for row in ordered] == [attempt(2), attempt(1), attempt(0)]
    ordered = sorted(shuffled, key=lambda row: order_key(by_id, PAYLOAD, row))
    assert [row.attempt_id for row in ordered] == [attempt(0), attempt(1), attempt(2)]
    assert [row.payload_position for row in ordered] == [2, 1, 0]


def test_the_leg_hook_makes_one_task_eligible_at_a_time() -> None:
    """The delayed transfer leg, as a plan: A_1, then its payload, then the filler, then B.

    Nothing here counts a lag. Each task waits for a fact the generation will hold, so the
    automaton is a predicate over what has happened rather than a second schedule of its own.
    """
    plan = ReleasePlan(
        RELEASE_AT_SEAL,
        PAYLOAD_FIRST,
        BY_POSITION,
        gates=[
            EligibilityGate(attempt(1), after_payload_position=0),
            EligibilityGate(attempt(2), after_sealed_attempt_id=attempt(1)),
        ],
    )
    rows = roster(3, plan)
    check_release(plan, rows, evaluation_only=False)

    def ready(state: ScheduleView) -> List[str]:
        return [row.attempt_id for row in eligible_tasks(plan, rows, state)]

    # The ungated head of the queue, and nothing behind it: both later tasks are gated shut.
    assert ready(view()) == [attempt(0)]
    assert ready(view(offered=[attempt(0)])) == []
    assert ready(view(offered=[attempt(0)], sealed=[attempt(0)])) == []
    after_payload = view(offered=[attempt(0)], sealed=[attempt(0)], presented=[0])
    assert ready(after_payload) == [attempt(1)]
    # And B waits for the filler's seal, not for its payload and not for a clock.
    after_filler = view(
        offered=[attempt(0), attempt(1)], sealed=[attempt(0), attempt(1)], presented=[0]
    )
    assert ready(after_filler) == [attempt(2)]
