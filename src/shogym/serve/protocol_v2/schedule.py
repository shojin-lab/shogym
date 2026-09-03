"""What a generation assigns before it behaves, and what releases it afterwards.

Two immutable records, and keeping them apart is the point. An :class:`Assignment` fixes one
task's positions and public identifiers and names the plan that will release its payload. It is
committed with the generation, before anything is offered, and nothing later can move it. A
:class:`ReleasePlan` says when an obligation becomes eligible, whether a payload outranks a
task, and how two messages that become eligible together are ordered. A release can change
readiness and it can change nothing else, which is why it is a second object rather than a
field of the first.

Two plans live here. :data:`IMMEDIATE` releases an obligation at the seal of the attempt it
belongs to: the complete ordered item list the candidate bundle built becomes eligible in the
seal transaction, not a narrowed part of it. :data:`NEVER` creates no obligation at all, so a
generation under it has no outbox and an evaluation-only generation pins it. A delayed plan,
its timers, the blocked state and the end-of-queue tail policies are not built here. What is
built is the interface they need: a named predicate with a version, a declared priority, a
declared tie key, and an eligibility hook.

The hook is the third thing a plan may declare. A gate names one task and one fact its
eligibility waits on: a payload being delivered, or another named task sealing. It is a
predicate over facts the generation already holds rather than a lag in milliseconds, which is
what lets a leg automaton be written as a schedule instead of as a second scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from shogym.serve.protocol_v2 import jcs
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.identity import length_prefixed
from shogym.serve.protocol_v2.records import require_opaque_id

# The release predicates a plan may declare. A predicate names the fact an obligation
# waits on, so a delayed one is a third name here and no change anywhere else.
RELEASE_AT_SEAL = "at_seal"
RELEASE_NEVER = "never"
RELEASE_PREDICATES: Tuple[str, ...] = (RELEASE_AT_SEAL, RELEASE_NEVER)

# Which kind of message a plan puts first when both are eligible.
PAYLOAD_FIRST = "payload_first"
TASK_FIRST = "task_first"
PRIORITIES: Tuple[str, ...] = (PAYLOAD_FIRST, TASK_FIRST)

# How simultaneously eligible messages of one kind are ordered.
BY_POSITION = "position"
BY_ATTEMPT_ID = "attempt_id"
TIE_KEYS: Tuple[str, ...] = (BY_POSITION, BY_ATTEMPT_ID)

PREDICATE_VERSION = "shogym.release.1"

TASK = "task"
PAYLOAD = "payload"


def _one_of(name: str, value: Any, allowed: Tuple[str, ...]) -> None:
    if value not in allowed:
        raise WireFormatError(f"{name} must be one of {', '.join(allowed)}")


def _position(name: str, value: Any) -> None:
    if type(value) is not int or value < 0:
        raise WireFormatError(f"{name} must be an integer position from 0")


@dataclass(frozen=True)
class EligibilityGate:
    """One task, and the single fact its eligibility waits on.

    ``after_payload_position`` is the leg boundary rule: the task becomes eligible once the
    payload at that position has been presented, so a plan that names one task there makes that
    task the only one a pull can reserve after payload k. ``after_sealed_attempt_id`` is the
    transfer rule: the task becomes eligible once another named task has sealed.

    Exactly one condition is declared, because a gate with two would be an order the plan has
    not written down.
    """

    attempt_id: str
    after_payload_position: Optional[int] = None
    after_sealed_attempt_id: Optional[str] = None

    def __post_init__(self) -> None:
        require_opaque_id("attempt_id", self.attempt_id)
        declared = [
            condition
            for condition in (self.after_payload_position, self.after_sealed_attempt_id)
            if condition is not None
        ]
        if len(declared) != 1:
            raise WireFormatError("a gate declares exactly one condition")
        if self.after_payload_position is not None:
            _position("after_payload_position", self.after_payload_position)
        if self.after_sealed_attempt_id is not None:
            require_opaque_id("after_sealed_attempt_id", self.after_sealed_attempt_id)
            if self.after_sealed_attempt_id == self.attempt_id:
                raise WireFormatError("a task cannot wait for its own seal")


@dataclass(frozen=True)
class ReleasePlan:
    """When an obligation becomes eligible, and what order eligibility is served in.

    Everything here is declared before the generation runs and read afterwards. A plan decides
    readiness only: there is no field it could select content with, so a release cannot change
    what an assignment fixed.
    """

    predicate: str
    priority: str
    tie_key: str
    gates: List[EligibilityGate] = field(default_factory=list)
    predicate_version: str = PREDICATE_VERSION

    def __post_init__(self) -> None:
        _one_of("predicate", self.predicate, RELEASE_PREDICATES)
        _one_of("priority", self.priority, PRIORITIES)
        _one_of("tie_key", self.tie_key, TIE_KEYS)
        if not isinstance(self.predicate_version, str) or not self.predicate_version:
            raise WireFormatError("predicate_version names the version of the declared rule")
        governed = [gate.attempt_id for gate in self.gates]
        if len(set(governed)) != len(governed):
            raise WireFormatError("a task has at most one gate")

    @property
    def release_plan_id(self) -> str:
        """This plan's identity: a hash of everything it declares.

        Two arms that declare the same plan carry the same plan ID, which makes a matched
        schedule skeleton something a test can compare rather than something a run asserts.
        """
        declared = {
            "predicate": self.predicate,
            "predicate_version": self.predicate_version,
            "priority": self.priority,
            "tie_key": self.tie_key,
            "gates": [
                {
                    "attempt_id": gate.attempt_id,
                    "after_payload_position": gate.after_payload_position,
                    "after_sealed_attempt_id": gate.after_sealed_attempt_id,
                }
                for gate in sorted(self.gates, key=lambda gate: gate.attempt_id)
            ],
        }
        digest = sha256(
            length_prefixed(b"release-plan-v2") + length_prefixed(jcs.encode(declared))
        )
        return digest.hexdigest()[:32]

    @property
    def creates_obligations(self) -> bool:
        """Whether this plan has an outbox at all. Never does not."""
        return self.predicate != RELEASE_NEVER

    def gate_for(self, attempt_id: str) -> Optional[EligibilityGate]:
        """The gate governing ``attempt_id``, or nothing when the queue governs it."""
        for gate in self.gates:
            if gate.attempt_id == attempt_id:
                return gate
        return None


IMMEDIATE = ReleasePlan(predicate=RELEASE_AT_SEAL, priority=PAYLOAD_FIRST, tie_key=BY_POSITION)
NEVER = ReleasePlan(predicate=RELEASE_NEVER, priority=PAYLOAD_FIRST, tie_key=BY_POSITION)


@dataclass(frozen=True)
class Assignment:
    """One roster row, committed before the behavior it can affect.

    It fixes the positions and the public identifiers of one task and names the plan that will
    release its payload. Pulls, retries, Waits, seal order, and wall time change none of them.
    There is no cell, arm, or seed here: what an experiment assigns on top of a position is the
    experiment package's, and the roster this kernel commits is the part every generation has.

    ``creates_payload_obligation`` says whether this generation creates a payload obligation
    for this position. A plan can say that there is no obligation anywhere, which is what Never
    says, and it has no field that could name which positions are silent, so the fact is
    resolved once here and fixed with the rest of the row. A row that creates none is a task
    the model works and the stream scores with nothing ever built or delivered against it,
    which is what a leg's filler is, and under Never every row is that. Reading the column
    alone is therefore reading what happened, which is what an analysis needs of it.
    """

    assignment_id: str
    attempt_id: str
    task_position: int
    payload_position: int
    task_message_id: str
    ack_message_id: str
    payload_message_id: str
    release_plan_id: str
    creates_payload_obligation: bool = True

    def __post_init__(self) -> None:
        if type(self.creates_payload_obligation) is not bool:
            raise WireFormatError("creates_payload_obligation is either true or false")
        for name in (
            "assignment_id",
            "attempt_id",
            "task_message_id",
            "ack_message_id",
            "payload_message_id",
            "release_plan_id",
        ):
            require_opaque_id(name, getattr(self, name))
        _position("task_position", self.task_position)
        _position("payload_position", self.payload_position)


def assignment_id_for(attempt_id: str) -> str:
    """Return the roster row identity for ``attempt_id``.

    It is derived rather than minted so that two generations built from one manifest produce
    the same roster, which is what lets their rows be compared row for row.
    """
    require_opaque_id("attempt_id", attempt_id)
    digest = sha256(
        length_prefixed(b"assignment-v2") + length_prefixed(attempt_id.encode("utf-8"))
    )
    return digest.hexdigest()[:32]


@dataclass(frozen=True)
class ScheduleView:
    """The facts a schedule is allowed to read.

    There is no score here, no cell, and no clock. A predicate that cannot see a verdict cannot
    release on one, and the restriction is cheaper to keep in the type than to test for later.
    """

    offered_attempts: FrozenSet[str]
    sealed_attempts: FrozenSet[str]
    presented_payload_positions: FrozenSet[int]


def gate_open(gate: EligibilityGate, view: ScheduleView) -> bool:
    """Whether the fact ``gate`` waits on has happened."""
    if gate.after_payload_position is not None:
        return gate.after_payload_position in view.presented_payload_positions
    return gate.after_sealed_attempt_id in view.sealed_attempts


def eligible_tasks(
    plan: ReleasePlan, assignments: Sequence[Assignment], view: ScheduleView
) -> List[Assignment]:
    """Every task the plan would let a pull reserve, before capacity is considered.

    An ungated task waits its turn: the first one not yet offered is eligible and the ones
    behind it are not, which is the ordinary closed queue. A gated task ignores the queue and
    waits for the fact its gate names, which is how a leg automaton leaves exactly one task
    eligible after a payload has been presented.
    """
    ready: List[Assignment] = []
    queue_head_taken = False
    for assignment in sorted(assignments, key=lambda row: row.task_position):
        if assignment.attempt_id in view.offered_attempts:
            continue
        gate = plan.gate_for(assignment.attempt_id)
        if gate is None:
            if not queue_head_taken:
                ready.append(assignment)
                queue_head_taken = True
        elif gate_open(gate, view):
            ready.append(assignment)
    return ready


def order_key(plan: ReleasePlan, kind: str, assignment: Assignment) -> Tuple[int, int, str]:
    """The declared total order over messages that are eligible at the same moment.

    The first element is the plan's payload-versus-task priority and the rest is its tie key.
    Positions are unique within a generation and so are attempt IDs, so either key orders any
    two messages of one kind and the priority orders the rest. The order is a function of the
    plan and the roster alone: nothing about when a pull arrived enters it.
    """
    leading = PAYLOAD if plan.priority == PAYLOAD_FIRST else TASK
    rank = 0 if kind == leading else 1
    if plan.tie_key == BY_ATTEMPT_ID:
        return (rank, 0, assignment.attempt_id)
    position = assignment.payload_position if kind == PAYLOAD else assignment.task_position
    return (rank, position, assignment.attempt_id)


def check_release(
    plan: ReleasePlan, assignments: Sequence[Assignment], *, evaluation_only: bool
) -> None:
    """Refuse a schedule this generation could not serve, before it serves anything.

    Everything checked here is fixed at construction, so a schedule that would leave a
    generation with a task nothing can make eligible, or with an outbox an evaluation must not
    have, fails while there is still nobody to answer.
    """
    if evaluation_only and plan.creates_obligations:
        raise WireFormatError(
            "an evaluation-only generation releases no payload, so its plan must be Never"
        )
    rows: Dict[str, Assignment] = {row.attempt_id: row for row in assignments}
    if len(rows) != len(assignments):
        raise WireFormatError("a generation assigns each attempt exactly once")
    for name, values in (
        ("assignment_id", [row.assignment_id for row in assignments]),
        ("task_position", [row.task_position for row in assignments]),
        ("payload_position", [row.payload_position for row in assignments]),
    ):
        if len(set(values)) != len(values):
            raise WireFormatError(f"{name} is unique within a generation")
    for row in assignments:
        if row.release_plan_id != plan.release_plan_id:
            raise WireFormatError("every assignment names the generation's own release plan")
        if row.creates_payload_obligation and not plan.creates_obligations:
            raise WireFormatError("this plan releases no payload, so no assignment creates one")
    positions = {row.payload_position for row in assignments}
    released = {row.payload_position for row in assignments if row.creates_payload_obligation}
    for gate in plan.gates:
        if gate.attempt_id not in rows:
            raise WireFormatError("a gate names a task this generation was assigned")
        if gate.after_payload_position is not None:
            if not plan.creates_obligations:
                raise WireFormatError("a gate cannot wait for a payload the plan never releases")
            if gate.after_payload_position not in positions:
                raise WireFormatError("a gate names a payload position this generation assigned")
            if gate.after_payload_position not in released:
                raise WireFormatError("a gate waits for a payload no assignment creates")
        if gate.after_sealed_attempt_id is not None and gate.after_sealed_attempt_id not in rows:
            raise WireFormatError("a gate names a task this generation was assigned")
    _check_gates_terminate(plan)


def _check_gates_terminate(plan: ReleasePlan) -> None:
    """Refuse gates that wait on each other.

    Two tasks each waiting for the other's seal is a generation that would Wait forever and
    never reach Done, which is a configuration mistake rather than a state to recover from.
    """
    for gate in plan.gates:
        seen = {gate.attempt_id}
        waiting_on = gate.after_sealed_attempt_id
        while waiting_on is not None:
            if waiting_on in seen:
                raise WireFormatError("these gates wait on each other, so no task could seal")
            seen.add(waiting_on)
            following = plan.gate_for(waiting_on)
            waiting_on = following.after_sealed_attempt_id if following else None
