"""The values that cross the durable stream's boundaries.

Three boundaries, one module. A stream is started with :class:`StreamStart`, whose queue,
capacity, public identifiers, and terminal tool are fixed for the life of the generation and
are the only things the workflow will ever treat as configuration. A caller reaches the
running stream through the Update arguments and results here. And the workflow reaches the
outside world through the Activity inputs and results here.

Every one of these is a plain frozen dataclass, because Temporal writes each of them into a
history that has to be replayed years later and reads it back by field name. The wire records
in :mod:`shogym.serve.protocol_v2.records` are not restated: an Update that carries a protocol
record carries that record. What this module adds around them is the operator-side material
the model never sees, which is why identifiers like the hidden execution ID and the seal ID
live here and not there.

A result the model can see travels as :class:`OfferedMessage`, which carries the canonical
bytes as text rather than a decoded record. The bytes are the authority: they are what the
digest covers, what the harness must insert verbatim, and what the presentation attests to.
Handing back a record instead would let a gateway re-serialize and present bytes nobody
committed to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Optional, Sequence

from shogym.serve.protocol_v2 import (
    IMMEDIATE,
    PROTOCOL_VERSION,
    SCHEDULE_VERSION,
    Assignment,
    BlobRef,
    ReleasePlan,
    TerminalMetadata,
    assignment_id_for,
    canonical_json,
    length_prefixed,
)


@dataclass(frozen=True)
class TaskItem:
    """One entry of the closed queue, with every public identifier it will ever use.

    The identifiers are preallocated rather than minted on the way out, so two generations
    built from the same manifest put the same IDs in the same positions and a transcript
    comparison between them is a comparison of content.
    """

    task_position: int
    attempt_id: str
    task_message_id: str
    ack_message_id: str
    payload_position: int
    payload_message_id: str
    body: str


@dataclass(frozen=True)
class TerminalTool:
    """The one tool that can end an attempt, and the argument names it declares.

    ``argument_names`` is the whole of the schema this kernel checks. A real environment's
    schema arrives with the environment.
    """

    public_tool_name: str
    native_terminal_name: str
    argument_names: List[str]


@dataclass(frozen=True)
class StreamStart:
    """Everything a generation is, fixed before it serves anything.

    Nothing here can be changed by an Update. ``id_key_hex`` and ``hidden_execution_id`` are
    operator material: the first keys the message IDs that are not preallocated, the second
    separates two executions that share a public attempt ID, and neither is ever copied into a
    model-visible message.

    ``assignments`` is the generation's roster and ``release`` is its schedule. They arrive
    together and before anything is served, which is what makes assignment a fact about the
    generation rather than a consequence of how it ran. An empty roster is filled in from the
    closed manifest at start, still before an offer. ``evaluation_only`` says this generation
    exists to score and not to deliver, and it is refused unless its plan is Never.

    ``configuration_hash`` is the environment's half of what this generation is: the contract,
    the instructions, the tools. The whole of it is :func:`configuration_hash`, which folds that
    value together with the manifest, the roster, the plan, the capacity, and the versions, and
    it is what a resume has to present. ``blob_root`` is the directory the blobs an event may
    reference are installed in. A generation without one verifies no reference, and says so.
    """

    configuration_hash: str
    consumer_claim_hash: str
    initial_cursor: str
    done_message_id: str
    id_key_hex: str
    hidden_execution_id: str
    canonicalization_version: str
    terminal_tool: TerminalTool
    tasks: List[TaskItem]
    capacity: int = 1
    wait_retry_after_ms: int = 1000
    execution_ordinal: int = 0
    release: ReleasePlan = field(default_factory=lambda: IMMEDIATE)
    assignments: List[Assignment] = field(default_factory=list)
    evaluation_only: bool = False
    blob_root: Optional[str] = None
    schedule_version: str = SCHEDULE_VERSION
    protocol_version: int = PROTOCOL_VERSION


def configuration_hash(start: StreamStart) -> str:
    """Return the immutable hash of everything this generation is.

    A resume presents this value and is refused when it does not match, so what goes in is
    everything a changed value would make the running generation a different one: the two
    versions, the environment's own configuration digest, the capacity, the closed manifest,
    the roster the generation will serve, the release plan, and the terminal tool. The roster is
    the derived one, because a generation started without rows gets the ones its manifest
    implies and a hash over the empty list would not cover them.

    Secrets are covered by their hashes rather than by value. What has to be true is that the
    key changed, not that a reader of this hash can tell what it changed to.

    The blob store's location is not here. Where a run keeps its bytes is deployment, and the
    same generation moved to another directory is the same generation.
    """
    roster = list(start.assignments) or assignments_for(start.tasks, start.release)
    plan = start.release
    declared = {
        "protocol_version": start.protocol_version,
        "schedule_version": start.schedule_version,
        "environment_configuration": start.configuration_hash,
        "consumer_claim_hash": start.consumer_claim_hash,
        "capacity": start.capacity,
        "wait_retry_after_ms": start.wait_retry_after_ms,
        "evaluation_only": start.evaluation_only,
        "canonicalization_version": start.canonicalization_version,
        "execution_ordinal": start.execution_ordinal,
        "initial_cursor": start.initial_cursor,
        "done_message_id": start.done_message_id,
        "id_key_sha256": sha256(start.id_key_hex.encode("utf-8")).hexdigest(),
        "hidden_execution_sha256": sha256(
            start.hidden_execution_id.encode("utf-8")
        ).hexdigest(),
        "terminal_tool": {
            "public_tool_name": start.terminal_tool.public_tool_name,
            "native_terminal_name": start.terminal_tool.native_terminal_name,
            "argument_names": list(start.terminal_tool.argument_names),
        },
        "manifest": [
            {
                "task_position": item.task_position,
                "attempt_id": item.attempt_id,
                "task_message_id": item.task_message_id,
                "ack_message_id": item.ack_message_id,
                "payload_position": item.payload_position,
                "payload_message_id": item.payload_message_id,
                "body_sha256": sha256(item.body.encode("utf-8")).hexdigest(),
            }
            for item in start.tasks
        ],
        "roster": [
            {
                "assignment_id": row.assignment_id,
                "attempt_id": row.attempt_id,
                "task_position": row.task_position,
                "payload_position": row.payload_position,
                "task_message_id": row.task_message_id,
                "ack_message_id": row.ack_message_id,
                "payload_message_id": row.payload_message_id,
                "release_plan_id": row.release_plan_id,
                "creates_payload_obligation": row.creates_payload_obligation,
            }
            for row in roster
        ],
        "release": {
            "release_plan_id": plan.release_plan_id,
            "predicate": plan.predicate,
            "predicate_version": plan.predicate_version,
            "priority": plan.priority,
            "tie_key": plan.tie_key,
            "gates": [
                {
                    "attempt_id": gate.attempt_id,
                    "after_payload_position": gate.after_payload_position,
                    "after_sealed_attempt_id": gate.after_sealed_attempt_id,
                }
                for gate in sorted(plan.gates, key=lambda gate: gate.attempt_id)
            ],
        },
    }
    return sha256(canonical_json(declared)).hexdigest()


def assignments_for(
    tasks: List[TaskItem], release: ReleasePlan, *, without_payload: Sequence[str] = ()
) -> List[Assignment]:
    """Return the roster a closed manifest implies under ``release``.

    The manifest and the roster are different objects and stay that way: the manifest is the
    queue a stream serves from, the roster is the set of rows an analysis counts. They are
    built together so a row cannot name a position the queue does not have.

    ``without_payload`` names the attempts this generation delivers nothing against. Their
    tasks are served, worked, and scored like any other, and no payload obligation is created
    for them, which is what a leg's filler needs and what the release plan cannot express.

    A plan that releases nothing at all leaves every row saying so. The column is what this
    generation does rather than what its manifest could have asked for, so a Never roster reads
    afterwards the way it behaved: no row created a payload, because none did.
    """
    silent = set(without_payload)
    return [
        Assignment(
            assignment_id=assignment_id_for(item.attempt_id),
            attempt_id=item.attempt_id,
            task_position=item.task_position,
            payload_position=item.payload_position,
            task_message_id=item.task_message_id,
            ack_message_id=item.ack_message_id,
            payload_message_id=item.payload_message_id,
            release_plan_id=release.release_plan_id,
            creates_payload_obligation=(
                release.creates_obligations and item.attempt_id not in silent
            ),
        )
        for item in tasks
    ]


@dataclass(frozen=True)
class Writer:
    """Who is calling, in the only terms the generation checks: an epoch and a token.

    Every call that can change the stream carries one. The epoch says which owner is speaking
    and the token proves it is that owner, so a writer whose epoch has been superseded is
    refused before it reads anything, including a writer whose call was already in flight when
    the new owner claimed.
    """

    ownership_epoch: int
    fencing_token: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class OwnershipClaim:
    """A claim to be the generation's writer, presented as a compare and swap.

    ``previous_epoch`` is the epoch the claimant read before claiming. It is the whole of the
    compare: a claimant that read a stale epoch loses, which is what makes two would-be owners
    resolve to one rather than to both. ``fencing_token`` is the new owner's unguessable secret,
    and only its hash is kept. ``configuration_hash`` is what the claimant believes it is
    resuming, and a claim that believes something else changes nothing.

    ``restored_checkpoints`` is what this claimant put back before it claimed: an attempt, and
    the exact task-start checkpoint it materialized for that attempt. An active attempt whose
    world the generation has since authorized a change to is continued only by a claimant that
    says this, because the alternative is a new owner carrying on in a world nobody restored.
    A claimant that restores nothing sends nothing, and is refused for exactly those attempts.
    """

    claimant_id: str
    previous_epoch: int
    fencing_token: str
    configuration_hash: str
    reason: str = "fresh"
    restored_checkpoints: Dict[str, str] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class OwnershipReceipt:
    """What a successful claim returns: the epoch it won, and what it replaced."""

    ownership_epoch: int
    previous_epoch: int
    fencing_token_hash: str
    configuration_hash: str
    claimant_id: str
    reason: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class VerifyBlobsInput:
    """Ask the store whether it holds the exact bytes these references name."""

    blob_root: str
    references: List[str]
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class BlobsVerified:
    """Which references the store could produce, and which it could not.

    A reference is unverified whether the object is absent or the bytes under its name hash to
    something else. The authority treats both the same way, because neither is the object the
    event would have been citing.
    """

    verified: List[str]
    unverified: List[str]
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class ConsumerClaim:
    """A caller's claim to be the generation's one logical consumer."""

    consumer_id: str
    claim_hash: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class ConsumerReceipt:
    """The bound consumer, its claim epoch, and where its cursor starts."""

    consumer_id: str
    claim_epoch: int
    initial_cursor: str
    configuration_hash: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class OfferedMessage:
    """One offered protocol result: the exact bytes, and enough to route them.

    ``visible_text`` is the canonical encoding of a Task, Payload, Wait, Done, SealAck, or
    SealReject. A harness presents those bytes and attests to their hash; it does not rebuild
    them from ``kind`` and ``message_id``.
    """

    message_id: str
    kind: str
    visible_text: str
    attempt_id: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SealRequest:
    """A terminal call, as the harness saw it, with the metadata only the harness can set."""

    metadata: TerminalMetadata
    public_tool_name: str
    native_terminal_name: str
    native_arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnvironmentCall:
    """One ordinary environment call, named so the stream can hold the generation for it.

    The call never reaches the stream and the world it changes is one the stream cannot see.
    What the stream can do is decide whether that call may happen at all, and stay held while
    it does, which is what makes the decision and the change one thing rather than two.
    """

    call_id: str
    attempt_id: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class EnvironmentLease:
    """The stream's permission for one environment call, and what it was granted against.

    ``held`` is the answer to giving one back: true when this call was the one holding the
    generation, false when it had already been taken from it, which is what a caller reading a
    lost answer needs to tell apart.
    """

    call_id: str
    attempt_id: str
    cursor: str
    held: bool
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class QueueClosed:
    """The controller's receipt for a closed queue."""

    task_count: int
    closed: bool = True
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class StreamState:
    """What the harness may ask a live generation about itself.

    This is the answer to a Query, so it writes nothing and can be asked at any time. It is
    harness-only, which is what lets it carry the hidden Wait reasons: a model that could read
    them would learn from a Wait exactly what the Wait record is shaped to withhold.

    Assignment, release, materialization, eligibility, offer, and presentation are six counts
    and not one. Collapsing them would make an offer look like a delivery, and only the last of
    them, and only for a Payload, is one: ``payload_delivery_count`` counts the payloads whose
    bytes were handed to the transport and nothing else. What the model consumed of them is
    attested by the harness transcript rather than by any count here.

    ``ownership_epoch`` is what a resume reads before it claims, so this Query is also the
    compare half of the compare and swap. ``blob_verification`` says whether this generation
    checks the references a presentation carries, which depends on whether it was given a store.

    ``pending_origin`` and ``pending_request_id`` say which call the reserved result is owed to,
    and ``environment_call`` names a hold the generation is still under. An owner that opened
    neither of them knows both from here: the cursor is the one the reserved result was offered
    against, because a reservation and an advanced cursor cannot both be true at once.

    ``prepared_seals`` is the same fact for a filing that has no result yet. A seal accepted by
    an owner that was fenced before it committed leaves its attempt prepared and reserves
    nothing, so the request it is waiting on is named per attempt instead. The filing itself is
    not here: the arguments a model wrote belong in the transcript that holds them, and what a
    replacement cannot rebuild from there is the identity the call was made under.

    ``task_checkpoints`` names the checkpoint each attempt would be restored from, per attempt,
    and ``restoration_required`` names the active attempts a claim may not simply continue:
    ones the generation has authorized a change to a world for since that checkpoint committed.
    A replacement reads both before it claims, because the claim it then makes has to say which
    of them it put back.

    ``graded_evidence`` names the object each committed score was taken out of. The score is a
    number, and a reader that has the run's store can go from it to the verdict behind it: the
    reference is the generation's own, and it is committed with the score rather than left in
    the result of a call nobody kept.
    """

    generation_state: str
    cursor: str
    configuration_hash: str
    stream_state_sha256: str
    ownership_epoch: int
    fencing_token_hash: Optional[str]
    ownership_claims: int
    blob_verification: str
    consumer_id: Optional[str]
    queue_closed: bool
    tasks_remaining: int
    capacity: int
    capacity_in_use: int
    pending_message_id: Optional[str]
    pending_kind: Optional[str]
    pending_origin: Optional[str]
    pending_request_id: Optional[str]
    environment_call: Optional[str]
    prepared_seals: Dict[str, str]
    task_checkpoints: Dict[str, str]
    graded_evidence: Dict[str, str]
    restoration_required: List[str]
    attempts: Dict[str, str]
    obligations: Dict[str, str]
    release_plan_id: str
    release_predicate: str
    assignment_count: int
    materialization_count: int
    eligibility_count: int
    offer_count: int
    presentation_count: int
    payload_delivery_count: int
    wait_count: int
    wait_reasons: Dict[str, int]
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class StreamOutcome:
    """What a finished generation returns: counts, not content.

    ``payloads_delivered`` counts the payloads handed to the transport, which is the fact this
    generation holds. Whether a model read one of them is not something a server can know, so
    the count is named for the delivery and not for the reading.
    """

    generation_state: str
    cursor: str
    sealed: int
    payloads_delivered: int
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SealAttemptInput:
    """Ask the environment for its canonical pre-verdict submission and its local seal.

    ``blob_root`` is where the environment installs the submission bytes it captured, so the
    reference it returns names an object a later event may cite. A generation without a store
    gets the reference anyway and nothing can be read back under it.
    """

    attempt_id: str
    seal_id: str
    native_terminal_name: str
    canonicalization_version: str
    native_arguments: Dict[str, Any] = field(default_factory=dict)
    blob_root: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SealAttemptResult:
    """The immutable submission, captured before any verifier ran.

    ``canonical_submission_text`` is the byte string the digest covers. It is carried inline
    because a kernel submission is small; ``canonical_submission`` names the same bytes by
    hash, which is the form a larger one takes.
    """

    attempt_id: str
    seal_id: str
    canonicalization_version: str
    canonical_submission_text: str
    canonical_submission: BlobRef
    environment_recovery_token: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class GradeAttemptInput:
    """Grade sealed evidence, out of process, against a world nobody can still change.

    ``blob_root`` is where the environment installs the verdict it took, for the reason the
    seal is given one: the reference the result carries is the run's own authoritative record
    of what the score was taken from, and a reference to bytes no store holds is a name nothing
    can resolve. A generation without a store gets the reference anyway and nothing can be read
    back under it.
    """

    attempt_id: str
    seal_id: str
    submission_digest: str
    canonical_submission_text: str
    environment_recovery_token: str
    blob_root: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class GradeAttemptResult:
    """A score and how the submission decoded.

    ``seal_id`` says which seal was graded, and the attempt ID does not: two branches of a fork
    share the public ID, so a score carrying only that could belong to either.

    ``decode_state`` distinguishes a submission that said nothing from one the grader could
    not read, and both are successful results: neither is an infrastructure failure and
    neither is retried.
    """

    attempt_id: str
    seal_id: str
    score: float
    decode_state: str
    evidence: BlobRef
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class GeneratePayloadBundleInput:
    """Build every candidate this obligation might deliver, before the acknowledgement.

    The score is deliberately not an input. A renderer that cannot see the verdict cannot
    leak it, and the restriction is cheaper to keep here than to test for later.
    """

    attempt_id: str
    payload_position: int
    payload_message_id: str
    submission_digest: str
    canonical_submission_text: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class PayloadCandidate:
    """One rendered candidate, with the proof metadata a family gate will need."""

    cell: str
    renderer_id: str
    match_group: str
    body: str
    inner_sha256: str
    visible_sha256: str
    visible_byte_count: int


@dataclass(frozen=True)
class PayloadBundle:
    """The complete candidate set for one obligation. The kernel family has one member.

    ``submission_digest`` names the filing these candidates were rendered from. The attempt ID
    and the position are public and survive a fork, so they say which obligation asked for a
    bundle and not which filing it answers.
    """

    attempt_id: str
    payload_position: int
    submission_digest: str
    candidates: List[PayloadCandidate]
    protocol_version: int = PROTOCOL_VERSION


def hidden_seal_id(hidden_execution_id: str, execution_ordinal: int, attempt_id: str) -> str:
    """Return the environment's idempotency key for sealing one attempt.

    The public attempt ID alone would not do: two branches of a fork share it, so an
    environment keyed on it would treat the second branch's filing as a replay of the first.
    Mixing in the hidden execution ID and the execution ordinal makes the key branch-local
    while leaving the public ID branch-neutral.
    """
    return sha256(
        length_prefixed(b"seal-id-v2")
        + length_prefixed(hidden_execution_id.encode("utf-8"))
        + execution_ordinal.to_bytes(8, "big")
        + length_prefixed(attempt_id.encode("utf-8"))
    ).hexdigest()
