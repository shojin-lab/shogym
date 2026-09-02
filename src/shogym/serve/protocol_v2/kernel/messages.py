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
    AGENT_FILED,
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
from shogym.serve.protocol_v2.policy import (
    LEGACY,
    GradeIdentity,
    MatchedFamily,
    PayloadDisposition,
    PolicyProvenance,
    PublicGrade,
    disposition_key,
    number_text,
)


# Why an attempt ended without a filing. The set is closed and the reasons are declared before a
# generation runs, so what ends an attempt is a property of the configuration rather than
# something a caller composes at the moment it gives up on one.
STEP_CAP = "step_cap"
DEADLINE = "deadline"
ABANDONED = "abandoned"
# The two reasons no caller asks for: they are written where the batch behind an accepted
# terminal cannot go on, and a controller that named either would be reporting something only
# the seal can know. They are told apart because the two say different things about the run. The
# first is the seal's own work failing for good, which is a step that did not happen. The second
# is a step that happened and came back with a result the seal cannot vouch for, which is what a
# reader looks at when the Activities all report success and the attempts still end.
SEAL_FAILED = "seal_failed"
SEAL_UNUSABLE = "seal_unusable"
# The third of those, and a subtype of the second rather than a new ending: the batch came back
# with a candidate built under something other than the policy this obligation was resolved to.
# It ends the attempt the way any unusable result does and is named apart, because a reader
# looking at a run that served the wrong bodies needs to see that rather than a generic result
# the seal could not vouch for.
SEAL_RENDERER = "seal_renderer_mismatch"
FINAL_FAILURE_REASONS = (STEP_CAP, DEADLINE, ABANDONED)
FINAL_FAILURES = FINAL_FAILURE_REASONS + (SEAL_FAILED, SEAL_UNUSABLE, SEAL_RENDERER)


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

    ``attempt_deadline_ms`` is how long an attempt may stay active before the generation ends it
    itself. Zero is off, which is the default: a deadline is a property of the run rather than of
    this kernel, so a generation that declares none has none and waits as long as it is asked to.

    ``profile`` and ``dispositions`` are what this generation delivers and under what. The
    profile is the run's own class rather than the name of whatever function composed it, and
    the dispositions are one resolved row per obligation per branch. Neither has a serving
    default: a generation created now says which profile it is and carries a row for every
    position, and one that says nothing is a history recorded before a policy was a fact about a
    generation, read as the legacy placeholder and never as honest.

    ``provenance`` is what entitles the generation to the profile it claims: the experiment that
    registered its rows, or the platform default it was stamped from, with the digest of the
    exact rows that authority answered for. A profile with no authority behind it is a word a
    caller wrote, so a generation created now carries both or does not start.

    ``families`` are the matched arms this generation's rows are cells of, each declaring the
    group its candidates are built in and the byte count they come to, so a concealed cell and
    an informative one cannot be told apart by their shape.

    ``grade`` is what the environment said its grader is. A generation may resolve an obligation
    to a policy that publishes the score only where that grader is the environment's own, so the
    claim is carried here and checked at start rather than being a property of whichever builder
    composed the generation.
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
    attempt_deadline_ms: int = 0
    execution_ordinal: int = 0
    release: ReleasePlan = field(default_factory=lambda: IMMEDIATE)
    assignments: List[Assignment] = field(default_factory=list)
    evaluation_only: bool = False
    blob_root: Optional[str] = None
    profile: str = LEGACY
    grade: Optional[GradeIdentity] = None
    dispositions: List[PayloadDisposition] = field(default_factory=list)
    provenance: Optional[PolicyProvenance] = None
    families: List[MatchedFamily] = field(default_factory=list)
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

    What a generation delivers is here, and it is folded in only where a generation declares it.
    A history recorded before a policy was a fact about a generation hashed exactly these keys,
    and adding one to what it presents would refuse every resume of it, so the legacy profile
    hashes what it always hashed and a generation that names a profile hashes its dispositions
    along with everything else. Each of those names its policy by the digest of the policy's
    preimage, so what a body was allowed to say is inside the identity a resume is held to,
    along with the authority that decided it and the matched families its rows are cells of.

    The grader is here whole, every field of it. Which grader, which version, whether its number
    is the environment's own, which measure the headline is, how fine that measure is, and what
    it may publish beside it are each a fact the record depends on, and a claimant composed over
    another of them is composing a different generation. The resolution is one of them for the
    reason the components' is: a run recorded under one headline precision and resumed by a
    process composed for another would take the generation over and disagree with it afterwards,
    at the first seal, rather than being refused before it owned anything.
    """
    roster = list(start.assignments) or assignments_for(start.tasks, start.release)
    plan = start.release
    declared: Dict[str, Any] = {
        "protocol_version": start.protocol_version,
        "schedule_version": start.schedule_version,
        "environment_configuration": start.configuration_hash,
        "consumer_claim_hash": start.consumer_claim_hash,
        "capacity": start.capacity,
        "wait_retry_after_ms": start.wait_retry_after_ms,
        "attempt_deadline_ms": start.attempt_deadline_ms,
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
    if start.profile != LEGACY:
        declared["profile"] = start.profile
        declared["grade"] = (
            None
            if start.grade is None
            else {
                "grader_id": start.grade.grader_id,
                "grader_version": start.grade.grader_version,
                "stand_in": start.grade.stand_in,
                "score_component": start.grade.score_component,
                "score_places": start.grade.score_places,
                "public_components": [
                    {
                        "name": number.name,
                        "minimum": number_text(number.minimum),
                        "maximum": number_text(number.maximum),
                        "places": number.places,
                    }
                    for number in start.grade.public_components
                ],
            }
        )
        declared["dispositions"] = [
            {
                "attempt_id": row.attempt_id,
                "payload_position": row.payload_position,
                "branch_slot": row.branch_slot,
                "kind": row.kind,
                "policy_digest": row.policy_digest,
                "cell": row.cell,
                "reason": row.reason,
                "resolution_source": row.resolution_source,
                "family_id": row.family_id,
            }
            for row in sorted(start.dispositions, key=disposition_key)
        ]
        declared["provenance"] = (
            None
            if start.provenance is None
            else {
                "authority": start.provenance.authority,
                "roster_digest": start.provenance.roster_digest,
                "experiment_id": start.provenance.experiment_id,
                "descriptor_digest": start.provenance.descriptor_digest,
            }
        )
        declared["families"] = [
            {
                "family_id": family.family_id,
                "match_group": family.match_group,
                "cells": [list(cell) for cell in family.cells],
                "visible_byte_count": family.visible_byte_count,
            }
            for family in sorted(start.families, key=lambda family: family.family_id)
        ]
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
    """A terminal call, as the harness saw it, with the metadata only the harness can set.

    ``terminal_source`` is who made it. The stream does not care which of the two filed: the
    seal, the grade and the acknowledgement are the same either way. What it does is write the
    answer down, because a reader counting how a generation's attempts ended is asking a
    question the score cannot answer, and an ending the agent chose and one its own step budget
    forced on it are two different things to count.
    """

    metadata: TerminalMetadata
    public_tool_name: str
    native_terminal_name: str
    native_arguments: Dict[str, Any] = field(default_factory=dict)
    terminal_source: str = AGENT_FILED


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
class FinalizeRequest:
    """A controller's request to end one attempt that nothing is going to finish.

    ``reason`` comes from the closed set above. With the attempt it is the whole of what the
    caller supplies: there is no score here and no message, because a finalization writes the
    floor and produces nothing the model can read.

    ``request_id`` is this call's logical identity, and it is here for the same reason a pull
    carries one. A controller that loses the answer retries the request it made, and the retry
    has to reach the answer it already has rather than a second ending or a stale refusal.
    """

    request_id: str
    attempt_id: str
    reason: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class AttemptFinalized:
    """The controller's receipt for an ended attempt: why, at what score, and what came back.

    ``capacity_in_use`` is read after the release, so the receipt says the capacity is free
    rather than promising that it will be.

    ``also_finalized`` names every other attempt the same transition floored: an attempt whose
    gate waits on a fact this ending made impossible has no way left to run, so it is ended
    here rather than left waiting for it. They are in the receipt because the floor is one
    atomic fact about several attempts, and a receipt naming only the one that was asked about
    would say less than happened.
    """

    attempt_id: str
    reason: str
    score: float
    capacity_in_use: int
    obligation_state: Optional[str] = None
    also_finalized: List[str] = field(default_factory=list)
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

    ``environment_calls`` is how many calls to a world this generation has authorized against
    each attempt since that attempt's task was presented or its checkpoint was restored. The
    calls themselves never reach the stream, so a grant is the only trace of one, and this is
    what a transport enforcing an environment's step budget is counting. A transport that kept
    none of its predecessor's memory reads the spent budget here rather than starting the
    attempt again at nothing. ``restoration_required`` is the same fact asked as a question
    about a claim.

    ``task_checkpoints`` names the checkpoint each attempt would be restored from, per attempt,
    and ``restoration_required`` names the active attempts a claim may not simply continue:
    ones the generation has authorized a change to a world for since that checkpoint committed.
    A replacement reads both before it claims, because the claim it then makes has to say which
    of them it put back.

    ``graded_evidence`` names the object each committed score was taken out of. The score is a
    number, and a reader that has the run's store can go from it to the verdict behind it: the
    reference is the generation's own, and it is committed with the score rather than left in
    the result of a call nobody kept.

    ``final_failures`` says why each ended attempt ended. It is here and nowhere else: a model
    that could read it would learn from an ending what the protocol gives it no message about.

    ``deadline_expired`` names the attempts whose deadline has passed and which the generation
    has not ended yet. There is one reason to be in it: the attempt is holding a call to a world
    this stream cannot see, so the ending is waiting for that call rather than cancelling an
    effect nothing here can observe. It is the operator's signal that a call is not coming back,
    and the takeover that ends the grant by name is what releases it.

    ``dispositions`` says what this generation delivers against each obligation and under what,
    one entry per obligation per branch. It is here rather than derivable, because what a run
    told its agent is a fact about the run and an operator reading a live generation should not
    have to infer it from which bodies happened to come out.

    ``profile`` and ``experiment_id`` are which kind of run this is and, where it is an
    experiment, which experiment registered it. They travel with the dispositions because the
    two questions an analysis asks of a body are what it said and who decided it would: a
    stamped honest body and a registered honest cell are the same delivery and different facts.
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
    environment_calls: Dict[str, int]
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
    final_failures: Dict[str, str]
    deadline_expired: List[str] = field(default_factory=list)
    dispositions: Dict[str, str] = field(default_factory=dict)
    profile: str = LEGACY
    experiment_id: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt as a record: what it was assigned, what it filed, what it scored.

    This is the row an analysis counts. It is harness-only: the acknowledgement commits to what
    was filed and says nothing about how good it was, and this row carries the assignment, the
    ending, the disposition and the score together, which is more than any policy publishes.
    What the agent is told is the payload, and the policy this row names is what decided it.

    The positions are the assignment's, so a row says where the attempt sat in its manifest
    rather than what order a stream happened to serve. The three message identifiers are the
    ones the manifest preallocated, and each is reported as delivered rather than as offered,
    because an offer is a reservation and a delivery is the exact bytes handed to the transport
    that carries them. What the model consumed of those bytes is attested by the harness
    transcript and not by this row.

    ``score`` is ``None`` until the seal that made it authoritative, and it stays ``None``
    rather than becoming a zero: an attempt nobody has sealed has no score, which is not the
    same fact as an attempt that scored nothing.

    An attempt that ended without a filing does have one, and it is the floor. That is a third
    fact again, and the number alone cannot say so: a floored attempt and one the environment
    graded at nothing carry the same zero. ``final_failure`` is what tells them apart, and it
    says which ending it was, because a run out of steps, a run out of time, a gate that can no
    longer open and a seal whose batch could not go on are four different things to count.
    ``deadline_expired`` is the clock on its own: an expiry the generation has recorded and not
    yet acted on, which is what a row carries while the attempt is still holding a call to a
    world nothing here can see.

    ``terminal_source`` is who filed, for the attempts that were filed at all. A generation over
    an environment whose horizon is a graded ending seals an attempt that spent its budget
    without waiting for the agent to file, so a sealed row is no longer proof the agent chose to
    end there. The two are separated here rather than left to be guessed from a step count, and
    an attempt nobody filed carries neither.

    ``creates_payload_obligation`` is the roster's own column, and ``payload_state`` is what
    became of the obligation it promised. Without them ``payload_delivered`` being false says
    two different things at once: a row this generation owed a payload and never delivered, and
    a row that was never going to have one, which is what a filler is. Those are a missed
    treatment and a structural absence, and an analysis that cannot tell them apart is counting
    fillers as failures. A row that creates no obligation has no obligation state.

    ``payload_policy`` and ``payload_disposition`` are what this row was told and under what
    rules. A generation recorded before a policy was a fact about one reads as the legacy
    placeholder rather than as honest, because absence is a history that predates the question
    and never an answer to it.

    ``profile`` and ``payload_resolution_source`` are who decided that. An analysis counting
    these rows is asking which of two mistakes a run made, if either: an experiment cell served
    as an ordinary default, or an ordinary run blinded by something nobody registered. Neither
    question can be answered from the policy name alone, because the same name is a correct
    answer to both.
    """

    attempt_id: str
    task_position: int
    payload_position: int
    state: str
    terminal_tool: Optional[str]
    terminal_source: Optional[str]
    canonicalization_version: str
    submission_digest: Optional[str]
    score: Optional[float]
    decode_state: Optional[str]
    seal_ordinal: Optional[int]
    final_failure: Optional[str]
    deadline_expired: bool
    task_message_id: str
    task_delivered: bool
    ack_message_id: str
    ack_delivered: bool
    payload_message_id: str
    payload_delivered: bool
    creates_payload_obligation: bool
    payload_state: Optional[str]
    payload_policy: Optional[str]
    payload_disposition: Optional[str]
    profile: str = LEGACY
    payload_resolution_source: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class PresentedMessage:
    """One message this generation committed to deliver, in the order it committed them.

    A row here is a commitment and not a handoff. What the generation verified is that the
    digest a transport claimed matched the bytes it was holding out, and what it then did is
    accept that claim and advance its cursor past the message. All of that happens before the
    transport has a result to hand anybody: an acknowledgement that never gets back to the call
    that asked for it leaves a message this generation has counted and no bytes anywhere, and a
    generation whose owner is replaced while such a result is owed never hands those bytes over
    at all.

    So the reconciliation is what establishes both of the things this cannot. Whether the bytes
    reached the transport, and whether a model then read them, are facts about the harness, and
    the harness's own transcript is where a claim about either has to be checked against these
    rows. A reader holding these rows alone holds an account of what was committed.

    An attempt's record says which of its three messages were committed, and that is the fact an
    analysis counts. This says which bytes each of them was, and it says it for every kind rather
    than for those three: a Wait, a SealReject and the Done that ends the generation are
    committed under the same attestation, and a harness reconciling its own transcript against
    what was committed would report a run as whole while its own record was missing one.

    ``visible_bytes_sha256`` is the digest the presentation was verified against. It is the whole
    of what a reconciliation needs and less than the bytes themselves: it does not say what the
    bytes were, and whoever compares already holds what the harness wrote down, so what is being
    asked is only whether the two are the same.
    """

    order: int
    kind: str
    message_id: str
    attempt_id: Optional[str]
    visible_bytes_sha256: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class GenerationRecords:
    """One generation's attempts and its commitments, read out of the same moment.

    The two are answered together because a generation that is still serving moves between two
    questions. Asked separately, a payload can be owed in the first answer and committed in the
    second, and the pair describes a generation that never existed in either state. Asked here,
    both halves are read off the one projection, so a row and the commitments beside it are the
    same run at the same point.
    """

    attempts: List[AttemptRecord]
    presentations: List[PresentedMessage]
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
    finalized: int = 0
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

    ``grade`` is the grader saying which grader it is. A generation is built over a declared
    grade identity and its honest bodies publish that grader's number, so the number arriving
    here carries the same identity or the generation is publishing one grader's verdict under
    another's name. The kernel's grade computes from the shape of a filing, reaches no world, and
    says so, which is what stops a transport fixture from being printed as a verdict. It is
    absent by default because a result recorded before graders said this is one nobody can now
    ask, and an absent one is read as a stand-in rather than as an environment's own.

    ``public_components`` is what the environment published for the agent beside the score, as
    numbers under token names. It is not the verdict and not the evidence: those stay in the
    reference this result carries, which a harness can resolve and a renderer cannot. What names
    may appear is the roster in the grade identity, declared before the run rather than taken
    from whatever the grader returned.

    Every field arrives as whatever the grader put in it. They are declared that way on purpose,
    and the reason is the same for the identity and the reference as it is for the numbers: a
    field with a type is a field the decoder has to make that type of, and a grader that returned
    a string, a list or an object under one would fail the decoding rather than the check. That
    failure is not an ending. It happens while the generation is being handed the result, before
    any code of its own runs, so the generation fails that step again on every retry, records
    nothing about why, and answers no question while it does. So the wire shape is permissive and
    the authority is strict: what a score, a name, an identity and a reference are gets decided
    where the result is read, and a value that is none of them ends the attempt with a reason.
    """

    attempt_id: Any
    seal_id: Any
    score: Any
    decode_state: Any
    evidence: Any
    grade: Any = None
    public_components: Any = field(default_factory=dict)
    protocol_version: Any = PROTOCOL_VERSION


@dataclass(frozen=True)
class GeneratePayloadBundleInput:
    """Build every candidate this obligation might deliver, before the acknowledgement.

    ``policy_digest`` is what this obligation was resolved to, and it decides which renderer
    runs. A digest this build does not implement is a failure rather than a body: there is no
    renderer to fall back to, because falling back is how a run comes to serve something other
    than what its record says it served. An empty digest is a request from a history recorded
    before policies existed and renders the placeholder it recorded.

    ``public_grade`` is the whole of what a renderer may know about the verdict, and it is
    present only where the resolved policy publishes it. A blinded renderer is not given a grade
    to withhold: it is handed a request with no grade in it, so leaking one is not a discipline
    it keeps but a value it does not have.
    """

    attempt_id: str
    payload_position: int
    payload_message_id: str
    submission_digest: str
    canonical_submission_text: str
    policy_digest: str = ""
    cell: str = ""
    public_grade: Optional[PublicGrade] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class PayloadCandidate:
    """One rendered candidate, with the proof metadata a family gate will need.

    ``policy_digest`` and ``renderer_version`` are echoed from the request rather than being the
    renderer's own opinion of itself. What they catch is a Worker running code the generation did
    not ask for: a build that never learned about policies returns neither, and the seal that
    asked for one refuses the candidate instead of serving whatever came back under an honest
    label. Both are empty where the request carried no policy, which is what a replayed legacy
    result looks like.
    """

    cell: str
    renderer_id: str
    match_group: str
    body: str
    inner_sha256: str
    visible_sha256: str
    visible_byte_count: int
    renderer_version: str = ""
    policy_digest: str = ""


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


def finalize_request_identity(request: FinalizeRequest) -> str:
    """Return the canonical identity of a finalization request.

    It covers the whole request and not the fields the ending happens to read, the declared
    protocol version among them. A retry carrying the same request ID and the same identity
    replays its receipt; one carrying the same ID and anything else is a conflict, which is the
    point of hashing the request rather than trusting the ID beside a few of its fields.
    """
    return sha256(
        length_prefixed(b"finalize-request-v2")
        + length_prefixed(
            canonical_json(
                {
                    "request_id": request.request_id,
                    "attempt_id": request.attempt_id,
                    "reason": request.reason,
                    "protocol_version": request.protocol_version,
                }
            )
        )
    ).hexdigest()


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
