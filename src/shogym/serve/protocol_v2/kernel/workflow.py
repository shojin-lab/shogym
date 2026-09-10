"""One stream generation, as a durable workflow.

The workflow is the authority. There is no second event store: what the stream knows is the
workflow's state, and what happened to it is the history that state replays from. Every fact
the protocol calls an event is a transition here, and the ones the protocol calls one atomic
batch are written in one stretch of code with no await in it, because a suspension point is
the only place a reader could see half of it.

The shape follows from two facts about the protocol. First, at most one call may be changing
the stream at a time, so every stream-affecting handler takes the lock synchronously, before
its first await, and a call that finds it taken is refused rather than queued. Second, an
offer is a reservation and a presentation is an observation, so an offered result sits in one
pending slot until the harness attests to the exact bytes, and until then a retry of the
originating request gets those bytes back and anything else gets an error.

Replay is what brings the state back after a worker dies, and it is not what makes the resumed
stream safe. One writer owns the generation at a time, under an epoch that only a compare and
swap moves and a token only that owner holds. Every call that can change the stream presents
both, at entry and again after every await, so the writer a resume replaced is refused rather
than raced, including the call it left in flight.

An attempt is state in this workflow rather than a workflow of its own. That is deliberate: a
seal has to make the result, the score, the candidate bundle, the schedule transition, the
released capacity, and the acknowledgement authoritative together, and Temporal has no
transaction spanning two histories.

Not every attempt is ended by a filing. A controller may end one that nothing is going to
finish, and the generation may end one whose deadline has passed. Either way the attempt fails
finally: capacity comes back, the payload it was owed is resolved without being rendered, the
assigned outcome is written at the floor, and no acknowledgement is minted, because an
acknowledgement is a fact about a submission and there is no submission here.

The refusals are protocol errors, raised as :class:`StreamProtocolError`. They carry a code
from the closed set and nothing else, they change no state, and they are never a message.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field as dataclass_field, fields, replace
from datetime import timedelta
from hashlib import sha256
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.api.common.v1 import Payload as WirePayload
from temporalio.exceptions import ActivityError, ApplicationError, FailureError
from temporalio.exceptions import TimeoutError as ActivityTimeout

with workflow.unsafe.imports_passed_through():
    from shogym.serve.protocol_v2 import (
        PROTOCOL_VERSION,
        RELEASE_AT_SEAL,
        SCHEDULE_VERSION,
        TERMINAL_SOURCES,
        Assignment,
        BlobRef,
        Done,
        Info,
        InfoRequest,
        Payload,
        PresentationAck,
        PresentationCommit,
        ProtocolError,
        PullRequest,
        ScheduleView,
        SealAck,
        SealReject,
        Task,
        Wait,
        WireFormatError,
        canonical_bytes,
        canonical_json,
        check_release,
        eligible_tasks,
        info_request_identity,
        order_key,
        presentation_request_identity,
        pull_request_identity,
        require_declaration,
        require_digest,
        require_opaque_id,
        require_step_budget,
        stream_message_id,
        submission_digest,
        terminal_request_identity,
        visible_bytes,
    )
    from shogym.serve.protocol_v2.artifact import (
        ELIGIBLE_CELLS,
        ReceiptContract,
        SourceArtifactManifest,
        check_receipt_contracts,
        manifest_fields,
        mask_spans,
        masked_body,
        payload_wire_count,
        read_source_artifact,
        source_commitment,
    )
    from shogym.serve.protocol_v2.kernel.activities import (
        EXPIRED_AUTHORITY_FAILURE,
        GENERATE_PAYLOAD_BUNDLE,
        ORIGIN_DISAGREEMENT_FAILURE,
        fork_availability_activity,
        generate_payload_bundle_activity,
        grade_attempt_activity,
        seal_attempt_activity,
        start_fork_child_activity,
        verify_blobs_activity,
        verify_fork_origin_activity,
    )
    from shogym.serve.protocol_v2.kernel.messages import (
        ABANDONED,
        CARRIER_SCHEMA_VERSIONS,
        CONFIRMED_EXISTING,
        CONTINUED_FIRST_DELIVERY,
        CONTRACT_DRIFT,
        CORRUPT_EVIDENCE,
        DEADLINE,
        EXISTENCE_UNCONFIRMED,
        EXPIRED_PREPARATION,
        FINAL_FAILURE_REASONS,
        FORK_ABANDONED,
        FORK_AVAILABILITY_STEP,
        FORK_CHILDREN_CONFIRMED,
        FORK_COMPLETE,
        FORK_CONFIGURATION_VIOLATION,
        FORK_CONFLICTED,
        FORK_EXPIRED_AUTHORITY,
        FORK_IN_FLIGHT,
        FORK_NOT_QUIET,
        FORK_MOVED_EXECUTION,
        FORK_ORIGIN_DISAGREEMENT,
        FORK_ORIGIN_STEP,
        FORK_PREPARED,
        FORK_REPAIRABLE_ABSENCE,
        FORK_REQUEST_CONFLICT,
        FORK_START_STEP,
        FORK_UNDECLARED_BRANCH,
        FORK_UNRECOVERABLE_EVIDENCE,
        FORK_WITNESS_MISMATCH,
        RETRYABLE_FORK_REFUSALS,
        SPENT_RECOVERY_RESERVE,
        OPERATION_OUTCOMES,
        OPERATION_PHASES,
        OPERATION_REASONS,
        OWNERSHIP_CLAIM,
        PAYLOAD_OFFER,
        RECOVERED_OPERATION,
        REFUSED_OPERATION,
        SEAL_FAILED,
        SEAL_RENDERER,
        SEAL_UNUSABLE,
        SOURCE_PUBLICATION,
        UNAVAILABLE_EVIDENCE,
        UNRECOVERABLE_OPERATION,
        WRONG_SOURCE,
        AttemptFinalized,
        AttemptRecord,
        AnsweredUpdate,
        CarriedAttempt,
        CarriedAttestation,
        CarriedBinding,
        CarriedFinalization,
        CarriedObligation,
        ConsumerClaim,
        ConsumerReceipt,
        EnvironmentCall,
        EnvironmentLease,
        FinalizeRequest,
        finalize_request_identity,
        ForkAvailability,
        ForkAvailabilityInput,
        ForkChildPlan,
        ForkChildReceipt,
        ForkChildStarted,
        ForkOrigin,
        ForkOriginVerified,
        ForkReceipt,
        ForkRequest,
        ForkStatusAnswer,
        ForkStatusQuestion,
        GeneratePayloadBundleInput,
        GenerationRecords,
        GradeAttemptInput,
        GradeAttemptResult,
        OfferedMessage,
        OperationFailure,
        OwnershipClaim,
        OwnershipReceipt,
        PayloadCandidate,
        PayloadCandidateResult,
        PreparedChild,
        PreparedFork,
        PresentedMessage,
        QueueClosed,
        SealAttemptInput,
        SealRequest,
        SelectedSourceReference,
        SourceOriginContext,
        SourceProvenance,
        StreamCarry,
        StreamOutcome,
        StreamStart,
        StartForkChildInput,
        StreamState,
        TaskItem,
        VerifyBlobsInput,
        VerifyForkOriginInput,
        Writer,
        assignments_for,
        carrier_version,
        check_child_configuration,
        check_complete_start_authorization,
        check_fork_origin,
        check_fork_request,
        check_origin_reply_within_bound,
        check_origin_within_bound,
        check_receipt_within_bound,
        check_start_within_bound,
        check_transmitted_size,
        canonical_location,
        child_blob_root,
        child_workflow_id,
        complete_start_digest,
        configuration_hash,
        continuation_argument,
        derived_selection,
        fork_activity_id,
        fork_origin_bound,
        fork_receipt_bound,
        fork_start_bound,
        fork_request_digest,
        hidden_seal_id,
        origin_digest,
        origin_fields,
        ownership_claim_operation_identity,
        read_source_origin,
        resolved_echo,
        source_seal_id,
        start_difference_projection,
        CarriedProjection,
        pack_carrier,
        unpack_carrier,
    )
    from shogym.serve.protocol_v2.policy import (
        ARTIFACT,
        DELIVER,
        HONEST,
        KERNEL_STAND_IN_GRADE,
        LEGACY,
        POLICIES,
        PROFILES,
        GradeIdentity,
        MatchedFamily,
        PayloadDisposition,
        PayloadPolicy,
        PolicyViolation,
        PublicGrade,
        PublishedNumber,
        check_dispositions,
        check_grade_result,
        describe_disposition,
        descriptor_digests,
        disposition_key,
        policy_name_of,
        published_grade,
        render_body,
        roster_digest,
    )
    from shogym.serve.protocol_v2.schedule import PAYLOAD, TASK

PLANNED = "planned"
TASK_OFFERED = "task_offered"
ACTIVE = "active"
SEALING = "sealing"
SEALED = "sealed"
ACK_PRESENTED = "ack_presented"
FINAL_FAILED = "final_failed"

ASSIGNED = "assigned"
MATERIALIZED = "materialized"
ELIGIBLE = "eligible"
OFFERED = "offered"
PRESENTED = "presented"

# The states Done has to look through. An attempt that has not reached a presented
# acknowledgement is still live, and an obligation that has not been presented is still owed.
# A final failure is neither: it is terminal and obligation fulfilling on both sides, so it is
# in neither tuple and Done reads it as resolved.
LIVE_ATTEMPT = (PLANNED, TASK_OFFERED, ACTIVE, SEALING, SEALED)
UNFULFILLED_OBLIGATION = (ASSIGNED, MATERIALIZED, ELIGIBLE, OFFERED)

# The states an obligation keeps its candidate in, which are the states an offer of it could
# still carry one. They are what the carrier writes the bytes for and what a restore requires
# them in: an obligation that has been presented, or that ended without being rendered, drops the
# body and will never be offered again, and one that has not been materialized has none yet.
_OFFERABLE_OBLIGATION = (MATERIALIZED, ELIGIBLE, OFFERED)

# The analysis outcome a finalized attempt is assigned. An attempt that was ended rather than
# filed has nothing to grade, and the floor is what the outcome is fixed at instead.
FLOOR = 0.0

OPEN = "open"
DONE = "done"
# And the third, which is a parent that has forked. It is terminal like Done and it is not Done: a
# generation that presented Done told its agent its work was over, and a generation that was forked
# told its agent nothing at all.
FORKED = "forked"

_ACTIVITY_TIMEOUT = timedelta(seconds=60)
# What the seal and the grade are given instead. They are the two Activities that reach an
# environment: stopping a world, copying what it persisted and running a grader over it are each
# bounded by the environment rather than by this stream, and those bounds are minutes. Sixty
# seconds is a cap the work would not fit under, and an Activity retried while its first attempt
# is still running is the case a finalizer key has to survive rather than the case to arrange.
_TERMINAL_ACTIVITY_TIMEOUT = timedelta(minutes=20)
_ACTIVITY_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=10),
    maximum_attempts=3,
)
# What a gated child's origin question is retried under, and why it has no attempt limit. An
# unavailable parent, an unavailable Worker, an unreadable history and a closed answer window are
# all infrastructure, and the child stays gated while they last: it owns nothing, serves nothing
# and costs nothing while it waits. An authenticated disagreement is not retried at all, because
# the answer that carries it is non-retryable and the row came from the parent's own record.
_ORIGIN_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=1),
    backoff_coefficient=2.0,
    maximum_interval=timedelta(seconds=60),
)

# How much of an Activity's failure a row keeps, in bytes, and the mark left where the cap cut a
# value short. The cap is one size for every text among them, because what is being bounded is a
# row rather than a particular field, and no name a service reports comes anywhere near it.
_FAILURE_TEXT_BYTES = 512
_CUT = " [truncated]"

# What the durable service does to bound one execution, and what this generation does about it.
#
# The service counts the Updates one execution accepted and refuses the next one past its cap.
# A roster longer than that cap needs more than one execution, so the generation continues as
# new at a quiet point well before the cap is reached, under the same workflow ID, carrying the
# whole logical projection. The five numbers below are that decision, and the inequality under
# them is what makes it safe rather than hopeful.
#
# SERVICE_UPDATE_CAP is the smallest cap a deployment this package supports may configure.
# TURNOVER_TRIGGER is where this generation latches, counted by its own counter rather than by
# the service's. ADMISSION_LEAD is the most the service can have admitted that this counter has
# not seen yet. ADMISSION_RESERVE is the most this generation will accept after the latch, which
# the admission gate enforces rather than infers. TURNOVER_MARGIN is what is left over.
#
# The reserve is a bound on the work that can still make the boundary quiet, and the whitelist
# below is exactly that work: the pending message's presentation commit (1), the held grant's
# end (1), a prepared seal's retry and the presentation of its acknowledgement (2), and a claim
# with the confirmation that follows it, which is how an abandoned grant is recovered (2), and
# closing the queue (1). Seven per latch, and the reserve is many times that, which is what
# leaves room for a claim that is refused and made again.
SERVICE_UPDATE_CAP = 2000
TURNOVER_TRIGGER = 1700
ADMISSION_LEAD = 10
ADMISSION_RESERVE = 40
TURNOVER_MARGIN = 200

# How much of the reserve the operations that can be repeated may spend, and why they do not all
# share one allowance.
#
# A confirmation mints a fresh identifier every time by design, and a consumer binding or a close
# of the queue can be sent again under another. A ceiling on the total alone would let those take
# every slot and leave none for the end of a held grant, which nothing but the caller holding it
# can send. So they have an allowance of their own.
#
# An ownership claim is repeatable in the same way, and it was given that same allowance, which
# was wrong in a way worth naming. A claim is the only thing that can recover a grant whose owner
# went away: the world is held, the owner that held it is gone, and nothing but a new owner can
# end the grant by name. Sharing the allowance with confirmations meant that spending it on
# confirmations locked out the one call that could clear the hold. It has an allowance of its own
# now, so a recovery claim is admissible whatever else has been sent.
ADMISSION_REPEATS = 8
ADMISSION_CLAIMS = 4
assert ADMISSION_REPEATS + ADMISSION_CLAIMS < ADMISSION_RESERVE, (
    "the operations that can be repeated must not be able to spend the whole reserve, or the "
    "one call that can clear a named hold would be refused for the want of a slot"
)
assert (
    TURNOVER_TRIGGER + ADMISSION_LEAD + ADMISSION_RESERVE + TURNOVER_MARGIN < SERVICE_UPDATE_CAP
), "the trigger, the admission lead, the reserve and the margin have to fit under the cap"

# What a parent admits once it has committed a fork barrier, and why it is a number of its own.
#
# The fork reserve replaces the post-latch reserve rather than adding to it: a generation that has
# latched for a turnover refuses a fork, and one that has committed a barrier never turns over, so
# the two allowances are never both in force. Twenty four leaves the same inequality standing under
# the smallest supported cap.
#
# All of them are the fork's own class, which is what the reserve is kept for. Nothing else can
# take one: status is read by Query, which charges nothing, and the barrier's whitelist admits no
# other writing class at all, so repeated reads and repeated recovery work cannot lock out the one
# call that can finish the fork. A build that widens that whitelist owes this class an allowance of
# its own inside this number.
FORK_RESERVE = 24
assert (
    TURNOVER_TRIGGER + ADMISSION_LEAD + FORK_RESERVE + TURNOVER_MARGIN < SERVICE_UPDATE_CAP
), "the trigger, the admission lead, the fork reserve and the margin have to fit under the cap"

# How long the parent's own fork work has to finish, and how long the answers it owes stand.
#
# The bound is a deadline on the parent's own work and never a bound on when it actually closes,
# because the SDK runs a durable timer's callback only inside a Worker activation: a parent whose
# Worker is away commits nothing when its deadline becomes due, and a Worker that comes back after
# it records the expiry against the recorded due time and restarts neither clock.
#
# The horizon is recorded at the barrier because the close cannot be bounded and the deletion clock
# is not the close clock either: the service schedules each execution's history for deletion from
# that execution's own close time, so a child that failed an hour after the barrier is deleted on
# its own clock while the parent is still prepared. The window is the earlier of the horizon and
# the parent's actual close plus the answer window, which is why a late close shortens the recovery
# window rather than extending the retention a deployment owes.
FORK_PREPARATION_BOUND_MS = 12 * 60 * 60 * 1000
FORK_ANSWER_WINDOW_MS = 24 * 60 * 60 * 1000
FORK_AUTHORITY_HORIZON_MS = FORK_PREPARATION_BOUND_MS + FORK_ANSWER_WINDOW_MS

# What one child start Activity is bounded by. The SDK defines this as covering the scheduling and
# every retry rather than one run, so the last in-flight outcome settles at a time the parent
# declared rather than at one an unavailable service chooses. A timeout is never proof that no
# child exists: the child it named moves into the attempted-unconfirmed class and never into never
# attempted or into proven absent.
_FORK_START_TIMEOUT = timedelta(hours=1)

# The endings a fork never moves out of once it stands at one, and every state its parent stops
# waiting at. An abandonment is the ending a spent reserve or an expired bound commits, and a
# conflict is the ending a permanent decision reached after the barrier: both retain every child
# the fork made, and neither is ever rewritten by a later status. A parent parks until the fork is
# at one of those or has finished, because there is nothing left for it to wait for at any of them.
_FORK_ENDINGS = (FORK_ABANDONED, FORK_CONFLICTED)
_FORK_SETTLED = (FORK_COMPLETE, *_FORK_ENDINGS)

# The refusal an arriving Update is rejected with once the latch is set. It is not a protocol
# error the agent is ever shown: the transport waits for the boundary and sends the same request,
# under the same Update ID, at the execution that comes next.
TURNOVER_PENDING = "turnover_pending"

# The refusal a call that would own or serve a child is rejected with while that child has still
# to authorize the lineage it carries. It is not a protocol error the agent is ever shown: nothing
# the caller sent is wrong, and the verification the child owes is its own first work, so the same
# call reaches a generation that has done it.
ORIGIN_UNVERIFIED = "origin_unverified"

# What a parked parent rejects an Update with once its barrier stands. It is not a protocol code
# the agent is ever shown, and neither is a fork refusal, which names its own reason as its type
# rather than sharing one.
FORK_BARRIER = "fork_barrier"

# Why a generation gave up on a boundary it had decided to take. Both are recorded where a
# launcher can read them, because both leave the generation serving out the execution it is in
# and therefore able to reach the service cap the boundary existed to avoid.
#
# CARRIER_TOO_LARGE is the profile being outside what this package supports: the projection would
# not fit in one payload. ADMISSION_EXHAUSTED is the other way a boundary becomes unreachable,
# and it is not about size at all. The work that clears a boundary can fail and be tried again:
# the transport builds a fresh attestation after a presentation is refused, and every failed
# attempt is an accepted Update. Once the reserve is spent, nothing more is admitted, and if the
# generation is not already quiet it never will be: the thing it is holding cannot be cleared,
# and the boundary cannot be taken. Rather than hold traffic back for a boundary that can no
# longer happen, the generation says so and goes back to serving.
CARRIER_TOO_LARGE = "carrier_too_large"
ADMISSION_EXHAUSTED = "admission_exhausted"
RECOVERY_CLAIMS_EXHAUSTED = "recovery_claims_exhausted"

# The service's own limit on one payload, which is what the replaced start has to fit under.
SERVICE_PAYLOAD_LIMIT_BYTES = 2 * 1024 * 1024

# The most the replaced start may encode to before this generation refuses to hand itself on. A
# generation over it keeps serving and records the size it measured, which is what a launcher
# reads to say why the run stopped where it did.
#
# The number is set from a measurement rather than from a feeling. The supported profile is the
# two hundred task AutomationBench roster, whose start encodes to about 600400 bytes with bodies
# of 1625 to 2760 bytes, driven to Done under the declared workload: fifteen world calls a task,
# a lost reply and the confirmation its recovery costs, identifiers shaped the way the transport
# mints them, and every boundary the generation decided on along the way. What is measured is the
# largest quiet point that roster has, which is the boundary after the last payload of the last
# task is presented; the recorded number is beside the test that checks this ceiling still leaves
# it the room it was chosen to leave. The ceiling sits 197152 bytes under the service's limit,
# which is room for the command envelope around a payload whose own bytes this measures exactly,
# and the measured profile sits well over a quarter under the ceiling, which is room for a roster
# that grows or bodies that do.
#
# Both numbers matter in different directions. Too high and the service refuses the continuation
# the generation was counting on. Too low and the generation refuses its own boundary and wedges
# at the cap, which is the failure this whole change exists to remove, so the margin above the
# measured profile is the one to keep honest as the profile moves.
TURNOVER_PAYLOAD_CEILING_BYTES = 1_900_000
assert TURNOVER_PAYLOAD_CEILING_BYTES < SERVICE_PAYLOAD_LIMIT_BYTES, (
    "a ceiling at or above the service's own limit would let the generation submit a "
    "continuation the service refuses"
)

# The first Activity ID, and the reason there is one. The SDK numbers Activities per execution
# from one, so a generation that names its own IDs from a carried ordinal produces exactly the
# strings the SDK would have produced while it has never continued, and keeps counting after it
# has. Without it, a failure recorded after a boundary would publish an ID the same run without a
# boundary would not.
_FIRST_ACTIVITY_ORDINAL = 1

# What an Activity that could not produce a receipt's evidence failed as, and which of the four
# reasons that is. The keys are the types a capture and a resolver declare their refusals under,
# and a failure outside this table is not an evidence failure at all: a grader that was down, a
# world that timed out and a candidate this generation would not vouch for each end the attempt
# with the fields that ending already carries and leave no row here.
_EVIDENCE_REASONS: Dict[str, str] = {
    "WrongSource": WRONG_SOURCE,
    "ContractDrift": CONTRACT_DRIFT,
    "ReceiptContractRefused": CONTRACT_DRIFT,
    "CorruptEvidence": CORRUPT_EVIDENCE,
    "PairRefused": CORRUPT_EVIDENCE,
    "SourceArtifactRefused": CORRUPT_EVIDENCE,
    "UnavailableEvidence": UNAVAILABLE_EVIDENCE,
}


class StreamProtocolError(ApplicationError):
    """A refusal carrying one code from the protocol's closed set.

    It travels as the transport's error, not as a result: it has no message ID and it advances
    nothing. What the model saw of it is recorded in the harness transcript. The canonical
    record is attached as a detail so a gateway can forward the same bytes it would have built
    itself.
    """

    def __init__(self, code: str) -> None:
        self.code = code
        record = ProtocolError(code=code)
        super().__init__(
            code,
            canonical_bytes(record).decode("utf-8"),
            type="ProtocolError",
            non_retryable=True,
        )


class TurnoverPending(ApplicationError):
    """The generation is between executions, so this Update was not admitted.

    It is deliberately not a protocol refusal. The protocol's codes are a closed set of things
    the agent's own request was wrong about, and this is not one of them: nothing the caller
    sent is wrong and nothing about the generation has changed. What has happened is that the
    generation has decided to continue as new and is waiting for the quiet point to do it, and
    until then it accepts only the work that can bring the boundary about.

    So this reaches the transport as its own type, the transport waits and sends the same
    request again under the same Update ID, and the agent sees latency. A rejection costs the
    generation nothing: the service does not count an Update it never accepted, which is what
    makes waiting free of the very limit the generation is avoiding.
    """

    def __init__(self) -> None:
        super().__init__(
            "this generation is continuing as new; the same request reaches the execution that "
            "follows",
            type=TURNOVER_PENDING,
            non_retryable=True,
        )


class OriginUnverified(ApplicationError):
    """This generation was cut from a fork and has not yet authorized that lineage.

    It is deliberately not a protocol refusal, for the reason a turnover's is not: the protocol's
    codes are a closed set of things the caller's own request was wrong about, and nothing the
    caller sent is wrong. A child comes up owning nothing, serving nothing and answering no agent
    until its own history has read the parent's record of it, because a child that served one
    message before discovering its origin was false has already spent the measurement it was
    created for.

    So this reaches the caller as its own type and the same call sent again reaches a generation
    that has done the reading. What the reading can conclude instead is a refusal of another kind
    entirely, permanent and never a retry.
    """

    def __init__(self) -> None:
        super().__init__(
            "this generation has still to authorize the lineage it was cut from, and it owns "
            "nothing and serves nothing until it has",
            type=ORIGIN_UNVERIFIED,
            non_retryable=True,
        )


class ForkRefused(ApplicationError):
    """A fork this generation will not take, naming the clause that failed.

    It is a controller refusal and never joins the agent-visible set: no fork failure is ever
    mapped onto a protocol code, and nothing about one reaches a model. ``reason`` is one of the
    closed refusal reasons and ``clause`` says which condition under it did not hold, so a fork
    that did not happen is diagnosable without reading a history.

    Retryable and permanent are kept apart because the experiment treats them differently: it
    retries infrastructure and never retries a decision, so a permanent refusal is recorded with
    its reason rather than attempted again.

    The reason is the failure's own type rather than a detail beside a shared one, because a
    refusal has to keep its reason wherever the failure is kept: the outcome journal holds a type
    and a message and no details, and an exact identifier answered from that journal after a
    boundary or from a closed parent has to say the same thing the live refusal said.
    """

    def __init__(self, reason: str, clause: str) -> None:
        self.reason = reason
        self.clause = clause
        super().__init__(
            f"this fork is refused: {clause}",
            reason,
            clause,
            type=reason,
            non_retryable=reason not in RETRYABLE_FORK_REFUSALS,
        )


class ForkBarrier(ApplicationError):
    """This generation has committed a fork barrier, so this Update was not admitted.

    A barriered parent is fenced and parked and never advances again. The whitelist of what it
    still accepts is smaller than a turnover's, because the parent is already quiet rather than
    working towards quiet: it admits the fork request under its own identity and the fresh logical
    retries that can finish an incomplete fork, and everything a controller wants to read is a
    Query, which reaches none of this and charges nothing.

    It is deliberately not a turnover's refusal, and not a protocol code. A caller told a turnover
    is pending waits for an execution that is coming; there is no such execution here, and a caller
    that waited for one would wait for ever.
    """

    def __init__(self, clause: str) -> None:
        self.clause = clause
        super().__init__(
            f"this generation has forked and accepts no further work: {clause}",
            clause,
            type=FORK_BARRIER,
            non_retryable=True,
        )


class _UnusableResult(ApplicationError):
    """A failure for an Activity result the seal cannot vouch for.

    It is not a refusal: nothing the caller sent is wrong, so it carries no protocol code. It is
    raised before the transition that would make an acknowledgement authoritative, which is the
    last place such a result can still be caught, so the filing that raised it offers no
    protocol result, commits no presentation, and moves no cursor.

    It does reach the caller, as the generic failure of the tool call that filed. A harness
    keeping its own transcript records that failure the way it records any other, so this is a
    thing the model can see happen even though the generation minted nothing it can read. What
    does not travel with it is why the result was unusable. That reason is this generation's, and
    it is kept where this generation keeps its own facts: the check that refused names itself in
    ``type`` and says what it found in the message, and both are written onto the attempt's row,
    which a harness Query answers with and the run's own records carry.

    The Activity that produced the result succeeded, so there is no step left to retry: the
    exact filing sent again asks for the same work and is handed the same result back. That is
    why it is one exception rather than one for each way a result can be wrong, and why the seal
    that raised it ends the attempt it prepared.

    ``final_failure`` is which of those endings gets written. It is one transition either way,
    and the reason is what a reader is left with: a candidate built under something other than
    the policy this obligation was resolved to is a run that served bodies its record does not
    describe, and counting that as a generic unusable result would hide the one failure an
    operator most needs to see.
    """

    def __init__(self, *args: Any, final_failure: str = SEAL_UNUSABLE, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.final_failure = final_failure


@dataclass
class _Attempt:
    item: TaskItem
    state: str = PLANNED
    # The checkpoint this attempt would be restored from, and how many calls to a world the
    # generation has authorized since it committed. A reference in the flat set of everything a
    # presentation carried says an object is cited somewhere. A resume needs to know which
    # attempt it belongs to, and whether anything has happened after it.
    task_start_checkpoint: Optional[str] = None
    environment_calls: int = 0
    terminal_request_id: Optional[str] = None
    terminal_identity: Optional[str] = None
    terminal_tool: Optional[str] = None
    terminal_source: Optional[str] = None
    seal_id: Optional[str] = None
    submission_digest: Optional[str] = None
    canonical_submission_text: Optional[str] = None
    environment_recovery_token: Optional[str] = None
    finalizer_key: Optional[str] = None
    score: Optional[float] = None
    decode_state: Optional[str] = None
    # The object the environment took this score out of, by the hash that names it. The score is
    # a number and this is what produced it, so a reader with the run's store can go from the
    # committed headline to the verdict behind it.
    graded_evidence: Optional[str] = None
    seal_ordinal: Optional[int] = None
    final_failure: Optional[str] = None
    # When this attempt's deadline expires, in milliseconds on the generation's clock. It is set
    # when the attempt becomes active and cleared the moment the attempt is no longer one a
    # deadline could end, so an armed deadline and a live attempt are the same fact.
    deadline_at: Optional[int] = None
    # Whether that deadline has passed. The timer firing is recorded here rather than acted on
    # where it fires, because the attempt may be holding a call to a world this stream cannot
    # see, and ending it under one would be deciding an effect nothing here can observe. The
    # expiry is the durable fact; the ending happens at the first moment the stream is quiet.
    deadline_expired: bool = False
    # What the work behind an accepted terminal failed with. The ending is a class and several
    # very different failures share it, so these five say which Activity Temporal gave up on,
    # what it failed as, what it said, and why the retries stopped. They are read out of what
    # the history recorded, which is why a replay produces them again, and every text among them
    # is bounded before it is kept.
    #
    # A result the seal could not vouch for fills the kind and the message and leaves the other
    # three alone. Its Activity succeeded, so naming it as the step that failed would be false
    # and there is no retry state to report: what this generation is recording there is the
    # check that refused the answer, not a step that never landed.
    failure_activity: Optional[str] = None
    failure_activity_id: Optional[str] = None
    failure_kind: Optional[str] = None
    failure_message: Optional[str] = None
    failure_retry_state: Optional[str] = None
    # The source this attempt's seal committed, and the selection made out of it. The descriptor
    # is held whole rather than by digest, because everything a later derivation reads is inside
    # it and nothing may have to open a store to find out what this attempt's cells were. The
    # commitment names its canonical bytes and the origin is the identity its seal id was minted
    # under, which is this generation's own here and is a field rather than an assumption so that
    # an inherited source can be checked against the generation that produced it.
    source_artifact: Optional[SourceArtifactManifest] = None
    source_commitment: Optional[str] = None
    source_origin: Optional[SourceOriginContext] = None
    # Which committed cell this attempt's obligation was served, by whose policy, under which
    # contract, and the reference the resolver was given. It lives on the attempt rather than on
    # the obligation because an obligation keeps a candidate only while one could still be
    # offered: a reference hung on the candidate alone would be dropped with the body at the first
    # boundary after a presentation, and the selection has to outlive the bytes.
    selected_cell: Optional[str] = None
    selected_body_reference: Optional[str] = None
    selected_policy_digest: Optional[str] = None
    receipt_contract_id: Optional[str] = None
    # The objects this attempt's own committed presentations cited, in commit order. A claim reads
    # the store for every one of them, and the flat inventory it reads them out of cannot say
    # which attempt required which object, so the association is kept here: a claim refused over a
    # presentation reference is refused over evidence this attempt needs, and the row it leaves
    # names it. A generation that declares no receipt contract keeps none of this.
    presentation_references: List[str] = dataclass_field(default_factory=list)


# The attempt fields one execution hands the next, taken from the carried row rather than listed
# twice. What is not here is what a boundary does not need: the task itself, which the start
# already carries, and the three the seal writes and nothing reads.
_CARRIED_ATTEMPT_FIELDS = tuple(
    row.name for row in fields(CarriedAttempt) if row.name != "attempt_id"
)
# The two that cross as the mappings they were written as rather than as typed values. They are
# written out here and read back by the strict readers at restore, for the reason the seal result
# carries its own descriptor the same way: a typed carrier field is one the decoder makes that
# type of before any code of this generation runs, so a member this build does not declare is
# dropped there and a size written as a float is rounded there, and what restore would then check
# is the record the decoder repaired rather than the record that crossed.
_RAW_ATTEMPT_FIELDS = ("source_artifact", "source_origin")
# The attempt fields that deliberately stay behind, so that a field added later is a choice
# somebody made rather than a value that quietly stopped crossing.
_UNCARRIED_ATTEMPT_FIELDS = (
    "item",
    "canonical_submission_text",
    "environment_recovery_token",
    "finalizer_key",
    "deadline_expired",
)


@dataclass
class _Obligation:
    item: TaskItem
    state: str = ASSIGNED
    candidate: Optional[PayloadCandidate] = None
    # Whether the candidate was ever built. It is kept apart from the candidate itself because
    # the two outlive each other by different lengths: the count of materializations is inside
    # the projection every presentation attests against and has to hold for the life of the
    # generation, and the bytes are needed only while an offer could still carry them.
    materialized: bool = False
    # Whether this obligation is a fork child's, selected against the parent's committed source
    # and waiting for the child's own preparation to build its body. It is the one state in which
    # an offerable obligation lawfully carries no candidate: the parent writes it into the child's
    # start, and preparation clears it in the transition that installs the candidate.
    pending_preparation: bool = False


@dataclass
class _Eligibility:
    """An obligation becoming eligible: which plan released it, on what, and in what order."""

    attempt_id: str
    release_plan_id: str
    causal_event: str
    priority: str
    order_key: Tuple[int, int, str]


@dataclass(frozen=True)
class _ReplyBounds:
    """The proved encoded upper bounds of the three replies one fork transmits.

    They are measured together and before the barrier, and every one of them is retained in the
    record the barrier commits, because a bound recomputed later from the value it is meant to
    bound is not a bound at all.
    """

    receipt: int
    origin: int
    start: int


@dataclass
class _Bound:
    """A logical request, its canonical identity, and the immutable result bound to it."""

    identity: str
    message: OfferedMessage


@dataclass
class _BoundFinalization:
    """A logical finalization, its canonical identity, and the receipt bound to it.

    A finalization mints no message, so what a retry has to reach is the receipt rather than
    reserved bytes. It is otherwise the same fact as a bound pull: one logical request, one
    answer, for as long as the generation runs.
    """

    identity: str
    receipt: AttemptFinalized


@dataclass
class _Pending:
    message: OfferedMessage
    origin: str
    request_id: str


@dataclass
class _Ownership:
    """One writer taking the generation from the last one, and the swap that proved it could.

    The witness is the epoch the claimant read before it claimed. Keeping it beside the epoch
    it won is what makes the compare and swap a fact in the record rather than a step that
    happened somewhere.
    """

    ownership_epoch: int
    previous_epoch: int
    witnessed_epoch: int
    fencing_token_hash: str
    replaced_token_hash: Optional[str]
    claimant_id: str
    consumer_claim_hash: str
    reason: str
    # The attempts this owner said it had put back before it claimed. A takeover that continues
    # a world somebody restored is a different fact from one that continues a world nothing
    # happened in, and the record says which of the two this was.
    restored_attempts: List[str]


@dataclass
class _Offer:
    """The server-side record of a reservation. An offer is not a delivery."""

    request_id: str
    identity: str
    origin: str
    cursor_before: str
    kind: str
    message_id: str
    attempt_id: Optional[str]
    visible_sha256: str
    wait_reason: Optional[str] = None


# The eleven Updates a generation answers, named once so the journal can say which handler gave
# an answer and the transport can decode it back into the type that handler returns.
OWN = "claim_ownership"
BIND = "claim_consumer"
PULL = "pull"
ASK = "info"
SEAL = "seal_attempt"
PRESENT = "commit_presentation"
FINALIZE = "finalize_attempt"
CLOSE = "close_queue"
GRANT = "begin_environment_call"
RELEASE = "end_environment_call"
CONFIRM = "confirm_state"
# And the twelfth, which no agent reaches and no gateway sends. A fork is a platform operation over
# two generations rather than a call in the protocol, and the exact Update returns the complete
# receipt, which is what makes the outcome journal a usable recovery path for it.
FORK = "fork_generation"

# What each of them answers with. A journal entry says which handler wrote it, and that is
# enough to read the value back as the thing it is rather than as an untyped map.
ANSWER_TYPES: Dict[str, Any] = {
    OWN: OwnershipReceipt,
    BIND: ConsumerReceipt,
    PULL: OfferedMessage,
    ASK: OfferedMessage,
    SEAL: OfferedMessage,
    PRESENT: PresentationAck,
    FINALIZE: AttemptFinalized,
    CLOSE: QueueClosed,
    GRANT: EnvironmentLease,
    RELEASE: EnvironmentLease,
    CONFIRM: StreamState,
    FORK: ForkReceipt,
}

# How a journal entry says where its answer is. A value the generation keeps nowhere else is
# written out; an answer that is one of the logical rows this generation already carries is a
# pointer at that row, so a message offered once is stored once however many identifiers reached
# it. The two failure kinds are apart because a protocol refusal is reproduced exactly from its
# code and anything else is reproduced as the declared type and words it carried.
BY_VALUE = "value"
BY_ROW = "row"
REFUSED = "protocol"
FAILED = "failure"


@dataclass
class _Answer:
    """What one exact Update was answered with, and where that answer is kept.

    One entry per exact Update identifier, written when the handler completes and never
    rewritten. Several identifiers may point at one logical row: a replacement owner asking for
    the ending an earlier owner already got is a second identifier reaching one receipt, and both
    identifiers have to keep working.

    The epoch is provenance rather than a second key. The service deduplicates on the identifier
    alone, and every identifier this runtime builds already carries its owner, so comparing the
    epoch again would make this stricter than the thing it is standing in for.

    A recipe is a second way of keeping a value, not a second kind of answer. Where an answer can
    be made again out of what the generation's start already holds, the recipe says how and the
    digest says which bytes it has to come out as, and the value itself is left out of the
    carrier. Anything a recipe cannot reproduce exactly keeps its literal value, and the choice
    is made by rebuilding at the time of writing rather than by reasoning about which offers
    ought to be reproducible.
    """

    handler: str
    epoch: int
    kind: str
    row: str = ""
    code: str = ""
    type_name: str = ""
    message: str = ""
    value: Optional[Dict[str, Any]] = None
    recipe: str = ""


@workflow.defn(name="ShogymStreamV2")
class StreamWorkflow:
    """One authenticated stream generation: its queue, its attempts, and its cursor."""

    @workflow.init
    def __init__(self, start: StreamStart) -> None:
        _check_start(start)
        self._start = start
        self._release = start.release
        self._id_key = bytes.fromhex(start.id_key_hex)
        self._configuration_hash = configuration_hash(start)
        self._generation_state = OPEN
        # Nobody owns the generation until somebody claims it, and until then no call that
        # could change the stream is accepted. Creation is a claim like a resume is.
        self._ownership_epoch = 0
        self._fencing_token_hash: Optional[str] = None
        self._ownership: List[_Ownership] = []
        self._consumer_id: Optional[str] = None
        self._claim_epoch = 0
        self._cursor = start.initial_cursor
        self._queue_closed = False
        self._items: Dict[str, TaskItem] = {item.attempt_id: item for item in start.tasks}
        self._assignments: Dict[str, Assignment] = {
            row.attempt_id: row for row in _roster(start)
        }
        self._attempts: Dict[str, _Attempt] = {
            item.attempt_id: _Attempt(item=item) for item in start.tasks
        }
        # Never creates no payload obligation, so under it there is no row to materialize,
        # release, offer, or wait for. The outbox does not exist rather than sitting empty.
        # A roster row that carries no payload is the other case, one position at a time: its
        # task is served and scored and nothing is ever delivered against it.
        self._obligations: Dict[str, _Obligation] = (
            {
                item.attempt_id: _Obligation(item=item)
                for item in start.tasks
                if self._assignments[item.attempt_id].creates_payload_obligation
            }
            if self._release.creates_obligations
            else {}
        )
        # What each obligation was resolved to on the branch this generation serves, which is
        # the only branch it may resolve: a row for a slot nothing has created is refused at the
        # start. The key keeps its branch anyway, because two children of one fork sharing one
        # obligation is the reason two rows can carry it.
        self._served: Dict[str, PayloadDisposition] = {
            row.attempt_id: row
            for row in start.dispositions
            if row.branch_slot == start.served_slot and row.kind == DELIVER
        }
        # Every row of the branch this generation serves, delivering or withholding. The seal
        # reads this one rather than the deliveries: which contract a capture is validated
        # against is named in the row, and an attempt that captures and delivers nothing has a
        # row that says so and no delivery at all.
        self._resolved: Dict[str, PayloadDisposition] = {
            row.attempt_id: row
            for row in start.dispositions
            if row.branch_slot == start.served_slot
        }
        # The matched arms this generation's rows are cells of, by the name a row claims.
        self._families: Dict[str, MatchedFamily] = {
            family.family_id: family for family in start.families
        }
        # And the receipt contracts declared beside them, in the same column a row names a family
        # in. The two lists hold distinct names, which is checked where the generation is, so a
        # row's column names one record or the other and never both.
        self._contracts: Dict[str, ReceiptContract] = {
            contract.contract_id: contract for contract in start.receipt_contracts
        }
        self._pending: Optional[_Pending] = None
        self._pull_requests: Dict[str, _Bound] = {}
        self._terminal_requests: Dict[str, _Bound] = {}
        # The info tool's own reservations, kept apart from the pull's so that a reservation is
        # scoped to the tool that made it: the same request ID names one reservation under each
        # tool rather than one across both. Neither map is what stops a caller collecting the
        # other tool's bytes, and neither is meant to be. Both handlers compare the identity
        # bound to a request ID before they hand anything back, and the two identities are taken
        # under different domain tags, so a request ID reused across the tools is a conflict
        # under one map and two separate reservations under these.
        self._info_requests: Dict[str, _Bound] = {}
        self._finalize_requests: Dict[str, _BoundFinalization] = {}
        self._attestations: Dict[str, PresentationAck] = {}
        self._attestation_identities: Dict[str, str] = {}
        # Every presented message, in order, with the kind it was and the digest of the bytes
        # it went as. The kind is what keeps payload delivery a count of its own rather than a
        # share of the presentations, and the digest is what a harness reconciling its own
        # transcript against these deliveries compares what it wrote down with.
        self._presented: Dict[str, PresentedMessage] = {}
        # Every object a committed presentation referenced, once each. A reference verified when
        # its event committed is a fact about that moment, and this is what makes it a question
        # a resume can ask again.
        #
        # The preimage of every policy this generation resolved to is in it from the start, and
        # so is every cell of every matched family it declares. A digest names bytes, and the
        # claim this generation makes is about the bytes: a run whose descriptor object is gone
        # is one whose record says what its bodies were allowed to contain and cannot produce the
        # saying, and an arm that cannot produce its counterpart's is one whose record says it
        # was a comparison and cannot say what against. So the descriptors are required objects
        # rather than a convenience installed beside the run, and a resume that cannot read them
        # back exactly is refused like any other unverifiable reference.
        self._committed_blobs: List[str] = (
            descriptor_digests(start.dispositions, start.families, start.receipt_contracts)
            if start.blob_root is not None
            else []
        )
        self._eligibilities: List[_Eligibility] = []
        self._issued_ids = {start.initial_cursor, start.done_message_id}
        for item in start.tasks:
            self._issued_ids.update({item.task_message_id, item.ack_message_id})
            self._issued_ids.add(item.payload_message_id)
        self._hidden_ordinal = 0
        self._offers: List[_Offer] = []
        self._seal_ordinal = 0
        self._wait_count = 0
        self._operation_in_flight = False
        # The environment call the stream is currently held for, if any, and the ticket it
        # holds it under. It holds the stream the way an Update does, and it is given back by
        # name rather than by returning.
        self._environment_call: Optional[str] = None
        self._environment_ticket = 0
        # Which call holds the stream. A handler releases the stream only when the ticket it
        # took is still the current one, because an owner that fenced it has already given the
        # stream to somebody else.
        self._operation_ticket = 0
        self._done_presented = False
        self._draining = False
        # What an earlier execution of this generation already did, kept as counts because the
        # rows behind them are read as counts and nothing else. Every reader of these adds the
        # carried number to what this execution has done since.
        self._carried_ownership_claims = 0
        self._carried_offers = 0
        self._carried_eligibilities = 0
        self._carried_wait_reasons: Dict[str, int] = {}
        self._carried_handed_out: Set[str] = set()
        # The Activity ordinal this generation numbers its own Activities from. The SDK would
        # number them per execution; this counts across the chain, so an ID says which Activity
        # of the generation it was rather than which of the execution.
        self._activity_ordinal = _FIRST_ACTIVITY_ORDINAL
        # The Updates this execution has accepted, and whether the generation has decided to
        # continue as new. The counter is this generation's own: the service counts what it
        # admitted, and only a count kept here is a fact a replay produces again.
        self._updates = 0
        self._post_latch = 0
        self._post_latch_repeats = 0
        self._post_latch_claims = 0
        self._turnover_requested = False
        # Whether the branch that turns this generation over is available to this execution.
        # It is settled once, at the latch, and it is what everything the latch switches on
        # reads: an execution the branch is unavailable to holds no traffic back and behaves
        # exactly as it did before there was a boundary to reach.
        self._turnover_available = False
        # How many boundaries this generation has crossed, and the size that stopped it crossing
        # another. Neither is configuration and neither is in any hash: the first is operational
        # metadata and the second is what a launcher reads to say the profile was not supported.
        self._turnovers = 0
        # Why this generation gave up on a boundary, and the size that decided it where size is
        # what decided it. A generation that has given up serves out the execution it is in.
        self._turnover_refused: Optional[str] = None
        self._turnover_refused_bytes: Optional[int] = None
        # The store reads this generation has in flight, and how many it has finished. A claim
        # reads the store before it may swap the epoch, and that read is an Activity with its own
        # timeout and its own retries: a batch can legitimately take minutes without moving
        # anything a projection would notice. Both are operational, in no hash and in no
        # projection, and they exist so a caller held off for a boundary can tell a generation
        # working towards one from a generation that has stopped.
        self._verifying = 0
        self._verification_batches = 0
        # The payload obligations whose evidence this execution has verified, keyed by attempt.
        # It is per execution and it is never carried: a set that crossed would say an object was
        # there in an execution that has ended, which is exactly the claim a resume exists to stop
        # anyone making. An obligation joins it only after its own dependencies verify, at the
        # publication that installed them or at the check that read them back, so the first offer
        # of one in a continued execution reads the store and every offer after it is pure. A
        # single flag would be set by the first receipt and would wave the second through
        # unchecked, and two receipt obligations in one continued execution is the ordinary case.
        self._verified_deliveries: Set[str] = set()
        # Every operation of this generation that could not produce the evidence it depended on,
        # in commit order. They are rows and not a field: a refusal and the recovery that follows
        # it each append, so the sequence says what happened rather than only what is true now,
        # and the latest word about an attempt is the last row naming it. They cross the boundary,
        # and they are read out of the same query the attempts and the presentations are.
        self._operation_failures: List[OperationFailure] = []
        # Which generation an operation of this generation belongs to. It is the workflow's own
        # identifier, which survives every boundary and is not the identifier of a child a fork
        # will create, so a row a child inherits reads as the parent's operation.
        self._generation_id = workflow.info().workflow_id
        # Every accepted Update this generation has completed, by the exact identifier it was
        # answered under. It is the whole of what a repeated identifier is answered with, and it
        # is kept for as long as the generation runs rather than for one execution: an owner
        # paused with an unanswered request is still owed its answer two boundaries later, and
        # nothing here can know when a caller has stopped asking.
        #
        # The service keeps the same record inside one execution and that record does not cross,
        # so this is that record made to cross. Several identifiers may name one logical row, and
        # every one of them keeps working.
        self._journal: Dict[str, _Answer] = {}
        # Which accepted handlers are running, by the exact Update identifier each was reached
        # under. A boundary waits for all of them, and a caller held off for that boundary has to
        # be able to see that it is waiting for something: an owner replaced while its filing was
        # still grading leaves work the projection no longer names anywhere, and a transport
        # comparing projections would read that as a generation that had stopped.
        #
        # The identifier is what makes it a ledger rather than a count. The SDK inserts an Update
        # into its own in-progress map before it runs that Update's validator and removes it in the
        # handler's finally, so a request that waited for the SDK's predicate in its own validator
        # or its own handler would wait for itself for ever. The public predicate takes no
        # exclusion argument, so the exclusion is this generation's to keep: an entry is made in
        # the accepted handler's prologue, which is after every validator has run, so a validator
        # sees exactly the handlers that were accepted and have not finished and never sees itself,
        # and a handler excludes its own identifier by name.
        #
        # This kernel registers no signal handler at all, so the signal half of the SDK's predicate
        # is empty by construction, and any signal handler added later joins this ledger under the
        # same rule.
        self._handlers: Dict[str, str] = {}
        # The fork this generation has committed a barrier for, if it has committed one, and the
        # receipt it answered with. The record is immutable provenance about the children and their
        # starts; the parent never advances again once it stands.
        self._fork: Optional[PreparedFork] = None
        self._fork_receipt: Optional[ForkReceipt] = None
        # When the parent's own fork work is due, and the absolute moment after which it can answer
        # nothing. Both are recorded at the barrier's commit, from the clock this generation reads,
        # because the close cannot be bounded and the deletion clock is not the close clock either.
        self._fork_deadline_at = 0
        self._fork_horizon_at = 0
        # What has been accepted since the barrier, which is what the reserve bounds. The gate
        # reads it rather than inferring it from the work this generation guessed at.
        self._post_barrier = 0
        # Whether the parent has stopped admitting fork work, which expiry and exhaustion both
        # reach: it admits no further work and schedules no new start work, it bounds the start
        # work already pending, and it closes only after the ledger settles.
        self._fork_admits_nothing = False
        # How many times each step of a fork has been dispatched, so every fork-only invocation
        # takes its identifier from the fork's own namespace and a child's ordinary Activity
        # numbering equals the number its inherited prefix left.
        self._fork_steps: Dict[str, int] = {}
        # Whether this execution was continued from another one of the same generation. It is
        # what makes a carrier legal, and it is also what tells the profile marker below that
        # this is not a generation being created.
        self._continued = workflow.info().continued_run_id is not None
        # Whether this execution has still to authorize the lineage its start carries. A child
        # comes up owning nothing and serving nothing until its own history has read the parent's
        # record of it, because a child that served one message before discovering its origin was
        # false has already spent the measurement it was created for. It is this execution's own
        # state and is never carried: what it gates is the comparison of a child against the
        # parent it was cut from, which is asked once, at the entry that was cut.
        self._origin_unverified = start.fork_origin is not None and not self._continued
        # And whether the reading that opens that gate came back saying the window it had to be
        # asked in has closed. It is a state of its own rather than a longer wait, because the
        # horizon a parent's answers stand behind is absolute: no later reading reaches an answer,
        # so a child that met it keeps its identity, owns and serves nothing for good, and says
        # which of the two it is to whatever asks it for something.
        self._origin_expired = False
        self._restore(start.carry)

    def _restore(self, carry: Optional[StreamCarry]) -> None:
        """Take the projection this execution was handed, or refuse the pair.

        A carry is legal in four entries and nowhere else, and lineage is kept apart from the
        authorization of the carry a start actually holds. A fresh ordinary generation carries
        nothing and was cut from no fork. A fresh child holds a lineage record and the carry its
        parent composed, and the identity inside that carry is the parent's, which the record
        names. A continued ordinary generation holds its own carry, authorized by the service's
        own continuation fact. A continued child holds its own carry under that same fact,
        compared against its own identity, and retains its lineage as provenance that authorizes
        nothing about the carry. A start with a carry and none of those authorizations is refused
        whole, and so is a start that names a lineage and carries no projection: a child is the
        prefix it inherited.

        The reason the two are checked against each other rather than the carrier being trusted
        is unchanged. A caller composing a projection could write a cursor, a score, a
        presentation or an ownership epoch into a generation that never served any of them, and an
        execution continued from another that was handed nothing would come back as an empty
        roster under a fenced writer and serve the whole run again.

        The version and the identity are checked next, and for the same reason. A carrier this
        code cannot read is refused rather than read half way, and one composed against another
        generation is refused before any of it is believed.

        A carrier that cannot be read at all is refused the same way, and reading it means all of
        it: the decoding, and the writing of what was decoded over the state the start rebuilt. A
        value whose shape is wrong is found in the second of those as often as the first, and what
        either can raise is a codec's failure rather than a generation's. Left alone that would
        fail the Workflow Task and be retried for ever on bytes that are never going to decode. So
        the whole reading is one refusal, and the execution ends saying it could not read what it
        was handed. The refusals raised inside it are already the answer and pass through.
        """
        origin = self._start.fork_origin
        if origin is not None:
            self._check_lineage(origin)
        if carry is None:
            if self._continued:
                raise _refuse_carrier("a continued execution was handed no carried projection")
            if origin is not None:
                raise _refuse_carrier(
                    "a generation cut from a fork was handed no carried projection, and a child "
                    "is the prefix it inherited"
                )
            return
        if self._continued:
            cut_from: Optional[ForkOrigin] = None
        elif origin is None:
            raise _refuse_carrier(
                "a generation started fresh was handed a carried projection, and carrying one "
                "is legal only where an execution continues another or a fork cut it"
            )
        else:
            cut_from = origin
        if carry.carrier_schema_version not in CARRIER_SCHEMA_VERSIONS:
            raise _refuse_carrier(
                f"the carried projection is version {carry.carrier_schema_version} and this "
                f"code reads versions {list(CARRIER_SCHEMA_VERSIONS)}"
            )
        try:
            projection = unpack_carrier(carry, workflow.payload_converter())
            self._authorize_carry(projection, cut_from)
            self._apply(projection, _check_carried_receipts(self._start, projection))
        except ApplicationError:
            raise
        except Exception as unreadable:  # noqa: BLE001 - an unreadable carrier is refused whole
            raise _refuse_carrier(
                f"the carried projection could not be read: {unreadable}"
            ) from unreadable

    def _check_lineage(self, origin: ForkOrigin) -> None:
        """Hold the lineage a start carries to what this build reads and to this child's identity.

        Both run at every entry, fresh or continued, because a lineage record is immutable
        provenance rather than an authorization of anything: a child that has continued twice
        still says which generation it was cut from, and still has to be the child that record was
        written about.
        """
        try:
            check_fork_origin(origin)
        except WireFormatError as error:
            raise _refuse_carrier(
                f"the lineage this start carries is not one this build reads: {error}"
            ) from error
        if origin.child_configuration_hash != self._configuration_hash:
            raise _refuse_carrier(
                "the lineage this start carries records another generation's identity as this "
                "child's, and a child is what its own start says it is"
            )

    def _authorize_carry(
        self, projection: CarriedProjection, cut_from: Optional[ForkOrigin]
    ) -> None:
        """Say which authorization admits this carry, and refuse a projection none of them does.

        A continued execution's carry is its own, whether or not the generation was cut from a
        fork, so it is compared against this start's own identity: the record's parent hash is
        what the inherited carrier was written under, and a child that has continued once has
        written its own since. Comparing against the record instead would refuse a child's own
        lawful continuation, which is the whole reason lineage and authorization are separate.

        A fresh child's carry is its parent's, so the identity inside it is the parent's and the
        record's own copy of that hash is what admits it. The cursor and the seal ordinal are the
        ones the record names, so a carrier cut somewhere other than the acknowledged boundary is
        refused before any of it is believed. The ordered presented rows are not compared here at
        all: the record commits to none of them, and the complete start digest in the parent's
        prepared record covers the carrier those rows ride in, so a row altered anywhere fails
        that authorization whether or not it changed an identifier.
        """
        if cut_from is None:
            if projection.configuration_hash != self._configuration_hash:
                raise _refuse_carrier(
                    "the carried projection was composed against another generation"
                )
            return
        if projection.configuration_hash != cut_from.parent_configuration_hash:
            raise _refuse_carrier(
                "the carried projection a fresh child was handed was composed against a "
                "generation other than the parent its lineage names"
            )
        if projection.cursor != cut_from.acknowledged_cursor:
            raise _refuse_carrier(
                f"this child was cut at the cursor {cut_from.acknowledged_cursor} and the "
                f"projection it was handed stands at {projection.cursor}"
            )
        if projection.seal_ordinal != cut_from.source_seal_ordinal:
            raise _refuse_carrier(
                f"this child was cut over {cut_from.source_seal_ordinal} filings and the "
                f"projection it was handed counts {projection.seal_ordinal}"
            )

    def _admit_lineage(self, record: PreparedChild) -> None:
        """Take the parent's own row for this child as the authority its start is checked against.

        This is the transition the origin verification reads its answer into, and it is the only
        thing that opens the gate. Until it commits the child owns nothing and serves nothing,
        because a child that served one message before discovering its origin was false has
        already spent the measurement it was created for.

        A disagreement is permanent. The row came from the parent's own record and the start is
        the one this execution is running, so there is nothing here for a retry to reach.
        """
        try:
            check_complete_start_authorization(
                self._start, workflow_id=workflow.info().workflow_id, record=record
            )
        except WireFormatError as error:
            raise _refuse_origin(str(error)) from error
        self._origin_unverified = False

    def _require_authorized_lineage(self) -> None:
        """Refuse a call that would own or serve a child that has not authorized its lineage.

        It is read in the validator, where a rejection costs the generation nothing, and read
        again in the two handler-side places a call that got past it would take effect: the claim
        that installs ownership and the writer check every stream-affecting call makes.

        A child whose reading came back expired says so instead. The two are different answers to
        the same caller: the gate is a wait, and a closed window is the end of the reading, so a
        controller told the first comes back and a controller told the second stops.
        """
        if self._origin_expired:
            raise ForkRefused(
                FORK_EXPIRED_AUTHORITY,
                f"the child {workflow.info().workflow_id} asked the parent its lineage names for "
                "the record it verifies against and the window that parent's answers stood in had "
                "closed, so it owns nothing and serves nothing",
            )
        if self._origin_unverified:
            raise OriginUnverified()

    def _apply(self, carry: CarriedProjection, read: Dict[str, _CarriedSource]) -> None:
        """Write the carried projection over the state the start alone rebuilt.

        The order is the order the fields depend on each other in: the issued identifiers are
        derived from the hidden ordinal, so the ordinal is taken first and the identifiers are
        rebuilt from it rather than carried. Everything else is written where it belongs.

        ``read`` is the descriptor and the origin the checks above read out of the mapping each
        crossed as. They are taken from there rather than decoded again, so what is installed on
        the attempt is exactly what was validated rather than a second reading of the same bytes.
        """
        self._ownership_epoch = carry.ownership_epoch
        self._fencing_token_hash = carry.fencing_token_hash
        self._carried_ownership_claims = carry.ownership_claims
        self._consumer_id = carry.consumer_id
        self._claim_epoch = carry.claim_epoch
        self._cursor = carry.cursor
        self._queue_closed = carry.queue_closed
        self._seal_ordinal = carry.seal_ordinal
        self._wait_count = carry.wait_count
        self._carried_wait_reasons = dict(carry.wait_reasons)
        self._carried_offers = carry.offer_count
        self._carried_eligibilities = carry.eligibility_count
        self._carried_handed_out = set(carry.handed_out_attempt_ids)
        self._activity_ordinal = carry.activity_ordinal
        self._verification_batches = carry.verification_batches
        self._turnovers = carry.turnovers
        self._operation_failures = list(carry.operation_failures)
        for row in carry.attempts:
            attempt = self._attempts[row.attempt_id]
            for name in _CARRIED_ATTEMPT_FIELDS:
                if name in _RAW_ATTEMPT_FIELDS:
                    continue
                carried = getattr(row, name)
                # A list is copied rather than adopted, so that appending to the attempt's own
                # never reaches back into the value the carrier was read as.
                setattr(attempt, name, list(carried) if isinstance(carried, list) else carried)
            source = read[row.attempt_id]
            attempt.source_artifact = source.manifest
            attempt.source_origin = source.origin
        for owed in carry.obligations:
            obligation = self._obligations[owed.attempt_id]
            obligation.state = owed.state
            obligation.materialized = owed.materialized
            obligation.candidate = owed.candidate
            obligation.pending_preparation = owed.pending_preparation
        self._presented = {row.message_id: row for row in carry.presented}
        self._committed_blobs = list(carry.committed_blobs)
        self._pull_requests = _bindings(carry.pull_requests)
        self._info_requests = _bindings(carry.info_requests)
        self._terminal_requests = _bindings(carry.terminal_requests)
        self._finalize_requests = {
            row.request_id: _BoundFinalization(identity=row.identity, receipt=row.receipt)
            for row in carry.finalize_requests
        }
        self._attestation_identities = {
            row.attestation_id: row.identity for row in carry.attestations
        }
        self._attestations = {row.attestation_id: row.ack for row in carry.attestations}
        # The journal is restored here, in the constructor, because the first Update of a fresh
        # execution can be validated before anything else runs and it has to find its answer.
        self._journal = {
            row[0]: _Answer(
                handler=row[1],
                epoch=row[2],
                kind=row[3],
                row=row[4],
                code=row[5],
                type_name=row[6],
                message=row[7],
                value=row[8],
                recipe=row[9],
            )
            for row in carry.journal
        }
        # Every recipe is rebuilt and checked here, against the digest the presentation of that
        # message committed to. A recipe is a promise that the bytes can be made again from what
        # the start already holds, and this is where the promise is kept rather than assumed: a
        # carrier whose recipe does not rebuild what was presented is refused, not served.
        for update_id, answer in self._journal.items():
            if answer.recipe:
                answer.value = self._rebuilt(update_id, answer)
        # The identifiers this generation has already minted, rebuilt rather than carried. Every
        # one of them is either preallocated by the manifest, which the start above already put
        # in the set, or drawn from the keyed stream at an ordinal below the one reached, which
        # is exactly what this walks. Carrying the set instead would carry a value derivable
        # from a number, and a set has no order to serialize deterministically.
        for ordinal in range(carry.hidden_ordinal):
            self._issued_ids.add(stream_message_id(self._id_key, ordinal))
        self._hidden_ordinal = carry.hidden_ordinal

    @workflow.run
    async def run(self, start: StreamStart) -> StreamOutcome:
        """Serve until Done has been presented, then let every accepted call finish.

        The generation waits here, and while it waits it is also the only thing watching the
        clock. A stream whose attempts have no deadline waits once and creates no timer at all,
        which is what a generation that declares none should cost.

        A generation created now says what each of its payloads may contain, and this is where
        that is true of every execution rather than of the calls that go through a builder. The
        one shape with nothing to say is a history recorded before the question existed, and a
        replay of one runs this line: the marker separates the two, because an execution started
        now writes it and a history recorded then has none. So the old generations keep
        replaying and a new one cannot be created without a profile, whichever entry point
        starts it.
        """
        if (
            start.profile == LEGACY
            and not self._continued
            and workflow.patched("profile-required-at-creation")
        ):
            raise StreamProtocolError("configuration_mismatch")
        if self._origin_unverified:
            await self._verify_the_lineage()
        if self._origin_expired:
            # There is no reading that opens this gate now, so there is nothing for this
            # generation to work towards. It keeps its identity and its objects, answers what it
            # is asked with the reason, and the link it belongs to is recorded incomplete.
            await workflow.wait_condition(lambda: not self._origin_expired)
        while not self._done_presented and self._fork is None:
            await self._wait_for_done_or_a_deadline()
            # Every deadline that has come due is applied before a boundary is even considered.
            # The wait above can end on the turnover term rather than on the clock, and an
            # attempt whose deadline passed while it did would otherwise cross still active,
            # with its ending written by the execution that came after rather than the one the
            # clock ran in.
            self._expire_what_is_due()
            out_of_reach = self._boundary_is_out_of_reach()
            if out_of_reach is not None:
                self._give_up_on_the_boundary(out_of_reach)
            elif self._turnover_ready():
                self._turn_over()
        if self._fork is not None:
            await self._park_for_the_fork()
        await workflow.wait_condition(workflow.all_handlers_finished)
        return StreamOutcome(
            generation_state=self._generation_state,
            cursor=self._cursor,
            sealed=self._seal_ordinal,
            payloads_delivered=sum(
                1 for o in self._obligations.values() if o.state == PRESENTED
            ),
            finalized=sum(1 for a in self._attempts.values() if a.state == FINAL_FAILED),
        )

    async def _wait_for_done_or_a_deadline(self) -> None:
        """Wait until Done, until the armed deadlines change, or until the earliest one expires.

        The timer is durable, so an attempt's deadline survives the worker that armed it and a
        resume neither loses it nor restarts it. It is the earliest one that is waited on: any
        change to the armed set brings the wait back here to choose again, which is what keeps
        one timer at a time correct for a generation serving more than one attempt.

        An expiry already recorded and not yet acted on is the other thing this waits for. It
        is applied at the top of every pass, so the wait below can end on the stream falling
        quiet as well as on the clock.

        A generation that has committed a fork barrier leaves here without waiting at all, because
        it is fenced and parked and the wait it belongs in is the fork's own.

        A generation that has decided to continue as new waits here for its boundary, which is
        the third thing. The expiry is applied first, so a deadline that came due never crosses
        unapplied, and the boundary is only ever reached from the wait rather than from inside
        a handler.
        """
        self._end_expired()
        armed = self._armed_deadlines()
        if not armed:
            await workflow.wait_condition(
                lambda: self._done_presented
                or self._fork is not None
                or bool(self._armed_deadlines())
                or self._expiry_can_be_applied()
                or self._turnover_ready()
                or self._boundary_is_out_of_reach() is not None
            )
            return
        attempt_id, deadline = min(armed.items(), key=lambda row: (row[1], row[0]))
        remaining = deadline - self._now_ms()
        if remaining > 0:
            try:
                await workflow.wait_condition(
                    lambda: self._done_presented
                    or self._fork is not None
                    or self._armed_deadlines() != armed
                    or self._expiry_can_be_applied()
                    or self._turnover_ready()
                    or self._boundary_is_out_of_reach() is not None,
                    timeout=timedelta(milliseconds=remaining),
                )
                return
            except asyncio.TimeoutError:
                pass
        self._expire(attempt_id, deadline)

    def _armed_deadlines(self) -> Dict[str, int]:
        """Every attempt with a deadline, and when it expires."""
        return {
            key: value.deadline_at
            for key, value in self._attempts.items()
            if value.deadline_at is not None
        }

    def _now_ms(self) -> int:
        """The generation's clock, in milliseconds. Deterministic, and replayed rather than read."""
        return int(workflow.now().timestamp() * 1000)

    def _arm_deadline(self, attempt: _Attempt) -> None:
        """Start this attempt's clock, if the generation declared one."""
        if self._start.attempt_deadline_ms > 0:
            attempt.deadline_at = self._now_ms() + self._start.attempt_deadline_ms

    def _quiet(self) -> bool:
        """Whether nothing is part way through: no call holding the stream, no result owed."""
        return not self._operation_in_flight and self._pending is None

    def _expiry_can_be_applied(self) -> bool:
        """Whether a recorded expiry is waiting and the stream is free to act on it."""
        return self._quiet() and any(
            attempt.deadline_expired for attempt in self._attempts.values()
        )

    def _expire(self, attempt_id: str, deadline: int) -> None:
        """Record that this attempt's deadline passed, and disarm the timer that said so.

        The firing is a fact and it is written down where it happens, rather than a decision
        deferred until the stream is free. Everything is checked again here rather than assumed
        from the timer, because the attempt may have sealed while the stream was busy, its
        terminal may be in flight, or the generation may be over: a deadline that no longer
        applies is dropped, and it never overtakes a filing.
        """
        attempt = self._attempts.get(attempt_id)
        if attempt is None or attempt.deadline_at != deadline:
            return
        attempt.deadline_at = None
        if self._generation_state != OPEN or self._draining:
            return
        if attempt.state != ACTIVE or attempt.terminal_request_id is not None:
            return
        attempt.deadline_expired = True
        self._end_expired()

    def _expire_what_is_due(self) -> None:
        """Record every armed deadline the clock has already passed, then act on what it can.

        The timed wait records an expiry when it ends on the clock, and that is the ordinary
        path. It is not the only one. The wait can also end on a generation deciding to continue
        as new, and it returns then without looking at whether a deadline came due while it
        waited: the flag would still be unset, the attempt would still be active, and a boundary
        that only ever checked the flag would carry the whole thing across and let the next
        execution write an ending the clock of this one had already reached.

        So this asks the question the flag cannot answer: is any armed deadline at or before the
        time this generation is at now. The order is fixed rather than incidental, because two
        that came due together have to be recorded in the same order on a replay.
        """
        now = self._now_ms()
        for attempt_id, deadline in sorted(self._armed_deadlines().items()):
            if deadline <= now:
                self._expire(attempt_id, deadline)
        self._end_expired()

    def _end_expired(self) -> None:
        """End every attempt whose deadline has passed, once nothing is part way through.

        Ending is held back from the moment the timer fires for one reason: the attempt may be
        holding a call to a world this stream cannot see. Finalizing under one would be this
        generation deciding an effect it cannot observe, so the expiry waits for that call to
        come back rather than cancelling it. What ends the wait is the stream falling quiet,
        which a call that returns does and which a resume that ends the grant by name does too.

        The expiry is dropped where it stopped applying. A filing that reached the stream first
        owns the attempt, and this never takes one back from a seal.
        """
        for attempt in list(self._attempts.values()):
            if not attempt.deadline_expired:
                continue
            if (
                self._generation_state != OPEN
                or self._draining
                or attempt.state != ACTIVE
                or attempt.terminal_request_id is not None
            ):
                attempt.deadline_expired = False
                continue
            if not self._quiet():
                continue
            self._finalize(attempt, DEADLINE)

    # Continuing before the service's cap.

    def _count_update(self, *, repeatable: bool = False, claim: bool = False) -> None:
        """Count one accepted Update, and latch the turnover when the count says to.

        This runs at the top of every handler, which is after the service accepted the Update
        and after any validator let it through, so what it counts is exactly what the service
        counts against its cap. A refusal raised inside a handler is counted for the same
        reason: the service counted it too.

        The service publishes a suggestion of its own, one boolean with three possible causes,
        and this generation cannot tell which of them fired. It is taken as a second trigger
        rather than the trigger: a count kept here is deterministic and replays, and it is the
        one that has to be right.

        The marker this code's branch is gated by is read here, once, at the moment the decision
        is taken and never before it. Where it is read decides what an execution that predates
        the branch does. Read at the top of the run method it would be read during every replay
        of such a history, answered no, and remembered as no, so the generation could never
        acquire a turnover when live traffic reached it. Read here, a history that never latched
        never asks, and the first ask is the live Update that latches.

        What the answer is used for matters as much as when it is asked. An execution whose
        replay latches and is then told the branch is not available cannot turn over, and it
        must not start holding traffic back either: it has no boundary to bring about, so
        gating would leave it unable to serve and unable to continue, which is worse than the
        cap it was trying to avoid. So the answer is kept, and everything the latch switches on
        reads it rather than the latch.
        """
        self._updates += 1
        # And the same counting once a barrier stands, which is a separate allowance rather than a
        # share of the one above: a generation that has latched for a turnover refuses a fork, and
        # one that has committed a barrier never turns over, so the two are never both in force.
        if self._fork is not None:
            self._post_barrier += 1
            return
        if self._turnover_requested:
            self._post_latch += 1
            if repeatable:
                self._post_latch_repeats += 1
            if claim:
                self._post_latch_claims += 1
            return
        if self._updates >= TURNOVER_TRIGGER or workflow.info().is_continue_as_new_suggested():
            self._turnover_requested = True
            self._turnover_available = workflow.patched(
                "generation-continues-before-the-update-cap"
            )

    def _reaching_for_a_boundary(self) -> bool:
        """Whether this generation is holding traffic back in order to reach a boundary.

        Three things have to be true together, and the middle one is the one that is easy to
        forget. The generation has to have decided to turn over. The branch that turns it over
        has to be available to this execution, which an execution recorded before the branch
        existed is told it is not. And the carrier has to still fit, because a generation that
        has measured one too large to hand on is serving out the execution it is in.

        Where any of the three fails, this generation holds nothing back and behaves exactly as
        it did before there was a boundary to reach. That is the whole of what an execution the
        branch is unavailable to should do: it cannot continue, so it must keep serving, and
        gating it would leave it able to do neither.

        It is also what a caller waits on. A transport rejected for a boundary reads this to
        tell a generation working towards one from a generation that has stopped, so it says
        the same thing to the gate and to the wait.
        """
        return self._turnover_available and self._turnover_refused is None

    def _admit(
        self,
        progress: bool,
        *,
        repeatable: bool = False,
        claim: bool = False,
        completion: bool = False,
    ) -> None:
        """Reject an Update that cannot bring this generation to its boundary.

        Once the generation is reaching for a boundary, the only Updates worth accepting are the
        ones that can make it quiet. Everything else is rejected here, before acceptance, which
        the service charges nothing for: a rejected Update writes no history and spends none of
        the cap. The transport waits and sends the same request, under the same Update ID, at
        the execution that comes next, so the agent sees latency and never a refusal.

        The reserve is a ceiling on the whole of it, and it is not the only one. Several of the
        operations on the whitelist can be sent again and again under fresh identifiers: a
        confirmation mints a new one every time by design, and a claim refused on a stale epoch
        can be made once more. A ceiling on the total alone would let those spend every slot,
        and the end of a held grant, which nothing else can send, would then be refused for the
        want of a slot that a repeat had taken. So the repeatable ones have a small allowance of
        their own inside the reserve, and what is left is kept for the work that is named: the
        presentation commit of the one message that can be pending, the end of the one grant
        that can be held, and the retry of the one seal that can be prepared.

        A generation that has still to authorize the lineage it was cut from rejects every one of
        them, whatever it is reaching for. It is the same shape of answer for the same reason:
        nothing the caller sent is wrong, a rejection costs the generation nothing, and the same
        request under the same identifier reaches a child that has done its own first work. Reads
        are Queries and reach none of this, which is what leaves a controller able to watch a
        gated child without spending anything.

        This runs in a validator, so it is synchronous and reads without writing. Every
        precondition it reads is read again in the handler, because state can move between the
        two and only the handler's reading decides anything.
        """
        self._require_authorized_lineage()
        self._admit_after_a_barrier(completion=completion)
        if not self._reaching_for_a_boundary():
            return
        if not progress or self._post_latch >= ADMISSION_RESERVE:
            raise TurnoverPending()
        if repeatable and self._post_latch_repeats >= ADMISSION_REPEATS:
            raise TurnoverPending()
        if claim and self._post_latch_claims >= ADMISSION_CLAIMS:
            raise TurnoverPending()

    def _admit_after_a_barrier(self, *, completion: bool) -> None:
        """Bound what a parked parent accepts, and keep the reserve for the class that can spend it.

        The whitelist is smaller than a turnover's, because the parent is already quiet rather than
        working towards quiet: it admits the fork request under its own identity and the fresh
        logical retries that can finish an incomplete fork, and nothing else. Everything a
        controller wants to read is a Query, which reaches none of this and charges nothing, so a
        permanently unavailable child service cannot turn polling into exhaustion of the reserve.

        The reserve therefore bounds the whole of what that one class may spend rather than capping
        the class inside it. Its slots are kept for the work this parent is named for, the way a
        turnover's are kept for the presentation commit and the end of a held grant, and nothing
        else here can take one. The counter is charged at the top of every accepted handler and read
        here, so what bounds the reserve is the work the service admitted rather than the work this
        generation guessed at.
        """
        if self._fork is None:
            return
        if not completion:
            raise ForkBarrier("this generation has forked and serves nothing further")
        if self._fork_admits_nothing:
            raise ForkBarrier("this fork admits no further work")
        if self._post_barrier >= FORK_RESERVE:
            raise ForkBarrier("this fork has spent its recovery reserve")

    def _turnover_ready(self) -> bool:
        """Whether this generation may continue as new at this exact moment.

        The boundary is stricter than either lock. Eight active attempts do not block it and a
        deadline still in the future does not block it: the first is state and the second is an
        absolute time that crosses and is armed again. What blocks it is anything part way
        through, because the far side of a boundary is a fresh history that cannot finish it: a
        message offered and not yet attested to, a grant held for a call to a world this stream
        cannot see, a seal prepared and not committed, an expiry that came due and could not be
        applied, and any handler still running, whose answer a boundary would strand.

        Whether this execution may take the branch at all was settled once, at the latch, and is
        read back rather than asked again. A generation that has already measured a carrier too
        large to hand on does not measure it again either: the carrier grows with what the
        generation serves and the ceiling does not move, so asking a second time would spend an
        activation to reach the same answer, and asking at every quiet point would spend all of
        them.
        """
        if not self._reaching_for_a_boundary():
            return False
        return self._at_a_boundary() and workflow.all_handlers_finished()

    def _boundary_is_out_of_reach(self) -> Optional[str]:
        """Why this generation can no longer get to the boundary it decided to take, if it cannot.

        The gate bounds what is accepted after the latch, and what it bounds is attempts rather
        than successes. The work that clears a boundary can fail and be sent again: a presentation
        whose references the store cannot produce is refused, the transport builds a fresh
        attestation for the same message and sends it again, and every one of those is an accepted
        Update. Nothing about that is misbehaviour, and it is the transport's supported repair
        path.

        Once an allowance that gates the work which could clear the boundary is spent, nothing
        that would clear it will be admitted again. If the generation is already quiet the
        boundary is taken and this never runs. If it is not, then the thing it is holding can
        only be cleared by an Update that will now be refused, and the boundary it is holding
        traffic back for can never happen. That is what this recognises, and it answers with
        which allowance it was so a launcher can say which.

        A handler still running is the case that is not yet lost, and it is excluded. A claim
        reading the store finishes on its own, without any further Update, and the generation may
        be quiet the moment it does.
        """
        if not self._reaching_for_a_boundary():
            return None
        if not workflow.all_handlers_finished():
            return None
        if self._at_a_boundary():
            return None
        if self._post_latch >= ADMISSION_RESERVE:
            return ADMISSION_EXHAUSTED
        # And the same thing one allowance smaller. A claim is the only call on the whitelist
        # that no other call can substitute for: an owner that went away leaves a grant that
        # only its replacement can end, and only a claim installs a replacement. A claim can
        # also fail without installing anything, because it reads the store before it swaps the
        # epoch and a store that cannot produce what it is asked for refuses the claim. Four of
        # those spend the claim allowance without anybody having taken the generation over, and
        # the claim that would have succeeded once the store came back is then refused for the
        # want of a slot. The total is nowhere near spent, so the line above does not see it.
        #
        # The other allowance is not here, and that is deliberate rather than an oversight. What
        # the calls sharing it do is read the state through the write path, bind a consumer and
        # close the queue, and none of those clears anything a boundary waits for: a pending
        # message needs its presentation commit, a held grant needs its end, a prepared seal
        # needs its filing, and every one of those draws on the reserve at large. So spending
        # that allowance closes no route, and giving a boundary up because it was spent would be
        # giving one up that was still perfectly reachable.
        if self._post_latch_claims >= ADMISSION_CLAIMS:
            return RECOVERY_CLAIMS_EXHAUSTED
        return None

    def _give_up_on_the_boundary(self, reason: str) -> None:
        """Stop holding traffic back for a boundary that can no longer be reached.

        This is the same ending as a carrier that will not fit, reached a different way, and it
        behaves the same way on purpose: the generation records why, releases the gate, and goes
        back to serving out the execution it is in. What it must not do is keep holding traffic
        back. A generation that can neither clear what it is holding nor hand itself on would be
        wedged where it stands, which is worse than the cap the boundary existed to avoid.

        The reason is recorded where a launcher reads it, because the residual is real: this
        generation will serve until it reaches the service cap, and a run that ends there ends
        without the boundary that was supposed to carry it past. Which allowance ran out is part
        of that reason, because the two say different things about the run: one is a generation
        whose clearing work failed over and over, and the other is a generation nobody could take
        over while the store was unreadable.
        """
        self._turnover_refused = reason

    def _at_a_boundary(self) -> bool:
        """Whether nothing about this generation is part way through."""
        return self._boundary_clause() is None

    def _boundary_clause(self) -> Optional[str]:
        """Which condition of a quiet boundary does not hold, or nothing where they all do.

        The clause is named rather than counted because a caller that asked for one and was
        refused has to be able to say which condition failed without reading a history. What is
        checked is unchanged: the generation is open, not draining, has not presented Done, holds
        no pending message, has no operation in flight, holds no environment grant, has no attempt
        sealing, and has no deadline due or overdue.
        """
        if self._generation_state != OPEN:
            return "generation_state"
        if self._draining:
            return "draining"
        if self._done_presented:
            return "done_presented"
        if self._pending is not None:
            return "pending_message"
        if self._operation_in_flight:
            return "operation_in_flight"
        if self._environment_call is not None:
            return "environment_grant"
        # A deadline that has come due is not something to carry across, whether or not its
        # expiry has been written down yet. The flag is the record of a clock that was read;
        # the comparison beside it is the clock itself, and a boundary that trusted only the
        # record would cross with an ending owed and let the next execution write it.
        now = self._now_ms()
        for attempt in self._attempts.values():
            if attempt.state == SEALING:
                return "attempt_sealing"
            if attempt.deadline_expired or (
                attempt.deadline_at is not None and attempt.deadline_at <= now
            ):
                return "deadline_due"
        return None

    def _turn_over(self) -> None:
        """Hand this generation to a fresh execution, or record why it could not be handed on.

        The replaced start is the original one with the projection beside it, so the identity a
        resume is held to does not move and the seal key an environment deduplicates on does
        not either. It is measured as the service will encode it, with the converter the service
        was configured with, because what has to fit is the argument and not the projection
        inside it.

        A carrier that will not fit is not a fault and it is not a drain. The generation records
        the size it measured and keeps serving: the run is outside the profile this package
        supports, the launcher reads the marker and says so, and nothing about the refusal is
        visible to the agent.
        """
        projection = self._projection()
        version = carrier_version(self._start, projection)
        replaced = replace(
            self._start,
            carry=pack_carrier(projection, workflow.payload_converter(), version=version),
        )
        argument = continuation_argument(replaced, workflow.payload_converter(), version)
        encoded = workflow.payload_converter().to_payloads([argument])[0].ByteSize()
        if encoded > TURNOVER_PAYLOAD_CEILING_BYTES:
            self._turnover_refused = CARRIER_TOO_LARGE
            self._turnover_refused_bytes = encoded
            return
        workflow.continue_as_new(argument)

    def _projection(self) -> CarriedProjection:
        """Write out everything the next execution has to find, in canonical order.

        Every unordered collection here is sorted, and every ordered one keeps its order. The
        distinction is not tidiness: this value becomes a command in a history that has to
        replay to the same bytes, and a set serialized in the order a hash table happened to
        hold it would not.

        Three groups of things are absent, each for its own reason. What the start already says
        is not repeated, because the start rides beside this. What a legal boundary forbids is
        not carried, because there is none of it to carry. And the identifiers already issued
        are not carried, because they are derivable from the ordinal that is.

        The value is built here and written outside, because what version it is written under is
        a question about the generation's configuration and its evidence together, and the writer
        answers it from both.
        """
        return CarriedProjection(
            configuration_hash=self._configuration_hash,
            ownership_epoch=self._ownership_epoch,
            fencing_token_hash=self._fencing_token_hash,
            ownership_claims=self._ownership_claims(),
            consumer_id=self._consumer_id,
            claim_epoch=self._claim_epoch,
            cursor=self._cursor,
            queue_closed=self._queue_closed,
            hidden_ordinal=self._hidden_ordinal,
            seal_ordinal=self._seal_ordinal,
            wait_count=self._wait_count,
            wait_reasons=dict(sorted(self._wait_reason_counts().items())),
            offer_count=self._offer_count(),
            eligibility_count=self._eligibility_count(),
            handed_out_attempt_ids=sorted(self._handed_out()),
            activity_ordinal=self._activity_ordinal,
            verification_batches=self._verification_batches,
            attempts=[
                CarriedAttempt(
                    attempt_id=attempt_id,
                    **{
                        name: getattr(attempt, name)
                        for name in _CARRIED_ATTEMPT_FIELDS
                        if name not in _RAW_ATTEMPT_FIELDS
                    },
                    **_carried_source(attempt),
                )
                for attempt_id, attempt in sorted(self._attempts.items())
            ],
            obligations=[
                CarriedObligation(
                    attempt_id=attempt_id,
                    state=obligation.state,
                    materialized=obligation.materialized,
                    # The bytes stay only while an offer could still carry them. An obligation
                    # that has been presented, or that ended without being rendered, will never
                    # be offered again, and its candidate is the largest thing in here.
                    candidate=(
                        obligation.candidate
                        if obligation.state in _OFFERABLE_OBLIGATION
                        else None
                    ),
                    pending_preparation=obligation.pending_preparation,
                )
                for attempt_id, obligation in sorted(self._obligations.items())
            ],
            presented=list(self._presented.values()),
            committed_blobs=sorted(set(self._committed_blobs)),
            pull_requests=self._carried_bindings(self._pull_requests),
            info_requests=self._carried_bindings(self._info_requests),
            terminal_requests=self._carried_bindings(self._terminal_requests),
            finalize_requests=[
                CarriedFinalization(
                    request_id=request_id,
                    identity=bound.identity,
                    receipt=bound.receipt,
                )
                for request_id, bound in sorted(self._finalize_requests.items())
            ],
            attestations=[
                CarriedAttestation(
                    attestation_id=attestation_id,
                    identity=identity,
                    ack=self._attestations[attestation_id],
                )
                for attestation_id, identity in sorted(self._attestation_identities.items())
            ],
            journal=[
                [
                    update_id,
                    answer.handler,
                    answer.epoch,
                    answer.kind,
                    answer.row,
                    answer.code,
                    answer.type_name,
                    answer.message,
                    # A recipe carries an offer instead of its bytes. The bytes are dropped
                    # here and made again at restore out of what the start already holds,
                    # which is what keeps a whole roster of offers from being written twice.
                    "" if answer.recipe else answer.value,
                    answer.recipe,
                ]
                for update_id, answer in sorted(self._journal.items())
            ],
            turnovers=self._turnovers + 1,
            # In commit order, which is the order they were appended in. Sorting them would lose
            # the one thing the list says that a set of them would not: which refusal a recovery
            # answered, and which of two refusals of one operation came first.
            operation_failures=list(self._operation_failures),
        )

    def _carried_bindings(self, table: Dict[str, _Bound]) -> List[CarriedBinding]:
        """One request table, written out with the bytes of what nobody can ask for again.

        A binding answers a retry of the request that made it, and it answers with the reserved
        bytes only while the message is unpresented. Afterwards the same request is told the
        message has been presented, and the bytes are never read again.

        At a legal boundary every bound message has been presented: a message is pending from the
        moment it is offered until its attestation commits, and a pending message is the first
        thing the boundary refuses. So the bytes are dropped here, and they are the largest thing
        the table would otherwise carry: one whole task body per attempt, for the life of the
        generation. The identity and the message's identifier stay, because those are what the
        retry is judged on.
        """
        return [
            CarriedBinding(
                request_id=request_id,
                identity=bound.identity,
                message=(
                    replace(bound.message, visible_text="")
                    if bound.message.message_id in self._presented
                    else bound.message
                ),
            )
            for request_id, bound in sorted(table.items())
        ]

    def _next_activity_id(self) -> str:
        """The ID for the next Activity this generation schedules, counted across the chain."""
        ordinal = self._activity_ordinal
        self._activity_ordinal += 1
        return str(ordinal)

    # The Updates a gateway calls.

    @workflow.update
    async def claim_ownership(self, claim: OwnershipClaim) -> OwnershipReceipt:
        """Take the generation from whoever held it, by compare and swap on the epoch.

        The exact claim sent again is answered with what it was answered with the first time,
        whether that was a receipt or a refusal. Within one execution the service does that
        itself; across a boundary it cannot, because its cache of an Update's outcome belongs to
        the execution that accepted it. A claim replayed after a boundary would otherwise read
        the epoch it already moved and be told it was fenced by itself.

        This is the first call a writer makes, at creation and again at every resume, and it is
        the only way an epoch moves. The compare is the epoch the claimant read: a claimant that
        read a stale one loses without changing anything, so two would-be owners resolve to one.
        The swap installs the new token's hash, which is what every later call is held to.

        The configuration is checked before the swap. A claimant resuming something other than
        what is running here is refused with nothing touched, because a generation whose queue,
        roster, plan, capacity, or versions have moved is a different generation and continuing
        it under the old history would serve a configuration nobody committed to.

        Every claim reads the store before it swaps, the first one included. Every reference a
        committed presentation carried was verified when that presentation committed, and a
        resume is the moment to ask again, because a history citing bytes the store can no longer
        produce is not one to hand a new owner. A creation has references of its own before it
        has served anything: the preimage of every policy it resolved to and of every cell of
        every matched family it declares is a required object, and a generation whose descriptor
        was never installed is one that would serve work while its record could not say what its
        bodies were allowed to contain or what the arm it is a leg of was a comparison against.
        Asking at creation is what makes that a refusal to start rather than something a later
        resume discovers. The read covers what the writer being replaced commits while it is
        running, because that is part of the same history.

        An active attempt is restored to active only when nothing has happened since the
        checkpoint it would be restored from. That is a precondition this call checks rather
        than assumes: an attempt the generation has granted a call to a world under is one whose
        world has moved past its checkpoint, and a claimant that has not put that world back
        would carry on in a different one, seal it, and have it graded. So the claim has to say
        which attempts it restored and from which checkpoint, and a claim that says nothing
        about an attempt in that state is refused with nothing touched. Restoring is the
        claimant's, because the world is not something this generation can reach; naming what
        was restored, and holding a claim to it, is this generation's.

        The claim also releases the stream. A call that was in flight when this epoch arrived is
        not in flight for the new owner: it will fail its own epoch check before it can commit,
        and until then it must not be holding the stream against the owner that replaced it. An
        environment call is the one hold carried over rather than released, because that call is
        changing a world this stream cannot reach and nothing here can fence it. The new owner
        ends it by name before the generation grants another one.
        """
        self._count_update(claim=True)
        return await self._answering(
            OWN, claim.previous_epoch, lambda: self._claim_ownership(claim)
        )

    @claim_ownership.validator
    def _claim_ownership_admitted(self, claim: OwnershipClaim) -> None:
        # A claim is on the progress whitelist, and it is the one call on it that nothing else
        # can substitute for: an environment call abandoned by an owner that went away is ended
        # by name, and only the owner that replaces it can end it. It is repeatable, because a
        # claim refused on a stale epoch can be made once more, so it has an allowance rather
        # than the reserve at large. That allowance is its own and not the one confirmations
        # share, because confirmations spending theirs must not lock out the claim a recovery
        # depends on.
        self._admit(True, claim=True)

    async def _claim_ownership(self, claim: OwnershipClaim) -> OwnershipReceipt:
        """Check the claim, read the store, and swap the epoch.

        The claim is one operation whatever it takes to make it. Its identity is built from the
        claimant and the epoch it witnessed and from nothing about the transport, so a claim
        refused over a missing object and the claim that repeats it after a repair, under a fresh
        fencing token and a different Update, are one operation with two rows rather than two
        operations with one each.
        """
        self._check_claim(claim)
        operation = ownership_claim_operation_identity(claim.claimant_id, claim.previous_epoch)
        if self._committed_blobs:
            await self._verify_committed_blobs(operation, claim)
            # The store was read outside this transition, so the claim is checked again on the
            # way back in. A claimant that another one overtook while this read was running
            # loses the swap it witnessed, and the swap below still has no await inside it.
            self._check_claim(claim)
        replaced = self._fencing_token_hash
        previous = self._ownership_epoch
        self._ownership_epoch = previous + 1
        self._fencing_token_hash = _token_hash(claim.fencing_token)
        self._operation_ticket += 1
        self._operation_in_flight = self._environment_call is not None
        if self._environment_call is not None:
            self._environment_ticket = self._operation_ticket
        restored = sorted(claim.restored_checkpoints)
        for attempt_id in restored:
            # The world is back at the checkpoint, so nothing has happened since it again.
            self._attempts[attempt_id].environment_calls = 0
        self._ownership.append(
            _Ownership(
                ownership_epoch=self._ownership_epoch,
                previous_epoch=previous,
                witnessed_epoch=claim.previous_epoch,
                fencing_token_hash=self._fencing_token_hash,
                replaced_token_hash=replaced,
                claimant_id=claim.claimant_id,
                consumer_claim_hash=self._start.consumer_claim_hash,
                reason=claim.reason,
                restored_attempts=restored,
            )
        )
        # The swap is what recovers a claim that was refused over an object somebody has since
        # put back. The claim certifies what it verified and nothing beyond it: neither eligible
        # body is in the set it read, so one can succeed while a selected body is still missing,
        # and what recovers that delivery is the delivery's own check passing.
        self._note_recovery(operation, OWNERSHIP_CLAIM, self._ownership_epoch)
        receipt = OwnershipReceipt(
            ownership_epoch=self._ownership_epoch,
            previous_epoch=previous,
            fencing_token_hash=self._fencing_token_hash,
            configuration_hash=self._configuration_hash,
            claimant_id=claim.claimant_id,
            reason=claim.reason,
        )
        return receipt

    @workflow.update
    async def claim_consumer(self, claim: ConsumerClaim, writer: Writer) -> ConsumerReceipt:
        """Bind the generation's one logical consumer.

        The same claim presented twice returns the same receipt, because a lost response must
        not cost a caller its stream. A different one is refused, before any message is
        offered and without touching state. The exact Update sent again is answered from what it
        was answered with, so a refusal stays a refusal across a boundary rather than becoming a
        receipt because the conditions behind it moved.
        """
        self._count_update(repeatable=True)
        return await self._answering(
            BIND, writer.ownership_epoch, _as_awaited(lambda: self._claim_consumer(claim, writer))
        )

    @claim_consumer.validator
    def _claim_consumer_admitted(self, claim: ConsumerClaim, writer: Writer) -> None:
        # Binding the consumer clears nothing a boundary is waiting for, so it waits.
        self._admit(False, repeatable=True)

    def _claim_consumer(self, claim: ConsumerClaim, writer: Writer) -> ConsumerReceipt:
        if self._generation_state != OPEN:
            raise StreamProtocolError("closed_stream")
        self._require_writer(writer)
        if claim.protocol_version != PROTOCOL_VERSION:
            raise StreamProtocolError("unsupported_version")
        if claim.claim_hash != self._start.consumer_claim_hash:
            raise StreamProtocolError("consumer_conflict")
        if self._consumer_id is None:
            self._consumer_id = claim.consumer_id
            self._claim_epoch = 1
        elif self._consumer_id != claim.consumer_id:
            raise StreamProtocolError("consumer_conflict")
        receipt = ConsumerReceipt(
            consumer_id=self._consumer_id,
            claim_epoch=self._claim_epoch,
            initial_cursor=self._start.initial_cursor,
            configuration_hash=self._configuration_hash,
        )
        return receipt

    @workflow.update
    async def pull(self, request: PullRequest, writer: Writer) -> OfferedMessage:
        """Return the one message this request is entitled to.

        A retry of the same request gets its own result back while that result is unpresented,
        and an error afterwards. A new request gets a new selection, but only from a current
        cursor and only when nothing is outstanding: a second request never inherits the first
        one's offer.

        It holds the stream across an await, which one entry path needs: the first offer in this
        execution of a payload obligation whose attempt carries a committed source reads the
        store before the bytes are reserved. Every other pull awaits nothing and behaves as it
        did.
        """
        self._count_update()
        return await self._answering(
            PULL,
            writer.ownership_epoch,
            lambda: self._locked_await(writer, self._pull, request, writer),
        )

    @pull.validator
    def _pull_admitted(self, request: PullRequest, writer: Writer) -> None:
        # A pull that selects anything reserves it, which is the opposite of a quiet point.
        self._admit(False)

    @workflow.update
    async def info(self, request: InfoRequest, writer: Writer) -> OfferedMessage:
        """Answer how much of this generation's queue there is, where it declared the tool.

        It is an Update and not a Query, and everything about how it behaves follows from that.
        The counts are read inside the transition that mints the record, so what the agent is
        told describes the stream that told it; the answer is offered like a pull's, so a retry
        of the same request gets the same bytes rather than a fresh reading; and a presentation
        commits it, so what the agent was told is in the ledger beside every other message it
        read. A Query would have none of those and could be answered by a transport the stream
        has already fenced.
        """
        self._count_update()
        return await self._answering(
            ASK,
            writer.ownership_epoch,
            _as_awaited(lambda: self._locked(writer, self._info, request)),
        )

    @info.validator
    def _info_admitted(self, request: InfoRequest, writer: Writer) -> None:
        # An answer minted here is reserved and then presented, so it waits like a pull.
        self._admit(False)

    @workflow.update
    async def seal_attempt(self, request: SealRequest, writer: Writer) -> OfferedMessage:
        """End an attempt, or say why the filing was not one this tool accepts.

        Nothing about the seal becomes authoritative until the last transition below, which
        contains no await. The acknowledgement is built there and returned after it, so a
        crash anywhere earlier leaves a stream that has not acknowledged anything.
        """
        self._count_update()
        return await self._answering(
            SEAL,
            writer.ownership_epoch,
            lambda: self._locked_await(writer, self._seal, request, writer),
        )

    @seal_attempt.validator
    def _seal_attempt_admitted(self, request: SealRequest, writer: Writer) -> None:
        # A seal prepared and not committed is one of the things a boundary refuses to cross,
        # and the exact filing that prepared it is the only thing that can finish it. Every
        # other filing waits. The state read here is read again in the handler.
        prepared = self._attempts.get(request.metadata.attempt_id)
        self._admit(
            prepared is not None
            and prepared.state == SEALING
            and prepared.terminal_request_id == request.metadata.request_id
        )

    @workflow.update
    async def commit_presentation(
        self, commit: PresentationCommit, writer: Writer
    ) -> PresentationAck:
        """Accept the harness's attestation that the exact offered bytes were delivered.

        Delivery is what the attestation says: the bytes were handed to the transport that
        carries them. What the model consumed is attested by the harness transcript and not
        here. Everything in the attestation is checked against something already held here, so
        this is a verification and not a report. The cursor advances only on the way out.
        """
        self._count_update()
        return await self._answering(
            PRESENT,
            writer.ownership_epoch,
            lambda: self._locked_await(writer, self._commit_presentation, commit, writer),
        )

    @commit_presentation.validator
    def _commit_presentation_admitted(
        self, commit: PresentationCommit, writer: Writer
    ) -> None:
        # Attesting to the message this generation is holding is exactly the work that makes a
        # boundary quiet, so it is admitted after the latch and nothing else here is.
        self._admit(
            self._pending is not None and self._pending.message.message_id == commit.message_id
        )

    @workflow.update
    async def finalize_attempt(
        self, request: FinalizeRequest, writer: Writer
    ) -> AttemptFinalized:
        """End one attempt that nothing is going to finish, and say why.

        This is a controller call and not a tool: no model reaches it, and it mints no result,
        so an attempt ended this way leaves the transcript exactly as it was and the next pull
        is the only thing that says anything happened. What it may end is narrow. An attempt
        whose terminal was accepted is the seal's, whatever became of it, and an attempt with a
        result outstanding is a caller's until that result is presented.
        """
        self._count_update()
        return await self._answering(
            FINALIZE,
            writer.ownership_epoch,
            _as_awaited(lambda: self._locked(writer, self._finalize_requested, request)),
        )

    @finalize_attempt.validator
    def _finalize_attempt_admitted(self, request: FinalizeRequest, writer: Writer) -> None:
        # Ending an attempt clears nothing a boundary is waiting for, so it waits.
        self._admit(False)

    @workflow.update
    async def close_queue(self, writer: Writer) -> QueueClosed:
        """Close the queue to insertion. It revokes nothing and seals nothing."""
        self._count_update(repeatable=True)
        return await self._answering(
            CLOSE, writer.ownership_epoch, _as_awaited(lambda: self._close(writer))
        )

    @close_queue.validator
    def _close_queue_admitted(self, writer: Writer) -> None:
        # Closing the queue is on the whitelist: a controller held off here would be held off
        # for the whole boundary. Nothing stops a controller sending it again under another
        # identifier, so it is repeatable and spends the allowance the repeats share.
        self._admit(True, repeatable=True)

    @workflow.update
    async def begin_environment_call(
        self, call: EnvironmentCall, writer: Writer
    ) -> EnvironmentLease:
        """Hold the generation for one call to a world this stream cannot see.

        An ordinary environment call never reaches the stream, so the stream cannot serialize
        it against its own Updates the way it serializes those against each other, and it
        cannot refuse it afterwards. What it can do is decide before the call happens and stay
        held while it does: the generation has to be open and held by nobody else, nothing may
        be outstanding, and the attempt has to be one this generation is still serving. The
        decision and the change are one thing that way, rather than a question answered and a
        world changed after the answer stopped being true.

        The stream is given back through :meth:`end_environment_call` rather than by returning,
        so every call the generation could otherwise take meanwhile is refused.

        Ownership is checked here like it is everywhere else, and this is the call that makes
        it reach the world: a writer that was fenced cannot change an environment either, and
        without this its only unfenced path would be the one the stream never sees.

        The exact Update sent again is answered from what it was answered with, receipt or
        refusal, rather than being granted a second time. A grant is a change to a world this
        stream cannot see and a durable count against the attempt, so a second one for the same
        call would leave a hold nothing out there matches. And a begin refused while something
        was outstanding must stay refused after the thing that was outstanding has gone, or the
        same Update ID would turn a refusal into a grant.
        """
        self._count_update()
        return await self._answering(
            GRANT,
            writer.ownership_epoch,
            _as_awaited(lambda: self._begin_environment_call(call, writer)),
        )

    @begin_environment_call.validator
    def _begin_environment_call_admitted(self, call: EnvironmentCall, writer: Writer) -> None:
        # A grant holds the stream, which is the opposite of a quiet point.
        self._admit(False)

    def _begin_environment_call(self, call: EnvironmentCall, writer: Writer) -> EnvironmentLease:
        ticket = self._take_lock(writer)
        try:
            try:
                require_opaque_id("call_id", call.call_id)
            except WireFormatError as error:
                raise StreamProtocolError("invalid_message") from error
            if self._pending is not None:
                raise StreamProtocolError("outstanding_response")
            attempt = self._attempts.get(call.attempt_id)
            if attempt is None or attempt.state != ACTIVE:
                raise StreamProtocolError("invalid_attempt")
        except BaseException:
            # Nothing was granted, so nothing of this call is holding the stream.
            self._release_lock(ticket)
            raise
        self._environment_call = call.call_id
        self._environment_ticket = ticket
        # The grant is the last thing this stream knows about that world. What the call does to
        # it happens somewhere this stream cannot see and is never reported back, so the grant
        # is counted as the change it authorized, and a later claim is held to it.
        self._attempts[call.attempt_id].environment_calls += 1
        return EnvironmentLease(
            call_id=call.call_id, attempt_id=call.attempt_id, cursor=self._cursor, held=True
        )

    @workflow.update
    async def end_environment_call(
        self, call: EnvironmentCall, writer: Writer
    ) -> EnvironmentLease:
        """Give the generation back once that call has settled, whatever became of it.

        The answer is the same whether or not this call was the one holding the stream, so a
        caller that never learned whether its grant arrived can give back a lease it is not
        sure it holds. Only the call named in the grant releases it: a lease taken from a
        caller is not one that caller may hand on afterwards, and a writer that was fenced
        while holding one has already had the stream taken from it by the claim.

        The exact Update sent again is answered with the lease it was answered with, its held
        flag and its cursor as they were returned. Across a boundary the grant is gone by
        construction, so an end replayed there would report that it held nothing when the
        execution before it reported that it held the stream.
        """
        self._count_update()
        return await self._answering(
            RELEASE,
            writer.ownership_epoch,
            _as_awaited(lambda: self._end_environment_call(call, writer)),
        )

    @end_environment_call.validator
    def _end_environment_call_admitted(self, call: EnvironmentCall, writer: Writer) -> None:
        # Ending the grant this generation is holding is what makes the boundary quiet. An end
        # for a grant it is not holding clears nothing, so it waits.
        self._admit(self._environment_call == call.call_id)

    @workflow.update
    async def confirm_state(self, writer: Writer) -> StreamState:
        """Report the generation's state to a caller the stream has to admit first.

        The query below answers whoever asks, because a read costs the generation nothing. It
        is therefore also answered for a writer this generation has fenced, which is the wrong
        question for a transport that is holding something it decided under an earlier epoch
        and is about to hand over. That caller asks here instead, and a fenced one is refused
        here as it would be on any other write, so reading around the stream stops being a way
        to serve what the stream would not.

        It takes no lock, because what it is asked about is often something outstanding, and it
        changes nothing, so asking twice is the same as asking once.
        """
        self._count_update(repeatable=True)
        return await self._answering(
            CONFIRM, writer.ownership_epoch, _as_awaited(lambda: self._confirm(writer))
        )

    @confirm_state.validator
    def _confirm_state_admitted(self, writer: Writer) -> None:
        # The confirmation that follows a claim is on the whitelist with the claim: an owner
        # that has just taken the generation over reads through the write path before it hands
        # anything on, and holding that off would hold off the recovery it is part of. It mints
        # a fresh identifier every time by design, so it is the most repeatable call there is
        # and it spends the allowance the repeats share rather than the reserve at large.
        self._admit(True, repeatable=True)

    # The fork: one platform operation over two generations, which no agent reaches.

    @workflow.update
    async def fork_generation(self, request: ForkRequest) -> ForkReceipt:
        """Fence this generation, prepare two children from its whole observed prefix, start them.

        The request's own handler makes the barrier transition, against the ledger of accepted
        handlers this generation keeps. The alternative is coordinating from the run method, which
        forces the exact Update to answer with admission rather than with the receipt and then needs
        a second public operation to deliver the receipt; the exact Update answers with the complete
        receipt instead, which is what makes the outcome journal a usable recovery path.

        Admission is decided in the validator, synchronously and without writing. Everything the
        validator read is read again here, then the one stage that has to be awaited runs, then the
        whole boundary is read a third time, and only then are the starts built and the barrier and
        the prepared record committed in one transition. There is no await anywhere between that
        final reading and the commit.
        """
        self._count_update()
        return await self._answering(
            FORK, self._ownership_epoch, lambda: self._fork_the_generation(request)
        )

    @fork_generation.validator
    def _fork_generation_admitted(self, request: ForkRequest) -> None:
        # A fork is the one call a parked parent still admits, and it is not on the turnover
        # whitelist: a generation that has latched refuses a fork as retryable until its successor
        # exists, and the clause below is what says so. Everything the handler decides is decided
        # again there; nothing here writes.
        self._admit(True, completion=True)
        self._refuse_a_fork(request)

    async def _fork_the_generation(self, request: ForkRequest) -> ForkReceipt:
        """Run one fork, in the order the barrier's own safety depends on.

        A retry that finds the barrier already standing skips straight to the children, because the
        record is what says which children this fork is: the starts are rebuilt from the same
        request over a parent that has not moved since, and each rebuilt start is held to the digest
        the record committed for it. Recovery preserves children that were created and replaces
        none that was.
        """
        self._refuse_a_fork(request, excluding=(self._update_id(),))
        if self._fork is None:
            available = await self._read_the_fork_evidence(request)
            # The stream could move while that ran, because the barrier does not stand yet, so the
            # whole boundary is read again here and the commit below follows with no await between.
            self._refuse_a_fork(request, excluding=(self._update_id(),))
            self._refuse_unavailable_evidence(available)
            children = self._built_children(request)
            self._commit_the_barrier(
                request, children, self._measured_replies(request, children)
            )
        built = self._built_children(request)
        await self._start_the_children(request, built)
        return self._fork_answer(request, built)

    async def _read_the_fork_evidence(self, request: ForkRequest) -> ForkAvailability:
        """Read the three objects one fork requires, before the barrier and nowhere else.

        An ordinary claim reads the manifest and deliberately verifies neither eligible body, so
        without this the fork would commit a barrier over two references whose bytes nobody had
        read. The manifest is named among the reads rather than left implicit in the commitment it
        binds to, because a carried descriptor is a value in state and says nothing about whether
        the object under its digest is still in the store.

        It takes its identifier from the fork's own namespace, like every other fork-only
        invocation, so a child's ordinary Activity numbering equals the number its inherited prefix
        left and maps onto an unforked twin's.

        Both shapes it transmits are measured as the configured converter would encode them, the
        question before it goes and the answer as it arrives, because every shape one fork
        transmits is measured before the barrier commits.
        """
        attempt = self._attempts[request.source_attempt_id]
        manifest = self._source_descriptor(request.source_attempt_id)
        if not attempt.source_commitment:
            raise ForkRefused(
                FORK_UNRECOVERABLE_EVIDENCE,
                f"the attempt {request.source_attempt_id} committed no source to fork over",
            )
        root = self._start.blob_root
        if root is None:
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                "a fork copies an object closure and this generation keeps no store",
            )
        converter = workflow.payload_converter()
        asked = ForkAvailabilityInput(
            blob_root=root,
            source_commitment=attempt.source_commitment,
            body_references=[
                manifest.cells[cell].sha256 for cell in sorted(ELIGIBLE_CELLS)
            ],
        )
        _fork_shape_measured(
            "the prebarrier availability read",
            asked,
            converter,
            ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
        )
        available: ForkAvailability = await workflow.execute_activity(
            fork_availability_activity,
            asked,
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
            activity_id=self._fork_activity(request.fork_id, FORK_AVAILABILITY_STEP),
        )
        _fork_shape_measured(
            "what the prebarrier availability read returned",
            available,
            converter,
            ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
        )
        return available

    def _source_descriptor(self, attempt_id: str) -> SourceArtifactManifest:
        """The descriptor one attempt committed, which a fork reads its cells out of."""
        manifest = self._attempts[attempt_id].source_artifact
        if manifest is None:
            raise ForkRefused(
                FORK_UNRECOVERABLE_EVIDENCE,
                f"the attempt {attempt_id} carries no committed descriptor",
            )
        return manifest

    def _refuse_unavailable_evidence(self, available: ForkAvailability) -> None:
        """Refuse a barrier over an object nobody could read, and say which one."""
        if available.missing:
            raise ForkRefused(
                FORK_REPAIRABLE_ABSENCE,
                f"the store could not produce {sorted(available.missing)}",
            )

    def _measured_replies(
        self, request: ForkRequest, built: List[Tuple[PreparedChild, StreamStart]]
    ) -> _ReplyBounds:
        """Measure the shapes this fork answers with, and return the bounds proved for them.

        Every shape one fork transmits is measured before the barrier commits, and three of them
        are replies rather than requests: the receipt this Update returns, the result each child's
        own start answers with, and the proof a child's origin verification asks for and is
        answered with. Measuring any of them only once it exists would be measuring it after the
        barrier that made it inevitable.

        None can be measured exactly, because each carries a value the service has still to
        generate and because the proof goes on being answered while the fork moves. So the rule is
        exact preflight for every known value and a proved encoded upper bound for everything else:
        each run id is allowed its declared ceiling, the proof's state members and the start's own
        observation of what it did are taken at their widest, the converter's own wrapper is
        measured empty for the receipt, and those bounds are what have to fit. They are returned so
        the record can retain them, and each reply that actually arrives is measured against the
        retained value rather than against one derived from itself.
        """
        converter = workflow.payload_converter()
        origin = 0
        start_reply = 0
        for row, _start in built:
            _fork_shape_measured(
                f"the origin verification question of child {row.child_ordinal}",
                VerifyForkOriginInput(
                    parent_workflow_id=workflow.info().workflow_id,
                    parent_run_id=workflow.info().run_id,
                    fork_id=request.fork_id,
                    child_ordinal=row.child_ordinal,
                    child_workflow_id=row.child_workflow_id,
                ),
                converter,
                ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
            )
            widest = fork_origin_bound(request.fork_id, row, converter)
            if widest > TURNOVER_PAYLOAD_CEILING_BYTES:
                raise ForkRefused(
                    FORK_CONFIGURATION_VIOLATION,
                    f"what the origin verification answers for child {row.child_ordinal} is "
                    f"bounded at {widest} bytes once the run id and the states this parent can "
                    f"answer in are allowed for, and one shape this operation transmits may be "
                    f"{TURNOVER_PAYLOAD_CEILING_BYTES}",
                )
            origin = max(origin, widest)
            answered = fork_start_bound(row, converter)
            if answered > TURNOVER_PAYLOAD_CEILING_BYTES:
                raise ForkRefused(
                    FORK_CONFIGURATION_VIOLATION,
                    f"what the start of child {row.child_ordinal} answers with is bounded at "
                    f"{answered} bytes once the run id and either observation of what it did are "
                    f"allowed for, and one shape this operation transmits may be "
                    f"{TURNOVER_PAYLOAD_CEILING_BYTES}",
                )
            start_reply = max(start_reply, answered)
        known = self._the_receipt(request, built)
        _fork_shape_measured(
            "the known parts of the fork receipt",
            known,
            converter,
            ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
        )
        bound = fork_receipt_bound(known, converter)
        if bound > TURNOVER_PAYLOAD_CEILING_BYTES:
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                f"the fork receipt is bounded at {bound} bytes once every child's run id is "
                f"allowed for, and one shape this operation transmits may be "
                f"{TURNOVER_PAYLOAD_CEILING_BYTES}",
            )
        return _ReplyBounds(receipt=bound, origin=origin, start=start_reply)

    def _commit_the_barrier(
        self,
        request: ForkRequest,
        built: List[Tuple[PreparedChild, StreamStart]],
        bounds: _ReplyBounds,
    ) -> None:
        """Fence the parent and record the fork, in one transition with no await inside it.

        The barrier, the fence and the prepared record are one thing, so a crash between two of them
        is impossible: a parent that is fenced has a record naming the children it prepared, and a
        record that exists names a parent nothing can move.

        The horizon is recorded here rather than derived from the close, because the close cannot
        be bounded: the service schedules each execution's history for deletion from that
        execution's own close time, so a child that failed an hour from now is deleted on its own
        clock while this parent is still prepared. The deadline beside it is a bound on this
        parent's own work and never a bound on when it actually closes.
        """
        self._fork = PreparedFork(
            fork_id=request.fork_id,
            request_digest=fork_request_digest(request),
            checkpoint_manifest_reference=request.checkpoint_manifest_reference,
            parent_workflow_id=workflow.info().workflow_id,
            parent_run_id=workflow.info().run_id,
            children=len(built),
            child_records=[row for row, _ in built],
            receipt_bound=bounds.receipt,
            origin_bound=bounds.origin,
            start_bound=bounds.start,
            status=FORK_PREPARED,
        )
        self._generation_state = FORKED
        self._fencing_token_hash = None
        now = self._now_ms()
        self._fork_deadline_at = now + FORK_PREPARATION_BOUND_MS
        self._fork_horizon_at = now + FORK_AUTHORITY_HORIZON_MS

    async def _start_the_children(
        self, request: ForkRequest, built: List[Tuple[PreparedChild, StreamStart]]
    ) -> None:
        """Create every child the record names, one at a time and never twice.

        The parent builds the starts and an Activity creates the children, rather than a controller
        creating them from a value the parent returned, because a controller that could compose a
        start could compose a projection holding a score nothing filed.

        The class a child is in is written before its start is dispatched, so a start whose reply is
        lost leaves that child attempted with its existence unconfirmed rather than never attempted.
        A timeout and a failure are never proof that no child exists, and neither of them ever moves
        a child into the class that would let a replacement be created under its identity.

        One thing a start can come back with is a decision rather than a failure. Something else
        running under a child's derived identity is authenticated and permanent, so it is turned
        into this fork's own refusal here, before the outcome is journalled, and the ending is
        retained with the fork: the journal keeps a type and a message and no details, and a reason
        that arrived as an Activity's own failure would be a reason nothing downstream could read.
        """
        for row, start in built:
            if self._child_row(row.child_ordinal).existence == CONFIRMED_EXISTING:
                continue
            if self._fork_admits_nothing:
                # The parent has stopped admitting fork work, so no new start is scheduled and
                # this Update fails rather than answering with a receipt naming a child nobody
                # created. The failure is journalled and stands under this identifier for ever.
                raise ForkBarrier(
                    f"this fork admits no further work and child {row.child_ordinal} is unstarted"
                )
            self._note_child(row.child_ordinal, EXISTENCE_UNCONFIRMED)
            try:
                started: ForkChildStarted = await workflow.execute_activity(
                    start_fork_child_activity,
                    StartForkChildInput(
                        fork_id=request.fork_id,
                        child_ordinal=row.child_ordinal,
                        child_workflow_id=row.child_workflow_id,
                        task_queue=workflow.info().task_queue,
                        start=start,
                    ),
                    schedule_to_close_timeout=_FORK_START_TIMEOUT,
                    retry_policy=_ACTIVITY_RETRY,
                    activity_id=self._fork_activity(request.fork_id, FORK_START_STEP),
                )
            except ActivityError as failure:
                raise self._what_the_start_decided(row, failure) from failure
            self._within_the_start_bound(started)
            self._note_child(
                row.child_ordinal, CONFIRMED_EXISTING, run_id=started.child_run_id
            )
        self._move_the_fork(FORK_CHILDREN_CONFIRMED)

    def _what_the_start_decided(
        self, row: PreparedChild, failure: ActivityError
    ) -> BaseException:
        """Return what one failed start is: a decision this fork ends on, or the failure it was.

        A start that could not be made is infrastructure and stays exactly what it was, retried by
        the policy it was made under and journalled as itself. A start that found another execution
        under this child's derived identity is neither: the reading is authenticated, no later
        reading reaches another answer, and replacing what is there is the one move a fork never
        makes. So that one is the fork's own permanent refusal, recorded with the fork before the
        outcome is kept, and the child it names keeps its identity and its class.
        """
        cause = failure.cause
        if (
            not isinstance(cause, ApplicationError)
            or cause.type != ORIGIN_DISAGREEMENT_FAILURE
        ):
            return failure
        return self._fork_ends_permanently(
            FORK_ORIGIN_DISAGREEMENT,
            f"the identity of child {row.child_ordinal} is held by another execution: "
            f"{cause.message}",
        )

    def _fork_ends_permanently(self, reason: str, clause: str) -> ForkRefused:
        """Retain one permanent ending with the fork, and return the refusal that says it.

        A decision is never retried, so what a controller meets afterwards is this decision rather
        than the work that would reach it again: the status a closed parent answers with carries
        it, and a fresh logical retry that still reaches a handler is refused from the record
        before anything is scheduled. The children the fork already made are left exactly as they
        are, because an ending is not a claim that they never existed.
        """
        assert self._fork is not None
        if self._fork.conflict_reason is None:
            self._fork = replace(
                self._fork, conflict_reason=reason, conflict_clause=clause
            )
        self._move_the_fork(FORK_CONFLICTED)
        assert self._fork.conflict_reason is not None
        return ForkRefused(self._fork.conflict_reason, self._fork.conflict_clause)

    def _within_the_start_bound(self, started: ForkChildStarted) -> None:
        """Hold what a start answered with to the bound proved for it before the barrier."""
        assert self._fork is not None
        try:
            check_start_within_bound(
                started, self._fork.start_bound, workflow.payload_converter()
            )
        except WireFormatError as oversize:
            raise ForkRefused(FORK_CONFIGURATION_VIOLATION, str(oversize)) from oversize

    def _the_receipt(
        self, request: ForkRequest, built: List[Tuple[PreparedChild, StreamStart]]
    ) -> ForkReceipt:
        """The complete evidence one fork answers with, from the rows it was handed.

        Before the barrier the rows are the ones just built and their run ids are empty, which is
        what the bound is proved over; afterwards they are the record's own, run ids and all. The
        shape is the same either way, which is what makes the earlier measurement a bound on the
        later value rather than a measurement of a different thing.
        """
        return ForkReceipt(
            fork_id=request.fork_id,
            parent_workflow_id=workflow.info().workflow_id,
            parent_run_id=workflow.info().run_id,
            checkpoint_manifest_reference=request.checkpoint_manifest_reference,
            children=len(built),
            child_receipts=[
                self._child_receipt(row, start, request.source_attempt_id)
                for row, start in built
            ],
            boundary_evidence=sorted(self._fork_boundary_evidence(request)),
        )

    def _fork_answer(
        self, request: ForkRequest, built: List[Tuple[PreparedChild, StreamStart]]
    ) -> ForkReceipt:
        """Build the complete receipt, measure it against the bound proved before the barrier."""
        assert self._fork is not None
        receipt = self._the_receipt(
            request,
            [(self._child_row(row.child_ordinal), start) for row, start in built],
        )
        try:
            check_receipt_within_bound(
                receipt, self._fork.receipt_bound, workflow.payload_converter()
            )
        except WireFormatError as oversize:
            raise ForkRefused(FORK_CONFIGURATION_VIOLATION, str(oversize)) from oversize
        self._fork_receipt = receipt
        self._move_the_fork(FORK_COMPLETE)
        return receipt

    def _child_receipt(
        self, row: PreparedChild, start: StreamStart, source_attempt_id: str
    ) -> ForkChildReceipt:
        """What the fork says about one child, read from that child's own start.

        The next task is read from the child rather than from the parent, so a reader can check that
        both children are about to work the same bytes rather than being told that they are. Same B
        is an identity here and not a comparison: neither child may change the manifest, the
        assignments or the release plan, so both carry the identical item.

        The roster row the schedule selected is what names that task, because the roster row is
        what a reader joining this receipt to a child's own schedule holds: the attempt the row
        stands for is a second identity of the same selection and finding it is the roster's job
        rather than a reader's.
        """
        roster_row, item = self._child_next_task(start, source_attempt_id) or (None, None)
        return ForkChildReceipt(
            child_ordinal=row.child_ordinal,
            child_workflow_id=row.child_workflow_id,
            child_run_id=row.child_run_id or "",
            configuration_hash=configuration_hash(start),
            complete_start_digest=row.complete_start_digest,
            origin_digest=row.origin_digest,
            branch_slot=row.branch_slot,
            target_cell=row.target_cell,
            selected_body_reference=row.selected_body_reference,
            acknowledged_cursor=self._cursor,
            projection_digest=(
                "" if start.fork_origin is None else start.fork_origin.projection_digest
            ),
            start_differences=list(row.start_differences),
            next_task_body_sha256=(
                "" if item is None else sha256(item.body.encode("utf-8")).hexdigest()
            ),
            next_assignment_id="" if roster_row is None else roster_row.assignment_id,
        )

    def _child_next_task(
        self, start: StreamStart, source_attempt_id: str
    ) -> Optional[Tuple[Assignment, TaskItem]]:
        """The task one child serves next, asked of that child's own schedule.

        The roster is not the queue of what is left. An attempt may end where it was planned,
        never offered and never handed out, and a controller may ask for exactly that: the task
        is on the roster for ever afterwards and no pull will ever be given it. So the first
        item nobody was handed is not the task a child works, and a receipt naming it would be
        telling a reader to compare two children against a task neither will see.

        What is asked instead is the plan, at the point a child stands at once the inherited
        payload has been presented: the declared gate, the declared order and the states each
        child carries. The attempt and obligation states are read from this generation, because
        those are the states the child is built to carry and the transformation moves none of
        them; the schedule, the roster and the item are read from the child's own start, which
        is what makes two receipts something a reader can compare rather than one value copied
        twice. Capacity does not enter it: the boundary refuses a fork with any live attempt, so
        nothing is in flight at the cut and nothing the child inherits is holding a slot.

        Both halves of the selection come back, because they are two identities of it and each
        one answers a different question: the roster row is what the schedule chose and what a
        receipt names it by, and the item behind it is the bytes that row hands over. A row whose
        task the manifest does not hold is no selection at all.
        """
        roster = list(start.assignments) or assignments_for(start.tasks, start.release)
        presented = {
            obligation.item.payload_position
            for obligation in self._obligations.values()
            if obligation.state == PRESENTED
        }
        inherited = self._obligations.get(source_attempt_id)
        if inherited is not None:
            presented.add(inherited.item.payload_position)
        ready = eligible_tasks(
            start.release,
            roster,
            ScheduleView(
                offered_attempts=frozenset(
                    key for key, value in self._attempts.items() if value.state != PLANNED
                ),
                sealed_attempts=frozenset(
                    key
                    for key, value in self._attempts.items()
                    if value.state in (SEALED, ACK_PRESENTED)
                ),
                presented_payload_positions=frozenset(presented),
            ),
        )
        if not ready:
            return None
        serves = min(ready, key=lambda row: order_key(start.release, TASK, row))
        item = next(
            (one for one in start.tasks if one.attempt_id == serves.attempt_id), None
        )
        return None if item is None else (serves, item)

    def _fork_boundary_evidence(self, request: ForkRequest) -> List[str]:
        """The clauses the parent actually checked, named so a reader need not read a history."""
        return [
            f"acknowledged_cursor={request.acknowledged_cursor}",
            f"attestation={request.attestation_id}",
            f"projection_digest={request.projection_digest}",
            f"source_attempt={request.source_attempt_id}",
        ]

    # Building one child, which is pure and produces the same two starts every time it runs.

    def _built_children(
        self, request: ForkRequest
    ) -> List[Tuple[PreparedChild, StreamStart]]:
        """Build both children's complete starts and the rows the parent commits about them.

        Each plan, each child start and the input that starts a child are measured here, as the
        configured converter would encode it, which is before the barrier on the first pass and
        before any child is started on every pass after it. A wrapper can exceed the limit while
        everything it wraps fits, so what is measured is the converter's actual output for each
        shape that crosses. The other shapes this operation transmits are measured where they
        cross: the request in the validator that admits it and again in the runtime that sends it,
        the availability read and the origin question at their own Activities, and the receipt as
        it is built. Every one of those measurements refuses the fork rather than raising, because
        bytes the service would never carry are a decision about a request.

        The build is pure and the parent does not move once its barrier stands, so a recovery run
        derives the same two starts. Where a record already stands, each rebuilt start is held to
        the digest that record committed for it, which is what makes recovery a resumption of the
        same fork rather than a second one.
        """
        projection = self._projection()
        converter = workflow.payload_converter()
        built: List[Tuple[PreparedChild, StreamStart]] = []
        for ordinal, plan in enumerate(request.child_plans, start=1):
            _fork_shape_measured(
                f"the plan of child {ordinal}",
                plan,
                converter,
                ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
            )
            start, origin = self._one_child(request, plan, ordinal, projection)
            _fork_shape_measured(
                f"the start of child {ordinal}",
                start,
                converter,
                ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
            )
            row = PreparedChild(
                child_ordinal=ordinal,
                child_workflow_id=child_workflow_id(
                    identity_namespace=workflow.info().namespace,
                    parent_workflow_id=workflow.info().workflow_id,
                    fork_id=request.fork_id,
                    child_ordinal=ordinal,
                ),
                complete_start_digest=complete_start_digest(start),
                origin_digest=origin_digest(origin),
                branch_slot=plan.branch_slot,
                target_cell=plan.target_cell,
                selected_body_reference=self._child_body_reference(request, plan),
                consumer_claim_hash=plan.consumer_claim_hash,
                hidden_execution_id=plan.hidden_execution_id,
                start_differences=list(origin.start_differences),
            )
            _fork_shape_measured(
                f"the start request for child {ordinal}",
                StartForkChildInput(
                    fork_id=request.fork_id,
                    child_ordinal=ordinal,
                    child_workflow_id=row.child_workflow_id,
                    task_queue=workflow.info().task_queue,
                    start=start,
                ),
                converter,
                ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
            )
            built.append((self._agreed_child(row), start))
        self._refuse_two_children_of_one_store(built)
        return built

    def _refuse_two_children_of_one_store(
        self, built: List[Tuple[PreparedChild, StreamStart]]
    ) -> None:
        """Refuse a pair of children the request kept apart and the build brought together.

        Each child is given a store of its own, and what makes two stores two is the place the
        children are actually created in rather than the text the plans were written in. Neither
        derivation on the way there is injective over strings: a directory keeps its objects under
        a fixed name beneath it, and what opens that place drops a separator at the end, an empty
        component and a component naming the directory it already stands in. So two directories
        that differ as text can name one store, and they are compared as the place they are.

        A parent that committed a barrier over such a pair would have fenced itself to create two
        generations promised separate objects and a manifest each, and the second attachment would
        find the first one's manifest already published in the place its own belongs. So the built
        values are compared here, before anything is committed, where a refusal installs no barrier
        and creates no child. The store's own directory is its parent, so comparing the stores
        compares the directories with them.
        """
        seen: Dict[str, int] = {}
        for row, start in built:
            store = self._one_store(start.blob_root or "")
            first = seen.get(store)
            if first is not None:
                raise ForkRefused(
                    FORK_CONFIGURATION_VIOLATION,
                    f"the children {first} and {row.child_ordinal} were given directories that "
                    f"name one store, {store}, and each child of one fork keeps its objects and "
                    "its manifest in a place of its own",
                )
            seen[store] = row.child_ordinal

    def _one_store(self, location: str) -> str:
        """The place one store is, as the one spelling two of them are compared as."""
        try:
            return canonical_location(location)
        except WireFormatError as error:
            raise ForkRefused(FORK_CONFIGURATION_VIOLATION, str(error)) from error

    def _agreed_child(self, row: PreparedChild) -> PreparedChild:
        """Hold a rebuilt row to the one the record committed, and keep what the record knows."""
        if self._fork is None:
            return row
        recorded = self._child_row(row.child_ordinal)
        for name in ("child_workflow_id", "complete_start_digest", "origin_digest"):
            if getattr(recorded, name) != getattr(row, name):
                raise ForkRefused(
                    FORK_REQUEST_CONFLICT,
                    f"the record for child {row.child_ordinal} names a {name} this request does "
                    "not rebuild",
                )
        return recorded

    def _one_child(
        self,
        request: ForkRequest,
        plan: ForkChildPlan,
        ordinal: int,
        projection: CarriedProjection,
    ) -> Tuple[StreamStart, ForkOrigin]:
        """Return one child's complete start and the lineage record that rides in it."""
        provenance = self._start.provenance
        if provenance is None:
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                "a child keeps the authority its parent's rows were registered under, and this "
                "generation names none",
            )
        attempt = self._attempts[request.source_attempt_id]
        manifest = self._source_descriptor(request.source_attempt_id)
        rows = list(plan.dispositions)
        digest = roster_digest(rows)
        bare = replace(
            self._start,
            consumer_claim_hash=plan.consumer_claim_hash,
            hidden_execution_id=plan.hidden_execution_id,
            blob_root=child_blob_root(plan.run_directory),
            served_slot=plan.branch_slot,
            dispositions=rows,
            provenance=replace(provenance, roster_digest=digest),
            carry=None,
            fork_origin=None,
        )
        try:
            check_child_configuration(self._start, bare)
        except WireFormatError as error:
            raise ForkRefused(FORK_CONFIGURATION_VIOLATION, str(error)) from error
        origin = ForkOrigin(
            parent_workflow_id=workflow.info().workflow_id,
            parent_run_id=workflow.info().run_id,
            parent_configuration_hash=self._configuration_hash,
            child_configuration_hash=configuration_hash(bare),
            parent_execution_ordinal=self._start.execution_ordinal,
            parent_hidden_execution_id=self._start.hidden_execution_id,
            acknowledged_cursor=self._cursor,
            projection_digest=request.projection_digest,
            attestation_id=request.attestation_id,
            acknowledged_visible_sha256=request.acknowledged_visible_sha256,
            checkpoint_manifest_reference=request.checkpoint_manifest_reference,
            source_attempt_id=attempt.item.attempt_id,
            source_seal_id=attempt.seal_id or "",
            source_submission_digest=attempt.submission_digest or "",
            source_canonicalization_version=self._start.canonicalization_version,
            source_score=attempt.score,
            source_seal_ordinal=attempt.seal_ordinal or 0,
            source_graded_evidence=attempt.graded_evidence or "",
            source_commitment=attempt.source_commitment or "",
            source_artifact_references=sorted(
                reference.sha256 for reference in manifest.cells.values()
            ),
            branch_slot=plan.branch_slot,
            dispositions_digest=digest,
            start_differences=start_difference_projection(self._start, bare),
            parent_turnovers=self._turnovers,
            fork_id=request.fork_id,
            child_ordinal=ordinal,
            children=len(request.child_plans),
        )
        carrier = child_carrier(
            projection,
            selections=[
                ChildSelection(
                    attempt_id=request.source_attempt_id,
                    cell=plan.target_cell,
                    policy_digest=self._child_policy(plan, request.source_attempt_id),
                )
            ],
        )
        holding = replace(bare, fork_origin=origin)
        composed = replace(
            holding,
            carry=pack_carrier(
                carrier,
                workflow.payload_converter(),
                version=carrier_version(holding, carrier),
            ),
        )
        self._refuse_a_child_its_own_restore_would(ordinal, composed, origin, carrier)
        return composed, origin

    def _refuse_a_child_its_own_restore_would(
        self,
        ordinal: int,
        start: StreamStart,
        origin: ForkOrigin,
        carrier: CarriedProjection,
    ) -> None:
        """Hold a start this parent built to the checks the child's own constructor makes of it.

        The structural transformation says the fields a child may change and what each of them
        becomes; it says nothing about whether the result is a generation. A plan whose target cell
        and whose own delivering row name different cells passes every classification and is
        refused by the child's restore, and a parent that committed a barrier over it would have
        fenced itself to create a generation nothing can start.

        So the reusable pure checks are run here, over the complete start with its carrier, before
        anything is committed: an invalid future configuration is a clean refusal that installs no
        barrier and creates no child, and it is the same code the child runs rather than a second
        opinion about it.

        The ordinary admission every constructor makes comes first, for the same reason and against
        the same objection. The classification says which fields a child may change and what each
        of them becomes; it does not say the schedule and the policy still describe a generation
        afterwards. A row saying the platform stamped it inside an experiment is one such start:
        the transformation admits it, the fork's own clauses admit it, and no generation is ever
        created from it.
        """
        try:
            _check_start(start)
        except StreamProtocolError as refusal:
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                f"the start this fork would build for child {ordinal} is one no generation is "
                f"created from: {refusal.code}",
            ) from refusal
        try:
            check_fork_origin(origin)
            _check_carried_receipts(start, carrier)
        except (WireFormatError, ApplicationError) as error:
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                f"the start this fork would build for child {ordinal} is one that child's own "
                f"restore refuses: {error}",
            ) from error

    def _child_policy(self, plan: ForkChildPlan, attempt_id: str) -> str:
        """The policy digest of the row this child resolves the inherited obligation under."""
        for row in plan.dispositions:
            if row.attempt_id == attempt_id and row.branch_slot == plan.branch_slot:
                return row.policy_digest or ""
        raise ForkRefused(
            FORK_CONFIGURATION_VIOLATION,
            f"the plan for {plan.branch_slot} resolves nothing for {attempt_id}",
        )

    def _child_body_reference(self, request: ForkRequest, plan: ForkChildPlan) -> str:
        """The committed entry one child delivers, read off the descriptor carried in state."""
        manifest = self._source_descriptor(request.source_attempt_id)
        return manifest.cells[plan.target_cell].sha256

    # The record the parent keeps about its children, and the two states it can move through.

    def _child_row(self, ordinal: int) -> PreparedChild:
        """One child's row out of the record this fork committed."""
        assert self._fork is not None
        for row in self._fork.child_records:
            if row.child_ordinal == ordinal:
                return row
        raise ForkRefused(
            FORK_REQUEST_CONFLICT, f"this fork prepared no child {ordinal}"
        )

    def _note_child(
        self, ordinal: int, existence: str, *, run_id: Optional[str] = None
    ) -> None:
        """Move one child into the class the evidence puts it in, and never out of one.

        The row that comes out of this is the one the parent answers a gated child's origin
        question with for as long as its window admits one, and the run id in it is the first value
        of that proof the service rather than this generation produced. So it is measured here,
        where it arrives, against the bound proved for it before the barrier: a reply over its
        bound is a recorded failure carrying the measurement rather than a silent oversize.
        """
        assert self._fork is not None
        moved = [
            replace(row, existence=existence, child_run_id=run_id or row.child_run_id)
            if row.child_ordinal == ordinal
            else row
            for row in self._fork.child_records
        ]
        for row in moved:
            if row.child_ordinal == ordinal and row.child_run_id:
                try:
                    check_origin_within_bound(
                        self._fork.fork_id,
                        row,
                        self._fork.origin_bound,
                        workflow.payload_converter(),
                    )
                except WireFormatError as oversize:
                    raise ForkRefused(
                        FORK_CONFIGURATION_VIOLATION, str(oversize)
                    ) from oversize
        self._fork = replace(self._fork, child_records=moved)

    def _move_the_fork(self, status: str) -> None:
        """Record where this fork now stands, without ever rewriting an ending."""
        assert self._fork is not None
        if self._fork.status in _FORK_ENDINGS:
            return
        self._fork = replace(self._fork, status=status)

    def _fork_activity(self, fork_id: str, step: str) -> str:
        """The identifier one fork-only Activity invocation is scheduled under.

        It comes from the fork's own namespace rather than from this generation's Activity ordinal,
        so a child's ordinary numbering equals the number its inherited prefix left and can be
        mapped onto an unforked twin's. The ordinal counts the attempts at one step of one fork, so
        two children and a logical retry of either are told apart.
        """
        key = f"{fork_id}.{step}"
        self._fork_steps[key] = self._fork_steps.get(key, 0) + 1
        return fork_activity_id(fork_id=fork_id, step=step, ordinal=self._fork_steps[key])

    # What a parent does after its barrier stands, and the two endings it commits itself.

    async def _park_for_the_fork(self) -> None:
        """Wait for the fork to finish, its reserve to be spent, or its preparation to expire.

        The parent returns once every child start is confirmed and the fork's own outcome is
        settled, so the Worker stops holding it and its records stay readable from the closed
        execution. Keeping it open for the life of the link would hold an execution and a workflow
        cache slot for the whole link, and it is not what this does.

        Expiry and exhaustion are one transition. The parent admits no further work and schedules
        no new start work, the start work already pending is bounded by its own schedule-to-close
        timeout, every accepted Update settles with a journalled outcome, and the abandonment is
        committed only after the ledger settles: an abandonment committed while a start Activity
        was still running would record an outcome the parent has not seen.
        """
        assert self._fork is not None
        while self._fork.status not in _FORK_SETTLED:
            remaining = self._fork_deadline_at - self._now_ms()
            if remaining <= 0:
                await self._abandon_the_fork(EXPIRED_PREPARATION)
                return
            if self._post_barrier >= FORK_RESERVE:
                await self._abandon_the_fork(SPENT_RECOVERY_RESERVE)
                return
            try:
                await workflow.wait_condition(
                    lambda: self._fork is not None
                    and self._fork.status in _FORK_SETTLED
                    or self._post_barrier >= FORK_RESERVE,
                    timeout=timedelta(milliseconds=remaining),
                )
            except asyncio.TimeoutError:
                pass

    async def _abandon_the_fork(self, reason: str) -> None:
        """Commit the terminal abandoned status, after the last in-flight outcome has settled."""
        self._fork_admits_nothing = True
        await workflow.wait_condition(workflow.all_handlers_finished)
        assert self._fork is not None
        if self._fork.status in _FORK_SETTLED:
            return
        self._fork = replace(self._fork, status=FORK_ABANDONED, abandoned_reason=reason)

    # A child's own first work, which happens before it owns or serves anything.

    async def _verify_the_lineage(self) -> None:
        """Read the parent's row for this child, and open the gate on what it says.

        The read is an Activity rather than a workflow call because the SDK's external workflow
        handle exposes signal and cancel and no Query at all, and it is recorded in this child's own
        history so a replay performs no fresh client I/O and reaches the same answer. It never waits
        for the parent to complete: creating a child confirms that the service created it, and
        readiness is a later state read separately.

        The question and the answer are both measured, like every other shape one fork transmits,
        which is why the answer is one row and never the parent's state.

        The one failure it keeps rather than raises is a window that has closed. A parent's answers
        stand behind an absolute horizon, so a reading that arrives after it is not a reading to
        try again: this child records that state, keeps its identity and everything it holds, and
        refuses every call that would own or serve it with the reason rather than with the wait.
        """
        origin = self._start.fork_origin
        assert origin is not None
        converter = workflow.payload_converter()
        question = VerifyForkOriginInput(
            parent_workflow_id=origin.parent_workflow_id,
            parent_run_id=origin.parent_run_id,
            fork_id=origin.fork_id,
            child_ordinal=origin.child_ordinal,
            child_workflow_id=workflow.info().workflow_id,
        )
        _fork_shape_measured(
            "the origin verification question",
            question,
            converter,
            ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
        )
        try:
            answer: ForkOriginVerified = await workflow.execute_activity(
                verify_fork_origin_activity,
                question,
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
                retry_policy=_ORIGIN_RETRY,
                activity_id=self._fork_activity(origin.fork_id, FORK_ORIGIN_STEP),
            )
        except ActivityError as error:
            if not _authority_expired(error):
                raise
            self._origin_expired = True
            return
        _fork_shape_measured(
            "what the origin verification answered",
            answer,
            converter,
            ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
        )
        self._admit_lineage(answer.record)

    # What a controller and a child read, both of which are Queries and cost the parent nothing.

    @workflow.query(name="fork_status")
    def fork_status(self, question: ForkStatusQuestion) -> ForkStatusAnswer:
        """Where one fork stands, answered from an execution that may already have closed.

        A closed workflow accepts no Update at all, refused by the service before any validator or
        handler runs, so a logical retry under a fresh identifier cannot reach a handler and a
        custom refusal cannot be raised from one. This is the route that stays: it is keyed by the
        fork id and the canonical request digest, it returns the recorded children for a matching
        request, it names a conflict for a changed one, and it says which state a controller write
        would now meet.

        A fork that finished is answered with the receipt it finished with, and that is the point
        of the route rather than a convenience of it. A retry under a fresh identifier cannot reach
        the outcome journal that holds the original Update's answer, so a completed fork read
        through here would otherwise come back as still in flight from a parent that can accept
        nothing: the evidence exists, and this is what returns it.

        It is a Query, so it charges nothing against the reserve: a permanently unavailable child
        service cannot turn controller polling into exhaustion of the one thing that can finish the
        fork.
        """
        standing = self._fork
        if standing is None or standing.fork_id != question.fork_id:
            return ForkStatusAnswer(
                found=False, conflict=False, parent_state=self._generation_state
            )
        conflict = standing.request_digest != question.request_digest
        return ForkStatusAnswer(
            found=True,
            conflict=conflict,
            parent_state=self._generation_state,
            record=standing,
            receipt=None if conflict else self._fork_receipt,
        )

    @workflow.query(name="fork_child")
    def fork_child(self, fork_id: str, child_ordinal: int) -> ForkOriginVerified:
        """The row this parent committed about one child, and nothing else.

        The answer is small and bounded on purpose: it is recorded in the child's own history, and a
        row that carried the parent's state would put a second copy of that state in every child.

        A question this parent has no answer to is answered rather than refused, with a row naming
        no child, because the caller has to be able to tell an authenticated disagreement from a
        parent it could not read at all.

        The reply is held to the bound the barrier proved for it, here, where it is given. The
        bound was proved over every reply this parent could ever answer with, so the one it
        actually answers with is measured against a value derived from something other than
        itself, and a reply over it says so with the measurement rather than crossing anyway.
        """
        standing = self._fork
        if standing is None or standing.fork_id != fork_id:
            return ForkOriginVerified(
                fork_id=fork_id, fork_status="", record=_no_such_child(child_ordinal)
            )
        for row in standing.child_records:
            if row.child_ordinal == child_ordinal:
                return self._within_the_origin_bound(
                    ForkOriginVerified(
                        fork_id=fork_id, fork_status=standing.status, record=row
                    )
                )
        return self._within_the_origin_bound(
            ForkOriginVerified(
                fork_id=fork_id, fork_status=standing.status, record=_no_such_child(child_ordinal)
            )
        )

    def _within_the_origin_bound(self, answer: ForkOriginVerified) -> ForkOriginVerified:
        """Return one origin reply, having measured it against the bound the barrier retained."""
        assert self._fork is not None
        check_origin_reply_within_bound(
            answer, self._fork.origin_bound, workflow.payload_converter()
        )
        return answer

    # The predicate, read in the validator, in the handler, and once more after the one await.

    def _refuse_a_fork(
        self, request: ForkRequest, *, excluding: Sequence[str] = ()
    ) -> None:
        """Refuse a fork this generation will not take, naming the clause that failed.

        The order is the order the clauses depend on each other in. What the request says about
        itself comes first, because a request this build cannot serve is refused before anything
        reads a generation, and its own encoded size comes with it: admission is decided here, and
        a request whose bytes the service would never carry is a decision this generation can make
        without reading a thing. The execution scope comes next, because a turnover between the
        moment a controller read the checkpoint evidence and the moment it submitted is an ordinary
        event and the answer to it is to read again and resubmit. A barrier that already stands
        comes next, because a fork that has been accepted is answered from its record rather than
        judged again.

        Then the conditions a boundary is, which are the continuation's own plus the ones a fork
        needs and a continuation does not. ``excluding`` is how a handler leaves itself out of the
        ledger: the SDK inserts an Update into its own map before its validator runs and removes it
        in the handler's finally, so a request that waited on the public predicate would wait for
        itself for ever, and the exclusion is this generation's to keep.

        A request arriving when any of this does not hold is refused rather than queued, because a
        fork that waited would hold the generation open against an agent still working.
        """
        try:
            check_fork_request(request)
        except WireFormatError as error:
            raise ForkRefused(FORK_CONFIGURATION_VIOLATION, str(error)) from error
        _fork_shape_measured(
            "the fork request",
            request,
            workflow.payload_converter(),
            ceiling=TURNOVER_PAYLOAD_CEILING_BYTES,
        )
        info = workflow.info()
        if request.parent_workflow_id != info.workflow_id:
            raise ForkRefused(
                FORK_MOVED_EXECUTION,
                f"this request was prepared against {request.parent_workflow_id!r} and this "
                f"generation is {info.workflow_id!r}",
            )
        if request.parent_run_id != info.run_id:
            raise ForkRefused(
                FORK_MOVED_EXECUTION,
                "this request was prepared against an execution other than the one running",
            )
        if request.parent_execution_ordinal != self._start.execution_ordinal:
            raise ForkRefused(
                FORK_MOVED_EXECUTION,
                f"this request was prepared at execution {request.parent_execution_ordinal} and "
                f"this generation is at {self._start.execution_ordinal}",
            )
        if request.parent_configuration_hash != self._configuration_hash:
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                "this request was prepared against another generation's configuration",
            )
        outstanding = sorted(set(self._handlers) - set(excluding))
        if self._fork is not None:
            if self._fork.fork_id != request.fork_id:
                raise ForkRefused(
                    FORK_IN_FLIGHT,
                    f"this generation holds a barrier for the fork {self._fork.fork_id}",
                )
            if self._fork.request_digest != fork_request_digest(request):
                raise ForkRefused(
                    FORK_REQUEST_CONFLICT,
                    f"the fork {request.fork_id} is bound to another checkpoint or other plans",
                )
            # A matching retry is answered from the record rather than judged again, and that is
            # not permission to run beside the handler already finishing this fork. Two handlers
            # admitted here would schedule the same child's start work and spend the same reserve,
            # so a fresh identifier arriving while one is in flight is refused as retryable and
            # comes back when the fork it would have raced is settled.
            racing = sorted(name for name in outstanding if self._handlers[name] == FORK)
            if racing:
                raise ForkRefused(
                    FORK_IN_FLIGHT,
                    f"the handler for {racing[0]} is finishing this fork and has not returned",
                )
            if self._fork.conflict_reason is not None:
                # A fork that ended on a decision is answered with that decision. Sending it again
                # under a fresh identifier is not a second chance at it: nothing a later reading
                # could find would change what was authenticated, so the recorded ending is what
                # comes back rather than the work that would reach it again.
                raise ForkRefused(
                    self._fork.conflict_reason, self._fork.conflict_clause
                )
            return
        if outstanding:
            raise ForkRefused(
                FORK_NOT_QUIET,
                f"the handler for {outstanding[0]} has been accepted and has not finished",
            )
        if self._reaching_for_a_boundary() and self._turnover_requested:
            raise ForkRefused(
                FORK_NOT_QUIET, "this generation has latched for a turnover"
            )
        clause = self._boundary_clause()
        if clause is not None:
            raise ForkRefused(FORK_NOT_QUIET, clause)
        self._refuse_the_fork_clauses(request)
        self._refuse_the_witnesses(request)
        self._refuse_the_plans(request)

    def _refuse_the_fork_clauses(self, request: ForkRequest) -> None:
        """The conditions a fork needs that quiescence does not give it."""
        attempt = self._attempts.get(request.source_attempt_id)
        if attempt is None or attempt.state != ACK_PRESENTED:
            raise ForkRefused(
                FORK_NOT_QUIET,
                f"the attempt {request.source_attempt_id} is not one whose acknowledgement has "
                "been presented",
            )
        for other in self._attempts.values():
            if other.state in (TASK_OFFERED, ACTIVE):
                raise ForkRefused(
                    FORK_NOT_QUIET,
                    f"the attempt {other.item.attempt_id} is {other.state}, so a successor has "
                    "been admitted or a world is open",
                )
        if self._start.capacity != 1:
            # Slots nobody is standing in are not the declared condition. A fork above capacity
            # one would clone attempts with live worlds, and cloning a world needs a world
            # snapshot contract this build does not write, so the declaration is what is read.
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                f"a fork is cut from a generation declaring one live attempt and this one "
                f"declares {self._start.capacity}",
            )
        if self._capacity_in_use() != 0:
            raise ForkRefused(FORK_NOT_QUIET, "this generation is serving a live attempt")
        for attempt_id in self._inherited_receipt_attempts(request):
            if self._last_row_for(attempt_id) is not None:
                raise ForkRefused(
                    FORK_UNRECOVERABLE_EVIDENCE,
                    f"the inherited attempt {attempt_id} reads unavailable, and a fork over "
                    "evidence the parent could not produce hands both children a dependency "
                    "neither can resolve",
                )
        self._refuse_the_inherited_obligation(request)

    def _inherited_receipt_attempts(self, request: ForkRequest) -> List[str]:
        """Every attempt whose receipt evidence both children inherit whole.

        The attempt a fork is cut over is one of them and never all of them. A prefix that withheld
        an earlier receipt keeps that attempt's committed source under the merged contract, both
        children carry it, and neither of them can produce a body of it that the parent could not.
        The prebarrier read opens the source of the one cell a child is selected for, so it
        discharges nothing about the others; what does is the last row naming each of them.
        """
        return sorted(
            {request.source_attempt_id}
            | {
                attempt.item.attempt_id
                for attempt in self._attempts.values()
                if attempt.source_artifact is not None
            }
        )

    def _last_row_for(self, attempt_id: str) -> Optional[OperationFailure]:
        """The standing refusal over one attempt's evidence, where the last row naming it is one.

        A reader takes an attempt's availability from the last row naming it in commit order, so
        that is what is read here rather than the presence of any row at all: an operation that was
        refused and then recovered reads available.
        """
        latest: Optional[OperationFailure] = None
        for row in self._operation_failures:
            if row.attempt_id == attempt_id:
                latest = row
        if latest is None or latest.outcome not in (
            REFUSED_OPERATION,
            UNRECOVERABLE_OPERATION,
        ):
            return None
        return latest

    def _refuse_the_inherited_obligation(self, request: ForkRequest) -> None:
        """Refuse a prefix whose payload obligations are not the one shape a child can inherit.

        Quiet at capacity one does not imply the inherited payload is what a child pulls first: the
        schedule ranks eligible tasks and eligible obligations together, and a plan that puts tasks
        first would serve the next task before the receipt. So the inherited obligation has to be
        the sole unresolved one, materialized and offerable and unpresented, and the schedule's next
        selection has to be precisely that payload.

        A fork over a prefix that already delivered a payload is refused too, so no child has to
        reconcile a new branch slot against a delivered row and the slot provenance question stays
        closed for this build.
        """
        unresolved = sorted(
            attempt_id
            for attempt_id, owed in self._obligations.items()
            if owed.state in UNFULFILLED_OBLIGATION
        )
        if unresolved != [request.source_attempt_id]:
            raise ForkRefused(
                FORK_NOT_QUIET,
                f"a child inherits one unresolved payload and this generation owes {unresolved}",
            )
        owed = self._obligations[request.source_attempt_id]
        if owed.state != ELIGIBLE or not owed.materialized:
            raise ForkRefused(
                FORK_NOT_QUIET,
                f"the inherited payload is {owed.state} and a child is cut over one that is "
                f"{ELIGIBLE}",
            )
        delivered = sorted(
            attempt_id
            for attempt_id, other in self._obligations.items()
            if other.state == PRESENTED
        )
        if delivered:
            raise ForkRefused(
                FORK_CONFIGURATION_VIOLATION,
                f"this generation already delivered the payloads of {delivered}, and this build "
                "forks only a prefix that delivered none",
            )
        selected = self._first_eligible()
        if selected != (PAYLOAD, request.source_attempt_id):
            raise ForkRefused(
                FORK_NOT_QUIET,
                f"the next selection of this generation's schedule is {selected} rather than the "
                "inherited payload",
            )

    def _refuse_the_witnesses(self, request: ForkRequest) -> None:
        """Compare the stream side of the freeze against this generation's own state.

        The message id must be the acknowledgement of the attempt this fork is cut over, the
        visible byte digest must be the one in the presented row, the cursor must be the current
        cursor, and the projection digest must be the current projection hash. The platform
        validates this half and the adapter attests the other; neither side proves the other's, and
        the origin records which manifest the comparison was made against.

        The two records are joined rather than read one beside the other. A presented row and an
        attestation both existing says nothing about whether either is about the other, so the row
        is required to be the named attempt's own acknowledgement and the attestation is required
        to be the one that committed that presentation. Read independently, an attestation of some
        other presentation would pass every one of these comparisons while attesting to nothing
        this fork is being cut at.

        The projection is compared against this generation's own rather than against the one the
        attestation recorded, and the difference is deliberate. The attestation holds the digest at
        the moment the acknowledgement committed, and a lawful ownership resume after that moves
        the digest without touching the acknowledgement, its bytes, its cursor or the attestation,
        because the ownership epoch is inside the projection. The answer to that is the one the
        precedence rule gives: the controller reads the checkpoint evidence again and resubmits
        against what this generation now stands at.
        """
        acknowledgement = self._attempts[request.source_attempt_id].item.ack_message_id
        if request.acknowledgement_message_id != acknowledgement:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                f"this checkpoint names {request.acknowledgement_message_id} and the "
                f"acknowledgement of {request.source_attempt_id} is {acknowledgement}",
            )
        presented = self._presented.get(request.acknowledgement_message_id)
        if presented is None:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                f"this generation presented no message {request.acknowledgement_message_id}",
            )
        if presented.visible_bytes_sha256 != request.acknowledged_visible_sha256:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                "the acknowledgement this checkpoint names went as other bytes than the ones this "
                "generation presented",
            )
        attested = self._attestations.get(request.attestation_id)
        if attested is None or self._attestation_identities.get(request.attestation_id) is None:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                f"this generation holds no attestation {request.attestation_id}",
            )
        if attested.cursor != request.acknowledgement_message_id:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                f"the attestation {request.attestation_id} committed the presentation of "
                f"{attested.cursor} and this checkpoint names {request.acknowledgement_message_id}",
            )
        if request.acknowledged_cursor != self._cursor:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                f"this checkpoint was taken at {request.acknowledged_cursor} and this generation "
                f"stands at {self._cursor}",
            )
        if request.projection_digest != self._projection_hash():
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                "this checkpoint names a projection digest other than this generation's own",
            )

    def _refuse_the_plans(self, request: ForkRequest) -> None:
        """Hold each child plan to the branches this generation declared before it forked."""
        for plan in request.child_plans:
            if plan.branch_slot not in self._start.forkable_slots:
                raise ForkRefused(
                    FORK_UNDECLARED_BRANCH,
                    f"this generation declared {sorted(self._start.forkable_slots)} and a plan "
                    f"names {plan.branch_slot!r}",
                )
            if plan.branch_slot == self._start.served_slot:
                raise ForkRefused(
                    FORK_UNDECLARED_BRANCH,
                    f"a child serves a branch of its own and a plan names {plan.branch_slot!r}, "
                    "which is the branch its parent serves",
                )

    @workflow.query
    def stream_state(self) -> StreamState:
        """Report the generation's state to the harness. Queries write nothing.

        Assignment, release, materialization, eligibility, offer, and presentation are reported
        as six facts. A reader that wants to know whether a payload was delivered has to read
        the presentation count for payloads, which is the only one of the six that says so.
        Whether the model consumed what was delivered belongs to the harness transcript, which
        is the record of that, and no count here answers it.

        It also reports what the generation is holding open, which is what an owner that did not
        open it has to know. A reserved result is owed to one request and no other, a held
        environment call is ended by name, and a prepared seal is continued by the exact filing
        that prepared it, so all three are named here: a replacement process that kept none of
        its predecessor's memory can still learn what is outstanding and which call may finish
        it.

        The checkpoints are the fourth of those, read before a claim rather than after one. Each
        active attempt names the checkpoint it would be restored from, and the ones a claim must
        restore before it may continue them are listed apart: those are the attempts whose world
        this generation has authorized a change to since that checkpoint committed.

        The count behind that list is reported too. An environment call never becomes an Update,
        so the transport that spends an environment's step budget is the only thing that sees it
        run out, and the grants counted here are the only durable record that they happened. A
        transport reads the count rather than keeping one: a budget kept in a process is a budget
        a second process starts over.
        """
        return StreamState(
            generation_state=self._generation_state,
            cursor=self._cursor,
            configuration_hash=self._configuration_hash,
            stream_state_sha256=self._projection_hash(),
            ownership_epoch=self._ownership_epoch,
            fencing_token_hash=self._fencing_token_hash,
            ownership_claims=self._ownership_claims(),
            blob_verification="unchecked" if self._start.blob_root is None else "required",
            consumer_id=self._consumer_id,
            queue_closed=self._queue_closed,
            tasks_remaining=sum(
                1 for attempt in self._attempts.values() if attempt.state == PLANNED
            ),
            capacity=self._start.capacity,
            capacity_in_use=self._capacity_in_use(),
            pending_message_id=None if self._pending is None else self._pending.message.message_id,
            pending_kind=None if self._pending is None else self._pending.message.kind,
            pending_origin=None if self._pending is None else self._pending.origin,
            pending_request_id=None if self._pending is None else self._pending.request_id,
            environment_call=self._environment_call,
            prepared_seals={
                key: value.terminal_request_id
                for key, value in self._attempts.items()
                if value.state == SEALING and value.terminal_request_id is not None
            },
            task_checkpoints={
                key: value.task_start_checkpoint
                for key, value in self._attempts.items()
                if value.task_start_checkpoint is not None
            },
            graded_evidence={
                key: value.graded_evidence
                for key, value in self._attempts.items()
                if value.graded_evidence is not None
            },
            environment_calls={
                key: value.environment_calls
                for key, value in self._attempts.items()
                if value.environment_calls > 0
            },
            restoration_required=self._restoration_required(),
            attempts={key: value.state for key, value in self._attempts.items()},
            obligations={key: value.state for key, value in self._obligations.items()},
            release_plan_id=self._release.release_plan_id,
            release_predicate=self._release.predicate,
            assignment_count=len(self._assignments),
            materialization_count=self._materialized(),
            eligibility_count=self._eligibility_count(),
            offer_count=self._offer_count(),
            presentation_count=len(self._presented),
            payload_delivery_count=sum(
                1 for message in self._presented.values() if message.kind == PAYLOAD
            ),
            wait_count=self._wait_count,
            wait_reasons=self._wait_reason_counts(),
            final_failures={
                key: value.final_failure
                for key, value in self._attempts.items()
                if value.final_failure is not None
            },
            deadline_expired=[
                key for key, value in self._attempts.items() if value.deadline_expired
            ],
            dispositions={
                disposition_key(row): describe_disposition(row)
                for row in self._start.dispositions
            },
            profile=self._start.profile,
            experiment_id=(
                None
                if self._start.provenance is None or not self._start.provenance.experiment_id
                else self._start.provenance.experiment_id
            ),
            turnovers=self._turnovers,
            turnover_requested=self._reaching_for_a_boundary(),
            turnover_refused=self._turnover_refused,
            verifying=self._verifying,
            verification_batches=self._verification_batches,
            unfinished_handlers=len(self._handlers),
            turnover_refused_bytes=self._turnover_refused_bytes,
        )

    @workflow.query
    def answered_update(self, update_id: str) -> AnsweredUpdate:
        """What one exact Update was answered with, for a caller that cannot send it again.

        A transport holding an Update it already sent has two places where it cannot ask the
        handler what became of it. One is a generation that has decided to continue as new and
        is holding traffic back: sending again is rejected until the boundary, and the boundary
        may be a while. The other is a generation that has finished, whose last execution accepts
        no Update at all. Both are places where the answer exists and only the route to it is
        missing, so this is that route.

        It is a Query, so it costs the generation nothing, writes nothing, and can be asked of an
        execution that has closed. What it needs is a Worker able to replay this generation: a
        Query is answered by the code, not by the service alone, so a deployment with no Worker
        for this task queue cannot answer one. That is the limit of this route, and a caller that
        cannot get an answer is left with the failure it already had.
        """
        answer = self._journal.get(update_id)
        if answer is None:
            return AnsweredUpdate(found=False)
        if answer.kind in (REFUSED, FAILED):
            return AnsweredUpdate(
                found=True,
                kind=answer.kind,
                handler=answer.handler,
                code=answer.code,
                type_name=answer.type_name,
                message=answer.message,
            )
        value = answer.value
        if answer.kind == BY_ROW:
            value = json.loads(
                workflow.payload_converter()
                .to_payloads([self._row(answer)])[0]
                .data.decode("utf-8")
            )
        return AnsweredUpdate(
            found=True, kind=BY_VALUE, handler=answer.handler, value=value
        )

    @workflow.query
    def attempt_records(self) -> List[AttemptRecord]:
        """Report one record per attempt, for a harness and never for a model.

        This is the read the analysis and the run's derived file are made of. It is a Query, so
        it writes nothing and can be asked of a generation that is still serving as easily as of
        one that finished, and it reads the same state the seal made authoritative rather than a
        second copy kept beside it.

        The rows come back in the order the manifest fixed, so two generations built from one
        manifest produce two files that line up row for row.
        """
        ordered = sorted(
            self._attempts.values(),
            key=lambda attempt: (attempt.item.task_position, attempt.item.attempt_id),
        )
        return [self._record(attempt) for attempt in ordered]

    @workflow.query
    def presented_messages(self) -> List[PresentedMessage]:
        """Report every message this generation committed to deliver, in commitment order.

        The attempt records answer what an attempt was and what it scored. This answers which
        bytes were committed to get there, which is the other half of the one question a harness
        asks after an episode: are the bytes in its own transcript the bytes this generation
        stands behind. Kinds the records have no column for are here for that reason, because a
        Wait, a SealReject, an Info answer and the Done are committed the same way, and a
        reconciliation blind to them would pass a run that lost one.

        Commitment is the whole of what this answers. It is accepted before the transport has a
        result to hand anybody, so a message counted here can be one whose bytes never left: an
        acknowledgement lost on the way back to the call that asked leaves exactly that. Whether
        the bytes reached the transport, and whether a model then read them, are the harness's
        facts, and its transcript is what a claim about either is reconciled against.

        Order is the order they were committed in, so the comparison is of two sequences rather
        than of two sets: a message that turned up somewhere else in the harness's own record is
        not the message that was committed at that point.
        """
        return list(self._presented.values())

    @workflow.query
    def generation_records(self) -> GenerationRecords:
        """Report the attempts and the commitments, both read out of this one call.

        A Query is answered against the projection as it stands when the handler runs, so two
        Queries are two moments. A generation that is still serving can commit a presentation
        between them, and a caller that asked for the rows and then for the commitments gets a
        payload that one half says is owed and the other says was committed. Neither answer is
        wrong and the pair describes no generation at all.

        So a reader that wants both asks for both here. The separate queries stay, because a
        caller that wants one half is not made wrong by the other being available, and a
        generation nobody is serving cannot move between two questions anyway.
        """
        return GenerationRecords(
            attempts=self.attempt_records(),
            presentations=self.presented_messages(),
            operation_failures=list(self._operation_failures),
        )

    def _record(self, attempt: _Attempt) -> AttemptRecord:
        """Return one attempt's record, read out of the attempt and the presentations.

        Presentation is read from the cursor's own history rather than from the attempt's state,
        because the three messages belong to three different transitions and a state can only
        say where the attempt ended up. A Payload is the case that matters: an attempt is
        acknowledged whether or not anything was ever delivered against it.

        Which is why the roster's column comes with it. A payload that was never presented is
        two different rows depending on whether one was owed, and the obligation is where that
        is written: a row that promised none has no obligation to be in any state, and a row
        that promised one carries how far it got.

        The ending comes with the score for the same reason. A floored attempt scores nothing,
        which is the number an attempt the environment graded at nothing also scores, and the
        reason is the only thing that separates them.

        The policy comes with the delivery for the third time round the same argument. A payload
        that was delivered is a different fact depending on what its body was allowed to say, and
        a row from a generation that recorded no policy reads as the placeholder rather than as
        an honest receipt nobody can now check.

        And who decided comes with the policy, for the fourth. The same policy name is the right
        answer for an ordinary run the platform stamped and for an experiment cell somebody
        registered, so the name on its own cannot say whether either of the two mistakes was
        made.

        And for a fifth, what the work behind a filing failed with comes with the ending it
        produced. The ending is a class, several very different failures share one, and the class
        alone leaves a reader with a hole and nothing to look into it with. What answers it is
        what this generation watched happen: the failure was reported to this workflow and
        recorded in its history, so a reader holding the row can tell a grader that was down from
        one that refused this task without holding anything else.

        And for a sixth, the source comes with the selection made from it. A row saying which cell
        it delivered and not which source that cell came out of would be a leg with no arm: the
        commitment is what two derivations of one source share, and it is the value a reader joins
        them by. The contract is named beside it whether or not anything has been captured, so an
        attempt still waiting to seal is legible as one under a contract rather than as a row that
        predates the question, and the visible digest of what was committed is named beside that,
        which is what a harness reconciles its own transcript against.
        """
        item = attempt.item
        obligation = self._obligations.get(item.attempt_id)
        declared = self._capture_contract(item.attempt_id)
        presented = self._presented.get(item.payload_message_id)
        resolved = next(
            (
                row
                for row in self._start.dispositions
                if row.attempt_id == item.attempt_id
                and row.branch_slot == self._start.served_slot
            ),
            None,
        )
        return AttemptRecord(
            attempt_id=item.attempt_id,
            task_position=item.task_position,
            payload_position=item.payload_position,
            state=attempt.state,
            terminal_tool=attempt.terminal_tool,
            terminal_source=attempt.terminal_source,
            canonicalization_version=self._start.canonicalization_version,
            submission_digest=attempt.submission_digest,
            score=attempt.score,
            decode_state=attempt.decode_state,
            seal_ordinal=attempt.seal_ordinal,
            final_failure=attempt.final_failure,
            deadline_expired=attempt.deadline_expired,
            task_message_id=item.task_message_id,
            task_delivered=item.task_message_id in self._presented,
            ack_message_id=item.ack_message_id,
            ack_delivered=item.ack_message_id in self._presented,
            payload_message_id=item.payload_message_id,
            payload_delivered=item.payload_message_id in self._presented,
            creates_payload_obligation=self._assignments[
                item.attempt_id
            ].creates_payload_obligation,
            payload_state=None if obligation is None else obligation.state,
            payload_policy=(
                None
                if obligation is None
                else policy_name_of(None if resolved is None else resolved.policy_digest)
            ),
            payload_disposition=None if resolved is None else describe_disposition(resolved),
            profile=self._start.profile,
            payload_resolution_source=(
                None if resolved is None else resolved.resolution_source
            ),
            failure_activity=attempt.failure_activity,
            failure_activity_id=attempt.failure_activity_id,
            failure_kind=attempt.failure_kind,
            failure_message=attempt.failure_message,
            failure_retry_state=attempt.failure_retry_state,
            receipt_contract_id=(
                attempt.receipt_contract_id
                or (None if declared is None else declared.contract_id)
            ),
            source_provenance=_provenance(attempt),
            payload_visible_sha256=(
                None if presented is None else presented.visible_bytes_sha256
            ),
        )

    # Pull.

    async def _pull(self, request: PullRequest, writer: Writer) -> OfferedMessage:
        """Answer one pull, reading the store only where this execution owes a read.

        A retry of a request this generation already answered is pure: the bytes were reserved
        for it and neither a moved cursor nor a lost object changes what it is owed. So is a
        refusal, which is what the exact-outcome journal replays. The one call that reads
        anything is the first offer in this execution of a payload obligation whose attempt
        carries a source, and it reads only what that obligation is about to deliver.
        """
        identity = pull_request_identity(request)
        bound = self._pull_requests.get(request.request_id)
        if bound is not None:
            if bound.identity != identity:
                raise StreamProtocolError("request_conflict")
            if bound.message.message_id in self._presented:
                raise StreamProtocolError("already_presented")
            return bound.message
        if request.last_presented_cursor != self._cursor:
            raise StreamProtocolError("invalid_cursor")
        if self._pending is not None:
            raise StreamProtocolError("outstanding_response")
        return await self._select(request.request_id, identity, writer)

    async def _select(self, request_id: str, identity: str, writer: Writer) -> OfferedMessage:
        choice = self._first_eligible()
        if choice is not None:
            kind, attempt_id = choice
            if kind == PAYLOAD:
                await self._require_evidence(attempt_id, writer, request_id, identity)
                offered = self._offer_payload(attempt_id, request_id, identity)
                # What recovers a refused delivery is this: the same pull, compared by its whole
                # canonical identity, passing the check it was refused on and being answered.
                # Putting the bytes back appends nothing, and neither does a claim over them.
                self._note_recovery(identity, CONTINUED_FIRST_DELIVERY, writer.ownership_epoch)
                return offered
            return self._offer_task(attempt_id, request_id, identity)
        if self._done_eligible():
            done = Done(message_id=self._start.done_message_id)
            return self._offer(done, "pull", request_id, identity, None)
        self._wait_count += 1
        wait = Wait(
            message_id=self._mint_message_id(),
            retry_after_ms=self._start.wait_retry_after_ms,
        )
        return self._offer(wait, "pull", request_id, identity, None, self._wait_reason())

    def _first_eligible(self) -> Optional[Tuple[str, str]]:
        """The message the declared order puts first, as its kind and the attempt it belongs to.

        Both kinds are ranked by the same declared key, so a payload outranking a task is the
        plan's priority rather than this method's opinion, and two messages that became eligible
        at the same moment are separated by the plan's tie key rather than by arrival.
        """
        ranked: List[Tuple[Tuple[int, int, str], str, str]] = [
            (
                order_key(self._release, PAYLOAD, self._assignments[obligation.item.attempt_id]),
                PAYLOAD,
                obligation.item.attempt_id,
            )
            for obligation in self._eligible_obligations()
        ]
        ranked += [
            (order_key(self._release, TASK, row), TASK, row.attempt_id)
            for row in self._eligible_tasks()
        ]
        if not ranked:
            return None
        _, kind, attempt_id = min(ranked)
        return kind, attempt_id

    def _eligible_obligations(self) -> List[_Obligation]:
        """Every obligation the plan has released and no pull has taken yet."""
        return [o for o in self._obligations.values() if o.state == ELIGIBLE]

    def _eligible_tasks(self) -> List[Assignment]:
        """Every task the schedule would let this pull reserve, when capacity allows one.

        Capacity is checked here rather than in the schedule because it is a property of the
        generation and not of the plan: the same plan under a wider capacity releases the same
        tasks and more of them can be in flight.
        """
        if self._capacity_in_use() >= self._start.capacity:
            return []
        return eligible_tasks(self._release, list(self._assignments.values()), self._view())

    def _view(self) -> ScheduleView:
        """The facts the schedule reads, taken from the generation's own state."""
        return ScheduleView(
            offered_attempts=frozenset(
                key for key, value in self._attempts.items() if value.state != PLANNED
            ),
            sealed_attempts=frozenset(
                key
                for key, value in self._attempts.items()
                if value.state in (SEALED, ACK_PRESENTED)
            ),
            presented_payload_positions=frozenset(
                o.item.payload_position
                for o in self._obligations.values()
                if o.state == PRESENTED
            ),
        )

    def _offer_payload(self, attempt_id: str, request_id: str, identity: str) -> OfferedMessage:
        obligation = self._obligations[attempt_id]
        candidate = obligation.candidate
        assert candidate is not None
        obligation.state = OFFERED
        result = Payload(
            message_id=obligation.item.payload_message_id,
            attempt_id=attempt_id,
            body=candidate.body,
        )
        return self._offer(result, "pull", request_id, identity, attempt_id)

    def _offer_task(self, attempt_id: str, request_id: str, identity: str) -> OfferedMessage:
        item = self._items[attempt_id]
        self._attempts[attempt_id].state = TASK_OFFERED
        result = Task(
            message_id=item.task_message_id,
            attempt_id=attempt_id,
            body=item.body,
            budget=self._start.budget,
        )
        return self._offer(result, "pull", request_id, identity, attempt_id)

    def _done_eligible(self) -> bool:
        """Done is monotonic and late: a live attempt or an unfulfilled payload keeps it away.

        The two tuples it reads cover every attempt and obligation state that exists here, so a
        state a later element adds has to be classified rather than silently counted as done. A
        caller reaches this only with nothing outstanding, which is the other condition.
        """
        if not self._queue_closed:
            return False
        if any(attempt.state in LIVE_ATTEMPT for attempt in self._attempts.values()):
            return False
        return not any(o.state in UNFULFILLED_OBLIGATION for o in self._obligations.values())

    def _wait_reason(self) -> str:
        """The hidden reason for a Wait. It is recorded here, and it is never on the wire."""
        if self._capacity_in_use() >= self._start.capacity:
            return "capacity"
        if any(attempt.state == PLANNED for attempt in self._attempts.values()):
            return "gate"
        if not self._queue_closed:
            return "queue_open"
        return "obligation_pending"

    def _materialized(self) -> int:
        """How many obligations have had their candidate built.

        The fact is read off the obligation rather than off the bytes. A candidate carries the
        renderer, the match group, the hashes and the byte count a family gate compares, and
        while an offer could still carry it the obligation holds it; once one has been presented
        or has ended without being rendered the bytes are of no further use and a generation
        that crossed a boundary will not be holding them. What has to survive either way is that
        it was built, because the count is inside the projection every presentation attests to.
        """
        return sum(1 for o in self._obligations.values() if o.materialized)

    def _ownership_claims(self) -> int:
        """How many owners this generation has had, over every execution it has run in."""
        return self._carried_ownership_claims + len(self._ownership)

    def _offer_count(self) -> int:
        """How many reservations this generation has made, over every execution."""
        return self._carried_offers + len(self._offers)

    def _eligibility_count(self) -> int:
        """How many obligations this generation's plan has released, over every execution."""
        return self._carried_eligibilities + len(self._eligibilities)

    def _handed_out(self) -> Set[str]:
        """Every attempt whose task this generation reserved, over every execution.

        An ending overwrites the state an attempt was in and leaves the offer that reserved its
        task exactly where it was, which is why this is read from the offers. An execution that
        continued another holds its predecessor's as a set of identifiers rather than as rows,
        because a count of who was handed out is the whole of what this is read for.
        """
        return self._carried_handed_out | {
            offer.attempt_id
            for offer in self._offers
            if offer.kind == "task" and offer.attempt_id is not None
        }

    def _wait_reason_counts(self) -> Dict[str, int]:
        """How many Waits each hidden reason accounts for, for the harness alone."""
        counts: Dict[str, int] = dict(self._carried_wait_reasons)
        for offer in self._offers:
            if offer.wait_reason is not None:
                counts[offer.wait_reason] = counts.get(offer.wait_reason, 0) + 1
        return counts

    def _capacity_in_use(self) -> int:
        occupied = (TASK_OFFERED, ACTIVE, SEALING)
        return sum(1 for attempt in self._attempts.values() if attempt.state in occupied)

    # Info.

    def _info(self, request: InfoRequest) -> OfferedMessage:
        """Reserve one answer for one info request, on the terms a pull is reserved under.

        A generation that did not declare the tool has nothing to answer with and refuses the
        call as the malformed thing it is. The refusal is here and not only where the tool is
        advertised, because this Update is reachable by anything holding the generation's writer,
        and what a generation serves is the generation's own fact rather than whichever
        transport happens to be in front of it.

        Everything after that is the pull's rule, and for the pull's reasons: a retry of the same
        request is answered from what that request reserved, the same ID carrying anything else
        is a conflict, a request made from a stale cursor is refused, and a second request while
        one answer is outstanding does not overtake it.

        A prepared seal is the one thing owed that reserves no result, and it is refused here
        along with the ones that do. A filing accepted by an owner that was then replaced leaves
        its attempt sealing, and the only thing that can still finish it is that exact request,
        which names the cursor it was made at. An answer minted here would be presented, the
        cursor would move, and that request would be refused for a stale cursor before it reached
        the filing it prepared, while a filing rebuilt at the new cursor is a different identity
        and therefore a second filing for an attempt that already has one. Nothing could then
        seal the attempt, finalize it, or let the generation reach Done. The question is worth
        less than the attempt, so it waits.
        """
        if not self._start.info:
            raise StreamProtocolError("invalid_message")
        identity = info_request_identity(request)
        bound = self._info_requests.get(request.request_id)
        if bound is not None:
            if bound.identity != identity:
                raise StreamProtocolError("request_conflict")
            if bound.message.message_id in self._presented:
                raise StreamProtocolError("already_presented")
            return bound.message
        if request.last_presented_cursor != self._cursor:
            raise StreamProtocolError("invalid_cursor")
        if self._pending is not None:
            raise StreamProtocolError("outstanding_response")
        if any(attempt.state == SEALING for attempt in self._attempts.values()):
            raise StreamProtocolError("outstanding_response")
        remaining, consumed, in_flight = self._queue_counts()
        answer = Info(
            message_id=self._mint_message_id(),
            remaining=remaining,
            consumed=consumed,
            in_flight=in_flight,
        )
        return self._offer(answer, "info", request.request_id, identity, None)

    def _queue_counts(self) -> Tuple[int, int, int]:
        """How much of the queue there is: remaining, consumed, and in flight.

        The three are defined here and nowhere else. Every other statement of them, in this
        module and outside it, says what this says.

        ``remaining`` is the attempts still to be handed out, which is PLANNED and only PLANNED:
        not handed out yet, and still able to be. An attempt a gate is holding back is planned,
        so it counts as still to come rather than as withheld.

        ``consumed`` is the attempts handed out so far, whatever became of them: TASK_OFFERED,
        ACTIVE, SEALING, SEALED, ACK_PRESENTED, and a FINAL_FAILED attempt that was offered
        before it ended. Handing out is the offer, so an attempt whose task was reserved counts
        from that moment even if the presentation of it never committed.

        ``in_flight`` is the ones among those that have not ended, which is TASK_OFFERED, ACTIVE
        and SEALING: exactly the attempts occupying capacity. Sealing releases capacity, so a
        sealed attempt whose acknowledgement is still to be presented is consumed and not in
        flight.

        So ``in_flight`` is inside ``consumed`` rather than beside it, and the three are not a
        partition of anything. What they leave out is the attempts this generation ended before
        ever handing them out: they were never handed out, so they are not consumed, and they can
        never be handed out now, so they are not remaining. Two mechanisms take an attempt there,
        and both write an ending over a PLANNED attempt: a controller finalizing one, and the
        cascade that floors every one whose gate can no longer open. Either can reach any number
        of attempts. A deadline and a spent step budget reach none of them, because each ends an
        attempt that was handed over already and leaves it consumed, although either can be the
        ending the cascade then follows from.

        The arithmetic, with the roster ``N`` and the never-handed-out endings ``U``, is
        ``remaining + consumed + U == N`` and ``in_flight <= consumed``. So ``remaining`` plus
        ``consumed`` is ``N`` wherever ``U`` is zero, which is every generation that has not
        ended an attempt before handing it out, and is short of ``N`` by ``U`` everywhere else.
        The three together need not sum to ``N``: they sum to ``N - U + in_flight``, which can
        fall below the roster, meet it, or pass it, so the sum of the three is not in general the
        roster.

        None of the three is a Wait's reason, and no record carries one. That is a fact about the
        fields rather than a guarantee about what can be worked out from them, and the difference
        matters here: a generation publishes its capacity in the control tool's own description,
        so an agent holding ``in_flight`` equal to that capacity has been told what a Wait it then
        receives is for, and one holding ``in_flight`` below it with ``remaining`` above zero has
        been told that something is gating the queue. A generation that declares this tool has
        decided to publish counts that carry that much.

        Whether an attempt was handed out is read from the offers this generation made, because
        that is where the fact is: an ending overwrites the state an attempt was in and leaves
        the offer that reserved its task exactly where it was.
        """
        remaining = sum(1 for attempt in self._attempts.values() if attempt.state == PLANNED)
        return remaining, len(self._handed_out()), self._capacity_in_use()

    # Seal.

    async def _seal(self, request: SealRequest, writer: Writer) -> OfferedMessage:
        metadata = request.metadata
        if request.terminal_source not in TERMINAL_SOURCES:
            raise StreamProtocolError("invalid_message")
        try:
            identity = terminal_request_identity(
                metadata,
                request.public_tool_name,
                request.native_terminal_name,
                request.native_arguments,
                request.terminal_source,
            )
        except WireFormatError as error:
            # Arguments with no canonical encoding have no identity either, so there is
            # nothing to deduplicate a retry against and nothing to seal.
            raise StreamProtocolError("invalid_message") from error
        bound = self._terminal_requests.get(metadata.request_id)
        if bound is not None:
            if bound.identity != identity:
                raise StreamProtocolError("request_conflict")
            if bound.message.message_id in self._presented:
                raise StreamProtocolError("already_presented")
            return bound.message
        if metadata.last_presented_cursor != self._cursor:
            raise StreamProtocolError("invalid_cursor")
        if self._pending is not None:
            raise StreamProtocolError("outstanding_response")
        attempt = self._attempts.get(metadata.attempt_id)
        if attempt is None:
            raise StreamProtocolError("invalid_attempt")
        if attempt.state == SEALING:
            # A prepared seal whose owner was fenced before it committed. The exact terminal
            # request continues it, keyed by the same seal ID, rather than starting a second
            # one: the submission was fixed by value when it was prepared, and this path grades
            # what was filed then. Any other filing for the attempt is a conflict.
            if (
                attempt.terminal_request_id == metadata.request_id
                and attempt.terminal_identity == identity
            ):
                return await self._seal_accepted(request, attempt, identity, writer)
            raise StreamProtocolError("conflicting_seal")
        if attempt.state in (SEALED, ACK_PRESENTED, FINAL_FAILED):
            # A finalized attempt refuses a filing for the same reason a sealed one does: it has
            # an outcome already, and this filing would be a second one. The refusal is the same
            # bytes however many times the request is retried, and it moves nothing.
            raise StreamProtocolError("conflicting_seal")
        if attempt.state != ACTIVE:
            raise StreamProtocolError("invalid_attempt")
        tool = self._start.terminal_tool
        if (
            request.public_tool_name != tool.public_tool_name
            or request.native_terminal_name != tool.native_terminal_name
        ):
            raise StreamProtocolError("invalid_message")
        complaint = _argument_complaint(tool.argument_names, request.native_arguments)
        if complaint is not None:
            reject = SealReject(
                message_id=self._mint_message_id(),
                attempt_id=attempt.item.attempt_id,
                body=complaint,
            )
            return self._offer(
                reject, "terminal", metadata.request_id, identity, attempt.item.attempt_id
            )
        return await self._seal_accepted(request, attempt, identity, writer)

    async def _seal_accepted(
        self, request: SealRequest, attempt: _Attempt, identity: str, writer: Writer
    ) -> OfferedMessage:
        """Run the seal's batch, and end the attempt if that batch cannot go on.

        An accepted terminal makes the attempt the seal's, and the exact filing sent again is
        what continues one that was interrupted. That reading holds while the seal can still
        make progress. It stops holding in two ways. The work behind the filing can fail for
        good: an Activity the retry policy has given up on, or one that declared itself
        non-retryable, is not a step this filing can take again. Or the work can succeed and
        hand back a result the seal cannot vouch for, which the exact filing sent again would
        ask for and be given a second time. Either way an attempt left prepared under it would
        stay live for ever with no acknowledgement, no outcome, and no way to Done.

        So a batch that cannot go on is an ending. It commits the final failure and nothing
        else: no seal, no score, no candidate, no acknowledgement, and the caller is told what
        failed rather than being answered. The two ways in are written apart, because a step
        that never happened and a step whose answer was unusable are different things to read
        afterwards. A writer that a resume fenced meanwhile commits nothing at all, and the
        attempt stays prepared for the owner that replaced it.

        A candidate built under a policy other than the one this obligation was resolved to is
        the second of those two, with a reason of its own. It is the same ending: the attempt is
        finalized, the capacity comes back, nothing is acknowledged, and the generation goes on
        serving. What the separate reason buys is a reader who can tell a Worker running the
        wrong renderer from a result that was merely malformed.

        Both endings are explained on the attempt before it is ended. A filing runs the seal and
        the grade, and the renderer as well when the attempt carries a payload obligation, and any
        of them can be the one that failed, so the ending on its own leaves a reader with a hole
        and no way to say which step made it. The other ending has no failed step to name, and
        what it needs written down instead is the check that refused the answer. Either way what
        is written is read out of what the history recorded and nothing else, so it costs no side
        effect and a replay produces the same words.
        """
        # Where the batch got to, for the row a failure leaves. The reference this seal selected
        # exists only while the batch is running, and the recorder is out here, so what the
        # resolver could not produce is written down as the batch reaches it rather than being
        # recovered afterwards from state a failed seal never committed.
        resolving: List[str] = []
        try:
            return await self._seal_batch(request, attempt, identity, writer, resolving)
        except StreamProtocolError:
            raise
        except ActivityError as error:
            self._require_writer(writer)
            _note_failure(attempt, error)
            self._note_capture_failure(
                attempt, identity, request.metadata.request_id, writer, resolving
            )
            self._finalize(attempt, SEAL_FAILED)
            raise
        except _UnusableResult as unusable:
            self._require_writer(writer)
            _note_unusable(attempt, unusable)
            self._finalize(attempt, unusable.final_failure)
            raise

    async def _seal_batch(
        self,
        request: SealRequest,
        attempt: _Attempt,
        identity: str,
        writer: Writer,
        resolving: List[str],
    ) -> OfferedMessage:
        metadata = request.metadata
        attempt.state = SEALING
        # The deadline is disarmed as the terminal is accepted, not when the seal commits. What
        # the deadline is for is an attempt nobody is finishing, and this one is being finished.
        attempt.deadline_at = None
        attempt.terminal_request_id = metadata.request_id
        attempt.terminal_identity = identity
        # Which tool ended it. One generation declares one terminal tool and a filing that named
        # another was refused above, so this is kept as the fact rather than derived later from a
        # configuration that a generation with two terminals would no longer answer for.
        attempt.terminal_tool = request.native_terminal_name
        # And who filed it. The seal is the same transaction either way, so this is written here
        # with the rest of what the filing fixed rather than being derived afterwards from a
        # budget the stream counted and an ending nothing recorded.
        attempt.terminal_source = request.terminal_source
        attempt.seal_id = hidden_seal_id(
            self._start.hidden_execution_id,
            self._start.execution_ordinal,
            attempt.item.attempt_id,
        )
        # Which shape this attempt's own bodies are admitted under, where its row names one. It
        # is read before the seal because the capture is bound to the contract at its first
        # write: an environment given none publishes no source at all, and one given the wrong
        # one would commit a capture nothing could later republish.
        contract = self._capture_contract(attempt.item.attempt_id)
        sealed = await workflow.execute_activity(
            seal_attempt_activity,
            SealAttemptInput(
                attempt_id=attempt.item.attempt_id,
                seal_id=attempt.seal_id,
                native_terminal_name=request.native_terminal_name,
                canonicalization_version=self._start.canonicalization_version,
                native_arguments=request.native_arguments,
                blob_root=self._start.blob_root,
                execution_ordinal=self._start.execution_ordinal,
                receipt_contract=contract,
            ),
            start_to_close_timeout=_TERMINAL_ACTIVITY_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
            activity_id=self._next_activity_id(),
        )
        if sealed.attempt_id != attempt.item.attempt_id or sealed.seal_id != attempt.seal_id:
            raise _unusable("the sealed submission is not the one this seal asked for")
        # And it was captured under the version this generation declared. That version is what
        # the acknowledgement names and what a reader takes the submission's capture rule from,
        # so a result answering with another one would put a rule in the record that the bytes
        # were not written under. The environment is asked under the declared version and every
        # port refuses any other, so this is the check on the answer rather than on the
        # question, and it is here because a submission whose capture rule this generation
        # cannot state is not one to grade or to acknowledge.
        #
        # It is behind a marker because it changes what a seal does with a result, and a history
        # recorded before it holds seals no build compared. A build from then could answer under
        # another version and be believed, and what it recorded next was a grade and an
        # acknowledgement; replaying that history against an unguarded check would end the
        # attempt where the history says the grade was scheduled, and the generation would fail
        # to replay rather than reconstruct what it served. The marker is read only where the
        # two versions differ, so a seal that agrees records nothing and every ordinary
        # generation keeps the history it had.
        if sealed.canonicalization_version != self._start.canonicalization_version and (
            workflow.patched("seal-answers-under-the-declared-version")
        ):
            raise _unusable(
                f"the submission was captured as {sealed.canonicalization_version!r} and this "
                f"generation was started as {self._start.canonicalization_version!r}"
            )
        # The prepared seal, recorded before the grader runs. It fixes the submission and its
        # digest by value, so a resumed seal grades what was filed and not what is there now.
        attempt.canonical_submission_text = sealed.canonical_submission_text
        attempt.environment_recovery_token = sealed.environment_recovery_token
        attempt.submission_digest = submission_digest(
            attempt.item.attempt_id,
            request.native_terminal_name,
            sealed.canonical_submission_text.encode("utf-8"),
        )
        attempt.finalizer_key = sealed.seal_id
        # The source, checked against what this generation is rather than against itself. The
        # descriptor carries the ordinal and not the hidden execution id, so the seal id it names
        # is recomputed here from this start's own identity: a descriptor cannot hand a
        # generation the value it is supposed to be checked against.
        source = self._verified_source(attempt, sealed, contract)
        # The result is read into a shape this generation decided on before anything is compared
        # against it. Nothing about the wire shape is a promise: every field arrives as whatever
        # was encoded, so a result that is not a grade is refused here, at the first line that
        # touches it, rather than failing an activation nothing can record.
        graded = _graded(
            await workflow.execute_activity(
                grade_attempt_activity,
                GradeAttemptInput(
                    attempt_id=attempt.item.attempt_id,
                    seal_id=sealed.seal_id,
                    submission_digest=attempt.submission_digest,
                    canonical_submission_text=sealed.canonical_submission_text,
                    environment_recovery_token=sealed.environment_recovery_token,
                    blob_root=self._start.blob_root,
                ),
                start_to_close_timeout=_TERMINAL_ACTIVITY_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                activity_id=self._next_activity_id(),
            )
        )
        # Each result is checked where it arrives, before the next Activity is asked for. The
        # seal ID and the digest are what make a result this filing's: the public attempt ID
        # and the payload position survive a fork, so a result that echoes only those says
        # which obligation asked and not which filing was answered.
        if graded.attempt_id != attempt.item.attempt_id or graded.seal_id != attempt.seal_id:
            raise _unusable("the score is not this seal's")
        # And it is the grader this generation was built over, whatever the agent is going to be
        # told. The score is committed either way: a concealed arm's number is the outcome that
        # arm reports and a position that owes no payload still records one, so a stand-in or
        # another implementation substituted behind either of those would put the wrong number
        # in the record with nothing about it visible in a body. The identity is therefore a
        # property of the seal rather than of the exposure.
        declared = self._start.grade or KERNEL_STAND_IN_GRADE
        returned = graded.grade or KERNEL_STAND_IN_GRADE
        if returned != declared:
            raise _unusable(
                f"this generation is graded by {declared.grader_id}/{declared.grader_version} "
                f"and the score that came back is {returned.grader_id}/"
                f"{returned.grader_version}'s"
            )
        # And the numbers are what the environment said its numbers are, before any of them is
        # committed or printed. A result is a decoded value rather than the object a grader
        # built: a field declared as a number comes back as one whatever was put in it, so the
        # projection is applied to what arrived rather than to what was meant to be sent. The
        # domains it is applied against are the ones this generation declared, which the check
        # above has just made the same ones the result carries.
        try:
            check_grade_result(
                score=graded.score,
                components=graded.components,
                grade=declared,
            )
        except PolicyViolation as violation:
            raise _unusable(str(violation)) from violation
        # An attempt with no obligation has nothing to build, so nothing is built: neither the
        # generation under Never nor the one position a roster gave no payload asks a renderer
        # for a candidate. An absent outbox row is not an outbox row nobody reads.
        candidate: Optional[PayloadCandidate] = None
        selected: Optional[SelectedSourceReference] = None
        disposition: Optional[PayloadDisposition] = None
        if attempt.item.attempt_id in self._obligations:
            disposition = self._served.get(attempt.item.attempt_id)
            policy = _policy_of(disposition)
            published = self._published(attempt, graded, policy, declared)
            # Which committed cell this obligation is served, where its policy delivers one. The
            # selection is the controller's and it is one reference: the store verifies a digest
            # without knowing which cell it names, so a resolver handed the map could return the
            # cell nobody selected and satisfy every hash and size check on the way back.
            selected = self._selected(source, disposition, policy)
            if selected is not None:
                # The one object the resolver is about to be asked for, written down before it is
                # asked. A payload Activity that cannot produce it fails out here, and the row
                # that records the failure has to name the reference the operation could not
                # produce rather than an empty list.
                resolving.append(selected.body_sha256)
            bundle = await workflow.execute_activity(
                generate_payload_bundle_activity,
                GeneratePayloadBundleInput(
                    attempt_id=attempt.item.attempt_id,
                    payload_position=attempt.item.payload_position,
                    payload_message_id=attempt.item.payload_message_id,
                    submission_digest=attempt.submission_digest,
                    canonical_submission_text=(
                        "" if selected is not None else sealed.canonical_submission_text
                    ),
                    policy_digest="" if disposition is None else (disposition.policy_digest or ""),
                    cell="" if disposition is None else (disposition.cell or ""),
                    public_grade=published,
                    selected=selected,
                ),
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                activity_id=self._next_activity_id(),
            )
            if len(bundle.candidates) != 1:
                raise _UnusableResult(
                    "the kernel payload family has exactly one candidate",
                    type="IncompleteCandidateBundle",
                    non_retryable=True,
                )
            if (
                bundle.attempt_id != attempt.item.attempt_id
                or bundle.payload_position != attempt.item.payload_position
                or bundle.submission_digest != attempt.submission_digest
            ):
                raise _unusable("the candidate bundle is not the one this obligation asked for")
            candidate = _read_candidate(bundle.candidates[0])
            _check_candidate(attempt.item, candidate)
            _check_echo(
                candidate,
                disposition,
                policy,
                attempt.item,
                attempt.submission_digest,
                published,
                selected=selected,
                source=source,
            )
            _check_family(
                candidate,
                None
                if disposition is None or not disposition.family_id
                else self._families.get(disposition.family_id),
            )
        # Everything above was awaited, so the owner is checked again before any of it is made
        # authoritative. A seal that was in flight when a resume fenced its writer commits
        # nothing: the attempt stays prepared, and the new owner's exact retry continues it.
        self._require_writer(writer)
        # One transition, no await inside it: the score, the bundle, the released capacity,
        # the obligation, and the acknowledgement all become authoritative together or not
        # at all. The bytes go out after this, which is what puts the seal before the Ack.
        attempt.score = float(graded.score)
        attempt.decode_state = graded.decode_state
        # The score and what it was taken from become authoritative together. A committed score
        # whose evidence the generation kept no name for is a headline with nothing under it, so
        # the reference is kept on the attempt and counted among the objects this history cites:
        # a claim reads the store for all of them again before it may continue the generation.
        attempt.graded_evidence = graded.evidence_sha256
        if graded.evidence_sha256 not in self._committed_blobs:
            self._committed_blobs.append(graded.evidence_sha256)
        self._commit_source(attempt, source, selected, disposition)
        self._seal_ordinal += 1
        attempt.seal_ordinal = self._seal_ordinal
        attempt.state = SEALED
        if candidate is not None:
            self._materialize(attempt.item, candidate)
        ack = SealAck(
            message_id=attempt.item.ack_message_id,
            attempt_id=attempt.item.attempt_id,
            submission_digest=attempt.submission_digest,
            canonicalization_version=self._start.canonicalization_version,
        )
        return self._offer(
            ack, "terminal", metadata.request_id, identity, attempt.item.attempt_id
        )

    def _commit_source(
        self,
        attempt: _Attempt,
        source: Optional[SourceArtifactManifest],
        selected: Optional[SelectedSourceReference],
        disposition: Optional[PayloadDisposition],
    ) -> None:
        """Write the committed source and the selection onto the attempt, inside the transition.

        The source outlives the candidate, so it is the attempt's state and not the obligation's.
        The seal builds one candidate and the position may carry none at all, and a generation
        that kept only the candidate would have a run whose record could say what one leg was told
        and could not say what it was a leg of. So the descriptor, its commitment, the identity it
        was sealed under and the selection made from it all become authoritative here, with the
        score and the acknowledgement.

        Two objects join the inventory a later claim reads the store for: the descriptor's own
        canonical bytes and the canonical submission it names. The cells are not among them,
        deliberately. Retention installs all three and keeps them; requiring them at every claim
        would let one attempt's unselected body block another attempt's delivery, and the body a
        delivery depends on is checked by the delivery that depends on it.

        The obligation joins this execution's verified set here rather than at the offer. What
        publication just did was install those objects and read every one of them back, in this
        execution, which is exactly what the barrier asks of a continued one; the first offer
        after a boundary finds an empty set and asks again.
        """
        if source is None:
            return
        attempt.source_artifact = source
        attempt.source_commitment = source_commitment(source)
        attempt.source_origin = SourceOriginContext(
            hidden_execution_id=self._start.hidden_execution_id,
            execution_ordinal=self._start.execution_ordinal,
        )
        attempt.receipt_contract_id = source.receipt_contract.contract_id
        for reference in (attempt.source_commitment, source.canonical_submission.sha256):
            if reference not in self._committed_blobs:
                self._committed_blobs.append(reference)
        if selected is None or disposition is None:
            return
        attempt.selected_cell = selected.cell
        attempt.selected_body_reference = selected.body_sha256
        attempt.selected_policy_digest = disposition.policy_digest
        self._verified_deliveries.add(attempt.item.attempt_id)

    def _capture_contract(self, attempt_id: str) -> Optional[ReceiptContract]:
        """Return the contract this attempt's capture is validated against, or nothing.

        A receipt-producing attempt names its contract in the same column a row names its
        matched family in, and a withholding names one too: capture is not conditional on
        exposure, so an attempt that captures and delivers nothing still has to say which shape
        its source was published under. A generation that declares no contract asks for no
        source, and its seal is the seal it always was.
        """
        row = self._resolved.get(attempt_id)
        if row is None or not row.family_id:
            return None
        return self._contracts.get(row.family_id)

    def _note_capture_failure(
        self,
        attempt: _Attempt,
        identity: str,
        request_id: str,
        writer: Writer,
        resolving: List[str],
    ) -> None:
        """Record a capture that could not produce its evidence, where that is what failed.

        Only an attempt whose row names a contract has a receipt to fail at, so nothing is
        synthesized for a legacy one, and only a failure that says something about the evidence
        writes a row: a grader that was down and a world that timed out are the seal's own ending
        and are described by the fields that ending already carries.

        ``resolving`` is the reference the batch had asked its resolver for, where it got that
        far. A row for the resolver's own phase names the object the operation could not produce,
        which is the whole of what a controller has to go and repair; the publication phase names
        none, because a seal that failed before it returned anything published no reference for
        this generation to have lost. Neither reaches an agent: the references are controller
        side, in this generation's own record.

        The outcome is the ending rather than a refusal, because that is what it is. This runs
        with the finalization that follows it: the attempt is over, there is no acknowledgement,
        no delivery and no filing that could be sent again, so nothing will ever recover it.
        """
        if self._capture_contract(attempt.item.attempt_id) is None:
            return
        reason = _EVIDENCE_REASONS.get(attempt.failure_kind or "")
        if reason is None:
            return
        resolver = attempt.failure_activity == GENERATE_PAYLOAD_BUNDLE
        self._note_operation(
            operation=identity,
            phase=PAYLOAD_OFFER if resolver else SOURCE_PUBLICATION,
            reason=reason,
            outcome=UNRECOVERABLE_OPERATION,
            refused_epoch=writer.ownership_epoch,
            attempt_id=attempt.item.attempt_id,
            payload_position=attempt.item.payload_position if resolver else None,
            references=list(resolving) if resolver else None,
            request_id=request_id,
        )

    def _verified_source(
        self,
        attempt: _Attempt,
        sealed: Any,
        contract: Optional[ReceiptContract],
    ) -> Optional[SourceArtifactManifest]:
        """Hold a returned source to this generation, or refuse it, before anything commits.

        The bindings are checked against values this generation already holds, and never against
        the descriptor's own. The attempt is the row's, the seal id is the one recomputed from
        this start's hidden execution id and the ordinal it asked under, the canonical submission
        reference is the text that came back beside it, the kernel digest is the one this
        workflow computed itself, the grade identity is the generation's, the contract is the one
        this generation declared, and the bundle digest is the bank source those contracts were
        admitted over. The commitment is recomputed from the descriptor's own canonical bytes,
        so a result naming one thing and committing to another is a refusal rather than a record.

        A generation that asked for no source and was handed one is refused too. A descriptor is
        provenance and a build that produced one unasked is a build this generation did not
        compose, so the answer is to end the attempt rather than to keep evidence nothing
        declared a contract for.

        The descriptor arrives as whatever was encoded and is read into a typed value here, at
        this boundary and nowhere earlier. A mapping carrying a name this build does not declare,
        or missing one it does, is refused with a reason recorded against the attempt: a field
        with a type on the result would have made that same shape a decoding failure instead,
        raised while the generation was being handed the result and before any code of its own
        ran, which the retries would reproduce for ever and the record would not explain.

        Taking the descriptor's canonical bytes is inside that same guard, for that same reason.
        The reading and the canonicalization are one boundary: a value the encoder cannot write
        would otherwise read back cleanly and raise where the commitment is computed, which is
        outside every recorded refusal there is, so the attempt would hang on an activation that
        fails for ever instead of ending with the reason this method exists to record.
        """
        returned = sealed.source_artifact
        if contract is None:
            if returned is not None or sealed.source_commitment:
                raise _unusable(
                    "this attempt declared no receipt contract and the seal returned a source"
                )
            return None
        if returned is None:
            raise _unusable(
                f"this attempt is captured under the contract {contract.contract_id} and the "
                "seal returned no source at all"
            )
        try:
            manifest = read_source_artifact(returned)
            committed = source_commitment(manifest)
        except WireFormatError as error:
            raise _unusable(str(error)) from error
        if sealed.source_commitment != committed:
            raise _unusable(
                "the seal committed to a source other than the one whose bytes it returned"
            )
        raw = sealed.canonical_submission_text.encode("utf-8")
        checks = (
            (manifest.source_attempt_id, attempt.item.attempt_id, "attempt"),
            (manifest.source_seal_id, attempt.seal_id, "seal"),
            (manifest.execution_ordinal, self._start.execution_ordinal, "execution"),
            (manifest.canonical_submission.sha256, sha256(raw).hexdigest(), "submission"),
            (manifest.canonical_submission.size, len(raw), "submission size"),
            (manifest.kernel_submission_digest, attempt.submission_digest, "filing digest"),
            (
                manifest.canonicalization_version,
                self._start.canonicalization_version,
                "canonicalization version",
            ),
            (manifest.bundle_digest, self._start.receipt_source, "bank source"),
            (manifest.receipt_contract, contract, "contract"),
            (manifest.grade_identity, self._start.grade or KERNEL_STAND_IN_GRADE, "grader"),
        )
        for returned, held, what in checks:
            if returned != held:
                raise _unusable(
                    f"the source this seal published names a {what} this generation is not: "
                    f"{returned!r} against {held!r}"
                )
        return manifest

    def _selected(
        self,
        source: Optional[SourceArtifactManifest],
        disposition: Optional[PayloadDisposition],
        policy: Optional[PayloadPolicy],
    ) -> Optional[SelectedSourceReference]:
        """Return the one committed cell this obligation delivers, where its policy delivers one.

        The disposition, the policy's declared cell, the manifest's kind and the reference all
        have to name the same cell, and they are made to here: the cell is the row's, the policy
        declares it, and the reference is the manifest's own entry for that kind rather than one
        this method chose. The oracle is never among them. It is named by the descriptor,
        retained with the rest, and is not a cell a live arm may be assigned, so a row naming it
        is refused before anything is resolved rather than served as a graded body.
        """
        if policy is None or policy.exposure != ARTIFACT or disposition is None:
            return None
        if source is None:
            raise _unusable(
                f"this obligation delivers {policy.policy_name} and its seal published no source "
                "for it to be a cell of"
            )
        cell = disposition.cell or ""
        if cell not in ELIGIBLE_CELLS or cell not in policy.cells:
            raise _unusable(
                f"an arm is served {sorted(ELIGIBLE_CELLS)} and this obligation was assigned "
                f"{cell!r}"
            )
        if self._start.blob_root is None:
            raise _unusable(
                f"this obligation delivers {policy.policy_name}, whose body is an object of this "
                "run's own store, and this generation was given no store"
            )
        contract = source.receipt_contract
        if disposition.family_id != contract.contract_id:
            raise _unusable(
                f"this obligation names the contract {disposition.family_id!r} and its source "
                f"was published under {contract.contract_id!r}"
            )
        try:
            return derived_selection(
                source=source,
                origin=SourceOriginContext(
                    hidden_execution_id=self._start.hidden_execution_id,
                    execution_ordinal=self._start.execution_ordinal,
                ),
                commitment=source_commitment(source),
                cell=cell,
                policy_digest=disposition.policy_digest or "",
                blob_root=self._start.blob_root,
            )
        except WireFormatError as error:
            raise _unusable(str(error)) from error

    def _published(
        self,
        attempt: _Attempt,
        graded: _Graded,
        policy: Optional[PayloadPolicy],
        declared: GradeIdentity,
    ) -> Optional[PublicGrade]:
        """Return what the renderer for this obligation may know of the verdict, or nothing.

        A policy that does not publish the grade is handed no grade. That is a property of the
        value rather than of the renderer's conduct: there is nothing in the request for a
        blinded body to leak, however the code on the other side is written.

        A policy that does publish it is refused a stand-in. The kernel's grade is a fact about
        the shape of a filing and says so, and a generation that told its agent that number as
        the environment's verdict would be publishing a transport fixture. That the number came
        from the grader this generation declared is settled before this, for every seal rather
        than for the ones a body publishes, so what is left here is the exposure's own half.

        What crosses is the attempt the authority assigned, the score, and the numbers the
        environment declared it publishes. A number under a name that roster does not hold does
        not cross: a name is text the grader wrote and a body prints it, so the names were fixed
        before the run and one that was not is an ending rather than a body. The evidence
        reference does not cross either: it names bytes a harness can resolve out of the run's
        own store, which is a different question from what the agent is told.
        """
        if policy is None or policy.exposure != HONEST:
            return None
        if declared.stand_in:
            raise _unusable(
                "this obligation publishes the environment's grade and the grade that came back "
                "is the kernel's stand-in"
            )
        try:
            return published_grade(
                attempt_id=attempt.item.attempt_id,
                score=graded.score,
                components=graded.components,
                grade=declared,
            )
        except PolicyViolation as violation:
            raise _unusable(str(violation)) from violation

    # Final failure.

    def _finalize_requested(self, request: FinalizeRequest) -> AttemptFinalized:
        """Check what a controller may end, then end it.

        The order of the refusals is the order of the facts. An attempt this generation never
        assigned is not an attempt. One whose terminal was accepted belongs to the seal, and
        that stays true after the seal has committed and after a finalization has already run,
        so all of those conflict rather than being told the attempt is unknown. And a result
        nobody has presented is a caller's turn, not a controller's.

        The request is bound first, for the reason a pull's is. A controller that loses the
        answer retries the request it made, and the retry has to reach the answer it already
        has: the receipt is kept against the logical request and handed back unchanged, rather
        than the retry being told the attempt it ended is now in conflict. One logical request
        is one ending, so the same ID carrying anything else is a conflict rather than a second
        one.
        """
        if request.protocol_version != PROTOCOL_VERSION:
            raise StreamProtocolError("unsupported_version")
        try:
            require_opaque_id("request_id", request.request_id)
        except WireFormatError as error:
            raise StreamProtocolError("invalid_message") from error
        identity = finalize_request_identity(request)
        bound = self._finalize_requests.get(request.request_id)
        if bound is not None:
            if bound.identity != identity:
                raise StreamProtocolError("request_conflict")
            return bound.receipt
        if request.reason not in FINAL_FAILURE_REASONS:
            raise StreamProtocolError("invalid_message")
        attempt = self._attempts.get(request.attempt_id)
        if attempt is None:
            raise StreamProtocolError("invalid_attempt")
        if attempt.terminal_request_id is not None or attempt.state in (
            SEALING,
            SEALED,
            ACK_PRESENTED,
            FINAL_FAILED,
        ):
            raise StreamProtocolError("conflicting_seal")
        if self._pending is not None:
            raise StreamProtocolError("outstanding_response")
        if attempt.state not in (PLANNED, ACTIVE):
            raise StreamProtocolError("invalid_attempt")
        receipt = self._finalize(attempt, request.reason)
        self._finalize_requests[request.request_id] = _BoundFinalization(
            identity=identity, receipt=receipt
        )
        return receipt

    def _finalize(self, attempt: _Attempt, reason: str) -> AttemptFinalized:
        """Fail one attempt finally, and with it every attempt that was waiting on it.

        One transition, no await inside it, and it mints nothing. There is no acknowledgement,
        because an acknowledgement says a submission was sealed under a digest and there is no
        submission. There is no payload either: the obligation is resolved where it stands,
        which before a seal is assigned and unbuilt, so nothing rendered is being thrown away
        and nothing unrendered is being invented.

        The ending reaches further than the attempt named in it, and it has to. A schedule may
        gate one task on another sealing or on a payload being presented, and an attempt that
        ended without a filing will never do either: the gate it holds shut can no longer open.
        Leaving those attempts planned would leave the generation waiting for a fact that
        cannot happen, with Done unreachable behind them. So they are floored here, in the same
        transition, which is also what the schedule asks for: a stop before an outcome is
        scored writes the floor over everything that outcome was going to cover, rather than
        letting one part of the scope be omitted and another wait for ever.
        """
        self._fail(attempt, reason)
        cascaded = self._fail_the_waiting()
        obligation = self._obligations.get(attempt.item.attempt_id)
        return AttemptFinalized(
            attempt_id=attempt.item.attempt_id,
            reason=reason,
            score=FLOOR,
            capacity_in_use=self._capacity_in_use(),
            obligation_state=None if obligation is None else obligation.state,
            also_finalized=cascaded,
        )

    def _fail(self, attempt: _Attempt, reason: str) -> None:
        """Write one attempt's ending: the floor, the reason, and the obligation it was owed."""
        attempt.state = FINAL_FAILED
        attempt.final_failure = reason
        attempt.score = FLOOR
        attempt.deadline_at = None
        attempt.deadline_expired = False
        obligation = self._obligations.get(attempt.item.attempt_id)
        if obligation is not None:
            obligation.state = FINAL_FAILED

    def _fail_the_waiting(self) -> List[str]:
        """Floor every planned attempt whose gate can no longer open, to a fixed point.

        The reason is abandonment rather than whatever ended the attempt in front: this one was
        never served, never given a deadline and never spent a step, and what happened to it is
        that the fact it was waiting for stopped being possible. Floors cascade, because a task
        gated on a task that was itself gated is one more attempt nothing will reach.
        """
        cascaded: List[str] = []
        moved = True
        while moved:
            moved = False
            for attempt_id, attempt in self._attempts.items():
                if attempt.state != PLANNED or not self._gate_is_shut_for_good(attempt_id):
                    continue
                self._fail(attempt, ABANDONED)
                cascaded.append(attempt_id)
                moved = True
        return cascaded

    def _gate_is_shut_for_good(self, attempt_id: str) -> bool:
        """Whether the fact this attempt's gate waits on can no longer happen.

        A gate names one of two facts, and an ending closes each of them the same way. A task
        waiting for another to seal waits for ever once that one has failed finally, because a
        finally failed attempt has an outcome and will never file. A task waiting for a payload
        waits for ever once the obligation at that position has been resolved without being
        rendered, which is what an ending does to it.
        """
        gate = self._release.gate_for(attempt_id)
        if gate is None:
            return False
        if gate.after_sealed_attempt_id is not None:
            blocking = self._attempts.get(gate.after_sealed_attempt_id)
            return blocking is not None and blocking.state == FINAL_FAILED
        return any(
            obligation.item.payload_position == gate.after_payload_position
            and obligation.state == FINAL_FAILED
            for obligation in self._obligations.values()
        )

    # Materialization and release.

    def _materialize(self, item: TaskItem, candidate: PayloadCandidate) -> None:
        """Record the built candidate against its obligation, then apply the release plan.

        Materialization and eligibility are two facts. A plan that released later would leave
        the obligation materialized here and make it eligible somewhere else, and the ledger
        would still say when each happened, which is the whole reason they are not one field.
        """
        obligation = self._obligations[item.attempt_id]
        obligation.candidate = candidate
        obligation.materialized = True
        obligation.state = MATERIALIZED
        if self._release.predicate == RELEASE_AT_SEAL:
            self._release_obligation(obligation, "seal")

    def _release_obligation(self, obligation: _Obligation, causal_event: str) -> None:
        """Make one obligation eligible, and record which plan did it and on what.

        The release decides readiness and nothing else. It reads no candidate content and it
        has no way to choose among candidates, so what an assignment fixed it cannot move.
        """
        obligation.state = ELIGIBLE
        assignment = self._assignments[obligation.item.attempt_id]
        self._eligibilities.append(
            _Eligibility(
                attempt_id=obligation.item.attempt_id,
                release_plan_id=self._release.release_plan_id,
                causal_event=causal_event,
                priority=self._release.priority,
                order_key=order_key(self._release, PAYLOAD, assignment),
            )
        )

    # Presentation.

    async def _commit_presentation(
        self, commit: PresentationCommit, writer: Writer
    ) -> PresentationAck:
        """Verify an attestation, then commit it.

        The replay of an attestation already committed is answered before anything else and
        without an await, so a lost acknowledgement costs a caller one round trip and nothing
        else. Everything after that is verification: the outstanding message, the cursor, the
        exact bytes, the pre-event projection, and the blobs the reference names. The commit
        itself is the last stretch, and it contains no await.
        """
        identity = presentation_request_identity(commit)
        known = self._attestation_identities.get(commit.attestation_id)
        if known is not None:
            if known != identity:
                raise StreamProtocolError("request_conflict")
            return self._attestations[commit.attestation_id]
        if commit.message_id in self._presented:
            raise StreamProtocolError("already_presented")
        pending = self._pending
        if pending is None or pending.message.message_id != commit.message_id:
            raise StreamProtocolError("invalid_message")
        if commit.cursor_before != self._cursor:
            raise StreamProtocolError("invalid_cursor")
        offered = pending.message.visible_text.encode("utf-8")
        if commit.visible_bytes_sha256 != sha256(offered).hexdigest():
            raise StreamProtocolError("invalid_message")
        if commit.stream_state_before_sha256 != self._projection_hash():
            raise StreamProtocolError("invalid_message")
        _check_blobs(pending.message.kind, commit)
        await self._verify_referenced_blobs(commit)
        # The store was read outside this transition, so the owner is checked again on the way
        # back in. A presentation from an epoch that has since been fenced commits nothing.
        self._require_writer(writer)
        # The projection is compared again for the same reason, and against the same clock. The
        # store read is the one await in here, and the deadline runs under it: an attestation
        # found current before the read can be describing a stream that has moved by the time
        # it comes back. Committing it then would file a description of a stream in which the
        # deadline had not passed, over one in which it had.
        if commit.stream_state_before_sha256 != self._projection_hash():
            raise StreamProtocolError("invalid_message")
        self._apply_presentation(pending.message.kind, pending.message.attempt_id, commit)
        self._presented[commit.message_id] = PresentedMessage(
            order=len(self._presented),
            kind=pending.message.kind,
            message_id=commit.message_id,
            attempt_id=pending.message.attempt_id,
            visible_bytes_sha256=commit.visible_bytes_sha256,
        )
        for reference in _references(commit):
            if reference not in self._committed_blobs:
                self._committed_blobs.append(reference)
        self._retain_presentation_references(pending.message.attempt_id, _references(commit))
        self._cursor = commit.message_id
        self._pending = None
        ack = PresentationAck(
            attestation_id=commit.attestation_id,
            cursor=self._cursor,
            stream_state_sha256=self._projection_hash(),
        )
        self._attestation_identities[commit.attestation_id] = identity
        self._attestations[commit.attestation_id] = ack
        return ack

    def _retain_presentation_references(
        self, attempt_id: Optional[str], references: List[str]
    ) -> None:
        """Keep, on the attempt, which objects its own committed presentations cited.

        The inventory a claim reads is flat, so it says an object is required and cannot say by
        whom. That is enough to refuse the claim and not enough to say whose evidence went
        missing, and a receipt whose transcript blob has gone would otherwise read available while
        the claim that needed it was being refused. So the association is kept here, in commit
        order, and a shared object is named by every attempt that cited it.

        A generation that declares no receipt contract keeps none of this. These names are a
        receipt's own retained inventory, a legacy attempt has no receipt to attribute a loss to,
        and synthesizing the list for one would move a carrier a legacy generation writes.
        """
        if attempt_id is None or self._capture_contract(attempt_id) is None:
            return
        attempt = self._attempts.get(attempt_id)
        if attempt is None:
            return
        for reference in references:
            if reference not in attempt.presentation_references:
                attempt.presentation_references.append(reference)

    def _apply_presentation(
        self, kind: str, attempt_id: Optional[str], commit: PresentationCommit
    ) -> None:
        if kind == "task" and attempt_id is not None:
            attempt = self._attempts[attempt_id]
            attempt.state = ACTIVE
            # The attempt keeps the checkpoint it would be restored from, so a resume can ask
            # about this attempt rather than about the set of everything anything referenced.
            attempt.task_start_checkpoint = commit.task_start_checkpoint_blob
            attempt.environment_calls = 0
            # An attempt's clock starts when the task is delivered, because that is when the
            # attempt is one a model could be working on and therefore one it could abandon.
            self._arm_deadline(attempt)
        elif kind == "seal_ack" and attempt_id is not None:
            self._attempts[attempt_id].state = ACK_PRESENTED
        elif kind == "payload" and attempt_id is not None:
            self._obligations[attempt_id].state = PRESENTED
        elif kind == "done":
            self._generation_state = DONE
            self._done_presented = True
            self._draining = True

    # What one exact Update was answered with.
    #
    # Two retry contracts exist and they stay apart. A retry that carries the same Update
    # identifier is asking for the answer that identifier already got, and inside one execution
    # the service answers it from its own record without the handler running at all. A logical
    # retry that carries a fresh identifier reaches the handler and is judged on the state it
    # finds, which is how a request for a message already presented is told so.
    #
    # A boundary destroys the first of those, because the service's record belongs to the
    # execution that accepted the Update. So this generation keeps the same record itself, for
    # every accepted Update it completes, and carries it. What is kept is the completed outcome:
    # a success, an accepted protocol refusal, or an accepted failure. A validator rejection is
    # none of those and is never kept, because the service does not keep one either and the same
    # identifier may legitimately be validated again. Neither is an exception that fails the
    # Workflow Task rather than the Update.

    def _update_id(self) -> str:
        """The exact Update identifier the handler now running was reached under."""
        current = workflow.current_update_info()
        return "" if current is None else current.id

    def _replayed(self) -> Optional[_Answer]:
        """The answer this exact identifier already got, if it got one.

        The identifier is the whole of the key, deliberately. The service deduplicates on it
        alone, and every identifier this runtime builds already names the owner that built it, so
        asking again who is writing would make this stricter than the record it stands in for and
        would refuse a paused owner the answer it is owed.
        """
        return self._journal.get(self._update_id())

    def _give_back(self, answer: _Answer) -> Any:
        """Hand back an answer this generation has already given, or raise it again."""
        if answer.kind == REFUSED:
            raise StreamProtocolError(answer.code)
        if answer.kind == FAILED:
            raise ApplicationError(answer.message, type=answer.type_name, non_retryable=True)
        if answer.kind == BY_ROW:
            return self._row(answer)
        return workflow.payload_converter().from_payload(
            _as_payload(answer.value),
            ANSWER_TYPES[answer.handler],
        )

    def _row(self, answer: _Answer) -> Any:
        """The logical row an answer points at, which is where the value itself is kept."""
        if answer.handler == PRESENT:
            return self._attestations[answer.row]
        return self._finalize_requests[answer.row].receipt

    def _remember(self, handler: str, epoch: int, answer: _Answer) -> None:
        """Write down what this exact Update was answered with, once and never again.

        The first completed outcome for an identifier is the outcome. A second write for the same
        identifier could only come from a replay of the same handler, and a replay has to produce
        what the history already holds.
        """
        self._journal.setdefault(self._update_id(), answer)

    async def _answering(self, handler: str, epoch: int, body: Any) -> Any:
        """Run one accepted handler, and keep whatever it answers with.

        The capture wraps the whole body, which is what makes it complete. A refusal raised
        before the lock is an answer; so is one raised by a table lookup that never reaches the
        lock at all; so is a failure from the work behind a filing. Every one of those is an
        accepted Update that the service recorded as completed, and every one of them has to be
        reproducible on the far side of a boundary.

        What is not kept is an exception that fails the Workflow Task rather than the Update. The
        service records no outcome for those, so neither does this: they are raised on, and the
        Task fails as it would have.

        The ledger of unfinished handlers is kept around the same body, in a finally, because a
        boundary waits for these and a caller waiting for the boundary has to be able to see them.
        The entry is made here, in the prologue and before the first suspension point, and it is
        keyed by the exact Update identifier, so a request that has to exclude itself from the
        ledger can name itself and a validator never sees the handler it is about to admit.
        """
        replayed = self._replayed()
        if replayed is not None:
            return self._give_back(replayed)
        self._handlers[self._update_id()] = handler
        try:
            result = await body()
        except StreamProtocolError as refusal:
            self._remember(handler, epoch, _Answer(handler, epoch, REFUSED, code=refusal.code))
            raise
        except (FailureError, asyncio.TimeoutError) as failure:
            kind = type(failure).__name__
            if isinstance(failure, ApplicationError) and failure.type:
                kind = failure.type
            said = getattr(failure, "message", None)
            self._remember(
                handler,
                epoch,
                _Answer(
                    handler,
                    epoch,
                    FAILED,
                    type_name=kind,
                    message=_bounded(said if isinstance(said, str) else str(failure)),
                ),
            )
            raise
        finally:
            self._handlers.pop(self._update_id(), None)
        self._remember(handler, epoch, self._answer_for(handler, epoch, result))
        return result

    def _answer_for(self, handler: str, epoch: int, result: Any) -> _Answer:
        """Where to keep one successful answer: at a row this generation holds, or by value.

        An acknowledgement and a finalization receipt are rows this generation carries whole, so
        those are pointed at and stored once however many identifiers reached them. An offered
        message is not: the request table keeps the identity a fresh request is judged against
        and drops the bytes, because the bytes are only ever owed to the exact identifier that
        was given them. So the bytes live here, inside the packed journal, which is the one place
        they are worth their size.
        """
        if handler == PRESENT:
            return _Answer(handler, epoch, BY_ROW, row=result.attestation_id)
        if handler == FINALIZE:
            return _Answer(handler, epoch, BY_ROW, row=self._finalized_by(result))
        value = _serialized(result)
        return _Answer(handler, epoch, BY_VALUE, value=value, recipe=self._recipe(handler, value))

    def _recipe(self, handler: str, value: Dict[str, Any]) -> str:
        """A way to make this answer again from the start, or nothing if there is not one.

        Only an offered task gets one. A task is the one offer built entirely out of things the
        start already carries: the work order's body, its message identifier and the generation's
        single budget, none of which any later call can change. An offered payload is not, because
        the bytes it carries are built during the run and are dropped from the projection once the
        obligation ends; an info answer is not, because it counts a queue as it stood when the
        question was asked. Those keep their literal values.

        The recipe is proved before it is used. What it would rebuild is rebuilt here and compared
        against what the handler actually answered with, and a recipe that does not reproduce
        those exact bytes is not written at all. So the fallback is not a judgement about which
        offers are reproducible, it is the outcome of trying.
        """
        if handler != PULL or value.get("kind") != "task":
            return ""
        attempt_id = value.get("attempt_id")
        if not isinstance(attempt_id, str) or attempt_id not in self._items:
            return ""
        recipe = f"task {attempt_id} {sha256(canonical_json(value)).hexdigest()}"
        if self._from_recipe(recipe) != value:
            return ""
        return recipe

    def _from_recipe(self, recipe: str) -> Optional[Dict[str, Any]]:
        """The answer a recipe denotes, or nothing if this code cannot make it.

        The task record is built exactly as the offer built it, from the same three immutable
        things, and the offered message is composed around it the same way. Nothing here reads
        the generation's current state, so what comes out is what went in however far the run
        moved on afterwards.
        """
        parts = recipe.split(" ")
        if len(parts) != 3 or parts[0] != "task":
            return None
        item = self._items.get(parts[1])
        if item is None:
            return None
        return _serialized(
            OfferedMessage(
                message_id=item.task_message_id,
                kind="task",
                visible_text=visible_bytes(
                    Task(
                        message_id=item.task_message_id,
                        attempt_id=item.attempt_id,
                        body=item.body,
                        budget=self._start.budget,
                    )
                ).decode("utf-8"),
                attempt_id=item.attempt_id,
            )
        )

    def _rebuilt(self, update_id: str, answer: _Answer) -> Dict[str, Any]:
        """Make a carried answer again from its recipe, and refuse the carrier if it differs.

        This is where the promise a recipe makes is kept. A recipe stands in for bytes a caller
        may still be owed, so a generation that cannot reproduce those exact bytes must not come
        up serving something else under the same identifier. It refuses the carrier instead, which
        is the same answer this code gives to a carrier written by another generation or by a
        version it cannot read.
        """
        made = self._from_recipe(answer.recipe)
        digest = "" if made is None else sha256(canonical_json(made)).hexdigest()
        if digest != answer.recipe.split(" ")[-1]:
            raise _refuse_carrier(
                f"the carried answer for {update_id} does not rebuild to the bytes it was "
                "originally answered with"
            )
        assert made is not None
        return made

    def _finalized_by(self, receipt: AttemptFinalized) -> str:
        """The logical request this receipt belongs to."""
        for request_id, bound in self._finalize_requests.items():
            if bound.receipt == receipt:
                return request_id
        raise ApplicationError(
            "a finalization answered with a receipt no logical request holds",
            type="JournalRowMissing",
            non_retryable=True,
        )

    # The bodies the eleven handlers run, apart from the handlers themselves, so that what is
    # captured is a whole handler rather than a handler minus whatever ran before the capture.

    def _locked(self, writer: Writer, body: Any, *arguments: Any) -> Any:
        """Run one body while holding the stream, and give the stream back however it ends."""
        ticket = self._take_lock(writer)
        try:
            return body(*arguments)
        finally:
            self._release_lock(ticket)

    async def _locked_await(self, writer: Writer, body: Any, *arguments: Any) -> Any:
        """The same, for a body that awaits."""
        ticket = self._take_lock(writer)
        try:
            return await body(*arguments)
        finally:
            self._release_lock(ticket)

    def _end_environment_call(self, call: EnvironmentCall, writer: Writer) -> EnvironmentLease:
        """Give the stream back, and say whether this call was the one holding it."""
        self._require_writer(writer)
        held = self._environment_call == call.call_id
        if held:
            self._environment_call = None
            self._release_lock(self._environment_ticket)
        return EnvironmentLease(
            call_id=call.call_id, attempt_id=call.attempt_id, cursor=self._cursor, held=held
        )

    def _close(self, writer: Writer) -> QueueClosed:
        """Close the queue to insertion."""
        return self._locked(
            writer, lambda: self._closed_queue()
        )

    def _closed_queue(self) -> QueueClosed:
        self._queue_closed = True
        return QueueClosed(task_count=len(self._start.tasks))

    def _confirm(self, writer: Writer) -> StreamState:
        """Read the state through the path a write takes, so a fenced writer is refused."""
        self._require_writer(writer)
        return self.stream_state()

    def _take_lock(self, writer: Writer) -> int:
        """Refuse a call the generation cannot accept, then hold the stream against overlap.

        This runs before the first await in every stream-affecting handler. Waiting on a lock
        would queue an overlapping call; the protocol rejects it instead. The ownership check is
        part of taking the stream rather than a step inside the call, so a fenced writer is
        refused before it has read a cursor, let alone written one.
        """
        if self._generation_state != OPEN or self._draining:
            raise StreamProtocolError("closed_stream")
        self._require_writer(writer)
        if self._consumer_id is None:
            raise StreamProtocolError("consumer_conflict")
        if self._operation_in_flight:
            raise StreamProtocolError("overlapping_call")
        self._operation_in_flight = True
        self._operation_ticket += 1
        return self._operation_ticket

    def _release_lock(self, ticket: int) -> None:
        """Release the stream, unless it has already been given to somebody else.

        A handler fenced part way through still runs its own exit. What it must not do then is
        clear a lock the new owner is holding, so the release is conditional on the ticket the
        call took still being the current one.
        """
        if self._operation_ticket == ticket:
            self._operation_in_flight = False

    def _check_claim(self, claim: OwnershipClaim) -> None:
        """Refuse a claim this generation cannot accept, without touching anything.

        It is separate from the swap because it runs twice on a resume, once before the store is
        read and once after, and because everything it does is a refusal: a claim that fails
        here leaves the epoch, the token, and the stream exactly where it found them.

        A child that has not authorized its lineage is refused before any of it is read. Ownership
        is what a claim installs, and a child owns nothing until its own history has read the
        parent's record of it.
        """
        self._require_authorized_lineage()
        if claim.protocol_version != PROTOCOL_VERSION:
            raise StreamProtocolError("unsupported_version")
        if self._generation_state != OPEN or self._draining:
            raise StreamProtocolError("closed_stream")
        if claim.configuration_hash != self._configuration_hash:
            raise StreamProtocolError("configuration_mismatch")
        if not _is_token(claim.fencing_token):
            raise StreamProtocolError("invalid_message")
        # The swap is compared before the claim's own account of itself, so a claimant that
        # read a stale epoch is told it was fenced whatever else its claim says.
        if claim.previous_epoch != self._ownership_epoch:
            raise StreamProtocolError("fenced_writer")
        if claim.reason not in ("fresh", "resume"):
            raise StreamProtocolError("invalid_message")
        if (claim.reason == "fresh") != (self._ownership_epoch == 0):
            raise StreamProtocolError("invalid_message")
        self._check_restorations(claim)

    def _check_restorations(self, claim: OwnershipClaim) -> None:
        """Refuse a claim that would continue an active attempt nobody restored.

        Every attempt the claim names has to be one this generation is holding active, under the
        exact checkpoint it retained for that attempt: a claim about some other attempt, or about
        bytes this generation never made the checkpoint, describes a restoration that did not
        happen here. And every attempt that needs one has to be named, which is the half that
        fails closed: the default claim restores nothing and is refused.
        """
        for attempt_id, checkpoint in claim.restored_checkpoints.items():
            attempt = self._attempts.get(attempt_id)
            if attempt is None or attempt.state != ACTIVE:
                raise StreamProtocolError("invalid_attempt")
            if attempt.task_start_checkpoint != checkpoint:
                raise StreamProtocolError("invalid_attempt")
        for attempt_id in self._restoration_required():
            if attempt_id not in claim.restored_checkpoints:
                raise StreamProtocolError("invalid_attempt")

    def _restoration_required(self) -> List[str]:
        """The active attempts a claim may not simply continue, in the order the queue holds them.

        An active attempt comes back as active only when nothing committed after the task-start
        checkpoint it would come back from. What this generation can commit against an attempt
        while it is active is one thing: permission for a call to a world this stream cannot
        see. A provider turn commits with an acknowledgement and a checkpoint with a Task, and
        the attempt is in neither state by then. So a granted call is the whole of the later
        commit here, and it is counted rather than inferred, because this stream never learns
        what the call did and a grant is the last moment at which it could have learned.
        """
        return [
            attempt_id
            for attempt_id, attempt in self._attempts.items()
            if attempt.state == ACTIVE and attempt.environment_calls > 0
        ]

    def _require_writer(self, writer: Writer) -> None:
        """Refuse a call that does not hold the generation's current ownership.

        The epoch says which owner is speaking and the token proves it. Both are checked here,
        at the start of every stream-affecting call and again after every await inside one,
        because a resume can arrive while a call is waiting on an Activity and the call that
        comes back is then speaking for an owner that no longer exists.

        A child that has not authorized its lineage is refused here too, and the two refusals are
        different facts. A gated child holds no ownership at all, so nothing could be speaking for
        it, and saying that it serves nothing yet is the honest answer rather than saying that the
        caller was fenced by an owner that never existed.
        """
        self._require_authorized_lineage()
        if writer.protocol_version != PROTOCOL_VERSION:
            raise StreamProtocolError("unsupported_version")
        if self._fencing_token_hash is None:
            raise StreamProtocolError("fenced_writer")
        if writer.ownership_epoch != self._ownership_epoch:
            raise StreamProtocolError("fenced_writer")
        if _token_hash(writer.fencing_token) != self._fencing_token_hash:
            raise StreamProtocolError("fenced_writer")

    async def _verify_referenced_blobs(self, commit: PresentationCommit) -> None:
        """Refuse a presentation whose blobs the store cannot produce.

        An event may cite a blob only once the complete object is installed and hashes to its
        own name, so this is a read of the store rather than a reading of the attestation. A
        generation given no store verifies nothing and reports that it does not, which is the
        honest answer: there is no object here to check the reference against.
        """
        root = self._start.blob_root
        if root is None:
            return
        await self._verify(root, _references(commit))

    async def _verify_committed_blobs(self, operation: str, claim: OwnershipClaim) -> None:
        """Refuse to hand the generation on over references the store can no longer produce.

        A presentation's references were read once, when it committed, and that read said the
        objects were there then. A resume is where to ask again: the object can change under its
        name afterwards, and a new owner would otherwise build on a history citing bytes nobody
        can produce. The refusal is the one an unverifiable reference gets anywhere else.

        The set is read until it stops growing, not once. The writer this claim replaces is
        still the writer while the store is being read, and a presentation it had already begun
        can commit in that window: its references would be in the history the claim is about to
        hand on and outside the set the claim checked. So each pass reads whatever the history
        cites that this claim has not read yet, and the last pass finds nothing new, which is
        what leaves no await between the check and the swap.
        """
        root = self._start.blob_root
        if root is None:
            return
        read: Set[str] = set()
        while True:
            outstanding = [
                reference for reference in self._committed_blobs if reference not in read
            ]
            if not outstanding:
                return
            missing = await self._unverified(root, outstanding)
            if missing:
                # The read happened outside this transition, so the claim is checked again
                # before its failure is recorded. A claimant that another one overtook while
                # this read was running has lost the swap it witnessed, and a row it wrote
                # afterwards would be a stale owner's word about evidence the owner that
                # replaced it has since verified.
                self._check_claim(claim)
                self._note_claim_refusal(operation, claim, missing)
                raise StreamProtocolError("invalid_message")
            read.update(outstanding)

    async def _require_evidence(
        self, attempt_id: str, writer: Writer, request_id: str, identity: str
    ) -> None:
        """Read the store before this execution's first dependent delivery of one obligation.

        What it reads is what this delivery depends on and nothing else: the descriptor that says
        what the cells of this source are, and the one committed body about to be offered. An
        obligation's delivery therefore never fails on another obligation's loss, and the oracle,
        the unselected eligible cell, the canonical submission and the grade evidence are none of
        this operation's business.

        The set that remembers the answer is this execution's, so the read happens once per
        obligation per execution and every later offer of it is pure. A read that refuses leaves
        the obligation unmarked, which is what makes a repair recoverable: the next entry asks
        again rather than finding a mark nothing has revisited.

        The refusal is the bare token and carries nothing else. Which object was missing, which
        cell it belonged to and which resolver looked for it are facts about this run's evidence,
        and none of them is a thing to hand the agent. They are written into this generation's own
        record instead, where a controller reads them, as a refused row naming the pull by its
        complete canonical identity: a later request carrying the same id under a moved cursor is
        a different identity and is not this operation.
        """
        if attempt_id in self._verified_deliveries:
            return
        attempt = self._attempts[attempt_id]
        if attempt.source_artifact is None:
            return
        root = self._start.blob_root
        references = [attempt.source_commitment or "", attempt.selected_body_reference or ""]
        if root is None or not all(references):
            self._note_delivery_refusal(
                attempt, writer, request_id, identity, [name for name in references if name]
            )
            raise StreamProtocolError("evidence_unavailable")
        missing = await self._unverified(root, references)
        # The read happened outside this transition, so the owner is checked again on the way back
        # in, and before either outcome is recorded rather than only before the good one. A resume
        # that arrived while the store was being read replaced the caller that asked, and a
        # replacement that has since repaired the body and offered the receipt must not then be
        # overwritten by the refusal its predecessor came back with: that row would stand as the
        # last word on an attempt whose delivery had already succeeded.
        self._require_writer(writer)
        if missing:
            self._note_delivery_refusal(attempt, writer, request_id, identity, missing)
            raise StreamProtocolError("evidence_unavailable")
        self._verified_deliveries.add(attempt_id)

    async def _verify(
        self, root: str, references: List[str], *, code: str = "invalid_message"
    ) -> None:
        """Read the store, and refuse when it cannot produce the exact bytes a name promises.

        ``code`` is which refusal an unproduceable name is. A presentation citing bytes the store
        cannot produce is a malformed message, because the attestation is what named them. A
        delivery whose own committed evidence has gone is not: nothing the caller sent is wrong,
        and what it is told is that the evidence is unavailable.
        """
        if await self._unverified(root, references):
            raise StreamProtocolError(code)

    async def _unverified(self, root: str, references: List[str]) -> List[str]:
        """Read the store and return the names it cannot produce the exact bytes for.

        The read is counted while it is happening and again when it finishes. Neither number is
        a fact about the generation's projection, and neither belongs in one. They are here
        because this is the one thing that can hold a boundary open for minutes while changing
        nothing else a caller could see: the Activity has its own timeout and its own retries,
        and a claim that is working through them is making progress that a projection hash
        cannot show. A caller held off for the boundary reads these and keeps waiting.

        What comes back is the list rather than a refusal, because the operation that asked is
        the one that knows what a name it cannot produce means: which refusal to raise, and which
        row to write about the objects it could not get.
        """
        self._verifying += 1
        try:
            verified = await workflow.execute_activity(
                verify_blobs_activity,
                VerifyBlobsInput(blob_root=root, references=references),
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
                activity_id=self._next_activity_id(),
            )
        finally:
            self._verifying -= 1
            self._verification_batches += 1
        return list(verified.unverified)

    # The rows an operation that could not get its evidence leaves behind.

    def _note_operation(
        self,
        *,
        operation: str,
        phase: str,
        reason: str,
        outcome: str,
        refused_epoch: int,
        attempt_id: Optional[str] = None,
        payload_position: Optional[int] = None,
        references: Optional[List[str]] = None,
        request_id: Optional[str] = None,
        recovered_epoch: Optional[int] = None,
    ) -> None:
        """Append one row, in commit order, and never rewrite one that is already there."""
        self._operation_failures.append(
            OperationFailure(
                operation=operation,
                phase=phase,
                reason=reason,
                outcome=outcome,
                generation=self._generation_id,
                refused_epoch=refused_epoch,
                attempt_id=attempt_id,
                payload_position=payload_position,
                references=sorted(references or []),
                request_id=request_id,
                recovered_epoch=recovered_epoch,
            )
        )

    def _standing_refusals(self, operation: str) -> List[OperationFailure]:
        """The refusals of one logical operation that nothing has answered yet.

        One row per attribution rather than one for the operation: a claim over a shared
        descriptor is refused for every sealed receipt attempt whose required set names it, and
        each of those is its own standing refusal. The last row for an attribution is the word on
        it, so a recovery already recorded is not recorded twice and an operation refused twice
        recovers once.
        """
        latest: Dict[Optional[str], OperationFailure] = {}
        for row in self._operation_failures:
            if row.operation == operation:
                latest[row.attempt_id] = row
        return [row for row in latest.values() if row.outcome == REFUSED_OPERATION]

    def _note_recovery(self, operation: str, phase: str, epoch: int) -> None:
        """Append the recovery of one operation, where a refusal under it stands.

        A recovered row is appended by one event and never by a repair: bytes going back into the
        store append nothing. What appends it is the operation itself passing the check it was
        refused on and completing, under the identity it was refused under, which is why the
        identity is what these are joined by rather than the transport call or the epoch.
        """
        for standing in self._standing_refusals(operation):
            self._note_operation(
                operation=operation,
                phase=phase,
                reason=standing.reason,
                outcome=RECOVERED_OPERATION,
                refused_epoch=standing.refused_epoch,
                attempt_id=standing.attempt_id,
                payload_position=standing.payload_position,
                references=list(standing.references),
                request_id=standing.request_id,
                recovered_epoch=epoch,
            )

    def _note_delivery_refusal(
        self,
        attempt: _Attempt,
        writer: Writer,
        request_id: str,
        identity: str,
        references: List[str],
    ) -> None:
        """Record one delivery refused because the objects it depends on are not there."""
        self._note_operation(
            operation=identity,
            phase=CONTINUED_FIRST_DELIVERY,
            reason=UNAVAILABLE_EVIDENCE,
            outcome=REFUSED_OPERATION,
            refused_epoch=writer.ownership_epoch,
            attempt_id=attempt.item.attempt_id,
            payload_position=attempt.item.payload_position,
            references=references,
            request_id=request_id,
        )

    def _note_claim_refusal(
        self, operation: str, claim: OwnershipClaim, missing: List[str]
    ) -> None:
        """Record one claim refused over objects the store can no longer produce.

        A claim carries no logical request of its own, so no request id and no payload position
        are invented for it, and the epoch it was refused under is the epoch it witnessed.

        The attribution is the whole of what is decided here. An object no attempt owns, a policy
        or a contract descriptor, is attributed to the operation and to every sealed receipt
        attempt whose required set names it, rather than to one of them chosen because it came
        first. What is left over is attributed to the operation alone, which is also the shape a
        claim refused before any receipt has sealed takes: one row, naming no attempt.

        A generation that declares no receipt contract records none of this. These rows are a
        receipt's own record of its evidence, and a generation with no receipt to lose has none:
        its refusal is the one it always was, and its carrier stays the number it always wrote.
        Provenance is never synthesized for a generation that has none.
        """
        if not self._contracts:
            return
        lost = set(missing)
        named = {
            attempt_id: sorted(self._required_by(attempt) & lost)
            for attempt_id, attempt in sorted(self._attempts.items())
            if self._required_by(attempt) & lost
        }
        for attempt_id, references in named.items():
            self._note_operation(
                operation=operation,
                phase=OWNERSHIP_CLAIM,
                reason=UNAVAILABLE_EVIDENCE,
                outcome=REFUSED_OPERATION,
                refused_epoch=claim.previous_epoch,
                attempt_id=attempt_id,
                references=references,
            )
        unattributed = sorted(lost.difference(*named.values()))
        if unattributed:
            self._note_operation(
                operation=operation,
                phase=OWNERSHIP_CLAIM,
                reason=UNAVAILABLE_EVIDENCE,
                outcome=REFUSED_OPERATION,
                refused_epoch=claim.previous_epoch,
                references=unattributed,
            )

    def _required_by(self, attempt: _Attempt) -> Set[str]:
        """Every object one sealed receipt attempt's own claim depends on.

        The descriptor and the canonical submission are its own, the grade evidence is what its
        score was taken from, the presentation references are the objects its own committed
        messages cited, and the two policy descriptors of its contract are shared with every
        other attempt under that contract, which is why a failure on one of those names all of
        them. The three cells are deliberately absent: a claim does not require them, and
        requiring them would let one attempt's unselected body block another attempt's delivery.
        """
        if attempt.source_artifact is None or not attempt.source_commitment:
            return set()
        required = {
            attempt.source_commitment,
            attempt.source_artifact.canonical_submission.sha256,
        }
        if attempt.graded_evidence:
            required.add(attempt.graded_evidence)
        required.update(attempt.presentation_references)
        contract = self._contracts.get(attempt.receipt_contract_id or "")
        if contract is not None:
            required.update(descriptor_digests([], [], [contract]))
        return required

    def _offer(
        self,
        result: Any,
        origin: str,
        request_id: str,
        identity: str,
        attempt_id: Optional[str],
        wait_reason: Optional[str] = None,
    ) -> OfferedMessage:
        """Reserve one result for one request, and record the reservation."""
        text = visible_bytes(result).decode("utf-8")
        message = OfferedMessage(
            message_id=result.message_id,
            kind=result.kind,
            visible_text=text,
            attempt_id=attempt_id,
        )
        self._pending = _Pending(message=message, origin=origin, request_id=request_id)
        self._offers.append(
            _Offer(
                request_id=request_id,
                identity=identity,
                origin=origin,
                cursor_before=self._cursor,
                kind=message.kind,
                message_id=message.message_id,
                attempt_id=attempt_id,
                visible_sha256=sha256(text.encode("utf-8")).hexdigest(),
                wait_reason=wait_reason,
            )
        )
        bindings = self._bound_by(origin)
        bindings[request_id] = _Bound(identity=identity, message=message)
        return message

    def _bound_by(self, origin: str) -> Dict[str, _Bound]:
        """The reservations one tool's requests are answered out of."""
        if origin == "pull":
            return self._pull_requests
        if origin == "info":
            return self._info_requests
        return self._terminal_requests

    def _mint_message_id(self) -> str:
        """Take the next ID from the keyed stream, for a message no manifest could count."""
        message_id = stream_message_id(self._id_key, self._hidden_ordinal)
        self._hidden_ordinal += 1
        if message_id in self._issued_ids:
            raise ApplicationError(
                "two messages drew the same public ID",
                type="MessageIdCollision",
                non_retryable=True,
            )
        self._issued_ids.add(message_id)
        return message_id

    def _projection_hash(self) -> str:
        """Hash the canonical projection of stream state.

        The projection is what a presentation attests against, so it holds the state the
        harness could have observed and nothing derived from it. It carries no hash of its
        own, which is what keeps hashing it well defined.

        A deadline that passed is part of it. The expiry is a durable transition the harness
        can read, and it happens on the generation's own clock rather than on a call, so it is
        the one transition that can land between an attestation being built and being
        committed. Without it here that attestation would still pass, and it would say the
        attempt's deadline had not passed when it had. When the deadline was armed is not
        here: no caller can read it, so nobody holding this generation's state could rebuild
        this hash from what they can observe, and every transition that moves it moves the
        attempt's state or its expiry with it.
        """
        projection = {
            "generation_state": self._generation_state,
            "cursor": self._cursor,
            "configuration_hash": self._configuration_hash,
            "release_plan_id": self._release.release_plan_id,
            "consumer_id": self._consumer_id,
            "claim_epoch": self._claim_epoch,
            "ownership_epoch": self._ownership_epoch,
            "queue_closed": self._queue_closed,
            "capacity_in_use": self._capacity_in_use(),
            "pending_message_id": (
                None if self._pending is None else self._pending.message.message_id
            ),
            "attempts": {key: value.state for key, value in self._attempts.items()},
            "final_failures": {
                key: value.final_failure
                for key, value in self._attempts.items()
                if value.final_failure is not None
            },
            "deadline_expired": sorted(
                key for key, value in self._attempts.items() if value.deadline_expired
            ),
            "obligations": {key: value.state for key, value in self._obligations.items()},
            "presented": list(self._presented),
            "materializations": self._materialized(),
            "eligibilities": self._eligibility_count(),
            "offers": self._offer_count(),
            "seal_ordinal": self._seal_ordinal,
        }
        return sha256(canonical_json(projection)).hexdigest()


def _serialized(result: Any) -> Dict[str, Any]:
    """One handler answer as the plain value a carrier can hold and the converter can read back."""
    return json.loads(workflow.payload_converter().to_payloads([result])[0].data.decode("utf-8"))


def _as_payload(value: Any) -> WirePayload:
    """One journal value as the payload the converter reads it back out of.

    The fields are set rather than passed, because this name belongs to the protocol's own
    payload record everywhere else in this module and the one wanted here is the service's.
    """
    payload = WirePayload()
    payload.metadata["encoding"] = b"json/plain"
    payload.data = canonical_json(value)
    return payload


def _as_awaited(body: Any) -> Any:
    """Present a synchronous body as one the answering wrapper can await.

    Eight of the eleven handlers do their work without awaiting anything, and three await
    Activities. The wrapper that keeps their answers should not have to know which is which.
    """

    async def awaited() -> Any:
        return body()

    return awaited


def _refuse_carrier(complaint: str) -> ApplicationError:
    """The failure a start whose carried projection does not belong to it is refused with.

    It fails the execution rather than one call, because there is no caller yet and nothing
    here can be repaired later: a generation restored from a projection that is not its own
    would serve a history nobody committed to.
    """
    return ApplicationError(complaint, type="CarrierRefused", non_retryable=True)


def _no_such_child(ordinal: int) -> PreparedChild:
    """A row naming no child, which is what a parent answers a question it has no answer to with.

    A refused Query would be indistinguishable from a parent nobody could read, and the two are
    different facts: one is authenticated and permanent, the other is infrastructure the child waits
    out while it stays gated. So the answer comes back and the caller compares.
    """
    return PreparedChild(
        child_ordinal=ordinal,
        child_workflow_id="",
        complete_start_digest="",
        origin_digest="",
        branch_slot="",
        target_cell="",
        selected_body_reference="",
        consumer_claim_hash="",
        hidden_execution_id="",
        start_differences=[],
    )


def _refuse_origin(complaint: str) -> ApplicationError:
    """The failure a child whose parent's record does not name it is refused with.

    It is the answer to a question already asked and answered: the row came from the parent's own
    record, so the two values are what they are and asking again returns them again. An
    unavailable parent, an unavailable Worker or a history nobody could read is a different thing
    entirely and is retryable infrastructure while the child stays gated.
    """
    return ApplicationError(complaint, type="OriginRefused", non_retryable=True)


def _authority_expired(error: ActivityError) -> bool:
    """Whether one origin reading failed because the window it had to be asked in had closed.

    It is read off the Activity's own failure rather than inferred from how often the reading was
    tried, because the two are different facts: a parent nobody could reach is retried until it
    answers, and a horizon that has passed is an answer of its own.
    """
    cause = error.cause
    return isinstance(cause, ApplicationError) and cause.type == EXPIRED_AUTHORITY_FAILURE


def _fork_shape_measured(name: str, value: Any, converter: Any, *, ceiling: int) -> int:
    """Measure one shape a fork transmits, and refuse the fork where the bytes will not fit.

    An oversize is a decision about the request rather than a fault in the generation, and the
    difference matters here more than anywhere else this measurement is taken. The wire failure a
    measurement raises is an ordinary Python error, so an accepted Update that let one out would
    not fail: the service would fail the Workflow Task instead, and this generation would replay
    the same activation for ever with nothing journalled, no barrier installed and no refusal
    anybody could read. So every shape one fork measures is measured through here, and what a
    caller gets back is a refusal of its own type, permanent, carrying the measurement it was
    refused over, with no barrier and no child behind it.
    """
    try:
        return check_transmitted_size(name, value, converter, ceiling=ceiling)
    except WireFormatError as oversize:
        raise ForkRefused(FORK_CONFIGURATION_VIOLATION, str(oversize)) from oversize


def _bindings(rows: List[CarriedBinding]) -> Dict[str, _Bound]:
    """One carried request table, back as the map a handler answers a retry out of."""
    return {row.request_id: _Bound(identity=row.identity, message=row.message) for row in rows}


def _carried_source(attempt: _Attempt) -> Dict[str, Any]:
    """Return one attempt's descriptor and origin as the mappings a carrier writes them as.

    They are written out rather than handed over as typed values so that the far side reads them
    back through the same strict reader the seal boundary uses. The descriptor's mapping is the
    one its canonical bytes are taken from, so what crosses and what was committed to are the
    same shape, and a member added on the way is a member the reader has to have declared.
    """
    return {
        "source_artifact": (
            None if attempt.source_artifact is None else manifest_fields(attempt.source_artifact)
        ),
        "source_origin": (
            None if attempt.source_origin is None else origin_fields(attempt.source_origin)
        ),
    }


@dataclass(frozen=True)
class _CarriedSource:
    """One carried attempt's descriptor and origin, after restore has read them."""

    manifest: Optional[SourceArtifactManifest] = None
    origin: Optional[SourceOriginContext] = None


def _read_carried_source(row: CarriedAttempt) -> _CarriedSource:
    """Read one carried attempt's raw descriptor and origin, or refuse the carrier over them.

    This is the carrier's half of the boundary the seal result has. Both arrived as whatever was
    written, and both become typed values here and nowhere earlier, so a descriptor carrying a
    member this build does not declare is refused rather than decoded without it, and a size
    written as a float is refused rather than rounded into agreement with the commitment.
    """
    try:
        return _CarriedSource(
            manifest=(
                None if row.source_artifact is None else read_source_artifact(row.source_artifact)
            ),
            origin=(
                None if row.source_origin is None else read_source_origin(row.source_origin)
            ),
        )
    except WireFormatError as error:
        raise _refuse_carrier(
            f"the source carried for attempt {row.attempt_id} is not one this build reads: "
            f"{error}"
        ) from error


@dataclass(frozen=True)
class ChildSelection:
    """What one fork child delivers from a source its parent committed.

    The attempt is the inherited one both children read, and the cell and the policy digest are
    the child's own row's rather than the parent's.
    """

    attempt_id: str
    cell: str
    policy_digest: str


def transformed_child_selection(
    projection: CarriedProjection,
    *,
    selections: Sequence[ChildSelection],
) -> CarriedProjection:
    """Return one parent's projection as a child's, the named deliveries transformed once.

    Source evidence is immutable and crosses exactly as it stands: the descriptor as the mapping
    it was written as, its commitment, the origin its seal was minted under, the seal id, the
    submission digest, the score, the decode state, the graded evidence, the seal ordinal and the
    attempt's own presentation references. Selected delivery evidence is transformed here and
    nowhere else: the cell becomes the child's target cell, the policy digest becomes the child's
    own row's, and the body reference becomes the descriptor's own entry for that cell. The
    contract id does not move, because both cells are the pair one contract declares.

    The transformation is pure. The descriptor is carried whole in state, so naming the opposite
    cell's reference reads no blob, renders nothing and needs no world.

    The candidate is dropped in the same construction and the obligation is marked pending
    preparation, so the child restores with a transformed selection, no candidate and the one
    state its own preparation clears. Dropping the candidate and leaving the selection alone would
    hand a placebo child the parent's graded selection, which its own restore refuses; dropping
    both would lose the selection a presented obligation is required to keep.

    The configuration identity stays the parent's, because a fresh child holds its parent's carry
    and nothing else: the hash inside it is the one that child's own origin records as its
    parent's, and a child's own hash reaches a carrier only at the child's own turnover.

    Nothing else moves. What a child must not inherit and what its counters become are the
    transformation the child's start is built by, and this is the half of it that is about the
    source and the delivery.
    """
    wanted: Dict[str, ChildSelection] = {}
    for selection in selections:
        if selection.attempt_id in wanted:
            raise ValueError(
                f"attempt {selection.attempt_id} is selected twice for one child, and one "
                "obligation is delivered once"
            )
        if selection.cell not in ELIGIBLE_CELLS:
            raise ValueError(
                f"a child is served {sorted(ELIGIBLE_CELLS)} and this selection names "
                f"{selection.cell!r}"
            )
        wanted[selection.attempt_id] = selection
    unknown = sorted(set(wanted) - {row.attempt_id for row in projection.attempts})
    if unknown:
        raise ValueError(
            f"the parent holds no attempt {unknown[0]}, and a child's selection is the "
            "transformation of one its parent made"
        )
    owing = sorted(set(wanted) - {owed.attempt_id for owed in projection.obligations})
    if owing:
        raise ValueError(
            f"the parent owes no payload against attempt {owing[0]}, and a child inherits the "
            "obligation it delivers against"
        )
    return replace(
        projection,
        attempts=[
            _child_attempt(row, wanted[row.attempt_id]) if row.attempt_id in wanted else row
            for row in projection.attempts
        ],
        obligations=[
            _child_obligation(owed) if owed.attempt_id in wanted else owed
            for owed in projection.obligations
        ],
    )


def _child_attempt(row: CarriedAttempt, selection: ChildSelection) -> CarriedAttempt:
    """Return one inherited attempt with the child's own selection in place of the parent's."""
    if row.source_artifact is None or not row.selected_cell:
        raise ValueError(
            f"the attempt {row.attempt_id} carries no committed selection, and a child's is the "
            "transformation of the one its parent made"
        )
    manifest = read_source_artifact(row.source_artifact)
    return replace(
        row,
        selected_cell=selection.cell,
        selected_body_reference=manifest.cells[selection.cell].sha256,
        selected_policy_digest=selection.policy_digest,
    )


def _child_obligation(owed: CarriedObligation) -> CarriedObligation:
    """Return one inherited obligation gated on the child's own preparation, its body dropped."""
    if owed.state not in _OFFERABLE_OBLIGATION:
        raise ValueError(
            f"the obligation for attempt {owed.attempt_id} is {owed.state}, and a child is cut "
            "over a payload nothing has delivered"
        )
    return replace(owed, candidate=None, pending_preparation=True)


def transformed_child_counters(projection: CarriedProjection) -> CarriedProjection:
    """Return one parent's projection with what a child must not inherit taken out of it.

    What is preserved describes the prefix both children inherited: the cursor, the closed queue,
    the hidden ordinal, the seal ordinal, the waits and their reasons, the offers, the
    eligibilities, the attempts handed out, every attempt and obligation row, the presented
    messages, the committed references and the operation failure rows the parent's own operations
    left. The Activity ordinal is preserved too, so a child's ordinary numbering equals the number
    its inherited prefix left, and the verification batch count is preserved and then incremented
    by the child's own claim exactly as any claim increments it.

    What resets is ownership and this execution's own bookkeeping. The epoch, the token, the
    consumer and the claim epoch go, so the child's own gateway takes it with a first claim and
    the parent's token can never write to it, and the cumulative claim count goes with them
    because a child's first claim is a first claim. The turnover count starts at zero and the
    parent's is recorded in the origin, so a lineage total is a sum over named source generations.

    The journal and the four tables are dropped rather than quietly absent. An identifier the
    parent answered is owed to the parent's caller, and a child answering it would hand a fenced
    transport bytes another generation produced. Nothing here is inside the projection hash, so
    dropping it moves no digest the inherited prefix committed to.
    """
    return replace(
        projection,
        ownership_epoch=0,
        fencing_token_hash=None,
        ownership_claims=0,
        consumer_id=None,
        claim_epoch=0,
        pull_requests=[],
        info_requests=[],
        terminal_requests=[],
        finalize_requests=[],
        attestations=[],
        journal=[],
        turnovers=0,
    )


def child_carrier(
    projection: CarriedProjection, *, selections: Sequence[ChildSelection]
) -> CarriedProjection:
    """Return the whole projection one fork child is handed, from the parent's own.

    Two transformations and no third: the selected delivery evidence of the named attempts
    becomes the child's, and the counters and tables become a fresh execution's. Everything else
    crosses as the parent wrote it.
    """
    return transformed_child_counters(
        transformed_child_selection(projection, selections=selections)
    )


def _provenance(attempt: _Attempt) -> Optional[SourceProvenance]:
    """Return what one attempt's committed source was, or nothing where none was committed.

    Every value is read off the descriptor the attempt is carrying rather than resolved from the
    store, which is what makes provenance readable after the world, the bank and the cell store
    are all gone. Nothing is recomputed here either: the commitment was taken over the
    descriptor's own canonical bytes when the source was committed and again at every restore,
    and a reader asking what a run came to is not the place to hash it a third time.
    """
    source = attempt.source_artifact
    if source is None:
        return None
    return SourceProvenance(
        source_commitment=attempt.source_commitment or "",
        bundle_digest=source.bundle_digest,
        environment_task_id=source.environment_task_id,
        source_attempt_id=source.source_attempt_id,
        source_seal_id=source.source_seal_id,
        execution_ordinal=source.execution_ordinal,
        canonicalization_version=source.canonicalization_version,
        renderer_configuration=source.renderer_configuration,
        canonical_submission_sha256=source.canonical_submission.sha256,
        kernel_submission_digest=source.kernel_submission_digest,
        selected_cell=attempt.selected_cell,
        selected_body_reference=attempt.selected_body_reference,
        selected_policy_digest=attempt.selected_policy_digest,
    )


def _note_failure(attempt: _Attempt, error: ActivityError) -> None:
    """Write on the attempt what the Activity behind its filing failed with.

    Everything read here is the failure Temporal recorded, which is what makes this safe to run
    inside a workflow: the same history hands back the same values on a replay, so a row rebuilt
    from it says what it said the first time. The failure read is the last one the service
    accepted: the attempt that stopped the retries, whether the retries were exhausted or the
    failure declared itself non-retryable.

    The kind is the semantic type wherever there is one. An environment that refuses a task
    raises a non-retryable application error whose type is what it refuses for, and reporting
    the class of the wrapper instead would put the same uninformative word on every deliberate
    refusal. An ordinary exception arrives with its own class name in that position already,
    because that is what the service records for one. A timeout carries which timeout instead,
    since its message is only the word.

    Every text is bounded and flattened to one line. A message is whatever an environment put in
    an exception, this runs where a Query and a JSON row will carry it, and a field that could be
    a megabyte of traceback is not one a row can hold.

    Every value is read as far as it can be read and no further. A cause is an object a caller's
    own failure converter may have built, so rendering one can raise, and a description that
    raised where it is being written would replace the failure it describes and leave the attempt
    it was meant to explain unfinished. A value whose rendering fails in the ordinary way is
    therefore absent, and the ending is written either way. A process-control signal raised while
    rendering one still stops the work, which is what such a signal is for.
    """
    cause = _as_far_as_it_reads(lambda: error.cause)
    attempt.failure_activity = _said(lambda: error.activity_type)
    attempt.failure_activity_id = _said(lambda: error.activity_id)
    attempt.failure_kind = _said(lambda: _kind_of(cause))
    attempt.failure_message = _said(lambda: _message_of(cause))
    attempt.failure_retry_state = _said(
        lambda: None if error.retry_state is None else error.retry_state.name
    )


def _kind_of(cause: Optional[BaseException]) -> Optional[str]:
    """Return what a recorded cause failed as: its semantic type, or the class carrying it."""
    if isinstance(cause, ApplicationError):
        return cause.type or type(cause).__name__
    if isinstance(cause, ActivityTimeout):
        return "TimeoutError" if cause.type is None else f"TimeoutError.{cause.type.name}"
    return None if cause is None else type(cause).__name__


def _message_of(cause: Optional[BaseException]) -> Optional[str]:
    """Return what a recorded cause said, from the failure's own message where it has one."""
    if isinstance(cause, (ApplicationError, ActivityTimeout)):
        return cause.message
    return None if cause is None else str(cause)


def _note_unusable(attempt: _Attempt, unusable: _UnusableResult) -> None:
    """Write on the attempt why the result behind its filing could not be vouched for.

    Two of the five, and deliberately not the other three. The Activity answered, so no step is
    named as having failed and there is no retry state to report: what happened is that this
    generation read the answer and would not commit it. The type and the message are the check
    that refused, which is what a reader is otherwise left to guess at, since one ending covers a
    seal for the wrong attempt, a score for another seal, a number outside what the grader
    declared, and a candidate the obligation did not ask for.
    """
    attempt.failure_kind = _said(lambda: unusable.type)
    attempt.failure_message = _said(lambda: unusable.message)


def _said(read: Any) -> Optional[str]:
    """Return what ``read`` says, bounded, or nothing when ordinarily it cannot be said."""
    said = _as_far_as_it_reads(read)
    return None if said is None else _as_far_as_it_reads(lambda: _bounded(said))


def _as_far_as_it_reads(read: Any) -> Any:
    """Return what ``read`` returns, or ``None`` where reading it raises an ordinary error.

    A description never replaces the failure it is describing. The guard stops at ``Exception``
    on purpose, and the claim it supports is bounded by that: a value that cannot be rendered is
    absent, and a process-control signal raised while rendering one is not a value that failed
    but an instruction to stop, so it goes past here as it would anywhere else.
    """
    try:
        return read()
    except Exception:  # noqa: BLE001 - an unreadable value is an absent one, not a second failure
        return None


def _bounded(text: str) -> str:
    """Return ``text`` as one line no longer than the cap, marking what the cap cut off.

    The cut is made on the encoded bytes so the cap is a size rather than a count of characters,
    and the mark goes inside the value, because a reader who cannot see that the harness stopped
    the message would read a partial explanation as a complete one. The mark is a convention and
    not a proof: a message that was short enough to keep whole is kept unchanged once flattened,
    ending included, so a value ending in those words is not by itself evidence of a cut.
    """
    flattened = " ".join(text.split())
    encoded = flattened.encode("utf-8")
    if len(encoded) <= _FAILURE_TEXT_BYTES:
        return flattened
    room = _FAILURE_TEXT_BYTES - len(_CUT.encode("utf-8"))
    return encoded[:room].decode("utf-8", "ignore") + _CUT


def _token_hash(token: str) -> str:
    """Return what the generation keeps of a fencing token: the hash of it, and not it."""
    return sha256(token.encode("utf-8")).hexdigest()


def _is_token(value: str) -> bool:
    """Whether ``value`` is shaped like a fencing token: 32 bytes, in lower-case hexadecimal."""
    return len(value) == 64 and set(value) <= set("0123456789abcdef")


def _check_start(start: StreamStart) -> None:
    """Refuse a generation this code cannot serve, before it serves anything.

    A malformed manifest has to fail the workflow rather than an Update, because there is no
    caller yet to refuse and nothing here can be repaired later. Both versions are checked, so
    a start input that mixes them, protocol two under a schedule this code does not implement,
    never serves a message either.
    """
    if start.protocol_version != PROTOCOL_VERSION:
        raise StreamProtocolError("unsupported_version")
    if start.schedule_version != SCHEDULE_VERSION:
        raise StreamProtocolError("unsupported_version")
    if start.wait_retry_after_ms < 0:
        raise StreamProtocolError("invalid_message")
    # Zero is the one value that turns the deadline off. A negative one is a configuration that
    # meant to declare a deadline and does not, and it fails here rather than serving an attempt
    # nothing will ever end.
    if start.attempt_deadline_ms < 0:
        raise StreamProtocolError("invalid_message")
    if len(start.id_key_hex) != 64 or not set(start.id_key_hex) <= set("0123456789abcdef"):
        raise StreamProtocolError("invalid_message")
    identifiers = [start.initial_cursor, start.done_message_id]
    try:
        # A declared budget goes on every task this generation offers, so a number no task
        # record would carry fails here rather than at the first offer, which is after the
        # generation has started and a consumer is already bound to it.
        if start.budget is not None:
            require_step_budget("budget", start.budget)
        # How many attempts may be live at once is a count and is held to the same rule. A
        # comparison against one lets a boolean, a float and an oversized integer through, and
        # each of them is a generation whose capacity is not the number it says it is.
        require_step_budget("capacity", start.capacity)
        # Whether the info tool is served is a yes or a no, and everything downstream reads it
        # as one: the manifest writes the tool or leaves it out, the hash writes the key or
        # leaves it out, and this workflow answers the call or refuses it. Anything else is
        # normalized by truthiness into one of the two, so a generation composed with a number
        # or a string would serve one surface and be identified as another without anything
        # having said which was meant.
        require_declaration("info", start.info)
        require_opaque_id("initial_cursor", start.initial_cursor)
        require_opaque_id("done_message_id", start.done_message_id)
        for item in start.tasks:
            identifiers += [item.task_message_id, item.ack_message_id, item.payload_message_id]
            for name in ("attempt_id", "task_message_id", "ack_message_id", "payload_message_id"):
                require_opaque_id(name, getattr(item, name))
    except WireFormatError as error:
        raise StreamProtocolError("invalid_message") from error
    if len(set(identifiers)) != len(identifiers):
        raise StreamProtocolError("invalid_message")
    _check_schedule(start)
    _check_policy(start)


def _roster(start: StreamStart) -> List[Assignment]:
    """Return the generation's assignment rows.

    A generation started without a roster gets the one its closed manifest implies, built here
    before anything is offered. That is still assignment before behavior: what the contract
    forbids is a row appearing after the behavior it could explain, not a row a manifest wrote.
    """
    return list(start.assignments) or assignments_for(start.tasks, start.release)


def _check_schedule(start: StreamStart) -> None:
    """Refuse a roster and a plan that do not describe this generation.

    The roster is checked against the queue rather than trusted beside it, because a row whose
    positions or public IDs disagree with the manifest would make two answers to the same
    question true at once.
    """
    roster = _roster(start)
    try:
        check_release(start.release, roster, evaluation_only=start.evaluation_only)
    except WireFormatError as error:
        raise StreamProtocolError("configuration_mismatch") from error
    rows = {row.attempt_id: row for row in roster}
    if set(rows) != {item.attempt_id for item in start.tasks}:
        raise StreamProtocolError("configuration_mismatch")
    for item in start.tasks:
        row = rows[item.attempt_id]
        fixed = (
            row.task_position,
            row.payload_position,
            row.task_message_id,
            row.ack_message_id,
            row.payload_message_id,
        )
        declared = (
            item.task_position,
            item.payload_position,
            item.task_message_id,
            item.ack_message_id,
            item.payload_message_id,
        )
        if fixed != declared:
            raise StreamProtocolError("configuration_mismatch")


def _check_policy(start: StreamStart) -> None:
    """Refuse a generation that has not resolved what it delivers.

    The refusal is where a generation is created rather than only in whatever composed it. A
    builder is a function a caller chooses, and what stops an experiment from being served as an
    ordinary run is not which function ran but that the generation itself carries a profile and
    a complete set of dispositions. So the profile is checked here, the coverage is checked here,
    and a start that resolves nothing may only be the legacy profile, which is what a history
    recorded before any of this decodes as.

    The matrix is the whole of the check. An ordinary generation delivers the honest policy
    against every payload it owes, withholds under one of the closed platform reasons where it
    owes none, and carries the platform's stamp on both rows. An experiment carries registered
    rows and covers every position. Neither profile can be reached from the other by an omission,
    and a start that mixes them is refused where the generation is created rather than trusted
    because some builder is supposed to have produced it.

    The profile itself is a word a caller wrote, so the authority behind it is checked here too.
    An experiment names the experiment that registered these rows and an ordinary run names the
    platform default it was stamped from, each with the digest of the rows it answered for, and
    a generation whose label has no such authority behind it is not created.

    The declared grader comes into it because honesty is a claim about a number. A generation
    whose environment is scored by a stand-in may not resolve an obligation to a policy that
    publishes the score, and that is refused before the first task is offered rather than
    discovered when the first receipt goes out.

    The legacy profile is the one thing here that cannot be refused. It is how a history recorded
    before any of this decodes, and replaying such a history runs this line: refusing it would
    refuse every resume of every generation recorded then. What is refused instead is creating
    one, which happens where a generation is started and never during a replay.
    """
    if start.profile not in PROFILES:
        raise StreamProtocolError("configuration_mismatch")
    if start.profile == LEGACY:
        if (
            start.dispositions
            or start.provenance is not None
            or start.families
            or start.receipt_contracts
            or start.receipt_source
        ):
            raise StreamProtocolError("configuration_mismatch")
        return
    roster = _roster(start)
    obligations = {
        row.attempt_id: row.payload_position
        for row in roster
        if row.creates_payload_obligation and start.release.creates_obligations
    }
    silent = {
        row.attempt_id: row.payload_position
        for row in roster
        if row.attempt_id not in obligations
    }
    try:
        check_dispositions(
            list(start.dispositions),
            profile=start.profile,
            obligations=obligations,
            silent=silent,
            grade=start.grade or KERNEL_STAND_IN_GRADE,
            provenance=start.provenance,
            families=list(start.families),
            contract_ids=[contract.contract_id for contract in start.receipt_contracts],
            served_slot=start.served_slot,
        )
        # And the shapes it admits an environment's own bodies under, with the rows that name
        # them. It is a second call rather than a field of the first because the record it admits
        # is declared where the descriptor it admits is, and that module reads the roster's.
        check_receipt_contracts(
            list(start.receipt_contracts),
            profile=start.profile,
            dispositions=list(start.dispositions),
            families=list(start.families),
            source=start.receipt_source,
        )
    except PolicyViolation as error:
        raise StreamProtocolError("configuration_mismatch") from error


def _check_carried_receipts(
    start: StreamStart, projection: CarriedProjection
) -> Dict[str, _CarriedSource]:
    """Refuse a carried projection whose receipt evidence does not hold, before any is believed.

    Every check here is pure and opens nothing. What a carrier holds is a provenance claim, and a
    claim nothing checked is a claim: the descriptor is read out of the mapping it crossed as, its
    own canonical bytes are hashed again, the seal id is recomputed from the identity the carrier
    says that seal was minted under, and every binding is compared against a value this start
    already holds rather than against another field of the same carrier.

    What comes back is what was read, by attempt, so that the values the projection is applied
    from are the values these checks were made against rather than a second decoding of the same
    bytes.

    Four things are refused whole rather than repaired, and all four are scoped to rows that
    deliver an environment's own committed bytes. A legacy presented obligation has no selection
    by design and has to stay readable, so none of these may be asked of one.

    The inventory is checked against the whole of what this generation's later operations require
    rather than against the two objects a source names. A claim reads the carried list and nothing
    else, so a name dropped from it on the way across is a name no claim will ever ask the store
    for: the descriptors a generation declaring a contract has to be able to produce are required
    from its first boundary, before it has captured anything, and every object a sealed receipt
    attempt's own claim depends on is required beside them.
    """
    contracts = {contract.contract_id: contract for contract in start.receipt_contracts}
    # The rows this generation serves, on the branch it declares it serves them on. A fork child
    # serves a slot of its own, and reading the default here instead would leave its inherited
    # source with no matching disposition and no capture contract to be validated against.
    resolved = {
        row.attempt_id: row
        for row in start.dispositions
        if row.branch_slot == start.served_slot
    }
    items = {item.attempt_id: item for item in start.tasks}
    inventory = set(projection.committed_blobs)
    attempts = {row.attempt_id: row for row in projection.attempts}
    if start.receipt_contracts:
        _require_carried(
            inventory,
            descriptor_digests(start.dispositions, start.families, start.receipt_contracts),
            "this generation's declared descriptors",
        )
    read = {row.attempt_id: _read_carried_source(row) for row in projection.attempts}
    for row in projection.attempts:
        _check_carried_attempt(start, row, read[row.attempt_id], contracts, resolved, inventory)
    for owed in projection.obligations:
        _check_carried_candidate(
            owed,
            attempts.get(owed.attempt_id),
            resolved,
            items,
            forked=start.fork_origin is not None,
        )
    for failure in projection.operation_failures:
        _check_carried_failure(failure, items)
    return read


def _check_carried_failure(row: OperationFailure, items: Dict[str, TaskItem]) -> None:
    """Hold one carried operation failure to the closed vocabulary it was written in.

    A row is read by a controller rather than acted on by this generation, which is exactly why
    it is checked: a phase, a reason or an outcome outside these sets is a record nobody can
    classify, and an attempt named by a row that is not this generation's is a row about another
    run. Nothing here is repaired.
    """
    if (
        row.phase not in OPERATION_PHASES
        or row.reason not in OPERATION_REASONS
        or row.outcome not in OPERATION_OUTCOMES
    ):
        raise _refuse_carrier(
            f"a carried operation failure is written as {row.phase}/{row.reason}/{row.outcome}, "
            "which is not a phase, a reason and an outcome this code declares"
        )
    if not row.operation or not row.generation:
        raise _refuse_carrier("a carried operation failure names no operation or no generation")
    if row.attempt_id is not None and row.attempt_id not in items:
        raise _refuse_carrier(
            f"a carried operation failure names the attempt {row.attempt_id}, which is not one "
            "of this generation's"
        )


def _authorized_origins(start: StreamStart) -> Tuple[SourceOriginContext, ...]:
    """Return the identities a carried source may name, per attempt rather than per generation.

    An ordinary generation's source origin is its own: a continuation hands the same start on, so
    the only source it can hold is one it captured itself, and a foreign origin is refused on
    sight. A fork child's is one of two, and which of them a row names is a fact about that
    attempt: a source it inherited was sealed under the parent's identity, which the child's own
    fork origin records, and one it captured locally was sealed under the child's own exactly as
    an ordinary generation's is. Both values sit inside the child's own start, so the comparison
    is pure; the authority of the origin record itself is established afterwards, before the child
    owns or serves anything.
    """
    here = SourceOriginContext(
        hidden_execution_id=start.hidden_execution_id,
        execution_ordinal=start.execution_ordinal,
    )
    if start.fork_origin is None:
        return (here,)
    return (
        here,
        SourceOriginContext(
            hidden_execution_id=start.fork_origin.parent_hidden_execution_id,
            execution_ordinal=start.fork_origin.parent_execution_ordinal,
        ),
    )


def _check_carried_attempt(
    start: StreamStart,
    row: CarriedAttempt,
    read: _CarriedSource,
    contracts: Dict[str, ReceiptContract],
    resolved: Dict[str, PayloadDisposition],
    inventory: Set[str],
) -> None:
    """Hold one carried attempt's source and selection to the generation that is restoring it.

    The retained presentation references are required before the source is, because an attempt
    keeps them from the moment it presents anything and a receipt generation reaches a boundary
    with a task delivered and nothing captured. A row on that side of its first seal names objects
    a later claim will read the store for, so a name dropped from the inventory on the way across
    is one no claim will ever ask for, exactly as it would be on the far side of the seal.

    The checkpoint the row names is required in both of those structures, because one transition
    wrote it to both: the task presentation that made this attempt active cited the checkpoint,
    which put it in the attempt's own association and in the generation's inventory together. A
    row that goes on naming a checkpoint missing from either is state this build never wrote, and
    the two omissions fail differently on the far side. Out of the inventory, the object is one
    no claim will read the store for. Out of the association, the loss is one no refusal can
    attribute, while the attempt still names the bytes a resume would be asked to restore from.
    """
    manifest = read.manifest
    selection = (row.selected_cell, row.selected_body_reference, row.selected_policy_digest)
    declared = resolved.get(row.attempt_id)
    checkpoint = row.task_start_checkpoint
    if (
        checkpoint is not None
        and declared is not None
        and declared.family_id in contracts
        and checkpoint not in row.presentation_references
    ):
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} would be restored from the checkpoint "
            f"{checkpoint[:16]} and its own presentations do not name it"
        )
    # The objects this attempt's own committed presentations cited, whether or not it has
    # captured anything, the checkpoint just required among them. A generation declaring no
    # receipt contract retains none of these, so nothing about a legacy row is asked here.
    _require_carried(
        inventory,
        row.presentation_references,
        f"the presentations of attempt {row.attempt_id}",
    )
    if manifest is None:
        if any(selection) or row.source_commitment or read.origin:
            raise _refuse_carrier(
                f"the carried attempt {row.attempt_id} holds a selection and no source for it to "
                "be a cell of"
            )
        if row.receipt_contract_id:
            raise _refuse_carrier(
                f"the carried attempt {row.attempt_id} names a capture contract and carries no "
                "source captured under it"
            )
        if (
            row.state in (SEALED, ACK_PRESENTED)
            and declared is not None
            and declared.family_id in contracts
        ):
            raise _refuse_carrier(
                f"the carried attempt {row.attempt_id} sealed under the receipt contract "
                f"{declared.family_id} and carries no source"
            )
        return
    commitment = row.source_commitment
    if commitment is None:
        raise _refuse_carrier(
            f"the source carried for attempt {row.attempt_id} names no commitment, and a "
            "descriptor nothing was committed over is a claim rather than evidence"
        )
    contract = contracts.get(row.receipt_contract_id or "")
    if contract is None:
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} names the capture contract "
            f"{row.receipt_contract_id!r} and this generation declares {sorted(contracts)}"
        )
    if manifest.receipt_contract != contract:
        raise _refuse_carrier(
            f"the source carried for attempt {row.attempt_id} was published under a contract "
            f"other than the {contract.contract_id} this generation declares"
        )
    if manifest.schema_version not in contract.manifest_schema_versions:
        raise _refuse_carrier(
            f"this generation admits {list(contract.manifest_schema_versions)} and the source "
            f"carried for attempt {row.attempt_id} is written as {manifest.schema_version!r}"
        )
    origin = read.origin
    if origin is None:
        raise _refuse_carrier(
            f"the source carried for attempt {row.attempt_id} names no origin, and the seal id "
            "is recomputed from the identity that seal was minted under rather than assumed"
        )
    if origin not in _authorized_origins(start):
        raise _refuse_carrier(
            f"the source carried for attempt {row.attempt_id} names an origin this generation is "
            "not and no fork origin of its own vouches for"
        )
    checks = (
        (commitment, source_commitment(manifest), "commitment"),
        (manifest.source_attempt_id, row.attempt_id, "attempt"),
        (manifest.source_seal_id, row.seal_id, "seal"),
        (manifest.source_seal_id, source_seal_id(origin, row.attempt_id), "recomputed seal"),
        (manifest.execution_ordinal, origin.execution_ordinal, "execution ordinal"),
        (manifest.kernel_submission_digest, row.submission_digest, "filing digest"),
        (
            manifest.canonicalization_version,
            start.canonicalization_version,
            "canonicalization version",
        ),
        (manifest.bundle_digest, start.receipt_source, "bank source"),
        (manifest.grade_identity, start.grade or KERNEL_STAND_IN_GRADE, "grader"),
    )
    for carried, held, what in checks:
        if carried != held:
            raise _refuse_carrier(
                f"the source carried for attempt {row.attempt_id} names a {what} this generation "
                f"is not: {carried!r} against {held!r}"
            )
    # Which capture this row was made under, whether or not anything is delivered from it. A
    # withholding names a contract like every other receipt-producing row, so the association is
    # compared here rather than inside the selection below: fresh capture reads the row's own
    # contract and validates against it, and a consumer of carried state that only compared a
    # delivering row would admit a withheld source captured under the other declared contract.
    if declared is None or declared.family_id != contract.contract_id:
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} carries a source captured under "
            f"{contract.contract_id!r} and its own row names "
            f"{(declared.family_id if declared is not None else None)!r}"
        )
    # The grade evidence is required rather than checked where it happens to be present. A source
    # and the object the score was taken from commit in one transition, so a carried row holding a
    # descriptor and naming no evidence is a row this build never wrote, and reading the name as
    # optional would quietly drop the store read a claim owes the object a score stands on.
    if not row.graded_evidence:
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} holds a committed source and names no grade "
            "evidence, and a source is committed with the score it was taken beside"
        )
    _require_carried(
        inventory,
        [
            commitment,
            manifest.canonical_submission.sha256,
            row.graded_evidence,
            *descriptor_digests([], [], [contract]),
        ],
        f"the claim for attempt {row.attempt_id}",
    )
    if not any(selection):
        return
    if not all(selection):
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} holds part of a selection, and a selection is "
            "a cell, a reference and a policy together"
        )
    cell = row.selected_cell or ""
    if cell not in ELIGIBLE_CELLS:
        raise _refuse_carrier(
            f"an arm is served {sorted(ELIGIBLE_CELLS)} and the carried attempt {row.attempt_id} "
            f"holds a selection for {cell!r}"
        )
    if row.selected_body_reference != manifest.cells[cell].sha256:
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} selected a reference other than its source's "
            f"own entry for the {cell} cell"
        )
    if declared is None or declared.kind != DELIVER:
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} holds a selection and this generation resolved "
            "no delivery for it"
        )
    if (declared.policy_digest or "") != (row.selected_policy_digest or "") or declared.cell != cell:
        raise _refuse_carrier(
            f"the carried attempt {row.attempt_id} holds a selection this generation's own row "
            "does not resolve"
        )


def _require_carried(inventory: Set[str], required: Sequence[str], what: str) -> None:
    """Refuse a carried inventory that has lost a name a later operation will ask the store for.

    A claim reads the carried list and nothing else, so an object dropped on the way across is
    one no operation will ever check: the refusal a missing object is owed would become a
    successful claim over evidence nobody looked for.
    """
    for reference in required:
        if reference not in inventory:
            raise _refuse_carrier(
                f"{what} requires the object {reference[:16]} and the carried inventory does not "
                "name it"
            )


def _check_carried_candidate(
    owed: CarriedObligation,
    row: Optional[CarriedAttempt],
    resolved: Dict[str, PayloadDisposition],
    items: Dict[str, TaskItem],
    *,
    forked: bool,
) -> None:
    """Hold one carried obligation to the selection its attempt carries, where it delivers one.

    An obligation an offer could still be made from has to carry both halves of what that offer
    is made of. The selection says which committed cell this row delivers and the candidate holds
    the bytes, and the offer path reads the candidate without asking whether there is one, so a
    carrier that dropped either would restore into a generation whose next pull for that position
    has nothing to answer with and no reason to give.

    There is one state where a selection stands with no candidate: a fork child's obligation,
    selected against the parent's committed source and unbuilt until the child builds it. It is
    admitted by the preparation gate the parent wrote into the child's start rather than by the
    absence of a candidate, either of which a damaged carry also has, and rather than by the fork
    origin alone, which a child retains for the whole of its life. So the gate is what is asked
    for here, and a generation holding no fork origin never carries one at all.

    An obligation this generation resolves to no artifact policy is asked for one thing before it
    is let past: that its candidate says nothing about a source. The row above it is already held
    to that, a carried attempt naming a commitment or a selection with no descriptor being refused
    whole, and the candidate is the other half of the same rule. Fresh capture refuses those values
    where the result comes back, so a carrier holding them is state this build never wrote, and
    admitting it would restore a generation whose next boundary writes the later carrier over
    evidence nothing ever captured.
    """
    declared = resolved.get(owed.attempt_id)
    policy = _policy_of(declared)
    if owed.pending_preparation and not forked:
        raise _refuse_carrier(
            f"the obligation for attempt {owed.attempt_id} is waiting on a preparation and this "
            "generation was cut from no fork, so nothing here has an inherited body to build"
        )
    if policy is None or policy.exposure != ARTIFACT:
        if owed.pending_preparation:
            raise _refuse_carrier(
                f"the obligation for attempt {owed.attempt_id} is waiting on a preparation and "
                "this generation resolves it to no committed cell to prepare one from"
            )
        if owed.candidate is not None and resolved_echo(owed.candidate):
            raise _refuse_carrier(
                f"the candidate carried for attempt {owed.attempt_id} describes a source it "
                "resolved and this generation resolves that obligation to no committed cell"
            )
        return
    selected = None if row is None else row.selected_cell
    if owed.state == PRESENTED and not selected:
        raise _refuse_carrier(
            f"the obligation for attempt {owed.attempt_id} delivered {policy.policy_name} and its "
            "attempt holds no selection to say which cell it was"
        )
    candidate = owed.candidate
    if owed.state in _OFFERABLE_OBLIGATION:
        if not selected:
            raise _refuse_carrier(
                f"the obligation for attempt {owed.attempt_id} is {owed.state} under "
                f"{policy.policy_name} and its attempt holds no selection to say which committed "
                "cell an offer of it would carry"
            )
        if candidate is None and not owed.pending_preparation:
            raise _refuse_carrier(
                f"the obligation for attempt {owed.attempt_id} is {owed.state} under "
                f"{policy.policy_name} and carries no candidate, and an obligation an offer could "
                "still be made from keeps the body that offer would carry"
            )
    # A gate stands on an obligation an offer is still owed from, and only while its body is
    # unbuilt: preparation installs the candidate and clears the gate in one transition, so a
    # carry holding both is state nothing wrote.
    if owed.pending_preparation:
        if owed.state not in _OFFERABLE_OBLIGATION:
            raise _refuse_carrier(
                f"the obligation for attempt {owed.attempt_id} is {owed.state} and waiting on a "
                "preparation, and only an obligation an offer could still be made from waits"
            )
        if candidate is not None:
            raise _refuse_carrier(
                f"the obligation for attempt {owed.attempt_id} carries the body of a preparation "
                "it is still waiting on, and a preparation clears its gate where it installs one"
            )
    if candidate is None:
        return
    if row is None or not selected:
        raise _refuse_carrier(
            f"the obligation for attempt {owed.attempt_id} carries a candidate for "
            f"{policy.policy_name} and its attempt holds no selection"
        )
    bindings = (
        (candidate.cell, selected, "cell"),
        (candidate.source_commitment, row.source_commitment, "source"),
        (candidate.body_reference, row.selected_body_reference, "reference"),
        (candidate.inner_sha256, row.selected_body_reference, "body"),
        (candidate.policy_digest, row.selected_policy_digest or "", "policy"),
    )
    for carried, held, what in bindings:
        if carried != held:
            raise _refuse_carrier(
                f"the candidate carried for attempt {owed.attempt_id} names a {what} its "
                f"attempt's selection does not: {carried!r} against {held!r}"
            )
    # And the identities the policy this obligation was resolved to declares. A fresh candidate is
    # held to them where it comes back from the Activity, and a carried one is held to them here
    # for the same reason: the renderer and the resolver are what a reader is told produced these
    # bytes, so a carrier that could move either would move the provenance of a body that nothing
    # else in the record contradicts.
    implementations = (
        (candidate.renderer_id, policy.renderer_id, "renderer"),
        (candidate.renderer_version, policy.renderer_version, "renderer version"),
        (candidate.resolver_id, policy.resolver_id, "resolver"),
        (candidate.resolver_version, policy.resolver_version, "resolver version"),
    )
    for carried, held, what in implementations:
        if carried != held:
            raise _refuse_carrier(
                f"the candidate carried for attempt {owed.attempt_id} came back from a {what} "
                f"{policy.policy_name} does not declare: {carried!r} against {held!r}"
            )
    item = items.get(owed.attempt_id)
    if item is None:
        raise _refuse_carrier(
            f"the obligation for attempt {owed.attempt_id} carries a candidate for a position "
            "this generation's manifest does not hold"
        )
    serialized = visible_bytes(
        Payload(
            message_id=item.payload_message_id,
            attempt_id=item.attempt_id,
            body=candidate.body,
        )
    )
    measured = (
        sha256(candidate.body.encode("utf-8")).hexdigest(),
        sha256(serialized).hexdigest(),
        len(serialized),
    )
    if measured != (candidate.inner_sha256, candidate.visible_sha256, candidate.visible_byte_count):
        raise _refuse_carrier(
            f"the candidate carried for attempt {owed.attempt_id} does not measure to what it "
            "says it measures to"
        )


def _unusable(what: str) -> _UnusableResult:
    """Say what is wrong with an Activity result, as the failure that ends the attempt."""
    return _UnusableResult(what, type="UnusableActivityResult", non_retryable=True)


# What a grade identity and one of its published numbers are made of, taken from the records
# themselves so the two cannot drift apart. A result carrying a field neither of them has is
# refused rather than quietly dropped: a Worker sending one was built against something other
# than this protocol, and a generation that ignored the difference would be reading an identity
# it cannot vouch for.
_GRADE_FIELDS = frozenset(entry.name for entry in fields(GradeIdentity))
_NUMBER_FIELDS = frozenset(entry.name for entry in fields(PublishedNumber))

# How much of a result this side will read before it calls the result the problem. Every text
# field it keeps is an identifier or a word from a closed set, every count in it is small, and
# every roster is bounded by the policy under it, so nothing a grader legitimately returns comes
# near these. What they stop is a field wide enough to hold something else. A string long enough
# fills the answer to a query, so the generation would stop saying what happened to it while
# still holding a value it took from a Worker; a roster long enough is held in memory to find out
# that none of it was declared; and an integer wide enough is one no double can be made of, so
# the check that would refuse it raises part way through instead, which is an activation that
# fails rather than an attempt that ends.
_MOST_TEXT = 1024
_MOST_ENTRIES = 64
_MOST_COUNT = 1_000_000


@dataclass(frozen=True)
class _Graded:
    """One grade Activity result, after the generation has decided it is one.

    The wire shape it was built from is permissive in every field, so this is where the result
    stops being whatever a Worker encoded and becomes something the seal can go on with. The
    score and the published numbers stay as they arrived: what a number is belongs to the
    projection, which rejects a boolean that this side would have to call an integer.
    """

    attempt_id: str
    seal_id: str
    score: Any
    decode_state: str
    evidence_sha256: str
    grade: Optional[GradeIdentity]
    components: Dict[str, Any]


def _graded(result: GradeAttemptResult) -> _Graded:
    """Read one grade Activity result, or refuse a shape that is not one.

    Every field this generation goes on to read is asked about here, at the first point it
    touches the result and before any of it is compared, committed or printed. The asking is
    this side's rather than the decoder's, and that is the whole reason the wire shape carries
    no types: a result that failed to decode never becomes a failure the generation can record.
    The activation itself fails, the same activation is retried for ever, nothing is written
    about why, and queries stop being answered, so a Worker's mistake becomes a generation that
    is neither running nor ended. Everything refused here ends the attempt with a reason
    instead. The score and the published numbers pass through as they arrived: what a number is
    belongs to the projection, which rejects a boolean that this side would have to call one.

    The result itself is one of the things asked about. A Worker that answered with nothing at
    all leaves the generation reading fields off it, and reading a field off nothing is the
    failure this whole function exists to avoid.

    The protocol version is read first, because it is what says the rest of the fields mean here
    what they meant where they were written. A Worker built against an earlier protocol returns
    an identity and numbers this side would otherwise accept, commit and publish as its own.

    Sizes are bounded here for the same reason the shapes are. A value too large is not a value
    this side can refuse cleanly further on: the refusal is written with the value in it, or the
    check that would make it raises on the way, and either way the ending the result was owed
    turns back into an activation nothing recorded.
    """
    if not isinstance(result, GradeAttemptResult):
        raise _unusable(
            f"a grade is a result with fields in it, and this seal was handed "
            f"{type(result).__name__}"
        )
    version = _whole(result.protocol_version, "protocol_version")
    if version != PROTOCOL_VERSION:
        raise _unusable(
            f"a grade is read under the protocol that asked for it, and this result was "
            f"written under version {version}"
        )
    return _Graded(
        attempt_id=_text(result.attempt_id, "attempt_id"),
        seal_id=_text(result.seal_id, "seal_id"),
        score=_measured(result.score, "score"),
        decode_state=_text(result.decode_state, "decode_state"),
        evidence_sha256=_evidence(result.evidence),
        grade=_identity(result.grade),
        components=_components(result),
    )


def _text(value: Any, field_name: str) -> str:
    """Return one field of a result that has to be text, or refuse what came instead."""
    if not isinstance(value, str):
        raise _unusable(
            f"a grade's {field_name} is text, and this result carried "
            f"{type(value).__name__}"
        )
    if len(value) > _MOST_TEXT:
        raise _unusable(
            f"a grade's {field_name} is a name or a word from a closed set, and this result "
            f"carried {len(value)} characters"
        )
    return value


def _measured(value: Any, field_name: str) -> Any:
    """Return one number of a result as it arrived, or refuse an integer no double can hold.

    What a number is belongs to the projection, so this is not that question. It is the narrower
    one the projection cannot ask for itself: an integer wider than a double raises where it is
    weighed rather than failing a comparison, and a check that raises part way through is an
    activation that fails instead of an attempt that ends.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        try:
            float(value)
        except OverflowError:
            raise _unusable(
                f"a grade's {field_name} is a number this side can weigh, and this result "
                f"carried an integer of {value.bit_length()} bits"
            ) from None
    return value


def _evidence(value: Any) -> str:
    """Return the digest of what a score was taken from, or refuse a reference that is not one.

    A reference arrives as the object a Worker encoded, which is a mapping rather than the class
    it was built from, so both are read here. What the generation keeps is the digest: it is the
    name a claim verifies the store for and the name a harness resolves the verdict under, so a
    reference whose name is not a digest is a score with nothing under it.
    """
    if isinstance(value, BlobRef):
        digest = value.sha256
    elif isinstance(value, dict):
        digest = value.get("sha256")
    else:
        raise _unusable(
            "a grade names the evidence its score was taken from, and this result carried "
            f"{type(value).__name__}"
        )
    try:
        return require_digest(digest)  # type: ignore[arg-type]
    except WireFormatError as error:
        raise _unusable(f"the evidence this grade names is not a blob: {error}") from error


def _identity(value: Any) -> Optional[GradeIdentity]:
    """Return the grader a result says it is, or refuse a shape that is not an identity.

    An absent identity is a result from before graders declared one and is read as the stand-in
    further on, so it is not refused here. Everything else is rebuilt field by field: the
    identity is compared against the one this generation was built over, and a comparison is
    only as good as the thing on the other side of it being an identity at all.
    """
    if value is None:
        return None
    if isinstance(value, GradeIdentity):
        return value
    if not isinstance(value, dict):
        raise _unusable(
            f"a grade says which grader it is, and this result carried {type(value).__name__}"
        )
    unknown = sorted(set(value) - _GRADE_FIELDS)
    if unknown:
        raise _unusable(f"a grade identity has no {unknown} in it")
    components = value.get("public_components", ())
    if not isinstance(components, (list, tuple)):
        raise _unusable(
            "a grade identity's roster is a list of published numbers, and this one carried "
            f"{type(components).__name__}"
        )
    if len(components) > _MOST_ENTRIES:
        raise _unusable(
            f"a grade identity's roster is the short list of what one grader publishes, and "
            f"this one carried {len(components)} entries"
        )
    return GradeIdentity(
        grader_id=_text(value.get("grader_id"), "grader_id"),
        grader_version=_text(value.get("grader_version"), "grader_version"),
        stand_in=_flag(value.get("stand_in")),
        score_component=_text(value.get("score_component"), "score_component"),
        score_places=_whole(value.get("score_places", 0), "score_places"),
        public_components=tuple(_number(entry) for entry in components),
    )


def _number(value: Any) -> PublishedNumber:
    """Return one entry of a declared roster, or refuse a shape that is not one."""
    if isinstance(value, PublishedNumber):
        return value
    if not isinstance(value, dict):
        raise _unusable(
            "a published number is a name and a domain, and this roster carried "
            f"{type(value).__name__}"
        )
    unknown = sorted(set(value) - _NUMBER_FIELDS)
    if unknown:
        raise _unusable(f"a published number has no {unknown} in it")
    return PublishedNumber(
        name=_text(value.get("name"), "name"),
        minimum=_real(value.get("minimum"), "minimum"),
        maximum=_real(value.get("maximum"), "maximum"),
        places=_whole(value.get("places", 0), "places"),
    )


def _flag(value: Any) -> bool:
    """Return a declared yes or no, or refuse what is neither."""
    if not isinstance(value, bool):
        raise _unusable(
            f"a grade says whether it is a stand-in, and this result carried "
            f"{type(value).__name__}"
        )
    return value


def _whole(value: Any, field_name: str) -> int:
    """Return a declared count, or refuse what is not one.

    Every count a result carries names a version, a resolution or a position, so one wider than
    a person would ever write is a value this side would go on to put in a refusal.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise _unusable(
            f"a declared {field_name} is a whole number, and this result carried "
            f"{type(value).__name__}"
        )
    if abs(value) > _MOST_COUNT:
        raise _unusable(
            f"a declared {field_name} is a small whole number, and this result carried one of "
            f"{value.bit_length()} bits"
        )
    return value


def _real(value: Any, field_name: str) -> float:
    """Return a declared bound, or refuse what is not a number at all."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _unusable(
            f"a declared {field_name} is a number, and this result carried "
            f"{type(value).__name__}"
        )
    return float(_measured(value, field_name))


def _components(result: GradeAttemptResult) -> Dict[str, Any]:
    """Return the numbers a grade says it published, or refuse a shape that is not a roster.

    A result comes back as whatever was encoded rather than as the object a grader meant to
    build, so the shape is a question this side has to ask. It is asked here rather than by the
    decoder because a decoding failure is not a result: the generation would fail its activation
    on every retry, record nothing about why, and answer no query while it did. A roster this is
    not is an ending with a reason, like every other result the seal cannot vouch for.
    """
    published = result.public_components
    if not isinstance(published, dict) or not all(
        isinstance(name, str) for name in published
    ):
        raise _unusable(
            "a grade publishes its numbers under names, and this result carried "
            f"{type(published).__name__}"
        )
    if len(published) > _MOST_ENTRIES:
        raise _unusable(
            f"a grade publishes a few numbers beside its score, and this result carried "
            f"{len(published)} of them"
        )
    return {
        _text(name, "component name"): _measured(value, name)
        for name, value in published.items()
    }


def _policy_of(disposition: Optional[PayloadDisposition]) -> Optional[PayloadPolicy]:
    """Return the policy one disposition delivers under, or nothing where none was resolved.

    Nothing is the legacy generation, which resolved no obligation and whose bodies are the
    placeholder its history recorded. It is never read as a licence to publish the grade.
    """
    if disposition is None or disposition.kind != DELIVER:
        return None
    return POLICIES.get(disposition.policy_digest or "")


def _check_echo(
    candidate: PayloadCandidate,
    disposition: Optional[PayloadDisposition],
    policy: Optional[PayloadPolicy],
    item: TaskItem,
    submission_digest: str,
    published: Optional[PublicGrade],
    *,
    selected: Optional[SelectedSourceReference] = None,
    source: Optional[SourceArtifactManifest] = None,
) -> None:
    """Hold a built candidate to the policy this obligation was resolved to, body included.

    The echo is what a Worker running code this generation did not ask for cannot produce. A
    build that never learned about policies returns a candidate with no digest and no renderer
    version in it, so a generation that resolved an obligation to a policy sees the absence and
    refuses, rather than serving a placeholder under an honest record.

    An echo on its own is metadata about a body rather than the body, and metadata is the easy
    half to get right: a renderer that is faulty, substituted, or half upgraded can copy the
    requested descriptor around any bytes it likes and every field would agree. So the body is
    built again here, out of the policy this generation resolved and the grade this generation
    committed, and the candidate is required to be exactly that. What the Activity is left
    deciding is nothing: it produces bytes and the authority produces the same bytes or the
    attempt ends. The rendering is deterministic and reaches nothing, which is what lets it run
    where the decision belongs.

    An artifact policy is the one that is not rebuilt, because there is nothing deterministic to
    rebuild it from: its body is the environment's own committed bytes, and its authority is the
    committed reference, which is why the entry comparison below is not optional.

    A legacy generation asked for nothing and is owed nothing. Its recorded results carry no
    echo and are read exactly as they were recorded, which is what keeps a stopped generation
    resumable by code that has since learned to ask. What it is still held to is being one. A
    generation that resolved no obligation has no source for a candidate to have resolved, so
    the four values a resolved one echoes are as empty on its result as its history says they
    are, and a result that filled one would put provenance on an attempt that captured none and
    carry it as the evidence that moves the whole generation to the later carrier.
    """
    if disposition is None:
        _check_unresolved(candidate, "this generation resolved no obligation to any policy")
        return
    expected = "" if disposition.policy_digest is None else disposition.policy_digest
    if policy is None:
        raise _UnusableResult(
            f"the obligation was resolved to a policy this build does not implement, so the "
            f"candidate under {expected[:16]} is one nothing can vouch for",
            type="RendererDescriptorMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )
    actual = (candidate.policy_digest, candidate.renderer_id, candidate.renderer_version)
    declared = (expected, policy.renderer_id, policy.renderer_version)
    if actual != declared:
        raise _UnusableResult(
            f"this obligation was resolved to {policy.policy_name} rendered by "
            f"{policy.renderer_id}/{policy.renderer_version}, and the candidate came back from "
            f"{candidate.renderer_id or 'a build that echoed nothing'}",
            type="RendererDescriptorMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )
    if candidate.cell != disposition.cell:
        raise _UnusableResult(
            f"this obligation was assigned the cell {disposition.cell!r} and the candidate "
            f"came back in {candidate.cell!r}",
            type="RendererDescriptorMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )
    if policy.exposure == ARTIFACT:
        _check_resolved(candidate, policy, item, selected, source)
        return
    _check_unresolved(candidate, f"{policy.policy_name} renders its body from a projection")
    try:
        declared_body = render_body(
            policy,
            grade=published,
            payload_position=item.payload_position,
            submission_digest=submission_digest,
        )
    except PolicyViolation as violation:
        raise _unusable(str(violation)) from violation
    if candidate.body != declared_body:
        raise _UnusableResult(
            f"the candidate echoes {policy.policy_name} and carries a body that policy does not "
            "render, so what the record says this agent was told and what it would be told are "
            "two different things",
            type="RendererDescriptorMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )


def _check_unresolved(candidate: PayloadCandidate, route: str) -> None:
    """Refuse a candidate that describes a source on a route where nothing resolved one.

    Two routes reach here and they are one condition. A policy that renders from a projection has
    no source to resolve, and a generation that resolved no obligation at all declared none to
    resolve one from. What either would be admitting is provenance nobody captured, standing on
    an attempt whose record is that it captured nothing, and read afterwards as the evidence that
    decides which carrier the generation writes. So the values are asked to be empty here rather
    than left to the comparisons an artifact route makes, which this route never runs.
    """
    if resolved_echo(candidate):
        raise _UnusableResult(
            f"{route}, and this candidate came back describing a source it resolved",
            type="RendererDescriptorMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )


def _check_resolved(
    candidate: PayloadCandidate,
    policy: PayloadPolicy,
    item: TaskItem,
    selected: Optional[SelectedSourceReference],
    source: Optional[SourceArtifactManifest],
) -> None:
    """Hold a resolved candidate to the source that was committed, without rebuilding it.

    Every check here is pure and every value it is made against is one this generation already
    held. The bytes are required to hash to the committed entry for the cell that was selected,
    which is what a candidate matching its own reported digest does not say: a body says what it
    is and never which cell was chosen. The size and the encoding are the contract's. The mask is
    expanded from the contract's own geometry and never from anything the candidate proposed, and
    the masked body is compared with the hash publication took while it held both cells, which
    makes this a consistency check on the selected body rather than a proof about the other one.
    And the wire count is the obligation's empty-body wrapper plus the stored count of the body,
    exactly, which is what makes two cells of one source come to one measurement.
    """
    if selected is None or source is None:
        raise _unusable(
            f"this obligation delivers {policy.policy_name} and no committed source was selected "
            "for it"
        )
    contract = source.receipt_contract
    resolver = (candidate.resolver_id, candidate.resolver_version)
    if resolver != (policy.resolver_id, policy.resolver_version):
        raise _UnusableResult(
            f"{policy.policy_name} resolves its body through "
            f"{policy.resolver_id}/{policy.resolver_version}, and this candidate came back from "
            f"{candidate.resolver_id or 'a build that echoed nothing'}",
            type="RendererDescriptorMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )
    if candidate.source_commitment != selected.source_commitment:
        raise _unusable(
            "the candidate names a source other than the one this obligation was selected from"
        )
    if candidate.body_reference != selected.body_sha256:
        raise _unusable(
            "the candidate resolved a reference other than the committed entry for its cell"
        )
    if candidate.inner_sha256 != selected.body_sha256:
        raise _unusable(
            "the body that came back does not hash to the entry this source committed for the "
            "cell this obligation was assigned"
        )
    if not candidate.body.isascii():
        raise _unusable(
            f"the contract {contract.contract_id} publishes {contract.body_encoding!r} bodies "
            "and this one is not that"
        )
    raw = candidate.body.encode("utf-8")
    if len(raw) != contract.body_size:
        raise _unusable(
            f"the contract {contract.contract_id} fixes a body at {contract.body_size} bytes and "
            f"this one is {len(raw)}"
        )
    masked = sha256(masked_body(raw, mask_spans(contract))).hexdigest()
    if masked != source.pair_parity.masked_body_sha256:
        raise _unusable(
            "the body that came back is not what publication masked under the slots this "
            "contract registers"
        )
    expected = payload_wire_count(
        payload_message_id=item.payload_message_id,
        attempt_id=item.attempt_id,
        encoded_body_bytes=source.pair_parity.encoded_body_bytes,
    )
    if candidate.visible_byte_count != expected:
        raise _unusable(
            f"this obligation's cells come to {expected} visible bytes and this candidate comes "
            f"to {candidate.visible_byte_count}"
        )


def _check_family(candidate: PayloadCandidate, family: Optional[MatchedFamily]) -> None:
    """Hold a candidate built for a matched arm to the shape that arm declared.

    A row outside a family is measured against nothing here: what its body comes to is what its
    policy renders, and there is no counterpart it has to be indistinguishable from. A row
    inside one has both, because the whole of what a matched arm claims is that its cells differ
    in what they say. A candidate longer than its family's declared count, or built in another
    group, is a cell an agent or a harness could pick out without reading it, and serving one
    would be running the comparison the registration says is being run while measuring something
    else.

    The count is the arm's rather than this generation's. Every leg of a matched arm declares
    the same family and each leg resolves the cell it serves, so what holds the cells to one
    another across legs is that each of them is held to the number the arm declared.
    """
    if family is None:
        return
    if candidate.match_group != family.match_group:
        raise _UnusableResult(
            f"the matched family {family.family_id} builds its cells in "
            f"{family.match_group!r} and this candidate came back in {candidate.match_group!r}",
            type="MatchedFamilyMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )
    if candidate.visible_byte_count != family.visible_byte_count:
        raise _UnusableResult(
            f"the matched family {family.family_id} declares cells of "
            f"{family.visible_byte_count} visible bytes and this candidate came to "
            f"{candidate.visible_byte_count}, which is a cell that can be told from its "
            "counterpart without being read",
            type="MatchedFamilyMismatch",
            non_retryable=True,
            final_failure=SEAL_RENDERER,
        )


def _read_candidate(returned: PayloadCandidateResult) -> PayloadCandidate:
    """Make a candidate of what came back, at the boundary the rest of the result is read at.

    The four values a resolved candidate echoes cross as whatever was encoded, so this is where
    they become a record. A mapping or a list written where a digest or an implementation name
    belongs is refused here with a reason recorded against the attempt: a field with a type on
    the result would have made that same shape a decoding failure instead, raised while the
    generation was being handed the result and before any code of its own ran, which the retries
    would reproduce for ever and the record would not explain.

    The comparisons below cannot stand in for this. An empty mapping is not a digest and is also
    not something any of them notices: a route that renders from a projection asks these four to
    be empty, and a value that is falsy without being text would pass that and go on to be
    carried, where the encoder that has to write it raises with nothing recording.

    A value the canonical encoding cannot write is refused for that last reason. What comes back
    here is state that crosses a boundary, and the reading and the writing are one question: a
    string holding a lone surrogate reads back cleanly and raises where the carrier is packed,
    which is outside every recorded refusal there is.
    """
    return PayloadCandidate(
        cell=returned.cell,
        renderer_id=returned.renderer_id,
        match_group=returned.match_group,
        body=returned.body,
        inner_sha256=returned.inner_sha256,
        visible_sha256=returned.visible_sha256,
        visible_byte_count=returned.visible_byte_count,
        renderer_version=returned.renderer_version,
        policy_digest=returned.policy_digest,
        source_commitment=_echoed(returned.source_commitment, "source commitment"),
        body_reference=_echoed(returned.body_reference, "body reference"),
        resolver_id=_echoed(returned.resolver_id, "resolver id"),
        resolver_version=_echoed(returned.resolver_version, "resolver version"),
    )


def _echoed(value: Any, field_name: str) -> str:
    """Return one value a candidate echoed about the source it resolved, or refuse it."""
    if not isinstance(value, str):
        raise _unusable(
            f"a candidate's {field_name} is text, and this result carried "
            f"{type(value).__name__}"
        )
    if len(value) > _MOST_TEXT:
        raise _unusable(
            f"a candidate's {field_name} is a digest or a name, and this result carried "
            f"{len(value)} characters"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise _unusable(
            f"a candidate's {field_name} is text the record can be written in, and this result "
            f"carried one that cannot: {error}"
        ) from error
    return value


def _check_candidate(item: TaskItem, candidate: PayloadCandidate) -> None:
    """Hold a built candidate to its own measurements before anything acknowledges it.

    The stream serves the candidate's body, and a family gate compares the hashes and byte
    counts the bundle recorded, so a candidate whose measurements do not describe what it
    carries is a build nothing downstream can check. Acknowledging it would attest to gates
    that never ran on the bytes a model would read.
    """
    if candidate.inner_sha256 != sha256(candidate.body.encode("utf-8")).hexdigest():
        raise _unusable("the candidate's inner hash does not cover the body it carries")
    serialized = visible_bytes(
        Payload(
            message_id=item.payload_message_id,
            attempt_id=item.attempt_id,
            body=candidate.body,
        )
    )
    if candidate.visible_sha256 != sha256(serialized).hexdigest():
        raise _unusable("the candidate's visible hash does not cover the result it would be")
    if candidate.visible_byte_count != len(serialized):
        raise _unusable("the candidate's byte count is not the result's byte count")


def _argument_complaint(names: List[str], arguments: Dict[str, Any]) -> Optional[str]:
    """Say what is wrong with a terminal call's native arguments, or nothing.

    The complaint is built from the declared schema and the submitted names alone. It is
    deterministic, and it says nothing an evaluation would say.
    """
    declared = set(names)
    submitted = set(arguments)
    missing = sorted(declared - submitted)
    unknown = sorted(submitted - declared)
    if not missing and not unknown:
        return None
    parts = []
    if missing:
        parts.append("missing " + ", ".join(missing))
    if unknown:
        parts.append("unknown " + ", ".join(unknown))
    return "; ".join(parts)


def _references(commit: PresentationCommit) -> List[str]:
    """Every object this presentation names, once each and in a fixed order."""
    named = [commit.transcript_blob, commit.provider_turn_blob, commit.task_start_checkpoint_blob]
    ordered: List[str] = []
    for reference in named:
        if reference is not None and reference not in ordered:
            ordered.append(reference)
    return ordered


def _check_blobs(kind: str, commit: PresentationCommit) -> None:
    """Hold a presentation to the blobs its kind requires and forbid the ones it does not.

    A Task presentation is the checkpoint a crash restores from, so it has to carry one. An
    acknowledgement is the last result of a completed provider turn, so it has to carry that
    turn and say the turn is complete. Everything else carries neither.
    """
    if kind == "task":
        valid = (
            commit.task_start_checkpoint_blob is not None
            and commit.provider_turn_blob is None
            and not commit.completed_turn
        )
    elif kind == "seal_ack":
        valid = (
            commit.provider_turn_blob is not None
            and commit.task_start_checkpoint_blob is None
            and commit.completed_turn
        )
    else:
        valid = (
            commit.provider_turn_blob is None
            and commit.task_start_checkpoint_blob is None
            and not commit.completed_turn
        )
    if not valid:
        raise StreamProtocolError("invalid_message")
