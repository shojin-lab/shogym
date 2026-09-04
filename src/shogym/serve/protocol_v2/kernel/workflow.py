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
from dataclasses import dataclass, fields
from datetime import timedelta
from hashlib import sha256
from typing import Any, Dict, List, Optional, Set, Tuple

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError
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
        order_key,
        presentation_request_identity,
        pull_request_identity,
        require_digest,
        require_opaque_id,
        require_step_budget,
        stream_message_id,
        submission_digest,
        terminal_request_identity,
        visible_bytes,
    )
    from shogym.serve.protocol_v2.kernel.activities import (
        generate_payload_bundle_activity,
        grade_attempt_activity,
        seal_attempt_activity,
        verify_blobs_activity,
    )
    from shogym.serve.protocol_v2.kernel.messages import (
        ABANDONED,
        DEADLINE,
        FINAL_FAILURE_REASONS,
        SEAL_FAILED,
        SEAL_RENDERER,
        SEAL_UNUSABLE,
        AttemptFinalized,
        AttemptRecord,
        ConsumerClaim,
        ConsumerReceipt,
        EnvironmentCall,
        EnvironmentLease,
        FinalizeRequest,
        finalize_request_identity,
        GeneratePayloadBundleInput,
        GenerationRecords,
        GradeAttemptInput,
        GradeAttemptResult,
        OfferedMessage,
        OwnershipClaim,
        OwnershipReceipt,
        PayloadCandidate,
        PresentedMessage,
        QueueClosed,
        SealAttemptInput,
        SealRequest,
        StreamOutcome,
        StreamStart,
        StreamState,
        TaskItem,
        VerifyBlobsInput,
        Writer,
        assignments_for,
        configuration_hash,
        hidden_seal_id,
    )
    from shogym.serve.protocol_v2.policy import (
        DELIVER,
        HONEST,
        KERNEL_STAND_IN_GRADE,
        LEGACY,
        POLICIES,
        PROFILES,
        SINGLETON_SLOT,
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

# The analysis outcome a finalized attempt is assigned. An attempt that was ended rather than
# filed has nothing to grade, and the floor is what the outcome is fixed at instead.
FLOOR = 0.0

OPEN = "open"
DONE = "done"

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

# How much of an Activity's failure a row keeps, in bytes, and the mark left where the cap cut a
# value short. The cap is one size for every text among them, because what is being bounded is a
# row rather than a particular field, and no name a service reports comes anywhere near it.
_FAILURE_TEXT_BYTES = 512
_CUT = " [truncated]"


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


@dataclass
class _Obligation:
    item: TaskItem
    state: str = ASSIGNED
    candidate: Optional[PayloadCandidate] = None


@dataclass
class _Eligibility:
    """An obligation becoming eligible: which plan released it, on what, and in what order."""

    attempt_id: str
    release_plan_id: str
    causal_event: str
    priority: str
    order_key: Tuple[int, int, str]


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
        # start. The key keeps its branch anyway, because the fork that will create those slots
        # is the reason two rows can share one obligation.
        self._served: Dict[str, PayloadDisposition] = {
            row.attempt_id: row
            for row in start.dispositions
            if row.branch_slot == SINGLETON_SLOT and row.kind == DELIVER
        }
        # The matched arms this generation's rows are cells of, by the name a row claims.
        self._families: Dict[str, MatchedFamily] = {
            family.family_id: family for family in start.families
        }
        self._pending: Optional[_Pending] = None
        self._pull_requests: Dict[str, _Bound] = {}
        self._terminal_requests: Dict[str, _Bound] = {}
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
            descriptor_digests(start.dispositions, start.families)
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
        if start.profile == LEGACY and workflow.patched("profile-required-at-creation"):
            raise StreamProtocolError("configuration_mismatch")
        while not self._done_presented:
            await self._wait_for_done_or_a_deadline()
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
        """
        self._end_expired()
        armed = self._armed_deadlines()
        if not armed:
            await workflow.wait_condition(
                lambda: self._done_presented
                or bool(self._armed_deadlines())
                or self._expiry_can_be_applied()
            )
            return
        attempt_id, deadline = min(armed.items(), key=lambda row: (row[1], row[0]))
        remaining = deadline - self._now_ms()
        if remaining > 0:
            try:
                await workflow.wait_condition(
                    lambda: self._done_presented
                    or self._armed_deadlines() != armed
                    or self._expiry_can_be_applied(),
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

    # The Updates a gateway calls.

    @workflow.update
    async def claim_ownership(self, claim: OwnershipClaim) -> OwnershipReceipt:
        """Take the generation from whoever held it, by compare and swap on the epoch.

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
        self._check_claim(claim)
        if self._committed_blobs:
            await self._verify_committed_blobs()
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
        return OwnershipReceipt(
            ownership_epoch=self._ownership_epoch,
            previous_epoch=previous,
            fencing_token_hash=self._fencing_token_hash,
            configuration_hash=self._configuration_hash,
            claimant_id=claim.claimant_id,
            reason=claim.reason,
        )

    @workflow.update
    async def claim_consumer(self, claim: ConsumerClaim, writer: Writer) -> ConsumerReceipt:
        """Bind the generation's one logical consumer.

        The same claim presented twice returns the same receipt, because a lost response must
        not cost a caller its stream. A different one is refused, before any message is
        offered and without touching state.
        """
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
        return ConsumerReceipt(
            consumer_id=self._consumer_id,
            claim_epoch=self._claim_epoch,
            initial_cursor=self._start.initial_cursor,
            configuration_hash=self._configuration_hash,
        )

    @workflow.update
    async def pull(self, request: PullRequest, writer: Writer) -> OfferedMessage:
        """Return the one message this request is entitled to.

        A retry of the same request gets its own result back while that result is unpresented,
        and an error afterwards. A new request gets a new selection, but only from a current
        cursor and only when nothing is outstanding: a second request never inherits the first
        one's offer.
        """
        ticket = self._take_lock(writer)
        try:
            return self._pull(request)
        finally:
            self._release_lock(ticket)

    @workflow.update
    async def seal_attempt(self, request: SealRequest, writer: Writer) -> OfferedMessage:
        """End an attempt, or say why the filing was not one this tool accepts.

        Nothing about the seal becomes authoritative until the last transition below, which
        contains no await. The acknowledgement is built there and returned after it, so a
        crash anywhere earlier leaves a stream that has not acknowledged anything.
        """
        ticket = self._take_lock(writer)
        try:
            return await self._seal(request, writer)
        finally:
            self._release_lock(ticket)

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
        ticket = self._take_lock(writer)
        try:
            return await self._commit_presentation(commit, writer)
        finally:
            self._release_lock(ticket)

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
        ticket = self._take_lock(writer)
        try:
            return self._finalize_requested(request)
        finally:
            self._release_lock(ticket)

    @workflow.update
    async def close_queue(self, writer: Writer) -> QueueClosed:
        """Close the queue to insertion. It revokes nothing and seals nothing."""
        ticket = self._take_lock(writer)
        try:
            self._queue_closed = True
            return QueueClosed(task_count=len(self._start.tasks))
        finally:
            self._release_lock(ticket)

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
        """
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
        """
        self._require_writer(writer)
        held = self._environment_call == call.call_id
        if held:
            self._environment_call = None
            self._release_lock(self._environment_ticket)
        return EnvironmentLease(
            call_id=call.call_id, attempt_id=call.attempt_id, cursor=self._cursor, held=held
        )

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
        self._require_writer(writer)
        return self.stream_state()

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
            ownership_claims=len(self._ownership),
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
            eligibility_count=len(self._eligibilities),
            offer_count=len(self._offers),
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
        Wait, a SealReject and the Done are committed the same way, and a reconciliation blind to
        them would pass a run that lost one.

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
            attempts=self.attempt_records(), presentations=self.presented_messages()
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
        """
        item = attempt.item
        obligation = self._obligations.get(item.attempt_id)
        resolved = next(
            (
                row
                for row in self._start.dispositions
                if row.attempt_id == item.attempt_id and row.branch_slot == SINGLETON_SLOT
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
        )

    # Pull.

    def _pull(self, request: PullRequest) -> OfferedMessage:
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
        return self._select(request.request_id, identity)

    def _select(self, request_id: str, identity: str) -> OfferedMessage:
        choice = self._first_eligible()
        if choice is not None:
            kind, attempt_id = choice
            if kind == PAYLOAD:
                return self._offer_payload(attempt_id, request_id, identity)
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
        """How many obligations have their candidate built.

        The candidate is the materialization: it carries the renderer, the match group, the
        hashes, and the byte count a family gate compares, so counting the obligations that
        hold one is reading the fact rather than a tally kept beside it.
        """
        return sum(1 for o in self._obligations.values() if o.candidate is not None)

    def _wait_reason_counts(self) -> Dict[str, int]:
        """How many Waits each hidden reason accounts for, for the harness alone."""
        counts: Dict[str, int] = {}
        for offer in self._offers:
            if offer.wait_reason is not None:
                counts[offer.wait_reason] = counts.get(offer.wait_reason, 0) + 1
        return counts

    def _capacity_in_use(self) -> int:
        occupied = (TASK_OFFERED, ACTIVE, SEALING)
        return sum(1 for attempt in self._attempts.values() if attempt.state in occupied)

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
        try:
            return await self._seal_batch(request, attempt, identity, writer)
        except StreamProtocolError:
            raise
        except ActivityError as error:
            self._require_writer(writer)
            _note_failure(attempt, error)
            self._finalize(attempt, SEAL_FAILED)
            raise
        except _UnusableResult as unusable:
            self._require_writer(writer)
            _note_unusable(attempt, unusable)
            self._finalize(attempt, unusable.final_failure)
            raise

    async def _seal_batch(
        self, request: SealRequest, attempt: _Attempt, identity: str, writer: Writer
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
        sealed = await workflow.execute_activity(
            seal_attempt_activity,
            SealAttemptInput(
                attempt_id=attempt.item.attempt_id,
                seal_id=attempt.seal_id,
                native_terminal_name=request.native_terminal_name,
                canonicalization_version=self._start.canonicalization_version,
                native_arguments=request.native_arguments,
                blob_root=self._start.blob_root,
            ),
            start_to_close_timeout=_TERMINAL_ACTIVITY_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
        )
        if sealed.attempt_id != attempt.item.attempt_id or sealed.seal_id != attempt.seal_id:
            raise _unusable("the sealed submission is not the one this seal asked for")
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
        if attempt.item.attempt_id in self._obligations:
            disposition = self._served.get(attempt.item.attempt_id)
            policy = _policy_of(disposition)
            published = self._published(attempt, graded, policy, declared)
            bundle = await workflow.execute_activity(
                generate_payload_bundle_activity,
                GeneratePayloadBundleInput(
                    attempt_id=attempt.item.attempt_id,
                    payload_position=attempt.item.payload_position,
                    payload_message_id=attempt.item.payload_message_id,
                    submission_digest=attempt.submission_digest,
                    canonical_submission_text=sealed.canonical_submission_text,
                    policy_digest="" if disposition is None else (disposition.policy_digest or ""),
                    cell="" if disposition is None else (disposition.cell or ""),
                    public_grade=published,
                ),
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
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
            candidate = bundle.candidates[0]
            _check_candidate(attempt.item, candidate)
            _check_echo(
                candidate,
                disposition,
                policy,
                attempt.item,
                attempt.submission_digest,
                published,
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

    # Shared machinery.

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
        """
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
        """
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

    async def _verify_committed_blobs(self) -> None:
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
            await self._verify(root, outstanding)
            read.update(outstanding)

    async def _verify(self, root: str, references: List[str]) -> None:
        """Read the store, and refuse when it cannot produce the exact bytes a name promises."""
        verified = await workflow.execute_activity(
            verify_blobs_activity,
            VerifyBlobsInput(blob_root=root, references=references),
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
        )
        if verified.unverified:
            raise StreamProtocolError("invalid_message")

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
        bindings = self._pull_requests if origin == "pull" else self._terminal_requests
        bindings[request_id] = _Bound(identity=identity, message=message)
        return message

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
            "eligibilities": len(self._eligibilities),
            "offers": len(self._offers),
            "seal_ordinal": self._seal_ordinal,
        }
        return sha256(canonical_json(projection)).hexdigest()


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
    if start.capacity < 1 or start.wait_retry_after_ms < 0:
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
        if start.dispositions or start.provenance is not None or start.families:
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
        )
    except PolicyViolation as error:
        raise StreamProtocolError("configuration_mismatch") from error


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

    A legacy generation asked for nothing and is owed nothing. Its recorded results carry no
    echo and are read exactly as they were recorded, which is what keeps a stopped generation
    resumable by code that has since learned to ask.
    """
    if disposition is None:
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
