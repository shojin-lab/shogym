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

The refusals are protocol errors, raised as :class:`StreamProtocolError`. They carry a code
from the closed set and nothing else, they change no state, and they are never a message.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from typing import Any, Dict, List, Optional, Set, Tuple

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from shogym.serve.protocol_v2 import (
        PROTOCOL_VERSION,
        RELEASE_AT_SEAL,
        SCHEDULE_VERSION,
        Assignment,
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
        require_opaque_id,
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
        ConsumerClaim,
        ConsumerReceipt,
        EnvironmentCall,
        EnvironmentLease,
        GeneratePayloadBundleInput,
        GradeAttemptInput,
        OfferedMessage,
        OwnershipClaim,
        OwnershipReceipt,
        PayloadCandidate,
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
    from shogym.serve.protocol_v2.schedule import PAYLOAD, TASK

PLANNED = "planned"
TASK_OFFERED = "task_offered"
ACTIVE = "active"
SEALING = "sealing"
SEALED = "sealed"
ACK_PRESENTED = "ack_presented"

ASSIGNED = "assigned"
MATERIALIZED = "materialized"
ELIGIBLE = "eligible"
OFFERED = "offered"
PRESENTED = "presented"

# The states Done has to look through. An attempt that has not reached a presented
# acknowledgement is still live, and an obligation that has not been presented is still owed.
# The failure states a later element adds are terminal and obligation fulfilling, so they join
# the fulfilled side rather than either of these.
LIVE_ATTEMPT = (PLANNED, TASK_OFFERED, ACTIVE, SEALING, SEALED)
UNFULFILLED_OBLIGATION = (ASSIGNED, MATERIALIZED, ELIGIBLE, OFFERED)

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
        self._pending: Optional[_Pending] = None
        self._pull_requests: Dict[str, _Bound] = {}
        self._terminal_requests: Dict[str, _Bound] = {}
        self._attestations: Dict[str, PresentationAck] = {}
        self._attestation_identities: Dict[str, str] = {}
        # Every presented message, in order, with the kind it was. The kind is what keeps
        # payload delivery a count of its own rather than a share of the presentations.
        self._presented: Dict[str, str] = {}
        # Every object a committed presentation referenced, once each. A reference verified when
        # its event committed is a fact about that moment, and this is what makes it a question
        # a resume can ask again.
        self._committed_blobs: List[str] = []
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
        """Serve until Done has been presented, then let every accepted call finish."""
        await workflow.wait_condition(lambda: self._done_presented)
        await workflow.wait_condition(workflow.all_handlers_finished)
        return StreamOutcome(
            generation_state=self._generation_state,
            cursor=self._cursor,
            sealed=self._seal_ordinal,
            payloads_delivered=sum(
                1 for o in self._obligations.values() if o.state == PRESENTED
            ),
        )

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

        A resume reads the store before it swaps. Every reference a committed presentation
        carried was verified when that presentation committed, and a resume is the moment to
        ask again, because a history citing bytes the store can no longer produce is not one to
        hand a new owner. The read covers what the writer being replaced commits while it is
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
        if claim.reason == "resume":
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
                1 for kind in self._presented.values() if kind == PAYLOAD
            ),
            wait_count=self._wait_count,
            wait_reasons=self._wait_reason_counts(),
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
        try:
            identity = terminal_request_identity(
                metadata,
                request.public_tool_name,
                request.native_terminal_name,
                request.native_arguments,
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
        if attempt.state in (SEALED, ACK_PRESENTED):
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
        metadata = request.metadata
        attempt.state = SEALING
        attempt.terminal_request_id = metadata.request_id
        attempt.terminal_identity = identity
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
        graded = await workflow.execute_activity(
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
        # Each result is checked where it arrives, before the next Activity is asked for. The
        # seal ID and the digest are what make a result this filing's: the public attempt ID
        # and the payload position survive a fork, so a result that echoes only those says
        # which obligation asked and not which filing was answered.
        if graded.attempt_id != attempt.item.attempt_id or graded.seal_id != attempt.seal_id:
            raise _unusable("the score is not this seal's")
        # An attempt with no obligation has nothing to build, so nothing is built: neither the
        # generation under Never nor the one position a roster gave no payload asks a renderer
        # for a candidate. An absent outbox row is not an outbox row nobody reads.
        candidate: Optional[PayloadCandidate] = None
        if attempt.item.attempt_id in self._obligations:
            bundle = await workflow.execute_activity(
                generate_payload_bundle_activity,
                GeneratePayloadBundleInput(
                    attempt_id=attempt.item.attempt_id,
                    payload_position=attempt.item.payload_position,
                    payload_message_id=attempt.item.payload_message_id,
                    submission_digest=attempt.submission_digest,
                    canonical_submission_text=sealed.canonical_submission_text,
                ),
                start_to_close_timeout=_ACTIVITY_TIMEOUT,
                retry_policy=_ACTIVITY_RETRY,
            )
            if len(bundle.candidates) != 1:
                raise ApplicationError(
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
        # Everything above was awaited, so the owner is checked again before any of it is made
        # authoritative. A seal that was in flight when a resume fenced its writer commits
        # nothing: the attempt stays prepared, and the new owner's exact retry continues it.
        self._require_writer(writer)
        # One transition, no await inside it: the score, the bundle, the released capacity,
        # the obligation, and the acknowledgement all become authoritative together or not
        # at all. The bytes go out after this, which is what puts the seal before the Ack.
        attempt.score = graded.score
        attempt.decode_state = graded.decode_state
        # The score and what it was taken from become authoritative together. A committed score
        # whose evidence the generation kept no name for is a headline with nothing under it, so
        # the reference is kept on the attempt and counted among the objects this history cites:
        # a claim reads the store for all of them again before it may continue the generation.
        attempt.graded_evidence = graded.evidence.sha256
        if graded.evidence.sha256 not in self._committed_blobs:
            self._committed_blobs.append(graded.evidence.sha256)
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
        self._apply_presentation(pending.message.kind, pending.message.attempt_id, commit)
        self._presented[commit.message_id] = pending.message.kind
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
            "obligations": {key: value.state for key, value in self._obligations.items()},
            "presented": list(self._presented),
            "materializations": self._materialized(),
            "eligibilities": len(self._eligibilities),
            "offers": len(self._offers),
            "seal_ordinal": self._seal_ordinal,
        }
        return sha256(canonical_json(projection)).hexdigest()


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
    if len(start.id_key_hex) != 64 or not set(start.id_key_hex) <= set("0123456789abcdef"):
        raise StreamProtocolError("invalid_message")
    identifiers = [start.initial_cursor, start.done_message_id]
    try:
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


def _unusable(what: str) -> ApplicationError:
    """A failure for an Activity result the seal cannot vouch for.

    It is not a refusal: nothing the caller sent is wrong, so it carries no protocol code and
    reaches no transcript. It is raised before the transition that would make an
    acknowledgement authoritative, which is the last place such a result can still be caught.
    """
    return ApplicationError(what, type="UnusableActivityResult", non_retryable=True)


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
