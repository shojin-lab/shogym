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
from typing import Any, Dict, List, Optional

from temporalio import workflow
from temporalio.common import RetryPolicy
from temporalio.exceptions import ApplicationError

with workflow.unsafe.imports_passed_through():
    from shogym.serve.protocol_v2 import (
        Done,
        Payload,
        PresentationAck,
        PresentationCommit,
        ProtocolError,
        PullRequest,
        SealAck,
        SealReject,
        Task,
        Wait,
        WireFormatError,
        canonical_bytes,
        canonical_json,
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
    )
    from shogym.serve.protocol_v2.kernel.messages import (
        ConsumerClaim,
        ConsumerReceipt,
        GeneratePayloadBundleInput,
        GradeAttemptInput,
        OfferedMessage,
        PayloadCandidate,
        QueueClosed,
        SealAttemptInput,
        SealRequest,
        StreamOutcome,
        StreamStart,
        StreamState,
        TaskItem,
        hidden_seal_id,
    )

PLANNED = "planned"
TASK_OFFERED = "task_offered"
ACTIVE = "active"
SEALING = "sealing"
SEALED = "sealed"
ACK_PRESENTED = "ack_presented"

ASSIGNED = "assigned"
ELIGIBLE = "eligible"
OFFERED = "offered"
PRESENTED = "presented"

OPEN = "open"
DONE = "done"

_ACTIVITY_TIMEOUT = timedelta(seconds=60)
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
    terminal_request_id: Optional[str] = None
    terminal_identity: Optional[str] = None
    seal_id: Optional[str] = None
    submission_digest: Optional[str] = None
    canonical_submission_text: Optional[str] = None
    environment_recovery_token: Optional[str] = None
    finalizer_key: Optional[str] = None
    score: Optional[int] = None
    decode_state: Optional[str] = None
    seal_ordinal: Optional[int] = None


@dataclass
class _Obligation:
    item: TaskItem
    state: str = ASSIGNED
    materialized: bool = False
    candidate: Optional[PayloadCandidate] = None


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
        self._id_key = bytes.fromhex(start.id_key_hex)
        self._generation_state = OPEN
        self._consumer_id: Optional[str] = None
        self._claim_epoch = 0
        self._cursor = start.initial_cursor
        self._queue_closed = False
        self._next_task = 0
        self._attempts: Dict[str, _Attempt] = {
            item.attempt_id: _Attempt(item=item) for item in start.tasks
        }
        self._obligations: Dict[str, _Obligation] = {
            item.attempt_id: _Obligation(item=item) for item in start.tasks
        }
        self._pending: Optional[_Pending] = None
        self._pull_requests: Dict[str, _Bound] = {}
        self._terminal_requests: Dict[str, _Bound] = {}
        self._attestations: Dict[str, PresentationAck] = {}
        self._attestation_identities: Dict[str, str] = {}
        self._presented: List[str] = []
        self._issued_ids = {start.initial_cursor, start.done_message_id}
        for item in start.tasks:
            self._issued_ids.update({item.task_message_id, item.ack_message_id})
            self._issued_ids.add(item.payload_message_id)
        self._hidden_ordinal = 0
        self._offers: List[_Offer] = []
        self._seal_ordinal = 0
        self._wait_count = 0
        self._operation_in_flight = False
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
    async def claim_consumer(self, claim: ConsumerClaim) -> ConsumerReceipt:
        """Bind the generation's one logical consumer.

        The same claim presented twice returns the same receipt, because a lost response must
        not cost a caller its stream. A different one is refused, before any message is
        offered and without touching state.
        """
        if self._generation_state != OPEN:
            raise StreamProtocolError("closed_stream")
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
            configuration_hash=self._start.configuration_hash,
        )

    @workflow.update
    async def pull(self, request: PullRequest) -> OfferedMessage:
        """Return the one message this request is entitled to.

        A retry of the same request gets its own result back while that result is unpresented,
        and an error afterwards. A new request gets a new selection, but only from a current
        cursor and only when nothing is outstanding: a second request never inherits the first
        one's offer.
        """
        self._take_lock()
        try:
            return self._pull(request)
        finally:
            self._operation_in_flight = False

    @workflow.update
    async def seal_attempt(self, request: SealRequest) -> OfferedMessage:
        """End an attempt, or say why the filing was not one this tool accepts.

        Nothing about the seal becomes authoritative until the last transition below, which
        contains no await. The acknowledgement is built there and returned after it, so a
        crash anywhere earlier leaves a stream that has not acknowledged anything.
        """
        self._take_lock()
        try:
            return await self._seal(request)
        finally:
            self._operation_in_flight = False

    @workflow.update
    async def commit_presentation(self, commit: PresentationCommit) -> PresentationAck:
        """Accept the harness's attestation that the exact offered bytes were delivered.

        Delivery is what the attestation says: the bytes were handed to the transport that
        carries them. What the model consumed is attested by the harness transcript and not
        here. Everything in the attestation is checked against something already held here, so
        this is a verification and not a report. The cursor advances only on the way out.
        """
        self._take_lock()
        try:
            return self._commit_presentation(commit)
        finally:
            self._operation_in_flight = False

    @workflow.update
    async def close_queue(self) -> QueueClosed:
        """Close the queue to insertion. It revokes nothing and seals nothing."""
        self._take_lock()
        try:
            self._queue_closed = True
            return QueueClosed(task_count=len(self._start.tasks))
        finally:
            self._operation_in_flight = False

    @workflow.query
    def stream_state(self) -> StreamState:
        """Report the generation's state to the harness. Queries write nothing."""
        return StreamState(
            generation_state=self._generation_state,
            cursor=self._cursor,
            configuration_hash=self._start.configuration_hash,
            stream_state_sha256=self._projection_hash(),
            consumer_id=self._consumer_id,
            queue_closed=self._queue_closed,
            tasks_remaining=len(self._start.tasks) - self._next_task,
            capacity=self._start.capacity,
            capacity_in_use=self._capacity_in_use(),
            pending_message_id=None if self._pending is None else self._pending.message.message_id,
            pending_kind=None if self._pending is None else self._pending.message.kind,
            attempts={key: value.state for key, value in self._attempts.items()},
            obligations={key: value.state for key, value in self._obligations.items()},
            offer_count=len(self._offers),
            presentation_count=len(self._presented),
            wait_count=self._wait_count,
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
        obligation = self._eligible_obligation()
        if obligation is not None:
            candidate = obligation.candidate
            assert candidate is not None
            obligation.state = OFFERED
            result = Payload(
                message_id=obligation.item.payload_message_id,
                attempt_id=obligation.item.attempt_id,
                body=candidate.body,
            )
            return self._offer(result, "pull", request_id, identity, obligation.item.attempt_id)
        item = self._eligible_task()
        if item is not None:
            self._next_task += 1
            self._attempts[item.attempt_id].state = TASK_OFFERED
            result = Task(
                message_id=item.task_message_id,
                attempt_id=item.attempt_id,
                body=item.body,
            )
            return self._offer(result, "pull", request_id, identity, item.attempt_id)
        if self._done_eligible():
            done = Done(message_id=self._start.done_message_id)
            return self._offer(done, "pull", request_id, identity, None)
        self._wait_count += 1
        wait = Wait(
            message_id=self._mint_message_id(),
            retry_after_ms=self._start.wait_retry_after_ms,
        )
        return self._offer(wait, "pull", request_id, identity, None, self._wait_reason())

    def _eligible_obligation(self) -> Optional[_Obligation]:
        """The eligible payload at the lowest declared position. Payloads outrank tasks."""
        eligible = [o for o in self._obligations.values() if o.state == ELIGIBLE]
        if not eligible:
            return None
        return min(eligible, key=lambda o: (o.item.payload_position, o.item.attempt_id))

    def _eligible_task(self) -> Optional[TaskItem]:
        """The next task, if the queue has one and capacity is free to reserve."""
        if self._capacity_in_use() >= self._start.capacity:
            return None
        if self._next_task >= len(self._start.tasks):
            return None
        return self._start.tasks[self._next_task]

    def _done_eligible(self) -> bool:
        """Done is monotonic and late: a live attempt or an unfulfilled payload keeps it away.

        A caller reaches this only with nothing outstanding, which is the third condition.
        """
        if not self._queue_closed:
            return False
        if any(attempt.state != ACK_PRESENTED for attempt in self._attempts.values()):
            return False
        return all(o.state == PRESENTED for o in self._obligations.values())

    def _wait_reason(self) -> str:
        """The hidden reason for a Wait. It is recorded, and it is never on the wire."""
        if self._capacity_in_use() >= self._start.capacity:
            return "capacity"
        if not self._queue_closed:
            return "queue_open"
        return "obligation_pending"

    def _capacity_in_use(self) -> int:
        occupied = (TASK_OFFERED, ACTIVE, SEALING)
        return sum(1 for attempt in self._attempts.values() if attempt.state in occupied)

    # Seal.

    async def _seal(self, request: SealRequest) -> OfferedMessage:
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
        if attempt.state in (SEALING, SEALED, ACK_PRESENTED):
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
        return await self._seal_accepted(request, attempt, identity)

    async def _seal_accepted(
        self, request: SealRequest, attempt: _Attempt, identity: str
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
            ),
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
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
            ),
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
            retry_policy=_ACTIVITY_RETRY,
        )
        # Each result is checked where it arrives, before the next Activity is asked for. The
        # seal ID and the digest are what make a result this filing's: the public attempt ID
        # and the payload position survive a fork, so a result that echoes only those says
        # which obligation asked and not which filing was answered.
        if graded.attempt_id != attempt.item.attempt_id or graded.seal_id != attempt.seal_id:
            raise _unusable("the score is not this seal's")
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
        # One transition, no await inside it: the score, the bundle, the released capacity,
        # the obligation, and the acknowledgement all become authoritative together or not
        # at all. The bytes go out after this, which is what puts the seal before the Ack.
        attempt.score = graded.score
        attempt.decode_state = graded.decode_state
        self._seal_ordinal += 1
        attempt.seal_ordinal = self._seal_ordinal
        attempt.state = SEALED
        obligation = self._obligations[attempt.item.attempt_id]
        obligation.candidate = candidate
        obligation.materialized = True
        obligation.state = ELIGIBLE
        ack = SealAck(
            message_id=attempt.item.ack_message_id,
            attempt_id=attempt.item.attempt_id,
            submission_digest=attempt.submission_digest,
            canonicalization_version=self._start.canonicalization_version,
        )
        return self._offer(
            ack, "terminal", metadata.request_id, identity, attempt.item.attempt_id
        )

    # Presentation.

    def _commit_presentation(self, commit: PresentationCommit) -> PresentationAck:
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
        self._apply_presentation(pending.message.kind, pending.message.attempt_id)
        self._presented.append(commit.message_id)
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

    def _apply_presentation(self, kind: str, attempt_id: Optional[str]) -> None:
        if kind == "task" and attempt_id is not None:
            self._attempts[attempt_id].state = ACTIVE
        elif kind == "seal_ack" and attempt_id is not None:
            self._attempts[attempt_id].state = ACK_PRESENTED
        elif kind == "payload" and attempt_id is not None:
            self._obligations[attempt_id].state = PRESENTED
        elif kind == "done":
            self._generation_state = DONE
            self._done_presented = True
            self._draining = True

    # Shared machinery.

    def _take_lock(self) -> None:
        """Refuse a call the generation cannot accept, then hold the stream against overlap.

        This runs before the first await in every stream-affecting handler. Waiting on a lock
        would queue an overlapping call; the protocol rejects it instead.
        """
        if self._generation_state != OPEN or self._draining:
            raise StreamProtocolError("closed_stream")
        if self._consumer_id is None:
            raise StreamProtocolError("consumer_conflict")
        if self._operation_in_flight:
            raise StreamProtocolError("overlapping_call")
        self._operation_in_flight = True

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
            "configuration_hash": self._start.configuration_hash,
            "consumer_id": self._consumer_id,
            "claim_epoch": self._claim_epoch,
            "queue_closed": self._queue_closed,
            "next_task": self._next_task,
            "capacity_in_use": self._capacity_in_use(),
            "pending_message_id": (
                None if self._pending is None else self._pending.message.message_id
            ),
            "attempts": {key: value.state for key, value in self._attempts.items()},
            "obligations": {key: value.state for key, value in self._obligations.items()},
            "presented": list(self._presented),
            "offers": len(self._offers),
            "seal_ordinal": self._seal_ordinal,
        }
        return sha256(canonical_json(projection)).hexdigest()


def _check_start(start: StreamStart) -> None:
    """Refuse a generation this code cannot serve, before it serves anything.

    A malformed manifest has to fail the workflow rather than an Update, because there is no
    caller yet to refuse and nothing here can be repaired later.
    """
    if start.protocol_version != 2:
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
