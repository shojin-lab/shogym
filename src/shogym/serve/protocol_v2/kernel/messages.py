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
from typing import Any, Dict, List, Optional

from shogym.serve.protocol_v2 import PROTOCOL_VERSION, TerminalMetadata, length_prefixed


@dataclass(frozen=True)
class BlobRef:
    """A reference to an immutable content-addressed object, by the hash that names it."""

    sha256: str
    size: int
    media_type: str


def blob_ref(text: str, media_type: str = "text/plain") -> BlobRef:
    """Return the reference that names ``text``'s bytes."""
    encoded = text.encode("utf-8")
    return BlobRef(sha256=sha256(encoded).hexdigest(), size=len(encoded), media_type=media_type)


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
class QueueClosed:
    """The controller's receipt for a closed queue."""

    task_count: int
    closed: bool = True
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class StreamState:
    """What the harness may ask a live generation about itself.

    This is the answer to a Query, so it writes nothing and can be asked at any time. It says
    nothing a model could use: no queue contents, no reason for a Wait, no score.
    """

    generation_state: str
    cursor: str
    configuration_hash: str
    stream_state_sha256: str
    consumer_id: Optional[str]
    queue_closed: bool
    tasks_remaining: int
    capacity: int
    capacity_in_use: int
    pending_message_id: Optional[str]
    pending_kind: Optional[str]
    attempts: Dict[str, str]
    obligations: Dict[str, str]
    offer_count: int
    presentation_count: int
    wait_count: int
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
    """Ask the environment for its canonical pre-verdict submission and its local seal."""

    attempt_id: str
    seal_id: str
    native_terminal_name: str
    canonicalization_version: str
    native_arguments: Dict[str, Any] = field(default_factory=dict)
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
    """Grade sealed evidence, out of process, against a world nobody can still change."""

    attempt_id: str
    seal_id: str
    submission_digest: str
    canonical_submission_text: str
    environment_recovery_token: str
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
    score: int
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
