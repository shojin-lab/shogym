"""The closed set of records protocol v2 puts on the wire.

A record is a frozen dataclass whose constructor validates every field, so an instance that
exists is already well formed and can be serialized without a second check. :meth:`from_wire`
is the decoder's half of the same rule: the payload must be a mapping whose key set is exactly
the record's, less any name in :data:`_OMISSIBLE` it leaves out, and every value must already
have the type and the range its field requires. No value that is there is coerced, and none is
dropped, so a payload that means something slightly different from what this protocol declares
is refused instead of quietly repaired. The one thing that is defaulted is an omissible name a
payload does not carry: it takes its field's default and is left out again on the way back, so
the absence survives the round trip rather than becoming a value.

Two conventions carry the validation. A field name has one meaning across every record here,
so its rule hangs off the name in :data:`_CHECKS`. And a field declared with a string default
is fixed on the wire, which is what makes ``kind`` a discriminator rather than a suggestion.

One key is exempt from the exact key set, and it is named in :data:`_OMISSIBLE` rather than
inferred from a type: a generation that declares no step budget serves the task record it
served before there was one to declare, byte for byte, so the key is left out rather than sent
empty.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Type, TypeVar, Union, cast

from shogym.serve.protocol_v2 import jcs
from shogym.serve.protocol_v2.errors import WireFormatError

PROTOCOL_VERSION = 2

_OPAQUE_ID = re.compile(r"[0-9a-f]{32}")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
_CANONICALIZATION_VERSION = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_RETRY_AFTER_MS_MAX = 4294967295
# The largest integer canonical JSON will write. A record refuses a number above it here rather
# than serializing well and failing when its bytes are taken.
_SAFE_INTEGER_MAX = 9007199254740991

# The complete code set. A code outside it is not a protocol error of an unknown sort, it is a
# malformed record.
PROTOCOL_ERROR_CODES: Tuple[str, ...] = (
    "unsupported_version",
    "invalid_message",
    "consumer_conflict",
    "overlapping_call",
    "invalid_cursor",
    "request_conflict",
    "outstanding_response",
    "already_presented",
    "invalid_attempt",
    "conflicting_seal",
    "frozen_stream",
    "failed_stream",
    "closed_stream",
    "fenced_writer",
    "configuration_mismatch",
)


# Who filed the terminal that ended an attempt. An agent's own call is one fact and a filing
# made on its behalf when its world work ran out is another, and a reader counting endings has
# to be able to tell them apart, so the two are named here rather than inferred from a step
# count somebody kept.
AGENT_FILED = "agent"
HORIZON_FILED = "horizon"
TERMINAL_SOURCES: Tuple[str, ...] = (AGENT_FILED, HORIZON_FILED)

# What reaching the environment's step budget does to an attempt. ``floor`` is the rule for an
# environment that says nothing: the world work is over, nothing was filed, and the attempt ends
# at the floor with the reason on it. ``graded`` is what an environment declares when its
# horizon is an ending its own scorer answers for: the world as it stands is the filing, and the
# attempt is sealed and graded there rather than floored.
FLOOR_HORIZON = "floor"
GRADED_HORIZON = "graded"
HORIZON_ENDINGS: Tuple[str, ...] = (FLOOR_HORIZON, GRADED_HORIZON)


def _version(name: str, value: Any) -> None:
    if type(value) is not int or value != PROTOCOL_VERSION:
        raise WireFormatError(f"{name} must be {PROTOCOL_VERSION}")


def _opaque_id(name: str, value: Any) -> None:
    if not isinstance(value, str) or _OPAQUE_ID.fullmatch(value) is None:
        raise WireFormatError(f"{name} must be 32 lower-case hexadecimal characters")


def _sha256_hex(name: str, value: Any) -> None:
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise WireFormatError(f"{name} must be 64 lower-case hexadecimal characters")


def _optional_sha256_hex(name: str, value: Any) -> None:
    if value is not None:
        _sha256_hex(name, value)


def _scalar_string(name: str, value: Any) -> None:
    if not isinstance(value, str):
        raise WireFormatError(f"{name} must be a string")
    for character in value:
        if 0xD800 <= ord(character) <= 0xDFFF:
            raise WireFormatError(f"{name} must be Unicode scalar values, and this one is not")


def _retry_after_ms(name: str, value: Any) -> None:
    if type(value) is not int or not 0 <= value <= _RETRY_AFTER_MS_MAX:
        raise WireFormatError(f"{name} must be an integer from 0 through {_RETRY_AFTER_MS_MAX}")


def _step_budget(name: str, value: Any) -> None:
    if type(value) is not int or not 1 <= value <= _SAFE_INTEGER_MAX:
        raise WireFormatError(f"{name} must be an integer from 1 through {_SAFE_INTEGER_MAX}")


def _canonicalization_version(name: str, value: Any) -> None:
    if not isinstance(value, str) or _CANONICALIZATION_VERSION.fullmatch(value) is None:
        raise WireFormatError(f"{name} must match [a-z0-9][a-z0-9._-]{{0,63}}")


def _boolean(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise WireFormatError(f"{name} must be true or false")


def _protocol_error_code(name: str, value: Any) -> None:
    if not isinstance(value, str) or value not in PROTOCOL_ERROR_CODES:
        raise WireFormatError(f"{name} must be one of the {len(PROTOCOL_ERROR_CODES)} known codes")


_CHECKS: Dict[str, Callable[[str, Any], None]] = {
    "protocol_version": _version,
    "request_id": _opaque_id,
    "last_presented_cursor": _opaque_id,
    "message_id": _opaque_id,
    "attempt_id": _opaque_id,
    "attestation_id": _opaque_id,
    "cursor_before": _opaque_id,
    "cursor": _opaque_id,
    "body": _scalar_string,
    "budget": _step_budget,
    "retry_after_ms": _retry_after_ms,
    "code": _protocol_error_code,
    "canonicalization_version": _canonicalization_version,
    "submission_digest": _sha256_hex,
    "visible_bytes_sha256": _sha256_hex,
    "transcript_blob": _sha256_hex,
    "stream_state_before_sha256": _sha256_hex,
    "stream_state_sha256": _sha256_hex,
    "provider_turn_blob": _optional_sha256_hex,
    "task_start_checkpoint_blob": _optional_sha256_hex,
    "completed_turn": _boolean,
}


# The names a record may leave out, and the whole of them. ``budget`` is here because the number
# of environment actions an attempt gets is a thing a generation may not have declared, and one
# that declared none has to serve the bytes it served before the key existed. Absence is the
# omission of the key: a name here is never sent with an empty value.
_OMISSIBLE = frozenset({"budget"})


def _fields(record: Any) -> Tuple[Any, ...]:
    return dataclasses.fields(record)


def _validate(record: Any) -> None:
    for field in _fields(record):
        value = getattr(record, field.name)
        if isinstance(field.default, str):
            if value != field.default:
                raise WireFormatError(f"{field.name} must be {field.default!r}")
        elif value is None and field.name in _OMISSIBLE:
            continue
        else:
            _CHECKS[field.name](field.name, value)


def _decode(cls: Any, payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        raise WireFormatError(f"a {cls.__name__} payload must be a JSON object")
    names = tuple(field.name for field in _fields(cls))
    unknown = sorted(repr(key) for key in payload if key not in names)
    if unknown:
        raise WireFormatError(f"a {cls.__name__} payload has no field {', '.join(unknown)}")
    missing = [name for name in names if name not in payload and name not in _OMISSIBLE]
    if missing:
        raise WireFormatError(f"a {cls.__name__} payload is missing {', '.join(missing)}")
    empty = [
        name for name in names if name in _OMISSIBLE and name in payload and payload[name] is None
    ]
    if empty:
        raise WireFormatError(
            f"a {cls.__name__} payload leaves out {', '.join(empty)} rather than sending it empty"
        )
    return cls(**{name: payload[name] for name in names if name in payload})


_R = TypeVar("_R", bound="_Record")


class _Record:
    """Validation, decoding, and the JSON object form, shared by every record below."""

    def __post_init__(self) -> None:
        _validate(self)

    @classmethod
    def from_wire(cls: Type[_R], payload: Any) -> _R:
        """Return the record ``payload`` denotes, or raise :class:`WireFormatError`."""
        return cast(_R, _decode(cls, payload))

    def to_wire(self) -> Dict[str, Any]:
        """Return this record as a JSON object, field for field, less any omissible name this
        record does not carry."""
        return {
            field.name: getattr(self, field.name)
            for field in _fields(self)
            if getattr(self, field.name) is not None or field.name not in _OMISSIBLE
        }


@dataclass(frozen=True)
class PullRequest(_Record):
    """The envelope the harness puts around the model's ``pull()``.

    The model's own call carries no arguments, so nothing here is model-authored. The request
    ID and the cursor are the harness's, and they are what let the server tell a retry of one
    request from a second request.
    """

    request_id: str
    last_presented_cursor: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class Task(_Record):
    """A work order, and the whole of one.

    ``body`` is what the renderer produced and nothing else. There is no field a queue
    position, a target, a cell, a regime, a branch, or a lease could be written into, so the
    redaction is structural rather than a rule someone has to keep applying.

    ``budget`` is how many environment actions the attempt gets, and the record carries it only
    where the generation declared one. It is that generation's single number, identical on every
    task it serves and drawn from what the environment publishes rather than from anything about
    this attempt, so it tells an agent what the work in its hands is allowed to spend and nothing
    about where that work sits in a queue.
    """

    message_id: str
    attempt_id: str
    body: str
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "task"
    budget: Optional[int] = None


@dataclass(frozen=True)
class Payload(_Record):
    """Content delivered against an attempt, in the fixed envelope its renderer produced.

    It is associated with an attempt. It does not claim to evaluate or answer one, and this
    layer reads nothing inside ``body``.
    """

    message_id: str
    attempt_id: str
    body: str
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "payload"


@dataclass(frozen=True)
class Wait(_Record):
    """Nothing to hand out yet.

    ``retry_after_ms`` is advisory and carries no reason: a Wait for a closed timer and a Wait
    for full capacity are the same record, so nothing about the schedule leaks through the one
    message an agent can provoke at will.
    """

    message_id: str
    retry_after_ms: int
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "wait"


@dataclass(frozen=True)
class Done(_Record):
    """Nothing will come again."""

    message_id: str
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "done"


@dataclass(frozen=True)
class SealAck(_Record):
    """A filing was accepted, said without saying anything about how good it was.

    The digest commits to the canonical submission, so the agent can tell that what it filed
    is what was recorded, and can learn nothing else from it.
    """

    message_id: str
    attempt_id: str
    submission_digest: str
    canonicalization_version: str
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "seal_ack"


@dataclass(frozen=True)
class SealReject(_Record):
    """A terminal call that did not match its native schema, refused without touching the
    attempt.

    ``body`` explains the mismatch from the public schema and the submitted arguments alone.
    It is not a verdict, and the attempt is exactly where it was.
    """

    message_id: str
    attempt_id: str
    body: str
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "seal_reject"
    code: str = "invalid_arguments"


@dataclass(frozen=True)
class ProtocolError(_Record):
    """A harness-level refusal, carrying a code from the closed set and nothing more.

    It has no message ID because it is never a message: it travels in the transport's error
    data, advances no state, and is never inserted into a model transcript.
    """

    code: str
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "protocol_error"


@dataclass(frozen=True)
class TerminalMetadata(_Record):
    """The harness-only envelope beside a terminal call's native arguments.

    It names the attempt a second time, next to the attempt ID in the model-visible wrapper,
    so the two can be compared before the request is looked up.
    """

    request_id: str
    last_presented_cursor: str
    attempt_id: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class PresentationCommit(_Record):
    """The harness's claim that exact bytes were delivered, offered for verification.

    The claim is about delivery to the transport that carries the bytes. What the model
    consumed is attested by the harness transcript and not by this record. Every hash here is
    checked against something the authority already holds, so the claim is verifiable rather
    than trusted. A blob that does not apply to this presentation is null, and null is a value
    the record requires rather than a field it may omit.
    """

    attestation_id: str
    cursor_before: str
    message_id: str
    visible_bytes_sha256: str
    transcript_blob: str
    provider_turn_blob: Optional[str]
    task_start_checkpoint_blob: Optional[str]
    completed_turn: bool
    stream_state_before_sha256: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class PresentationAck(_Record):
    """What the authority returns once a presentation has committed: the advanced cursor and
    the stream hash it advanced to."""

    attestation_id: str
    cursor: str
    stream_state_sha256: str
    protocol_version: int = PROTOCOL_VERSION
    kind: str = "presentation_ack"


PullResult = Union[Task, Payload, Wait, Done]
TerminalResult = Union[SealAck, SealReject]
Record = Union[
    PullRequest,
    TerminalMetadata,
    PresentationCommit,
    PresentationAck,
    ProtocolError,
    PullResult,
    TerminalResult,
]

_PULL_RESULTS: Dict[str, Type[_Record]] = {
    "task": Task,
    "payload": Payload,
    "wait": Wait,
    "done": Done,
}
_TERMINAL_RESULTS: Dict[str, Type[_Record]] = {
    "seal_ack": SealAck,
    "seal_reject": SealReject,
}
_PRESENTABLE = (Task, Payload, Wait, Done, SealAck, SealReject)


def _decode_union(table: Dict[str, Type[_Record]], label: str, payload: Any) -> Any:
    if not isinstance(payload, Mapping):
        raise WireFormatError(f"{label} must be a JSON object")
    kind = payload.get("kind")
    cls = table.get(kind) if isinstance(kind, str) else None
    if cls is None:
        raise WireFormatError(f"{label} has kind {kind!r}, not one of {sorted(table)}")
    return cls.from_wire(payload)


def decode_pull_result(payload: Any) -> PullResult:
    """Return the pull result ``payload`` denotes. A record from another union is refused
    here even when it is a perfectly good record of its own kind."""
    return cast(PullResult, _decode_union(_PULL_RESULTS, "a pull result", payload))


def decode_terminal_result(payload: Any) -> TerminalResult:
    """Return the terminal result ``payload`` denotes, on the same terms."""
    return cast(TerminalResult, _decode_union(_TERMINAL_RESULTS, "a terminal result", payload))


def canonical_bytes(record: Record) -> bytes:
    """Return the canonical JSON bytes of any record in this module."""
    if not isinstance(record, _Record):
        raise WireFormatError(f"{type(record).__name__} is not a protocol record")
    return jcs.encode(record.to_wire())


def visible_bytes(result: Union[PullResult, TerminalResult]) -> bytes:
    """Return the complete bytes a pull or terminal result commits to the model transcript.

    These are the bytes the presentation hash covers, so this is the function every later
    layer hashes rather than re-deriving the serialization for itself.
    """
    if not isinstance(result, _PRESENTABLE):
        raise WireFormatError(f"a {type(result).__name__} is never presented to the model")
    return canonical_bytes(result)


def mcp_text_content(result: Union[PullResult, TerminalResult]) -> Tuple[Dict[str, str], ...]:
    """Return the content items that carry ``result`` to the model: exactly one text item,
    whose text is :func:`visible_bytes`, and no structured content beside it."""
    return ({"type": "text", "text": visible_bytes(result).decode("utf-8")},)


def require_opaque_id(name: str, value: Any) -> None:
    """Raise unless ``value`` is an opaque ID, for callers outside a record constructor."""
    _opaque_id(name, value)


def require_step_budget(name: str, value: Any) -> None:
    """Raise unless ``value`` is a step budget, for callers outside a record constructor.

    A generation declares the number long before a task record carries it, and the layers in
    between compose, hash and start on it. They hold it to this rule rather than to one of their
    own, so a number a record would refuse is refused where it is declared instead of surviving
    as far as the first offer.
    """
    _step_budget(name, value)
