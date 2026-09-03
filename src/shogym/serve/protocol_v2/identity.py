"""Request identities, the submission digest, and the message ID stream.

Each identity is a SHA-256 over length-prefixed fields, so no field can borrow bytes from its
neighbour and two different tuples of inputs cannot collide by rearranging the same characters.
Each starts with its own domain tag, so a preimage built for one purpose is not a preimage for
another. The message ID stream is keyed instead of hashed openly, because an agent that could
recompute the next ID would learn how many messages are still coming.
"""

from __future__ import annotations

import hmac
from hashlib import sha256
from typing import Any, Mapping

from shogym.serve.protocol_v2 import jcs
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.records import (
    PresentationCommit,
    PullRequest,
    TerminalMetadata,
    canonical_bytes,
    require_opaque_id,
)

_UINT64_MAX = 2**64 - 1
_ID_KEY_BYTES = 32
_MESSAGE_ID_BYTES = 16


def length_prefixed(value: bytes) -> bytes:
    """Return ``value`` behind its unsigned 64-bit big-endian byte length."""
    return len(value).to_bytes(8, "big") + value


def _utf8(name: str, text: Any) -> bytes:
    if not isinstance(text, str):
        raise WireFormatError(f"{name} must be a string")
    try:
        return text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise WireFormatError(f"{name} must be Unicode scalar values, and this one is not") from exc


def pull_request_identity(request: PullRequest) -> str:
    """Return the canonical identity of a pull request.

    A retry that carries the same request ID and the same identity replays its result. A retry
    that carries the same ID and a different identity is a conflict, which is the whole point
    of hashing the envelope rather than trusting the ID alone.
    """
    return sha256(
        length_prefixed(b"pull-request-v2") + length_prefixed(canonical_bytes(request))
    ).hexdigest()


def terminal_request_identity(
    metadata: TerminalMetadata,
    public_tool_name: str,
    native_terminal_name: str,
    native_arguments: Mapping[str, Any],
) -> str:
    """Return the canonical identity of a terminal request.

    Both tool names are covered, so the same arguments filed through a different tool are a
    different request rather than a replay of the first.
    """
    return sha256(
        length_prefixed(b"terminal-request-v2")
        + length_prefixed(canonical_bytes(metadata))
        + length_prefixed(_utf8("public_tool_name", public_tool_name))
        + length_prefixed(_utf8("native_terminal_name", native_terminal_name))
        + length_prefixed(jcs.encode(native_arguments))
    ).hexdigest()


def presentation_request_identity(commit: PresentationCommit) -> str:
    """Return the canonical identity of a presentation commit."""
    return sha256(
        length_prefixed(b"presentation-v2") + length_prefixed(canonical_bytes(commit))
    ).hexdigest()


def submission_digest(
    attempt_id: str, native_terminal_name: str, canonical_submission: bytes
) -> str:
    """Return the digest that commits to one filing.

    The inputs are the attempt, the tool that filed, and the canonical submission bytes. A
    verdict, a score, a hidden rule, a cell, a branch, a schedule, a failure class, a timing,
    payload bytes, and delivery state are all absent, so the digest cannot move when any of
    them moves, and an agent cannot read a verdict off it.
    """
    require_opaque_id("attempt_id", attempt_id)
    if not isinstance(canonical_submission, (bytes, bytearray)):
        raise WireFormatError("canonical_submission must be bytes")
    return sha256(
        length_prefixed(b"task-stream-submission-v2")
        + length_prefixed(_utf8("attempt_id", attempt_id))
        + length_prefixed(_utf8("native_terminal_name", native_terminal_name))
        + length_prefixed(bytes(canonical_submission))
    ).hexdigest()


def stream_message_id(id_key: bytes, hidden_ordinal: int) -> str:
    """Return the message ID at ``hidden_ordinal`` in the branch-neutral stream.

    The key is branch neutral, so two children of one fork mint identical IDs at identical
    ordinals and a transcript comparison across branches is a comparison of content. The
    ordinal is hidden: it is the position in the stream, never a queue position, and nothing
    about it is recoverable from the ID without the key.
    """
    if not isinstance(id_key, (bytes, bytearray)) or len(id_key) != _ID_KEY_BYTES:
        raise WireFormatError(f"the id key must be {_ID_KEY_BYTES} bytes")
    if type(hidden_ordinal) is not int or not 0 <= hidden_ordinal <= _UINT64_MAX:
        raise WireFormatError("the hidden ordinal must be an unsigned 64-bit integer")
    message = length_prefixed(b"message-v2") + hidden_ordinal.to_bytes(8, "big")
    return hmac.new(bytes(id_key), message, sha256).digest()[:_MESSAGE_ID_BYTES].hex()
