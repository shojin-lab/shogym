"""Domain-separated randomness, and the opaque identifiers the agent sees.

Every draw this environment makes comes from a keyed stream with an explicit
domain label. There is no ambient PRNG anywhere: no `random.seed`, no module-level
generator, no reuse of one seed for two purposes. Two streams under different
labels are independent given the master key, so knowing task A's surface tells you
nothing about task B's, about the filler, or about the drawn convention.

The master key is controller-side and never leaves it. It is what makes the task
identifiers opaque: an identifier is an HMAC over the label and the coordinates,
so it encodes no seed, no family index and no draw ordinal, and it cannot be
inverted without the key. An agent that collects every identifier it has ever been
served learns the number of them and nothing else.

The labels are declared here rather than passed as strings at each call site, so
the domain separation is a fact about this module and not a convention someone has
to remember.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import random
from typing import Iterable

#: The declared streams. Adding one means adding a name here, which is the point:
#: a label typed at a call site is a label nobody checked.
SURFACE_A = "surface-a"
SURFACE_B = "surface-b"
CONVENTION = "convention"
FILLER = "filler"
TASK_ID = "task-id"
REVIEW_FILING = "review-filing"

LABELS = (SURFACE_A, SURFACE_B, CONVENTION, FILLER, TASK_ID, REVIEW_FILING)

#: Bytes of key material. 32 is one SHA-256 block's worth of output.
KEY_BYTES = 32


def new_master_key() -> bytes:
    """A fresh controller-side master key."""
    return os.urandom(KEY_BYTES)


def _parts(coordinates: Iterable[object]) -> bytes:
    """Coordinates to bytes, length-prefixed so no two tuples collide.

    Joining on a separator is not enough: a separator that can appear inside a
    coordinate makes ("a|b", "c") and ("a", "b|c") the same message, and two
    streams that should be independent become one.
    """
    out = bytearray()
    for part in coordinates:
        raw = str(part).encode("utf-8")
        out += len(raw).to_bytes(4, "big") + raw
    return bytes(out)


def derive(master: bytes, label: str, *coordinates: object) -> bytes:
    """Key material for one stream at one set of coordinates."""
    if label not in LABELS:
        raise ValueError(f"undeclared stream label {label!r}; the declared ones are {LABELS}")
    message = _parts((label, *coordinates))
    return hmac.new(master, message, hashlib.sha256).digest()


def rng(master: bytes, label: str, *coordinates: object) -> random.Random:
    """A generator seeded from one stream. Deterministic, and independent per label."""
    return random.Random(int.from_bytes(derive(master, label, *coordinates), "big"))


def task_identifier(master: bytes, *coordinates: object, width: int = 16) -> str:
    """An opaque identifier for one served task.

    It is a truncated HMAC, so it is stable for the same coordinates, unguessable
    without the key, and carries none of the coordinates it was made from.
    """
    if not 8 <= width <= 64:
        raise ValueError("a task identifier is between 8 and 64 hex characters")
    return derive(master, TASK_ID, *coordinates).hex()[:width]


def filler_stream(master: bytes, alphabet: str, length: int, *coordinates: object) -> str:
    """The committed filler: fixed before launch, drawn from the family's alphabet.

    It pads every cell to the envelope size. It is drawn from its own stream so
    nothing about it moves when a surface or a convention moves, which is what
    lets the envelope check assert that padding bytes never carry anything.
    """
    if not alphabet:
        raise ValueError("the filler alphabet cannot be empty")
    generator = rng(master, FILLER, *coordinates)
    return "".join(generator.choice(alphabet) for _ in range(length))


def digest(*chunks: bytes) -> str:
    """A content hash over ordered chunks, length-prefixed the same way."""
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(len(chunk).to_bytes(8, "big"))
        hasher.update(chunk)
    return hasher.hexdigest()


def file_digest(path: str) -> str:
    """The content hash of one file, for the manifest's hash set."""
    with open(path, "rb") as handle:
        return digest(handle.read())


__all__ = [
    "CONVENTION",
    "FILLER",
    "KEY_BYTES",
    "LABELS",
    "REVIEW_FILING",
    "SURFACE_A",
    "SURFACE_B",
    "TASK_ID",
    "derive",
    "digest",
    "file_digest",
    "filler_stream",
    "new_master_key",
    "rng",
    "task_identifier",
]
