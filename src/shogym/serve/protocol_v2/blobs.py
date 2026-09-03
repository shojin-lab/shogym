"""Immutable objects, named by the hash of their own bytes.

An event references a snapshot, a transcript, or a provider turn by hash rather than by value,
so the authority stays small while what it points at stays complete. That only works if a
reference can be checked: the store here is content addressed, and reading is a verification,
because a file whose bytes do not hash to its own name is not the object the name promised.

The store is an interface with one implementation. :class:`FilesystemBlobStore` is a directory,
one file per object, which is what a local run needs and what a test can inspect. A remote
store is another implementation of the same three methods, and nothing above this module knows
which one it has.

Nothing here imports Temporal. A blob is installed by whoever holds the bytes, which is the
harness, and verified by the authority before it lets an event cite it.
"""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Protocol, Union

from shogym.serve.protocol_v2.errors import WireFormatError

# The store's directory inside a run directory. It is a fixed name so a reader given the run
# finds the blobs without being told a second path.
BLOB_DIRECTORY = "blobs"

_DIGEST = re.compile(r"[0-9a-f]{64}")


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


def require_digest(value: str) -> str:
    """Return ``value`` if it names a blob, and refuse it otherwise.

    A name is checked before it is used to build a path, so a reference that is not a hash
    cannot reach the filesystem at all.
    """
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise WireFormatError("a blob is named by 64 lower-case hexadecimal characters")
    return value


class BlobStore(Protocol):
    """Where referenced bytes live, and how a reader proves they are the right ones."""

    def put(self, data: bytes, *, media_type: str = "application/octet-stream") -> BlobRef:
        """Install ``data`` and return the reference that names it."""
        ...

    def read(self, digest: str) -> bytes:
        """Return the exact bytes ``digest`` names, or refuse."""
        ...

    def unverified(self, digests: Iterable[str]) -> List[str]:
        """Return the digests this store cannot produce the exact bytes for."""
        ...


def flush_directory(directory: Path) -> None:
    """Make a rename into ``directory`` durable, so the name a file was published under lasts.

    A file whose own bytes reached the disk inside a directory whose entry did not is no more
    durable than that entry. A filesystem that will not flush a directory must not fail the
    write over it, so a refusal here is the end of it.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def create_directory(directory: Path) -> None:
    """Make ``directory``, and make the name of every level it had to create durable too.

    A directory is a name in the directory above it exactly as a file is. An object flushed
    into a directory whose own entry never reached the disk is no more durable than that entry,
    and a machine that comes back holds neither: the first object in a new shard, and every
    object in a new run, are published under names that were themselves just created. So each
    level that did not exist is created from the top down and flushed into the level above it,
    before anything is written inside the last one.
    """
    missing: List[Path] = []
    current = directory
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    for path in reversed(missing):
        path.mkdir(exist_ok=True)
        flush_directory(path.parent)


@dataclass(frozen=True)
class FilesystemBlobStore:
    """A blob store that is a directory: one file per object, named by its own hash.

    A write is a temporary file in the object's own directory, then a rename, then a flush of
    the directory entry: a reader finds either nothing under a name or the complete object, and
    a machine that comes back finds what it had before it went away. A shard the object is the
    first in is a new name as well, so it is created the same way, from the store's own root
    down, before the object goes into it.

    Installing an object that is already installed writes nothing, which is what makes
    installation idempotent under retry. What counts as installed is read rather than assumed:
    a name is a promise about bytes, so an object that no longer keeps that promise is not
    installed, and putting the right bytes there again is how a damaged store is repaired.
    """

    root: Path

    @classmethod
    def under(cls, run_directory: Union[str, Path]) -> "FilesystemBlobStore":
        """Return the store a run directory keeps its blobs in."""
        return cls(Path(run_directory) / BLOB_DIRECTORY)

    def path_for(self, digest: str) -> Path:
        """Where the object named ``digest`` lives."""
        checked = require_digest(digest)
        return self.root / checked[:2] / checked

    def put(self, data: bytes, *, media_type: str = "application/octet-stream") -> BlobRef:
        """Install ``data`` under its own hash and return the reference to it.

        A file already under that name is read before it is believed. If it is the object the
        name promises this writes nothing, and if it is not, these bytes replace it: a store
        that lost an object cannot be repaired by an install that takes the damage as proof
        the work is done.
        """
        digest = sha256(data).hexdigest()
        path = self.path_for(digest)
        if not self.holds(digest):
            create_directory(path.parent)
            descriptor, staged = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
            try:
                with os.fdopen(descriptor, "wb") as opened:
                    opened.write(data)
                    opened.flush()
                    os.fsync(opened.fileno())
                os.replace(staged, path)
            except BaseException:
                Path(staged).unlink(missing_ok=True)
                raise
            flush_directory(path.parent)
        return BlobRef(sha256=digest, size=len(data), media_type=media_type)

    def holds(self, digest: str) -> bool:
        """Whether the object this name promises is installed. It reads the bytes to say so."""
        try:
            self.read(digest)
        except WireFormatError:
            return False
        return True

    def read(self, digest: str) -> bytes:
        """Return the bytes ``digest`` names.

        The content is hashed on the way out. A name is a promise about bytes, so a file that
        no longer keeps that promise is a missing object rather than a damaged one.
        """
        path = self.path_for(digest)
        try:
            data = path.read_bytes()
        except OSError as error:
            raise WireFormatError(f"no blob is installed under {digest}") from error
        if sha256(data).hexdigest() != digest:
            raise WireFormatError(f"the object under {digest} does not hash to that name")
        return data

    def unverified(self, digests: Iterable[str]) -> List[str]:
        """Return the digests this store cannot produce the exact bytes for, in order."""
        return [digest for digest in digests if not self.holds(digest)]
