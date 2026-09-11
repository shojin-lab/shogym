"""The directory a durable generation keeps its blobs and its resume manifest in.

A run directory is what a new owner is handed when it takes over a generation nobody is
serving any more. It holds two things: the content-addressed blobs the authority's events
reference, and one small manifest saying which generation lives here, under which versions,
and against which immutable configuration hash. The authority itself is the workflow history,
so nothing here is a second event store.

The version rules are the whole reason the manifest exists. A directory that says nothing
about its protocol version, says version one, or holds a version one log beside a version two
manifest is refused before anything is claimed. A version one run directory stays exactly as
readable as it was, by the reader that wrote it, and it can never become a version two stream:
its rows cannot say whether a payload was ever delivered, so there is no state to resume into.

A generation is written down twice, because it is created in two steps. The starting record
says which generation this directory is about to start, and it is written before the stream
exists; the manifest says which generation it holds, and it is written once the stream does.
A directory left holding only the starting record is a run that died in between, and what it
names is a generation nothing points at: the next attempt reads it, ends what it names, and
starts its own.

One run can hold more than one generation. A generation cut from another lives in a directory
of its own, with its own manifest and its own blobs, under a run root the whole lineage shares:
one history, one task queue, one service. So a manifest says two more things than it used to. It
says where the shared history is, as a path relative to the directory holding the manifest, so a
lineage copied somewhere else resolves inside the copy rather than pointing back at the original.
And it says where this generation was cut from, which is what makes a directory readable as a
child rather than as a run that started at a cursor nobody can explain.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from shogym import __version__ as shogym_version
from shogym.serve.protocol_v2.blobs import (
    BLOB_DIRECTORY,
    FilesystemBlobStore,
    create_directory,
    flush_directory,
)
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.records import PROTOCOL_VERSION
from shogym.serve.protocol_v2.schedule import SCHEDULE_VERSION

if TYPE_CHECKING:  # pragma: no cover - the kernel reaches this module, so the arrow is one way
    from shogym.serve.protocol_v2.kernel.messages import LineageOrigin

# The manifest one generation writes about itself, once, when it is created.
MANIFEST_FILE = "generation.json"

# What a run writes down before it starts the stream the manifest will name.
STARTING_FILE = "generation.starting.json"

# What the process serving this run says about itself: the package version whose Worker is
# polling this run's task queue. It is beside the manifest rather than in it, and nothing reads
# it to decide anything: the manifest holds a closed set of names, so a directory with a name
# outside it would be refused by code that predates that name, and a fact recorded for an
# operator must not be a fact that stops a run resuming.
#
# What it records is a deployment invariant. A generation may run in more than one execution,
# because it continues as new before the durable service's limits bound it, and every execution
# of the chain decodes the projection the one before it wrote. A Worker whose package predates
# that projection would decode the start, drop the field it does not know, and serve the roster
# again from the beginning. Nothing in the older code could refuse it, because the refusal did
# not exist there. What excludes it is the deployment: one Worker per run, on that run's own
# task queue, from one package. This is where that package writes its name down.
SERVING_FILE = "serving.json"

# The logs the version one serving path appends. Their presence is what makes a directory a
# version one run directory, whatever else is in it.
V1_LOGS = ("dispenses.jsonl", "results.jsonl")

_FIELDS = ("protocol_version", "schedule_version", "workflow_id", "task_queue", "configuration_hash")

# What a manifest may hold beside those five, and what an older one simply does not have. They
# are optional rather than required because every run directory written before a run could hold
# more than one generation has neither, and a fact recorded for a lineage must not be a fact that
# stops an ordinary run resuming.
_LINEAGE_FIELDS = ("database_root", "origin")

# The members of the origin a child's manifest holds, all of them or none.
_ORIGIN_FIELDS = (
    "parent_workflow_id",
    "parent_run_id",
    "acknowledged_cursor",
    "branch_slot",
    "fork_id",
)

# Where a generation's own directory says the shared history is, for one that shares nobody's.
OWN_DATABASE_ROOT = "."


class ResumeRefused(WireFormatError):
    """A run directory this protocol will not resume, carrying the code that says why."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RunManifest:
    """What one generation says about itself: where it runs, what it is, and where it came from.

    ``database_root`` is where the history this generation's records are read out of lives,
    written as a path relative to the directory holding this manifest. A generation with a
    history of its own says so with the current directory, and a generation sharing a lineage's
    says how to walk to it. It is relative on purpose: an absolute path in a copied lineage would
    send a read of the copy back to the original's live database.

    ``origin`` is the generation this one was cut from, for one that was cut. A directory holding
    it is a child, and a table over it says which generation it inherited its prefix from.
    """

    workflow_id: str
    task_queue: str
    configuration_hash: str
    protocol_version: int = PROTOCOL_VERSION
    schedule_version: str = SCHEDULE_VERSION
    database_root: str = OWN_DATABASE_ROOT
    origin: Optional["LineageOrigin"] = None

    def to_wire(self) -> Dict[str, Any]:
        """Return the manifest as the JSON object the directory holds.

        A generation that shares nobody's history and was cut from nothing writes exactly the
        five fields it always wrote, so a directory this build creates for an ordinary run is
        the file the build before it created.
        """
        payload: Dict[str, Any] = {
            "protocol_version": self.protocol_version,
            "schedule_version": self.schedule_version,
            "workflow_id": self.workflow_id,
            "task_queue": self.task_queue,
            "configuration_hash": self.configuration_hash,
        }
        if self.database_root != OWN_DATABASE_ROOT:
            payload["database_root"] = self.database_root
        if self.origin is not None:
            payload["origin"] = {
                "parent_workflow_id": self.origin.parent_workflow_id,
                "parent_run_id": self.origin.parent_run_id,
                "acknowledged_cursor": self.origin.acknowledged_cursor,
                "branch_slot": self.origin.branch_slot,
                "fork_id": self.origin.fork_id,
            }
        return payload


@dataclass(frozen=True)
class RunDirectory:
    """One generation's directory: its manifest, and the blobs its events reference."""

    root: Path
    manifest: RunManifest

    @property
    def blobs(self) -> FilesystemBlobStore:
        """The store this run installs its blobs in."""
        return FilesystemBlobStore.under(self.root)

    @property
    def database_root(self) -> Path:
        """The directory holding the history this generation's records are read out of.

        It is resolved against this directory rather than assumed to be it, because a generation
        cut from another shares its lineage's history and keeps only its own manifest and blobs.
        Resolving here is what makes a relocated lineage read: the manifest says how to walk from
        this directory to the shared one, and the walk lands wherever the copy is.
        """
        return (self.root / self.manifest.database_root).resolve()


def prepare_run_directory(root: Union[str, Path]) -> Path:
    """Make the directory a generation will run out of, before there is a generation to name.

    This is every refusal a new run can meet in its directory and none of the recording. A
    caller that has to create the generation before it can say what the generation is prepares
    the directory first and writes the manifest once the generation exists, so a run that dies
    in between leaves a directory the next attempt can still use rather than a manifest naming
    a generation that was never started.

    The run and its store are created as durably as what goes into them. Everything this run
    records is published inside these two directories, and a name published inside a directory
    whose own entry never reached the disk goes when that entry does.
    """
    directory = Path(root)
    _refuse_v1_logs(directory)
    path = directory / MANIFEST_FILE
    if path.exists():
        raise ResumeRefused(
            "configuration_mismatch", f"{directory} already holds a generation manifest"
        )
    create_directory(directory)
    create_directory(FilesystemBlobStore.under(directory).root)
    return directory


def stage_run_directory(
    root: Union[str, Path],
    *,
    workflow_id: str,
    task_queue: str,
    configuration_hash: str,
) -> RunManifest:
    """Write down the generation this directory is about to start, before it is started.

    A stream is created and then recorded, and a run that dies in between would otherwise leave
    a live generation whose name nothing on the disk holds: no manifest points at it, its
    identifier was minted at random, and the next attempt mints another one and leaves the
    first running with no consumer and no record. So the identifier goes down first. The next
    attempt out of this directory reads it and ends what it names.

    What it names never served a message. The manifest is written before the consumer is
    claimed, so a directory holding a starting record and no manifest holds a generation that
    no transport was ever bound to.
    """
    directory = prepare_run_directory(root)
    staged = RunManifest(
        workflow_id=workflow_id, task_queue=task_queue, configuration_hash=configuration_hash
    )
    _publish(directory / STARTING_FILE, json.dumps(staged.to_wire(), sort_keys=True) + "\n")
    return staged


def staged_generation(root: Union[str, Path]) -> Optional[RunManifest]:
    """Return the generation a previous attempt started here and never recorded, if there is one.

    A record this code cannot read names nothing anybody can act on, so it is answered the same
    way an absent one is. That leaves the run no worse off than it was without the record.
    """
    path = Path(root) / STARTING_FILE
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return _manifest(path, payload)
    except (OSError, ValueError, ResumeRefused):
        return None


def create_run_directory(
    root: Union[str, Path],
    *,
    workflow_id: str,
    task_queue: str,
    configuration_hash: str,
) -> RunDirectory:
    """Create the directory one generation will run out of, and write its manifest.

    The manifest is written once and never rewritten, because a resume compares against it: a
    directory whose recorded configuration could be edited to match a changed generation would
    check nothing. Writing it says the generation it names exists, so it is written by a caller
    that has one.

    The starting record goes once the manifest is there. The directory now holds a generation,
    which is what the record was standing in for, and a record left behind by a crash between
    the two names the generation the manifest names.
    """
    directory = prepare_run_directory(root)
    manifest = RunManifest(
        workflow_id=workflow_id, task_queue=task_queue, configuration_hash=configuration_hash
    )
    _record_generation(directory, manifest)
    _discard(directory / STARTING_FILE)
    return RunDirectory(root=directory, manifest=manifest)


def attach_run_directory(
    root: Union[str, Path],
    *,
    workflow_id: str,
    task_queue: str,
    configuration_hash: str,
    database_root: str = OWN_DATABASE_ROOT,
    origin: Optional["LineageOrigin"] = None,
) -> RunDirectory:
    """Adopt the directory a generation somebody else created already lives in, or make it.

    This is the door a generation that was created before its directory comes in by. A fork
    creates its children gated and unowned and their directories are prepared afterwards, so the
    stream exists first and there is nothing to stage: the manifest is written for a generation
    that is already running rather than to reserve a name for one about to start.

    It is idempotent, and that is the point of it rather than a convenience. Publishing a
    manifest, installing a closure and claiming ownership are three steps a crash can be between,
    and the repair is to do them again: the ordinary creation path refuses a directory that
    already holds a manifest, so a retry through it would refuse the child it had just registered.
    A manifest naming this same generation, hash, history and origin is adopted and the directory
    is returned as it stands; one naming anything else is refused, because a directory already
    holding a generation is not somewhere to put a second.

    The store is not made here, the way the creation path makes one. A generation created before
    its directory names the store it verifies against inside its own start, and installing an
    object makes what it needs, so a directory laid out for a store the generation does not use
    would be a directory with an empty one in it.
    """
    directory = Path(root)
    _refuse_v1_logs(directory)
    manifest = RunManifest(
        workflow_id=workflow_id,
        task_queue=task_queue,
        configuration_hash=configuration_hash,
        database_root=database_root,
        origin=origin,
    )
    path = directory / MANIFEST_FILE
    if path.is_file():
        standing = open_run_directory(directory)
        if standing.manifest != manifest:
            raise ResumeRefused(
                "configuration_mismatch",
                f"{directory} already holds {standing.manifest.workflow_id}, and this is "
                f"{workflow_id} being attached to it",
            )
        return standing
    create_directory(directory)
    _record_generation(directory, manifest)
    return RunDirectory(root=directory, manifest=manifest)


def _record_generation(directory: Path, manifest: RunManifest) -> None:
    """Write down the generation this directory holds, and the package serving it.

    The manifest arrives whole or not at all. A file that exists is what says this directory
    holds a generation, so a partial one is the worst of both: it names no generation anybody can
    resume, and the next attempt is refused by it rather than being able to run out of the
    directory.
    """
    _publish(directory / MANIFEST_FILE, json.dumps(manifest.to_wire(), sort_keys=True) + "\n")
    _publish(
        directory / SERVING_FILE,
        json.dumps(
            {
                "package": "shogym",
                "version": shogym_version,
                "task_queue": manifest.task_queue,
            },
            sort_keys=True,
        )
        + "\n",
    )


def serving_record(root: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """What the package that created this run directory said about itself, if it said anything.

    A directory written before this record existed simply has none, which is why the answer is
    allowed to be nothing. Nothing decides anything on it: it is what an operator reads to know
    which package version's Worker a chain of executions was served by.
    """
    path = Path(root) / SERVING_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _publish(path: Path, payload: str) -> None:
    """Put a file in place whole: write it beside itself, get it to the disk, then rename.

    The rename is the publication, and it is the one step a crash can be either side of. The
    directory entry is flushed afterwards, because a file made durable inside a directory whose
    own entry is not is no more durable than that entry.
    """
    directory = path.parent
    descriptor, temporary = tempfile.mkstemp(dir=str(directory), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    flush_directory(directory)


def _discard(path: Path) -> None:
    """Remove a file that has served its purpose, and flush the removal."""
    try:
        os.unlink(path)
    except OSError:
        return
    flush_directory(path.parent)


def run_directory_of(blob_root: Union[str, Path]) -> Path:
    """Return the directory a store laid out inside a run directory belongs to.

    It is the inverse of the layout every generation has, and it exists because a generation
    something else created is authenticated by its start and a start names its store rather than
    its directory. Deriving the directory from that authenticated value is what keeps a caller
    attaching to one from being handed a second opinion about where the generation lives.
    """
    root = Path(blob_root)
    if root.name != BLOB_DIRECTORY:
        raise ValueError(
            f"{str(root)!r} is not a store this build laid out: a generation keeps its objects "
            f"under {BLOB_DIRECTORY!r} inside its own directory"
        )
    return root.parent


def open_run_directory(root: Union[str, Path]) -> RunDirectory:
    """Return the generation this directory holds, or refuse to resume it.

    Every refusal happens here, before an owner is claimed and before anything is read from the
    authority, which is what keeps a rejected directory a directory nobody touched.
    """
    directory = Path(root)
    _refuse_v1_logs(directory)
    path = directory / MANIFEST_FILE
    if not path.is_file():
        raise ResumeRefused(
            "configuration_mismatch",
            f"{directory} holds no completed protocol v2 generation manifest",
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ResumeRefused("configuration_mismatch", f"{path} is not a JSON manifest") from error
    return RunDirectory(root=directory, manifest=_manifest(path, payload))


def _manifest(path: Path, payload: Any) -> RunManifest:
    """Read the manifest a directory holds, refusing every version this code cannot serve."""
    if not isinstance(payload, dict):
        raise ResumeRefused("configuration_mismatch", f"{path} is not a JSON object")
    if "protocol_version" not in payload:
        raise ResumeRefused("unsupported_version", f"{path} names no protocol version")
    if payload["protocol_version"] != PROTOCOL_VERSION:
        raise ResumeRefused(
            "unsupported_version",
            f"{path} is protocol version {payload['protocol_version']!r} and this is version "
            f"{PROTOCOL_VERSION}",
        )
    if payload.get("schedule_version") != SCHEDULE_VERSION:
        raise ResumeRefused(
            "unsupported_version",
            f"{path} names schedule version {payload.get('schedule_version')!r}, which this "
            f"code does not serve",
        )
    missing = [name for name in _FIELDS if name not in payload]
    if missing or not set(payload) <= set(_FIELDS) | set(_LINEAGE_FIELDS):
        raise ResumeRefused("configuration_mismatch", f"{path} is not a complete manifest")
    return RunManifest(
        workflow_id=payload["workflow_id"],
        task_queue=payload["task_queue"],
        configuration_hash=payload["configuration_hash"],
        protocol_version=payload["protocol_version"],
        schedule_version=payload["schedule_version"],
        database_root=_database_root(path, payload),
        origin=_origin(path, payload),
    )


def _database_root(path: Path, payload: Dict[str, Any]) -> str:
    """Where this manifest says the shared history is, refusing anything but a relative walk.

    An absolute path is refused rather than followed, because the whole point of the field is
    that a copied lineage resolves inside the copy: a manifest carrying the original's own
    location would send a read of the copy back to the database the original is still serving.
    """
    root = payload.get("database_root", OWN_DATABASE_ROOT)
    if not isinstance(root, str) or not root or Path(root).is_absolute():
        raise ResumeRefused(
            "configuration_mismatch",
            f"{path} names the shared history at {root!r}, and what a manifest holds is where to "
            f"walk to it from the directory holding the manifest",
        )
    return root


def _origin(path: Path, payload: Dict[str, Any]) -> Optional["LineageOrigin"]:
    """The generation this directory was cut from, whole or not at all.

    The type is imported here rather than at the top, because the kernel that declares it is what
    reaches this module: a run directory is what a kernel resumes out of, so the arrow between
    them points one way and this is the one place that has to look back along it.
    """
    from shogym.serve.protocol_v2.kernel.messages import LineageOrigin

    origin = payload.get("origin")
    if origin is None:
        return None
    if not isinstance(origin, dict) or set(origin) != set(_ORIGIN_FIELDS):
        raise ResumeRefused(
            "configuration_mismatch", f"{path} does not name whole the generation it was cut from"
        )
    return LineageOrigin(
        parent_workflow_id=str(origin["parent_workflow_id"]),
        parent_run_id=str(origin["parent_run_id"]),
        acknowledged_cursor=str(origin["acknowledged_cursor"]),
        branch_slot=str(origin["branch_slot"]),
        fork_id=str(origin["fork_id"]),
    )


def _refuse_v1_logs(directory: Path) -> None:
    """Refuse a directory the version one serving path wrote, whatever else it holds.

    A version one log beside a version two manifest is the mixed case, and it is refused for
    the same reason as a plain version one directory: two protocols would be claiming the same
    run, and only one of them records whether a message was delivered.
    """
    for name in V1_LOGS:
        if (directory / name).exists():
            raise ResumeRefused(
                "unsupported_version",
                f"{directory} holds the protocol v1 log {name}, which is readable offline and "
                f"is never resumed as protocol v2",
            )
