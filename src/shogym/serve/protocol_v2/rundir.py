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
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from shogym.serve.protocol_v2.blobs import (
    FilesystemBlobStore,
    create_directory,
    flush_directory,
)
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.records import PROTOCOL_VERSION
from shogym.serve.protocol_v2.schedule import SCHEDULE_VERSION

# The manifest one generation writes about itself, once, when it is created.
MANIFEST_FILE = "generation.json"

# What a run writes down before it starts the stream the manifest will name.
STARTING_FILE = "generation.starting.json"

# The logs the version one serving path appends. Their presence is what makes a directory a
# version one run directory, whatever else is in it.
V1_LOGS = ("dispenses.jsonl", "results.jsonl")

_FIELDS = ("protocol_version", "schedule_version", "workflow_id", "task_queue", "configuration_hash")


class ResumeRefused(WireFormatError):
    """A run directory this protocol will not resume, carrying the code that says why."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class RunManifest:
    """What one generation says about itself: where it runs, and what it is."""

    workflow_id: str
    task_queue: str
    configuration_hash: str
    protocol_version: int = PROTOCOL_VERSION
    schedule_version: str = SCHEDULE_VERSION

    def to_wire(self) -> Dict[str, Any]:
        """Return the manifest as the JSON object the directory holds."""
        return {
            "protocol_version": self.protocol_version,
            "schedule_version": self.schedule_version,
            "workflow_id": self.workflow_id,
            "task_queue": self.task_queue,
            "configuration_hash": self.configuration_hash,
        }


@dataclass(frozen=True)
class RunDirectory:
    """One generation's directory: its manifest, and the blobs its events reference."""

    root: Path
    manifest: RunManifest

    @property
    def blobs(self) -> FilesystemBlobStore:
        """The store this run installs its blobs in."""
        return FilesystemBlobStore.under(self.root)


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

    It arrives whole or not at all. A file that exists is what says this directory holds a
    generation, so a partial one is the worst of both: it names no generation anybody can
    resume, and the next attempt is refused by it rather than being able to run out of the
    directory.

    The starting record goes once the manifest is there. The directory now holds a generation,
    which is what the record was standing in for, and a record left behind by a crash between
    the two names the generation the manifest names.
    """
    directory = prepare_run_directory(root)
    manifest = RunManifest(
        workflow_id=workflow_id, task_queue=task_queue, configuration_hash=configuration_hash
    )
    _publish(directory / MANIFEST_FILE, json.dumps(manifest.to_wire(), sort_keys=True) + "\n")
    _discard(directory / STARTING_FILE)
    return RunDirectory(root=directory, manifest=manifest)


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
    if missing or set(payload) != set(_FIELDS):
        raise ResumeRefused("configuration_mismatch", f"{path} is not a complete manifest")
    return RunManifest(
        workflow_id=payload["workflow_id"],
        task_queue=payload["task_queue"],
        configuration_hash=payload["configuration_hash"],
        protocol_version=payload["protocol_version"],
        schedule_version=payload["schedule_version"],
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
