"""What a reference points at, and what a directory may be resumed as.

A blob store is content addressed, so the name of an object is a claim about its bytes and
reading it is how that claim is checked. These tests hold the store to that: an object that no
longer hashes to its own name is missing rather than damaged, and a name that is not a hash
never becomes a path at all.

The run directory tests are the version rules. A protocol v1 run directory stays exactly as
readable as it was, by the reader that wrote it, and it is never resumed as v2. So is a
directory that says nothing about its version, and so is one that holds both.

The last of them are about a run holding more than one generation: what a directory says about
the history it shares and the generation it was cut from, and the door a generation created
before its directory comes in by, which adopts what it finds rather than refusing it.

Nothing here needs Temporal or a server: a store is a directory and a manifest is a file.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import List, Tuple

import pytest

from shogym.serve.protocol_v2 import FilesystemBlobStore, WireFormatError, blob_ref
from shogym.serve.protocol_v2.kernel.messages import LineageOrigin
from shogym.serve.protocol_v2.rundir import (
    MANIFEST_FILE,
    STARTING_FILE,
    ResumeRefused,
    attach_run_directory,
    create_run_directory,
    open_run_directory,
    prepare_run_directory,
    serving_record,
    stage_run_directory,
    staged_generation,
)
from shogym.serve.v1_runs import read_dispenses

CONFIGURATION = "a" * 64


def refusal(directory: Path) -> str:
    """Return the protocol code a refused run directory carries."""
    with pytest.raises(ResumeRefused) as caught:
        open_run_directory(directory)
    return caught.value.code


def test_a_blob_is_named_by_the_bytes_it_holds(tmp_path: Path) -> None:
    """Installing is idempotent, reading returns the exact bytes, and the name is the hash."""
    store = FilesystemBlobStore.under(tmp_path)
    reference = store.put(b"a transcript", media_type="text/plain")
    assert reference.sha256 == blob_ref("a transcript").sha256
    assert reference.size == len(b"a transcript")
    assert store.read(reference.sha256) == b"a transcript"
    assert store.holds(reference.sha256)
    assert store.unverified([reference.sha256]) == []

    # The same bytes again are the same object, and no second file is written.
    assert store.put(b"a transcript").sha256 == reference.sha256
    assert len(list(store.root.rglob("*"))) == 2

    absent = blob_ref("never installed").sha256
    assert store.holds(absent) is False
    assert store.unverified([absent, reference.sha256]) == [absent]
    with pytest.raises(WireFormatError):
        store.read(absent)


def test_an_object_that_stopped_hashing_to_its_name_is_a_missing_one(tmp_path: Path) -> None:
    """A name is a promise about bytes. A file that broke it is not the object.

    Which is what makes putting the bytes there again a repair. A store that read the file's
    existence rather than its content would have no way back from this: the right bytes are in
    the caller's hand, and the one call that installs them would decline to.
    """
    store = FilesystemBlobStore.under(tmp_path)
    reference = store.put(b"the original bytes")
    store.path_for(reference.sha256).write_bytes(b"something else entirely")
    assert store.holds(reference.sha256) is False
    assert store.unverified([reference.sha256]) == [reference.sha256]
    with pytest.raises(WireFormatError):
        store.read(reference.sha256)

    # The same bytes again, and what the name promises is what the store produces.
    assert store.put(b"the original bytes").sha256 == reference.sha256
    assert store.read(reference.sha256) == b"the original bytes"
    assert store.unverified([reference.sha256]) == []
    assert list(store.root.rglob("*.partial")) == []


def test_an_object_arrives_under_its_name_or_not_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rename is the publication, and the directory entry it made is flushed after it.

    An event may cite a reference only once the object is installed, so the install has to
    outlast the machine: bytes that reached the disk under a name whose directory entry did not
    are an object the history cites and the store cannot produce. This cuts the run at the
    rename, then watches what a completed one flushes.
    """
    store = FilesystemBlobStore.under(tmp_path)
    digest = blob_ref("an object nobody will find").sha256

    def dies(*args: object, **kwargs: object) -> None:
        raise OSError("the disk went away between the write and the rename")

    monkeypatch.setattr("shogym.serve.protocol_v2.blobs.os.replace", dies)
    with pytest.raises(OSError, match="between the write and the rename"):
        store.put(b"an object nobody will find")
    assert store.holds(digest) is False
    assert list(store.root.rglob("*.partial")) == []
    monkeypatch.undo()

    flushed: List[bool] = []
    fsync = os.fsync

    def watched(descriptor: int) -> None:
        flushed.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        fsync(descriptor)

    monkeypatch.setattr("shogym.serve.protocol_v2.blobs.os.fsync", watched)
    reference = store.put(b"an object nobody will find")
    monkeypatch.undo()
    assert flushed == [False, True]
    assert reference.sha256 == digest
    assert store.read(digest) == b"an object nobody will find"


def test_the_directory_a_name_was_created_in_is_published_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shard is a name in the store, and a run is a name in its parent, as an object is.

    The first object of a hash shard creates that shard, and the first thing a run publishes
    creates the run and its store. Flushing the object and stopping there leaves it under
    directory entries that never reached the disk, and a machine that comes back holds neither
    those entries nor what was published inside them, which is an object the history cites and
    the store cannot produce. So every level a publication had to create is flushed into the
    level above it, before anything inside it is reported installed.
    """
    flushed: List[Tuple[int, int]] = []
    fsync = os.fsync

    def watched(descriptor: int) -> None:
        info = os.fstat(descriptor)
        flushed.append((info.st_dev, info.st_ino))
        fsync(descriptor)

    def entry(path: Path) -> Tuple[int, int]:
        info = path.stat()
        return (info.st_dev, info.st_ino)

    monkeypatch.setattr("shogym.serve.protocol_v2.blobs.os.fsync", watched)
    store = FilesystemBlobStore.under(tmp_path / "run")
    reference = store.put(b"the first object in its shard")
    monkeypatch.undo()
    assert store.read(reference.sha256) == b"the first object in its shard"
    # Every directory that holds a name this call created: the one holding the run, the run
    # holding the store, the store holding the shard, and the shard holding the object.
    shard = store.path_for(reference.sha256).parent
    for holder in (tmp_path, tmp_path / "run", store.root, shard):
        assert entry(holder) in flushed

    # A run directory is the same question one level up, and its manifest goes inside it.
    flushed.clear()
    monkeypatch.setattr("shogym.serve.protocol_v2.blobs.os.fsync", watched)
    run = create_run_directory(
        tmp_path / "runs" / "second",
        workflow_id="stream/durable/1",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
    )
    monkeypatch.undo()
    assert open_run_directory(run.root).manifest == run.manifest
    assert run.blobs.root.is_dir()
    for holder in (tmp_path, tmp_path / "runs", run.root):
        assert entry(holder) in flushed


def test_a_name_that_is_not_a_hash_never_becomes_a_path(tmp_path: Path) -> None:
    """The name is checked before it is joined to anything."""
    store = FilesystemBlobStore.under(tmp_path)
    for name in ("../../etc/passwd", "A" * 64, "abc", ""):
        with pytest.raises(WireFormatError):
            store.path_for(name)


def test_a_run_directory_records_the_generation_it_holds(tmp_path: Path) -> None:
    """One manifest, written once, and the store beside it."""
    run = create_run_directory(
        tmp_path / "run",
        workflow_id="stream/run-1/gen-1",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
    )
    assert run.blobs.root == tmp_path / "run" / "blobs"
    reopened = open_run_directory(tmp_path / "run")
    assert reopened.manifest == run.manifest
    assert reopened.manifest.configuration_hash == CONFIGURATION

    # The manifest is what a resume is checked against, so it is never rewritten in place.
    with pytest.raises(ResumeRefused):
        create_run_directory(
            tmp_path / "run",
            workflow_id="stream/run-1/gen-2",
            task_queue="shogym-stream-v2",
            configuration_hash="b" * 64,
        )


def test_a_version_one_run_directory_is_read_offline_and_never_resumed(tmp_path: Path) -> None:
    """The v1 reader still reads it. A v2 resume refuses it, and refuses the mixture too."""
    directory = tmp_path / "v1"
    directory.mkdir()
    row = {"task_id": "17", "dispensed_at": 1}
    (directory / "dispenses.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    assert read_dispenses(directory) == [row]
    assert refusal(directory) == "unsupported_version"

    # A v1 log beside a v2 manifest is the mixed case, and it is refused on the same terms.
    mixed = tmp_path / "mixed"
    create_run_directory(
        mixed,
        workflow_id="stream/run-1/gen-1",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
    )
    (mixed / "results.jsonl").write_text("", encoding="utf-8")
    assert refusal(mixed) == "unsupported_version"


def test_a_directory_that_says_nothing_about_its_version_is_refused(tmp_path: Path) -> None:
    """A missing version, another version, and an incomplete creation are all refusals."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert refusal(empty) == "configuration_mismatch"

    manifest = {
        "protocol_version": 2,
        "schedule_version": "shogym.schedule.1",
        "workflow_id": "stream/run-1/gen-1",
        "task_queue": "shogym-stream-v2",
        "configuration_hash": CONFIGURATION,
    }
    for change, code in (
        ({"protocol_version": None}, "unsupported_version"),
        ({"protocol_version": 1}, "unsupported_version"),
        ({"schedule_version": "shogym.schedule.0"}, "unsupported_version"),
        ({"configuration_hash": None}, "configuration_mismatch"),
    ):
        directory = tmp_path / f"case-{len(list(tmp_path.iterdir()))}"
        directory.mkdir()
        written = dict(manifest)
        for key, value in change.items():
            if value is None:
                written.pop(key)
            else:
                written[key] = value
        (directory / MANIFEST_FILE).write_text(json.dumps(written), encoding="utf-8")
        assert refusal(directory) == code


def test_a_directory_prepared_for_a_generation_is_not_one_yet(tmp_path: Path) -> None:
    """The manifest is the claim that the generation exists, so it is written last.

    A caller has to create the generation before it can say what the generation is, which puts
    a crash between the directory and the manifest. What that leaves has to be a directory
    nothing resumes and the next attempt can still use, rather than a manifest naming a stream
    that was never started.
    """
    directory = prepare_run_directory(tmp_path / "run")
    assert FilesystemBlobStore.under(directory).root.is_dir()
    assert (directory / MANIFEST_FILE).exists() is False
    assert refusal(directory) == "configuration_mismatch"

    # Preparing it again is not a second generation, because there is no first one yet.
    assert prepare_run_directory(directory) == directory
    run = create_run_directory(
        directory,
        workflow_id="stream/prepared/1",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
    )
    assert open_run_directory(directory).manifest == run.manifest

    # Once it names a generation it is that generation's directory and no other's.
    with pytest.raises(ResumeRefused) as caught:
        prepare_run_directory(directory)
    assert caught.value.code == "configuration_mismatch"


def test_a_directory_names_the_generation_it_is_starting_before_it_starts(tmp_path: Path) -> None:
    """A record between the two, so a run that dies mid creation leaves a name behind.

    The identifier is minted by the process that starts the stream. A crash before the manifest
    lands would otherwise leave an authority nothing on the disk names, and the next attempt
    mints another and leaves the first one running. The starting record is not a generation, so
    it resumes nothing and refuses nothing: it is one name for the next attempt to act on.
    """
    root = tmp_path / "run"
    staged = stage_run_directory(
        root,
        workflow_id="stream/starting/1",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
    )
    assert staged_generation(root) == staged
    assert refusal(root) == "configuration_mismatch"

    # A record this code cannot read names nobody, and is answered like an absent one.
    (root / STARTING_FILE).write_text("half a record", encoding="utf-8")
    assert staged_generation(root) is None

    # The manifest is what it was standing in for, so it goes once the manifest is there.
    run = create_run_directory(
        root,
        workflow_id="stream/starting/1",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
    )
    assert (root / STARTING_FILE).exists() is False
    assert open_run_directory(root).manifest == run.manifest


def a_lineage_origin(fork_id: str = "fork-1", slot: str = "kept") -> LineageOrigin:
    """What a child's directory says about the generation it was cut from."""
    return LineageOrigin(
        parent_workflow_id="stream/run-1/gen-1",
        parent_run_id="run-abcdef",
        acknowledged_cursor="0" * 32,
        branch_slot=slot,
        fork_id=fork_id,
    )


def test_a_directory_written_before_a_run_could_hold_a_lineage_is_read_as_it_was(
    tmp_path: Path,
) -> None:
    """A manifest that names one generation and no lineage is the file it always was.

    A run with one generation keeps its history beside its own manifest and was cut from
    nothing, so it writes the five fields it wrote before either question existed and reads back
    saying its history is here. That is what stops a directory an older build left behind from
    becoming unreadable, and it is checked against the file rather than against the value.
    """
    run = create_run_directory(
        tmp_path / "run",
        workflow_id="stream/plain/1",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
    )
    written = json.loads((run.root / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert set(written) == {
        "protocol_version",
        "schedule_version",
        "workflow_id",
        "task_queue",
        "configuration_hash",
    }
    reopened = open_run_directory(run.root)
    assert reopened.manifest.origin is None
    assert reopened.database_root == run.root.resolve()


def test_a_child_directory_says_where_its_lineages_history_is_and_what_cut_it(
    tmp_path: Path,
) -> None:
    """The two things a generation cut from another writes down, and how they read back.

    The history is a walk from this directory rather than a place, because the walk is what
    survives the lineage being copied somewhere else: a manifest holding the original's own
    location would send a read of the copy back to the database the original is serving. So an
    absolute one is refused rather than followed.
    """
    root = tmp_path / "run"
    origin = a_lineage_origin()
    child = attach_run_directory(
        root / "child-1",
        workflow_id="stream/plain/1.fork.1.abcd",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
        database_root="..",
        origin=origin,
    )
    assert child.database_root == root.resolve()

    reopened = open_run_directory(child.root)
    assert reopened.manifest.origin == origin
    assert reopened.manifest.database_root == ".."
    assert reopened.database_root == root.resolve()

    # An origin is whole or it is not there, and the history is a walk or it is refused.
    for change, code in (
        ({"origin": {"fork_id": "fork-1"}}, "configuration_mismatch"),
        ({"database_root": str(root.resolve())}, "configuration_mismatch"),
        ({"database_root": ""}, "configuration_mismatch"),
    ):
        elsewhere = tmp_path / f"edited-{len(list(tmp_path.iterdir()))}"
        elsewhere.mkdir()
        payload = {**json.loads((child.root / MANIFEST_FILE).read_text(encoding="utf-8")), **change}
        (elsewhere / MANIFEST_FILE).write_text(json.dumps(payload), encoding="utf-8")
        assert refusal(elsewhere) == code


def test_attaching_a_directory_twice_adopts_it_and_never_makes_a_second(tmp_path: Path) -> None:
    """The repair for a crash mid registration is to do it again, so the door is idempotent.

    Publishing a manifest, installing a closure and claiming ownership are three steps a crash
    can be between. The ordinary creation door refuses a directory that already holds a manifest,
    so a retry through it would refuse the child it had just registered; this one adopts the
    manifest it finds where the manifest says the same thing, and refuses where it does not.
    """
    root = tmp_path / "run" / "child-1"
    first = attach_run_directory(
        root,
        workflow_id="stream/plain/1.fork.1.abcd",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
        database_root="..",
        origin=a_lineage_origin(),
    )
    again = attach_run_directory(
        root,
        workflow_id="stream/plain/1.fork.1.abcd",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
        database_root="..",
        origin=a_lineage_origin(),
    )
    assert again == first
    assert open_run_directory(root).manifest == first.manifest

    # A directory already holding a generation is not somewhere to put a second, whichever half
    # of what it says has moved.
    for change in (
        {"workflow_id": "stream/plain/1.fork.2.beef"},
        {"configuration_hash": "b" * 64},
        {"database_root": "."},
        {"origin": a_lineage_origin(fork_id="fork-2")},
    ):
        with pytest.raises(ResumeRefused) as caught:
            attach_run_directory(
                root,
                **{
                    "workflow_id": "stream/plain/1.fork.1.abcd",
                    "task_queue": "shogym-stream-v2",
                    "configuration_hash": CONFIGURATION,
                    "database_root": "..",
                    "origin": a_lineage_origin(),
                    **change,
                },
            )
        assert caught.value.code == "configuration_mismatch"


def test_a_generation_created_before_its_directory_still_gets_one(tmp_path: Path) -> None:
    """Attaching makes the directory and the store where there is nothing there yet.

    A fork creates its children before anything serves them, so the stream exists first and
    there is no name to reserve: the manifest is written for a generation that is already
    running rather than to stand in for one about to start. What that leaves behind is a
    directory the ordinary reader opens like any other.
    """
    root = tmp_path / "run" / "child-2"
    child = attach_run_directory(
        root,
        workflow_id="stream/plain/1.fork.2.beef",
        task_queue="shogym-stream-v2",
        configuration_hash=CONFIGURATION,
        database_root="..",
        origin=a_lineage_origin(fork_id="fork-1", slot="placebo"),
    )
    assert child.root.is_dir()
    assert (root / STARTING_FILE).exists() is False
    assert serving_record(root) is not None
    assert open_run_directory(root) == child
