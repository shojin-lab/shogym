"""``orca_bench`` dataset: the pins, the cache, and the index the port builds over it.

The dataset is downloaded on demand and never vendored, so everything here runs offline against
a synthetic dataset in the real on-disk shape, except one ``network``-marked test, which pulls a
single real task into the cache and re-checks the loader and the redaction property against real
bytes. The offline suite (``-m "not network"``) therefore passes with an empty cache and no
network at all.
"""

from __future__ import annotations

import contextlib
import io
import itertools
import json
import re
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import pytest

import shogym
from shogym.envs.orca_bench import dataset, tasks
from tests._fixtures import orca_bench_dataset as synth

_SNAPSHOT_RE = re.compile(r"^\d{8}T\d{6}Z-[0-9a-f]{16}$")


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    return synth.write_dataset(tmp_path / "orca")


# ----- pins -----


def test_dataset_pin_is_recorded() -> None:
    assert dataset.DATASET == "orca-bench/orca-bench"
    assert dataset.DATASET_REVISION == 2
    assert re.fullmatch(r"[0-9a-f]{64}", dataset.DATASET_CONTENT_HASH)
    assert dataset.DATASET_TASK_COUNT == 755
    assert dataset.DATASET_LICENSE == "CC BY 4.0"


def test_cache_dir_honors_the_shogym_conventions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SHOGYM_ORCA_BENCH_DATA_DIR", raising=False)
    monkeypatch.setenv("SHOGYM_CACHE", "/tmp/shogym-cache-probe")
    assert dataset.cache_dir() == Path("/tmp/shogym-cache-probe/orca_bench")
    monkeypatch.setenv("SHOGYM_ORCA_BENCH_DATA_DIR", "/tmp/orca-probe")
    assert dataset.cache_dir() == Path("/tmp/orca-probe")
    # The revision hash keys the tree, so a re-pin lands beside the old one.
    assert dataset.dataset_dir().name == dataset.DATASET_CONTENT_HASH


def test_archive_path_is_content_addressed() -> None:
    task = dataset.HubTask(name="deadbeefdeadbeef", content_hash="0" * 64)
    assert task.archive_path == f"packages/orca-bench/deadbeefdeadbeef/{'0' * 64}/dist.tar.gz"


# ----- index -----


def test_index_is_ordered_by_name(dataset_dir: Path) -> None:
    refs = tasks.load_index(dataset_dir)
    names = [ref.name for ref in refs]
    assert names == sorted(names)
    assert [ref.dataset_index for ref in refs] == list(range(len(refs)))


def test_slicing_preserves_the_canonical_dataset_index(dataset_dir: Path) -> None:
    # A slice keeps each ref's position in the full dataset, which is exactly why that number is
    # provenance and not a selector: it no longer matches the position in the sliced list.
    refs = tasks.load_index(dataset_dir)
    sliced = tasks.select(refs, difficulty="easy")
    assert [ref.dataset_index for ref in sliced] == [0, 3]
    assert [ref.name for ref in sliced] == [synth.TASKS[0].name, synth.TASKS[3].name]


def test_index_reads_the_labels_a_caller_slices_on(dataset_dir: Path) -> None:
    refs = {ref.name: ref for ref in tasks.load_index(dataset_dir)}
    for task in synth.TASKS:
        ref = refs[task.name]
        assert ref.difficulty == task.difficulty
        assert ref.section == task.section
        assert ref.snapshot == task.snapshot
        # Upstream's definition: a control task is one with no incident to find.
        assert ref.is_control is task.is_control
        assert ref.is_control is (ref.section == "control")


def test_difficulty_ladder_drift_is_refused(tmp_path: Path) -> None:
    # Reading `granularity` instead of `difficulty` (the ladders overlap on "hard") would
    # silently redefine every per-tier number, so an unexpected ladder is an error, not a shrug.
    root = synth.write_dataset(tmp_path / "orca")
    toml_path = root / synth.TASKS[0].name / "task.toml"
    toml_path.write_text(
        toml_path.read_text(encoding="utf-8").replace('difficulty = "easy"', 'difficulty = "universal"'),
        encoding="utf-8",
    )
    with pytest.raises(tasks.TaskIndexError, match="difficulty"):
        tasks.load_index(root)


def test_missing_snapshot_is_refused(tmp_path: Path) -> None:
    root = synth.write_dataset(tmp_path / "orca")
    (root / synth.TASKS[0].name / "environment" / "Dockerfile").write_text(
        f"FROM {synth.BASE_IMAGE}\n", encoding="utf-8"
    )
    with pytest.raises(tasks.TaskIndexError, match="SNAPSHOT_NAME"):
        tasks.load_index(root)


def test_slicing_and_snapshot_grouping(dataset_dir: Path) -> None:
    refs = tasks.load_index(dataset_dir)
    assert [r.name for r in tasks.select(refs, difficulty="easy")] == [
        synth.TASKS[0].name,
        synth.TASKS[3].name,
    ]
    assert [r.name for r in tasks.select(refs, is_control=True)] == [synth.TASKS[3].name]
    assert len(tasks.select(refs, is_control=False)) == 3
    assert tasks.select(refs, section="exact_range")[0].name == synth.TASKS[1].name
    # Staging a snapshot is the expensive part of a run; the groups are what a runner walks.
    groups = tasks.group_by_snapshot(refs)
    assert sorted(groups) == sorted({synth.SNAPSHOT_A, synth.SNAPSHOT_B})
    assert [r.name for r in groups[synth.SNAPSHOT_A]] == [
        synth.TASKS[0].name,
        synth.TASKS[1].name,
    ]


def test_answer_strings_name_the_ground_truth_but_not_the_prompt(dataset_dir: Path) -> None:
    task = synth.TASKS[0]
    answers = tasks.answer_strings(dataset_dir / task.name)
    assert task.flag in answers
    assert task.events[0].event_time in answers
    assert task.qid in answers
    # Not secrets: the benchmark hands these to the agent in instruction.md on purpose, and
    # claiming otherwise would make the redaction test assert something false.
    assert task.complaint not in answers
    assert not any(task.reported in a for a in answers)


def test_env_exposes_the_index(dataset_dir: Path) -> None:
    env = shogym.make("orca_bench", config={"dataset_dir": str(dataset_dir)})
    assert env.num_tasks == len(synth.TASKS)
    assert [ref.name for ref in env.refs] == [task.name for task in synth.TASKS]
    loaded = env.load_task(1)
    assert loaded["task_name"] == synth.TASKS[1].name
    assert loaded["snapshot"] == synth.TASKS[1].snapshot
    assert loaded["is_control"] is False


def test_an_empty_dataset_dir_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="no tasks"):
        shogym.make("orca_bench", config={"dataset_dir": str(tmp_path / "empty")})
    with pytest.raises(tasks.TaskIndexError, match="no dataset directory"):
        tasks.load_index(tmp_path / "absent")


# ----- the pin authenticates the task set, not just the count -----


def test_the_pinned_manifest_is_the_revision(tmp_path: Path) -> None:
    manifest = dataset.pinned_manifest()
    assert len(manifest) == dataset.DATASET_TASK_COUNT == 755
    assert all(re.fullmatch(r"[0-9a-f]{16}", name) for name in manifest)
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in manifest.values())


def test_the_pinned_manifest_cannot_be_mutated_through_the_accessor() -> None:
    """The manifest is a process-global trust anchor, so handing out a mutable view of it makes
    an ordinary inspection (popping entries to derive a subset) a way to redefine what every
    later listing, warm-cache, authentication, residue and indexing check believes."""
    manifest = dataset.pinned_manifest()
    name = next(iter(manifest))
    expected = manifest[name]

    with pytest.raises(TypeError):
        manifest[name] = "0" * 64  # type: ignore[index]
    with pytest.raises((TypeError, AttributeError)):
        manifest.pop(name)  # type: ignore[attr-defined]
    with pytest.raises((TypeError, AttributeError)):
        manifest.clear()  # type: ignore[attr-defined]

    assert dataset.pinned_manifest()[name] == expected
    assert len(dataset.pinned_manifest()) == 755


def test_a_resolved_judges_environment_cannot_be_mutated() -> None:
    # Same rule one layer up: what the verifier is equipped with is also what the score is
    # audited against, so it is not a scratch dict for a caller to edit.
    from shogym.envs.orca_bench.judge import JudgeConfig

    resolved = JudgeConfig().resolve({"OPENAI_API_KEY": "sk-test"})
    with pytest.raises(TypeError):
        resolved.environment["OPENAI_API_KEY"] = "sk-other"  # type: ignore[index]
    assert resolved.environment["OPENAI_API_KEY"] == "sk-test"


def test_a_modified_manifest_file_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forged = tmp_path / "task_manifest.txt"
    original = dataset._MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
    original[0] = original[0].split(" ")[0] + " " + "0" * 64
    forged.write_text("\n".join(original) + "\n", encoding="utf-8")
    monkeypatch.setattr(dataset, "_MANIFEST_PATH", forged)
    dataset.pinned_manifest.cache_clear()
    try:
        with pytest.raises(dataset.DatasetUnavailableError, match="has been modified"):
            dataset.pinned_manifest()
    finally:
        dataset.pinned_manifest.cache_clear()


def _listing(monkeypatch: pytest.MonkeyPatch, rows: List[dataset.HubTask]) -> None:
    """Serve a raw listing, bypassing the hub, so the authentication is what is under test."""
    monkeypatch.setattr(dataset, "resolve_dataset_version", lambda: "version-id")
    pages = [[{"task_version": {"content_hash": row.content_hash,
                                "package": {"name": row.name}}} for row in rows]]

    def _hub_json(path: str, **_kwargs: object) -> object:
        return pages.pop(0) if pages else []

    monkeypatch.setattr(dataset, "_hub_json", _hub_json)


def test_a_duplicated_listing_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The count is not the benchmark. 755 copies of one row satisfies it, provisions a single
    directory, and would report success."""
    pinned = dataset.pinned_manifest()
    one = next(iter(pinned.items()))
    _listing(monkeypatch, [dataset.HubTask(name=one[0], content_hash=one[1])] * 755)
    with pytest.raises(dataset.DatasetUnavailableError, match="more than once"):
        dataset.list_dataset_tasks()


def test_a_listing_with_a_forged_hash_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every archive is authenticated against the hash in the listing, so a forged listing hash
    # would authenticate forged bytes. It is checked against the pin first.
    pinned = dataset.pinned_manifest()
    rows = [dataset.HubTask(name=name, content_hash=value) for name, value in pinned.items()]
    rows[7] = dataset.HubTask(name=rows[7].name, content_hash="0" * 64)
    _listing(monkeypatch, rows)
    with pytest.raises(dataset.DatasetUnavailableError, match="but the pin says"):
        dataset.list_dataset_tasks()


def test_a_listing_of_the_wrong_tasks_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    pinned = dataset.pinned_manifest()
    rows = [dataset.HubTask(name=name, content_hash=value) for name, value in pinned.items()]
    rows[3] = dataset.HubTask(name="deadbeefdeadbeef", content_hash=rows[3].content_hash)
    _listing(monkeypatch, rows)
    with pytest.raises(dataset.DatasetUnavailableError, match="different task set"):
        dataset.list_dataset_tasks()


def test_provisioning_asserts_the_tree_it_leaves_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"No worker raised" is not "the benchmark is here"."""
    root = tmp_path / "cache"
    cache = dataset.dataset_dir(root)
    present, absent = synth.TASKS[0], synth.TASKS[1]
    source = synth.write_task(tmp_path / "source" / present.name, present)
    _pin(monkeypatch, {present.name: dataset.content_hash(source), absent.name: "0" * 64})
    monkeypatch.setattr(
        dataset,
        "list_dataset_tasks",
        lambda: [dataset.HubTask(name=n, content_hash=h) for n, h in dataset.pinned_manifest().items()],
    )
    # A provisioner that quietly publishes only some of what it was asked for.
    monkeypatch.setattr(
        dataset,
        "_download_task",
        lambda task, dest: (
            synth.write_task(dest / task.name, present) if task.name == present.name else None
        ),
    )
    with pytest.raises(dataset.DatasetUnavailableError, match="absent or incomplete"):
        dataset.ensure_dataset(root=root)
    assert dataset._is_task_complete(cache, present.name)


def test_a_modified_cached_task_is_not_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache is warm when it holds the pinned bytes, not when it holds the pinned filenames.

    The cold path authenticates what it publishes and then never looks again, so any later change
    to an instruction, a compose file, a verifier or an expected answer would be served as the
    pinned revision forever. Phase 2 executes several of those files straight out of this cache.
    """
    root = tmp_path / "cache"
    cache = dataset.dataset_dir(root)
    synth.write_dataset(cache)
    _pin(monkeypatch, {task.name: dataset.content_hash(cache / task.name) for task in synth.TASKS})
    assert len(dataset._cached_task_names(cache)) == len(synth.TASKS)

    verifier = cache / synth.TASKS[0].name / "tests" / "check_prediction.py"
    verifier.write_text(verifier.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")

    assert dataset.modified_task_names(cache) == [synth.TASKS[0].name]
    assert not dataset._is_task_authentic(cache, synth.TASKS[0].name)
    assert dataset._is_task_complete(cache, synth.TASKS[0].name)  # the files are all still there
    assert dataset._cached_task_names(cache) == []


def test_a_modified_cached_task_is_repaired_like_an_incomplete_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Corruption is repaired, not refused: unlike residue, the pin says exactly what these bytes
    must be, so re-fetching is deterministic and cannot destroy anything a person authored."""
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    _fake_hub(monkeypatch, name, _tarball(source), content_hash=dataset.content_hash(source))

    root = tmp_path / "cache"
    task_dir = dataset.ensure_task(name, root=root)
    verifier = task_dir / "tests" / "check_prediction.py"
    verifier.write_text("raise SystemExit('tampered judge')\n", encoding="utf-8")
    assert not dataset._is_task_authentic(dataset.dataset_dir(root), name)

    assert dataset.ensure_task(name, root=root) == task_dir
    assert dataset._is_task_authentic(dataset.dataset_dir(root), name)
    assert "tampered" not in verifier.read_text(encoding="utf-8")


def test_provisioning_asserts_the_bytes_it_leaves_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The postcondition authenticates too: a publisher that writes the right filenames with the
    # wrong bytes does not finish a provision.
    root = tmp_path / "cache"
    wanted = synth.TASKS[0]
    source = synth.write_task(tmp_path / "source" / wanted.name, wanted)
    _pin(monkeypatch, {wanted.name: dataset.content_hash(source)})
    monkeypatch.setattr(
        dataset,
        "list_dataset_tasks",
        lambda: [dataset.HubTask(name=n, content_hash=h) for n, h in dataset.pinned_manifest().items()],
    )

    def _download(task: dataset.HubTask, dest: Path) -> None:
        written = synth.write_task(dest / task.name, wanted)
        (written / "instruction.md").write_text("not the pinned instruction\n", encoding="utf-8")

    monkeypatch.setattr(dataset, "_download_task", _download)
    with pytest.raises(dataset.DatasetUnavailableError, match="modified"):
        dataset.ensure_dataset(root=root)


def test_a_constrained_index_still_reports_canonical_positions(dataset_dir: Path) -> None:
    """``dataset_index`` means one thing everywhere: the position in the dataset's own order.

    Enumerating only the requested directories would number them 0, 1, 2 within the request, so a
    subset would claim canonical provenance it does not have. The number is taken from the full
    name-sorted order of the dataset directory and then filtered, which is the same number the
    unconstrained load reports."""
    canonical = {ref.name: ref.dataset_index for ref in tasks.load_index(dataset_dir)}
    wanted = [synth.TASKS[2].name, synth.TASKS[3].name]

    refs = tasks.load_index(dataset_dir, names=wanted)
    assert [ref.name for ref in refs] == sorted(wanted)
    assert [ref.dataset_index for ref in refs] == [canonical[name] for name in sorted(wanted)]
    assert [ref.dataset_index for ref in refs] != [0, 1]


def test_a_constrained_index_refuses_duplicate_identities(dataset_dir: Path) -> None:
    name = synth.TASKS[0].name
    with pytest.raises(tasks.TaskIndexError, match=name):
        tasks.load_index(dataset_dir, names=[name, name])


def test_a_pinned_cache_with_unpinned_residue_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every pinned task present and complete is not the same claim as "this is the benchmark".

    A leftover task-shaped directory from an older or interrupted provisioner is indexed like any
    other task: it changes ``num_tasks``, shifts every numeric id after it, and silently moves
    what a slice or a published result refers to. The cache has to be an exact set, not a
    superset.
    """
    root = tmp_path / "cache"
    cache = dataset.dataset_dir(root)
    synth.write_dataset(cache)
    _pin(monkeypatch, {task.name: dataset.content_hash(cache / task.name) for task in synth.TASKS})
    residue = cache / "zzzzzzzzzzzzzzzz"
    shutil.copytree(cache / synth.TASKS[0].name, residue)

    assert dataset.unpinned_task_dirs(cache) == ["zzzzzzzzzzzzzzzz"]
    assert dataset._cached_task_names(cache) == []  # not warm, so the cold path decides
    with pytest.raises(dataset.DatasetUnavailableError) as raised:
        dataset.ensure_dataset(root=root)

    message = str(raised.value)
    assert "zzzzzzzzzzzzzzzz" in message and str(cache) in message
    assert "remove" in message.lower()
    # Refused, never cleaned up behind the user's back: this might be their work, or evidence of
    # a bug worth keeping.
    assert residue.is_dir() and (residue / "task.toml").is_file()


def test_indexing_is_constrained_to_the_pinned_identities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Even handed a directory with residue in it, indexing the authenticated identities yields
    # the benchmark and nothing else, with ids that do not move.
    cache = tmp_path / "orca"
    synth.write_dataset(cache)
    shutil.copytree(cache / synth.TASKS[0].name, cache / "zzzzzzzzzzzzzzzz")
    names = [task.name for task in synth.TASKS]

    refs = tasks.load_index(cache, names=names)
    assert [ref.name for ref in refs] == sorted(names)
    assert [ref.dataset_index for ref in refs] == list(range(len(names)))
    # The unconstrained scan is what saw five.
    assert len(tasks.load_index(cache)) == len(names) + 1


def test_indexing_pinned_identities_refuses_a_missing_one(tmp_path: Path) -> None:
    cache = synth.write_dataset(tmp_path / "orca")
    with pytest.raises(tasks.TaskIndexError, match="absent"):
        tasks.load_index(cache, names=[synth.TASKS[0].name, "deadbeefdeadbeef"])


def test_residue_appearing_during_provisioning_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The postcondition is an exact set too, not just "the pinned ones arrived"."""
    root = tmp_path / "cache"
    wanted = synth.TASKS[0]
    source = synth.write_task(tmp_path / "source" / wanted.name, wanted)
    _pin(monkeypatch, {wanted.name: dataset.content_hash(source)})
    monkeypatch.setattr(
        dataset,
        "list_dataset_tasks",
        lambda: [dataset.HubTask(name=n, content_hash=h) for n, h in dataset.pinned_manifest().items()],
    )

    def _download(task: dataset.HubTask, dest: Path) -> None:
        synth.write_task(dest / task.name, wanted)
        synth.write_task(dest / "zzzzzzzzzzzzzzzz", wanted)  # a provisioner that leaves litter

    monkeypatch.setattr(dataset, "_download_task", _download)
    with pytest.raises(dataset.DatasetUnavailableError, match="zzzzzzzzzzzzzzzz"):
        dataset.ensure_dataset(root=root)


def test_a_cache_of_unpinned_tasks_is_not_warm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Complete directories under names the pin never mentions are not this benchmark, however
    # many of them there are.
    root = synth.write_dataset(tmp_path / "orca")
    _pin(monkeypatch, {f"other{index}": "0" * 64 for index in range(len(synth.TASKS))})
    assert dataset._cached_task_names(root) == []


def test_an_unpinned_task_is_never_fetched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin(monkeypatch, {"aaaa000000000001": "0" * 64})
    with pytest.raises(dataset.DatasetUnavailableError, match="is not a task of"):
        dataset.ensure_task("deadbeefdeadbeef", root=tmp_path / "cache")


# ----- fetching one task's archive -----


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_exc: object) -> bool:
        return False


def _fake_object_store(monkeypatch: pytest.MonkeyPatch, *, missing: bool = False) -> List[str]:
    """Serve (or 404) the object store, recording the URLs actually requested."""
    requested: List[str] = []

    def _urlopen(request, timeout=None):  # noqa: ANN001 - urllib's own loose signature
        url = request.full_url
        requested.append(url)
        if missing:
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)  # type: ignore[arg-type]
        return _FakeResponse(b"ARCHIVE")

    monkeypatch.setattr(dataset.urllib.request, "urlopen", _urlopen)
    return requested


def test_the_archive_fetch_does_not_wait_on_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The deterministic, content-addressed URL is tried first and on its own.

    The registry call is a fallback for a layout change, so a transiently unavailable registry
    must not stop a fetch that would have worked: a cold 755-task provision would otherwise fail
    without ever issuing a single valid archive request (and pay 755 needless registry calls when
    it did work)."""
    requested = _fake_object_store(monkeypatch)

    def _resolver_must_not_run(_task: dataset.HubTask) -> str:
        raise dataset.DatasetUnavailableError("registry RPC is transiently unavailable")

    monkeypatch.setattr(dataset, "_resolved_archive_path", _resolver_must_not_run)
    task = dataset.HubTask(name="deadbeefdeadbeef", content_hash="0" * 64)
    assert dataset._archive_bytes(task) == b"ARCHIVE"
    assert requested == [f"{dataset.HUB_URL}/storage/v1/object/packages/{task.archive_path}"]


def test_the_registry_is_asked_only_after_a_404(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = _fake_object_store(monkeypatch, missing=True)
    calls: List[str] = []

    def _resolver(task: dataset.HubTask) -> str:
        calls.append(task.name)
        return f"packages/moved/{task.name}/dist.tar.gz"

    monkeypatch.setattr(dataset, "_resolved_archive_path", _resolver)
    task = dataset.HubTask(name="deadbeefdeadbeef", content_hash="0" * 64)
    with pytest.raises(dataset.DatasetUnavailableError, match="no archive published"):
        dataset._archive_bytes(task)
    # Exactly one resolver call, and the relocated path really was tried before giving up.
    assert calls == [task.name]
    assert len(requested) == 2
    assert requested[1].endswith("packages/moved/deadbeefdeadbeef/dist.tar.gz")


# ----- the pin is cryptographic, not just a URL -----


def test_the_content_hash_covers_every_published_byte_and_its_path(tmp_path: Path) -> None:
    """The registry's own algorithm: sha256 over ``<relpath>\\0<sha256 of the file>\\n`` for each
    collected file, in path order. What matters here is that it is sensitive to everything that
    ships: any byte of any file, and where that file sits."""
    task_dir = synth.write_task(tmp_path / synth.TASKS[0].name, synth.TASKS[0])
    original = dataset.content_hash(task_dir)
    assert re.fullmatch(r"[0-9a-f]{64}", original)
    assert dataset.content_hash(task_dir) == original  # stable

    judge_file = task_dir / "tests" / "check_prediction.py"
    body = judge_file.read_text(encoding="utf-8")
    judge_file.write_text(body + "\nraise SystemExit('tampered judge')\n", encoding="utf-8")
    assert dataset.content_hash(task_dir) != original
    judge_file.write_text(body, encoding="utf-8")
    assert dataset.content_hash(task_dir) == original

    # The path is hashed alongside the bytes, so moving a file is a different package.
    (task_dir / "tests" / "renamed.py").write_bytes(judge_file.read_bytes())
    judge_file.unlink()
    assert dataset.content_hash(task_dir) != original


def test_the_content_hash_ignores_what_the_publisher_ignores(tmp_path: Path) -> None:
    task_dir = synth.write_task(tmp_path / synth.TASKS[0].name, synth.TASKS[0])
    original = dataset.content_hash(task_dir)
    (task_dir / "tests" / "__pycache__").mkdir()
    (task_dir / "tests" / "__pycache__" / "check_prediction.cpython-312.pyc").write_bytes(b"x")
    (task_dir / "tests" / "stray.pyc").write_bytes(b"x")
    (task_dir / ".DS_Store").write_bytes(b"x")
    assert dataset.content_hash(task_dir) == original
    # ...but a real file in a collected directory is not ignored.
    (task_dir / "tests" / "extra.json").write_text("{}", encoding="utf-8")
    assert dataset.content_hash(task_dir) != original


def test_a_tampered_archive_is_refused_and_publishes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin has to authenticate the bytes, not merely address them.

    Phase 2 executes what this cache holds: the compose file, `tests/test.sh`, and the judge. An
    archive served under a task's pinned hash whose judge has been replaced must never reach the
    cache, however complete it looks."""
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    pinned = dataset.content_hash(source)
    (source / "tests" / "check_prediction.py").write_text(
        "raise SystemExit('tampered judge')\n", encoding="utf-8"
    )
    tampered = dataset.content_hash(source)
    _fake_hub(monkeypatch, name, _tarball(source), content_hash=pinned)

    root = tmp_path / "cache"
    with pytest.raises(dataset.DatasetUnavailableError) as raised:
        dataset.ensure_task(name, root=root)
    message = str(raised.value)
    assert name in message and pinned in message and tampered in message
    assert not (dataset.dataset_dir(root) / name).exists()


# ----- what counts as a cached task -----


def _tarball(task_dir: Path, *, only: Optional[List[str]] = None) -> bytes:
    """A task package archive, flat at its root exactly as the hub publishes them."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for path in sorted(task_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(task_dir).as_posix()
            if only is not None and rel not in only:
                continue
            tf.add(path, arcname=rel)
    return buf.getvalue()


def _pin(monkeypatch: pytest.MonkeyPatch, manifest: Dict[str, str]) -> None:
    """Stand in for the committed manifest, so a test can pin a synthetic dataset."""
    monkeypatch.setattr(dataset, "pinned_manifest", lambda: dict(manifest))
    monkeypatch.setattr(dataset, "DATASET_TASK_COUNT", len(manifest))


def _fake_hub(
    monkeypatch: pytest.MonkeyPatch, name: str, archive: bytes, *, content_hash: str
) -> None:
    """Serve one task's listing + archive, so provisioning runs with no network.

    ``content_hash`` is what the registry claims for this task, which is what the fetched bytes
    are authenticated against, and what the pin says it should be."""
    _pin(monkeypatch, {name: content_hash})
    monkeypatch.setattr(
        dataset,
        "list_dataset_tasks",
        lambda: [dataset.HubTask(name=name, content_hash=content_hash)],
    )

    def _urlopen(request, timeout=None):  # noqa: ANN001 - urllib's own loose signature
        return _FakeResponse(archive)

    monkeypatch.setattr(dataset.urllib.request, "urlopen", _urlopen)


def test_the_required_files_are_the_ones_the_port_reads(dataset_dir: Path) -> None:
    task_dir = dataset_dir / synth.TASKS[0].name
    assert dataset.missing_task_files(task_dir) == []
    # The set is the port's contract with a cached task, across both phases: what the index and
    # `describe` read now, plus the verifier files a graded episode will need.
    assert set(dataset.REQUIRED_TASK_FILES) == {
        "task.toml",
        "instruction.md",
        "environment/Dockerfile",
        "environment/docker-compose.yaml",
        "tests/test.sh",
        "tests/check_prediction.py",
        "tests/expected.json",
    }


@pytest.mark.parametrize("required", dataset.REQUIRED_TASK_FILES)
def test_a_task_missing_any_required_file_is_not_complete(dataset_dir: Path, required: str) -> None:
    """``task.toml`` alone is not a task. Trusting a partial tree hides the damage until
    something downstream reads the missing file, which for the verifier files is phase 2."""
    task_dir = dataset_dir / synth.TASKS[0].name
    (task_dir / required).unlink()
    assert dataset.missing_task_files(task_dir) == [required]
    assert not dataset._is_task_complete(dataset_dir, synth.TASKS[0].name)


def test_the_warm_path_repairs_an_incomplete_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A damaged cached task is re-fetched, not returned as warm and not raised on."""
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    _fake_hub(monkeypatch, name, _tarball(source), content_hash=dataset.content_hash(source))

    root = tmp_path / "cache"
    task_dir = dataset.ensure_task(name, root=root)
    assert dataset.missing_task_files(task_dir) == []

    (task_dir / "instruction.md").unlink()
    assert dataset.ensure_task(name, root=root) == task_dir
    assert (task_dir / "instruction.md").is_file()
    assert tasks.load_ref(task_dir, 0).instructions()


def test_a_malformed_archive_is_never_published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Authentic but inadequate: the package really is what the registry pinned, and it still does
    # not carry what the port reads. That is what the completeness check is for, so the archive
    # here is served under the hash of its own (truncated) contents.
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    truncated = tmp_path / "truncated" / name
    truncated.mkdir(parents=True)
    (truncated / "task.toml").write_bytes((source / "task.toml").read_bytes())
    _fake_hub(
        monkeypatch,
        name,
        _tarball(truncated),
        content_hash=dataset.content_hash(truncated),
    )

    root = tmp_path / "cache"
    with pytest.raises(dataset.DatasetUnavailableError, match="instruction.md"):
        dataset.ensure_task(name, root=root)
    # Nothing half-good is left behind for the next run to trust.
    assert not (dataset.dataset_dir(root) / name).exists()


def test_a_concurrent_publisher_is_accepted_not_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two provisioners publishing one task must be redundant, not destructive.

    ``_locked`` degrades to no inter-process exclusion on a filesystem that cannot flock, and its
    contract is that concurrent cold starts stay redundant-but-correct there. So both callers can
    be inside ``_download_task`` for the same task, and the one that loses the race must accept
    the winner's complete tree rather than delete it out from under a reader that is already
    holding the path the winner returned.
    """
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    archive = _tarball(source)
    cache = dataset.dataset_dir(tmp_path / "cache")
    cache.mkdir(parents=True)
    task = dataset.HubTask(name=name, content_hash=dataset.content_hash(source))
    _pin(monkeypatch, {name: task.content_hash})

    order = itertools.count()
    order_lock = threading.Lock()
    release_loser = threading.Event()

    def _archive_bytes(_task: dataset.HubTask) -> bytes:
        with order_lock:
            nth = next(order)
        if nth > 0:
            # The loser is held past the point of no return: it has committed to publishing, and
            # is released only once the winner has published and a reader has acted on it.
            assert release_loser.wait(timeout=30)
        return archive

    monkeypatch.setattr(dataset, "_archive_bytes", _archive_bytes)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dataset._download_task, task, cache) for _ in range(2)]
        deadline = time.monotonic() + 30
        while not dataset._is_task_complete(cache, name):
            assert time.monotonic() < deadline, "the winner never published"
            time.sleep(0.01)
        # A reader holding the winner's returned task_dir, doing something with it.
        sentinel = cache / name / "a-reader-was-here"
        sentinel.write_text("written after the winner published", encoding="utf-8")
        release_loser.set()
        for future in futures:
            future.result()  # neither call fails

    assert dataset.missing_task_files(cache / name) == []
    assert sentinel.is_file(), "the loser deleted the winner's complete published tree"


# How long a repairer is held between deciding and acting. A correct implementation makes that
# interleaving impossible (the peer cannot publish while this one holds the decision), so the hold
# is a timeout rather than an assertion, and the property under test holds either way.
_DECISION_HOLD_SECONDS = 1.5


def _plant_lock(cache: Path, name: str, *, host: str, pid: int) -> Path:
    """A lock directory as some other holder would have left it."""
    lock = cache / f".publish-{name}"
    lock.mkdir(parents=True)
    (lock / "planted-token").write_text(
        json.dumps({"host": host, "pid": pid, "start": None}), encoding="utf-8"
    )
    return lock


def _dead_pid() -> int:
    """A pid that is not running: a child started, exited, and reaped, so it is provably gone."""
    child = subprocess.Popen([sys.executable, "-c", ""])
    child.wait(timeout=30)
    return child.pid


def test_a_live_holder_is_never_broken(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A lease cannot fence a holder that is still running.

    The owner token made an expired holder's *cleanup* safe, but its critical-section *body* is
    not something a wall-clock deadline can stop: a filesystem operation may stall past any lease,
    and the holder then resumes and mutates the destination on an observation it made before its
    successor published. So the lease no longer decides anything. A waiter waits, and if the
    holder is alive when its patience runs out, the waiter fails closed instead of taking a lock
    that is still in use.
    """
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    name = "aaaa000000000001"
    lock = cache / f".publish-{name}"
    monkeypatch.setattr(dataset, "_PUBLISH_LOCK_TIMEOUT_SECONDS", 0.2)

    holder = dataset._publish_lock(cache, name)
    holder.__enter__()
    owner_token = {entry.name for entry in lock.iterdir()}
    try:
        with pytest.raises(dataset.DatasetUnavailableError) as raised:
            with dataset._publish_lock(cache, name):
                pass  # pragma: no cover - acquiring here would be the bug
    finally:
        holder.__exit__(None, None, None)

    message = str(raised.value)
    assert str(lock) in message
    assert str(os.getpid()) in message and socket.gethostname() in message
    assert "still running" in message or "still holds" in message
    # The live holder's lock was left exactly as it was.
    assert {entry.name for entry in lock.iterdir()} == owner_token if lock.exists() else True


def test_a_provably_dead_holder_is_broken(tmp_path: Path) -> None:
    # The one case where taking someone else's lock is sound: same host, and that pid is gone.
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    name = "aaaa000000000001"
    lock = _plant_lock(cache, name, host=socket.gethostname(), pid=_dead_pid())
    stale_token = next(lock.iterdir()).name

    with dataset._publish_lock(cache, name):
        held = {entry.name for entry in lock.iterdir()}
        assert held and stale_token not in held  # broken and re-acquired, not adopted
    assert not lock.exists()
    assert list(cache.iterdir()) == []  # the corpse is reclaimed too


def test_a_holder_on_another_host_is_never_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A pid means nothing across hosts, so a shared cache's foreign lock is never assumed dead,
    # however long it has been there.
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    name = "aaaa000000000001"
    _plant_lock(cache, name, host="some-other-host", pid=_dead_pid())
    monkeypatch.setattr(dataset, "_PUBLISH_LOCK_TIMEOUT_SECONDS", 0.2)

    with pytest.raises(dataset.DatasetUnavailableError, match="some-other-host"):
        with dataset._publish_lock(cache, name):
            pass  # pragma: no cover - acquiring here would be the bug


def test_an_unreadable_lock_is_never_broken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    name = "aaaa000000000001"
    lock = cache / f".publish-{name}"
    lock.mkdir(parents=True)
    (lock / "token").write_text("not json", encoding="utf-8")
    monkeypatch.setattr(dataset, "_PUBLISH_LOCK_TIMEOUT_SECONDS", 0.2)

    with pytest.raises(dataset.DatasetUnavailableError, match="unknown"):
        with dataset._publish_lock(cache, name):
            pass  # pragma: no cover - acquiring here would be the bug


def test_a_stale_proof_never_evicts_the_holder_that_replaced_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Recovering a dead holder's lock has to name the generation that was proved dead.

    Two waiters read the same dead holder and both prove it gone. One clears it and acquires; the
    other is still carrying a proof about a generation that no longer occupies the path. If its
    recovery acts on the path rather than on that generation, it evicts a live successor and walks
    into the critical section beside it, which is the concurrent repair the lock exists to prevent.
    """
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    name = "aaaa000000000001"
    lock = cache / f".publish-{name}"
    _plant_lock(cache, name, host=socket.gethostname(), pid=_dead_pid())

    proofs = itertools.count()
    proof_lock = threading.Lock()
    both_proved = threading.Barrier(2, timeout=30)
    first_entered = threading.Event()
    prove = dataset._holder_is_provably_dead

    def _prove_then_wait(holder: object) -> bool:
        dead = prove(holder)  # type: ignore[arg-type]
        if not dead:
            return dead
        with proof_lock:
            nth = next(proofs)
        if nth < 2:
            # Neither prover acts until both hold a proof about the SAME generation, which is the
            # only way the second one's proof can be stale by the time it is spent.
            both_proved.wait()
        if nth == 1:
            # The second prover resumes once the first has cleared that generation and is inside.
            assert first_entered.wait(timeout=30)
        return dead

    monkeypatch.setattr(dataset, "_holder_is_provably_dead", _prove_then_wait)

    inside = threading.Lock()
    live = [0]
    overlap: List[bool] = []
    entered: List[str] = []
    undisturbed: List[bool] = []

    def _publisher() -> None:
        with dataset._publish_lock(cache, name):
            held_at_entry = {entry.name for entry in lock.iterdir()}
            with inside:
                live[0] += 1
                if live[0] > 1:
                    overlap.append(True)
            first_entered.set()
            time.sleep(0.2)
            undisturbed.append({entry.name for entry in lock.iterdir()} == held_at_entry)
            with inside:
                live[0] -= 1
            entered.append("in")

    threads = [threading.Thread(target=_publisher) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert not overlap, "a stale proof evicted the holder that replaced the generation it proved"
    assert entered == ["in", "in"], "a publisher never acquired the lock"
    assert all(undisturbed), "a holder's lock was replaced underneath it"
    assert not lock.exists()
    assert list(cache.iterdir()) == []


def test_two_breakers_of_one_dead_lock_do_not_both_enter(tmp_path: Path) -> None:
    """Both prove the same holder dead and both try to break it. The rename decides."""
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    name = "aaaa000000000001"
    _plant_lock(cache, name, host=socket.gethostname(), pid=_dead_pid())

    inside = threading.Semaphore(0)
    overlap = []
    entered = []
    concurrent = threading.Lock()
    live = [0]

    def _breaker() -> None:
        with dataset._publish_lock(cache, name):
            with concurrent:
                live[0] += 1
                if live[0] > 1:
                    overlap.append(True)
            entered.append(True)
            inside.release()
            time.sleep(0.05)
            with concurrent:
                live[0] -= 1

    threads = [threading.Thread(target=_breaker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
    assert entered == [True, True], "one breaker never acquired"
    assert not overlap, "both breakers were inside the critical section at once"
    assert not (cache / f".publish-{name}").exists()
    assert list(cache.iterdir()) == []


def test_a_displaced_holders_cleanup_leaves_its_successor_alone(tmp_path: Path) -> None:
    """The round-five property, restated for the only path that can now displace a lock.

    A died holding the lock; B proved it dead, broke it, and is inside. A's cleanup (which in
    reality never runs, since A is dead, but a resurrected or forked-off caller could try) must
    not free B's lock."""
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    name = "aaaa000000000001"
    lock = _plant_lock(cache, name, host=socket.gethostname(), pid=_dead_pid())
    stale_token = next(lock.iterdir()).name

    holder_b = dataset._publish_lock(cache, name)
    holder_b.__enter__()
    dataset._release_publish_lock(lock, stale_token)  # the dead holder's cleanup
    assert lock.is_dir() and any(lock.iterdir()), "the dead holder's cleanup freed B's lock"

    entered = threading.Event()

    def _c() -> None:
        with dataset._publish_lock(cache, name):
            entered.set()

    third = threading.Thread(target=_c)
    third.start()
    try:
        assert not entered.wait(timeout=0.15), "a third holder entered while the second was inside"
        holder_b.__exit__(None, None, None)
        assert entered.wait(timeout=30), "the lock was never released"
    finally:
        third.join(timeout=30)
    assert not lock.exists()


def test_the_publish_lock_leaves_nothing_behind(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    cache.mkdir(parents=True)
    with dataset._publish_lock(cache, "aaaa000000000001"):
        assert (cache / ".publish-aaaa000000000001").is_dir()
    assert list(cache.iterdir()) == []


def test_a_stalled_publisher_is_waited_out_not_overtaken(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reviewer's end-to-end probe, at the level a reader actually sees.

    A holds the publish lock, decides the damaged destination needs replacing, and stalls there
    for longer than any lease. B must not take the lock and repair the task underneath it: if it
    does, A resumes with a decision made before B published and displaces the very tree B handed
    out. B waiting, or failing closed, costs a provision. B overtaking costs a reader its files.
    """
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    cache = dataset.dataset_dir(tmp_path / "cache")
    damaged = cache / name
    damaged.mkdir(parents=True)
    (damaged / "task.toml").write_text("truncated", encoding="utf-8")
    task = dataset.HubTask(name=name, content_hash=dataset.content_hash(source))
    _pin(monkeypatch, {name: task.content_hash})

    monkeypatch.setattr(dataset, "_archive_bytes", lambda _task: _tarball(source))
    monkeypatch.setattr(dataset, "_PUBLISH_LOCK_TIMEOUT_SECONDS", 0.2)
    decided = threading.Event()
    stall = threading.Lock()
    stalled = []
    observe = dataset._is_task_complete

    def _observe_then_stall(cache_dir: Path, task_name: str) -> bool:
        complete = observe(cache_dir, task_name)
        with stall:
            first = not complete and not stalled
            if first:
                stalled.append(True)
        if first:
            decided.set()
            time.sleep(0.6)  # far past the lease, still inside the critical section
        return complete

    monkeypatch.setattr(dataset, "_is_task_complete", _observe_then_stall)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dataset._download_task, task, cache) for _ in range(2)]
        assert decided.wait(timeout=30)
        deadline = time.monotonic() + 60
        while not observe(cache, name):
            assert time.monotonic() < deadline, "no publisher ever completed the task"
            time.sleep(0.01)
        sentinel = cache / name / "a-reader-was-here"
        sentinel.write_text("written after a publisher completed", encoding="utf-8")
        outcomes = [future.exception() for future in futures]

    assert sentinel.is_file(), "a stalled publisher overtook the tree a reader was handed"
    assert dataset.missing_task_files(cache / name) == []
    # The waiter failed closed rather than taking a lock that was still in use.
    refused = [error for error in outcomes if isinstance(error, dataset.DatasetUnavailableError)]
    assert len(refused) == 1 and "still holds the publish lock" in str(refused[0])


def test_two_concurrent_repairs_do_not_undo_each_other(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same guarantee for the repair path, which starts from an *incomplete* destination.

    Both repairers observe the damaged tree and decide to replace it, so whichever acts second is
    acting on something it learned before its peer's tree existed. Deciding and replacing have to
    happen together: re-checking just before the rename only narrows the window it is checking
    for. The invariant asserted here does not depend on who wins: whatever tree first became
    complete was handed to a reader, so the reader's file has to still be there at the end.
    """
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    cache = dataset.dataset_dir(tmp_path / "cache")
    damaged = cache / name
    damaged.mkdir(parents=True)
    (damaged / "task.toml").write_text("truncated", encoding="utf-8")
    task = dataset.HubTask(name=name, content_hash=dataset.content_hash(source))
    _pin(monkeypatch, {name: task.content_hash})

    monkeypatch.setattr(dataset, "_archive_bytes", lambda _task: _tarball(source))
    decided_to_replace = threading.Event()
    may_proceed = threading.Event()
    held = threading.Lock()
    already_held = []
    observe = dataset._is_task_complete

    def _observe_then_hold(cache_dir: Path, task_name: str) -> bool:
        complete = observe(cache_dir, task_name)
        with held:
            first = not complete and not already_held
            if first:
                already_held.append(True)
        if first:
            # Parked between deciding to replace and replacing.
            decided_to_replace.set()
            may_proceed.wait(timeout=_DECISION_HOLD_SECONDS)
        return complete

    monkeypatch.setattr(dataset, "_is_task_complete", _observe_then_hold)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(dataset._download_task, task, cache) for _ in range(2)]
        assert decided_to_replace.wait(timeout=30)
        deadline = time.monotonic() + _DECISION_HOLD_SECONDS + 30
        while not observe(cache, name):
            assert time.monotonic() < deadline, "no repair ever published"
            time.sleep(0.01)
        sentinel = cache / name / "a-reader-was-here"
        sentinel.write_text("written after a repair published", encoding="utf-8")
        may_proceed.set()
        for future in futures:
            future.result()

    assert dataset.missing_task_files(cache / name) == []
    assert sentinel.is_file(), "a repair undid the published tree a reader was handed"


def test_an_incomplete_destination_is_still_repaired(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The destructive path stays available where it is the right answer: a destination that is
    # actually incomplete is replaced, not accepted.
    name = synth.TASKS[0].name
    source = synth.write_task(tmp_path / "source" / name, synth.TASKS[0])
    cache = dataset.dataset_dir(tmp_path / "cache")
    damaged = cache / name
    damaged.mkdir(parents=True)
    (damaged / "task.toml").write_text("truncated", encoding="utf-8")
    (damaged / "stale-file").write_text("from the damaged tree", encoding="utf-8")
    monkeypatch.setattr(dataset, "_archive_bytes", lambda _task: _tarball(source))
    _pin(monkeypatch, {name: dataset.content_hash(source)})

    dataset._download_task(
        dataset.HubTask(name=name, content_hash=dataset.content_hash(source)), cache
    )
    assert dataset.missing_task_files(damaged) == []
    assert not (damaged / "stale-file").exists()


def test_a_peer_repair_is_honored_even_with_no_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's decision to fetch is re-taken under the lock, and holds when there is none.

    On a filesystem that cannot flock, ``_locked`` yields without exclusion, so a peer may have
    completed this very task between the caller's first check and its second. The second check is
    what makes that peer's work count: nothing is fetched and nothing is replaced.
    """

    @contextlib.contextmanager
    def _degraded(_directory: Path):
        yield  # what `_locked` does on a filesystem that cannot lock

    monkeypatch.setattr(dataset, "_locked", _degraded)
    fetches: List[str] = []

    def _must_not_fetch(task: dataset.HubTask) -> bytes:
        fetches.append(task.name)
        raise AssertionError("a complete task was re-fetched")

    monkeypatch.setattr(dataset, "_archive_bytes", _must_not_fetch)

    name = synth.TASKS[0].name
    cache = dataset.dataset_dir(tmp_path / "cache")
    synth.write_task(cache / name, synth.TASKS[0])
    _pin(monkeypatch, {name: dataset.content_hash(cache / name)})
    assert dataset.ensure_task(name, root=tmp_path / "cache") == cache / name
    assert fetches == []


def test_an_incomplete_cache_is_not_a_warm_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = synth.write_dataset(tmp_path / "orca")
    _pin(monkeypatch, {task.name: dataset.content_hash(root / task.name) for task in synth.TASKS})
    assert len(dataset._cached_task_names(root)) == len(synth.TASKS)
    (root / synth.TASKS[0].name / "tests" / "expected.json").unlink()
    # The all-or-nothing warm check counts complete tasks only, so provisioning re-fetches this
    # one instead of indexing a dataset that will fail later.
    assert dataset._cached_task_names(root) == []


def test_unreachable_hub_is_an_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # A cold cache with no network must say what it wanted and where it caches it, not unwind a
    # urllib traceback into the caller.
    def _boom(*_args, **_kwargs):
        raise OSError("Network is unreachable")

    monkeypatch.setattr(dataset.urllib.request, "urlopen", _boom)
    with pytest.raises(dataset.DatasetUnavailableError, match="downloaded on demand"):
        dataset.resolve_dataset_version()


# ----- the one network test -----


@pytest.mark.network
def test_one_real_task_downloads_and_indexes(tmp_path: Path) -> None:
    """Fetch a single real task into the cache and re-check the loader + redaction on real bytes.

    Deliberately one task, not the 755-task dataset: this exercises the whole provisioning path
    (resolve the pinned revision, list it, fetch the archive, publish the directory) at ~30 KB.
    """
    listed = dataset.list_dataset_tasks()
    assert len(listed) == dataset.DATASET_TASK_COUNT
    name = sorted(t.name for t in listed)[0]

    task_dir = dataset.ensure_task(name, root=tmp_path)
    assert task_dir == dataset.dataset_dir(tmp_path) / name
    # The golden vector for the hash: the registry's own value for a real published task.
    pinned = {t.name: t.content_hash for t in listed}[name]
    assert dataset.content_hash(task_dir) == pinned
    assert (task_dir / "instruction.md").is_file()
    assert (task_dir / "task.toml").is_file()
    # Idempotent: a warm cache re-resolves to the same directory.
    assert dataset.ensure_task(name, root=tmp_path) == task_dir

    ref = tasks.load_ref(task_dir, 0)
    assert ref.difficulty in tasks.DIFFICULTIES
    assert ref.section in tasks.SECTIONS
    assert _SNAPSHOT_RE.fullmatch(ref.snapshot), ref.snapshot
    assert ref.is_control is (ref.section == "control")

    answers = tasks.answer_strings(task_dir)
    assert answers, "a real task with no ground truth would make this check vacuous"
    env = shogym.make("orca_bench", config={"tasks": [ref]})
    spec = env.describe(ref.name)
    assert spec.instructions.startswith(ref.instructions().rstrip())
    assert [a for a in answers if a in spec.model_dump_json()] == []
