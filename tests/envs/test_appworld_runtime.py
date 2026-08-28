"""The appworld port's runtime seams: provisioning order, concurrency, and what blocks the loop.

Offline and upstream-free. Every failure defended here was found by running the port rather than
by reading it, and each is the kind that looks like nothing until the day it matters: a cold CI
runner that hangs instead of provisioning, two forks of one clone deriving the same task at the
same moment, a finalizer that stops every other episode while it waits, a cache path that produces
a tree of links resolving to nothing.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shogym.envs.appworld import adapter, world


# ----- provisioning order -----


def test_provisioning_the_corpus_does_not_wait_on_a_lock_it_already_holds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A genuinely cold machine has to get past this, and it is CI's ordinary path.

    The interpreter and the corpus live under one cache directory and both were guarded by an
    ``flock`` on it. Two ``flock`` calls through two opens are two lock requests even inside one
    process, so provisioning the interpreter from inside the corpus's lock is a process waiting on
    itself, with no error and no timeout. The fix is an ordering, so the test is over the
    ordering: nothing may be locked while that same path is already held."""
    held: List[Path] = []

    class _recorder:
        def __init__(self, directory: Path, *, required: bool = False) -> None:
            self.directory = Path(directory)
            self.required = required

        def __enter__(self) -> None:
            assert self.directory not in held, f"{self.directory} locked while already held"
            held.append(self.directory)

        def __exit__(self, *exc: Any) -> None:
            held.remove(self.directory)

    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path))
    monkeypatch.delenv(adapter.ROOT_ENV_VAR, raising=False)
    monkeypatch.setattr(adapter, "_locked", _recorder)

    ordered: List[str] = []
    monkeypatch.setattr(adapter, "runtime", lambda: ordered.append("runtime") or Path("python"))
    monkeypatch.setattr(adapter, "ensure_apps", lambda: ordered.append("apps"))

    def _fetch(root: Path) -> None:
        ordered.append("corpus")
        (root / "data" / "tasks").mkdir(parents=True)

    monkeypatch.setattr(adapter, "_fetch_corpus", _fetch)
    adapter.ensure_corpus()
    # And the ordering is the one the fix is: the interpreter exists before the corpus is fetched.
    assert ordered == ["runtime", "apps", "corpus"]
    assert not held


def test_a_relative_cache_root_still_produces_links_that_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A derived corpus is a tree of symlinks, and a symlink's target is read relative to the
    link, not to the directory the run was launched from. A relative cache root therefore built a
    tree whose every link pointed at nothing, silently."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SHOGYM_CACHE", "cache")
    adapter._private_tag.cache_clear()
    assert adapter.cache_root().is_absolute()
    assert adapter.graded_root().is_absolute()

    original = tmp_path / "corpus" / "data"
    (original / "tasks").mkdir(parents=True)
    (original / "shared").write_text("base databases")
    derived = adapter.cache_root() / "derived" / "data"
    world.derive_root(original=original, derived=derived)
    materialised = derived / "shared"
    # No symlink to resolve at all now, which is the stronger form of the same guarantee.
    assert not materialised.is_symlink()
    assert materialised.exists() and materialised.read_text() == "base databases"


# ----- concurrency -----


def test_two_streams_deriving_one_cold_task_both_get_a_world(tmp_path: Path) -> None:
    """Paired forks are launched together and both derive on first use, so this is the ordinary
    case rather than a hypothetical. Before the lock, one would delete the other's staging tree or
    its published target and leave no world at all."""
    original = tmp_path / "corpus"
    task = original / "tasks" / "abc_1"
    (task / "dbs").mkdir(parents=True)
    (task / "specs.json").write_text("{}")
    (task / "ground_truth").mkdir()
    (task / "ground_truth" / "answer.json").write_text('"the answer"')
    (task / "dbs" / "todoist.jsonl").write_text("")
    (task / "dbs" / "gmail.jsonl").write_text("mail")

    derived = tmp_path / "derived"
    graded = tmp_path / "graded"
    for root in (derived, graded):
        (root / "tasks").mkdir(parents=True)

    written: List[int] = []

    def write_log(source: Path, into: Path) -> None:
        # Slow enough that the other thread is certainly inside `derive_task` while this one is
        # halfway through building, which is the window the bug lived in.
        time.sleep(0.2)
        written.append(1)
        into.write_text("seeded")

    failures: List[BaseException] = []
    results: List[Path] = []

    def derive() -> None:
        try:
            results.append(
                world.derive_task(
                    original=original,
                    derived=derived,
                    graded=graded,
                    task_id="abc_1",
                    write_log=write_log,
                )
            )
        except BaseException as exc:  # noqa: BLE001 (the point is that none escapes)
            failures.append(exc)

    threads = [threading.Thread(target=derive) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not failures, failures
    # Both callers get a world, exactly one of them built it, and no staging tree is left behind.
    assert len(results) == 2 and len(set(results)) == 1
    assert written == [1]
    assert (derived / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").read_text() == "seeded"
    assert (derived / "tasks" / "abc_1" / "dbs" / "gmail.jsonl").read_text() == "mail"
    assert not list((derived / "tasks").glob(".*building*"))


def test_a_write_through_the_served_tree_changes_nothing_else(tmp_path: Path) -> None:
    """Served inputs are copies, not links of any kind.

    A hard link removes the pathname that led back to the corpus and keeps the file: same inode,
    two names, and the worker runs as the user that made it. A write through the served name would
    then change the corpus every later episode is derived from and the baseline the grader diffs
    against, which is a served episode editing the thing it is scored on.

    The copies are half of it and the seal is the other half. The shared task is what every
    episode's view is built from, so it is read-only from the moment it is published; only the
    per-episode copy of it is writable."""
    original = tmp_path / "corpus"
    task = original / "tasks" / "abc_1"
    (task / "dbs").mkdir(parents=True)
    (task / "specs.json").write_text("{}")
    (task / "ground_truth").mkdir()
    (task / "ground_truth" / "answer.json").write_text('"the answer"')
    (task / "dbs" / "todoist.jsonl").write_text("")
    (task / "dbs" / "gmail.jsonl").write_text("mail")

    derived, graded = tmp_path / "derived", tmp_path / "graded"
    world.derive_task(
        original=original,
        derived=derived,
        graded=graded,
        task_id="abc_1",
        write_log=lambda source, into: into.write_text("seeded"),
    )
    shared = derived / "tasks" / "abc_1" / "dbs" / "gmail.jsonl"
    source = task / "dbs" / "gmail.jsonl"
    baseline = graded / "tasks" / "abc_1" / "dbs" / "gmail.jsonl"
    assert shared.stat().st_ino != source.stat().st_ino
    assert shared.stat().st_ino != baseline.stat().st_ino

    # The shared task is the pristine source every later episode's view is copied out of, so it
    # is sealed along with the rest of the derived tree: an episode that could write here would be
    # writing into what the next one, or the other arm of its own pair, starts from.
    for name in ("gmail.jsonl", "todoist.jsonl"):
        with pytest.raises(PermissionError):
            (derived / "tasks" / "abc_1" / "dbs" / name).write_text("rewritten by the agent")
    # And a name cannot be added or taken away either, which is the other half of owning a cache.
    with pytest.raises(PermissionError):
        (derived / "tasks" / "abc_1" / "dbs" / "planted.jsonl").write_text("hello")
    with pytest.raises(PermissionError):
        (derived / "tasks" / "planted_1").mkdir()

    # A write through the episode's own copy reaches nothing but itself, which is the property the
    # copies are for and which the seal alone would not give.
    view = world.derive_view(derived=derived, view=tmp_path / "a", task_id="abc_1")
    (view / "data" / "tasks" / "abc_1" / "dbs" / "gmail.jsonl").write_text("rewritten by the agent")
    assert source.read_text() == "mail"
    assert baseline.read_text() == "mail"
    assert shared.read_text() == "mail"
    # And the seeded log the episode is scored against is the grader's own copy too.
    assert (graded / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").read_text() == "seeded"
    world._unseal(derived)
    world._unseal(graded)


def test_nothing_in_a_served_task_names_where_it_came_from(tmp_path: Path) -> None:
    """The served tree is what the worker gets `APPWORLD_ROOT` pointing at, so any symlink in it
    is a path to the corpus, and the corpus has every task's answers one directory over."""
    original = tmp_path / "corpus"
    task = original / "tasks" / "abc_1"
    (task / "dbs").mkdir(parents=True)
    (task / "specs.json").write_text("{}")
    (task / "ground_truth").mkdir()
    (task / "ground_truth" / "answer.json").write_text('"the answer"')
    (task / "dbs" / "todoist.jsonl").write_text("")
    (task / "dbs" / "gmail.jsonl").write_text("mail")
    (original / "base_dbs").mkdir()
    (original / "base_dbs" / "admin.db").write_text("base")

    derived, graded = tmp_path / "derived", tmp_path / "graded"
    world.derive_root(original=original, derived=derived)
    world.derive_task(
        original=original,
        derived=derived,
        graded=graded,
        task_id="abc_1",
        write_log=lambda source, into: into.write_text("seeded"),
    )
    links = [p for p in derived.rglob("*") if p.is_symlink()]
    assert links == [], links
    # The content is really there, so this is not a tree of empty files.
    assert (derived / "tasks" / "abc_1" / "specs.json").read_text() == "{}"
    assert (derived / "base_dbs" / "admin.db").read_text() == "base"
    assert not (derived / "tasks" / "abc_1" / "ground_truth").exists()


def test_the_world_an_agent_is_given_carries_no_answers(tmp_path: Path) -> None:
    """The answers ship in the corpus in plaintext and agent-authored code runs with that corpus's
    root in its environment, so the directory holding them is simply not in the tree the world is
    served from. The grader gets its own view of the same task, with the same database files."""
    original = tmp_path / "corpus"
    task = original / "tasks" / "abc_1"
    (task / "dbs").mkdir(parents=True)
    (task / "specs.json").write_text("{}")
    (task / "ground_truth").mkdir()
    (task / "ground_truth" / "answer.json").write_text('"the answer"')
    (task / "dbs" / "todoist.jsonl").write_text("")

    derived, graded = tmp_path / "derived", tmp_path / "graded"
    for root in (derived, graded):
        (root / "tasks").mkdir(parents=True)
    world.derive_task(
        original=original,
        derived=derived,
        graded=graded,
        task_id="abc_1",
        write_log=lambda source, into: into.write_text("seeded"),
    )
    served = derived / "tasks" / "abc_1"
    assert not (served / "ground_truth").exists()
    assert "the answer" not in _read_tree(served)
    # The grader's view has them, and shares the one seeded database file rather than copying it.
    grader = graded / "tasks" / "abc_1"
    assert (grader / "ground_truth" / "answer.json").read_text() == '"the answer"'
    assert (grader / "dbs" / "todoist.jsonl").read_text() == "seeded"


def _read_tree(root: Path) -> str:
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                out.append(path.read_text())
            except UnicodeDecodeError:
                pass
    return "\n".join(out)


# ----- the event loop -----


async def test_finalizing_one_episode_does_not_stop_the_others() -> None:
    """``finalize`` makes two blocking calls into another process, one of which grades the base
    task. Made from the coroutine itself they stop every other episode this serving process is
    running, and the serve layer's deadline, which is an ``asyncio.wait_for`` around exactly this
    coroutine, cannot fire on the episode that is holding it."""
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    async def finalize_like() -> str:
        # What the env does: the blocking work goes to a thread, and the coroutine yields.
        await asyncio.to_thread(time.sleep, 0.3)
        await asyncio.to_thread(time.sleep, 0.3)
        return "scored"

    beat = asyncio.create_task(ticker())
    try:
        assert await finalize_like() == "scored"
    finally:
        beat.cancel()
    # A coroutine that blocked would leave this at one or two.
    assert ticks > 20, ticks


async def test_a_deadline_can_still_fire_on_a_finalizer_that_is_waiting() -> None:
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.to_thread(time.sleep, 5.0), timeout=0.05)


# ----- the published budget -----


def test_the_run_fingerprint_covers_everything_that_changes_what_a_score_means() -> None:
    """Two runs whose rows are meant to be one measurement have to agree on all of these. The
    digest is not agent-visible feedback: it is a short hash over a small integer pulse and
    otherwise public material, so an agent handed it could enumerate pulses until one matched and
    then compute every later key."""
    from shogym.envs.appworld.env_v1 import run_fingerprint

    base = run_fingerprint(pulse=0, report="graded", blocks=60)
    assert base == run_fingerprint(pulse=0, report="graded", blocks=60)
    assert base != run_fingerprint(pulse=1, report="graded", blocks=60)
    assert base != run_fingerprint(pulse=0, report="drawn", blocks=60)
    assert base != run_fingerprint(pulse=0, report="graded", blocks=61)
    assert len(base) == 16
    assert base != run_fingerprint(pulse=0, report="graded", blocks=60, corpus="a different one")
    # And it moves when the way a score is read moves, which no input of the run would show.
    from shogym.envs.appworld import adapter, env_v1, payload

    for module, name in (
        (env_v1, "SCORING_VERSION"),
        (adapter, "DERIVATION_VERSION"),
        # Every constant a published payload is generated from, and this is the one that was
        # missing. `DRAWN_BASIS` seeds the drawn arm's whole visible vector: changing it re-rolls
        # every drawn payload, which is a change to what the agent is told, and a record could
        # resume across it under an identity that had not moved.
        (payload, "DRAWN_BASIS"),
    ):
        original = getattr(module, name)
        try:
            setattr(module, name, f"{original}-moved")
            assert base != run_fingerprint(pulse=0, report="graded", blocks=60), name
        finally:
            setattr(module, name, original)


def test_what_is_recorded_and_never_revealed_stays_that_way() -> None:
    """Two halves, and both matter. A resumed directory is checked against what its rows say, so
    the fingerprint has to be on every row. It is a digest over a usually small integer pulse, so
    it must reach no agent: one handed it can enumerate pulses until it matches and then compute
    every later key.

    Inference level is exactly that contract, and it is the only channel an env has that satisfies
    both: `_revealable` filters a row's feedback to episode level, so even `Immediate`, which
    reveals everything else a row records, cannot reach an inference item."""
    import inspect

    from shogym.envs.appworld import env_v1
    from shogym.serve import stream as stream_module

    verify = inspect.getsource(env_v1.AppWorldEnv._verify)
    # On the row, at the level that is recorded rather than surfaced, and nowhere among the items
    # a terminating call can reveal: not as a named episode append, and not in the tuples of names
    # the loops build episode items out of.
    assert 'InferenceFeedback(name="config_digest"' in verify
    assert 'EpisodeFeedback(name="config_digest"' not in verify
    assert verify.count('"config_digest"') == verify.count('InferenceFeedback(name="config_digest"')
    # And off the terminal evidence, which a direct caller reads back verbatim.
    assert "config_digest" not in inspect.getsource(env_v1.AppWorldEnv.finalize)
    # The filter that makes the level mean what it is being relied on to mean: what a terminating
    # call can reveal is the row's episode-level items and nothing else.
    revealable = inspect.getsource(stream_module._revealable)
    assert "_EPISODE_LEVEL" in revealable
    assert stream_module._EPISODE_LEVEL == "episode"


def test_a_worker_environment_carries_nothing_it_was_not_given(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Agent-authored code runs as the worker, so everything the serving process exported is one
    ``os.environ`` away from it unless it is taken away first."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-also-secret")
    monkeypatch.setenv("SHOGYM_APPWORLD_PROV", "/runs/somewhere")
    scrubbed: Dict[str, str] = adapter._worker_environment(tmp_path)
    assert "ANTHROPIC_API_KEY" not in scrubbed
    assert "OPENAI_API_KEY" not in scrubbed
    assert "SHOGYM_APPWORLD_PROV" not in scrubbed
    assert scrubbed["HOME"] == str(tmp_path)
    assert set(scrubbed) <= set(adapter._ENV_ALLOW_LIST) | {"HOME", "APPWORLD_CACHE"}


def test_two_episodes_of_one_task_do_not_share_their_served_inputs(tmp_path: Path) -> None:
    """A write through one episode's served view is not in the next episode's starting inputs.

    The end-to-end version of this drives two real episodes; this is the same property at the
    level that decides it, so it runs everywhere and in a second. The derived corpus was one
    deterministic global root handed to every worker, writable by the process that runs
    agent-authored code, with nothing putting it back: episode A's write was still there when
    episode B started. Two arms of a pair are the same task served at the same time, so the arm
    meant to differ only in what it was told could also differ in the world it was given.
    """
    original = tmp_path / "corpus" / "data"
    (original / "tasks" / "abc_1" / "dbs").mkdir(parents=True)
    (original / "base_dbs").mkdir()
    (original / "base_dbs" / "big.jsonl").write_text("shared base")

    derived = world.derive_root(original=original, derived=tmp_path / "derived" / "data")
    (derived / "tasks" / "abc_1" / "dbs").mkdir(parents=True)
    (derived / "tasks" / "abc_1" / "dbs" / "gmail.jsonl").write_text("pristine")

    first = world.derive_view(derived=derived, view=tmp_path / "a", task_id="abc_1")
    second = world.derive_view(derived=derived, view=tmp_path / "b", task_id="abc_1")
    assert first != second

    served = first / "data" / "tasks" / "abc_1" / "dbs" / "gmail.jsonl"
    served.write_text("written by an earlier episode")

    assert (second / "data" / "tasks" / "abc_1" / "dbs" / "gmail.jsonl").read_text() == "pristine"
    assert (derived / "tasks" / "abc_1" / "dbs" / "gmail.jsonl").read_text() == "pristine"
    # The 129 MB of shared databases are named rather than copied, which is what makes a view
    # cheap enough to build per episode. Naming rather than copying is safe because the shared
    # base is sealed read-only: an episode reads it and cannot leave anything in it for the next
    # one, or for the other arm of its own pair. Both halves are asserted, because the link on its
    # own is what the previous head shipped and it is the writability that decides the property.
    assert (first / "data" / "base_dbs").is_symlink()
    shared = first / "data" / "base_dbs" / "big.jsonl"
    assert shared.read_text() == "shared base"
    with pytest.raises(PermissionError):
        shared.write_text("reaching into the next episode")
    with pytest.raises(PermissionError):
        (first / "data" / "base_dbs" / "planted.jsonl").write_text("hello, twin")
    assert (second / "data" / "base_dbs" / "big.jsonl").read_text() == "shared base"
    # Same-uid permissions, so this is a boundary against writing and not against a process that
    # sets out to defeat it; shojin-lab/shogym#140 mounts the base read-only in the container,
    # which is. Undone here so the temporary directory can be removed.
    world._unseal(derived)


def test_the_shared_parent_cannot_be_renamed_around(tmp_path: Path) -> None:
    """The other half of the same invariant, and the half sealing each entry does not give.

    A view names the shared entries by absolute path, so what an episode resolves is the entry's
    bytes *and* the name that reaches them — and a name lives in its parent. The previous head
    sealed every entry and left their parent owner-writable, so `base_dbs` could be renamed aside
    and a directory of the episode's own choosing put there under the same name; every view that
    resolved it afterwards, this episode's and the other arm of its pair's, would follow.
    """
    original = tmp_path / "corpus" / "data"
    (original / "tasks").mkdir(parents=True)
    (original / "base_dbs").mkdir()
    (original / "base_dbs" / "big.jsonl").write_text("shared base")
    (original / "version.txt").write_text("1.0")

    derived = world.derive_root(original=original, derived=tmp_path / "derived" / "data")
    (derived / "tasks" / "abc_1" / "dbs").mkdir(parents=True)
    (derived / "tasks" / "abc_1" / "dbs" / "gmail.jsonl").write_text("pristine")
    view = world.derive_view(derived=derived, view=tmp_path / "a", task_id="abc_1")

    # The links really are absolute paths into the shared parent, which is what makes the parent
    # part of what an episode resolves rather than an implementation detail above it.
    link = view / "data" / "base_dbs"
    assert link.is_symlink()
    assert os.readlink(link) == str(derived / "base_dbs")

    assert not (derived.lstat().st_mode & 0o222), oct(derived.lstat().st_mode)
    # A name cannot be moved aside, replaced, added or taken away. Each of these needs write
    # permission on the parent and none of them touches the entry's own mode, which is exactly why
    # the entry seal did not cover them.
    with pytest.raises(PermissionError):
        os.rename(derived / "base_dbs", derived / "moved_aside")
    with pytest.raises(PermissionError):
        (derived / "planted").mkdir()
    with pytest.raises(PermissionError):
        (derived / "version.txt").unlink()
    with pytest.raises(PermissionError):
        (derived / "swapped").symlink_to(tmp_path / "elsewhere")
    # And what the episode resolves is still what it was built from.
    assert (view / "data" / "base_dbs" / "big.jsonl").read_text() == "shared base"

    # The residual, stated by exercising it: the worker runs as the user that owns these files, so
    # a process that means to defeat the mode can put it back. This is a boundary against a rename
    # and not against an adversary; shojin-lab/shogym#140 mounts the shared base into the worker's
    # container read-only, which is a boundary rather than a convention. Two ancestors above this
    # one stay writable as well — the seeded root holds the port's cache stamp and the cache root
    # is where it provisions — so the name `data` itself is movable by a process willing to work a
    # level up, and the container mount is what closes that too.
    os.chmod(derived, 0o755)
    os.rename(derived / "base_dbs", derived / "moved_aside")
    assert (derived / "moved_aside" / "big.jsonl").read_text() == "shared base"
    world._unseal(derived)


# ----- stopping a worker, and stopping what it started -----


def _sleeper(seconds: float) -> subprocess.Popen:
    """A process of its own session, standing in for a worker or for a stranger."""
    return subprocess.Popen(
        [sys.executable, "-c", f"import time; time.sleep({seconds})"],
        start_new_session=True,
    )


def test_a_second_close_does_not_signal_whoever_holds_the_pid_now(tmp_path: Path) -> None:
    """An ordinary episode closes its worker twice, and the two are ten minutes apart.

    The first close happens when the sealed world has been read; the second comes from teardown.
    Between them runs the grader, which is allowed 600 seconds, and a pid released at the start of
    that window can belong to something else by the end of it. The second close used to resolve a
    process group by asking the kernel which group the stored pid was in, so it signalled whoever
    held the pid by then, and the comment beside it called the operation idempotent.

    The reuse is made explicit rather than waited for: the worker's process is given the pid a
    stranger now holds, which is exactly what the kernel does when it hands the number out again.
    """
    worker_process = _sleeper(30)
    worker = adapter.Worker(
        root=tmp_path,
        process=worker_process,
        port=0,
        token="unused",
        scratch=tmp_path / "scratch",
        pgid=adapter._group_of(worker_process),
    )
    (tmp_path / "scratch").mkdir()
    worker.close()
    assert worker_process.poll() is not None

    stranger = _sleeper(30)
    try:
        # The kernel hands the number out again.
        worker.process.pid = stranger.pid
        worker.close()
        time.sleep(0.2)
        assert stranger.poll() is None
    finally:
        stranger.kill()
        stranger.wait(timeout=10)


def test_a_worker_is_stopped_through_the_group_it_was_spawned_in() -> None:
    """The other half of the same fix, at the helper that does the signalling.

    The group is taken once, while the answer is still about this worker; the signal goes to that
    group and never to the stored pid. A `killpg` that finds an empty group means there is nothing
    left to stop, which used to fall through to `send_signal` on the pid: the one value that may
    since have been handed to somebody else."""
    worker_process = _sleeper(30)
    real_pid = worker_process.pid
    pgid = adapter._group_of(worker_process)
    assert pgid == real_pid
    stranger = _sleeper(30)
    try:
        # The kernel hands the number out again.
        worker_process.pid = stranger.pid
        adapter._stop(worker_process, signal.SIGTERM, pgid)
        time.sleep(0.3)
        # The stranger is untouched, and the worker's own group got the signal.
        assert stranger.poll() is None
        worker_process.pid = real_pid
        assert worker_process.wait(timeout=10) is not None
    finally:
        stranger.kill()
        stranger.wait(timeout=10)




# ----- a stop that cannot be confirmed is not a stop -----


def _worker(tmp_path: Path, process: subprocess.Popen) -> Any:
    scratch = tmp_path / f"scratch-{process.pid}"
    scratch.mkdir(parents=True, exist_ok=True)
    return adapter.Worker(
        root=tmp_path,
        process=process,
        port=0,
        token="unused",
        scratch=scratch,
        pgid=adapter._group_of(process),
    )


def test_a_confirmed_close_needs_the_process_table_to_have_answered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """"I could not look" is not "there is nothing there", and it used to be recorded as one.

    `_group_members` mapped every failure of `ps` onto an empty sequence, which is the same value
    a group that had emptied gives. A confirmed stop built on that confirmed nothing, and the
    episode graded on the strength of it was graded on a tree something might still be writing
    to."""
    process = _sleeper(30)
    worker = _worker(tmp_path, process)
    monkeypatch.setattr(adapter, "_group_members", lambda pgid: None)
    with pytest.raises(adapter.WorkerError, match="could not be confirmed stopped"):
        worker.close(confirm=True)
    # The worker itself is gone either way: what failed is the evidence, not the stop.
    assert process.poll() is not None
    assert worker.stopped is False


def test_a_confirmed_close_needs_the_group_to_be_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A descendant that outlives SIGKILL is a domain that was not contained, and grading a tree
    it can still write to is grading a state no instant of the episode had."""
    process = _sleeper(30)
    worker = _worker(tmp_path, process)
    monkeypatch.setattr(adapter, "_group_members", lambda pgid: [999999])
    monkeypatch.setattr(adapter, "_CLOSE_SECONDS", 0.05)
    with pytest.raises(adapter.WorkerError, match="could not be confirmed stopped"):
        worker.close(confirm=True)
    assert worker.stopped is False


def test_a_leader_something_else_reaped_leaves_a_number_nobody_may_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Saving a pgid at spawn does not reserve the number; reaping the leader releases it.

    A leader that merely exited is still holding its group, so the close below signals it and
    confirms it, which is what stops an episode buying an unscored row by killing its own world.
    A leader something *else* has already reaped is a different fact: the number may have been
    handed on, so it is neither signalled nor enumerated."""
    process = _sleeper(0.01)
    worker = _worker(tmp_path, process)
    process.wait(timeout=10)  # reaped here, before the close

    signalled: List[Any] = []
    monkeypatch.setattr(adapter, "_signal_group", lambda pgid, how: signalled.append((pgid, how)))
    monkeypatch.setattr(adapter, "_group_members", lambda pgid: signalled.append(pgid) or [])
    with pytest.raises(adapter.WorkerError, match="could not be confirmed stopped"):
        worker.close(confirm=True)
    # Nothing was signalled and nothing was enumerated: the number is not this worker's to use.
    assert signalled == []


def test_a_worker_that_exited_on_its_own_is_still_stopped_and_confirmed(tmp_path: Path) -> None:
    """The exploit the obvious ordering would have opened.

    Reaping the leader first and refusing an already-exited one hands an episode a way out of a
    bad score: kill the world from inside a block, and the seal cannot be confirmed, so the row is
    unscored rather than low. A pid is reserved until its parent reaps it and a group exists while
    any member does, so an exited-but-unreaped leader is still holding this group: it is signalled,
    reaped and confirmed like any other, and the episode is graded on what upstream persisted."""
    process = _sleeper(0.01)
    worker = _worker(tmp_path, process)
    time.sleep(0.3)
    # Exited, and deliberately not polled: the pid is a zombie this process still holds.
    assert process.returncode is None
    worker.close(confirm=True)
    assert worker.stopped is True


def test_an_ordinary_teardown_close_still_never_raises(tmp_path: Path) -> None:
    """The confirmation is the finalizer's demand, not teardown's. Teardown runs on a shared loop
    and its job is to release, so it takes the same stop without the assertion."""
    process = _sleeper(0.01)
    worker = _worker(tmp_path, process)
    process.wait(timeout=10)
    worker.close()
    assert worker.closed is True and worker.stopped is False


def test_a_process_table_that_cannot_be_read_is_not_an_empty_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The value beneath both of the above, at the helper that produces it."""
    monkeypatch.setenv("PATH", "")
    assert adapter._group_members(os.getpid()) is None


# ----- what the grader reads, and what it refuses -----


def test_an_output_tree_with_a_link_in_it_is_refused_rather_than_graded(tmp_path: Path) -> None:
    """The grading process is pointed at the root that holds the answers and also has to read the
    state to grade, which was writable by the process that ran the agent's code. A link planted
    under the output tree resolves in the grader, so it could make the filing, the digest and the
    evaluator read the graded tree instead of what the episode submitted."""
    outputs = tmp_path / "episode"
    (outputs / "tasks" / "abc_1" / "dbs").mkdir(parents=True)
    (outputs / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").write_text("[]")
    snapshot = adapter.snapshot_outputs(outputs, into=tmp_path / "episode.graded")
    assert (snapshot / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").read_text() == "[]"

    (outputs / "tasks" / "abc_1" / "answers").symlink_to(tmp_path)
    with pytest.raises(adapter.SnapshotError, match="symbolic link"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "episode.graded")


def test_an_episode_that_never_started_has_no_tree_to_grade(tmp_path: Path) -> None:
    with pytest.raises(adapter.SnapshotError, match="no output tree"):
        adapter.snapshot_outputs(tmp_path / "never", into=tmp_path / "into")


# ----- the caches say what they were built from -----


def test_two_corpora_under_one_cache_root_are_two_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Source A then source B, which is the case the cache name could not tell apart.

    The name carried the data version and the generator's constants and nothing about the corpus,
    while `already_derived` trusted a path existing. A process pointed at a second corpus therefore
    computed a fingerprint for that one and served task material derived from the first."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    adapter.cache_root.cache_clear() if hasattr(adapter.cache_root, "cache_clear") else None

    def _corpus(where: Path, mail: str) -> Path:
        task = where / "data" / "tasks" / "abc_1"
        (task / "dbs").mkdir(parents=True)
        (task / "dbs" / "gmail.jsonl").write_text(mail)
        (where / "data" / "version.txt").write_text("0.1.0")
        (where / "data" / "base_dbs").mkdir()
        (where / "data" / "base_dbs" / "big.jsonl").write_text(mail)
        return where

    first = adapter.corpus_digest(_corpus(tmp_path / "a", "one"))
    second = adapter.corpus_digest(_corpus(tmp_path / "b", "two"))
    assert first != second
    assert adapter.derived_root(first) != adapter.derived_root(second)
    assert adapter.graded_root(first) != adapter.graded_root(second)
    # And the shared base is inside the digest, which it was not: only `version.txt` and the task
    # tree were, so 134 MB of starting state every episode reads could change invisibly.
    (tmp_path / "a" / "data" / "base_dbs" / "big.jsonl").write_text("edited")
    assert adapter.corpus_digest(tmp_path / "a") != first


def test_a_cache_that_was_built_from_something_else_is_refused(tmp_path: Path) -> None:
    """The name cannot cover a tree edited, moved or restored in place under the old name, and a
    cache is the material a run is scored against, so the stamp inside it is checked too."""
    root = tmp_path / "seeded"
    adapter.stamp_cache(root, source="aaaa")
    adapter.stamp_cache(root, source="aaaa")  # idempotent
    with pytest.raises(adapter.ProvisioningError, match="was built from"):
        adapter.stamp_cache(root, source="bbbb")


# ----- the seal is verified, not inferred -----


def test_a_nested_chmod_that_failed_leaves_the_entry_unsealed(tmp_path: Path) -> None:
    """The warm path used to read the top-level mode alone, on the reasoning that `_seal` sets it
    last. That held only while every nested chmod succeeded, and `_chmod` swallowed the ones that
    did not, so one failure left a writable child permanently behind a read-only top."""
    tree = tmp_path / "entry"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "file.jsonl").write_text("x")
    world._seal(tree)
    assert world._sealed(tree) is True

    # Exactly the state a swallowed failure produced: read-only top, writable child.
    os.chmod(tree, 0o755)
    os.chmod(tree / "nested" / "file.jsonl", 0o644)
    os.chmod(tree, 0o555)
    assert world._sealed(tree) is False
    world._unseal(tree)


def test_a_seal_that_cannot_be_taken_fails_the_derivation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_chmod` swallowed every error, which turned a filesystem that cannot hold the invariant
    into a derivation that claims it does."""
    monkeypatch.setattr(world.os, "chmod", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    with pytest.raises(OSError):
        world._chmod(tmp_path, 0o555)


def test_a_derivation_publishes_rather_than_building_in_place(tmp_path: Path) -> None:
    """A crash leaves a staging directory and never a half-made target.

    `derive_root` used to unseal, delete, copy and mark the live target while holding a lock the
    helper may decline to take at all, so two cold processes on a lockless filesystem rebuilt the
    same directory under each other."""
    original = tmp_path / "corpus" / "data"
    (original / "base_dbs").mkdir(parents=True)
    (original / "base_dbs" / "big.jsonl").write_text("shared")
    derived = tmp_path / "derived" / "data"

    published: List[Path] = []
    real_publish = world._publish

    def _watch(building: Path, target: Path, *, replacing: bool = False) -> None:
        # Whatever is published is complete and sealed before it has a name.
        assert (building / world._COMPLETE).exists()
        assert world._sealed(building)
        published.append(target)
        real_publish(building, target, replacing=replacing)

    world._publish = _watch
    try:
        world.derive_root(original=original, derived=derived)
    finally:
        world._publish = real_publish
    assert published == [derived / "base_dbs"]
    # Nothing staged survives a completed build.
    assert [p.name for p in derived.iterdir() if ".building" in p.name] == []
    # And the warm path publishes nothing at all.
    world.derive_root(original=original, derived=derived)
    assert len(published) == 1
    world._unseal(derived)


# ----- a seal that failed publishes no verdict -----


def test_a_failed_terminal_publishes_the_failure_and_neither_arm() -> None:
    """The row an unconfirmed stop leaves, and what used to be on it.

    A finalize that fails closed used to publish a row of zeroed fractions with an empty receipt
    and an empty notice beside them. That is a scored-looking row for an episode nothing scored:
    the zeros average into a mean, and an empty receipt is still an item a paired policy selects,
    renames and reveals. There is no verdict behind such an episode, so the honest record of it is
    that fact alone."""
    from shogym.envs.appworld import env_v1
    from shogym.serve.lifecycle import TerminalEvidence

    class _Env:
        _config_digest = "fingerprint"

    evidence = TerminalEvidence(
        source="explicit_tool", status="finalize_error", verdict={}, diagnostic="the stop was not"
    )
    fb = env_v1.AppWorldEnv._verify(
        _Env(),  # pyright: ignore[reportArgumentType]
        trajectory=None,  # pyright: ignore[reportArgumentType]
        task={},
        terminated=True,
        evidence=evidence,
    )
    published = {item.name for item in fb.episode}
    # The failure, and nothing that could be read as a score or revealed as a dose.
    assert published == {"finalize_error"}
    assert [item.value for item in fb.episode] == [True]
    # The record still says which configuration the failure happened under, off every wire.
    assert [(item.name, item.value) for item in fb.inference] == [("config_digest", "fingerprint")]


def test_an_ordinary_terminal_still_publishes_both_arms() -> None:
    """The other side of the same branch, so the check above is about the failure rather than
    about `_verify` having stopped publishing."""
    from shogym.envs.appworld import env_v1
    from shogym.serve.lifecycle import TerminalEvidence

    class _Env:
        _config_digest = "fingerprint"

    evidence = TerminalEvidence(
        source="explicit_tool",
        status="ok",
        verdict={"ledger_fraction": 1.0, "report": "a receipt", "notice": "a digest"},
    )
    fb = env_v1.AppWorldEnv._verify(
        _Env(),  # pyright: ignore[reportArgumentType]
        trajectory=None,  # pyright: ignore[reportArgumentType]
        task={},
        terminated=True,
        evidence=evidence,
    )
    published = {item.name: item.value for item in fb.episode}
    assert published["ledger_fraction"] == 1.0
    assert published["report"] == "a receipt"
    assert published["notice"] == "a digest"
    assert "finalize_error" not in published


# ----- what a stop is signalled with, and in what order -----


def test_the_group_is_killed_after_a_short_grace_and_reaped_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two orderings, both of which the docs claimed and the code did not do.

    SIGTERM used to be followed by up to ten seconds of waiting, and SIGTERM is catchable: every
    one of those seconds is a second in which the process that ran agent-authored code can still
    write into the tree about to be graded. It gets a short grace and then a signal it cannot
    decline.

    And the leader was reaped *before* the group was enumerated. Reaping releases the pid, and a
    group exists only while it has a member, so the enumeration and the escalation after it were
    about whatever held the number by then. The group is read while the leader is still a zombie
    holding it, and the reap is last."""
    order: List[str] = []
    real_members = adapter._group_members

    def _watch_signal(pgid: int, how: int) -> None:
        order.append("SIGKILL" if how == signal.SIGKILL else "SIGTERM")
        try:
            os.killpg(pgid, how)
        except OSError:
            # A group that has already emptied, which is what the real helper tolerates too: the
            # kill after the grace is sent whatever the group did, and a no-op is one of the
            # things it can be.
            pass

    def _watch_members(pgid: int) -> Any:
        order.append("enumerate")
        return real_members(pgid)

    process = _sleeper(30)
    worker = _worker(tmp_path, process)
    monkeypatch.setattr(adapter, "_signal_group", _watch_signal)
    monkeypatch.setattr(adapter, "_group_members", _watch_members)
    monkeypatch.setattr(adapter, "_TERM_GRACE_SECONDS", 0.05)

    began = time.monotonic()
    worker.close(confirm=True)
    elapsed = time.monotonic() - began

    assert worker.stopped is True
    # A kill follows the grace whatever the group did, and every enumeration happened before the
    # reap, which is the last thing that runs.
    assert order[0] == "SIGTERM"
    assert "SIGKILL" in order
    assert order.index("SIGKILL") < order.index("enumerate", order.index("SIGKILL"))
    assert process.returncode is not None
    # The grace is the bound, not ten seconds of a signal the process may ignore.
    assert elapsed < 5.0


# ----- what the grader may be handed -----


def test_a_symlinked_output_root_is_refused(tmp_path: Path) -> None:
    """`resolve()` erased the question. A root that was itself a link came back as whatever it
    named, and only its descendants were inspected afterwards, so substituting the episode's own
    output directory substituted the whole tree the grade is computed over."""
    real = tmp_path / "real"
    (real / "tasks").mkdir(parents=True)
    (real / "tasks" / "a.jsonl").write_text("[]")
    link = tmp_path / "outputs"
    link.symlink_to(real)
    with pytest.raises(adapter.SnapshotError, match="output root .* is a symbolic link"):
        adapter.snapshot_outputs(link, into=tmp_path / "into")
    # And the ordinary root is still accepted, so the refusal is about the link.
    assert adapter.snapshot_outputs(real, into=tmp_path / "into").is_dir()


def test_a_snapshot_is_bounded_in_every_direction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An episode wrote this tree, so walking and copying it without a bound lets the episode
    decide how long finalization takes and how much of the host's disk it uses. None of the
    deadlines above cover it: a deadline cancels an await, and the thread doing the copying does
    not stop for that."""
    outputs = tmp_path / "outputs"
    (outputs / "dbs").mkdir(parents=True)
    for index in range(6):
        (outputs / "dbs" / f"{index}.jsonl").write_text("x" * 64)

    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_NODES", 3)
    with pytest.raises(adapter.SnapshotError, match="more than 3 entries"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "a")

    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_NODES", 20_000)
    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_BYTES", 100)
    with pytest.raises(adapter.SnapshotError, match="larger than 100 bytes"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "b")

    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_BYTES", 1 << 30)
    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_DEPTH", 0)
    with pytest.raises(adapter.SnapshotError, match="deeper than 0 directories"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "c")

    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_DEPTH", 24)
    monkeypatch.setattr(adapter, "_SNAPSHOT_SECONDS", -1.0)
    with pytest.raises(adapter.SnapshotError, match="longer than"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "d")


def test_an_abandoned_snapshot_stops_at_the_next_file(tmp_path: Path) -> None:
    """Cancelling the await does not stop the thread, so the thread is given something to stop
    for, and it is checked once per file rather than once per tree."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for index in range(4):
        (outputs / f"{index}.jsonl").write_text("x")
    stop = threading.Event()
    stop.set()
    with pytest.raises(adapter.SnapshotError, match="abandoned"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "into", stop=stop)


# ----- what the identity covers -----


def test_the_fingerprint_covers_the_realized_runtime_and_the_authored_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four inputs the digest claimed to cover and did not.

    The runtime cache is named for the direct AppWorld release while it is built by resolving
    that release's ranges against whatever the host offers, so the realized interpreter and
    distribution set were outside the identity entirely. The guide, the tool guide and the
    appended paragraph are the text every episode is given, which is authored treatment rather
    than scenery. And the generator digest already decided which derived cache was served."""
    from shogym.envs.appworld import adapter, env_v1, world
    from shogym.envs.appworld.env_v1 import run_fingerprint

    base = run_fingerprint(pulse=0, report="graded", blocks=60)
    for module, name, moved in (
        (adapter, "runtime_digest", lambda: "a different runtime"),
        (env_v1, "_WORLD_GUIDE", "a different guide"),
        (env_v1, "_TOOL_GUIDE", "a different tool guide"),
        (world, "APPENDED_PARAGRAPH", "a different chore"),
        (adapter, "_generator_digest", lambda: "a different generator"),
    ):
        original = getattr(module, name)
        try:
            monkeypatch.setattr(module, name, moved)
            assert base != run_fingerprint(pulse=0, report="graded", blocks=60), name
        finally:
            monkeypatch.setattr(module, name, original)


def test_a_spec_edited_in_place_is_not_served_from_a_cache(tmp_path: Path) -> None:
    """The identity moved and the task did not.

    `task_specs` was memoized on `(root, task id)`, which says where a spec was rather than what
    it said. Editing a corpus in place produced a new `corpus_digest` and a new cache name while
    the same process went on serving the instruction it had read the first time."""
    task = tmp_path / "data" / "tasks" / "abc_1"
    task.mkdir(parents=True)
    (task / "specs.json").write_text('{"instruction": "first"}')
    assert adapter.task_specs(tmp_path, "abc_1")["instruction"] == "first"
    (task / "specs.json").write_text('{"instruction": "second"}')
    assert adapter.task_specs(tmp_path, "abc_1")["instruction"] == "second"


def test_a_corpus_with_a_link_in_it_is_refused_rather_than_half_digested(
    tmp_path: Path,
) -> None:
    """The digest skipped links; derivation follows them. So a served world held bytes the
    identity had never read, and changing what a link pointed at changed the world without moving
    the digest that claims to say what the world is."""
    data = tmp_path / "data"
    (data / "tasks" / "abc_1").mkdir(parents=True)
    (data / "version.txt").write_text("0.1.0")
    (data / "tasks" / "abc_1" / "specs.json").write_text("{}")
    assert len(adapter.corpus_digest(tmp_path)) == 16

    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text('{"answer": 1}')
    (data / "tasks" / "abc_1" / "linked.json").symlink_to(elsewhere)
    with pytest.raises(adapter.ProvisioningError, match="symbolic link"):
        adapter.corpus_digest(tmp_path)


# ----- what the runtime's identity covers, and what the pins enforce -----


def _fake_runtime(home: Path) -> Path:
    """A provisioned interpreter's shape, without provisioning one.

    Enough of it for the two things that read a runtime off the filesystem: a venv config, a
    ``site-packages`` holding the pinned distribution, and an interpreter to name."""
    packages = home / "lib" / "python3.12" / "site-packages"
    (packages / "appworld").mkdir(parents=True)
    (packages / "appworld" / "__init__.py").write_text("VERSION = 'one'\n")
    dist = packages / f"appworld-{adapter.UPSTREAM_VERSION}.dist-info"
    dist.mkdir()
    (dist / "RECORD").write_text("appworld/__init__.py,sha256=aaaa,16\n")
    (home / "pyvenv.cfg").write_text("home = /usr/bin\nversion = 3.12.0\n")
    (home / "bin").mkdir()
    (home / "bin" / "python").write_text("")
    return home / "bin" / "python"


def test_the_runtime_digest_moves_when_the_installed_code_does(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest hashed labels and called them an identity.

    It covered the platform, the venv config and the `.dist-info` directory *names*, which is the
    list of what was asked for written down a second time. Two different artifacts published under
    one version were therefore one identity, and so was a module edited in place inside the
    interpreter every world runs under. The realized code has to be what moves it, and this
    perturbs the tree rather than replacing the function.

    The edit below is the same length as what it replaces, because a digest that only noticed
    lengths would pass a weaker version of this test."""
    home = tmp_path / "runtime"
    python = _fake_runtime(home)
    monkeypatch.setattr(adapter, "runtime", lambda: python)
    packages = home / "lib" / "python3.12" / "site-packages"

    before = adapter.runtime_digest()
    assert adapter.runtime_digest() == before, "the same tree has to give the same answer"

    (packages / "appworld" / "__init__.py").write_text("VERSION = 'two'\n")
    edited = adapter.runtime_digest()
    assert edited != before

    (packages / "appworld" / "models.py").write_text("# a module that was not installed before\n")
    added = adapter.runtime_digest()
    assert added != edited

    (packages / "widget-2.0.dist-info").mkdir()
    (packages / "widget-2.0.dist-info" / "RECORD").write_text("widget/__init__.py,,\n")
    assert adapter.runtime_digest() != added

    # And it reaches the number every row carries, which is the only reason the digest exists.
    from shogym.envs.appworld.env_v1 import run_fingerprint

    stamped = run_fingerprint(pulse=0, report="graded", blocks=60)
    (packages / "appworld" / "__init__.py").write_text("VERSION = 'three'\n")
    assert run_fingerprint(pulse=0, report="graded", blocks=60) != stamped


def test_a_bytecode_cache_is_not_part_of_the_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest reads installed bytes, and `__pycache__` is not one of them.

    A `.pyc` is written lazily by whichever process imported the module first, so hashing one
    would make the interpreter's identity depend on what ran before rather than on what is
    installed. Nothing hides in one either: Python discards a cache whose source is newer."""
    home = tmp_path / "runtime"
    python = _fake_runtime(home)
    monkeypatch.setattr(adapter, "runtime", lambda: python)
    cache = home / "lib" / "python3.12" / "site-packages" / "appworld" / "__pycache__"

    before = adapter.runtime_digest()
    cache.mkdir()
    (cache / "__init__.cpython-312.pyc").write_bytes(b"\x00compiled")
    assert adapter.runtime_digest() == before


def test_a_runtime_is_reused_only_when_it_is_the_one_the_pins_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test for reuse was that `bin/python` existed.

    That is true of a venv whose install died after the interpreter was created, of one a later
    pin change should have rebuilt, and of one somebody edited. The pins now name the directory
    and are written inside it, so what is reused is a tree this code finished building under these
    pins."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    built: List[Path] = []

    def _build(home: Path) -> None:
        # Staged and published exactly as the real builder does it, because publishing over an
        # incumbent is one of the cases this test walks through.
        staging = home.with_name(home.name + ".building")
        _fake_runtime(staging)
        (staging / adapter._RUNTIME_FILE).write_text(adapter._runtime_stamp())
        adapter._publish_runtime(staging, home)
        built.append(home)

    monkeypatch.setattr(adapter, "_build_runtime", _build)

    python = adapter.runtime()
    home = python.parent.parent
    assert built == [home]
    # Both pins are in the name, so moving either one is a second interpreter rather than a reused
    # first one.
    assert adapter.UPSTREAM_VERSION in home.name
    assert adapter.UPSTREAM_SHA[:12] in home.name

    adapter.runtime()
    assert len(built) == 1, "a stamped runtime is reused"

    (home / adapter._RUNTIME_FILE).unlink()
    adapter.runtime()
    assert len(built) == 2, "an interpreter nobody stamped is not one this code built"

    (home / adapter._RUNTIME_FILE).write_text('{"sha": "some other commit"}')
    with pytest.raises(adapter.ProvisioningError, match="says it was built as"):
        adapter.runtime()


def test_a_runtime_that_resolved_another_release_never_gets_published(tmp_path: Path) -> None:
    """The requirement string asks and the resolver answers, and nothing read the answer.

    A build that resolved a different release, or an index that moved under the name, was served
    out of a cache whose name said the pin had been honored. This is checked inside the staging
    tree, before the rename, so the published name never holds an interpreter that failed it."""
    home = tmp_path / "runtime"
    _fake_runtime(home)
    adapter._check_pin(home)  # the pinned release passes

    packages = adapter._site_packages(home)[0]
    (packages / f"appworld-{adapter.UPSTREAM_VERSION}.dist-info").rename(
        packages / "appworld-0.1.2.dist-info"
    )
    with pytest.raises(adapter.ProvisioningError, match="but this port pins"):
        adapter._check_pin(home)


# ----- what an env goes on serving after it has said what corpus it serves -----


def _one_task_corpus(root: Path, *, instruction: str, moment: str) -> Path:
    """A corpus holding one task, shaped like the pinned one."""
    task = root / "data" / "tasks" / "abc_1"
    task.mkdir(parents=True)
    (root / "data" / "version.txt").write_text("0.1.0")
    (task / "specs.json").write_text(
        json.dumps(
            {
                "instruction": instruction,
                "datetime": moment,
                "supervisor": {
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                    "phone_number": "555",
                },
            }
        )
    )
    return root


def test_a_corpus_is_read_once_for_its_digest_and_for_its_authored_text(tmp_path: Path) -> None:
    """The digest and the specs are one observation, not two.

    A digest computed in the constructor and a `specs.json` read later are two readings of a tree
    that can change between them, and the gap is exactly where a corpus edited in place served new
    authored text under the old identity. They come out of one walk now, and a manifest task the
    walk never reached is a manifest and a corpus that are not describing the same split."""
    root = _one_task_corpus(tmp_path / "corpus", instruction="do the thing", moment="2023-05-18T12:00:00")
    snapshot = adapter.corpus_snapshot(root, task_ids=("abc_1",))
    assert snapshot.digest == adapter.corpus_digest(root)
    assert snapshot.specs["abc_1"]["instruction"] == "do the thing"

    with pytest.raises(adapter.ProvisioningError, match="no specification for"):
        adapter.corpus_snapshot(root, task_ids=("abc_1", "not_in_this_corpus_1"))


def test_an_env_serves_the_corpus_it_was_constructed_against(tmp_path: Path) -> None:
    """The env's own time-of-check gap, which the spec reread left open.

    The corpus digest, the served cache's name and the grader's cache name were fixed in the
    constructor, and the instruction, the supervisor and the datetime went on being reread from
    the live corpus every time a task was described, seeded or scored. So a corpus edited after
    construction served new authored text out of caches named for the old bytes, under a
    fingerprint that had never seen it, and nothing in the record said so.

    Refusing would need the corpus rehashed on every read, which is two seconds a time; serving
    what was read costs nothing and is a contract that can be stated. This is the contract."""
    from shogym.envs.appworld import env_v1

    root = _one_task_corpus(tmp_path / "corpus", instruction="the first", moment="2023-05-18T12:00:00")
    snapshot = adapter.corpus_snapshot(root, task_ids=("abc_1",))

    env = env_v1.AppWorldEnv.__new__(env_v1.AppWorldEnv)
    env._original = root / "data"
    env._task_ids = ("abc_1",)
    env._specs = snapshot.specs
    env._corpus = snapshot.digest

    # The corpus changes under the env, in place, after it has stated what it is serving.
    (root / "data" / "tasks" / "abc_1" / "specs.json").write_text(
        json.dumps(
            {
                "instruction": "the second",
                "datetime": "2024-01-01T12:00:00",
                "supervisor": {
                    "first_name": "Mallory",
                    "last_name": "Elsewhere",
                    "email": "mallory@example.com",
                    "phone_number": "999",
                },
            }
        )
    )
    assert adapter.corpus_digest(root) != snapshot.digest, "the edit really did move the corpus"

    specs = env._task_specs("abc_1")
    assert specs["instruction"] == "the first"
    assert specs["datetime"] == "2023-05-18T12:00:00"
    # And every place the env hands that text on: the instructions an agent is given, and the
    # supervisor whose accounts its world is driven with.
    assert "the first" in env._instructions(0)
    assert "the second" not in env._instructions(0)
    assert env._load_task(0)["supervisor_email"] == "ada@example.com"


# ----- exclusion the filesystem cannot give -----


def test_a_mount_that_cannot_lock_refuses_the_builders_and_still_serves_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback the concurrency test above never exercises, on both sides of the fork.

    `_locked` yielded with no exclusion at all when the filesystem could not provide `flock`,
    which is right for the upstream-source download it was written for: that publishes by one
    atomic rename and a loser validates the winner, so the loss is redundant work. It is wrong for
    every caller here. The runtime and corpus builders stage under fixed `.building` names they
    delete first, so two of them remove and publish each other's half-built tree, and the
    permission windows open a published directory and seal it again, which a second process inside
    them closes under the first's feet.

    Simulated at the errno, because a filesystem that cannot lock is not a thing a test suite
    has."""
    from shogym.envs import _upstream

    def _cannot_lock(descriptor: int, operation: int) -> None:
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(_upstream.fcntl, "flock", _cannot_lock)
    monkeypatch.setattr(_upstream, "_warned_unlocked", False)

    # The download path is unchanged: it says so once and gets on with it.
    with pytest.warns(RuntimeWarning, match="cannot provide flock"):
        with _upstream._locked(tmp_path):
            pass
    with pytest.raises(_upstream.ExclusionUnavailable, match="needs real exclusion"):
        with _upstream._locked(tmp_path, required=True):
            pass

    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv(adapter.ROOT_ENV_VAR, raising=False)
    # The runtime builder.
    with pytest.raises(_upstream.ExclusionUnavailable):
        adapter.runtime()
    # The corpus builder, past the interpreter it provisions first.
    monkeypatch.setattr(adapter, "runtime", lambda: Path("python"))
    monkeypatch.setattr(adapter, "ensure_apps", lambda: None)
    with pytest.raises(_upstream.ExclusionUnavailable):
        adapter.ensure_corpus()

    original = tmp_path / "corpus" / "data"
    (original / "tasks" / "abc_1" / "dbs").mkdir(parents=True)
    (original / "tasks" / "abc_1" / "ground_truth").mkdir()
    (original / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").write_text("")
    (original / "shared").write_text("base databases")
    # The permission window over the shared entries.
    with pytest.raises(_upstream.ExclusionUnavailable):
        world.derive_root(original=original, derived=tmp_path / "derived" / "data")
    # And the one over the published tasks directory.
    with pytest.raises(_upstream.ExclusionUnavailable):
        world.derive_task(
            original=original,
            derived=tmp_path / "derived2" / "data",
            graded=tmp_path / "graded2" / "data",
            task_id="abc_1",
            write_log=lambda source, into: into.write_text("seeded"),
        )


# ----- the snapshot's bounds cover the operations that spend them -----


def test_the_walk_stops_inside_a_directory_rather_than_after_reading_all_of_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The enumeration was the one part of the walk no bound could reach.

    `sorted(source.iterdir())` read and sorted a whole directory before the first check of the
    node count, the deadline or the stop flag, so an episode that left a million names in one
    directory spent all of that time and memory after the budget was gone and after the
    finalization it belonged to had been cancelled. The entries arrive one at a time now, and the
    bound is spent as they arrive.

    Counted at the seam rather than timed, because "it stopped early" is the claim and a
    stopwatch is not how to say it."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    for index in range(400):
        (outputs / f"{index}.jsonl").write_text("x")

    consumed = 0
    real_scandir = os.scandir

    class _counting:
        def __init__(self, inner: Any) -> None:
            self._inner = inner

        def __enter__(self) -> "_counting":
            return self

        def __exit__(self, *exc: Any) -> None:
            self._inner.__exit__(*exc)

        def __iter__(self) -> "_counting":
            return self

        def __next__(self) -> Any:
            nonlocal consumed
            entry = next(self._inner)
            consumed += 1
            return entry

    monkeypatch.setattr(os, "scandir", lambda path: _counting(real_scandir(path)))

    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_NODES", 5)
    with pytest.raises(adapter.SnapshotError, match="more than 5 entries"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "a")
    assert 0 < consumed <= 6, "the walk read the whole directory before it refused it"

    # The deadline is read inside the enumeration too, not only between directories.
    consumed = 0
    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_NODES", 20_000)
    monkeypatch.setattr(adapter, "_SNAPSHOT_SECONDS", -1.0)
    with pytest.raises(adapter.SnapshotError, match="longer than"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "b")
    assert consumed <= 1


def test_a_cancelled_snapshot_stops_partway_through_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Once per file is not a bound on a tree with one big file in it.

    `shutil.copyfile` is one call with no way in, so a finalization that was cancelled while the
    copy was inside a large file waited out the whole of it. The bytes move in chunks now and the
    flag is read between them, which is what makes cancellation a bound on the work rather than on
    the gaps between pieces of it."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "one.jsonl").write_bytes(b"x" * (1 << 16))
    # Sixty-four chunks for one file, so a flag raised on the tenth reading is raised well inside
    # it however the checks before the copy are arranged.
    monkeypatch.setattr(adapter, "_SNAPSHOT_CHUNK_BYTES", 1024)

    class _after:
        """A stop flag that is not set when the copy starts and is set once it is under way."""

        def __init__(self, checks: int) -> None:
            self.checks = checks
            self.seen = 0

        def is_set(self) -> bool:
            self.seen += 1
            return self.seen > self.checks

    stop = _after(checks=10)
    with pytest.raises(adapter.SnapshotError, match="abandoned"):
        adapter.snapshot_outputs(
            outputs,
            into=tmp_path / "into",
            stop=stop,  # pyright: ignore[reportArgumentType]
        )
    # It stopped with the file part copied, which is the whole claim: the refusal came from inside
    # one copy rather than from the gap after it.
    partial = tmp_path / "into" / "one.jsonl"
    assert partial.exists()
    assert 0 < partial.stat().st_size < (1 << 16)


def test_the_deadline_is_read_while_one_file_is_still_being_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same gap, told by the clock instead of by the flag.

    A copy that crosses the sixty-second budget used to run to the end of the file and only then
    be noticed, so an episode with one enormous file decided how long finalization took. The
    bounds are shrunk here rather than the file grown: a chunk of one byte over a megabyte is work
    that certainly outlasts a fiftieth of a second, and the assertion is that it was stopped in
    the middle of it."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "one.jsonl").write_bytes(b"x" * (1 << 20))

    monkeypatch.setattr(adapter, "_SNAPSHOT_CHUNK_BYTES", 1)
    monkeypatch.setattr(adapter, "_SNAPSHOT_SECONDS", 0.05)
    with pytest.raises(adapter.SnapshotError, match="longer than"):
        adapter.snapshot_outputs(outputs, into=tmp_path / "into")
    partial = tmp_path / "into" / "one.jsonl"
    assert partial.exists()
    assert 0 < partial.stat().st_size < (1 << 20)


def test_the_previous_snapshot_is_removed_under_the_same_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination is one character away from a name the world is handed.

    AppWorld is given the episode's output root by absolute path, and the snapshot goes to that
    name with `.graded` on the end, in a process running as the same user. So the tree removed
    before the copy is a tree the episode can size, and it was removed by an `rmtree` outside
    every bound: the deadline had not started, the stop flag was not read, and nothing counted the
    entries."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "one.jsonl").write_text("what the episode submitted")

    into = tmp_path / "outputs.graded"
    (into / "planted").mkdir(parents=True)
    for index in range(400):
        (into / "planted" / f"{index}.jsonl").write_text("x")

    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_NODES", 5)
    with pytest.raises(adapter.SnapshotError, match="more than 5 entries"):
        adapter.snapshot_outputs(outputs, into=into)

    # And a cancelled finalization does not wait out the removal either.
    stop = threading.Event()
    stop.set()
    monkeypatch.setattr(adapter, "_SNAPSHOT_MAX_NODES", 20_000)
    with pytest.raises(adapter.SnapshotError, match="abandoned"):
        adapter.snapshot_outputs(outputs, into=into, stop=stop)
    assert (into / "planted").exists(), "nothing was removed after the flag was seen"


def test_a_provisioning_subprocess_that_never_finishes_is_not_waited_on_forever(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Construction waits on these, and a construction that never returns reports nothing at all.

    `pip install` against an index that accepts the connection and then stops sending has no
    timeout of its own, and neither has an unpack whose child wedged. A run that says which
    command hung is strictly better than a queue that never starts."""
    began = time.monotonic()
    with pytest.raises(adapter.ProvisioningError, match="timed out after"):
        adapter._run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert time.monotonic() - began < 10.0
