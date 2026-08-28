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
from typing import Any, Dict, List, Optional, Tuple

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


def test_a_block_budget_that_is_not_a_count_of_blocks_is_refused_at_construction() -> None:
    """Zero blocks used to mean one block, and it is the mutating one that counts.

    The env stored a zero budget and published a serve-layer horizon of one. The guard on the
    served tool reads the budget for truth before it compares anything, so zero was read as no
    limit at all and the first `execute` reached the world; only after that mutation did the serve
    layer's own horizon end the episode. A negative budget half-works in the other direction: it
    refuses the first block and still spends the call that ends the horizon.

    A budget is a count of blocks, so the smallest honest one is one, and the boundary that can
    say so is this one. It is refused before the constructor provisions anything, and refused
    rather than coerced: this value is a member of the fingerprint two runs are compared on, and
    a string that quietly became a number would be a second configuration wearing the first's
    identity."""
    from shogym.envs.appworld.env_v1 import AppWorldEnv

    for refused in (0, -1, True, 2.0, "2", None):
        with pytest.raises(ValueError, match="positive whole number of blocks"):
            AppWorldEnv(horizon=refused)  # pyright: ignore[reportArgumentType]


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
    # No cache is written back, which is what keeps every `.pyc` a worker can consult a hash-based
    # one and lets the runtime digest leave `__pycache__` out and still be true about what runs.
    assert scrubbed["PYTHONDONTWRITEBYTECODE"] == "1"
    assert set(scrubbed) <= set(adapter._ENV_ALLOW_LIST) | {
        "HOME",
        "APPWORLD_CACHE",
        "PYTHONDONTWRITEBYTECODE",
    }


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




# ----- a worker whose parent is gone -----


def _keepalive_worker_script(tmp_path: Path) -> Path:
    """A worker that arms the real parent-death watch and then never stops on its own.

    It calls `worker.watch_parent`, which is the code under test, rather than a copy of it. That
    module imports nothing but the standard library at its top level, so it can be loaded by an
    interpreter that has never heard of `appworld` or of shogym."""
    script = tmp_path / "keepalive_worker.py"
    script.write_text(
        "import json, sys, time\n"
        f"sys.path.insert(0, {str(adapter.WORKER.parent)!r})\n"
        "import worker\n"
        "opening = json.loads(sys.stdin.readline())\n"
        "sys.stdin.close()\n"
        "worker.watch_parent(opening.get('keepalive'))\n"
        "sys.stdout.write(json.dumps({'port': 1}) + '\\n')\n"
        "sys.stdout.flush()\n"
        "while True:\n"
        "    time.sleep(1)\n"
    )
    return script


def _supervisor_script(tmp_path: Path) -> Path:
    """A serving process that spawns one worker, says which pid it got, and then waits."""
    script = tmp_path / "supervisor.py"
    script.write_text(
        "import sys, time\n"
        "from pathlib import Path\n"
        "from shogym.envs.appworld import adapter\n"
        "adapter.runtime = lambda: Path(sys.executable)\n"
        "adapter.WORKER = Path(sys.argv[1])\n"
        "worker = adapter.Worker.spawn(Path(sys.argv[2]))\n"
        "Path(sys.argv[3]).write_text(str(worker.process.pid))\n"
        "time.sleep(300)\n"
    )
    return script


def test_a_worker_stops_itself_when_the_process_that_started_it_dies(tmp_path: Path) -> None:
    """Teardown needs a parent, and the case it cannot reach is the parent dying with an episode
    open.

    A worker is started in a session of its own, so nothing reaps it when its owner goes: it was
    handed to init and went on serving a world, holding a port and a scratch directory, while the
    only handle on it, the port, the token, the process and the group number, died with the
    process that held them. Every other close test keeps the owning parent alive, so none of them
    could have seen this.

    The supervisor here is a real process running the real spawn, and it is killed with a signal
    it cannot handle rather than asked to tidy up. What the worker is left with is the reading end
    of a pipe, which reaches end of file because the kernel closed the writing end, and that is a
    fact about the parent rather than a message from one."""
    root = tmp_path / "root"
    root.mkdir()
    told = tmp_path / "worker.pid"
    supervisor = subprocess.Popen(
        [
            sys.executable,
            str(_supervisor_script(tmp_path)),
            str(_keepalive_worker_script(tmp_path)),
            str(root),
            str(told),
        ],
        env={**os.environ, "SHOGYM_CACHE": str(tmp_path / "cache")},
    )
    try:
        deadline = time.monotonic() + 60
        while not told.exists() and time.monotonic() < deadline:
            assert supervisor.poll() is None, "the supervisor exited instead of spawning"
            time.sleep(0.05)
        worker_pid = int(told.read_text())
        os.kill(worker_pid, 0)
        # And it was written down, which is the other half of what a later run has to work from.
        written = (tmp_path / "cache" / "appworld" / "workers.txt").read_text()
        assert f'"pid": {worker_pid}' in written

        supervisor.kill()
        supervisor.wait(timeout=30)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                os.kill(worker_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("the worker outlived the process that started it")
    finally:
        supervisor.kill()
        supervisor.wait(timeout=30)


def test_a_worker_whose_owner_is_gone_is_reclaimed_and_a_live_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable half: what a construction after a crash has instead of a `Worker` object.

    Nothing outlived the serving process. A resumed harness could neither adopt a worker nor name
    one for teardown, so it started another beside it. Every worker is written down now with the
    pid and the start time of the process that started it, and a construction reads that file.

    The start time is the load-bearing half. A pid alone answers "is something running under that
    number", and pids are handed out again, so an owner's number can belong to a stranger by the
    time anybody asks: reclaiming a live episode's world is worse than the failure this fixes. So
    the owner below is a real process that really exited, and the live one is a real process
    running under its own recorded birth."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    running: List[subprocess.Popen] = []

    def _worker_of(
        name: str, owner: subprocess.Popen, birth: str
    ) -> Tuple[subprocess.Popen, Path]:
        """One sleeper standing in for a worker, written down as ``owner``'s."""
        held = _sleeper(120)
        running.append(held)
        scratch = tmp_path / f"shogym-appworld-{name}"
        scratch.mkdir()
        adapter._append(
            "+"
            + json.dumps(
                {
                    "name": name,
                    "parent": owner.pid,
                    "birth": birth,
                    "boot": adapter._boot_id(),
                    "pid": held.pid,
                    "pid_birth": adapter.process_birth(held.pid),
                    "pgid": adapter._group_of(held),
                    "scratch": str(scratch),
                }
            )
        )
        return held, scratch

    gone = _sleeper(120)
    gone_birth = adapter.process_birth(gone.pid)
    assert gone_birth, "the process table has to answer for this test to mean anything"
    gone.kill()
    gone.wait(timeout=30)
    living = _sleeper(120)
    running.append(living)

    try:
        abandoned, abandoned_scratch = _worker_of("orphan", gone, gone_birth)
        kept, kept_scratch = _worker_of("live", living, adapter.process_birth(living.pid))

        assert adapter.reap() == ["orphan"]
        assert abandoned.wait(timeout=30) is not None, "the abandoned world was stopped"
        assert not abandoned_scratch.exists(), "and its scratch directory went with it"
        assert kept.poll() is None and kept_scratch.exists(), "the live one is untouched"
        assert [record["name"] for record in adapter.outstanding()] == ["live"]
        assert adapter.reap() == [], "and a second sweep has nothing left to do"
    finally:
        for process in running:
            process.kill()
            process.wait(timeout=30)


def test_an_orphan_nothing_can_be_told_about_keeps_its_record_and_its_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sweep may only spend a record it has actually dealt with.

    The stop used to report nothing whatever it had done, so every one of its silent paths reached
    the caller as a delivered kill. An entry whose owner is gone, whose worker pid is live, and
    whose birth the process table will not answer for was therefore reported reclaimed: its
    scratch directory was deleted and its ledger line tombstoned without a signal ever being sent,
    which leaves a world still serving with nothing left that names it. That is the failure the
    ledger exists to prevent, arrived at through the ledger.

    Two shapes of the same ambiguity below. One record never had a birth written down; the other
    has one the table has stopped answering for. Neither says the worker is gone and neither can
    be signalled safely, so both keep everything they have for the next construction to try."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    gone = _sleeper(120)
    gone_birth = adapter.process_birth(gone.pid)
    assert gone_birth, "the process table has to answer for this test to mean anything"
    gone.kill()
    gone.wait(timeout=30)

    running: List[subprocess.Popen] = []
    scratches: Dict[str, Path] = {}
    try:
        for name, birth in (("unrecorded", ""), ("unreadable", "1700000000")):
            held = _sleeper(120)
            running.append(held)
            scratch = tmp_path / f"shogym-appworld-{name}"
            scratch.mkdir()
            scratches[name] = scratch
            adapter._append(
                "+"
                + json.dumps(
                    {
                        "name": name,
                        "parent": gone.pid,
                        "birth": gone_birth,
                        "boot": adapter._boot_id(),
                        "pid": held.pid,
                        # Either never written, or written and no longer confirmable.
                        "pid_birth": birth,
                        "pgid": adapter._group_of(held),
                        "scratch": str(scratch),
                    }
                )
            )
        # The table answers about the owner (which is gone, so `kill` settles it without asking)
        # and refuses to answer about either worker.
        monkeypatch.setattr(adapter, "process_birth", lambda pid: "")

        assert adapter.reap() == [], "nothing was stopped, so nothing may be reported reclaimed"
        assert all(scratch.exists() for scratch in scratches.values())
        assert sorted(r["name"] for r in adapter.outstanding()) == ["unreadable", "unrecorded"]
        assert all(held.poll() is None for held in running), "and both worlds are still running"
    finally:
        for process in running:
            process.kill()
            process.wait(timeout=30)


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


def test_a_grader_that_outruns_its_bound_takes_its_whole_group_with_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The advertised ten minutes was a bound on nothing.

    The grader was an ordinary child with captured pipes: a timeout killed the leader alone and
    then read those pipes again with no deadline at all, so a descendant holding either of them
    kept the call, and with it a sealed episode's terminal, open indefinitely. Measured with a
    one-second bound, the call had not returned twenty seconds later and the descendant was still
    running.

    The grader leads a session of its own now, the group is what is signalled, its emptying is
    what is waited for, and the final read of the pipes is bounded like every other wait on the
    way down. The child below is the case the old code could not survive: it outlives its parent
    and it inherited both pipes."""
    script = tmp_path / "wedged_grader.py"
    told = tmp_path / "descendant.pid"
    script.write_text(
        "import subprocess, sys, time\n"
        "sys.stdin.readline()\n"
        # Inherits stdout and stderr, so it holds the pipes the caller reads.
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        f"open({str(told)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(120)\n"
    )
    monkeypatch.setattr(adapter, "runtime", lambda: Path(sys.executable))
    monkeypatch.setattr(adapter, "WORKER", script)
    monkeypatch.setattr(adapter.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))

    failure: List[BaseException] = []

    def _grade() -> None:
        try:
            adapter.grade(
                root=tmp_path, task_id="abc_1", outputs=tmp_path, ignore=[], filing={}, timeout=1.0
            )
        except BaseException as exc:  # noqa: BLE001 - the point is which one, and how soon
            failure.append(exc)

    # In a thread with a join, so a teardown that does not return fails this test rather than
    # hanging the suite the way the failure it is about hung a run.
    grading = threading.Thread(target=_grade, daemon=True)
    began = time.monotonic()
    grading.start()
    grading.join(60)
    assert not grading.is_alive(), "the grader's timeout never returned"
    assert time.monotonic() - began < 40, "it returned, but not inside anything like its bound"
    assert failure and isinstance(failure[0], adapter.WorkerError)
    assert "did not finish within" in str(failure[0])

    descendant = int(told.read_text())
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(descendant, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(descendant, signal.SIGKILL)
        raise AssertionError("what the grader started outlived the grader's own timeout")
    assert not list(tmp_path.glob("shogym-appworld-grade-*")), "and the scratch went with it"
    assert adapter.outstanding() == [], "and the ledger entry it was written down under"


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
    held = "0123456789abcdef"
    assert adapter.derived_root(first, runtime=held) != adapter.derived_root(second, runtime=held)
    assert adapter.graded_root(first, runtime=held) != adapter.graded_root(second, runtime=held)
    # And the shared base is inside the digest, which it was not: only `version.txt` and the task
    # tree were, so 134 MB of starting state every episode reads could change invisibly.
    (tmp_path / "a" / "data" / "base_dbs" / "big.jsonl").write_text("edited")
    assert adapter.corpus_digest(tmp_path / "a") != first


def test_a_cache_is_named_by_the_code_and_the_interpreter_that_filled_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The name said which corpus a cache came from and nothing about what produced it.

    Two things fill a seeded cache besides the corpus. One is this port's own generator: the
    constants were in the name, the code that reads them was not, so an edit to how a backlog is
    drawn or how a task is materialised left the key alone and compared it against rows an older
    implementation had seeded. The other is the provisioned interpreter, which is the process that
    writes a task's database log through upstream's model layer: the run fingerprint moved when it
    changed and the cache did not, so a run could hold a name saying one runtime over a world
    another had built."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    corpus = "0f0f0f0f0f0f0f0f"

    served = adapter.derived_root(corpus, runtime="aaaaaaaaaaaaaaaa")
    graded = adapter.graded_root(corpus, runtime="aaaaaaaaaaaaaaaa")
    assert served == adapter.derived_root(corpus, runtime="aaaaaaaaaaaaaaaa")
    # A different realized interpreter is a different cache, on both roots.
    assert served != adapter.derived_root(corpus, runtime="bbbbbbbbbbbbbbbb")
    assert graded != adapter.graded_root(corpus, runtime="bbbbbbbbbbbbbbbb")

    # And so is a generator whose constants did not move but whose code did. The modules whose
    # bytes are read are the ones a world is generated from, and they are read as bytes rather
    # than named, so an edit that touches no constant still moves the key.
    named = dict(adapter._generator_sources())
    assert {
        "shogym.envs.appworld.env_v1",
        "shogym.envs.appworld.ledger",
        "shogym.envs.appworld.world",
        "shogym.envs.appworld.worker",
    } <= set(named)
    assert all(path.is_file() for path in named.values())
    # `env_v1` above is the one the hand-kept list did not have. `_backlog_seed` decides the
    # backlog written into a derived task and `_world_seed` decides the episode's own generator,
    # and both live there, so an implementation change to either reused a world and a run identity
    # naming the generator before it.
    assert "_backlog_seed" in named["shogym.envs.appworld.env_v1"].read_text()

    copies = tmp_path / "generator"
    copies.mkdir()
    for name, path in adapter._generator_sources():
        (copies / name).write_bytes(path.read_bytes())
    order = tuple((name, copies / name) for name, _ in adapter._generator_sources())
    monkeypatch.setattr(adapter, "_generator_sources", lambda: order)
    try:
        adapter._generator_digest.cache_clear()
        copied = adapter.derived_root(corpus, runtime="aaaaaaaaaaaaaaaa")
        assert copied == served, "the same bytes under another directory are the same generator"
        # Every one of them, not a representative: a module in the closure that did not move the
        # name would be a module the identity does not cover.
        for name, path in order:
            before = path.read_bytes()
            path.write_bytes(before + b"\n# an edit that changes no constant\n")
            adapter._generator_digest.cache_clear()
            assert copied != adapter.derived_root(corpus, runtime="aaaaaaaaaaaaaaaa"), name
            path.write_bytes(before)
            adapter._generator_digest.cache_clear()
            assert copied == adapter.derived_root(corpus, runtime="aaaaaaaaaaaaaaaa"), name
    finally:
        # The digest is memoized for the process, so a test that patched what it reads has to
        # leave the cache empty or every later test reads this one's answer.
        adapter._generator_digest.cache_clear()


def test_the_generator_names_itself_by_what_it_imports_rather_than_by_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The closure is walked, so a module that joins the generator joins the identity by itself.

    The failure this is against is not an edit to a listed file; it is a helper written into a
    file nobody thought to list. So the walk is exercised on a package of its own, where a module
    can be added to an entry point's imports and the answer checked, which is the operation the
    real closure cannot demonstrate without editing the port.

    Three things are established. Imports are followed transitively, and through a body rather
    than only at the top level, because this port defers several of its own. A module reached only
    from outside the port's own roots is not pulled in, which is what keeps the name of a 134 MB
    cache off the serve layer's source. And a new import moves the name."""
    root = tmp_path / "src" / "shogym"
    package = root / "envs" / "appworld"
    package.mkdir(parents=True)
    (root / "envs" / "_upstream.py").write_text("HELD = 1\n")
    (root / "serve").mkdir()
    (root / "serve" / "stream.py").write_text("UNRELATED = 1\n")
    (package / "entry.py").write_text(
        "from shogym.envs.appworld import middle\n"
        "from shogym.serve.stream import UNRELATED\n"
        "def later():\n"
        "    from shogym.envs._upstream import HELD\n"
        "    return HELD\n"
    )
    (package / "middle.py").write_text("VALUE = 1\n")
    (package / "spare.py").write_text("VALUE = 2\n")
    monkeypatch.setattr(adapter, "_PACKAGE_ROOT", root)
    monkeypatch.setattr(adapter, "_GENERATOR_ENTRY_POINTS", ("shogym.envs.appworld.entry",))

    adapter._generator_sources.cache_clear()
    adapter._generator_digest.cache_clear()
    try:
        walked = dict(adapter._generator_sources())
        assert set(walked) == {
            "shogym.envs.appworld.entry",
            "shogym.envs.appworld.middle",
            "shogym.envs._upstream",
        }, "transitively, through a function body, and no further than this port's own roots"
        assert "shogym.serve.stream" not in walked
        assert "shogym.envs.appworld.spare" not in walked, "a file beside them is not an import"

        before = adapter._generator_digest()
        (package / "entry.py").write_text(
            (package / "entry.py").read_text() + "from shogym.envs.appworld import spare\n"
        )
        adapter._generator_sources.cache_clear()
        adapter._generator_digest.cache_clear()
        assert "shogym.envs.appworld.spare" in dict(adapter._generator_sources())
        assert adapter._generator_digest() != before, "a module that joined the generator moved it"
    finally:
        adapter._generator_sources.cache_clear()
        adapter._generator_digest.cache_clear()


def test_a_cache_that_was_built_from_something_else_is_refused(tmp_path: Path) -> None:
    """The name cannot cover a tree edited, moved or restored in place under the old name, and a
    cache is the material a run is scored against, so the stamp inside it is checked too."""
    root = tmp_path / "seeded"
    adapter.stamp_cache(root, source="aaaa", runtime="rrrr")
    adapter.stamp_cache(root, source="aaaa", runtime="rrrr")  # idempotent
    with pytest.raises(adapter.ProvisioningError, match="was built from"):
        adapter.stamp_cache(root, source="bbbb", runtime="rrrr")
    # The interpreter that filled it is in the stamp too, so a cache reused under a runtime that
    # did not write it is refused rather than served.
    with pytest.raises(adapter.ProvisioningError, match="was built from"):
        adapter.stamp_cache(root, source="aaaa", runtime="ssss")


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


def test_a_publish_that_fails_puts_the_displaced_tree_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replacement that fails used to destroy the thing it was replacing.

    The incumbent is renamed aside and the staged tree renamed in, and the displaced copy was then
    removed whatever had happened in between. So a failure at the second rename deleted the only
    remaining copy and left the name absent: injected once, the probe found neither tree. This
    path covers served tasks, grading views and the shared base entries, and for the last of those
    an episode already running resolves absolute names through the name that disappears.

    Two outcomes below, and the second is why the restore is not enough on its own. When the
    incumbent goes back, the caller is told the publish failed rather than left to serve an
    episode out of the entry it had already judged unusable. When the restore itself cannot
    happen, the displaced copy is kept under its own name, because it is then the only copy of
    that tree there is."""
    incumbent = tmp_path / "base_dbs"
    incumbent.mkdir()
    (incumbent / "big.jsonl").write_text("what live episodes resolve through")

    def _publishing(failing: Tuple[int, ...]) -> None:
        """Stage a replacement and publish it with those renames, by position, refused.

        Rename 1 moves the incumbent aside, rename 2 moves the staged tree in, and rename 3 is the
        restore that only happens because rename 2 did not."""
        building = world._staging(tmp_path, "base_dbs")
        building.mkdir(parents=True)
        (building / "big.jsonl").write_text("the replacement")
        real_replace = os.replace
        renames: List[int] = []

        def _fails(source: Any, destination: Any, **kwargs: Any) -> None:
            renames.append(1)
            if len(renames) in failing:
                raise OSError(errno.ENOSPC, "no space left on device")
            real_replace(source, destination, **kwargs)

        monkeypatch.setattr(world.os, "replace", _fails)
        try:
            with pytest.raises(OSError):
                world._publish(building, incumbent, replacing=True)
        finally:
            monkeypatch.setattr(world.os, "replace", real_replace)

    # The publish fails: the incumbent goes back under its own name, with its own bytes.
    _publishing(failing=(2,))
    assert incumbent.is_dir()
    assert (incumbent / "big.jsonl").read_text() == "what live episodes resolve through"
    assert not list(tmp_path.glob("*.displaced")), "and nothing is left staged aside"
    assert not list(tmp_path.glob("*.building")), "nor left half-built"

    # The restore fails too: the only copy left is retained rather than removed.
    _publishing(failing=(2, 3))
    displaced = list(tmp_path.glob("*.displaced"))
    assert not incumbent.exists(), "this is the case where the name really is gone"
    assert len(displaced) == 1, "so the tree itself has to still be somewhere"
    assert (displaced[0] / "big.jsonl").read_text() == "what live episodes resolve through"


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


def test_the_headline_is_published_under_the_name_a_row_is_summarised_from() -> None:
    """A durable row's summary is filled from `reward` or `partial_credit` and from nothing else.

    This port calls its headline `ledger_fraction`, which no stream reads a headline out of, so
    every complete run of it recorded rows that had a score object and no score in it: `reward`
    and `success` both empty, and every shipped `results.py` reporting `scored 0/N` for a run in
    which every task was graded. The headline is published under both names now. An alias and not
    a rename, because `ledger_fraction` is what the scorer, the port's README and the analysis all
    call it.

    It stays off both arms' wires, which is what makes it safe to add: `Information` reveals the
    item named `report` and `Placebo` the one named `notice`, one item each and neither of them
    this one."""
    from shogym.envs.appworld import env_v1
    from shogym.serve import stream as stream_module
    from shogym.serve.lifecycle import TerminalEvidence

    class _Env:
        _config_digest = "fingerprint"

    fb = env_v1.AppWorldEnv._verify(
        _Env(),  # pyright: ignore[reportArgumentType]
        trajectory=None,  # pyright: ignore[reportArgumentType]
        task={},
        terminated=True,
        evidence=TerminalEvidence(
            source="explicit_tool",
            status="ok",
            verdict={"ledger_fraction": 0.75, "report": "a receipt", "notice": "a digest"},
        ),
    )
    # In the shape a row records them: episode level is what a terminal may summarise or reveal.
    published = [{"name": item.name, "value": item.value, "level": "episode"} for item in fb.episode]
    summary = {item["name"]: item["value"] for item in published}
    assert summary["reward"] == summary["ledger_fraction"] == 0.75
    # Read the way a row is read, through the stream's own funnel rather than by eye.
    assert stream_module._pick_float(published, stream_module._REWARD_NAMES) == 0.75
    # And revealed by neither arm of the pair.
    from shogym.feedback.wire import NOTICE_FEEDBACK_NAME, REPORT_FEEDBACK_NAME

    for channel in (REPORT_FEEDBACK_NAME, NOTICE_FEEDBACK_NAME):
        revealed = stream_module._channel(published, channel)
        assert [item["value"] for item in revealed] == [summary[channel]]


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


def test_a_bytecode_cache_is_part_of_the_runtime_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.pyc` is executable input, and the digest used to skip every one of them.

    The defence of skipping them was that provisioning leaves the runtime holding hash-based
    caches, which the import system validates against the source's own hash. That validation
    binds the *source hash written in the cache's header* to the source, and binds nothing to the
    marshalled code after it. So a cache whose header still matched its source and whose payload
    had been changed was executable code the run's identity had never read, and the identity did
    not move.

    This builds precisely that: a real checked-hash cache, then one byte of its payload changed
    with the sixteen-byte header left exactly as it was."""
    import py_compile

    home = tmp_path / "runtime"
    python = _fake_runtime(home)
    monkeypatch.setattr(adapter, "runtime", lambda: python)
    installed = home / "lib" / "python3.12" / "site-packages" / "appworld" / "__init__.py"

    compiled = Path(
        py_compile.compile(
            str(installed),
            doraise=True,
            invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH,
        )
    )
    before = adapter.runtime_digest()
    source, original = installed.read_bytes(), compiled.read_bytes()
    edited = bytearray(original)
    edited[-1] ^= 0xFF
    compiled.write_bytes(bytes(edited))

    assert compiled.read_bytes()[:16] == original[:16], "the source hash it records is untouched"
    assert installed.read_bytes() == source, "and so is the source it claims to stand for"
    assert adapter.runtime_digest() != before


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


# ----- the bytes a derivation reads are the bytes the run was built against -----


def _derivable_corpus(root: Path, *, answer: str = "the answer", shared: str = "shared") -> Path:
    """A corpus holding one whole task: text, databases and ground truth, plus a shared base."""
    task = root / "data" / "tasks" / "abc_1"
    (task / "dbs").mkdir(parents=True)
    (task / "dbs" / "gmail.jsonl").write_text("mail")
    (task / "dbs" / "todoist.jsonl").write_text("[]")
    (task / "ground_truth").mkdir()
    (task / "ground_truth" / "answer.json").write_text(json.dumps(answer))
    (task / "specs.json").write_text(
        json.dumps(
            {
                "instruction": "do the thing",
                "datetime": "2023-05-18T12:00:00",
                "supervisor": {
                    "first_name": "Ada",
                    "last_name": "Lovelace",
                    "email": "ada@example.com",
                    "phone_number": "555",
                },
            }
        )
    )
    (root / "data" / "version.txt").write_text("0.1.0")
    (root / "data" / "base_dbs").mkdir()
    (root / "data" / "base_dbs" / "big.jsonl").write_text(shared)
    return root


class _StubBacklog:
    """Enough of a backlog for `world.seeding` to describe, and nothing that costs a second."""

    description = "a project description"
    requests: List[Any] = []


class _StubSeeder:
    """A seeding worker that writes the one file the derivation asks it to write."""

    def __init__(self) -> None:
        self.calls = 0

    def call(self, command: str, **body: Any) -> Any:
        self.calls += 1
        Path(body["into"]).write_text("[]")
        return {}

    def close(self, *, confirm: bool = False) -> None:
        pass


def _stub_env(root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """An env holding one corpus's snapshot, without the provisioning a constructor does."""
    from functools import partial

    from shogym.envs.appworld import env_v1

    snapshot = adapter.corpus_snapshot(root, task_ids=("abc_1",))
    env = env_v1.AppWorldEnv.__new__(env_v1.AppWorldEnv)
    env._original = root / "data"
    env._task_ids = ("abc_1",)
    env._specs = snapshot.specs
    env._corpus = snapshot.digest
    env._backlogs = {}
    env._blocks = 60
    env._source_check = partial(snapshot.verify, root)
    env._derived = tmp_path / "served" / "data"
    env._graded = tmp_path / "graded" / "data"
    monkeypatch.setattr(env_v1, "build_backlog", lambda seed, reference: _StubBacklog())
    return env


def test_a_task_edited_after_the_snapshot_is_not_derived_into_a_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pinning a task's authored text pinned the text and nothing else it is made of.

    The snapshot fixed the instruction, the supervisor and the date at construction; the
    databases, and the ground truth the grader diffs against, went on being read out of the live
    corpus at the moment a task was first served, which can be hours and two hundred episodes
    later. So an in-place edit in that window built a world and a grading baseline out of bytes the
    run's own identity had never seen, under an unchanged `config_digest`. The unit is checked
    before it is read, and a mismatch is an episode that does not happen rather than one that is
    scored against something else."""
    root = _derivable_corpus(tmp_path / "corpus")
    env = _stub_env(root, tmp_path, monkeypatch)
    (root / "data" / "tasks" / "abc_1" / "ground_truth" / "answer.json").write_text('"moved"')

    seeder = _StubSeeder()
    with pytest.raises(adapter.ProvisioningError, match="no longer holds"):
        env._derive(seeder, "abc_1")
    assert seeder.calls == 0, "nothing was written out of the changed corpus"
    assert not (env._derived / "tasks" / "abc_1").exists()

    # And the same env derives the task it was built against, so what is refused is the change.
    (root / "data" / "tasks" / "abc_1" / "ground_truth" / "answer.json").write_text(
        json.dumps("the answer")
    )
    env._derive(seeder, "abc_1")
    assert (env._derived / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").exists()
    assert (env._graded / "tasks" / "abc_1" / "ground_truth").exists()


def test_a_shared_entry_edited_after_the_snapshot_is_not_derived_into_a_root(
    tmp_path: Path,
) -> None:
    """The other derivation path, and the larger one.

    `derive_root` copies the base databases and the documentation, which is 134 MB of starting
    state every episode of the run reads as input. It read them from the live corpus with nothing
    saying they were still what the digest had been computed over."""
    from functools import partial

    root = _derivable_corpus(tmp_path / "corpus")
    snapshot = adapter.corpus_snapshot(root, task_ids=("abc_1",))
    check = partial(snapshot.verify, root)
    derived = tmp_path / "served" / "data"

    (root / "data" / "base_dbs" / "big.jsonl").write_text("a different starting state")
    with pytest.raises(adapter.ProvisioningError, match="no longer holds"):
        world.derive_root(original=root / "data", derived=derived, verify=check)
    assert not (derived / "base_dbs").exists()

    (root / "data" / "base_dbs" / "big.jsonl").write_text("shared")
    world.derive_root(original=root / "data", derived=derived, verify=check)
    assert (derived / "base_dbs" / "big.jsonl").read_text() == "shared"


# ----- a task is reused only when it is still the task that was derived -----


def test_a_task_that_is_no_longer_what_was_derived_is_built_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reuse was decided by two path existences, which is not a task.

    The question asked was whether the served `dbs/todoist.jsonl` and the graded `ground_truth`
    were there. That says yes to a tree with everything else missing, to one whose databases were
    changed after derivation, and to one whose read-only seal has come off; and what is reused is
    the world every episode of the task starts in and the baseline it is graded against. Each of
    the three is built here on a real derivation and the answer read.

    A rebuild rather than a refusal, because a task that is not what was derived is a task this
    can make correctly: the seeder is called again and the tree afterwards is the tree that was
    published the first time."""
    root = _derivable_corpus(tmp_path / "corpus")
    env = _stub_env(root, tmp_path, monkeypatch)
    seeder = _StubSeeder()
    env._derive(seeder, "abc_1")
    assert seeder.calls == 1
    served = env._derived / "tasks" / "abc_1"
    graded = env._graded / "tasks" / "abc_1"
    assert world.already_derived(derived=env._derived, graded=env._graded, task_id="abc_1")
    intact = {
        path: path.read_bytes()
        for path in sorted(served.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }

    def _damage(how: Any) -> None:
        world._unseal(served)
        how()
        world._seal(served)

    # A file removed, a file's bytes changed, and a node whose write bit came back. The third is
    # left unsealed on purpose: it is the state a partial chmod leaves, and a shared task an
    # episode can write to is one it can leave changed for the next episode and for the other arm
    # of its own pair.
    for damage in (
        lambda: _damage(lambda: (served / "dbs" / "gmail.jsonl").unlink()),
        lambda: _damage(lambda: (served / "dbs" / "gmail.jsonl").write_text("nail")),
        lambda: world._unseal(served / "dbs" / "gmail.jsonl"),
    ):
        damage()
        assert not world.already_derived(
            derived=env._derived, graded=env._graded, task_id="abc_1"
        )
        before = seeder.calls
        env._derive(seeder, "abc_1")
        assert seeder.calls == before + 1, "rebuilt rather than trusted"
        assert world.already_derived(derived=env._derived, graded=env._graded, task_id="abc_1")
        assert {
            path: path.read_bytes()
            for path in sorted(served.rglob("*"))
            if path.is_file() and not path.is_symlink()
        } == intact

    # And the grader's own view is held to the same standard: it is the baseline the evaluator
    # diffs against, and half of it was never checked at all.
    world._unseal(graded)
    (graded / "ground_truth" / "answer.json").write_text('"moved"')
    world._seal(graded)
    assert not world.already_derived(derived=env._derived, graded=env._graded, task_id="abc_1")
    env._derive(seeder, "abc_1")
    assert json.loads((graded / "ground_truth" / "answer.json").read_text()) == "the answer"

    # A warm episode pays for that check once per task, and this is what it costs.
    began = time.monotonic()
    for _ in range(20):
        assert world.already_derived(derived=env._derived, graded=env._graded, task_id="abc_1")
    assert (time.monotonic() - began) / 20 < 0.05


# ----- what a finalizer may do before it yields -----


async def test_a_warm_finalize_draws_its_backlog_off_the_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production path the synthetic loop test could not reach.

    A task served for the first time draws its backlog while it is being seeded. A task already on
    disk skips that, so the first draw happened at the top of `finalize`, synchronously, before
    that coroutine had yielded once: between a tenth of a second and three seconds of auditing
    depending on the task, during which every other episode on this loop is stopped and the
    `wait_for` that is supposed to be able to time this one out cannot fire."""
    from shogym.envs.appworld import env_v1, mcp_server
    from shogym.serve.lifecycle import FinalizeRequest

    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    def _slow_draw(seed: int, reference: Any) -> Any:
        time.sleep(0.4)
        return _StubBacklog()

    monkeypatch.setattr(env_v1, "build_backlog", _slow_draw)

    class _wedged:
        def close(self, *, confirm: bool = False) -> None:
            raise adapter.WorkerError("the group could not be confirmed stopped")

    env = env_v1.AppWorldEnv.__new__(env_v1.AppWorldEnv)
    env._pulse = 0
    env._backlogs = {}
    env._specs = {"abc_1": {"datetime": "2023-05-18T12:00:00"}}
    mcp_server.begin_session(
        "warm",
        mcp_server.Session(
            worker=_wedged(),
            task_id="abc_1",
            supervisor_email="ada@example.com",
            experiment="/nowhere",
        ),
    )
    beat = asyncio.create_task(ticker())
    try:
        with pytest.raises(adapter.WorkerError):
            await env.finalize(
                FinalizeRequest(
                    source="explicit_tool", finalization_id="f", session_id="warm"
                )
            )
    finally:
        beat.cancel()
        mcp_server.end_session("warm")
    # A draw made from the coroutine itself would leave this at one or two.
    assert ticks > 20, ticks


def test_session_setup_draws_the_backlog_for_a_task_that_is_already_derived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the reason the finalizer normally finds one waiting.

    The draw used to happen only on the branch that seeds a cold task, so every later episode of
    that task reached the terminal without one. The serve layer runs this hook in a thread, which
    is where an audit that can take three seconds belongs."""
    from shogym.envs.appworld import mcp_server

    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    root = _derivable_corpus(tmp_path / "corpus")
    env = _stub_env(root, tmp_path, monkeypatch)

    class _opened:
        def call(self, command: str, **body: Any) -> Any:
            return {}

        def close(self, *, confirm: bool = False) -> None:
            pass

    monkeypatch.setattr(world, "already_derived", lambda **kw: True)
    monkeypatch.setattr(world, "derive_view", lambda **kw: tmp_path / "view")
    monkeypatch.setattr(adapter.Worker, "spawn", classmethod(lambda cls, root: _opened()))

    try:
        env._begin_session("warm", {"task_id": "abc_1", "supervisor_email": "ada@example.com"})
        assert "abc_1" in env._backlogs, "the warm path drew nothing and left it to finalize"
    finally:
        mcp_server.end_session("warm")


# ----- construction is bounded, and owns what it starts -----


def test_locating_the_installed_package_is_not_waited_on_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one provisioning subprocess that had no deadline at all.

    It starts a provisioned interpreter and waits for it to import a large package. An import that
    wedges on a broken shared library or a filesystem that stopped answering held construction
    open for the life of the run, and construction runs before there is a task to file a timeout
    row against."""
    monkeypatch.setattr(adapter, "_IMPORT_TIMEOUT_SECONDS", 0.5)
    wedged = tmp_path / "python"
    wedged.write_text("#!/bin/sh\nsleep 30\n")
    wedged.chmod(0o755)

    began = time.monotonic()
    with pytest.raises(adapter.ProvisioningError, match="did not finish importing"):
        adapter._installed_package(wedged)
    assert time.monotonic() - began < 10.0


def test_a_handshake_that_stops_halfway_through_a_line_is_not_waited_on_forever() -> None:
    """Readability is not a line.

    The descriptor was waited on once and then read with `readline`, which has no deadline of its
    own: a worker that wrote half a line and then wedged made the wait bounded and the read
    unbounded, so the spawn timeout it was supposed to be under never applied to it."""
    half = subprocess.Popen(
        [sys.executable, "-c", "import sys, time; sys.stdout.write('{\"po'); "
         "sys.stdout.flush(); time.sleep(30)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        began = time.monotonic()
        assert adapter._first_line(half, 0.5) == ""
        assert time.monotonic() - began < 10.0
    finally:
        half.kill()
        half.wait()

    whole = subprocess.Popen(
        [sys.executable, "-c", "import sys, time; sys.stdout.write('{\"port\": 1}\\n'); "
         "sys.stdout.flush(); time.sleep(30)"],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert json.loads(adapter._first_line(whole, 5.0))["port"] == 1
    finally:
        whole.kill()
        whole.wait()


def _stub_worker_script(tmp_path: Path) -> Path:
    """A worker that records its pid, says whatever it is told to say, and then never stops."""
    script = tmp_path / "stub_worker.py"
    script.write_text(
        "import os, sys, time\n"
        "sys.stdin.readline()\n"
        "open(sys.argv[0] + '.pid', 'w').write(str(os.getpid()))\n"
        "sys.stdout.write(open(sys.argv[0] + '.line').read())\n"
        "sys.stdout.flush()\n"
        "time.sleep(60)\n"
    )
    return script


def test_a_handshake_that_fails_leaves_no_process_and_no_scratch_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only one of the ways this fails used to clean up after itself.

    The empty-line branch killed the worker, reaped it and removed its scratch. Everything else
    walked out of the constructor with the process still running, its whole group with it, and a
    temporary home directory on disk that nothing held a reference to: a first line that is not
    JSON, an object with no port in it, a port that is not a number. All of them happen before
    there is an episode to record a failure against, so nothing else would have said so either."""
    script = _stub_worker_script(tmp_path)
    monkeypatch.setattr(adapter, "runtime", lambda: Path(sys.executable))
    monkeypatch.setattr(adapter, "WORKER", script)
    monkeypatch.setattr(adapter.tempfile, "tempdir", str(tmp_path))
    said = Path(str(script) + ".line")
    recorded = Path(str(script) + ".pid")

    for line, failure in (
        ("not json at all\n", json.JSONDecodeError),
        ("{}\n", KeyError),
        ('{"port": "not a number"}\n', ValueError),
    ):
        said.write_text(line)
        recorded.unlink(missing_ok=True)
        with pytest.raises(failure):
            adapter.Worker.spawn(tmp_path / "root")
        pid = int(recorded.read_text())
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert not list(tmp_path.glob("shogym-appworld-*")), line


def test_a_handshake_that_ends_at_end_of_file_signals_before_anything_reaps_the_leader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one handshake failure the tests above do not cover, and the ordering it broke.

    A worker that closes its pipe without printing is a worker that died, and the branch that
    reports it read the leader's exit status to say so. ``poll`` reaps an exited child, and a
    reaped leader releases both its pid and its group number for the kernel to hand on, so the
    cleanup that followed signalled a number that might already have been somebody else's. That is
    the stale-group ordering `Worker._stop_the_group` refuses by name, reached through the one
    path nothing exercised.

    The group is signalled while the leader is still unreaped now, which is the window in which
    the number is unambiguously this worker's, and a worker that started something before dying
    has that something stopped with it."""
    script = tmp_path / "eof_worker.py"
    script.write_text(
        "import os, subprocess, sys\n"
        "sys.stdin.readline()\n"
        # A descendant that does not hold the handshake pipe, so the leader's exit is an
        # end-of-file rather than a wait for the whole spawn timeout.
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "open(sys.argv[0] + '.child', 'w').write(str(child.pid))\n"
        # And then it dies without ever saying which port it bound.
        "sys.exit(3)\n"
    )
    monkeypatch.setattr(adapter, "runtime", lambda: Path(sys.executable))
    monkeypatch.setattr(adapter, "WORKER", script)
    monkeypatch.setattr(adapter.tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))

    # `_group_of` is asked once per worker, on the worker, which is how this test gets a handle on
    # the leader without standing between the constructor and the process it starts.
    leader: List[subprocess.Popen] = []
    real_group_of = adapter._group_of
    real_killpg = os.killpg
    reaped_when_signalled: List[Optional[int]] = []

    def _capture(process: subprocess.Popen) -> Optional[int]:
        leader.append(process)
        return real_group_of(process)

    def _watch(pgid: int, how: int) -> None:
        # What the leader's status was at the instant the group was signalled. `None` is an
        # unreaped leader, which is the only state in which this number is certainly its group's.
        reaped_when_signalled.append(leader[0].returncode if leader else -1)
        real_killpg(pgid, how)

    monkeypatch.setattr(adapter, "_group_of", _capture)
    monkeypatch.setattr(adapter.os, "killpg", _watch)

    with pytest.raises(adapter.WorkerError, match="never bound a port"):
        adapter.Worker.spawn(tmp_path / "root")

    assert reaped_when_signalled, "the group was never signalled at all"
    assert reaped_when_signalled[0] is None, (
        "the group was signalled after the leader had been reaped, so the number may have been "
        "handed on before the signal reached it"
    )
    child = int(Path(str(script) + ".child").read_text())
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(child, signal.SIGKILL)
        raise AssertionError("what the worker started outlived the failed handshake")
    assert not list(tmp_path.glob("shogym-appworld-*")), "and the scratch went with it"


def test_a_leader_something_already_reaped_is_abandoned_without_a_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard itself, in the state a caller can arrive in however it got there.

    A pid is reserved until its parent reaps it, and a process group exists while any member holds
    it: once the leader has been waited on, both numbers are the kernel's to hand out again. So a
    cleanup handed an already-reaped leader may signal nothing, which is the same rule
    :meth:`Worker._stop_the_group` applies and the same one this used to break unconditionally.
    The scratch directory is still cleared, because that is this call's own to remove."""
    leader = _sleeper(120)
    pgid = adapter._group_of(leader)
    leader.kill()
    leader.wait(timeout=30)  # reaped: the number is no longer this worker's

    signalled: List[Tuple[int, int]] = []
    monkeypatch.setattr(
        adapter.os, "killpg", lambda group, how: signalled.append((group, how))
    )
    scratch = tmp_path / "shogym-appworld-reaped"
    scratch.mkdir()

    adapter._abandon(leader, pgid, scratch)

    assert signalled == [], "a released group number was signalled"
    assert not scratch.exists(), "the scratch directory is still this call's to remove"


def test_a_session_that_never_opens_leaves_no_served_view_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The episode's own copy of the task is written before there is a worker to hold it.

    `_end_session` removes what a *published* session names, and a session that never got
    published names nothing: a copy that failed part way through, a spawn that failed or a world
    that would not open left the copied task directory and the episode's output tree behind, and
    left the worker running."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    root = _derivable_corpus(tmp_path / "corpus")
    env = _stub_env(root, tmp_path, monkeypatch)
    view = tmp_path / "view"

    closed: List[bool] = []

    class _unopenable:
        def call(self, command: str, **body: Any) -> Any:
            raise adapter.WorkerError("the world would not open")

        def close(self, *, confirm: bool = False) -> None:
            closed.append(True)

    monkeypatch.setattr(world, "already_derived", lambda **kw: True)
    monkeypatch.setattr(adapter, "episode_view", lambda session_id: view)
    monkeypatch.setattr(world, "derive_view", lambda **kw: kw["view"].mkdir(exist_ok=True))
    monkeypatch.setattr(adapter.Worker, "spawn", classmethod(lambda cls, root: _unopenable()))

    with pytest.raises(adapter.WorkerError):
        env._begin_session("orphan", {"task_id": "abc_1", "supervisor_email": "ada@example.com"})
    assert closed == [True]
    assert not view.exists()
    assert not adapter.episode_outputs("orphan").exists()


# ----- what the runtime digest leaves out, and why that is safe -----


def test_a_stale_bytecode_cache_is_not_what_the_worker_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The digest skips `__pycache__`, and the usual defence of that is not quite true.

    Python discards a cache whose recorded source mtime and size have changed, which is weaker
    than "whose source has changed": an ordinary `.pyc` carrying the right two numbers is executed
    whatever is in it. This builds exactly that and proves the interpreter runs it, rather than
    asserting that it would not; then it rebuilds the same cache the way provisioning does, as a
    hash-based one the import system checks against the source's own hash, and proves the source
    is what runs.

    That is what an edited *source* costs, and it is only half the question. An edited *payload*
    under a matching header is executed by either kind, which is why the digest reads these bytes
    rather than reasoning about them: see
    `test_a_bytecode_cache_is_part_of_the_runtime_identity`."""
    import py_compile

    site = tmp_path / "site"
    package = site / "widget"
    package.mkdir(parents=True)
    source = package / "__init__.py"

    def _plant(mode: "py_compile.PycInvalidationMode") -> None:
        source.write_text("VALUE = 'cached'\n")
        py_compile.compile(str(source), doraise=True, invalidation_mode=mode)
        stamped = source.stat()
        # The same length, so the cache's recorded size still matches, and the same modification
        # time, so its recorded timestamp does too. The cache and the source now disagree, and a
        # timestamp cache has nothing that could say so.
        source.write_text("VALUE = 'source'\n")
        os.utime(source, (stamped.st_atime, stamped.st_mtime))

    def _value() -> str:
        finished = subprocess.run(
            [sys.executable, "-c", "import widget; print(widget.VALUE)"],
            capture_output=True,
            text=True,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": str(site),
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            timeout=60,
        )
        assert finished.returncode == 0, finished.stderr
        return finished.stdout.strip()

    _plant(py_compile.PycInvalidationMode.TIMESTAMP)
    assert _value() == "cached", "a metadata-valid timestamp cache is executable code"

    _plant(py_compile.PycInvalidationMode.CHECKED_HASH)
    assert _value() == "source", "a hash-based cache is checked against the source it claims"

    # And both halves of what puts the runtime in that state: every cache in it is rewritten as a
    # hash-based one at provisioning, and a worker writes none back.
    scratch = tmp_path / "scratch"
    assert adapter._worker_environment(scratch)["PYTHONDONTWRITEBYTECODE"] == "1"


def test_the_runtime_digest_covers_the_interpreter_the_venv_borrows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`bin/python` is a link out of the tree, and nothing read what it points at.

    The digest walked `site-packages` and read `pyvenv.cfg`, which names the base interpreter's
    *directory*. A directory is a label. The executable every world actually runs under was
    outside the identity, so a base interpreter replaced or repointed under the same
    configuration was one identity with the one before it."""
    home = tmp_path / "runtime"
    python = _fake_runtime(home)
    base = tmp_path / "base" / "python3.12"
    base.parent.mkdir()
    base.write_bytes(b"an interpreter\n")
    python.unlink()
    python.symlink_to(base)
    monkeypatch.setattr(adapter, "runtime", lambda: python)

    before = adapter.runtime_digest()
    assert adapter.runtime_digest() == before
    # The same length, because a digest that only noticed lengths would pass a weaker test.
    base.write_bytes(b"another one!!!\n")
    assert adapter.runtime_digest() != before


# ----- an unpack is finished or it is not -----


def test_a_half_unpacked_runtime_is_unpacked_again_rather_than_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime is published and stamped before its app sources are unpacked at all.

    Reuse was decided by whether one file, `apps/todoist/models.py`, existed. Upstream's installer
    goes on doing in-place work after the individual app files are there, so an unpack interrupted
    anywhere past that point left a runtime under a name and a stamp that both say it is finished,
    and every later construction skipped it. What says the unpack is done is now a stamp written
    after the unpacking process exited zero."""
    home = tmp_path / "runtime"
    python = _fake_runtime(home)
    installed = home / "lib" / "python3.12" / "site-packages" / "appworld"
    monkeypatch.setattr(adapter, "runtime", lambda: python)

    located: List[int] = []
    monkeypatch.setattr(
        adapter, "_installed_package", lambda _: (located.append(1), installed)[1]
    )
    ran: List[List[str]] = []

    def _unpack(command: List[str], **kw: Any) -> None:
        ran.append(command)
        (installed / "apps" / "todoist").mkdir(parents=True, exist_ok=True)
        (installed / "apps" / "todoist" / "models.py").write_text("")

    monkeypatch.setattr(adapter, "_run", _unpack)

    # Exactly what an interruption leaves behind: the file the old test read, and no more.
    (installed / "apps" / "todoist").mkdir(parents=True)
    (installed / "apps" / "todoist" / "models.py").write_text("")
    adapter.ensure_apps()
    assert [command[-1] for command in ran][:1] == ["install"], (
        "the sentinel file alone is not proof the unpack finished"
    )
    # And the interpreter is left with hash-based bytecode caches, which is what makes the runtime
    # digest's silence about `__pycache__` a true statement rather than a hopeful one.
    compiled = ran[-1]
    assert "compileall" in compiled and "checked-hash" in compiled and "-f" in compiled
    unpacked = len(ran)

    adapter.ensure_apps()
    assert len(ran) == unpacked, "and an unpack that got to the end is not repeated"
    # The warm path does not start the interpreter to find out where the package lives, which is
    # most of a second on every env construction.
    assert len(located) == 1
