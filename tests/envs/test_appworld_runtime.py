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

    The image and the corpus live under one cache directory and both were guarded by an
    ``flock`` on it. Two ``flock`` calls through two opens are two lock requests even inside one
    process, so provisioning the interpreter from inside the corpus's lock is a process waiting on
    itself, with no error and no timeout. The fix is an ordering, so the test is over the
    ordering: nothing may be locked while that same path is already held. The image is what the
    bundle is opened by, so it is built before the corpus lock is taken."""
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
    monkeypatch.setattr(adapter, "ensure_image", lambda: ordered.append("image") or "image")

    def _fetch(root: Path) -> None:
        ordered.append("corpus")
        (root / "data" / "tasks").mkdir(parents=True)

    monkeypatch.setattr(adapter, "_fetch_corpus", _fetch)
    adapter.ensure_corpus()
    # And the ordering is the one the fix is: the interpreter exists before the corpus is fetched.
    assert ordered == ["image", "corpus"]
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
    # A shared entry this port names: the derived root holds what `world.SHARED_ENTRIES` says
    # and nothing else, so a fixture inventing a name would be testing a tree production
    # never builds.
    (original / "version.txt").write_text("base databases")
    derived = adapter.cache_root() / "derived" / "data"
    world.derive_root(original=original, derived=derived)
    materialised = derived / "version.txt"
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
    # Matched on the name rather than on the call written on one line: this branch's appends carry
    # the row's own step, which puts the name on a line of its own.
    inference = verify.count("fb.inference.append(")
    assert inference and verify.count('name="config_digest"') == inference
    assert 'EpisodeFeedback(name="config_digest"' not in verify
    assert verify.count('"config_digest"') == verify.count('name="config_digest"')
    # And off the terminal evidence, which a direct caller reads back verbatim.
    assert "config_digest" not in inspect.getsource(env_v1.AppWorldEnv.finalize)
    # The filter that makes the level mean what it is being relied on to mean: what a terminating
    # call can reveal is the row's episode-level items and nothing else.
    revealable = inspect.getsource(stream_module._revealable)
    assert "_EPISODE_LEVEL" in revealable
    assert stream_module._EPISODE_LEVEL == "episode"


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
    # cheap enough to build per episode. What the link names is the derived tree, which has no
    # answers in it, and that is the property being asserted here. It is shared and writable by
    # this uid, so an episode that goes looking can leave something in it for the next one, which
    # is the host worker's boundary and what shojin-lab/shogym#140 closes by binding it read-only.
    assert (first / "data" / "base_dbs").is_symlink()
    shared = first / "data" / "base_dbs" / "big.jsonl"
    assert shared.read_text() == "shared base"
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


def test_a_worker_container_is_given_its_whole_environment_rather_than_a_filtered_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Agent-authored code runs as the worker, so everything the serving process exported used to
    be one ``os.environ`` away from it unless an allow-list took it away first. An allow-list is
    the same list with a worse failure mode: a name nobody thought of still gets through.

    A container is given the image's own environment and what ``docker run -e`` names, and nothing
    else, so this is a test that the port never *offers* a host variable rather than a test that
    it removes the ones it knows about."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-also-secret")
    monkeypatch.setenv("SHOGYM_APPWORLD_PROV", "/runs/somewhere")
    given: Dict[str, str] = adapter._worker_environment("/corpus")
    assert set(given) == {"APPWORLD_ROOT", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE"}
    assert given["APPWORLD_ROOT"] == "/corpus"
    assert "sk-secret" not in "".join(given.values())


def _worker_module() -> Any:
    """The worker, imported by path the way the port runs it.

    `read` and `quiesce` are about files, processes and threads, so neither needs `appworld` and
    both can be exercised under the interpreter this suite runs on."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("shogym_worker_under_test", adapter.WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubWorld:
    """A world whose saved state is whatever `contents` says at the moment it is asked."""

    class _Absent:
        """The model layer, with nobody in it: `_read_filing` gives up at the first lookup."""

        @staticmethod
        def find_one(**_: Any) -> None:
            return None

    def __init__(self, home: Path, contents: List[str]) -> None:
        self.output_db_home_path_on_disk = str(home)
        self.models = self
        self.todoist = self
        self.User = self._Absent
        self._home = home
        self._contents = contents

    def reset_db_home_path(self) -> None:
        pass

    def _save_state(self, directory: str) -> None:
        (Path(directory) / "todoist.jsonl").write_text(self._contents[-1])



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
    writes a task's database log through upstream's model layer: the run fingerprint moves with
    the runtime pin and the cache did not, so a run could hold a name saying one runtime over a
    world another had built."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    corpus = "0f0f0f0f0f0f0f0f"

    served = adapter.derived_root(corpus, runtime="aaaaaaaaaaaaaaaa")
    graded = adapter.graded_root(corpus, runtime="aaaaaaaaaaaaaaaa")
    assert served == adapter.derived_root(corpus, runtime="aaaaaaaaaaaaaaaa")
    # A different runtime pin is a different cache, on both roots.
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


# ----- a derivation publishes rather than building in place -----


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
        # Whatever is published is complete before it has a name.
        assert (building / world._COMPLETE).exists()
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
    """The row a finalize that failed leaves, and what used to be on it.

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


# ----- what the identity covers -----


def test_the_fingerprint_covers_the_runtime_pin_and_the_authored_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four inputs the digest claimed to cover and did not.

    The runtime pin names the release, the commit and the interpreter series the worlds run
    under. The guide, the tool guide and the appended paragraph are the text every episode is
    given, which is authored treatment rather than scenery. And the generator digest already
    decided which derived cache was served."""
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


# ----- one reading of one corpus -----


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


# ----- exclusion the filesystem cannot give -----


def test_a_mount_that_cannot_lock_refuses_the_builders_and_still_serves_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback the concurrency test above never exercises, on both sides of the fork.

    `_locked` yielded with no exclusion at all when the filesystem could not provide `flock`,
    which is right for the upstream-source download it was written for: that publishes by one
    atomic rename and a loser validates the winner, so the loss is redundant work. It is wrong for
    the builders here. The corpus is the material every score is computed against, and the
    permission windows open a published directory and seal it again, which a second process inside
    them closes under the first's feet.

    The runtime builder this test also covered upstream is gone: there is no virtual environment
    on the host any more, and the image is built by a daemon that serializes its own tags.

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
    # The corpus builder, past the image it makes sure of first.
    monkeypatch.setattr(adapter, "ensure_image", lambda: "an-image")
    with pytest.raises(_upstream.ExclusionUnavailable):
        adapter.ensure_corpus()

    original = tmp_path / "corpus" / "data"
    (original / "tasks" / "abc_1" / "dbs").mkdir(parents=True)
    (original / "tasks" / "abc_1" / "ground_truth").mkdir()
    (original / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").write_text("")
    (original / "version.txt").write_text("base databases")
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
    """A stand-in for the seeding container, which is what writes a task's log on this branch.

    Upstream's version of this test hands a worker into `_derive`; here seeding is a short-lived
    container of its own and `adapter.seed` is what starts it, so the count is of calls to that."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, root: Path, source_dbs: Path, into: Path, rows: Any) -> Any:
        self.calls += 1
        into.write_text("[]")
        return {}


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
    monkeypatch.setattr(adapter, "seed", seeder)
    with pytest.raises(adapter.ProvisioningError, match="no longer holds"):
        env._derive("abc_1")
    assert seeder.calls == 0, "nothing was written out of the changed corpus"
    assert not (env._derived / "tasks" / "abc_1").exists()

    # And the same env derives the task it was built against, so what is refused is the change.
    (root / "data" / "tasks" / "abc_1" / "ground_truth" / "answer.json").write_text(
        json.dumps("the answer")
    )
    env._derive("abc_1")
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
        # A container this finalization cannot stop. `settle` answers first on this branch,
        # because a terminal may overtake an ordinary call and a world in the middle of the save
        # upstream ends every block with must not be removed underneath it.
        poisoned = ""

        def settle(self, timeout: float) -> bool:
            return True

        def close(self, *, confirm: bool = False) -> None:
            raise adapter.WorkerError("the container could not be confirmed removed")

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
    monkeypatch.setattr(
        adapter.Worker, "spawn", classmethod(lambda cls, root, **kw: _opened())
    )

    try:
        env._begin_session("warm", {"task_id": "abc_1", "supervisor_email": "ada@example.com"})
        assert "abc_1" in env._backlogs, "the warm path drew nothing and left it to finalize"
    finally:
        mcp_server.end_session("warm")


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
    # Seeding is a container of its own on this branch, so what is counted is the call that
    # starts it rather than a worker handed in.
    monkeypatch.setattr(adapter, "seed", seeder)
    env._derive("abc_1")
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
        env._derive("abc_1")
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
    env._derive("abc_1")
    assert json.loads((graded / "ground_truth" / "answer.json").read_text()) == "the answer"

    # A warm episode pays for that check once per task, and this is what it costs.
    began = time.monotonic()
    for _ in range(20):
        assert world.already_derived(derived=env._derived, graded=env._graded, task_id="abc_1")
    assert (time.monotonic() - began) / 20 < 0.05


def test_a_task_member_this_port_does_not_name_is_not_derived_or_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The boundary inside a task was the same denylist the shared half used to be.

    A derived task is mounted whole at `/corpus/data/tasks/<id>`, and what was copied into it was
    every member of the source task except `dbs` and the exact name `ground_truth`. Since
    `APPWORLD_ROOT` takes any directory with a `data/tasks` in it, an artifact somebody's own
    corpus carried beside the three the pinned bundle ships was derived and mounted because it was
    neither of the two names that had been thought of. The answers stay out by not being on the
    list now rather than by being remembered, which is the difference between a boundary and a
    habit."""
    root = _derivable_corpus(tmp_path / "corpus")
    source = root / "data" / "tasks" / "abc_1"
    # What a custom corpus might carry beside the three, and what the pinned one does not.
    (source / "notes.md").write_text("somebody's own working notes")
    (source / "solution").mkdir()
    (source / "solution" / "walkthrough.txt").write_text("how this task is solved")

    env = _stub_env(root, tmp_path, monkeypatch)
    monkeypatch.setattr(adapter, "seed", _StubSeeder())
    env._derive("abc_1")

    def _corpus_members(tree: Path) -> List[str]:
        """What came out of the corpus, which is everything but this port's own bookkeeping."""
        return sorted(
            entry.name for entry in tree.iterdir() if not entry.name.startswith(".shogym")
        )

    served = env._derived / "tasks" / "abc_1"
    assert _corpus_members(served) == ["dbs", "specs.json"]
    assert not (served / "notes.md").exists()
    assert not (served / "solution").exists()
    # The answers are absent for the same reason rather than a different one.
    assert not (served / "ground_truth").exists()

    # And the per-episode view an episode is actually given carries only what the task holds.
    view = world.derive_view(derived=env._derived, view=tmp_path / "view", task_id="abc_1")
    assert _corpus_members(view / "data" / "tasks" / "abc_1") == ["dbs", "specs.json"]

    # The grader's own view is built from the derived task, so it inherits the same list, plus the
    # answers it exists to hold.
    graded = env._graded / "tasks" / "abc_1"
    assert _corpus_members(graded) == ["dbs", "ground_truth", "specs.json"]
