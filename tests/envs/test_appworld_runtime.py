"""The appworld port's runtime seams: provisioning order, concurrency, and what blocks the loop.

Offline and upstream-free. Every failure defended here was found by running the port rather than
by reading it, and each is the kind that looks like nothing until the day it matters: a cold CI
runner that hangs instead of provisioning, two forks of one clone deriving the same task at the
same moment, a finalizer that stops every other episode while it waits, a cache path that produces
a tree of links resolving to nothing.
"""

from __future__ import annotations

import asyncio
import json
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
        def __init__(self, directory: Path) -> None:
            self.directory = Path(directory)

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
    against, which is a served episode editing the thing it is scored on."""
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
    served = derived / "tasks" / "abc_1" / "dbs" / "gmail.jsonl"
    source = task / "dbs" / "gmail.jsonl"
    baseline = graded / "tasks" / "abc_1" / "dbs" / "gmail.jsonl"
    assert served.stat().st_ino != source.stat().st_ino
    assert served.stat().st_ino != baseline.stat().st_ino

    served.write_text("rewritten by the agent")
    assert source.read_text() == "mail"
    assert baseline.read_text() == "mail"
    # And the seeded log the episode is scored against is the grader's own copy too.
    (derived / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").write_text("a different backlog")
    assert (graded / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").read_text() == "seeded"


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
    # And it moves when the way a score is read moves, which no input of the run would show.
    from shogym.envs.appworld import env_v1

    original = env_v1.SCORING_VERSION
    try:
        env_v1.SCORING_VERSION = original + 1
        assert base != run_fingerprint(pulse=0, report="graded", blocks=60)
    finally:
        env_v1.SCORING_VERSION = original


def test_what_is_recorded_and_never_revealed_stays_that_way() -> None:
    """Two halves, and both matter. A resumed directory is checked against what its rows say, so
    the fingerprint has to be on every row. It is a digest over a usually small integer pulse, so
    it must reach no agent: one handed it can enumerate pulses until it matches and then compute
    every later key. The seal note is the same shape of fact: the record needs to know that an
    episode's snapshot may have been taken of a moving world, and an agent must not learn that the
    work it retained went unstopped.

    Inference level is exactly that contract, and it is the only channel an env has that satisfies
    both: `_revealable` filters a row's feedback to episode level, so even `Immediate`, which
    reveals everything else a row records, cannot reach an inference item."""
    import inspect

    from shogym.envs.appworld import env_v1
    from shogym.serve import stream as stream_module

    verify = inspect.getsource(env_v1.AppWorldEnv._verify)
    published, _, private = verify.partition("fb.inference.append(")
    # Split at the first inference append, which is sound because everything episode-level is
    # built before it. Asserted rather than assumed, since it is what the two checks below rest on.
    assert private and "fb.episode.append(" not in private
    for name in ("config_digest", "seal_note"):
        # On the row, at the level that is recorded rather than surfaced.
        assert f'name="{name}"' in private
        # And nowhere among the items a terminating call can reveal, in either shape: the named
        # appends, and the tuples of names the loops above build episode items out of.
        assert name not in published
    # The fingerprint is also off the terminal evidence, which a direct caller reads back
    # verbatim. The seal note is not: it is a fact about how this episode's own end state was
    # taken, published beside the digests of that state, and it is key material for nothing.
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


# ----- quiescence -----

#: Runs the worker's own `quiesce` in a process of its own session, so the group it enumerates
#: holds nothing but that process and whatever the probe deliberately starts. Imported by path,
#: the way the port runs it, and it needs no `appworld`: quiesce is about processes and threads.
_QUIESCE_PROBE = """
import importlib.util, json, os, subprocess, sys, threading, time

spec = importlib.util.spec_from_file_location("shogym_worker_probe", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

episode = module.Episode()
started = []
if sys.argv[2] == "descendant":
    # A subprocess an earlier `execute` left running, in the worker's own group.
    started.append(subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"]))
if sys.argv[2] == "thread":
    stop = threading.Event()
    thread = threading.Thread(target=stop.wait, name="left-running", daemon=True)
    thread.start()
if sys.argv[2] == "unreadable":
    # No `ps` on the path, so the process table genuinely cannot be read.
    os.environ["PATH"] = ""

began = time.monotonic()
answer = episode.quiesce({})
answer["elapsed"] = time.monotonic() - began
answer["escaped"] = [p.pid for p in started if p.poll() is None]
print(json.dumps(answer))
"""


def _quiesce(mode: str) -> Dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, "-c", _QUIESCE_PROBE, str(adapter.WORKER), mode],
        capture_output=True,
        text=True,
        timeout=120,
        # Its own group, which is what a real worker has: `quiesce` enumerates the group it is in.
        start_new_session=True,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_quiescing_an_episode_that_left_nothing_running_is_prompt() -> None:
    """Every submission pays for this, and it used to pay ten seconds.

    `quiesce` enumerates its own process group with `ps`, and `ps` inherited that group, so it
    appeared in the very listing it was producing. The group was therefore never seen empty: both
    five-second settle windows ran to the end on every submission, whether or not the episode had
    retained anything, and a terminal that would have been prompt was ten seconds closer to its
    deadline. The helper runs in a session of its own now."""
    answer = _quiesce("clean")
    assert answer["quiesced"] is True
    assert answer["stopped"] == 0
    assert answer["descendants"] == []
    assert answer["note"] == ""
    # Well under a second. The old path took a hair over ten.
    assert answer["elapsed"] < 1.0


def test_a_process_the_episode_left_running_is_stopped_before_the_state_is_read() -> None:
    """The thing quiescence is for: a subprocess an earlier `execute` started is still writing
    while the filing is taken and the snapshot saved, so the evaluator scores a state no instant of
    the episode ever had."""
    answer = _quiesce("descendant")
    assert answer["stopped"] >= 1
    assert answer["escaped"] == []
    assert answer["descendants"] == []
    assert answer["quiesced"] is True


def test_a_thread_the_episode_left_running_is_reported_rather_than_implied() -> None:
    """A thread is not signallable, so this is the one part of the execution domain quiescence
    cannot stop. It used to come back as `stopped: 0`, which is the same answer a clean episode
    gives, so a snapshot taken of a moving world was indistinguishable from one taken of a still
    one. It says so now, and `read` proves the pair separately."""
    answer = _quiesce("thread")
    assert answer["threads"] == ["left-running"]
    assert answer["quiesced"] is False
    assert "cannot be signalled" in answer["note"]


def test_a_process_table_that_cannot_be_read_is_not_an_empty_one() -> None:
    """Failure used to be reported as success. An unreadable table meant an empty list of
    descendants, which is the answer a quiet group gives, so the seal claimed a property it had no
    evidence for."""
    answer = _quiesce("unreadable")
    assert answer["quiesced"] is False
    assert "could not be read" in answer["note"]


# ----- the pair the evaluator is handed -----


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


def test_the_filing_and_the_snapshot_are_proved_to_be_one_instant(tmp_path: Path) -> None:
    """A still world reports a stable pair, which is what makes the unstable answer mean anything.

    The filing is read off the live models and the snapshot is written from those same models, so
    the two are one instant only if nothing touched a model between them. Quiescence stops
    processes and cannot stop threads, so this is measured rather than assumed: the state is saved
    and digested on both sides of the read."""
    module = _worker_module()
    episode = module.Episode()
    episode.world = _StubWorld(tmp_path, ["one instant"])
    answer = episode.read(
        {"supervisor_email": "who@example.com", "project": "p", "title": "t", "label": "l"}
    )
    assert answer["stable"] is True
    assert answer["filing"]["filed"] is False


def test_a_world_that_moved_during_the_read_says_so_rather_than_being_scored(
    tmp_path: Path,
) -> None:
    """The failure quiescence cannot prevent, made visible.

    A thread an earlier `execute` left running is not signallable, so it can write while the filing
    is being taken. The filing then describes one moment and the snapshot the grader replays
    describes another, and the episode's number is about a state no instant of it ever had. The
    stub writes between the two saves, which is exactly that window."""
    module = _worker_module()
    episode = module.Episode()
    contents = ["before the read"]
    episode.world = _StubWorld(tmp_path, contents)

    original = module._read_filing

    def _read_filing(models: Any, body: Dict[str, Any]) -> Dict[str, Any]:
        # The retained thread, writing while the filing is taken.
        contents.append("after the read")
        return original(models, body)

    module._read_filing = _read_filing
    answer = episode.read(
        {"supervisor_email": "who@example.com", "project": "p", "title": "t", "label": "l"}
    )
    assert answer["stable"] is False
    # And the digest published is of the state the grader will actually replay.
    assert answer["world_digest"] == module._directory_digest(str(tmp_path))


def test_a_seal_that_was_not_clean_says_which_half_was_not() -> None:
    """Both halves reach the record, and a clean seal is the empty string, so any text is signal.

    A quiesce that raised used to be swallowed whole: the episode was graded and nothing anywhere
    said that its execution domain had never been stopped."""
    from shogym.envs.appworld import env_v1

    clean = {"quiesced": True, "note": ""}
    assert env_v1._seal_note(clean, {"stable": True}) == ""

    threaded = {"quiesced": False, "note": "thread(s) still running"}
    assert "not quiesced" in env_v1._seal_note(threaded, {"stable": True})
    assert "thread(s) still running" in env_v1._seal_note(threaded, {"stable": True})

    # A worker too wedged to answer at all.
    refused = {"quiesced": False, "note": "quiesce failed: WorkerError"}
    assert "quiesce failed" in env_v1._seal_note(refused, {"stable": True})

    both = env_v1._seal_note(threaded, {"stable": False})
    assert "not quiesced" in both and "different moments" in both
