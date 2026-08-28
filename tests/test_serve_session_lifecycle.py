"""One session, one release, and neither hook on the shared loop.

An env's ``_begin_session`` and ``_end_session`` are the two places the serve layer hands control
to code that can do anything: spawn a world in another process, take a port, copy a corpus, signal
and reap a child. These tests are about what happens when that code is slow and the caller is not
patient, which is where the serve layer used to release one episode's resources twice:

* a cancelled or failed setup closed the env beside the rollback it had just arranged,
* and a teardown that gave up waiting left its hook running while ``close`` entered the same hook
  again through ``env.close()``.

They are timing reproductions, written the way the reviewer wrote them: a bound far shorter than
the hook, and an assertion about how many times the hook was entered and how far the caller was
held. The counting is done inside the env, because "how many releases" is a property of the env's
own hook and not of anything the serve layer reports about itself.

The last group is about the loop rather than the session: constructing an env is blocking work
too, and a stream that builds one on the loop stops every other episode it is serving.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from shogym.serve import ServedEpisode
from shogym.serve import episode as episode_module
from shogym.serve.lifecycle import FinalizationRecord, FinalizationStore
from shogym.serve.stream import TaskRef, TaskStream

from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [{"id": "q0", "question": "2+2?", "answer": "4"}]


class _SlowSessionEnv(_FixtureScoreEnv):
    """A score env whose session hooks take a measurable amount of time and count their entries.

    ``releases`` gets one entry per *entry* into ``_end_session``, not per completion, which is
    the distinction the double-release bug lived in: the old failure path entered the hook a
    second time while the first was still inside it, so counting completions would have counted
    two of them as one. ``peak_releases`` records how many were ever inside at once."""

    def __init__(
        self,
        *,
        begin_seconds: float = 0.0,
        end_seconds: float = 0.0,
        begin_error: Optional[BaseException] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._begin_seconds = begin_seconds
        self._end_seconds = end_seconds
        self._begin_error = begin_error
        self.begins: List[float] = []
        self.begin_returned: Optional[float] = None
        self.releases: List[Dict[str, Any]] = []
        self.releases_finished = 0
        self.peak_releases = 0
        self._inside = 0
        self._counting = threading.Lock()
        super().__init__(tasks=tasks or list(TASKS))

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        self.begins.append(time.perf_counter())
        time.sleep(self._begin_seconds)
        if self._begin_error is not None:
            self.begin_returned = time.perf_counter()
            raise self._begin_error
        super()._begin_session(session_id, task)
        self.begin_returned = time.perf_counter()

    def _end_session(self, session_id: str) -> None:
        with self._counting:
            self._inside += 1
            self.peak_releases = max(self.peak_releases, self._inside)
            self.releases.append(
                {"at": time.perf_counter(), "thread": threading.current_thread().name}
            )
        try:
            time.sleep(self._end_seconds)
            super()._end_session(session_id)
            with self._counting:
                self.releases_finished += 1
        finally:
            with self._counting:
                self._inside -= 1


def _until(predicate: Any, seconds: float = 5.0) -> bool:
    """Wait for something a thread is doing, without a loop and without a sleep that guesses."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


# ----- a setup that was abandoned is rolled back once, by one owner -----


async def test_a_cancelled_setup_releases_the_session_exactly_once() -> None:
    # The reproduction: give up on `open_env` while the setup hook is still inside the env, and
    # count the releases. Two of them, one starting before setup had finished, is the failure:
    # `Env.begin_session` records the session id before entering the hook, so the failure path's
    # `env.close()` used to end a session `_begin_session` was still in the middle of creating,
    # and the rollback arranged for when the hook landed then ended it again.
    env = _SlowSessionEnv(begin_seconds=0.3)
    opening = asyncio.ensure_future(ServedEpisode.open_env(env, task=0))
    await asyncio.sleep(0.02)  # inside the hook, before it can have returned
    assert env.begin_returned is None
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening

    assert len(env.releases) == 1, env.releases
    assert env.peak_releases == 1
    assert env.releases_finished == 1
    # And it ran *after* the hook, not beside it. The release is queued behind the setup on the
    # episode's one hook thread, so this ordering is the thread's rather than a race that went
    # the right way this time.
    assert env.begin_returned is not None
    assert env.releases[0]["at"] >= env.begin_returned


async def test_a_setup_that_raises_releases_the_session_exactly_once() -> None:
    # The same double release by the other route: the hook fails rather than being abandoned, so
    # the failure is immediate and the rollback and the `env.close()` beside it used to land on
    # top of each other rather than one after the other.
    # The release is slow so a second one has to overlap it rather than follow it: two releases
    # that happen to run one after the other still leave a session released twice, and the
    # ordering they arrive in is not something either caller decided.
    env = _SlowSessionEnv(begin_error=RuntimeError("boom during setup"), end_seconds=0.2)
    with pytest.raises(RuntimeError, match="boom during setup"):
        await ServedEpisode.open_env(env, task=0)
    # Give a second release somewhere to arrive from before counting. The one this replaced was a
    # task on the loop, so counting the moment the error surfaced would have counted a release
    # that had not been scheduled yet as no release at all.
    for _ in range(5):
        await asyncio.sleep(0)
    await asyncio.sleep(0.3)
    assert len(env.releases) == 1, env.releases
    assert env.peak_releases == 1


def test_a_cancelled_setups_rollback_outlives_the_loop_it_was_cancelled_on() -> None:
    # Sync, and driving the loop by hand, because the property under test is what happens with no
    # loop at all. A rollback that is a callback on the event loop returns early when there is no
    # running loop to schedule on, which leaves a session the hook is still creating with nothing
    # left to release it. This rollback is the hook thread's work, so ending the loop changes
    # nothing about whether it happens.
    env = _SlowSessionEnv(begin_seconds=0.5)
    loop = asyncio.new_event_loop()
    # A task still parked when the loop closes is exactly the state under test, and asyncio
    # reports it at collection; silenced here so the report is not read as a failure.
    loop.set_exception_handler(lambda _loop, _context: None)

    async def give_up() -> None:
        opening = loop.create_task(ServedEpisode.open_env(env, task=0))
        await asyncio.sleep(0.05)
        opening.cancel()
        # Enough turns for the handler to arrange the rollback and no more: the loop ends with
        # the hook still running and the caller's task parked in a wait that will never finish.
        for _ in range(5):
            await asyncio.sleep(0)

    try:
        loop.run_until_complete(give_up())
    finally:
        loop.close()

    assert env.releases == []  # the hook has not landed, so nothing has been released yet
    assert _until(lambda: len(env.releases) == 1), env.releases
    assert _until(lambda: env.releases_finished == 1)
    assert env.peak_releases == 1


# ----- a teardown that gave up waiting does not let close start a second release -----


async def test_a_timed_out_teardown_is_abandoned_and_never_reissued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reviewer's reproduction, with their shape: a bound far shorter than the hook. The old
    # behaviour was two releases at concurrency two, and a `close()` that took as long as the
    # hook, because the base env counts a session as open until its hook returns and `env.close()`
    # therefore entered the same hook again on the shared loop.
    monkeypatch.setattr(episode_module, "_END_SESSION_SECONDS", 0.02)
    env = _SlowSessionEnv(end_seconds=0.4)
    ep = await ServedEpisode.open_env(env, task=0)

    sealed = time.perf_counter()
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    # The terminal call is bounded by the timeout, not by the hook: teardown abandons the wait.
    assert time.perf_counter() - sealed < 0.3

    closing = time.perf_counter()
    await ep.close()
    closed = time.perf_counter() - closing

    assert len(env.releases) == 1, env.releases
    assert env.peak_releases == 1
    # `close()` is bounded too. The release is still inside the hook when it runs, and it neither
    # waits for it a second time nor enters it again.
    assert closed < 0.3, closed
    # The release the bound abandoned still finishes, in its own thread, on its own time.
    assert _until(lambda: env.releases_finished == 1)
    assert len(env.releases) == 1, env.releases


async def test_an_ordinary_episode_releases_its_session_exactly_once() -> None:
    # The same count on the path nothing goes wrong on, so "exactly one" is a property of the
    # design rather than of the failure handling: seal, finalize, teardown, close.
    env = _SlowSessionEnv()
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    await ep.close()
    await ep.close()  # idempotent, and still one release
    assert len(env.releases) == 1, env.releases
    assert env.peak_releases == 1


async def test_a_wedged_teardown_does_not_stop_another_episodes_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A deadline is an `asyncio.wait_for` on the loop, so an env that releases slowly on the loop
    # stops every watchdog in the process. Counted as ticks rather than measured as latency: a
    # release made from the coroutine would leave this at one or two.
    monkeypatch.setattr(episode_module, "_END_SESSION_SECONDS", 5.0)
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    env = _SlowSessionEnv(end_seconds=0.3)
    ep = await ServedEpisode.open_env(env, task=0)
    beat = asyncio.ensure_future(ticker())
    try:
        await ep.call(SUBMIT_TOOL, {"answer": "4"})
        await ep.close()
    finally:
        beat.cancel()
    assert ticks > 20, ticks
    assert len(env.releases) == 1


# ----- constructing an env is blocking work, and it is not the loop's -----


async def test_a_slow_construction_does_not_stop_the_rest_of_the_loop(tmp_path: Path) -> None:
    # `TaskStream.get_task` used to evaluate the env factory in the argument list of its await,
    # which is on the loop. For an env that provisions a corpus and walks two views of it, a cold
    # first construction is seconds of it, and every other episode's deadline is held for all of
    # them.
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    def slow_factory(_name: str) -> _FixtureScoreEnv:
        time.sleep(0.3)
        return _FixtureScoreEnv(tasks=list(TASKS))

    stream = TaskStream(slow_factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov")
    beat = asyncio.ensure_future(ticker())
    try:
        async with stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    finally:
        beat.cancel()
    assert ticks > 20, ticks


async def test_start_builds_its_env_off_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    # The same for the single-episode entry point, whose `make()` was on the loop for the same
    # reason and with the same effect.
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    def slow_make(_name: str, config: Optional[Dict[str, Any]] = None) -> _FixtureScoreEnv:
        time.sleep(0.3)
        return _FixtureScoreEnv(tasks=list(TASKS))

    monkeypatch.setattr(episode_module, "make", slow_make)
    beat = asyncio.ensure_future(ticker())
    try:
        ep = await ServedEpisode.start(ENV_NAME, task=0)
        await ep.close()
    finally:
        beat.cancel()
    assert ticks > 20, ticks


async def test_an_abandoned_construction_is_closed_rather_than_left_running() -> None:
    # Offloading the factory introduces a moment the synchronous call did not have: the env is
    # built and its caller is gone. A thread cannot be cancelled, so the env exists either way,
    # and something has to close it.
    closed = threading.Event()

    class _NoticingEnv(_FixtureScoreEnv):
        async def _close(self) -> None:
            closed.set()

    def slow_factory() -> _NoticingEnv:
        time.sleep(0.3)
        return _NoticingEnv(tasks=list(TASKS))

    building = asyncio.ensure_future(episode_module._built(slow_factory))
    await asyncio.sleep(0.02)
    building.cancel()
    with pytest.raises(asyncio.CancelledError):
        await building
    assert _until(closed.is_set), "the env nobody took was never closed"


# ----- restart recovery is a startup question, asked once -----


def test_recovery_reads_one_store_once_per_process(tmp_path: Path) -> None:
    # Recovery reads every record in the store, and the no-trace store is one directory shared by
    # every session ever run on the machine. Asked once per episode it is O(the machine's whole
    # history) per episode: on a developer machine with 79k records that is seconds of reading
    # JSON before an episode does anything, and a suite that opens eighty of them spends minutes
    # in it. The answer cannot change because this process opened another episode.
    store = FinalizationStore(tmp_path / "finalizations")
    store.write(
        FinalizationRecord(
            session_id="prior", finalization_id="f-crash", status="PENDING",
            source="explicit_tool",
        )
    )
    assert [r.finalization_id for r in store.recover_once()] == ["f-crash"]
    assert store.read("f-crash").status == "FAILED"
    # Asked again, by this object and by any other naming the same directory, it does not read.
    assert store.recover_once() == []
    assert FinalizationStore(tmp_path / "finalizations").recover_once() == []
    # And the underlying pass is still there for a caller that means to run one.
    assert store.recover() == []


async def test_opening_many_episodes_reads_the_store_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads = {"n": 0}
    real = FinalizationStore.load_all

    def counting(self: FinalizationStore) -> Any:
        reads["n"] += 1
        return real(self)

    monkeypatch.setattr(FinalizationStore, "load_all", counting)
    for _ in range(3):
        ep = await ServedEpisode.open_env(
            _FixtureScoreEnv(tasks=list(TASKS)),
            task=0,
            trace_path=tmp_path / "run.jsonl",
        )
        await ep.close()
    assert reads["n"] == 1, reads
