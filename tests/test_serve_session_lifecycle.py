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
import contextlib
import contextvars
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from shogym.serve import ServedEpisode
from shogym.serve import episode as episode_module
from shogym.serve.lifecycle import FinalizationRecord, FinalizationStore
from shogym.serve.stream import Immediate, TaskRef, TaskStream

from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [{"id": "q0", "question": "2+2?", "answer": "4"}]

#: A request-scoped value of the kind a caller sets before opening an episode: a tenant, an auth
#: subject, a trace id. The session hooks run in a thread, and a thread that does not carry the
#: caller's context reads the default here and begins or releases the wrong one.
_TENANT: contextvars.ContextVar[str] = contextvars.ContextVar("tenant", default="unset")


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
        end_error: Optional[BaseException] = None,
        describe_error: Optional[BaseException] = None,
        close_error: Optional[BaseException] = None,
        close_seconds: float = 0.0,
        bind_loop: bool = False,
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._begin_seconds = begin_seconds
        self._end_seconds = end_seconds
        self._begin_error = begin_error
        self._end_error = end_error
        self._close_error = close_error
        self._close_seconds = close_seconds
        self.begins: List[float] = []
        self.begin_returned: Optional[float] = None
        self.releases: List[Dict[str, Any]] = []
        self.releases_finished = 0
        self.peak_releases = 0
        self.closed = threading.Event()
        self.closed_during_release = False
        self.closed_after_sessions: Optional[bool] = None
        self.closed_during_begin = False
        self.sessions_closed = threading.Event()
        self.close_entries = 0
        self.peak_closes = 0
        self.closed_on_a_foreign_loop = False
        self.begin_context: Optional[str] = None
        self.end_context: Optional[str] = None
        self.owner = asyncio.get_running_loop() if bind_loop else None
        self._inside_close = 0
        self.describe_error = describe_error
        self._inside = 0
        self._counting = threading.Lock()
        super().__init__(tasks=tasks or list(TASKS))

    def describe(self, task_id: Optional[str] = None) -> Any:
        if self.describe_error is not None:
            raise self.describe_error
        return super().describe(task_id)

    async def _close(self) -> None:
        """The env-level half of cleanup, which releasing a session is not. A `close()` that runs
        while a release is still inside `_end_session` is tearing down underneath it, and a
        second one running beside the first tears the same thing down twice."""
        with self._counting:
            if self._inside:
                self.closed_during_release = True
            if self.closed_after_sessions is None:
                self.closed_after_sessions = self.sessions_closed.is_set()
            self._inside_close += 1
            self.close_entries += 1
            self.peak_closes = max(self.peak_closes, self._inside_close)
        try:
            if self.owner is not None and asyncio.get_running_loop() is not self.owner:
                self.closed_on_a_foreign_loop = True
                raise RuntimeError("closed on a loop that does not own this env's resources")
            # A yield, so a second close arriving while this one is in flight overlaps it rather
            # than queueing behind it by accident.
            await asyncio.sleep(self._close_seconds or 0.05)
            if self._close_error is not None:
                raise self._close_error
        finally:
            with self._counting:
                self._inside_close -= 1
            self.closed.set()

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        if self.close_entries:
            self.closed_during_begin = True
        self.begin_context = _TENANT.get()
        self.begins.append(time.perf_counter())
        time.sleep(self._begin_seconds)
        if self._begin_error is not None:
            self.begin_returned = time.perf_counter()
            raise self._begin_error
        super()._begin_session(session_id, task)
        self.begin_returned = time.perf_counter()

    def _end_session(self, session_id: str) -> None:
        self.end_context = _TENANT.get()
        with self._counting:
            self._inside += 1
            self.peak_releases = max(self.peak_releases, self._inside)
            self.releases.append(
                {"at": time.perf_counter(), "thread": threading.current_thread().name}
            )
        try:
            time.sleep(self._end_seconds)
            if self._end_error is not None:
                raise self._end_error
            super()._end_session(session_id)
            with self._counting:
                self.releases_finished += 1
        finally:
            with self._counting:
                self._inside -= 1


async def _awaited(predicate: Any, seconds: float = 5.0) -> bool:
    """Wait for something the loop itself has to run.

    A close arranged from the hook thread is scheduled back onto the loop that built the env, so
    a caller that blocks that loop waiting for it is waiting for work it is preventing. This
    yields instead."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


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
    # And the env itself was closed, which releasing its session is not: `open_env` promises that
    # ownership transfers and a setup failure closes what it was given, and an env's `_close` is
    # where a constructor's processes, clients and temp directories go.
    assert _until(env.closed.is_set), "the env was released but never closed"
    assert env.closed_during_release is False


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
    assert _until(env.closed.is_set), "the env was released but never closed"
    assert env.closed_during_release is False


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
    # And the env is closed too, which is the half that used to be decided by the coroutine after
    # the wait. A coroutine parked on a loop that then closes never decides anything, so the
    # session was released and the env left open. Both halves are queued when the rollback is
    # made, so a loop that goes away cannot orphan the second one.
    assert _until(env.closed.is_set), "the loop went away and took the env close with it"
    assert env.close_entries == 1, env.close_entries


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
    # The release the bound abandoned still finishes, in its own thread, on its own time, and the
    # env close follows it rather than running beside it. `Env.close` states that order, and a
    # `_close` that tears down what `_end_session` is still using is the same use-after-free by
    # another route. Bounding the caller's latency is not the same as declaring cleanup done.
    assert await _awaited(lambda: env.releases_finished == 1)
    assert len(env.releases) == 1, env.releases
    assert await _awaited(env.closed.is_set), "the env was never closed"
    assert env.closed_during_release is False
    assert env.close_entries == 1, env.close_entries


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


async def test_a_declared_off_loop_factory_does_not_stop_the_rest_of_the_loop(
    tmp_path: Path,
) -> None:
    # `TaskStream.get_task` used to evaluate the env factory in the argument list of its await,
    # which is on the loop. For an env that provisions a corpus and walks two views of it, a cold
    # per-task construction is a second of it, and every other episode's deadline is held for all
    # of them. Moved only for a factory whose caller has said it may be.
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    def slow_factory(_name: str) -> _FixtureScoreEnv:
        time.sleep(0.3)
        return _FixtureScoreEnv(tasks=list(TASKS))

    stream = TaskStream(
        slow_factory,
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        off_loop_factory=True,
    )
    beat = asyncio.ensure_future(ticker())
    try:
        async with stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    finally:
        beat.cancel()
    assert ticks > 20, ticks


async def test_start_builds_its_env_off_the_loop_when_told_it_may(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The same for the single-episode entry point, whose `make()` was on the loop for the same
    # reason and with the same effect, and behind the same declaration.
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
        ep = await ServedEpisode.start(ENV_NAME, task=0, off_loop_factory=True)
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

    lifecycle = episode_module._Lifecycle("probe")
    building = asyncio.ensure_future(episode_module._built(slow_factory, lifecycle))
    await asyncio.sleep(0.02)
    building.cancel()
    with pytest.raises(asyncio.CancelledError):
        await building
    # Closed on the loop it was built on, which is the lifecycle's, so there is no loop to find
    # and none to give up on. This caller's loop is not involved at all.
    assert _until(closed.is_set), "the env nobody took was never closed"
    assert _until(lambda: not lifecycle.running)


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
    real = FinalizationStore._load_all

    def counting(self: FinalizationStore) -> Any:
        reads["n"] += 1
        return real(self)

    # The primitive that walks the directory, which is what costs, rather than the public reader
    # that calls it.
    monkeypatch.setattr(FinalizationStore, "_load_all", counting)
    for _ in range(3):
        ep = await ServedEpisode.open_env(
            _FixtureScoreEnv(tasks=list(TASKS)),
            task=0,
            trace_path=tmp_path / "run.jsonl",
        )
        await ep.close()
    assert reads["n"] == 1, reads


async def test_a_setup_error_after_begin_releases_off_the_loop() -> None:
    # `describe()` raising, or the constructor's own score/finalize guard, is a supported failure
    # and it happens after the session exists. The release used to go through `env.close()`, and
    # `Env.close` runs the hook inline, so a slow release on this path froze the loop and every
    # deadline on it: measured at one 5 ms tick while a 300 ms hook ran on the main thread.
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    env = _SlowSessionEnv(end_seconds=0.3, describe_error=RuntimeError("no contract"))
    beat = asyncio.ensure_future(ticker())
    try:
        with pytest.raises(RuntimeError, match="no contract"):
            await ServedEpisode.open_env(env, task=0)
    finally:
        beat.cancel()
    assert len(env.releases) == 1, env.releases
    assert env.releases[0]["thread"] != "MainThread", env.releases
    assert ticks > 20, ticks
    assert _until(env.closed.is_set), "the env was released but never closed"
    assert env.closed_during_release is False


# ----- the factory runs where its caller said it may -----


class _LoopBoundEnv(_FixtureScoreEnv):
    """An env that binds the running loop in its constructor. The stream contract permits this
    (`test_a_catalog_env_is_never_closed_on_a_foreign_loop` codifies it), so a serve layer that
    quietly moves the factory to a worker thread breaks a supported env."""

    def __init__(self, **kwargs: Any) -> None:
        self.loop = asyncio.get_running_loop()
        super().__init__(**kwargs)


async def test_a_loop_affine_factory_still_gets_its_loop_on_every_dispense(
    tmp_path: Path,
) -> None:
    # Not only at construction. The catalog instance is built on the caller's loop either way;
    # this is about the *second* invocation, the per-task one, which is the one that moved.
    stream = TaskStream(
        lambda _name: _LoopBoundEnv(tasks=list(TASKS)),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})


async def test_an_off_loop_factory_is_called_in_a_thread_with_the_callers_context(
    tmp_path: Path,
) -> None:
    # The opt-in, and what it means: no running loop where the factory runs, a thread that is not
    # the caller's, and the caller's context variables still readable, because a factory that
    # reads one is reading a value its caller set rather than a loop it is bound to.
    marker: contextvars.ContextVar[str] = contextvars.ContextVar("marker", default="unset")
    marker.set("the caller's")
    seen: Dict[str, Any] = {}

    def factory(_name: str) -> _FixtureScoreEnv:
        try:
            seen["loop"] = asyncio.get_running_loop()
        except RuntimeError:
            seen["loop"] = None
        seen["thread"] = threading.current_thread().name
        seen["marker"] = marker.get()
        return _FixtureScoreEnv(tasks=list(TASKS))

    stream = TaskStream(
        factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov", off_loop_factory=True
    )
    # The catalog call was the caller's, on the caller's thread. That is not an oversight: this
    # constructor is synchronous, so no thread it delegates to would give the caller its loop back
    # (see below).
    assert seen["thread"] == threading.main_thread().name
    caller = asyncio.get_running_loop()
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    # On a loop, and on the right one: the episode's own. An env that binds
    # `asyncio.get_running_loop()` in its constructor binds the loop that will also close it, and
    # that loop outlives every caller, so `off_loop_factory` moves construction off the *caller's*
    # loop rather than off every loop.
    assert seen["loop"] is not None
    assert seen["loop"] is not caller
    assert seen["thread"] != threading.main_thread().name
    assert seen["marker"] == "the caller's"


async def test_the_first_construction_is_the_callers_and_a_serving_caller_builds_off_the_loop(
    tmp_path: Path,
) -> None:
    # The cold call for an env whose data is fetched lazily is this one, not the per-task one, and
    # no flag on this constructor can move it: it is synchronous, so a caller that makes it from
    # inside a running loop blocks that loop for its duration whichever thread does the work. The
    # fix is the caller's, and it is one line, so it is tested as one.
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    def slow_factory(_name: str) -> _FixtureScoreEnv:
        time.sleep(0.3)
        return _FixtureScoreEnv(tasks=list(TASKS))

    beat = asyncio.ensure_future(ticker())
    try:
        on_the_loop = TaskStream(
            slow_factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "on-loop"
        )
        held = ticks
        off_the_loop = await asyncio.to_thread(
            TaskStream,
            slow_factory,
            [TaskRef(ENV_NAME, 0)],
            prov_dir=tmp_path / "off-loop",
            off_loop_factory=True,
        )
        moved = ticks - held
    finally:
        beat.cancel()
    await on_the_loop.aclose()
    await off_the_loop.aclose()
    # Built on the loop: the loop did not advance. Built off it: it did, through the same 300 ms.
    assert held <= 2, held
    assert moved > 20, moved


def test_a_scan_that_did_not_finish_is_not_remembered_as_one(tmp_path: Path) -> None:
    # Remembering the directory before the pass ran turned one transient write failure into a
    # record that stays PENDING for the life of the process: the first call raised, and every
    # call after it found the path already answered and read nothing. Only a pass that resolved
    # what it found counts.
    store = FinalizationStore(tmp_path / "finalizations")
    store.write(
        FinalizationRecord(
            session_id="prior", finalization_id="f-crash", status="PENDING",
            source="explicit_tool",
        )
    )
    real = FinalizationStore.write
    failures = {"left": 1}

    def flaky(self: FinalizationStore, record: FinalizationRecord) -> None:
        if failures["left"]:
            failures["left"] -= 1
            raise OSError("the store is read-only for one call")
        return real(self, record)

    FinalizationStore.write = flaky  # type: ignore[method-assign]
    try:
        with pytest.raises(OSError):
            store.recover_once()
    finally:
        FinalizationStore.write = real  # type: ignore[method-assign]
    assert store.read("f-crash").status == "PENDING"
    # Writability is back, and the next caller is the one that has to notice.
    assert [r.finalization_id for r in store.recover_once()] == ["f-crash"]
    assert store.read("f-crash").status == "FAILED"


def test_an_unreadable_entry_is_quarantined_rather_than_re_reading_the_store(
    tmp_path: Path,
) -> None:
    # A file that cannot be read is skipped so the rest of the directory still loads, and that is
    # right for a reader and not enough for a scan: the record it could not read may be exactly
    # the one recovery exists for. Naming it is what settles both. The directory is dealt with,
    # so the known-good records beside it are never read again; the entry that was not a record
    # is, and only it.
    directory = tmp_path / "finalizations"
    store = FinalizationStore(directory)
    store.write(
        FinalizationRecord(
            session_id="prior", finalization_id="f-crash", status="PENDING",
            source="explicit_tool",
        )
    )
    (directory / "finalization-f-torn.json").write_text("{not json", encoding="utf-8")
    with pytest.warns(RuntimeWarning, match="f-torn"):
        assert [r.finalization_id for r in store.recover_once()] == ["f-crash"]
    with _reads_counted(store) as reads:
        assert store.recover_once() == []
    assert reads["full"] == 0, "a corrupt file sent the next pass back over the whole store"


@contextlib.contextmanager
def _reads_counted(store: FinalizationStore) -> Any:
    """Count full walks of a store's directory while the body runs."""
    counted = {"full": 0}
    real = FinalizationStore._load_all

    def counting(self: FinalizationStore) -> Any:
        counted["full"] += 1
        return real(self)

    FinalizationStore._load_all = counting  # type: ignore[method-assign]
    try:
        yield counted
    finally:
        FinalizationStore._load_all = real  # type: ignore[method-assign]


def test_a_forked_child_asks_the_recovery_question_for_itself(tmp_path: Path) -> None:
    # "Once per process" means this process. The parent left a record alone because its owner was
    # alive when the parent looked; a child that inherits that answer never looks again, and the
    # record outlives the owner it was waiting on. Sync, and forking before any loop exists, for
    # the reason the stream's fork test gives.
    directory = tmp_path / "finalizations"
    store = FinalizationStore(directory)
    store.write(
        FinalizationRecord(
            session_id="prior", finalization_id="f-crash", status="PENDING",
            source="explicit_tool", owner_pid=os.getpid(),
        )
    )
    assert store.recover_once() == []  # the owner is alive: nothing to resolve, and remembered

    held = store.read("f-crash")
    held.owner_pid = _a_dead_pid()
    store.write(held)

    verdict = tmp_path / "child-said"
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns to the test runner
        try:
            resolved = FinalizationStore(directory).recover_once()
            verdict.write_text(
                ",".join(r.finalization_id for r in resolved), encoding="utf-8"
            )
        finally:
            os._exit(0)
    os.waitpid(pid, 0)
    assert verdict.read_text(encoding="utf-8") == "f-crash"
    assert store.read("f-crash").status == "FAILED"


def _a_dead_pid() -> int:
    """A pid nothing is running under: spawn a child, reap it, and use its id. Made rather than
    guessed, because a number picked out of the air is a number something may be using."""
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns to the test runner
        os._exit(0)
    os.waitpid(pid, 0)
    return pid


async def test_a_declared_off_loop_env_is_built_and_closed_on_the_same_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `TaskStream` promises its envs are closed on the loop that built them. An env whose factory
    # the caller has declared safe off their loop is built on the episode's own lifecycle loop,
    # which is also where it is closed, so the promise holds by construction: there is no second
    # loop to find and nothing to decide when the release runs late. The bound is short and the
    # hook is long here, so the close is arranged behind a release that has outrun it, which is
    # the case that used to reach for a throwaway loop.
    monkeypatch.setattr(episode_module, "_END_SESSION_SECONDS", 0.02)
    built: List[_SlowSessionEnv] = []

    def factory(_name: str) -> _SlowSessionEnv:
        env = _SlowSessionEnv(end_seconds=0.3, bind_loop=True)
        built.append(env)
        return env

    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        off_loop_factory=True,
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    served = built[1]  # built[0] is the catalog instance, built on the caller's thread
    assert served.closed.is_set(), "the env was never closed"
    assert served.closed_on_a_foreign_loop is False
    assert served.close_entries == 1, served.close_entries


async def test_a_second_close_joins_the_deferred_one_rather_than_starting_another(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first close times out and arranges the close behind the release. The second arrives
    # after the release has finished, so a check of "has the release landed" sends it straight
    # into `_close` while the arranged one is already inside it. One owner, and everyone else
    # joins.
    monkeypatch.setattr(episode_module, "_END_SESSION_SECONDS", 0.02)
    env = _SlowSessionEnv(end_seconds=0.3)
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    await ep.close()
    await asyncio.sleep(0.35)  # the release lands; the arranged close starts
    await ep.close()
    await asyncio.sleep(0.2)
    assert env.close_entries == 1, env.close_entries
    assert env.peak_closes == 1, env.peak_closes
    assert env.closed_during_release is False


async def test_the_session_hooks_carry_the_callers_context() -> None:
    # `asyncio.to_thread` copies the caller's context and a raw `submit` does not, so moving the
    # hooks onto a dedicated thread silently took request-scoped state away from them: a hook
    # that reads a tenant, an auth subject or a trace id read the default instead, and began or
    # released the wrong one.
    _TENANT.set("tenant-a")
    env = _SlowSessionEnv()
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    await ep.close()
    assert env.begin_context == "tenant-a"
    assert env.end_context == "tenant-a"


@pytest.mark.parametrize(
    "blob",
    [
        pytest.param("[]", id="valid json that is not an object"),
        pytest.param("{}", id="an object with none of a record's fields"),
        pytest.param(
            json.dumps(
                {
                    "session_id": "s", "finalization_id": "f-bad", "status": "PENDING",
                    "source": "explicit_tool", "verdict": [1, 2],
                }
            ),
            id="a record shaped field that is the wrong shape",
        ),
    ],
)
def test_a_file_that_is_not_a_record_does_not_stop_an_episode_opening(
    tmp_path: Path, blob: str
) -> None:
    # The store is shared with every session the machine has run and holds files this process did
    # not write. "Not a record" is not only invalid JSON: valid JSON that is not an object raises
    # on the mapping, an object missing the fields raises from the constructor, and a field of
    # the wrong shape raises where it is first used, three frames away in recovery. All of it
    # used to reach `ServedEpisode.__init__`, which caught `OSError` alone, so one such file in
    # the machine-global store stopped every later score episode from opening.
    directory = tmp_path / "finalizations"
    directory.mkdir(parents=True)
    (directory / "finalization-bad.json").write_text(blob, encoding="utf-8")
    store = FinalizationStore(directory)
    assert store.load_all() == []
    # Named rather than silently skipped: if one of these was a record left mid-finalize, an
    # episode has not been resolved fail-closed, and nobody downstream can tell from an empty
    # list that it happened.
    with pytest.warns(RuntimeWarning, match="are not finalization records"):
        assert store.recover_once() == []


async def test_an_episode_opens_against_a_store_it_could_not_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The containment, end to end: recovery may not decide whether an episode opens.
    def unreadable(self: FinalizationStore) -> Any:
        raise RuntimeError("this store cannot be read at all")

    monkeypatch.setattr(FinalizationStore, "_load_all", unreadable)
    ep = await ServedEpisode.open_env(
        _FixtureScoreEnv(tasks=list(TASKS)), task=0, trace_path=tmp_path / "run.jsonl"
    )
    try:
        result = await ep.call(SUBMIT_TOOL, {"answer": "4"})
        assert result.terminated is True
    finally:
        await ep.close()


# ----- a slot is not free until the env in it is closed -----


async def test_a_slot_is_not_dispensed_again_until_its_env_is_closed(tmp_path: Path) -> None:
    # `_Live.released` says an env is closed when the release task is done, and `TaskStream`
    # refuses to dispense over an episode still closing. `close()` is bounded on purpose, so a
    # release still inside a wedged hook leaves the env close arranged behind it and returns:
    # right for latency, and the wrong answer to the ownership question. Answered on it, a
    # capacity of one handed out the next task while the first env still held its worker, its
    # port and its directory.
    monkeypatch_bound = 0.02
    built: List[_SlowSessionEnv] = []

    def factory(_name: str) -> _SlowSessionEnv:
        env = _SlowSessionEnv(end_seconds=0.4, tasks=list(TASKS + [dict(TASKS[0], id="q1")]))
        built.append(env)
        return env

    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        max_in_flight=1,
    )
    original = episode_module._END_SESSION_SECONDS
    episode_module._END_SESSION_SECONDS = monkeypatch_bound
    try:
        async with stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
            first = built[1]  # built[0] is the catalog instance, which never opens a session
            await stream.get_task()
            assert first.closed.is_set(), "the next task was dispensed over a closing env"
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    finally:
        episode_module._END_SESSION_SECONDS = original


# ----- a caller that awaits a close is told what it did -----


async def test_close_raises_what_the_envs_close_raised() -> None:
    # Before the close had an owner, `ServedEpisode.close()` awaited `env.close()` and let its
    # failure through. Recording it in a private field instead made every caller's close silently
    # best-effort, including the ones that are not a stream with a containment boundary of its
    # own.
    env = _SlowSessionEnv(close_error=RuntimeError("close boom"))
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    with pytest.raises(RuntimeError, match="close boom"):
        await ep.close()
    # And the episode is still finished with: the hook thread is not leaked over the failure.
    assert _until(lambda: not _session_threads_alive())


async def test_a_cancelled_close_finishes_the_close_and_hands_the_cancellation_back() -> None:
    # A cancellation arriving mid-close is the caller going away, not an instruction to leave an
    # env half torn down.
    env = _SlowSessionEnv(close_seconds=0.2)
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    closing = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert await _awaited(env.closed.is_set), "the close was abandoned half way"
    assert env.close_entries == 1


def test_a_lifecycle_owned_close_does_not_need_the_callers_loop_at_all() -> None:
    # What replaces four rounds of "will that loop ever run this?". The env is built on the
    # episode's own loop and closed there, so a caller's loop closing, pausing, or never turning
    # again is not a question anyone has to answer: no work is ever posted to it. Driven by hand,
    # with the caller's loop closed before the close is asked for.
    env: Dict[str, Any] = {}
    lifecycle = episode_module._Lifecycle("probe")
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _loop, _context: None)

    async def build() -> None:
        env["it"] = await episode_module._built(
            lambda: _SlowSessionEnv(), lifecycle
        )

    try:
        loop.run_until_complete(build())
    finally:
        loop.close()
    built = env["it"]
    assert built.owner is None  # `bind_loop` is off; what matters is the close below

    cleanup = episode_module._EnvClose(built, lifecycle, None)
    outcome = cleanup.close_env()
    outcome.result(timeout=10.0)
    assert built.closed.is_set()
    assert built.close_entries == 1
    lifecycle.stop(5.0)


# ----- a CancelledError an env raised is the env's, not the caller's -----


async def test_an_env_that_raises_cancelled_from_its_release_still_gets_closed() -> None:
    # `CancelledError` is two unrelated things wearing one type. Caught as `Exception` it is
    # neither: the env's own cancellation went straight past the env close on the rollback path,
    # and out through the terminating call on the ordinary one, answering the agent with a
    # traceback instead of the constant.
    env = _SlowSessionEnv(end_error=asyncio.CancelledError())
    ep = await ServedEpisode.open_env(env, task=0)
    result = await ep.call(SUBMIT_TOOL, {"answer": "4"})
    assert result.terminated is True
    await ep.close()
    assert await _awaited(env.closed.is_set), "the env's cancellation skipped its close"


async def test_a_rollback_whose_release_raises_cancelled_still_closes_the_env() -> None:
    env = _SlowSessionEnv(
        end_error=asyncio.CancelledError(), describe_error=RuntimeError("no contract")
    )
    with pytest.raises(RuntimeError, match="no contract"):
        await ServedEpisode.open_env(env, task=0)
    assert await _awaited(env.closed.is_set), "the env's cancellation skipped its close"


def test_a_loop_loss_rollback_does_not_leak_its_hook_thread() -> None:
    # The executor is this episode's alone and used to be shut down by the caller's continuation.
    # A continuation on a loop that closed never runs, so every abandoned setup left a live
    # non-daemon thread behind. Shutdown belongs to the rollback, which the thread itself runs.
    env = _SlowSessionEnv(begin_seconds=0.4)
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _loop, _context: None)

    async def give_up() -> None:
        opening = loop.create_task(ServedEpisode.open_env(env, task=0))
        await asyncio.sleep(0.05)
        opening.cancel()
        for _ in range(5):
            await asyncio.sleep(0)

    try:
        loop.run_until_complete(give_up())
    finally:
        loop.close()
    assert _until(env.closed.is_set)
    assert _until(lambda: not _session_threads_alive()), _session_threads_alive()


def _session_threads_alive() -> List[str]:
    return [t.name for t in threading.enumerate() if t.name.startswith("shogym-episode")]


# ----- one bad file does not restore the scan it took a review pass to remove -----


@pytest.mark.parametrize(
    "owner_pid",
    [pytest.param("not-a-pid", id="a string"), pytest.param(True, id="a boolean")],
)
def test_a_record_whose_owner_pid_is_not_a_pid_is_not_a_record(
    tmp_path: Path, owner_pid: Any
) -> None:
    # `owner_pid` is not read back, it is *operated on*: it reaches `os.kill`, three frames from
    # the decode, in a caller that can only suppress what it raises. And `True` is an `int` to
    # Python, so a boolean is pid 1, which is alive on every machine: the record is left untouched
    # forever and the pass records itself as having dealt with it.
    directory = tmp_path / "finalizations"
    directory.mkdir(parents=True)
    (directory / "finalization-bad.json").write_text(
        json.dumps(
            {
                "session_id": "s", "finalization_id": "f-bad", "status": "PENDING",
                "source": "explicit_tool", "owner_pid": owner_pid,
            }
        ),
        encoding="utf-8",
    )
    store = FinalizationStore(directory)
    assert store.load_all() == []
    with pytest.warns(RuntimeWarning, match="are not finalization records"):
        assert store.recover_once() == []


def test_one_unreadable_entry_does_not_make_every_episode_read_the_whole_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The performance half, which is the whole point of the mechanism: against a store holding
    # every record the machine has written, one malformed or unreadable file used to restore the
    # exact O(episodes x history) scan this replaced. The directory is remembered with its bad
    # entries named, and a later pass reads those and nothing else.
    directory = tmp_path / "finalizations"
    directory.mkdir(parents=True)
    (directory / "finalization-bad.json").write_text("[]", encoding="utf-8")
    store = FinalizationStore(directory)
    scans = {"n": 0}
    real = FinalizationStore._load_all

    def counting(self: FinalizationStore) -> Any:
        scans["n"] += 1
        return real(self)

    monkeypatch.setattr(FinalizationStore, "_load_all", counting)
    with pytest.warns(RuntimeWarning):
        store.recover_once()
    for _ in range(2):
        store.recover_once()
    assert scans["n"] == 1, scans


def test_a_quarantined_entry_is_re_read_and_resolved_once_it_becomes_a_record(
    tmp_path: Path,
) -> None:
    # The correctness half. A file half-written when its writer died is not a record now and may
    # be one later, so the entries that could not be read are the entries a later pass comes back
    # to. Nothing else does.
    directory = tmp_path / "finalizations"
    directory.mkdir(parents=True)
    torn = directory / "finalization-f-crash.json"
    torn.write_text('{"session_id": "s", "finaliz', encoding="utf-8")
    store = FinalizationStore(directory)
    with pytest.warns(RuntimeWarning, match="are not finalization records"):
        assert store.recover_once() == []
    torn.write_text(
        json.dumps(
            {
                "session_id": "s", "finalization_id": "f-crash", "status": "PENDING",
                "source": "explicit_tool",
            }
        ),
        encoding="utf-8",
    )
    assert [r.finalization_id for r in store.recover_once()] == ["f-crash"]
    assert store.read("f-crash").status == "FAILED"
    # And it stops being quarantined once it is readable.
    assert store.recover_once() == []


# ----- the slot waits for the close itself, and nothing shorter -----


async def test_a_slot_is_not_freed_by_a_join_that_gave_up(tmp_path: Path) -> None:
    # The join the slot owner makes used to be bounded, and a bound that expires reads exactly
    # like a close that finished: the release task completed, `_Live.released` said the env was
    # gone, and the next task was dispensed over an env that still held its worker and its port.
    # With both lifecycle bounds at 10 ms and a 200 ms release there is no bound left to expire.
    built: List[_SlowSessionEnv] = []

    def factory(_name: str) -> _SlowSessionEnv:
        env = _SlowSessionEnv(end_seconds=0.2)
        built.append(env)
        return env

    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0), TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        max_in_flight=1,
    )
    original = (episode_module._END_SESSION_SECONDS, episode_module._ROLLBACK_SECONDS)
    episode_module._END_SESSION_SECONDS = 0.01
    episode_module._ROLLBACK_SECONDS = 0.01
    try:
        async with stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
            first = built[1]
            await stream.get_task()
            assert first.closed.is_set(), "a join that gave up freed the slot"
            assert first.releases_finished == 1
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    finally:
        (
            episode_module._END_SESSION_SECONDS,
            episode_module._ROLLBACK_SECONDS,
        ) = original


# ----- a live owner is never taken over, however long it stays quiet -----


def test_a_paused_but_open_owner_loop_is_never_taken_over() -> None:
    # No elapsed time proves an open loop is dead. A loop between two `run_until_complete` calls
    # looks exactly like a loop nobody will ever turn again, so a takeover on a timer reclaims a
    # live owner and closes a loop-affine env somewhere it does not belong. Only the owner itself
    # can say, and `is_closed` is the one thing that says it.
    lifecycle = episode_module._Lifecycle("probe")
    owner = asyncio.new_event_loop()
    try:
        env = owner.run_until_complete(_made_on(owner))
        cleanup = episode_module._EnvClose(env, lifecycle, owner)
        cleanup.close_env()
        # The owner is open and idle, which is what a pause looks like. Nothing may happen.
        time.sleep(0.6)
        assert not env.closed.is_set(), "an idle owner was treated as a dead one"
        assert env.closed_on_a_foreign_loop is False
        # It comes back, and closes its own env on its own loop, exactly as it always could.
        owner.run_until_complete(cleanup.here())
        assert env.closed.is_set()
        assert env.close_entries == 1
        assert env.closed_on_a_foreign_loop is False
    finally:
        owner.close()
        lifecycle.stop(5.0)


async def _made_on(_owner: "asyncio.AbstractEventLoop") -> "_SlowSessionEnv":
    return _SlowSessionEnv(bind_loop=True)


async def test_a_close_asked_for_from_a_foreign_loop_hands_off_and_waits() -> None:
    # Asking a foreign running loop to run `asyncio.run` is a nested-run `RuntimeError`, and
    # asking it to block while something is polled is a serving loop held for a minute. It does
    # neither: the close belongs to the loop that built the env, and a caller elsewhere waits on
    # the outcome.
    lifecycle = episode_module._Lifecycle("probe")
    env = await _built_on(lifecycle)
    cleanup = episode_module._EnvClose(env, lifecycle, None)
    started = time.perf_counter()
    cleanup.close_env()
    await cleanup.joined()
    assert time.perf_counter() - started < 1.0
    assert env.closed.is_set()
    assert cleanup.failure is None
    lifecycle.stop(5.0)


async def _built_on(lifecycle: Any) -> "_SlowSessionEnv":
    return await episode_module._built(lambda: _SlowSessionEnv(bind_loop=True), lifecycle)


# ----- the rollback closes the sessions before the env -----


async def test_a_rollback_closes_the_mcp_sessions_before_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `Env.close` releases what the *constructor* made, and the sessions opened for this episode
    # are clients that may still be using it. The ordinary close path has always closed them
    # first; the rollback used to arrange the env close the moment the release landed, beside a
    # caller that was only then awaiting the sessions.
    # The release lands at once and the sessions take a while, which is the order that exposes
    # it: a rollback that arranges the env close the moment the release is out is arranging it
    # while the caller is still closing the clients that use what the env is about to release.
    env = _SlowSessionEnv(begin_seconds=0.1)
    real_open = episode_module._open_session_for_spec

    async def opening_session(spec: Any, **kwargs: Any) -> Any:
        session = await real_open(spec, **kwargs)
        closing = session.close

        async def close() -> None:
            # A close that yields for a while, so an env close scheduled beside it runs first
            # rather than merely happening to run second.
            await asyncio.sleep(0.3)
            await closing()
            env.sessions_closed.set()

        session.close = close  # type: ignore[method-assign]
        return session

    monkeypatch.setattr(episode_module, "_open_session_for_spec", opening_session)
    opening = asyncio.ensure_future(ServedEpisode.open_env(env, task=0))
    await asyncio.sleep(0.02)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert env.closed.is_set()
    assert env.closed_after_sessions is True


# ----- one builder per queue position -----


async def test_a_cancelled_pull_hands_its_builder_to_the_next_one(tmp_path: Path) -> None:
    # A thread cannot be told to stop, so a pull that gives up mid-build leaves a constructor
    # running. If the next pull for the same still-owed position starts its own, a client that
    # cancels repeatedly runs as many constructors at once as it likes, none of them inside
    # `max_in_flight`, and each wedged one leaves a thread behind.
    started = threading.Semaphore(0)
    release = threading.Event()
    builds = {"n": 0}

    def factory(_name: str) -> _SlowSessionEnv:
        builds["n"] += 1
        started.release()
        release.wait(10.0)
        return _SlowSessionEnv()

    stream = TaskStream(
        factory,
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        off_loop_factory=True,
    )
    catalog_builds = builds["n"]
    try:
        for _ in range(3):
            pull = asyncio.ensure_future(stream.get_task())
            assert started.acquire(timeout=5.0) or True
            await asyncio.sleep(0.05)
            pull.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pull
        assert builds["n"] - catalog_builds == 1, builds
    finally:
        release.set()
        await stream.aclose()


# ----- a cleanup failure is a note on the setup failure, not a replacement -----


async def test_a_close_that_fails_during_setup_does_not_replace_the_setup_failure() -> None:
    # The caller asked why setup failed. "The env would not close" is not that answer: it is
    # something that happened while answering, and a caller handed it instead has been given the
    # wrong problem to debug.
    env = _SlowSessionEnv(close_error=RuntimeError("close boom"))

    def refuse(_idx: Any) -> Any:
        raise ValueError("setup boom")

    env.load_task = refuse  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="setup boom") as raised:
        await ServedEpisode.open_env(env, task=0)
    assert any("close boom" in note for note in getattr(raised.value, "__notes__", []))


# ----- every joiner is told what the close did -----


async def test_every_close_caller_is_told_the_same_failure() -> None:
    # A single-owner operation needs a retained outcome and not only a completion event: the
    # loser of the claim used to wait for the event and return as though the close had gone well.
    env = _SlowSessionEnv(close_error=RuntimeError("close boom"), close_seconds=0.15)
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    outcomes = await asyncio.gather(ep.close(), ep.close(), return_exceptions=True)
    assert [type(outcome).__name__ for outcome in outcomes] == ["RuntimeError", "RuntimeError"]
    assert all("close boom" in str(outcome) for outcome in outcomes)
    # And a caller arriving after the fact is told too, rather than finding a quiet success.
    with pytest.raises(RuntimeError, match="close boom"):
        await ep.close()


# ----- the abandoned build closes under the context it was built in -----


async def test_an_abandoned_build_is_closed_under_the_callers_context() -> None:
    # `_built` copies the caller's context into the factory; the discard callback fires later, in
    # whichever thread finished the build, after that context has been left behind. An env whose
    # constructor took a tenant-scoped resource releases one, and the abandoned path is not where
    # that should quietly become the wrong tenant.
    _TENANT.set("tenant-a")
    seen: Dict[str, Any] = {}
    closed = threading.Event()

    class _Noticing(_FixtureScoreEnv):
        async def _close(self) -> None:
            seen["tenant"] = _TENANT.get()
            closed.set()

    def slow_factory() -> _Noticing:
        time.sleep(0.3)
        return _Noticing(tasks=list(TASKS))

    lifecycle = episode_module._Lifecycle("probe")
    building = asyncio.ensure_future(episode_module._built(slow_factory, lifecycle))
    await asyncio.sleep(0.02)
    building.cancel()
    with pytest.raises(asyncio.CancelledError):
        await building
    assert _until(closed.is_set), "the env nobody took was never closed"
    assert seen["tenant"] == "tenant-a"


# ----- the decoder refuses values that are not what they claim to be -----


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("owner_pid", 0, id="pid zero is every process in a group"),
        pytest.param("owner_pid", -1, id="a negative pid is a process group"),
        pytest.param("owner_pid", 2**63, id="a pid past what the platform can hold"),
        pytest.param("schema_version", 2, id="a schema this reader does not have"),
    ],
)
def test_a_value_that_is_not_what_it_claims_is_not_a_record(
    tmp_path: Path, field: str, value: Any
) -> None:
    directory = tmp_path / "finalizations"
    directory.mkdir(parents=True)
    record = {
        "session_id": "s", "finalization_id": "f-bad", "status": "PENDING",
        "source": "explicit_tool",
    }
    record[field] = value
    (directory / "finalization-bad.json").write_text(json.dumps(record), encoding="utf-8")
    store = FinalizationStore(directory)
    assert store.load_all() == []
    with pytest.warns(RuntimeWarning, match="are not finalization records"):
        assert store.recover_once() == []


def test_a_record_carrying_nan_is_quarantined_rather_than_rewritten_forever(
    tmp_path: Path,
) -> None:
    # `json.loads` reads `NaN` and the writer refuses it, so a record carrying one can be read
    # and never written back: recovery rewrote it, the write raised, and the same failure came
    # round again on every pass, with the directory never cached and every episode rescanning.
    directory = tmp_path / "finalizations"
    directory.mkdir(parents=True)
    (directory / "finalization-nan.json").write_text(
        '{"session_id": "s", "finalization_id": "f-nan", "status": "PENDING", '
        '"source": "explicit_tool", "verdict": {"confidence": NaN}}',
        encoding="utf-8",
    )
    store = FinalizationStore(directory)
    with pytest.warns(RuntimeWarning, match="are not finalization records"):
        assert store.recover_once() == []
    with _reads_counted(store) as reads:
        assert store.recover_once() == []
    assert reads["full"] == 0


# ----- the lifecycle owns every decision; callers only observe -----


async def test_closing_a_stream_does_not_discard_a_build_a_pull_is_waiting_for(
    tmp_path: Path,
) -> None:
    # A builder is in the map both while a live pull awaits it and while it waits for the next
    # pull after a cancellation, and those are not the same thing. Shutdown used to treat both as
    # abandoned, so it closed an env the pull went on to open a session on.
    made = threading.Event()
    release = threading.Event()
    built: List[_SlowSessionEnv] = []

    def factory(_name: str) -> _SlowSessionEnv:
        env = _SlowSessionEnv()
        built.append(env)
        made.set()
        release.wait(10.0)
        return env

    stream = TaskStream(
        factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov", off_loop_factory=True
    )
    made.clear()
    release.clear()
    pull = asyncio.ensure_future(stream.get_task())
    assert await asyncio.to_thread(made.wait, 5.0)
    closing = asyncio.ensure_future(stream.aclose())
    await asyncio.sleep(0.05)
    release.set()
    with contextlib.suppress(BaseException):
        await pull
    with contextlib.suppress(BaseException):
        await closing
    served = built[-1]
    # Whatever the race decided about the task, the env the pull was holding was not closed out
    # from under it: no close began before its session did.
    assert served.begins == [] or served.close_entries <= 1
    assert served.closed_during_begin is False


async def test_a_cancelled_start_closes_the_env_its_factory_finally_returns() -> None:
    # `_built` arranges the discard, and `start()` used to stop the lifecycle beside it: marked
    # stopped, the lifecycle could no longer take that discard when the build landed, so the env
    # was built and then nothing could close it.
    closed = threading.Event()

    class _Noticing(_FixtureScoreEnv):
        async def _close(self) -> None:
            closed.set()

    def slow_make(_name: str, config: Optional[Dict[str, Any]] = None) -> _Noticing:
        time.sleep(0.3)
        return _Noticing(tasks=list(TASKS))

    original = episode_module.make
    episode_module.make = slow_make  # type: ignore[assignment]
    try:
        starting = asyncio.ensure_future(
            ServedEpisode.start(ENV_NAME, task=0, off_loop_factory=True)
        )
        await asyncio.sleep(0.02)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        assert _until(closed.is_set), "the env the factory returned was never closed"
    finally:
        episode_module.make = original  # type: ignore[assignment]


async def test_a_rollback_that_outran_its_bound_is_not_raced_by_the_caller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `settled()` says whether the release actually landed, and the handler ignored it: it
    # claimed the close over a release still inside the hook, which is the exact overlap this
    # design exists to remove.
    monkeypatch.setattr(episode_module, "_ROLLBACK_SECONDS", 0.02)
    env = _SlowSessionEnv(end_seconds=0.3, describe_error=RuntimeError("no contract"))
    with pytest.raises(RuntimeError, match="no contract"):
        await ServedEpisode.open_env(env, task=0)
    assert env.closed_during_release is False
    assert await _awaited(lambda: env.releases_finished == 1)
    assert await _awaited(env.closed.is_set)
    assert env.close_entries == 1


def test_a_loop_loss_rollback_stops_its_own_lifecycle() -> None:
    # The rollback used to end after arranging the close and leave the shutdown to a caller
    # continuation. A continuation on a loop that closed never runs, so the env was closed and
    # its thread stayed alive for the life of the process.
    env = _SlowSessionEnv(begin_seconds=0.3)
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _loop, _context: None)

    async def give_up() -> None:
        opening = loop.create_task(ServedEpisode.open_env(env, task=0))
        await asyncio.sleep(0.05)
        opening.cancel()
        for _ in range(5):
            await asyncio.sleep(0)

    try:
        loop.run_until_complete(give_up())
    finally:
        loop.close()
    assert _until(env.closed.is_set)
    assert _until(lambda: not _session_threads_alive()), _session_threads_alive()


async def test_a_cancelled_close_still_closes_the_env_and_stops_the_lifecycle() -> None:
    # A cancellation while waiting for the release used to leave `close()` before it had arranged
    # either the env close or the shutdown. What the `finally` does is arrangement rather than
    # waiting, so it runs the same way on a cancellation as on a return.
    env = _SlowSessionEnv(end_seconds=0.3)
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    closing = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert await _awaited(env.closed.is_set), "the cancelled close left the env open"
    assert _until(lambda: not ep._lifecycle.running)


async def test_two_closes_join_rather_than_moving_the_env_to_another_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A flag meaning "somebody started a wait" let the second close skip it, find the release
    # incomplete and hand a caller-built env to the lifecycle loop: one caller saw the affinity
    # failure and the other reported success over the same env.
    monkeypatch.setattr(episode_module, "_END_SESSION_SECONDS", 0.05)
    env = _SlowSessionEnv(end_seconds=0.3, bind_loop=True)
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    outcomes = await asyncio.gather(ep.close(), ep.close(), return_exceptions=True)
    assert await _awaited(env.closed.is_set)
    assert env.closed_on_a_foreign_loop is False, outcomes
    assert env.close_entries == 1


async def test_a_flagged_envs_finalize_runs_on_the_loop_it_was_built_on(
    tmp_path: Path,
) -> None:
    # The contract says a flagged env may bind `get_running_loop()` in its constructor because
    # its work and its close use that loop. `finalize` is env work, and running it on the serving
    # caller's loop broke exactly that promise: the affinity error came back as a fail-closed
    # verdict, so a valid episode scored zero.
    seen: Dict[str, Any] = {}

    class _LoopChecking(_FixtureScoreEnv):
        def __init__(self, **kwargs: Any) -> None:
            self.loop = asyncio.get_running_loop()
            super().__init__(**kwargs)

        async def finalize(self, req: Any) -> Any:
            seen["foreign"] = asyncio.get_running_loop() is not self.loop
            return await super().finalize(req)

    stream = TaskStream(
        lambda _name: _LoopChecking(tasks=list(TASKS)),
        [TaskRef(ENV_NAME, 0)],
        prov_dir=tmp_path / "prov",
        off_loop_factory=True,
        feedback=Immediate(),
    )
    async with stream:
        await stream.get_task()
        answer = await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    assert seen["foreign"] is False
    assert "finalize_error" not in json.loads(answer.content[0].text)


async def test_the_teardown_bound_is_spent_once_across_release_and_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The release and the env close behind it are two halves of one operation. A full bound for
    # each turned a stated bound into twice one for the env that needs it least: the one whose
    # release never comes back.
    monkeypatch.setattr(episode_module, "_END_SESSION_SECONDS", 0.1)
    env = _SlowSessionEnv(end_seconds=5.0)
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    started = time.perf_counter()
    await ep.close()
    assert time.perf_counter() - started < 0.15, time.perf_counter() - started


async def test_an_adopted_build_is_closed_under_the_context_it_was_built_in(
    tmp_path: Path,
) -> None:
    # A build made by one pull and adopted by another was closed under the *adopting* caller's
    # context, so an env whose constructor took a tenant-scoped resource released a different
    # tenant's.
    seen: Dict[str, Any] = {}
    release = threading.Event()
    made = threading.Event()

    class _Noticing(_FixtureScoreEnv):
        def __init__(self, **kwargs: Any) -> None:
            seen.setdefault("built", []).append(_TENANT.get())
            super().__init__(**kwargs)

        async def _close(self) -> None:
            seen.setdefault("closed", []).append(_TENANT.get())

    def factory(_name: str) -> _Noticing:
        made.set()
        release.wait(10.0)
        return _Noticing(tasks=list(TASKS))

    _TENANT.set("tenant-a")
    release.set()  # the catalog build runs inside the constructor and may not be held
    stream = TaskStream(
        factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov", off_loop_factory=True
    )
    made.clear()
    release.clear()
    pull = asyncio.ensure_future(stream.get_task())
    assert await asyncio.to_thread(made.wait, 5.0)
    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull
    _TENANT.set("tenant-b")
    release.set()
    async with stream:
        await stream.get_task()
        await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    # Two envs are built here: the catalog instance, inside the constructor, and the served one,
    # which the first pull started and the second adopted. The served one closes first, when its
    # task is sealed, and that is the close this is about. (The catalog env is closed by `aclose`
    # under whatever context that caller has, which is a different question.)
    assert seen["built"] == ["tenant-a", "tenant-a"], seen
    assert seen["closed"][0] == "tenant-a", seen


async def test_a_cancelled_pull_does_not_open_a_span_it_cannot_close(tmp_path: Path) -> None:
    # Spans were opened before the retained build was joined, so a cancelled pull left one open
    # with nothing to finalize it: three cancels and a retry opened four and closed one, and
    # whatever the three abandoned ones had taken was never given back. Opened after the build
    # now, which is the step a cancellation can survive.
    opened = {"n": 0}
    release = threading.Event()
    made = threading.Event()
    real_spans = TaskStream._begin_spans

    async def counting(self: TaskStream, ref: TaskRef) -> Any:
        opened["n"] += 1
        return await real_spans(self, ref)

    def factory(_name: str) -> _FixtureScoreEnv:
        made.set()
        release.wait(10.0)
        return _FixtureScoreEnv(tasks=list(TASKS))

    release.set()  # the catalog build runs inside the constructor and may not be held
    stream = TaskStream(
        factory, [TaskRef(ENV_NAME, 0)], prov_dir=tmp_path / "prov", off_loop_factory=True
    )
    TaskStream._begin_spans = counting  # type: ignore[method-assign]
    try:
        made.clear()
        release.clear()
        for _ in range(3):
            pull = asyncio.ensure_future(stream.get_task())
            assert await asyncio.to_thread(made.wait, 5.0)
            pull.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pull
        release.set()
        async with stream:
            await stream.get_task()
            await stream.dispatch(SUBMIT_TOOL, {"answer": "4"})
    finally:
        TaskStream._begin_spans = real_spans  # type: ignore[method-assign]
    assert opened["n"] == 1, opened


def test_a_number_json_cannot_write_back_is_refused_however_deep_it_is(
    tmp_path: Path,
) -> None:
    # The writer serialises with `allow_nan=False`, which walks the whole structure, so a check
    # that stopped at the first level passed records the writer would never accept: the rewrite
    # raised on every pass and the store was never cached.
    directory = tmp_path / "finalizations"
    directory.mkdir(parents=True)
    (directory / "finalization-nested.json").write_text(
        '{"session_id": "s", "finalization_id": "f", "status": "PENDING", '
        '"source": "explicit_tool", "provenance": {"nested": {"value": NaN}}}',
        encoding="utf-8",
    )
    store = FinalizationStore(directory)
    assert store.load_all() == []
    with pytest.warns(RuntimeWarning, match="are not finalization records"):
        assert store.recover_once() == []
    with _reads_counted(store) as reads:
        assert store.recover_once() == []
    assert reads["full"] == 0
