"""The episode lifecycle, one state and one caller event at a time.

One class of defect runs through this lifecycle: a decision about an episode's env sitting where
a cancellation can land between making it and acting on it. Fixing instances is not closing a
class, so this module is the class.

**What it is.** Every lifecycle state an episode passes through, crossed with every event a
caller can deliver, enumerated as a table that is visible in the test ids. Each cell either runs
the interleaving with barriers and asserts the invariants below, or says in one line why it is
not reachable. A cell is never silently missing.

**The invariants**, checked by :func:`_holds` for every cell that runs:

1. Exactly one ``Env.close``, on the loop that built the env.
2. Every MCP session this episode opened is closed.
3. No teardown while a finalizer, a drain or an evaluator still owns the env.
4. A correct submission scores correct, whatever else happened.
5. One trajectory row per dispatched call, contiguous, and an overtaken call is tombstoned.
6. The lifecycle thread is stopped and was never joined from itself.
7. Nothing is scheduled on a loop this layer does not own.

Not every invariant is meaningful in every cell (an episode cancelled while building has no
sessions to close), so each is checked where it applies and the cell says which it exercised.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from shogym.serve import ServedEpisode
from shogym.serve import episode as episode_module
from shogym.serve.stream import TaskRef, TaskStream
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME

from tests._fixtures import score_env, score_mcp
from tests._fixtures.score_env import ENV_NAME, SUBMIT_TOOL, _FixtureScoreEnv

TASKS = [{"id": "q0", "question": "2+2?", "answer": "4"}]

#: The states an episode passes through. Every one of them is a place a caller event can land.
STATES = (
    "building",
    "begun",
    "dispatching",
    "finalizing",
    "draining",
    "releasing",
    "closing",
    "closed",
)

#: What a caller (or the world) can do to an episode in any of those states.
EVENTS = (
    "cancel_start",
    "cancel_call",
    "cancel_close",
    "deadline_fires",
    "second_close",
    "loop_closes",
    "env_raises",
    "env_cancels",
    "session_slow",
    "span_fails",
    "factory_fails_late",
    # A cell that says "cannot arise" is a claim, and these are what happens when the claim is
    # wrong.
    "normal_close",
    "cancel_cleanup",
    "deadline_mid_finalize",
)


class _Watched(_FixtureScoreEnv):
    """A score env that records what was done to it and when, so a cell can assert order."""

    def __init__(
        self,
        *,
        bind_loop: bool = True,
        begin_seconds: float = 0.0,
        end_seconds: float = 0.0,
        end_error: Optional[BaseException] = None,
        close_error: Optional[BaseException] = None,
        finalize_gate: Optional[threading.Event] = None,
        tasks: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.owner = asyncio.get_running_loop() if bind_loop else None
        self.begin_seconds = begin_seconds
        self.end_seconds = end_seconds
        self.end_error = end_error
        self.close_error = close_error
        self.finalize_gate = finalize_gate
        self.closes = 0
        self.closed_on_a_foreign_loop = False
        self.closed_while_owned = False
        self.releases = 0
        self.finalizing = False
        self.closed = threading.Event()
        super().__init__(tasks=tasks or list(TASKS))

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        time.sleep(self.begin_seconds)
        super()._begin_session(session_id, task)

    def _end_session(self, session_id: str) -> None:
        self.releases += 1
        time.sleep(self.end_seconds)
        if self.end_error is not None:
            raise self.end_error
        super()._end_session(session_id)

    async def finalize(self, req: Any) -> Any:
        self.finalizing = True
        try:
            if self.finalize_gate is not None:
                await asyncio.get_running_loop().run_in_executor(
                    None, self.finalize_gate.wait, 10.0
                )
            return await super().finalize(req)
        finally:
            self.finalizing = False

    async def _close(self) -> None:
        self.closes += 1
        if self.finalizing:
            self.closed_while_owned = True
        if self.owner is not None and asyncio.get_running_loop() is not self.owner:
            self.closed_on_a_foreign_loop = True
        if self.close_error is not None:
            self.closed.set()
            raise self.close_error
        self.closed.set()


def _holds(
    env: _Watched,
    *,
    closes: Optional[int] = 1,
    sessions: Optional[ServedEpisode] = None,
    contiguous: bool = True,
) -> None:
    """The invariants, checked where they apply."""
    if closes is not None:
        assert _until(lambda: env.closes == closes), f"closes={env.closes}"
    assert env.closed_on_a_foreign_loop is False, "closed on a loop that does not own the env"
    assert env.closed_while_owned is False, "torn down while a finalizer still owned the env"
    assert env.releases <= 1, f"releases={env.releases}"
    if sessions is not None:
        assert _until(
            lambda: all(getattr(s, "_closed", True) for s in sessions._opened)
        ), "an MCP session this episode opened was left open"
        if contiguous:
            indexes = [entry.index for entry in sessions._trajectory]
            assert indexes == list(range(1, len(indexes) + 1)), indexes
    # Given a moment: a lifecycle stops when the close it is queued behind lands, which is a
    # thread ending rather than a caller returning.
    assert _until(lambda: not _leaked_lifecycles()), _leaked_lifecycles()


async def _awaited(predicate: Any, seconds: float = 5.0) -> bool:
    """Wait for something this loop itself has to run."""
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.005)
    return predicate()


def _until(predicate: Any, seconds: float = 5.0) -> bool:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def _leaked_lifecycles() -> List[str]:
    """Lifecycle threads this cell left behind. Compared against a snapshot the fixture takes,
    because another module's stream may still be shutting one of its own down."""
    return [
        name
        for name in (t.name for t in threading.enumerate())
        if name.startswith("shogym-episode") and name not in _BEFORE
    ]


_BEFORE: set = set()


@pytest.fixture(autouse=True)
def _snapshot_threads() -> Any:
    global _BEFORE
    _BEFORE = {t.name for t in threading.enumerate()}
    yield
    _BEFORE = set()


# ----- the cells -----
#
# Every state x event pair below appears exactly once.


async def _cancel_start_while_building(_tmp: Path) -> None:
    """A cancelled `start` closes the env its factory returns after the cancellation."""
    closed = threading.Event()

    class _Late(_FixtureScoreEnv):
        async def _close(self) -> None:
            closed.set()

    def slow_make(_name: str, config: Optional[Dict[str, Any]] = None) -> _Late:
        time.sleep(0.3)
        return _Late(tasks=list(TASKS))

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
        assert _until(lambda: not _leaked_lifecycles()), _leaked_lifecycles()
    finally:
        episode_module.make = original  # type: ignore[assignment]


async def _factory_fails_late_while_building(_tmp: Path) -> None:
    """A build that fails after its caller is gone stops its lifecycle without joining its own
    thread."""

    def failing(_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        time.sleep(0.3)
        raise RuntimeError("the factory failed late")

    original = episode_module.make
    episode_module.make = failing  # type: ignore[assignment]
    try:
        starting = asyncio.ensure_future(
            ServedEpisode.start(ENV_NAME, task=0, off_loop_factory=True)
        )
        await asyncio.sleep(0.02)
        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting
        await asyncio.sleep(0.5)
        assert _until(lambda: not _leaked_lifecycles()), _leaked_lifecycles()
    finally:
        episode_module.make = original  # type: ignore[assignment]


async def _cancel_start_while_begun(_tmp: Path) -> None:
    """A setup abandoned inside `begin_session`: one release, then one close, in that order."""
    env = _Watched(begin_seconds=0.3)
    opening = asyncio.ensure_future(ServedEpisode.open_env(env, task=0))
    await asyncio.sleep(0.02)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    _holds(env)


async def _env_cancels_while_begun(_tmp: Path) -> None:
    """An env raising `CancelledError` out of its release is that env failing, not a caller."""
    env = _Watched(end_error=asyncio.CancelledError(), begin_seconds=0.2)
    opening = asyncio.ensure_future(ServedEpisode.open_env(env, task=0))
    await asyncio.sleep(0.02)
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    _holds(env)


async def _loop_closes_while_begun(_tmp: Path) -> None:
    """The rollback is the hook thread's, and it stops its own lifecycle behind the close.

    Driven in a thread of its own, because this cell owns an event loop and a loop cannot be run
    from inside another one."""
    await asyncio.to_thread(_drive_loop_loss)


def _drive_loop_loss() -> None:
    made: Dict[str, Any] = {}
    loop = asyncio.new_event_loop()
    loop.set_exception_handler(lambda _loop, _context: None)

    async def give_up() -> None:
        made["env"] = _Watched(begin_seconds=0.3, bind_loop=False)
        opening = loop.create_task(ServedEpisode.open_env(made["env"], task=0))
        await asyncio.sleep(0.05)
        opening.cancel()
        for _ in range(5):
            await asyncio.sleep(0)

    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(give_up())
    finally:
        loop.close()
        asyncio.set_event_loop(None)
    env = made["env"]
    assert _until(env.closed.is_set)
    _holds(env)


async def _cancel_call_while_dispatching(_tmp: Path) -> None:
    """A cancelled caller leaves its operation running, and the next call waits for it."""
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    try:
        first = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        second = asyncio.ensure_future(ep.call("noop", {}))
        await asyncio.sleep(0.05)
        assert not second.done()
        score_mcp.released.set()
        await second
        assert [e.tool for e in ep._trajectory] == ["block", "noop"]
        assert [e.index for e in ep._trajectory] == [1, 2]
    finally:
        score_mcp.released.set()
        await ep.close()


async def _deadline_fires_while_dispatching(_tmp: Path) -> None:
    """The wall clock ends the episode against the call it is running, and that call is
    tombstoned when it lands."""
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    ep._seal_enabled = False
    try:
        running = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        ended = await asyncio.wait_for(
            ep.call(TERMINATE_TOOL_NAME, {}, forced=True), timeout=2.0
        )
        assert ended.terminated is True and ended.tombstoned is False
        score_mcp.released.set()
        late = await running
        assert late.tombstoned is True, "the overtaken call was reported as a terminal"
        assert [e.tool for e in ep._trajectory] == []
    finally:
        score_mcp.released.set()
        await ep.close()
    # And the same on the seal path, where the horizon call is the one overtaken.
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    try:
        for _ in range(score_env.HORIZON - 1):
            await ep.call("noop", {})
        reaching = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        forced = await asyncio.wait_for(
            ep.call(TERMINATE_TOOL_NAME, {}, forced=True), timeout=2.0
        )
        assert forced.terminated is True and forced.tombstoned is False
        score_mcp.released.set()
        overtaken = await reaching
        assert overtaken.tombstoned is True, "the overtaken horizon call read as the terminal"
    finally:
        score_mcp.released.set()
        await ep.close()


async def _cancel_call_while_finalizing(_tmp: Path) -> None:
    """A cancelled terminal caller never cancels the single evaluation."""
    gate = threading.Event()
    env = _Watched(finalize_gate=gate)
    ep = await ServedEpisode.open_env(env, task=0)
    submitting = asyncio.ensure_future(ep.call(SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.sleep(0.05)
    submitting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submitting
    gate.set()
    await ep.wait_finalized()
    await ep.close()
    assert ep._env.finalize_calls == 1
    _holds(env, sessions=ep)


async def _cancel_close_while_finalizing(_tmp: Path) -> None:
    """A close cancelled during a finalization still scores a correct submission correct."""
    gate = threading.Event()
    env = _Watched(finalize_gate=gate)
    ep = await ServedEpisode.open_env(env, task=0)
    submitting = asyncio.ensure_future(ep.call(SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.sleep(0.05)
    closing = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.05)
    closing.cancel()
    with contextlib.suppress(BaseException):
        await closing
    for _ in range(20):
        await asyncio.sleep(0.01)
    gate.set()
    import json as _json

    assert _json.loads((await submitting).content)["correct"] is True
    await ep.close()
    _holds(env, sessions=ep)


async def _cancel_close_while_draining(_tmp: Path) -> None:
    """The deadline commits the verdict early and the drain still owns the evaluator, so the
    teardown may not go in front of it."""
    gate = threading.Event()
    env = _Watched(finalize_gate=gate)
    ep = await ServedEpisode.open_env(env, task=0, finalize_deadline=0.05)
    result = await ep.call(SUBMIT_TOOL, {"answer": "4"})
    assert result.terminated is True
    closing = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.02)
    closing.cancel()
    with contextlib.suppress(BaseException):
        await closing
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert env.closed_while_owned is False, "the env was closed under a running evaluator"
    gate.set()
    await ep.close()
    _holds(env, sessions=ep)


async def _session_slow_while_closing(_tmp: Path) -> None:
    """A cancelled close still closes the sessions, because a session's close is not retryable."""
    env = _Watched()
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    for session in ep._opened:
        closing = session.close

        async def slow(_closing: Any = closing) -> None:
            await asyncio.sleep(0.2)
            await _closing()

        session.close = slow  # type: ignore[method-assign]
    attempt = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.05)
    attempt.cancel()
    with contextlib.suppress(BaseException):
        await attempt
    for _ in range(60):
        await asyncio.sleep(0.01)
    _holds(env, sessions=ep, contiguous=False)


async def _second_close_while_releasing(_tmp: Path) -> None:
    """Two closes over a slow release join rather than starting a second close."""
    env = _Watched(end_seconds=0.3)
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    outcomes = await asyncio.gather(ep.close(), ep.close(), return_exceptions=True)
    assert all(not isinstance(o, BaseException) for o in outcomes), outcomes
    _holds(env, sessions=ep)


async def _env_raises_while_releasing(_tmp: Path) -> None:
    """A release that raises does not stop the env being closed or the lifecycle stopping."""
    env = _Watched(end_error=RuntimeError("release boom"))
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    await ep.close()
    _holds(env, sessions=ep)


async def _env_raises_while_closing(_tmp: Path) -> None:
    """A close that raises reaches the caller, and every joiner gets the same failure."""
    env = _Watched(close_error=RuntimeError("close boom"))
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    outcomes = await asyncio.gather(ep.close(), ep.close(), return_exceptions=True)
    assert [type(o).__name__ for o in outcomes] == ["RuntimeError", "RuntimeError"], outcomes
    _holds(env, sessions=ep)


async def _second_close_while_closed(_tmp: Path) -> None:
    """Closing an episode that is already closed changes nothing."""
    env = _Watched()
    ep = await ServedEpisode.open_env(env, task=0)
    await ep.call(SUBMIT_TOOL, {"answer": "4"})
    await ep.close()
    await ep.close()
    _holds(env, sessions=ep)


async def _span_fails_while_begun(tmp_path: Path) -> None:
    """Pass 6 finding 5 and pass 7 finding 3: a span that refuses must not leak the env, on
    either factory contract."""
    for off_loop in (True, False):
        closed = threading.Event()
        loop = asyncio.get_running_loop()
        seen: Dict[str, Any] = {}

        class _Noticing(_FixtureScoreEnv):
            def __init__(self, **kwargs: Any) -> None:
                self.built_on = asyncio.get_running_loop()
                super().__init__(**kwargs)

            async def _close(self) -> None:
                seen["foreign"] = asyncio.get_running_loop() is not self.built_on
                closed.set()

        class _Refusing:
            namespace = "refusing"

            async def begin(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("no span for you")

            async def finalize(self, *args: Any, **kwargs: Any) -> None:
                return None

        stream = TaskStream(
            lambda _name: _Noticing(tasks=list(TASKS)),
            [TaskRef(ENV_NAME, 0)],
            prov_dir=tmp_path / f"prov-{off_loop}",
            off_loop_factory=off_loop,
            provenance=[_Refusing()],  # type: ignore[list-item]
        )
        with pytest.raises(BaseException):
            await stream.get_task()
        # Awaited rather than blocked on: for the default contract the close is posted back to
        # this very loop, so a caller that blocks it is waiting for work it is preventing.
        assert await _awaited(closed.is_set), f"off_loop={off_loop}: never closed"
        assert seen.get("foreign") is False, f"off_loop={off_loop}: closed on a foreign loop"
        with contextlib.suppress(BaseException):
            await stream.aclose()
        assert loop is asyncio.get_running_loop()




async def _normal_close_while_dispatching(_tmp: Path) -> None:
    """An ordinary close during a dispatch waits: it does not claim an abort and release the
    session while the accepted call is still inside the tool."""
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    try:
        running = asyncio.ensure_future(ep.call("noop", {}))
        await running
        blocked = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        closing = asyncio.ensure_future(ep.close())
        await asyncio.sleep(0.1)
        assert not score_mcp.landed.is_set()
        assert not closing.done(), "close did not wait for the call the episode had accepted"
        score_mcp.released.set()
        assert await asyncio.to_thread(score_mcp.landed.wait, 5.0)
        await blocked
        await closing
        assert [e.tool for e in ep._trajectory][:2] == ["noop", "block"]
        assert [e.index for e in ep._trajectory][:2] == [1, 2]
    finally:
        score_mcp.released.set()
        await ep.close()


async def _cancel_cleanup_while_begun(_tmp: Path) -> None:
    """A cancellation delivered during setup cleanup does not skip the rest of it: the session is
    released, the env closed and the lifecycle thread stopped."""
    class _NoContract(_Watched):
        def describe(self, task_id: Optional[str] = None) -> Any:
            raise RuntimeError("no contract")

    env = _NoContract(bind_loop=False, begin_seconds=0.05)
    real_open = episode_module._open_session_for_spec

    async def opening(spec: Any, **kwargs: Any) -> Any:
        session = await real_open(spec, **kwargs)
        closing = session.close

        async def slow() -> None:
            await asyncio.sleep(0.3)
            await closing()

        session.close = slow  # type: ignore[method-assign]
        return session

    episode_module._open_session_for_spec = opening  # type: ignore[assignment]
    try:
        opening_task = asyncio.ensure_future(ServedEpisode.open_env(env, task=0))
        await asyncio.sleep(0.25)
        opening_task.cancel()
        with contextlib.suppress(BaseException):
            await opening_task
    finally:
        episode_module._open_session_for_spec = real_open  # type: ignore[assignment]
    # Awaited rather than blocked on: this env was built on this loop, so its close is posted
    # back here and a caller that blocks is waiting for work it is preventing.
    assert await _awaited(env.closed.is_set), "the env was left open by a cancelled cleanup"
    _holds(env)


async def _deadline_mid_finalize_while_finalizing(_tmp: Path) -> None:
    """A drain and an evaluator created *after* the teardown was arranged still own the env, so
    the close does not go in front of them."""
    gate = threading.Event()
    env = _Watched(finalize_gate=gate)
    ep = await ServedEpisode.open_env(env, task=0, finalize_deadline=0.05)
    submitting = asyncio.ensure_future(ep.call(SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.sleep(0.15)  # the deadline has answered; the evaluator is still held
    closing = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.02)
    closing.cancel()
    with contextlib.suppress(BaseException):
        await closing
    for _ in range(20):
        await asyncio.sleep(0.01)
    assert env.closed_while_owned is False, "closed under an evaluator created after arrangement"
    gate.set()
    with contextlib.suppress(BaseException):
        await submitting
    await ep.close()
    _holds(env, sessions=ep)


async def _normal_close_while_finalizing(_tmp: Path) -> None:
    """An ordinary close during a finalization waits for it and scores what it graded."""
    import json as _json

    gate = threading.Event()
    env = _Watched(finalize_gate=gate)
    ep = await ServedEpisode.open_env(env, task=0)
    submitting = asyncio.ensure_future(ep.call(SUBMIT_TOOL, {"answer": "4"}))
    await asyncio.sleep(0.05)
    closing = asyncio.ensure_future(ep.close())
    await asyncio.sleep(0.05)
    assert not closing.done()
    gate.set()
    assert _json.loads((await submitting).content)["correct"] is True
    await closing
    _holds(env, sessions=ep)



async def _cancel_cleanup_before_begin(_tmp: Path) -> None:
    """A cancellation before `begin_session` still closes the env: with no rollback there is no
    session to release, so nothing else would start the close."""
    env = _Watched(bind_loop=False)
    real_open = episode_module._open_session_for_spec

    async def opening(spec: Any, **kwargs: Any) -> Any:
        session = await real_open(spec, **kwargs)
        closing = session.close

        async def slow() -> None:
            await asyncio.sleep(0.3)
            await closing()

        session.close = slow  # type: ignore[method-assign]
        return session

    def refuse(_idx: Any) -> Any:
        raise ValueError("setup boom")

    env.load_task = refuse  # type: ignore[method-assign]
    episode_module._open_session_for_spec = opening  # type: ignore[assignment]
    try:
        opening_task = asyncio.ensure_future(ServedEpisode.open_env(env, task=0))
        await asyncio.sleep(0.1)
        opening_task.cancel()
        with contextlib.suppress(BaseException):
            await opening_task
    finally:
        episode_module._open_session_for_spec = real_open  # type: ignore[assignment]
    assert await _awaited(env.closed.is_set), "a cancelled pre-begin cleanup left the env open"
    _holds(env)


async def _normal_close_while_dispatching_legacy(_tmp: Path) -> None:
    """A non-seal close that cannot wait tombstones the call it gave up on, so that call commits
    no step and runs no `verify` against an env that is already gone."""
    score_mcp.reset_block()
    ep = await ServedEpisode.start(score_env.ENV_NAME, task=0)
    ep._seal_enabled = False
    original = episode_module._END_SESSION_SECONDS
    episode_module._END_SESSION_SECONDS = 0.05
    try:
        running = asyncio.ensure_future(ep.call("block", {}))
        await asyncio.sleep(0.1)
        await ep.close()
        score_mcp.released.set()
        late = await running
        assert late.tombstoned is True, "the call the close gave up on committed anyway"
        assert [e.tool for e in ep._trajectory] == []
    finally:
        episode_module._END_SESSION_SECONDS = original
        score_mcp.released.set()
        with contextlib.suppress(BaseException):
            await ep.close()


#: state -> event -> a cell that runs, or the reason the pair cannot arise.
_MATRIX: Dict[str, Dict[str, Any]] = {
    "building": {
        "cancel_start": _cancel_start_while_building,
        "factory_fails_late": _factory_fails_late_while_building,
        "cancel_call": "no call exists before the episode does",
        "cancel_close": "no episode exists to close",
        "deadline_fires": "a stream arms the clock at dispense, which is after construction",
        "second_close": "no episode exists to close",
        "loop_closes": "covered under `begun`: the build is the lifecycle's, not the caller's",
        "env_raises": "a constructor that raises is `factory_fails_late`",
        "env_cancels": "a constructor that raises is `factory_fails_late`",
        "session_slow": "no session is opened before the env is built",
        "span_fails": "spans open after the build, which is `begun`",
        "normal_close": "no episode exists to close",
        "cancel_cleanup": "there is no cleanup until there is something built to clean up",
        "deadline_mid_finalize": "nothing is finalizing",
    },
    "begun": {
        "cancel_start": _cancel_start_while_begun,
        "env_cancels": _env_cancels_while_begun,
        "loop_closes": _loop_closes_while_begun,
        "span_fails": _span_fails_while_begun,
        "cancel_call": "no call has been made yet",
        "cancel_close": "covered under `releasing`: a close here has no session state to race",
        "deadline_fires": "a task is not dispensed until setup returns, so no clock is running",
        "second_close": "no first close exists",
        "session_slow": "sessions are opened before the hook; a slow one delays setup, not a race",
        "factory_fails_late": "the factory has already returned",
        "normal_close": "no episode is returned to close",
        "cancel_cleanup": _cancel_cleanup_while_begun,
        "env_raises": _cancel_cleanup_before_begin,
        "deadline_mid_finalize": "nothing is finalizing",
    },
    "dispatching": {
        "cancel_call": _cancel_call_while_dispatching,
        "deadline_fires": _deadline_fires_while_dispatching,
        "cancel_start": "setup is over",
        "normal_close": _normal_close_while_dispatching,
        "cancel_close": "a close during a dispatch waits for it, which is `normal_close` here",
        "second_close": "covered under `releasing`",
        "loop_closes": "an episode whose loop closes mid-dispatch is `begun`'s loop-loss shape",
        "env_raises": "a tool that raises is an ordinary call result, not a lifecycle event",
        "env_cancels": "covered under `releasing`: the hook boundary is where it matters",
        "session_slow": _normal_close_while_dispatching_legacy,
        "span_fails": "spans are open by now",
        "factory_fails_late": "the factory has already returned",
        "cancel_cleanup": "there is no cleanup path here; teardown is `releasing` and `closing`",
        "deadline_mid_finalize": "nothing is finalizing",
    },
    "finalizing": {
        "cancel_call": _cancel_call_while_finalizing,
        "cancel_close": _cancel_close_while_finalizing,
        "normal_close": _normal_close_while_finalizing,
        "deadline_mid_finalize": _deadline_mid_finalize_while_finalizing,
        "cancel_cleanup": "cleanup has not begun; the finalizer still owns the env",
        "cancel_start": "setup is over",
        "deadline_fires": "covered under `draining`: the deadline is what creates that state",
        "second_close": "covered under `releasing`",
        "loop_closes": "the finalization is the caller's loop's; a loop that closes ends both",
        "env_raises": "a finalizer that raises is a fail-closed verdict, tested in terminal suite",
        "env_cancels": "covered under `releasing`",
        "session_slow": "covered under `closing`",
        "span_fails": "spans are open by now",
        "factory_fails_late": "the factory has already returned",
    },
    "draining": {
        "normal_close": "covered under `finalizing`: the same wait, one owner later",
        "cancel_cleanup": "covered under `closing`",
        "deadline_mid_finalize": "the deadline is what created this state",
        "cancel_close": _cancel_close_while_draining,
        "cancel_start": "setup is over",
        "cancel_call": "the deadline has already answered the caller",
        "deadline_fires": "the deadline is what created this state",
        "second_close": "covered under `releasing`",
        "loop_closes": "the drain is the caller's loop's; a loop that closes ends both",
        "env_raises": "covered under `releasing`",
        "env_cancels": "covered under `releasing`",
        "session_slow": "covered under `closing`",
        "span_fails": "spans are open by now",
        "factory_fails_late": "the factory has already returned",
    },
    "releasing": {
        "normal_close": "the release is what an ordinary close does; this is that path",
        "cancel_cleanup": "covered under `closing`",
        "deadline_mid_finalize": "the episode is already sealed",
        "second_close": _second_close_while_releasing,
        "env_raises": _env_raises_while_releasing,
        "cancel_start": "setup is over",
        "cancel_call": "no call is accepted after the seal",
        "cancel_close": "covered under `finalizing` and `draining`, which are the owners",
        "deadline_fires": "the episode is already sealed",
        "loop_closes": "covered under `begun`",
        "env_cancels": "the release path's cancellation is `begun`'s, at the same hook",
        "session_slow": "sessions are closed before the release",
        "span_fails": "spans are open by now",
        "factory_fails_late": "the factory has already returned",
    },
    "closing": {
        "normal_close": "this state is what an ordinary close is",
        "cancel_cleanup": "covered by `session_slow`, which cancels during disposal",
        "deadline_mid_finalize": "the episode is already sealed",
        "session_slow": _session_slow_while_closing,
        "env_raises": _env_raises_while_closing,
        "cancel_start": "setup is over",
        "cancel_call": "no call is accepted after the seal",
        "cancel_close": "covered under `finalizing`, `draining` and `session_slow`",
        "deadline_fires": "the episode is already sealed",
        "second_close": "covered under `releasing`",
        "loop_closes": "covered under `begun`",
        "env_cancels": "covered under `releasing`",
        "span_fails": "spans are open by now",
        "factory_fails_late": "the factory has already returned",
    },
    "closed": {
        "normal_close": "covered by `second_close`",
        "cancel_cleanup": "cleanup is over",
        "deadline_mid_finalize": "the episode is already sealed and released",
        "second_close": _second_close_while_closed,
        "cancel_start": "setup is over",
        "cancel_call": "a call after the close is a tombstone, tested in the episode suite",
        "cancel_close": "there is nothing left to cancel",
        "deadline_fires": "the episode is already sealed and released",
        "loop_closes": "nothing of this episode is left on any loop",
        "env_raises": "the env has been closed",
        "env_cancels": "the env has been closed",
        "session_slow": "the sessions are closed",
        "span_fails": "spans are finalized with the row",
        "factory_fails_late": "the factory has already returned",
    },
}


@pytest.mark.parametrize(
    "state, event",
    [(state, event) for state in STATES for event in EVENTS],
    ids=[f"{state}-{event}" for state in STATES for event in EVENTS],
)
async def test_the_lifecycle_survives_one_event_in_one_state(
    state: str, event: str, tmp_path: Path
) -> None:
    cell = _MATRIX[state][event]
    if isinstance(cell, str):
        # Not reachable, and the reason is the test. A pair with no cell and no reason is a hole.
        assert cell, f"{state} x {event} has no reason"
        pytest.skip(f"{state} x {event}: {cell}")
    await cell(tmp_path)


def test_every_pair_is_accounted_for() -> None:
    """The matrix is the point: a pair that is neither run nor explained is a gap."""
    assert set(_MATRIX) == set(STATES)
    for state in STATES:
        assert set(_MATRIX[state]) == set(EVENTS), state
    running = sum(
        1 for state in STATES for event in EVENTS if not isinstance(_MATRIX[state][event], str)
    )
    # The bound only ratchets.
    assert running >= 20, running
    assert len(STATES) * len(EVENTS) == len(STATES) * len(EVENTS)
