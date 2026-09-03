"""The episode-serving engine (RFC 008): drive a served env one tool call at a time.

A tool call *is* the step. :class:`ServedEpisode` opens the env's essential MCP sessions
directly (the terminate server + the env-mandatory servers), dispatches each incoming call
to the right session, records a flat :class:`~shogym.trajectory.Trajectory`, gates the
horizon env-side, and runs the env's pure ``verify`` on each call. Feedback rides back on
the ``_meta`` sidecar (episode-level hidden until the terminal result) and every step is
appended to the JSONL trace. No gym ``step``/``Observation`` anywhere.

**Seal-before-verdict (durable + middleware-gated).** An env opts in by declaring a
``score``-terminal tool (``terminal_kind="score"`` in its manifest) *and* a ``finalize`` hook.
For such an env, a score-terminal call runs a *terminal transaction* instead of an ordinary
step: :meth:`call` (1) validates the args against the full advertised JSON schema while the
episode is still OPEN — an invalid request is a normal validation error, never a sealed
finalizer with unvalidated evidence; (2) atomically seals the episode (``OPEN -> SEALED``);
(3) runs ``finalize`` as a single, cancellation-safe finalization producing core-owned
:class:`~shogym.serve.lifecycle.TerminalEvidence`; (4) records the evidence in the **durable
finalization store** and appends a versioned ``terminal`` trace event; (5) tears down; (6)
returns a public-safe, sanitized payload. After the seal every ``tools/call`` is tombstoned
with no inward dispatch. The state machine is
``OPEN -> SEALED -> FINALIZING -> FINALIZED -> TEARING_DOWN -> CLOSED``.

An env that marks nothing ``score`` never enters this path: ``_seal_enabled`` is False, so
:meth:`call` runs *only* the legacy single-step body — its behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import contextlib
import contextvars
import functools
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jsonschema

from shogym.envs import make
from shogym.feedback.wire import build_meta, dump_item, select_inband
from shogym.mcp.session import MCPSession
from shogym.mcp.toolset import _open_session_for_spec
from shogym.serve.lifecycle import (
    FinalizationRecord,
    FinalizationStore,
    FinalizeRequest,
    LifecycleState,
    TerminalEvidence,
    args_digest,
    fail_closed_verdict,
    failure_summary,
)
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.task import TaskSpec
from shogym.trace import (
    append_terminal_event,
    append_trace,
    step_record,
    terminal_event_record,
)
from shogym.trajectory import Step, Trajectory
from shogym.utils.uuid7 import uuid7


def _named(key: Any) -> str:
    """The schema key a validation error is *about*, or a placeholder when naming it raises.

    A schema key is the env's own object and ``repr`` is that object's code, run here only to say
    which argument the caller has to fix. Unguarded it replaces the validation error with its own
    exception, which is the one outcome this check exists to prevent: the episode is left OPEN
    with the terminal call the agent made answered by a traceback, and the harness above ends up
    composing the task's outcome from a call it never carried. The refusal is the point and the
    name is the decoration, so the name is what gives way — and nothing of the failure is echoed
    into it, because this message goes to the caller and an env's exception text is not the
    caller's business."""
    try:
        return repr(key)
    except Exception:  # noqa: BLE001 — the refusal outranks its own decoration
        return "<a key this schema cannot name>"


def _json_safe(obj: Any) -> bool:
    """True iff ``obj`` serializes to strict JSON (no NaN/Inf, only JSON types) — the same
    contract the trace store enforces with ``allow_nan=False``."""
    try:
        json.dumps(obj, allow_nan=False)
        return True
    except (ValueError, TypeError):
        return False


def _wire_form(spec: TaskSpec) -> TaskSpec:
    """The published contract as the wire carries it: every advertised tool round-tripped through
    JSON, so the values this episode *enforces* are the values a client is *shown*.

    A schema is the env's object, and a JSON scalar in it may be a subclass — the models coerce
    one away at construction but do not validate on assignment, so it reaches here verbatim. It
    serializes like the scalar it subclasses, which is exactly what makes it invisible: a server
    advertises ordinary text while this episode validates against something that answers a
    comparison differently. The agent then sends what it was told to send and the terminal
    transaction refuses it, which the harness above can only record as a wrong answer.

    **Contained, and that is not a hole.** A schema that will not serialize keeps the env's own
    object, because the only honest alternative here is to fail opening an episode that a caller
    may have no other way to refuse — and refusing it is a decision for the layer that knows
    whether an alternative exists. A stream makes that decision one step later and stops the run:
    it compares this contract against the one its endpoint published, and a schema that cannot be
    serialized cannot be compared either (see :meth:`TaskStream._require_published_manifest`), so
    the task is never dispensed. What is left is a single-episode server enforcing exactly what it
    advertises, which is what it did before."""
    tools = []
    changed = False
    for manifest in spec.tools:
        try:
            # The whole advertised tool, not the schema alone. A name is identity here — the
            # score terminal is found by looking this episode's own key up against the name a
            # call arrives under — so a `str` subclass answering that comparison its own way is
            # a terminal call dispatched as an ordinary step, sealing nothing and scoring
            # nothing, on a tool the endpoint advertised as the way to finish the task.
            wire = json.loads(
                json.dumps(
                    {
                        "name": manifest.name,
                        "description": manifest.description,
                        "input_schema": manifest.input_schema,
                    },
                    allow_nan=False,
                )
            )
        except (ValueError, TypeError):
            tools.append(manifest)
            continue
        if not isinstance(wire["name"], str) or not isinstance(wire["description"], str):
            tools.append(manifest)
            continue
        changed = True
        tools.append(manifest.model_copy(update=wire))
    return spec.model_copy(update={"tools": tools}) if changed else spec


#: How long a caller waits on a lifecycle operation before going on without it. A bound on the
#: *wait*, never on the work: the lifecycle thread owns what it was given and finishes it whether
#: or not anyone is still listening.
_ROLLBACK_SECONDS = 60.0

#: How long teardown waits for an env to release one episode's resources before going on without
#: it. Teardown runs on the shared loop, so this is a bound on the wait rather than a kill: an env
#: whose release has to be certain makes its own hook bounded. Spent once per session.
_END_SESSION_SECONDS = 60.0

#: Every lifecycle this process has started and not yet stopped. Held so the exit hook can stop
#: them: the threads are not daemons, because a close abandoned half way through is worse than an
#: exit that waits for it, and an interpreter that waits forever is worse than either.
_LIVE: "set[_Lifecycle]" = set()
_LIVE_LOCK = threading.Lock()

#: Closes posted onto a loop that owns an env. A task nothing holds may be collected before it
#: runs, so each one is kept until it is done.
_PENDING_CLOSES: "set[asyncio.Task[None]]" = set()

#: How long the exit hook gives one lifecycle to finish what it is doing.
_EXIT_SECONDS = 10.0


def _running_loop() -> "Optional[asyncio.AbstractEventLoop]":
    """The loop this call is on, or ``None`` when it is not on one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class _Lifecycle:
    """One episode's own thread, running one event loop that belongs to nobody else.

    Everything an episode does *to its env* happens here: the env is built on this loop when the
    caller has said its factory may be built off theirs, the session hooks run on this thread, and
    the env is closed on this loop. The caller's loop only ever awaits futures, with bounds it
    chooses, and a caller that stops waiting, is cancelled, or takes its whole loop down changes
    nothing about what happens to the env.

    **Why a loop of our own rather than a worker thread and a rulebook.** The env has to be closed
    on the loop that built it, closing has to follow the session release, and neither may run on
    the loop that is serving other episodes. Trying to satisfy those with a worker thread that
    hands work back to somebody else's loop needs an answer to "is that loop ever going to run
    this?", and there is no answer: a loop that is open and idle is a loop between two
    ``run_until_complete`` calls and also a loop nobody will ever turn again, and no amount of
    elapsed time tells them apart. Owning the loop removes the question. This one is running from
    the moment the episode exists until the moment it is closed, so "the loop that built the env"
    and "a loop that will run this" are the same loop by construction.

    **The env still sees a loop.** A factory built here runs *on* this loop, so an env that binds
    ``asyncio.get_running_loop()`` in its constructor binds this one, which outlives every caller.
    That is what makes ``off_loop_factory`` safe rather than merely fast: it moves construction off
    the *caller's* loop, not off every loop.

    An env built on the caller's loop keeps the caller's loop, and this class never closes it while
    that loop can still run (see :meth:`close_env`)."""

    def __init__(self, name: str) -> None:
        self._loop = asyncio.new_event_loop()
        started = threading.Event()
        # A daemon, and that is a trade rather than an oversight. A non-daemon thread cannot both
        # own cleanup that an env may take as long as it likes over and promise the process a
        # finite exit: Python joins it before anything of ours runs again, so a wedged hook stops
        # the interpreter for ever. The exit hook below gives every lifecycle `_EXIT_SECONDS` to
        # finish properly first, and one still inside a hook after that is reported rather than
        # waited for.
        self._thread = threading.Thread(
            target=self._serve, args=(started,), name=f"shogym-episode-{name}", daemon=True
        )
        self._stopped = False
        self._lock = threading.Lock()
        self._on_stop: "List[Callable[[], None]]" = []
        self._thread.start()
        started.wait()
        with _LIVE_LOCK:
            _LIVE.add(self)

    def _serve(self, started: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.call_soon(started.set)
        try:
            self._loop.run_forever()
        finally:
            try:
                self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            except BaseException:  # noqa: BLE001 - nothing above this to report to
                pass
            self._loop.close()
            self._mark_stopped()

    @property
    def loop(self) -> "asyncio.AbstractEventLoop":
        return self._loop

    @property
    def running(self) -> bool:
        return not self._stopped and self._thread.is_alive()

    def run(
        self, coro: "Any", context: "Optional[contextvars.Context]" = None
    ) -> "concurrent.futures.Future[Any]":
        """Run one coroutine on this lifecycle's loop, from anywhere.

        The future it hands back retains its outcome, so every joiner gets the same answer rather
        than only the first one, and a joiner that arrives after the work is done gets it without
        waiting at all.

        ``context`` is the caller's, when the caller has one worth carrying. A task copies the
        context of whatever *creates* it, and what creates a task submitted from another thread
        is the loop thread, so a coroutine sent here without one runs under the lifecycle's own
        context rather than under the context of whoever asked for it. An env's close releases
        what its constructor took, and a constructor that read a tenant out of a context variable
        wants that same tenant released."""
        with self._lock:
            if self._stopped:
                coro.close()
                raise RuntimeError("this episode's lifecycle has already stopped")
            if context is None:
                return asyncio.run_coroutine_threadsafe(coro, self._loop)
            out: "concurrent.futures.Future[Any]" = concurrent.futures.Future()

            async def _in_context() -> None:
                task = asyncio.get_running_loop().create_task(coro, context=context)
                try:
                    out.set_result(await task)
                except BaseException as exc:  # noqa: BLE001 - handed to the future, not raised
                    out.set_exception(exc)

            asyncio.run_coroutine_threadsafe(_in_context(), self._loop)
            return out

    def call_in(
        self, context: "contextvars.Context", fn: "Callable[..., Any]", *args: Any
    ) -> "concurrent.futures.Future[Any]":
        """One synchronous env hook on this thread, under a context somebody kept.

        The release of a session is the counterpart of opening it: a hook that reads a tenant out
        of a context variable to pick which resource to take must read the same one to give it
        back. Copying the *closer's* context released whoever happened to be closing, and left
        the opener's held."""

        async def _run() -> Any:
            return fn(*args)

        return self.run(_run(), context)

    def call(self, fn: "Callable[..., Any]", *args: Any) -> "concurrent.futures.Future[Any]":
        """Run one **synchronous** env hook on this lifecycle's thread.

        Called on the loop's own thread rather than handed to a pool: this loop is this episode's
        and nothing else is waiting on it, so a hook may hold it for as long as it holds. That is
        also what serialises the hooks. A release submitted while a setup hook is still running
        waits for it, which is the ordering an env's ``_end_session`` is entitled to and the
        reason two of them never run on one episode's resources.

        The context is copied per call, as :func:`asyncio.to_thread` does, so a hook reading a
        request-scoped value reads the caller's."""
        context = contextvars.copy_context()

        async def _run() -> Any:
            return context.run(fn, *args)

        return self.run(_run())

    def at_stop(self, callback: "Callable[[], None]") -> None:
        """Run ``callback`` when this lifecycle stops, so nothing is left waiting on work that
        this loop was going to do and now never will."""
        with self._lock:
            if self._stopped:
                stopped = True
            else:
                self._on_stop.append(callback)
                stopped = False
        if stopped:
            callback()

    def _mark_stopped(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            callbacks, self._on_stop = self._on_stop, []
        with _LIVE_LOCK:
            _LIVE.discard(self)
        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001 - a stop may not fail over what it is telling
                pass

    def stop_when(self, outcome: "concurrent.futures.Future[Any]") -> None:
        """End this lifecycle once ``outcome`` has landed, without waiting for it here.

        What a caller wants at the end of an episode is both "do not hold me" and "do not cut the
        cleanup off", and joining the thread gives only the second. This gives both: the stop is
        hung off the outcome itself, so a release still inside a hook and the env close behind it
        both finish first, and the caller returns now.

        A callback on the outcome rather than a coroutine on the loop, because a coroutine
        submitted to a loop that then closes before running it is a coroutine nobody awaits, and
        the collector says so every time. This has nothing to leave behind."""

        def stop_now(_finished: "concurrent.futures.Future[Any]") -> None:
            self.request_stop()

        outcome.add_done_callback(stop_now)

    def request_stop(self) -> None:
        """Ask the loop to stop, without waiting for it."""
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            pass

    def stop(self, timeout: Optional[float] = None) -> None:
        """Stop the loop now and let the thread end. Idempotent; never raises.

        A hook already running on this thread finishes: it is blocking the loop, so the stop is
        not seen until it returns. A coroutine *awaiting* is not resumed, which is why anything
        with somebody waiting on it registers through :meth:`at_stop`. For an end that waits for
        the work rather than cutting it off, see :meth:`stop_when`."""
        with self._lock:
            if self._stopped:
                return
        self.request_stop()
        if threading.current_thread() is self._thread:
            # Asked for from inside: a build's own completion callback runs on this thread, and
            # joining it here is joining the caller to itself, which Python answers with
            # `RuntimeError: cannot join current thread` out of a future callback nobody reads.
            # The stop is requested; the thread ends when this callback returns.
            return
        bound = timeout if timeout is not None else _EXIT_SECONDS
        self._thread.join(bound)
        # A stop that asked for no wait has not timed out; it has not waited. Warning there said
        # a hook was still running when nothing had been given a chance to finish, which is the
        # message an unknown-env construction failure was producing with no env in existence.
        if bound > 0 and self._thread.is_alive():
            # Still inside a hook. Said out loud rather than dropped quietly: this lifecycle is
            # no longer tracked and nothing will wait for it again, so if it is holding a
            # process, a port or a directory, this line is the only place that says so.
            warnings.warn(
                f"{self._thread.name} did not finish within {bound}s and is no longer waited "
                "for; an env hook is still running and whatever it holds is still held",
                RuntimeWarning,
                stacklevel=2,
            )
        self._mark_stopped()


def _stop_live_lifecycles() -> None:
    """At interpreter exit: stop every lifecycle still running, with a bound.

    The threads are not daemons, so an episode still closing an env keeps the process alive rather
    than having its cleanup cut off mid-way. A bound, because "still closing" and "wedged" look
    the same from here and an interpreter that will not exit is its own failure.

    **Registered where it has to be, not where it reads best.** ``threading`` joins every
    non-daemon thread *before* ``atexit`` runs, so a plain ``atexit`` hook that stops these loops
    would run after the join it exists to make possible, and a lifecycle nobody closed would hold
    the interpreter open for ever. ``threading._register_atexit`` runs before that join, which is
    the same hook ``concurrent.futures`` uses to shut its own pools down and for the same reason.

    Every stop is requested before any is waited for, so one wedged env cannot make the others
    wait their turn."""
    with _LIVE_LOCK:
        live = list(_LIVE)
    for lifecycle in live:
        lifecycle.request_stop()
    for lifecycle in live:
        lifecycle.stop(_EXIT_SECONDS)


def _lifecycle_for(name: str) -> "_Lifecycle":
    """One episode's lifecycle. A function so a test can watch them being made."""
    return _Lifecycle(name)


def _forget_live_lifecycles() -> None:
    """A forked child inherits the set and none of the threads in it."""
    global _LIVE, _LIVE_LOCK, _ENV_CLOSE_LOCK
    _LIVE = set()
    _LIVE_LOCK = threading.Lock()
    # And this one. A child that inherits it held by a thread that no longer exists blocks in
    # `_env_close` for ever, which is the same fork hazard the pool had and the same repair.
    _ENV_CLOSE_LOCK = threading.Lock()


_register_atexit = getattr(threading, "_register_atexit", None)
if _register_atexit is not None:  # pragma: no branch - present on every supported interpreter
    _register_atexit(_stop_live_lifecycles)
else:  # pragma: no cover - a interpreter without the private hook still gets a late stop
    atexit.register(_stop_live_lifecycles)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_live_lifecycles)


#: Where an env keeps the one close that belongs to it. On the env rather than in a table, so it
#: lives exactly as long as the env does and a second asker finds the first one's.
_ENV_CLOSE_ATTR = "_shogym_env_close"
_ENV_CLOSE_LOCK = threading.Lock()


def _env_close(
    env: Any,
    lifecycle: "_Lifecycle",
    owner: "Optional[asyncio.AbstractEventLoop]",
    context: "Optional[contextvars.Context]" = None,
) -> "_EnvClose":
    """The one close for this env, whoever asks and from wherever.

    Guarded at the env and not at the call site. Every call site had its own guard, and each one
    was right about its own path and blind to the others: a shutdown and a cancelled pull could
    both decide, correctly by their own reading, that a build had no owner, and construct one
    `_EnvClose` each. Two owners are two closes, because what makes a close single is the claim
    inside one of these objects."""
    with _ENV_CLOSE_LOCK:
        existing = getattr(env, _ENV_CLOSE_ATTR, None)
        if isinstance(existing, _EnvClose):
            return existing
        cleanup = _EnvClose(env, lifecycle, owner, context)
        try:
            setattr(env, _ENV_CLOSE_ATTR, cleanup)
        except Exception:  # noqa: BLE001 - an env that refuses attributes still gets a close
            pass
        return cleanup


class _EnvClose:
    """The one close of one env, run on the loop that owns the env and nowhere else.

    Which loop that is, is decided when the env is built and never guessed afterwards:

    * built on its episode's lifecycle loop, and it is closed there, by the lifecycle thread,
      whatever the caller is doing;
    * built on the caller's loop, and the caller closes it there. The lifecycle steps in only when
      that loop is **closed**, which is a fact rather than an inference: a closed loop can never
      run anything again. It is never taken from a loop that is merely idle, however long it stays
      idle, because a loop between two ``run_until_complete`` calls looks exactly like a loop
      nobody will turn again and only the loop itself can say which it is.

    The outcome is retained rather than signalled, so every caller that joins gets the same answer,
    including the failure."""

    def __init__(
        self,
        env: Any,
        lifecycle: "_Lifecycle",
        owner: "Optional[asyncio.AbstractEventLoop]",
        context: "Optional[contextvars.Context]" = None,
    ) -> None:
        self._context = context if context is not None else contextvars.copy_context()
        self._env = env
        self._lifecycle = lifecycle
        #: The caller loop the env was built on, or ``None`` when it was built on the lifecycle
        #: loop and is therefore this lifecycle's to close.
        self._owner = owner
        self._lock = threading.Lock()
        self._claimed = False
        self._closing: "Optional[concurrent.futures.Future[None]]" = None
        self._outcome: "concurrent.futures.Future[None]" = concurrent.futures.Future()
        lifecycle.at_stop(self._abandoned)

    def _abandoned(self) -> None:
        """The lifecycle stopped. Whatever it was going to do it will not do now, so anybody
        waiting is told rather than left waiting: an unresolved outcome is a joiner that never
        returns, including the stream holding this episode's slot."""
        if self._outcome.done():
            return
        self._finish(
            RuntimeError(
                "this episode's lifecycle stopped before its env was closed; whatever its "
                "constructor made is still held"
            )
        )

    @property
    def owned_by_caller(self) -> bool:
        return self._owner is not None

    @property
    def done(self) -> bool:
        return self._outcome.done()

    @property
    def outcome(self) -> "concurrent.futures.Future[None]":
        """The close itself, retained: every joiner gets the same answer, failure included."""
        return self._outcome

    @property
    def failure(self) -> Optional[BaseException]:
        if not self._outcome.done():
            return None
        return self._outcome.exception()

    def _claim(self) -> bool:
        with self._lock:
            if self._claimed:
                return False
            self._claimed = True
            return True

    def _finish(self, error: Optional[BaseException]) -> None:
        if self._outcome.done():
            return
        if error is None:
            self._outcome.set_result(None)
        else:
            self._outcome.set_exception(error)

    async def closed_by(self, timeout: Optional[float] = None) -> None:
        """Close the env wherever it belongs, and tell this caller what that did.

        The caller's loop when the caller built it there, this lifecycle's loop when this
        lifecycle built it. Setup cleanup used to call :meth:`here` on both, which runs
        ``env.close()`` on whatever loop is asking: a flagged env that had bound its constructor
        loop was closed on the caller's, which is the one thing building it off that loop was
        meant to make impossible."""
        if self._owner is not None and _running_loop() is self._owner:
            await self.here()
            return
        # Not this env's loop. Post it to the one that owns it (or run it here, if this lifecycle
        # is the owner) and wait for the outcome rather than closing an env's constructor state
        # from a loop it never met.
        self.close_env()
        await self.joined(timeout)

    async def here(self) -> None:
        """Close on the caller's loop, which is where a caller-built env belongs.

        Raises what the close raised: a caller that awaited a close is the one place there is
        somebody to tell. A cancellation arriving mid-close is the caller going away rather than
        an instruction to leave an env half torn down, so the close is shielded to completion and
        the cancellation handed back afterwards."""
        if not self._claim():
            await self.joined()
            return
        closing = asyncio.ensure_future(self._run())
        closing.add_done_callback(_swallow)
        try:
            await asyncio.shield(closing)
        except asyncio.CancelledError:
            with contextlib.suppress(BaseException):
                await asyncio.shield(closing)
            raise
        await self.joined()

    def close_env(self) -> "concurrent.futures.Future[None]":
        """Arrange the close on whichever loop owns the env, and hand back its outcome.

        Called from the lifecycle thread, behind the session release. For a lifecycle-built env
        this is immediate. For a caller-built one it waits: the caller closes it, or the caller's
        loop closes and this takes it over on the loop it has."""
        with self._lock:
            if self._closing is not None:
                return self._outcome
            self._closing = self._lifecycle.run(
                self._on_the_owning_loop(), self._context
            )
        return self._outcome

    def watching(self) -> "concurrent.futures.Future[Any]":
        """The lifecycle's own half of the close, for a caller that is stopping the lifecycle.

        Not :attr:`outcome`. For a caller-owned env the outcome is completed by the caller's own
        close, while the lifecycle-side watcher that was following it is still running; stopping
        on the outcome halted that loop under it, and Python said so twice at every process exit.
        This is the thing the loop is actually doing."""
        self.close_env()
        with self._lock:
            return self._closing if self._closing is not None else self._outcome

    async def _on_the_owning_loop(self) -> None:
        """The close, on whichever loop owns the env, run by the lifecycle and by nobody else.

        For an env this lifecycle built there is one loop and this is it. For an env the caller
        built, the close belongs to the caller's loop: it is *posted* there once the release is
        out, and this keeps watching. Nothing is inferred from how long that takes. A loop that
        never runs what it was given keeps the close pending, which is what an env owned by a
        loop nobody turns actually is; a loop that **closes** can run nothing ever again, and
        that is the one fact this acts on."""
        owner = self._owner
        if owner is None:
            if self._claim():
                await self._run(warn_on_failure=True)
            return
        posted = False
        while not owner.is_closed():
            if self._outcome.done():
                return
            if not self._lifecycle.running:
                # This loop is being stopped. Watching from a task on a loop that is going away
                # is watching nothing, and a task still parked when it goes is a diagnostic at
                # interpreter exit rather than a fact anyone can act on. `_abandoned` has already
                # told every joiner what happened.
                return
            if not posted and self._claimed:
                # A caller took it. Theirs to finish, and this keeps watching rather than
                # waiting: a close its loop never finishes because that loop closed is a joiner
                # who would otherwise wait for ever, and the branch below is the one that says so.
                await asyncio.sleep(0.01)
                continue
            if not posted:
                posted = self._post_to(owner)
            await asyncio.sleep(0.01)
        # The owner is closed. Whatever it was holding it will not run, so this takes what is
        # left: an untaken close is run here, and a taken one that its loop never finished is
        # reported, because a closed loop cannot come back to it.
        if self._claim():
            await self._run(warn_on_failure=True, orphaned=True)
            return
        if not self._outcome.done():
            lost = RuntimeError(
                "the loop that built this env closed while its own close was still running"
            )
            self._finish(lost)
            self._warn(lost, orphaned=True)

    def _post_to(self, owner: "asyncio.AbstractEventLoop") -> bool:
        """Ask the owning loop to run the close. ``True`` if it accepted the request."""

        def on_owner() -> None:
            if self._claim():
                task = owner.create_task(self._run())
                _PENDING_CLOSES.add(task)
                task.add_done_callback(_PENDING_CLOSES.discard)

        try:
            owner.call_soon_threadsafe(on_owner)
        except RuntimeError:
            return False
        return True

    async def _run(self, *, warn_on_failure: bool = False, orphaned: bool = False) -> None:
        self.orphaned = orphaned
        try:
            await self._env.close()
        except BaseException as exc:  # noqa: BLE001 - recorded; there may be no caller to raise to
            self._finish(exc)
            # `GeneratorExit` is not the env refusing; it is this task being torn down with the
            # loop under it, which `_abandoned` already reports in the one voice that belongs to
            # it. Warning here would say the env would not close, which is not what happened.
            if warn_on_failure and not isinstance(exc, GeneratorExit):
                self._warn(exc, orphaned=orphaned)
            return
        self._finish(None)

    def _warn(self, error: BaseException, *, orphaned: bool) -> None:
        where = (
            "the loop that built it had closed, so this ran on its episode's own loop; an env "
            "whose resources belong to the loop that built it cannot be closed that way, and "
            if orphaned
            else "no caller was waiting for it, so this is where it is reported, and "
        )
        warnings.warn(
            f"an env could not be closed: {type(error).__name__}: {error}. "
            f"{where}whatever its constructor made is still held.",
            RuntimeWarning,
            stacklevel=2,
        )

    async def joined(self, timeout: Optional[float] = None) -> None:
        """Wait for the close and raise what it raised, for **every** caller and not only the one
        that ran it. A bound on the wait; ``None`` waits for the close itself."""
        waiter = asyncio.wrap_future(self._outcome)
        if timeout is None:
            await asyncio.shield(waiter)
            return
        if timeout <= 0:
            if self._outcome.done():
                await asyncio.shield(waiter)
            return
        try:
            await asyncio.wait_for(asyncio.shield(waiter), timeout=timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return


class _SetupRollback:
    """The single owner of the rollback for a setup that was abandoned or failed mid-flight.

    A caller that gives up leaves the setup hook running and whatever it goes on to create with
    nobody to release it. The whole rollback is queued on the episode's lifecycle loop when this
    is made: the release runs when the hook ahead of it finishes, the env close is arranged after
    the release, and neither depends on a coroutine that may never be resumed or a loop that may
    not be there.

    **The MCP sessions come first.** ``Env.close`` releases what the *constructor* made, and the
    sessions opened for this episode are clients that may still be using it. The ordinary close
    path closes them and then the env; the rollback waits for the caller to say they are gone
    before it arranges the env close, and goes ahead on its own only if the caller's loop closes
    first, since a session entered on a loop that is gone cannot be exited at all."""

    def __init__(
        self,
        lifecycle: "_Lifecycle",
        env: Any,
        session_id: str,
        cleanup: "_EnvClose",
    ) -> None:
        self._cleanup = cleanup
        self._released = threading.Event()
        self._sessions_closed = threading.Event()
        self._caller = _running_loop()
        self._lifecycle = lifecycle
        self._done = lifecycle.run(self._rollback(env, session_id))

    async def _rollback(self, env: Any, session_id: str) -> None:
        try:
            env.end_session(session_id)
        except BaseException:  # noqa: BLE001 - see below
            # `BaseException`, and the one that matters is `CancelledError`. An env's hook can
            # raise it like any other code, and this loop is nobody's caller, so one arriving
            # here is the env's rather than an instruction. Caught as `Exception` it went past
            # the env close below and left the env open.
            pass
        finally:
            self._released.set()
        caller = self._caller
        while not self._sessions_closed.is_set():
            if caller is None or caller.is_closed():
                # A session entered on a loop that is gone cannot be exited on any other, so
                # there is nothing left to wait for.
                break
            await asyncio.sleep(0.01)
        # And the lifecycle stops itself behind that close. Left to a caller continuation it was
        # left to a coroutine that may never resume: a loop-loss rollback closed the env and then
        # its own thread stayed alive for the life of the process.
        self._lifecycle.stop_when(self._cleanup.watching())

    @property
    def released(self) -> bool:
        return self._released.is_set()

    def sessions_closed(self) -> None:
        """Told by the caller once every MCP session it opened has finished closing."""
        self._sessions_closed.set()

    async def settled(self, timeout: Optional[float] = None) -> bool:
        """Wait for the whole rollback, and say whether the session release landed.

        A bound rather than a promise: the caller is already carrying an error and must not be
        held by a hook that is not coming back. Nothing depends on this wait, which is only what
        lets the rollback finish before the error reaches the caller."""
        bound = _ROLLBACK_SECONDS if timeout is None else timeout
        try:
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(self._done)), timeout=bound
            )
        except BaseException:
            # Including a `CancelledError`: this runs inside a handler that re-raises the failure
            # it was called for, so the wait giving way never loses it.
            pass
        return self.released


def _with_preemption(diagnostic: Optional[str], preempted: Optional[str]) -> Optional[str]:
    """Say, privately, that this terminal overtook an ordinary call that had already been
    accepted. Only the wall clock does that, and only for an episode whose call is not coming
    back, so the fact belongs in the record even though the call itself is tombstoned out of the
    trajectory it would have been part of."""
    if preempted is None:
        return diagnostic
    note = f"the deadline's terminal overtook an accepted call to {preempted!r}"
    return note if not diagnostic else f"{diagnostic}; {note}"


async def _disposed(
    sessions: "List[MCPSession]", carried: Optional[BaseException] = None
) -> None:
    """Close every session this setup opened, whatever any one of them does.

    Failures are carried rather than dropped. Continuing past one is right, because the sessions
    after it still have to be closed; losing it is not, because the caller is being handed a setup
    error and "a session would not close" is part of what happened while answering."""
    for session in sessions:
        try:
            await session.close()
        except BaseException as failed:  # noqa: BLE001 - noted; one may not stop the rest
            if carried is not None:
                _noted(carried, "an MCP session", failed)
            else:
                warnings.warn(
                    f"an MCP session could not be closed: {type(failed).__name__}: {failed}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def _closed_when_disposed(
    disposing: "asyncio.Future[None]",
    lifecycle: "_Lifecycle",
    cleanup: "_EnvClose",
    rollback: "Optional[_SetupRollback]",
) -> None:
    """Close the env and stop the lifecycle once the sessions have finished disposing.

    Chained rather than called, because the caller that would have called it is being cancelled.
    A rollback, when there is one, releases the session first and closes the env after; without
    one there is no session and the close is the whole of it."""

    def then(_finished: "asyncio.Future[None]") -> None:
        if rollback is None:
            cleanup.close_env()
        lifecycle.stop_when(cleanup.watching())

    if disposing.done():
        then(disposing)
    else:
        disposing.add_done_callback(then)


def _noted(carried: BaseException, what: str, failure: BaseException) -> None:
    """Attach a cleanup failure to the failure already on its way out, without replacing it.

    The caller asked why setup failed, and "the env would not close" is not that answer: it is
    something that happened while answering. A note keeps both, in the order they matter."""
    try:
        carried.add_note(
            f"cleanup: closing {what} also failed: {type(failure).__name__}: {failure}"
        )
    except Exception:  # noqa: BLE001 - a note is decoration; the failure outranks it
        pass


def _settled() -> "concurrent.futures.Future[None]":
    """A future that is already done: the release for a session somebody else has claimed."""
    done: "concurrent.futures.Future[None]" = concurrent.futures.Future()
    done.set_result(None)
    return done


def _caller_cancelled() -> bool:
    """Is the task running this the one somebody asked to cancel?

    ``CancelledError`` is two unrelated things wearing one type: a caller withdrawing, and
    third-party code raising it. The count of cancellations requested on this task and not yet
    withdrawn tells them apart, and no exception can, which is why this module asks the task."""
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def _swallow(finished: "asyncio.Future[Any]") -> None:
    """Read an abandoned wrapper's outcome so asyncio does not report it as never retrieved.

    The failure it carries is the one already on its way to the caller through the shield, and
    the work it describes has an owner of its own."""
    if not finished.cancelled():
        finished.exception()


async def _built(factory: "Callable[[], Any]", lifecycle: "_Lifecycle") -> Any:
    """Construct one env on its episode's lifecycle loop, off the caller's.

    Constructing an env is blocking work, and for some envs it is real work: provisioning a
    corpus, walking and copying two views of it, taking a file lock. Run on the shared loop it
    stops every other episode the process is serving, along with their watchdogs and deadlines,
    which is the whole reason the session hooks are off that loop too.

    **On a loop, and on the right one.** The factory runs on the lifecycle loop's own thread, so
    an env that binds ``asyncio.get_running_loop()`` binds the loop that will also close it, and
    that loop outlives every caller. There is nothing to hand back and no loop to guess at later.

    A caller that stops waiting does not stop the build, because a thread cannot be told to stop.
    What it does is leave an env with no owner, so the wait is shielded and an abandoned build is
    closed on the lifecycle loop that made it."""
    building = lifecycle.run(_build(factory))
    waiter = asyncio.wrap_future(building)
    try:
        return await asyncio.shield(waiter)
    except BaseException:
        waiter.add_done_callback(_swallow)
        _discard_when_built(building, lifecycle)
        raise


async def _build(factory: "Callable[[], Any]") -> Any:
    context = contextvars.copy_context()
    return context.run(factory)


def _discard_when_built(
    building: "concurrent.futures.Future[Any]", lifecycle: "_Lifecycle"
) -> None:
    """Close an env whose caller stopped waiting for it to be built, once it exists.

    A constructed env holds whatever its constructor made (for an out-of-process world, another
    process), and the caller that asked for it has gone. It is closed on the loop it was built on,
    which is this lifecycle's, so there is no loop to find and none to give up on. The caller's
    context is carried into the callback, because the close of a tenant-scoped env releases a
    tenant-scoped resource and the abandoned path is not where that should stop being true."""
    # Captured here, before the callback below runs: that callback fires in whichever thread
    # finished the build, after the caller's context has been left behind, and the close of a
    # tenant-scoped env releases a tenant-scoped resource.
    context = contextvars.copy_context()

    def discard(finished: "concurrent.futures.Future[Any]") -> None:
        if finished.cancelled() or finished.exception() is not None:
            lifecycle.stop(0.0)
            return
        env = finished.result()
        context.run(_close_and_stop, lifecycle, env)

    building.add_done_callback(discard)


def _discarded(
    lifecycle: "_Lifecycle",
    building: "Optional[concurrent.futures.Future[Any]]",
    context: "Optional[contextvars.Context]" = None,
) -> "concurrent.futures.Future[None]":
    """Close whatever a build hands back that nobody is going to take, and say when that is done.

    The future it returns is what a lifecycle is stopped behind: a build still running is a
    constructor still making an env, and stopping the loop under it would leave what it made with
    nobody to close it."""
    done: "concurrent.futures.Future[None]" = concurrent.futures.Future()
    if building is None:
        done.set_result(None)
        return done
    # Captured here, before the callback below runs: that callback fires in whichever thread
    # finished the build, long after the caller's `Context.run` has exited, so a close scheduled
    # from it would otherwise release a tenant-scoped resource under no tenant at all.
    closing_context = context if context is not None else contextvars.copy_context()

    def discard(finished: "concurrent.futures.Future[Any]") -> None:
        if finished.cancelled() or finished.exception() is not None:
            done.set_result(None)
            return
        cleanup = _env_close(finished.result(), lifecycle, None, closing_context)
        closing = cleanup.close_env()
        closing.add_done_callback(lambda _f: done.set_result(None))

    building.add_done_callback(discard)
    return done


def _discarded_env(
    lifecycle: "_Lifecycle",
    env: Any,
    context: "Optional[contextvars.Context]" = None,
    owner: "Optional[asyncio.AbstractEventLoop]" = None,
) -> "concurrent.futures.Future[None]":
    """Close an env that was built and then never handed to an episode, and say when that is
    done. The future is what a lifecycle is stopped behind.

    ``owner`` is the loop the env was built on, and it is not optional in the sense that matters:
    hard-coded to ``None`` this closed a default-factory env, built on its caller's loop, on the
    lifecycle loop instead, which is the one place a loop-bound env refuses."""
    cleanup = _env_close(
        env,
        lifecycle,
        owner,
        context if context is not None else contextvars.copy_context(),
    )
    return cleanup.close_env()


def _close_and_stop(lifecycle: "_Lifecycle", env: Any) -> None:
    cleanup = _env_close(env, lifecycle, None, contextvars.copy_context())
    cleanup.close_env()
    lifecycle.stop_when(cleanup.watching())


@dataclass
class CallResult:
    """The outcome of one tool call: the tool's functional ``content`` (the observation the
    harness needs), the ``meta`` sidecar (feedback + terminate flag), and whether the episode
    is now over."""

    content: str
    meta: Dict[str, Any]
    terminated: bool
    # True when this call arrived after the episode had already ended and was answered with a
    # tombstone: nothing was dispatched, nothing was sealed, and ``terminated`` reports the
    # episode's state rather than anything this call did. A caller that attributes the ending
    # to the call it made — "this request is what aborted the task" — has to tell the two
    # apart, because every call after the first terminal is answered this way.
    tombstoned: bool = False


class ServedEpisode:
    """One episode of one env, driven by external tool calls.

    Construct via :meth:`start` (it loads the task and opens the tool sessions), then call
    :meth:`call` per tool invocation until :attr:`terminated`. :meth:`describe` returns the
    task contract to hand the harness at setup.
    """

    def __init__(
        self,
        env,
        env_name: str,
        task_id: Optional[str],
        session_id: str,
        task: Dict[str, Any],
        sessions: Dict[str, MCPSession],
        opened: List[MCPSession],
        trace_path: Optional[Union[str, Path]],
        spec: TaskSpec,
        finalize_deadline: Optional[float] = None,
        lifecycle: Optional["_Lifecycle"] = None,
        cleanup: Optional["_EnvClose"] = None,
        opened_context: "Optional[contextvars.Context]" = None,
    ) -> None:
        self._env = env
        self._env_name = env_name
        self._task_id = task_id
        self._session_id = session_id
        self._task = task
        self._sessions = sessions  # advertised tool name -> session
        self._opened = opened  # every session opened, for teardown
        self._trace_path = Path(trace_path) if trace_path is not None else None
        self._trajectory: Trajectory = []
        self._step = 0
        self._terminated = False
        # The terminal step's feedback in wire form (inference + episode), retained so
        # the in-process `evaluate()` can report the score without a trace file. Same
        # list `result_from_trace` reconstructs from the terminal row.
        self._terminal_feedback: List[Dict[str, Any]] = []
        # Serialize calls: one episode is a single sequential trajectory. `call()`
        # mutates shared step/trajectory/terminated state across an `await`, so
        # concurrent MCP requests on this session must not interleave. This same lock is the
        # lifecycle-state lock: seal, evidence commit, close-race arbitration, and teardown all
        # take it.
        self._lock = asyncio.Lock()

        # The task contract, described once and snapshotted here. Everything that publishes this
        # episode's contract or enforces it reads THIS object: `describe()` hands it out, and the
        # score schemas below — what a terminal call's arguments are validated against — are the
        # schemas in it. Describing again per reader is what that replaces: `Env.describe` is
        # ordinary env code and nothing obliges two calls to agree, so a stateful env could
        # publish one contract to whoever frames the agent and enforce another when the agent
        # acts on it. The agent then sends the arguments it was told to send, the seal refuses
        # them against a schema it was never shown, and the task is recorded as a wrong answer.
        #
        # Copied rather than kept, because `describe` hands back the env's own object and an env
        # is free to hold on to it: a snapshot the publisher can still reach is not a snapshot.
        #
        # And normalized, not merely copied, because a copy of an env's object is still the env's
        # object. What a server advertises is this contract *as JSON*, and what the terminal
        # transaction refuses a call against is the schema in here — so the two have to be the
        # same value, not two renderings of one. A JSON scalar has subclasses, and the models do
        # not validate on assignment, so a `const` that is a `str` subclass whose `__eq__` answers
        # false is advertised as ordinary text and matches nothing the agent can send: the agent
        # sends exactly what it was shown, the seal refuses it, and the row says it answered
        # wrong. The round trip strips the subclass and leaves the value the wire carries.
        #
        # Normalized *first*, and the order is load-bearing. Copying is the one step here that
        # runs code the env wrote — `__deepcopy__` belongs to whatever object is being copied —
        # so a copy taken of the env's own spec lets an env decide whether an episode can be
        # opened at all, on a value the round trip was about to replace with plain data anyway.
        # Nothing contains that: it leaves `open_env` as the env's own exception, before the
        # layer above has a task to attribute it to. Normalized first, every tool the round trip
        # replaced is copied as data and that code is never reached.
        #
        # It is a narrowing and not a seal. What the round trip does not replace — a tool it
        # could not serialize, and the advisory fields it does not touch — is still the env's
        # object, and copying one still runs the env's code. The first is refused a step later
        # by a stream, which cannot compare a contract it cannot serialize; the second is not,
        # and is what a per-field detachment here would have to cover.
        self._spec = _wire_form(spec).model_copy(deep=True)

        # ----- seal-before-verdict lifecycle -----
        # The score-terminal tool name -> its advertised outer schema, read off the *published
        # contract* above (the manifest is the enforcement point, not a marker scan). At most one
        # tool is `score`; empty for every non-score env, so `call()` never leaves its legacy
        # path there.
        self._score_schemas: Dict[str, Dict[str, Any]] = {
            m.name: m.input_schema for m in self._spec.tools if m.terminal_kind == "score"
        }
        # `finalize` is the env's terminal hook (None unless the env overrides it).
        self._finalize = getattr(env, "finalize", None)
        # A published `score` terminal is a promise: this call is authoritatively sealed and
        # finalized. Enforce that promise HERE — the single boundary every served env passes
        # through — not only in the env's construction check. An env that hand-builds its
        # TaskSpec/manifest in `describe()` instead of declaring `score_terminal_tool` never runs
        # that check, and would otherwise slip a score terminal past the serve layer with no
        # callable finalize, silently leaving `_seal_enabled` False and routing its advertised,
        # authoritative scoring through the legacy marker/trajectory path — reopening the
        # grade->read->fix->grade exploit for an env that expected the seal to protect it. Refuse
        # to run, loudly, rather than downgrade.
        if self._score_schemas and not callable(self._finalize):
            name = next(iter(self._score_schemas))
            raise TypeError(
                f"env advertises a `score` terminal {name!r} but provides no callable "
                "finalize() hook; refusing to run its authoritative scoring through the legacy "
                "path (a `score` terminal must seal and finalize)"
            )
        # After the guard, a score manifest guarantees a callable finalize, so a `score`
        # terminal ALWAYS seals. An env that marks nothing `score` opts out to the unchanged
        # legacy path (`_seal_enabled` False) — the correct behaviour for a non-migrated env.
        self._seal_enabled = bool(self._score_schemas)
        self._state = LifecycleState.OPEN
        # The single in-flight finalization. Created exactly once, at seal; shielded by every
        # awaiter so a client cancellation/close can never abandon or re-dispatch it. At most
        # one per episode.
        self._finalization: Optional["asyncio.Future[CallResult]"] = None
        self._finalization_id: Optional[str] = None
        self._finalization_source: Optional[str] = None
        # The tool that entered the terminal transaction, recorded AT the seal — so a caller
        # whose terminal call was cancelled before its verdict landed can still learn how the
        # episode ended, from the episode itself.
        self._finalization_tool: Optional[str] = None
        # The committed, core-owned terminal evidence (None until FINALIZED).
        self._evidence: Optional[TerminalEvidence] = None
        # Teardown is idempotent and runs exactly once (owned by the finalizer, after
        # evidence; close() routes through the same guard). `_teardown_runs` is a test hook.
        self._torn_down = False
        self._teardown_runs = 0
        # This episode's own thread and event loop, and the ONE release future for its session.
        # The lifecycle is where the env was built (when the caller declared its factory safe
        # off their loop), where the session hooks run, and where the env is closed; `close()`
        # stops it. `_released` is issued by whichever of teardown and close reaches it first and
        # then only observed (see `_release`), and every caller that wants it joins that same
        # future rather than a flag saying somebody once waited on it.
        self._lifecycle = lifecycle if lifecycle is not None else _Lifecycle(session_id[-8:])
        self._released: Optional["concurrent.futures.Future[None]"] = None
        # The one deadline the whole teardown shares: the release and the env close behind it are
        # two halves of one operation and get one bound between them (see `_teardown_budget`).
        self._teardown_deadline: Optional[float] = None
        # The one clock a terminal transaction answers against: the evaluator and the verifier
        # share it rather than each getting the whole of it (see `_verify_budget`).
        self._answer_deadline: Optional[float] = None
        # Whether the env close and this lifecycle's shutdown have been handed over. Once, and
        # never in front of a finalization that is still writing its verdict (`_arrange_teardown`).
        self._teardown_arranged = False
        # The context this episode was opened in, kept so its session is released under it.
        self._opened_context = (
            opened_context if opened_context is not None else contextvars.copy_context()
        )
        # The one disposal of this episode's MCP sessions (see `_dispose_sessions`).
        self._disposing: Optional["asyncio.Future[None]"] = None
        # The one close of this env. Every path that wants the env closed goes through it, so
        # exactly one of them runs `Env.close` and the rest join and are told what it did.
        self._cleanup = (
            cleanup
            if cleanup is not None
            else _env_close(env, self._lifecycle, _running_loop())
        )
        # Set if a durable-record write ever failed (best-effort persistence — never fatal).
        self._persist_degraded = False
        # The in-flight evaluator task (retained so teardown drains it — see `_run_finalize`).
        self._eval_task: Optional["asyncio.Future[Any]"] = None
        # A background drain+teardown task, used only on the deadline path so the caller gets
        # the fail-closed result AT the deadline while resource cleanup waits for the evaluator.
        self._drain_task: Optional["asyncio.Future[None]"] = None
        # The one ordinary dispatch this episode owns, if there is one in flight.
        #
        # **Owned rather than awaited.** An ordinary tool is dispatched into the env, and for a
        # synchronous handler FastMCP runs that in a worker thread: cancelling the coroutine that
        # awaits it abandons the *await*, never the operation.
        #
        # So the operation is a task of this episode's, every caller waits on it *shielded*, and
        # it commits its own step when it lands whether or not anybody is still listening. This
        # is what the next ordinary call waits for (see `_settled`), which is a stronger gate
        # than the lock because a cancellation cannot drop it.
        self._inflight: Optional["asyncio.Future[tuple[CallResult, int]]"] = None
        # The tool the in-flight ordinary dispatch is running, and the one a forced terminal
        # overtook if it ever did. The second is private accounting: the overtaken call lands
        # into a sealed episode and is tombstoned out of the trajectory, so this is the only
        # place a run says there was one.
        self._inflight_tool: Optional[str] = None
        self._preempted: Optional[str] = None
        # Optional finalize deadline (seconds): the evaluator is awaited up to this bound; on
        # timeout the episode fails closed to a `finalize_error` verdict. None disables it.
        self._finalize_deadline = finalize_deadline
        # The durable finalization store — a local directory of fsync'd JSON records next to
        # the trace (zero user setup). Built lazily only for a seal-enabled env.
        self._store: Optional[FinalizationStore] = None
        if self._seal_enabled:
            self._store = FinalizationStore(
                FinalizationStore.resolve_dir(session_id, self._trace_path)
            )
            # Restart recovery, transport-independent: resolve any finalization records left
            # dangling (SEALED/PENDING) by a crashed prior run to a fail-closed verdict —
            # the evaluator is never re-invoked. Done here at construction (not only in
            # `run_stdio`) so `evaluate()` and every in-process caller get the same guarantee.
            # This episode has not sealed yet, so it owns no record here; only prior/other
            # sessions' dangling records are resolved. Best-effort: a read-only-store I/O error
            # never blocks startup.
            #
            # Once per store directory per process, not once per episode. The no-trace store is
            # one directory shared by every session ever run on the machine, so reading it is
            # O(the machine's whole history), and the answer, "what did a process that is gone
            # leave behind", is the same for every episode this process opens. Asked per episode
            # it turned a suite that opens eighty of them into minutes of reading JSON (see
            # :meth:`FinalizationStore.recover_once`).
            try:
                self._store.recover_once()
            except Exception:  # noqa: BLE001 - recovery may not decide whether an episode opens
                # Not just `OSError`. This directory is shared with every session the machine has
                # run and holds files this process did not write, so what it can raise is not a
                # list this line gets to be right about. Recovery is best-effort by design and a
                # store it could not deal with is left uncached for a later pass, which is a
                # worse recovery guarantee and not a reason to refuse to open an episode.
                pass

    @classmethod
    async def start(
        cls,
        env_name: str,
        *,
        task: Optional[Union[int, str]] = None,
        trace_path: Optional[Union[str, Path]] = None,
        env_config: Optional[Dict[str, Any]] = None,
        finalize_deadline: Optional[float] = None,
        off_loop_factory: bool = False,
    ) -> "ServedEpisode":
        """Build the env, load the task instance, open the essential MCP sessions, and push
        per-episode state into the (in-process) tool servers.

        ``off_loop_factory`` declares that this env's constructor may be run on a loop that is
        not the caller's: it binds no thread-affine resources, and any loop it binds is a loop it
        is content to be closed on. Constructing an env is blocking work and some envs make it
        real work (provisioning a corpus, walking and copying a cache, taking a file lock), so a
        cold or contended construction on the shared loop stops every other episode this process
        is serving, and their watchdogs and deadlines with them. Such an env is built on this
        episode's own lifecycle loop, which is also where it is closed, so loop affinity holds by
        construction rather than by anybody keeping track.

        Off by default: an env is allowed to bind the caller's loop in its constructor, and a
        caller that has not said this one does not is a caller whose env this may not move."""
        lifecycle = _Lifecycle("start")
        if off_loop_factory:
            # No handler here, deliberately. A cancelled `_built` has already arranged the
            # discard of an env its factory is still making, and stopping the lifecycle beside
            # that arrangement is what made it useless: marked stopped, it can no longer take the
            # discard when the build lands, so the env is built and then nothing can close it.
            # The lifecycle stops itself behind the discard, which is the same rule as everywhere
            # else here.
            env = await _built(lambda: make(env_name, config=env_config), lifecycle)
        else:
            try:
                env = make(env_name, config=env_config)
            except BaseException:
                # Built on this caller's loop and never built at all, so there is nothing for the
                # lifecycle to be holding.
                lifecycle.stop(0.0)
                raise
        return await cls.open_env(
            env,
            env_name=env_name,
            task=task,
            trace_path=trace_path,
            finalize_deadline=finalize_deadline,
            lifecycle=lifecycle,
            built_on_lifecycle=off_loop_factory,
        )

    @classmethod
    async def open_env(
        cls,
        env,
        *,
        env_name: Optional[str] = None,
        task: Optional[Union[int, str]] = None,
        trace_path: Optional[Union[str, Path]] = None,
        finalize_deadline: Optional[float] = None,
        lifecycle: Optional["_Lifecycle"] = None,
        built_on_lifecycle: bool = False,
        context: "Optional[contextvars.Context]" = None,
    ) -> "ServedEpisode":
        """Start an episode on an **already-constructed** env, which this episode then owns.

        Same contract as :meth:`start` except the caller supplies the env instance instead of
        a name, which lets a caller that serves several episodes at once give each one its own
        env. Ownership transfers: :meth:`close` closes this env, and a failure during setup
        closes it here — so the caller must hand over a *fresh* instance per episode rather
        than a shared one (``Env.close`` ends **every** session the instance tracks, which
        would tear down any sibling episode sharing the instance).
        """
        opened: List[MCPSession] = []
        # The episode's own thread and loop. Built here so the setup hook and its rollback are
        # the same loop's work, and handed to the episode below; on every path out of this method
        # it belongs to exactly one owner, the episode that was returned or the failure handler.
        # An env this caller built itself was built on *their* loop, which is where it is closed;
        # one this module built (see `_built`) was built on the lifecycle loop and is closed
        # there. Which of the two is decided here, once, and never inferred afterwards.
        lifecycle = lifecycle if lifecycle is not None else _Lifecycle("setup")
        cleanup = _env_close(
            env, lifecycle, None if built_on_lifecycle else _running_loop(), context
        )
        rollback: Optional[_SetupRollback] = None
        # The session id, once `begin_session` has returned and the env therefore holds one. It
        # is what the failure handler below needs to release, and it is a value rather than a
        # flag so that "there is a session to release" and "here is which one" are one answer.
        began: Optional[str] = None
        handed_over = False
        env_name = env_name if env_name is not None else env.name
        # Every env method, on the loop that built the env. A flagged env may have bound its
        # constructor loop, and `load_task`, `essential_specs` and `describe` are env work like
        # any other: run here they broke the same promise `finalize` was breaking.
        async def on_env(fn: "Callable[..., Any]", *called: Any) -> Any:
            if built_on_lifecycle:
                return await asyncio.wrap_future(lifecycle.call(fn, *called))
            return fn(*called)

        try:
            task_idx = int(task) if task is not None else None
            task_data = await on_env(env.load_task, task_idx)
            # Publish the *resolved* task identity so a random-default episode (task
            # omitted) is still attributable: an env that indexes tasks records the
            # chosen index in task_data (Wordle: "task_idx"), so a `shogym serve wordle_v1`
            # run traces a concrete task rather than null.
            if task is not None:
                resolved_task: Optional[str] = str(task)
            elif "task_idx" in task_data:
                resolved_task = str(task_data["task_idx"])
            else:
                resolved_task = None
            session_id = str(uuid7())

            sessions: Dict[str, MCPSession] = {}
            for spec in await on_env(env.essential_specs):
                session = await _open_session_for_spec(spec, session_id=session_id)
                opened.append(session)
                for tool_config in await session.list_tools():
                    sessions[tool_config.name] = session
            # Off the event loop. `_begin_session` is an env hook and some envs make it do real
            # work: an env whose episode is a world in another process spawns it here, and a slow
            # or wedged one would otherwise freeze every other episode this server is running,
            # along with their watchdogs and deadlines.
            #
            # A thread cannot be cancelled, so a caller that gives up on this await leaves the
            # hook running, and whatever it goes on to create (a process, a port, a directory)
            # has nobody left to release it. The wait is therefore shielded and, if it is
            # abandoned, the rollback is queued behind the hook on this same thread: one owner,
            # running whether or not the caller's task or its loop still exist.
            beginning = lifecycle.call(env.begin_session, session_id, task_data)
            waiter = asyncio.wrap_future(beginning)
            try:
                await asyncio.shield(waiter)
            except BaseException:
                waiter.add_done_callback(_swallow)
                rollback = _SetupRollback(lifecycle, env, session_id, cleanup)
                raise
            # From here the env holds a session, so every failure below owes it a release, and
            # that release goes the same way this one would have: on the lifecycle loop, never
            # on the caller's. `env.close()` is what used to do it, and `Env.close` runs the hook
            # inline, so a slow release on a `describe` that raised froze the whole server.
            began = session_id
            # The one description this episode ever asks for. Everything published about the task
            # and everything enforced on it comes off this single answer — see the snapshot the
            # constructor takes of it.
            spec = await on_env(env.describe, resolved_task)
            # Construct inside the try so the cleanup below also covers the constructor's own
            # fail-loud guard (a `score` manifest with no callable finalize): the sessions
            # opened above are released before the error propagates.
            episode = cls(
                env,
                env_name,
                resolved_task,
                session_id,
                task_data,
                sessions,
                opened,
                trace_path,
                spec=spec,
                finalize_deadline=finalize_deadline,
                lifecycle=lifecycle,
                cleanup=cleanup,
                opened_context=contextvars.copy_context(),
            )
            handed_over = True
            return episode
        except BaseException as setup_failed:
            # Setup failed, so no ServedEpisode is returned for the caller to close: release
            # everything here. Nothing this does may replace the failure being carried: a caller
            # told "close boom" instead of "setup boom" has been handed the wrong problem, so a
            # cleanup failure is attached to the setup one as a note rather than raised over it.
            # Arranged before anything is awaited, so a cancellation delivered part-way through
            # cleanup cannot leave the rest of it undone. It used to: a cancel while one session
            # close was pending re-raised at once and skipped the sessions after it, the rollback,
            # the env close and the lifecycle's own shutdown.
            if began is not None:
                rollback = rollback or _SetupRollback(lifecycle, env, began, cleanup)
            disposing = asyncio.ensure_future(_disposed(opened, setup_failed))
            disposing.add_done_callback(_swallow)
            if rollback is not None:
                # Told by the *task*, not by whoever gets to the line after the await. A caller
                # cancelled mid-disposal used to say the sessions were closed while one of them
                # was still closing, and the rollback then ran `Env.close` underneath it.
                disposing.add_done_callback(lambda _f: rollback.sessions_closed())
            try:
                await asyncio.shield(disposing)
            except BaseException as closing_failed:  # noqa: BLE001 - noted, not raised
                if isinstance(closing_failed, asyncio.CancelledError) and _caller_cancelled():
                    # Handed back, and everything after it is *chained from the disposal* rather
                    # than started here. Started here, the close ran before the sessions this
                    # setup opened had finished closing, on both branches: with a rollback,
                    # because `watching()` starts the close itself; and without one, because this
                    # was the only line that ever would. `Env.close` releases what the constructor
                    # made and those clients are still using it, so it goes behind them.
                    if not handed_over:
                        _closed_when_disposed(disposing, lifecycle, cleanup, rollback)
                    raise
                _noted(setup_failed, "an MCP session", closing_failed)
            if rollback is not None:
                # Arranged before any of the waiting below, so a cancellation anywhere in it
                # leaves the whole sequence owned: the rollback runs the release and then the
                # close, and the lifecycle stops behind that. What follows is a wait, not a
                # decision, and a wait is the one thing a cancellation may take.
                rollback.sessions_closed()
                # The sessions above are closed before the env is, because `Env.close` releases
                # what the *constructor* made and those clients may still be using it. The
                # rollback is told rather than raced: it waits for this line before it arranges
                # the env close, and goes ahead on its own only if this loop closes first, since
                # a session entered on a loop that is gone cannot be exited anywhere.
                # The rollback itself was queued on the lifecycle loop when it was made: the
                # release runs when the setup hook ahead of it finishes, and the env close after
                # that. Neither depends on this coroutine resuming, which is the difference
                # between a promise and a hope. This wait only lets it finish before the error
                # reaches the caller.
                if await rollback.settled():
                    # The release has landed, so this is the moment the env close is allowed to
                    # start, and this loop is the one that built the env. Closing it here is what
                    # keeps loop affinity on the ordinary failure path; the rollback finds it
                    # claimed and stands down.
                    try:
                        await cleanup.closed_by()
                    except BaseException as closing_failed:  # noqa: BLE001 - noted, not raised
                        # Including a `CancelledError` the env raised: contained, so it neither
                        # replaces the setup failure nor skips the lifecycle shutdown below.
                        if isinstance(closing_failed, asyncio.CancelledError) and (
                            _caller_cancelled()
                        ):
                            raise
                        _noted(setup_failed, "the env", closing_failed)
                # And if it has not landed, nothing here starts the close. `settled()` says
                # whether the release is out, and claiming the close over a release still inside
                # the hook is the overlap this whole design exists to remove. The rollback has
                # the close queued behind that release and will run it.
            else:
                # No session was ever begun, so there is nothing for the close to follow.
                try:
                    await cleanup.closed_by()
                except BaseException as closing_failed:  # noqa: BLE001 - noted, not raised
                    if isinstance(closing_failed, asyncio.CancelledError) and (
                        _caller_cancelled()
                    ):
                        raise
                    _noted(setup_failed, "the env", closing_failed)
            if not handed_over:
                # Behind the cleanup, not on top of it: a rollback whose release outran the wait
                # above is still running, and stopping the loop under it would abandon the env
                # close it is about to arrange. And behind the *lifecycle's* half of that close,
                # because a caller-owned close completes its outcome while the loop-side watcher
                # is still running.
                lifecycle.stop_when(cleanup.watching())
            raise

    async def env_closed(self, timeout: Optional[float] = None) -> None:
        """Wait for this episode's env to be **actually** closed, and say whether it is.

        :meth:`close` is bounded on purpose: a release still inside a wedged hook may not hold
        the caller past the bound it was promised, so a close arranged behind that release is
        left to finish on its own and ``close`` returns. That is right for latency and wrong for
        ownership, and they are different questions. A caller holding the slot this episode
        occupies (a stream deciding whether the next task may be dispensed) has to ask the second
        one, because an env still closing is still holding a worker, a port and a directory, and
        a slot freed on the first answer lets those accumulate past the configured capacity.

        **No bound by default, because a bound here is the wrong answer twice.** A slot freed on
        a wait that expired is a slot freed over an open env, which is the thing this exists to
        prevent; and there is nobody else the wait could be protecting, since the lifecycle owns
        the close and will finish it. A caller that passes a bound is asking for the other
        answer and gets it: the close still finishes, and it is told this call could not say so.
        A close that failed raises here, so a slot owner is not told an env is gone when it is
        not."""
        await self._cleanup.joined(timeout)

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def seal_enabled(self) -> bool:
        """True iff this episode runs the seal transaction (the env declares a score terminal
        and a ``finalize`` hook). False for a non-seal env."""
        return self._seal_enabled

    @property
    def sealed(self) -> bool:
        """True once the lifecycle has left OPEN — the ingress gate consults this to tombstone
        post-seal traffic. Always False for a non-seal episode."""
        return self._state is not LifecycleState.OPEN

    @property
    def terminal_feedback(self) -> List[Dict[str, Any]]:
        """The terminal step's feedback (wire form), or ``[]`` until the episode ends."""
        return self._terminal_feedback

    @property
    def terminal_source(self) -> Optional[str]:
        """How the terminal transaction was entered — ``explicit_tool``, ``abort`` or
        ``horizon`` — or ``None`` while the episode is still open.

        Recorded AT the seal, so it is readable even when the call that requested it was
        cancelled before its verdict landed."""
        return self._finalization_source

    @property
    def terminal_tool(self) -> Optional[str]:
        """The tool that entered the terminal transaction (``None`` for an abort, which has no
        tool, and until the episode seals). Recorded at the seal, like :attr:`terminal_source`."""
        return self._finalization_tool

    @property
    def terminal_payload(self) -> Optional[Dict[str, Any]]:
        """The public-safe terminal payload — the same sanitized verdict a terminal call
        returns — or ``None`` until the terminal transaction has committed its evidence."""
        return None if self._evidence is None else self._sanitize_terminal(self._evidence)

    @property
    def terminal_failure(self) -> Optional[Dict[str, Any]]:
        """What failed, structurally, when the terminal transaction failed closed. ``None`` when
        it did not, or when it has not committed yet.

        Read by the **harness**, and deliberately not part of :attr:`terminal_payload`: that
        payload is what a terminal call answers the agent with, and this says something about the
        env's own state. It is safe for a row and not for a reply, so it travels on its own
        channel rather than being widened into the shared one. It carries neither message text nor
        field locations either way (see ``failure_summary``): only the failure's type, a count,
        and error kinds drawn from the validator's own fixed vocabulary.
        """
        return None if self._evidence is None else self._evidence.failure

    async def wait_finalized(self) -> None:
        """Wait for an in-flight terminal transaction to commit, if there is one.

        A terminal call leaves its finalization running when its own caller is cancelled, so an
        episode can be sealed with its verdict still landing. Anyone who needs to classify the
        outcome must wait for it first — the alternative is reading a not-yet-terminated episode
        and reporting an ending that never happened. The wait is shielded: waiting for the single
        evaluation never cancels it."""
        finalization = self._finalization
        if finalization is None:
            return
        try:
            await asyncio.shield(finalization)
        except asyncio.CancelledError:
            # An env's own `CancelledError`, raised out of a hook the finalization ran, is that
            # env failing: contained like any other failure, because letting it out cancels the
            # caller that only wanted to know whether the episode had finished, and the verdict
            # is already committed. A cancellation asked for against this task is a different
            # thing wearing the same type and goes back to whoever asked.
            if _caller_cancelled():
                raise
        except Exception:  # noqa: BLE001 — a failed finalization fails closed onto the episode
            pass

    def describe(self) -> TaskSpec:
        """The contract this episode was opened on — the same snapshot for every reader, for as
        long as the episode lives.

        A copy per reader, because the snapshot is also what a terminal call's arguments are
        validated against: handed out directly, whoever framed the agent could rewrite the schema
        the seal enforces, or the tools a server had already registered, from under the episode
        that is running on them."""
        return self._spec.model_copy(deep=True)

    async def call(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        *,
        forced: bool = False,
    ) -> CallResult:
        """Execute one tool call as one step; return its result + feedback sidecar.

        For a non-seal env this is a plain step: dispatch, record, verify. For a seal-enabled
        env, a score-terminal call runs the terminal transaction (validate -> seal ->
        evaluate), and after termination (legacy terminate/horizon) or after a seal, further
        calls are tombstoned with no inward dispatch."""
        # Phase 0 (no lock): an ordinary dispatch this episode owns may still be running, left
        # by a caller that was cancelled. Wait for it to commit before deciding anything, because
        # every decision below reads `_step` and `_terminated`, which it is about to write.
        #
        # A terminal does not wait, and that asymmetry is the point. The deadline's forced
        # terminal exists precisely for the episode whose ordinary call is not coming back: made
        # to queue behind it, the one case a wall clock matters most is the one case it could
        # never act on. So a terminal seals now, and the finalization it starts is what unblocks
        # the ordinary call, by stopping the world it is waiting on.
        # Only a *seal* env has terminals that do not dispatch. On a non-seal env `terminate` is
        # an ordinary call like any other, and it waits like one.
        terminal_call = self._seal_enabled and (
            tool_name in self._score_schemas or tool_name == TERMINATE_TOOL_NAME
        )
        # A deadline has to be enforceable for every env this layer serves, not only the ones
        # that seal. On a non-seal env `terminate` is an ordinary call, so made to queue it could
        # only fire once the thing it was timing had already finished: the one episode a wall
        # clock exists for is the one it could never end. What it does *not* need is a different
        # ending. The reserved terminate server is not the env's blocked tool and answers at
        # once, so the forced call only has to skip the queue, not the dispatch.
        overtakes = forced and (terminal_call or tool_name == TERMINATE_TOOL_NAME)
        if not overtakes:
            await self._settled()
        elif self._inflight is not None and not self._inflight.done():
            # Overtaking one, and saying which. The call it passes will land into a sealed
            # episode and be tombstoned out of the trajectory, so a row that carries no trace of
            # it should at least carry, privately, the fact that there was one.
            self._preempted = self._inflight_tool

        dispatch: Optional["asyncio.Future[tuple[CallResult, int]]"] = None
        is_horizon = False
        finalization: Optional["asyncio.Future[CallResult]"] = None
        # Phase 1: decide and, for a terminal call, seal + spawn the finalization — all under
        # the lock so state transitions are atomic against concurrent ingress.
        async with self._lock:
            # Post-seal tombstone: once the lifecycle has left OPEN, no `tools/call` reaches
            # task/env state. (Read-only `describe`/`tools/list`/resource reads bypass
            # `call()` entirely, so they stay readable — see the ingress gate in server.py.)
            if self._state is not LifecycleState.OPEN:
                return self._sealed_tombstone()
            # Legacy terminate/horizon tombstone. Only reachable for a non-seal env (a seal env
            # leaves OPEN via a seal above).
            if self._terminated:
                return CallResult(
                    content="<episode already terminated>",
                    meta=build_meta(terminate=True),
                    terminated=True,
                    tombstoned=True,
                )

            args = dict(arguments or {})
            # `_session_id` is a reserved hidden field the transport injects with the real id.
            # Strip any caller-supplied value before *both* dispatch and Step construction, so
            # a forged id can't run against the real session nor land in the trajectory.
            args.pop("_session_id", None)

            if not self._seal_enabled:
                if overtakes and self._inflight is not None and not self._inflight.done():
                    # The deadline, over a call this episode has already accepted. Dispatching a
                    # second one here put two `_legacy_step` coroutines on the same next index:
                    # the forced one ended the episode, then the old one landed, appended after
                    # `end_session`, and set `_terminated` back to False over a row the stream
                    # had already published. So this ends the episode against the operation that
                    # is running rather than starting another beside it, and the operation is
                    # tombstoned when it lands.
                    return await self._forced_legacy_ending(tool_name)
                # Non-seal env: the single-step path. Owned and awaited outside the lock for the
                # same reasons the seal-enabled one is: a cancelled caller would otherwise
                # abandon an operation that is already in the env.
                dispatch = self._begin_dispatch(tool_name, args, legacy=True)
            # ----- seal-enabled env -----
            elif tool_name in self._score_schemas:
                # validate -> seal (evaluate happens after the lock is released)
                invalid = self._validate_terminal_args(tool_name, args)
                if invalid is not None:
                    # A malformed terminal request is a NORMAL validation error while the
                    # episode is still OPEN — no seal, no verdict, no evidence. The harness
                    # may correct and re-submit.
                    return invalid
                finalization = self._begin_finalization("explicit_tool", tool_name, args)
            elif tool_name == TERMINATE_TOOL_NAME:
                # An explicit abort on a scoring env: seal with a no-score abort evidence.
                finalization = self._begin_finalization("abort", None, None)
            else:
                # An ordinary tool in a scoring env. If this call reaches the horizon it IS the
                # terminal step: dispatch it, defer its trace row, and let the horizon
                # finalization write that same step as the single terminal row (no phantom
                # ``<horizon>`` step). Otherwise it's a normal mid-episode step.
                horizon = self._env.horizon
                is_horizon = horizon is not None and (self._step + 1) >= horizon
                dispatch = self._begin_dispatch(
                    tool_name, args, write_trace=not is_horizon, seals_on_horizon=is_horizon
                )

        # Phase 2 (lock released). The lock is deliberately not held across an ordinary
        # dispatch: a blocked handler would otherwise hold the whole episode shut, including
        # against the deadline's forced terminal, which is the one caller that has to be able to
        # end an episode whose ordinary call has stopped answering. What keeps a *second
        # ordinary* call out meanwhile is `_settled` above, which a cancellation cannot drop.
        if dispatch is not None:
            result, _ = await self._settle(dispatch)
            if not is_horizon:
                return result
            # The horizon finalization was begun by the dispatch itself, as part of committing
            # the step that reached the budget (see `_begin_dispatch`). Deciding it here, in the
            # caller's continuation, meant a caller cancelled while the horizon-reaching tool was
            # blocked left the step committed and nobody to seal it: the episode stayed open past
            # its budget, accepted another call, and closed as an abort rather than a horizon.
            if result.tombstoned:
                # Sealed under this call while it ran: only the deadline's forced terminal does
                # that, and the dispatch said so. Reading `self._finalization` instead could not
                # tell this apart, because the forced seal is exactly what populated it, so the
                # overtaken caller was handed the deadline's own terminal result: two callers
                # with the same terminal content, one of which ended nothing, and a stream that
                # takes the second for an agent seal.
                return result
            finalization = self._finalization
            if finalization is None:
                return self._sealed_tombstone()

        # Await the single in-flight finalization, *shielded* so a cancellation/disconnect of
        # THIS request never cancels the evaluator or re-dispatches it. If we are cancelled the
        # finalization keeps running to completion in the background; a later close() awaits the
        # same future. Exactly one evaluation.
        assert finalization is not None
        await asyncio.shield(finalization)
        return finalization.result()

    # ----- legacy (non-seal) step: dispatch, record, verify -----

    async def _legacy_step(
        self, tool_name: str, args: Dict[str, Any]
    ) -> "tuple[CallResult, int]":
        # Prospective step: don't advance `self._step` until the call actually completes. If
        # `call_tool` raises, the counter stays put so the next call reuses this number, and
        # the trajectory stays contiguous, one Step per completed call. A *cancelled* caller does
        # not reach this: the operation is the episode's and commits when it lands (see
        # `_begin_dispatch`), so what the next call follows is a call that really ran.
        step = self._step + 1
        session = self._sessions.get(tool_name)
        if session is None:
            content = f"<unknown tool {tool_name!r}>"
        else:
            result = await session.call_tool(tool_name, args, tool_call_id=f"call-{step}")
            content = result.result
        # Ended while this was in the env: only the deadline's forced terminal does that, and the
        # row it published is the episode's outcome. Committing now would append after
        # `end_session`, take an index the terminal already accounted for, and set `_terminated`
        # back to False over a result the stream has already filed. Checked before anything is
        # written rather than after, because the write is the damage.
        if self._terminated:
            return (
                CallResult(
                    content="<episode ended while this call was running>",
                    meta=build_meta(terminate=True),
                    terminated=True,
                    tombstoned=True,
                ),
                self._step,
            )
        # The await completed; commit the step atomically with its Step. Everything from here
        # on is synchronous, so no cancellation point can split them.
        self._step = step
        self._trajectory.append(
            Step(index=step, tool=tool_name, arguments=args, result=content)
        )

        horizon = self._env.horizon
        terminated = tool_name == TERMINATE_TOOL_NAME or (
            horizon is not None and step >= horizon
        )
        self._terminated = terminated

        feedback = await self._env_verify(terminated=terminated)
        items = [*feedback.inference, *feedback.episode]

        if terminated:
            # Retain the terminal feedback so the no-trace `evaluate()` path can report the
            # score directly off the episode (not only via the trace).
            self._terminal_feedback = [dump_item(item) for item in items]

        if self._trace_path is not None:
            append_trace(
                self._trace_path,
                step_record(
                    session_id=self._session_id,
                    env_name=self._env_name,
                    task_id=self._task_id,
                    step=step,
                    tool=tool_name,
                    feedback=items,  # trace records everything, in or out of band
                    terminated=terminated,
                ),
            )

        # Eval-safe default: dense inference feedback is recorded (above) but not surfaced
        # in-band. v0 exposes no per-tool opt-in, so surface_inference stays False; episode
        # feedback rides out only on the terminal result.
        inband = select_inband(items, terminal=terminated, surface_inference=False)
        return (
            CallResult(
                content=content,
                meta=build_meta(inband, terminate=terminated),
                terminated=terminated,
            ),
            step,
        )

    # ----- seal-enabled ordinary step (mid-episode, non-terminal) -----

    async def _settled(self) -> None:
        """Wait until no ordinary dispatch this episode owns is still running.

        The ingress gate for an ordinary call, and it is the lock's job done properly: an
        ``asyncio.Lock`` is released as a cancellation unwinds, so a caller that goes away leaves
        the episode open to the next call while its own operation is still in the env. This is a
        future rather than a lock, and a future a cancelled caller cannot drop.

        Shielded, and it never re-raises a *previous* operation's failure: what a waiter needs
        from the one ahead of it is that it is over, not that it succeeded. It loops rather than
        returning on one, because between that failure and this line another call may have
        installed itself, and a waiter that returns without looking walks straight past it.

        **This waiter's own cancellation is not the operation ahead failing.** Returning on it
        let the same coroutine carry on into `_begin_dispatch` and replace `_inflight` while the
        first operation was still inside the env: two ordinary calls in one episode at once, the
        second committing its step over the first. A cancellation asked for against this task
        goes back to its caller."""
        while True:
            inflight = self._inflight
            if inflight is None or inflight.done():
                return
            try:
                await asyncio.shield(inflight)
            except asyncio.CancelledError:
                if _caller_cancelled():
                    raise
            except Exception:  # noqa: BLE001 (a previous caller's failure is not this one's)
                pass

    async def _sealing_on_horizon(
        self, runner: "Any", tool_name: str
    ) -> "tuple[CallResult, int]":
        """Run the dispatch and, in the same operation, seal the episode it just finished.

        The horizon terminal is not a second decision made about a committed step; it is part of
        committing it. Left to the caller, a cancellation between the two produced an episode
        whose budget was spent, whose step was in the trajectory, and which nothing had ended:
        the next call was accepted over it, and a close recorded an abort where the horizon
        outcome belonged."""
        outcome = await runner
        async with self._lock:
            # The world may have been sealed under this call while it ran: only the deadline's
            # forced terminal does that, and then there is no horizon terminal to begin.
            if self._state is LifecycleState.OPEN and not self._terminated:
                # Horizon has no submission: finalize with source=horizon and no args, but keep
                # the real tool name so the terminal trace row is labelled with the call that hit
                # the budget.
                self._begin_finalization("horizon", tool_name, None)
        return outcome

    async def _forced_legacy_ending(self, tool_name: str) -> CallResult:
        """End a non-seal episode against the ordinary call it is still running.

        **Called under the lock, and it dispatches nothing.** A non-seal env has no seal
        transaction to end an episode with, so the deadline's terminal used to be an ordinary
        call like any other; forced past the queue it became a *second* operation on one
        episode's next index. This ends the episode where it stands instead: what the trajectory
        holds is what was committed, the verifier is run over exactly that, and the call still in
        the env is tombstoned when it lands (see `_legacy_step`)."""
        self._terminated = True
        self._preempted = self._inflight_tool
        feedback = await self._env_verify(terminated=True)
        items = [*feedback.inference, *feedback.episode]
        self._terminal_feedback = [dump_item(item) for item in items]
        # **No trace row.** A step row is one dispatched call, and this dispatches nothing: written
        # at `self._step` it either duplicated the previous call's index or, when the deadline
        # overtook the first call, invented a step 0 for a `terminate` that never ran. The trace
        # ends at the last call that really happened, which is what it is for; the verdict is on
        # this result and in `_terminal_feedback`, which is where a non-seal ending has always
        # been read from.
        inband = select_inband(items, terminal=True, surface_inference=False)
        return CallResult(
            content="<episode ended by the deadline; no further tool calls are dispatched>",
            meta=build_meta(inband, terminate=True),
            terminated=True,
        )

    def _begin_dispatch(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        write_trace: bool = True,
        legacy: bool = False,
        seals_on_horizon: bool = False,
    ) -> "asyncio.Future[tuple[CallResult, int]]":
        """Start the one ordinary dispatch this episode owns. **Called under the lock.**

        ``seals_on_horizon`` makes the seal part of committing rather than something a caller
        does afterwards. The call that reaches the budget *is* the terminal step, and a caller
        that is cancelled while it is still in the env is a caller who never gets to say so."""
        runner = (
            self._legacy_step(tool_name, args)
            if legacy
            else self._dispatch_step(
                tool_name, args, terminated=False, write_trace=write_trace
            )
        )
        if seals_on_horizon:
            runner = self._sealing_on_horizon(runner, tool_name)
        dispatch: "asyncio.Future[tuple[CallResult, int]]" = asyncio.ensure_future(runner)
        # Read whatever it ends with, so an operation whose caller went away does not leave an
        # unretrieved exception for asyncio to complain about at collection. Nothing acts on it:
        # the failure belongs to the caller that is still waiting, if there is one.
        dispatch.add_done_callback(lambda done: done.cancelled() or done.exception())
        self._inflight = dispatch
        self._inflight_tool = tool_name
        return dispatch

    async def _settle(
        self, dispatch: "asyncio.Future[tuple[CallResult, int]]"
    ) -> "tuple[CallResult, int]":
        """Await the owned dispatch, shielded, and clear it once it has actually landed.

        Cleared on completion rather than in a ``finally``: a caller that is cancelled here leaves
        the operation running and must leave it *set*, because that is what holds the next call
        out until it has committed."""
        try:
            return await asyncio.shield(dispatch)
        finally:
            if dispatch.done() and self._inflight is dispatch:
                self._inflight = None

    async def _dispatch_step(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        terminated: bool,
        write_trace: bool = True,
    ) -> "tuple[CallResult, int]":
        """Dispatch one ordinary tool call in a seal-enabled env and record it as a normal
        (non-terminal) step. Termination is owned by the finalizer, so ``terminated`` is False
        here; the caller decides whether the horizon was reached. Returns the result and the
        committed step index.

        ``write_trace=False`` records the trajectory Step but **defers the trace row**: used for
        the call that *reaches the horizon*, so the horizon finalization can write that same step
        as the single terminal (``terminated=True``) row — the ordinary call that hits the budget
        IS the terminal step, never a fabricated extra one."""
        step = self._step + 1
        session = self._sessions.get(tool_name)
        if session is None:
            content = f"<unknown tool {tool_name!r}>"
        else:
            result = await session.call_tool(tool_name, args, tool_call_id=f"call-{step}")
            content = result.result
        if self._terminated or self._state is not LifecycleState.OPEN:
            # Sealed while this ran. Only the deadline's forced terminal does that, and it does it
            # on an episode whose ordinary call had already stopped answering, so the terminal
            # step is already in the trajectory and this one may not be appended after it: a
            # trajectory whose steps are out of order is worse than one missing a call the row it
            # belongs to was never scored on. The result is still returned, so a caller still
            # waiting is answered rather than left.
            return (
                CallResult(
                    content=content,
                    meta=build_meta(terminate=True),
                    terminated=True,
                    tombstoned=True,
                ),
                self._step,
            )
        self._step = step
        self._trajectory.append(
            Step(index=step, tool=tool_name, arguments=args, result=content)
        )
        feedback = await self._env_verify(terminated=terminated)
        if self._terminated or self._state is not LifecycleState.OPEN:
            # Sealed while this was verifying. The check before the step was not enough: the
            # verifier is awaited, and the deadline's forced terminal can seal in that gap, so
            # this caller used to come back holding the forced terminal's own result with
            # `tombstoned` false, as a second terminal caller over one episode.
            return self._sealed_tombstone(), self._step
        items = [*feedback.inference, *feedback.episode]
        if self._trace_path is not None and write_trace:
            append_trace(
                self._trace_path,
                step_record(
                    session_id=self._session_id,
                    env_name=self._env_name,
                    task_id=self._task_id,
                    step=step,
                    tool=tool_name,
                    feedback=items,
                    terminated=terminated,
                ),
            )
        inband = select_inband(items, terminal=terminated, surface_inference=False)
        return (
            CallResult(
                content=content,
                meta=build_meta(inband, terminate=terminated),
                terminated=terminated,
            ),
            step,
        )

    # ----- terminal transaction -----

    def _validate_terminal_args(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[CallResult]:
        """Validate a ``score``-terminal call's args against its advertised outer schema.

        Returns a normal validation-error :class:`CallResult` (episode stays OPEN, not sealed,
        no verdict) when the request is malformed, else ``None``. Validates the **complete**
        advertised JSON Schema BEFORE any lifecycle mutation, so a request FastMCP would reject
        downstream (a wrong type, an unknown extra field, a missing required key) is a normal
        error the harness can correct and re-submit, never a sealed finalizer that irrevocably
        scores 0. On top of the schema, a required string must be non-blank (schema
        ``type: string`` accepts ``""``).

        **Only the caller's request is answered here.** The schema is the env's own object and
        validating against it runs the env's code — a key checked against the instance, a key
        formatted into a message — so this can fail for reasons the caller could never fix. Those
        are deliberately *not* turned into a validation error the caller is invited to retry:
        nothing it sends will satisfy a contract that cannot be read, and answering as if it
        might leaves the harness above composing an outcome for a task nobody could finish. They
        propagate, and the layer that owns the task's record classifies it (see
        :meth:`shogym.serve.stream.TaskStream.dispatch`). What does not propagate is the *name* of
        an argument this refusal is about — see :func:`_named`."""
        schema = self._score_schemas.get(tool_name, {})
        try:
            jsonschema.validate(instance=args, schema=schema)
        except jsonschema.ValidationError as exc:
            # `exc.message` describes only the caller's own (already public) input — no gold
            # answer, no verdict, no oracle.
            return self._validation_error(f"invalid arguments: {exc.message}")
        except jsonschema.SchemaError:  # a broken advertised schema: fail closed, don't seal
            return self._validation_error("tool schema is invalid")

        required = schema.get("required", []) if isinstance(schema, dict) else []
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        for key in required:
            if key == "_session_id":  # transport-injected; never client-supplied
                continue
            if props.get(key, {}).get("type") == "string":
                value = args.get(key)
                if not isinstance(value, str) or not value.strip():
                    return self._validation_error(
                        f"argument {_named(key)} must be a non-blank string"
                    )
        return None

    def _validation_error(self, message: str) -> CallResult:
        """A terminal-arg validation failure: a plain error result while still OPEN. Carries no
        seal, no terminate flag, and does not advance the step — rejected at ingress before any
        dispatch."""
        return CallResult(
            content=json.dumps({"error": message, "validation_error": True}),
            meta=build_meta(),
            terminated=False,
        )

    def _begin_finalization(
        self, source: str, tool_name: Optional[str], args: Optional[Dict[str, Any]]
    ) -> "asyncio.Future[CallResult]":
        """Atomically seal the episode (OPEN -> SEALED -> FINALIZING), persist the durable
        ``SEALED`` record, and spawn the single finalization task. Called under the lock."""
        # One clock from here: the evaluator and the verifier both answer inside it.
        self._start_answer_clock()
        finalization_id = str(uuid7())
        self._finalization_id = finalization_id
        self._finalization_source = source
        self._finalization_tool = tool_name
        self._state = LifecycleState.SEALED
        self._write_record("SEALED", source, args_digest(args))
        finalization: "asyncio.Future[CallResult]" = asyncio.ensure_future(
            self._run_finalize(source, tool_name, args, finalization_id)
        )
        self._finalization = finalization
        self._state = LifecycleState.FINALIZING
        return finalization

    async def _run_finalize(
        self,
        source: str,
        tool_name: Optional[str],
        args: Optional[Dict[str, Any]],
        finalization_id: str,
    ) -> CallResult:
        """The single finalization: run the evaluator (``finalize``) on the *already-sealed*
        episode, stamp non-forgeable provenance onto the evidence, persist it durably, append
        the versioned ``terminal`` trace event, score via ``verify(evidence)``, tear down, and
        return the sanitized public payload.

        Runs as its own task (created at seal), shielded by every awaiter, so a cancellation of
        the awaiting request or a racing close() cannot abandon it or spawn a second evaluation.
        Fail-closed: an evaluator timeout/crash yields a ``finalize_error`` verdict
        (``correct=False``) rather than propagating, with only a private diagnostic (never
        exception text to the agent)."""
        self._write_record("PENDING", source, args_digest(args))
        confidence = args.get("confidence") if isinstance(args, dict) else None
        try:
            if source == "abort":
                # An abort is a no-score path: core synthesizes the evidence directly (no
                # evaluator is invoked at teardown).
                evidence = TerminalEvidence(
                    source="abort",
                    status="ok",
                    verdict={"correct": False, "aborted": True},
                )
            else:
                req = FinalizeRequest(
                    source=source,  # type: ignore[arg-type]
                    finalization_id=finalization_id,
                    session_id=self._session_id,
                    args=args,
                    deadline=self._finalize_deadline,
                    tool_name=tool_name,
                )
                # Run the evaluator as its own retained task so a deadline timeout can fail the
                # verdict closed WITHOUT abandoning the evaluator: `_teardown` drains this task
                # before releasing env/session resources, so a late evaluator can never touch
                # torn-down state or run a second time.
                eval_task: "asyncio.Future[Any]" = self._evaluated(req)
                self._eval_task = eval_task
                if self._finalize_deadline is not None:
                    # shield inside wait_for: the deadline must not cancel the evaluator
                    # mid-flight (that could wedge the in-process client); it only bounds how
                    # long we *await* the verdict before failing closed. The evaluator keeps
                    # running and is drained by `_teardown`.
                    evidence = await asyncio.wait_for(
                        asyncio.shield(eval_task), timeout=self._verify_budget()
                    )
                else:
                    evidence = await eval_task
                if not isinstance(evidence, TerminalEvidence):
                    raise TypeError(
                        f"finalize must return TerminalEvidence, got {type(evidence)!r}"
                    )
        except (asyncio.CancelledError, asyncio.TimeoutError, Exception) as exc:
            # Fail-closed on ANY evaluator failure — including cancellation. A cancellation here
            # can only come from the *evaluator's own* awaited work being cancelled: the
            # finalization task is shielded from every caller (`call`/`close`), so an awaiter's
            # cancellation never propagates into `_run_finalize`. Swallowing it and committing a
            # fail-closed verdict is therefore correct (the episode never strands FINALIZING with
            # a PENDING record) and preserves the caller-cancellation rule, which lives at the
            # shielded await in `call()`/`close()`, not here.
            evidence = TerminalEvidence(
                source=source,  # type: ignore[arg-type]
                status="finalize_error",
                verdict=fail_closed_verdict(confidence),
                diagnostic=f"finalize failed: {type(exc).__name__}: {exc}",
                # The same failure, structurally, for the harness-side row. The diagnostic above
                # stays private because it renders the failure's values; the summary carries only
                # a type, a count and a fixed vocabulary, which is what a reader of an unscored
                # row needs and all they may safely be told.
                failure=failure_summary(exc),
            )

        # A verdict that isn't a JSON-object dict — a non-dict (e.g. a list, which
        # `_sanitize_terminal`'s `dict(evidence.verdict)` would reject) or one that isn't
        # JSON-serializable with allow_nan=False (a NaN/Inf, a non-JSON value) — would raise
        # mid-commit: `_sanitize_terminal` at the terminal-step append (and the trace/store
        # writes) run AFTER the episode reached FINALIZING with a PENDING record, so the raise
        # would strand that record at PENDING and surface an exception to the client instead of
        # the documented fail-closed result. Catch it here, before any commit, and fail closed to
        # the canonical safe verdict so the terminal transaction always completes FINALIZED
        # (fail-closed) and returns the safe result to the caller.
        if not isinstance(evidence.verdict, dict) or not _json_safe(evidence.verdict):
            evidence = TerminalEvidence(
                source=source,  # type: ignore[arg-type]
                status="finalize_error",
                verdict=fail_closed_verdict(confidence),
                diagnostic="finalize returned a non-dict or non-serializable verdict",
            )

        # Core stamps the non-forgeable fields — source, finalization_id, provenance — so a
        # verdict is only ever trusted with core-owned provenance the harness cannot supply.
        evidence.source = source  # type: ignore[assignment]
        evidence.finalization_id = finalization_id
        evidence.provenance = {
            "core": "shogym-serve",
            "session_id": self._session_id,
            "finalization_id": finalization_id,
            "sealed_source": source,
        }
        if evidence.args is None and source == "explicit_tool":
            evidence.args = dict(args or {})

        # Commit evidence atomically under the lifecycle lock (FINALIZING -> FINALIZED). The
        # commit is wrapped so an unexpected failure in verify/trace still fails closed and,
        # via the `finally`, ALWAYS reaches teardown — a committed episode can never leak its
        # env/session state or return an exception instead of a terminal result.
        items: List[Any] = []
        try:
            async with self._lock:
                self._evidence = evidence
                self._terminated = True

                # Append the terminal step to the trajectory BEFORE calling verify, so a
                # migrated verifier sees the COMPLETE call history — including the terminal
                # action it is scoring (turn counts, a terminal inference signal, ...). The
                # horizon path already recorded its budget-reaching call as this step;
                # explicit_tool/abort append it now.
                if source == "horizon":
                    # The ordinary call that hit the budget was already appended to the
                    # trajectory (with its trace row deferred). Reuse it as the terminal step —
                    # never fabricate a phantom step or a tool invocation that didn't happen.
                    step = self._step
                    terminal_tool = tool_name or "<horizon>"
                else:
                    # explicit_tool / abort: the terminal action (submit / terminate) is a real
                    # call with no prior step — append it as the terminal step now.
                    step = self._step + 1
                    self._step = step
                    terminal_tool = tool_name or "terminate"
                    self._trajectory.append(
                        Step(
                            index=step,
                            tool=terminal_tool,
                            arguments=dict(args or {}),
                            result=json.dumps(self._sanitize_terminal(evidence)),
                        )
                    )

                # A verifier bug must not strand the episode: fail closed on a verify() raise.
                #
                # **Bounded, and for the same reason the evaluator is.** `verify` runs on the loop
                # that owns the env, and on the deadline path that loop is still holding the
                # evaluator this transaction just timed out: awaiting it there put the fail-closed
                # answer *behind* the thing the deadline exists to stop waiting for, so the
                # promised bound was not a bound at all. What is left of the budget is what it
                # gets, and a verifier that cannot run inside it fails closed like one that
                # raised.
                try:
                    feedback = await asyncio.wait_for(
                        self._env_verify(terminated=True, evidence=evidence),
                        timeout=self._scoring_budget(),
                    )
                    items = [*feedback.inference, *feedback.episode]
                except BaseException as exc:  # noqa: BLE001 (verifier failure => fail closed)
                    # `BaseException`, so a `CancelledError` the verifier itself raised is
                    # contained like any other verifier failure. Let out, it cancelled this
                    # transaction after the in-memory evidence was set and before the durable
                    # record was replaced: an episode neither finalized nor recoverable. A
                    # cancellation asked for against this task is a different thing and goes back.
                    if isinstance(exc, asyncio.CancelledError) and _caller_cancelled():
                        raise
                    evidence = TerminalEvidence(
                        source=source,  # type: ignore[arg-type]
                        status="finalize_error",
                        verdict=fail_closed_verdict(confidence),
                        provenance=evidence.provenance,
                        finalization_id=finalization_id,
                        diagnostic="verify() raised while scoring the terminal evidence",
                        # This boundary fails closed like the evaluator's, so it owes the row the
                        # same account of what happened. A verifier defect and an evaluator defect
                        # are different repairs, and a row that names neither makes them look alike.
                        failure=failure_summary(exc),
                    )
                    self._evidence = evidence
                    items = []
                self._terminal_feedback = [dump_item(item) for item in items]
                # `verify` is awaited now, so this transaction yields between appending the step
                # and committing it, and the deadline's forced terminal can seal in that gap.
                # Rechecked rather than assumed: two callers came back with the same abort payload
                # and neither tombstoned, over a trajectory that had grown a second terminal.
                if self._finalization_id is not None and finalization_id != self._finalization_id:
                    return self._sealed_tombstone()

                # Durable state leads the public trace. Persist the terminal record FIRST —
                # FINALIZED (ok) or FAILED (fail-closed) — so a crash can never leave a public
                # trace that says `ok` while the recoverable record still says PENDING. If we
                # crash after this fsync but before the trace event, recovery replays this
                # FINALIZED/FAILED evidence and the trace merely lacks its (derivable) terminal
                # event — never a contradiction. Confidential diagnostic + provenance live only
                # here, never in the user-readable trace.
                self._write_record(
                    "FAILED" if evidence.finalize_error else "FINALIZED",
                    source,
                    args_digest(args),
                    verdict=evidence.verdict,
                    provenance=evidence.provenance,
                    diagnostic=evidence.diagnostic,
                )
                self._state = LifecycleState.FINALIZED
                # Write the terminal step row (keeps evaluate()/result_from_trace working) then
                # the versioned terminal event after it (the terminal event is the last row).
                # Best-effort, like the durable record: a trace-store I/O failure is flagged
                # degraded, never allowed to strand a FINALIZED episode.
                if self._trace_path is not None:
                    try:
                        append_trace(
                            self._trace_path,
                            step_record(
                                session_id=self._session_id,
                                env_name=self._env_name,
                                task_id=self._task_id,
                                step=step,
                                tool=terminal_tool,
                                feedback=items,
                                terminated=True,
                            ),
                        )
                        append_terminal_event(
                            self._trace_path,
                            terminal_event_record(
                                session_id=self._session_id,
                                env_name=self._env_name,
                                task_id=self._task_id,
                                step=step,
                                source=source,
                                status=evidence.status,
                                verdict=evidence.verdict,
                                finalization_id=finalization_id,
                                args_digest=args_digest(args),
                            ),
                        )
                    except Exception:  # noqa: BLE001 — trace is best-effort; never strand
                        self._persist_degraded = True
        finally:
            # Teardown ALWAYS runs after the commit — even if the commit path failed unexpectedly.
            # If the evaluator is STILL running (the deadline fired and we failed closed while it
            # keeps going, shielded), defer the drain+teardown to the background so the caller
            # receives the fail-closed result AT the deadline — resources are released only once
            # the evaluator actually finishes (`_teardown` drains it first, no use-after-free). On
            # the normal path the evaluator is already done, so teardown runs inline.
            if self._eval_task is not None and not self._eval_task.done():
                self._drain_task = asyncio.ensure_future(self._teardown())
            else:
                await self._teardown()

        public = self._sanitize_terminal(evidence)
        inband = select_inband(items, terminal=True, surface_inference=False)
        return CallResult(
            content=json.dumps(public),
            meta=build_meta(inband, terminate=True),
            terminated=True,
        )

    def _write_record(
        self,
        status: str,
        source: str,
        digest: Optional[str],
        *,
        verdict: Optional[Dict[str, Any]] = None,
        provenance: Optional[Dict[str, Any]] = None,
        diagnostic: Optional[str] = None,
    ) -> None:
        """Persist one durable record — **best-effort**. A local-file I/O failure (ENOSPC,
        permissions) must never strand the lifecycle: the seal, the finalization, and the
        in-memory verdict proceed regardless, so a sealed episode always yields an outcome. A
        persistence failure only degrades crash-recovery for *this* record (a normal crash still
        works, since the store is normally writable); it is flagged for audit, never raised."""
        if self._store is None or self._finalization_id is None:
            return
        try:
            self._store.write(
                FinalizationRecord(
                    session_id=self._session_id,
                    finalization_id=self._finalization_id,
                    status=status,  # type: ignore[arg-type]
                    source=source,  # type: ignore[arg-type]
                    args_digest=digest,
                    verdict=verdict,
                    provenance=provenance,
                    diagnostic=_with_preemption(diagnostic, self._preempted),
                    owner_pid=os.getpid(),
                )
            )
        except Exception:  # noqa: BLE001 — durability is best-effort; never strand the seal
            self._persist_degraded = True

    def _sanitize_terminal(self, evidence: TerminalEvidence) -> Dict[str, Any]:
        """The public-safe terminal payload: the core-stamped verdict + a ``finalize_error``
        flag. The verdict an env returns is already public-safe (the RFC forbids returning
        judge reasoning / extracted answers / exception text); the private diagnostic and
        provenance are dropped here and live only in the durable store."""
        public = dict(evidence.verdict)
        public["finalize_error"] = evidence.finalize_error
        return public

    def _sealed_tombstone(self) -> CallResult:
        """The generic post-seal tombstone: no inward dispatch, no verdict re-exposed. Used for
        every `tools/call` (repeat terminal, unknown tool) after the seal."""
        return CallResult(
            content="<episode sealed; no further tool calls are dispatched>",
            meta=build_meta(terminate=True),
            terminated=True,
            tombstoned=True,
        )

    async def _teardown(self) -> None:
        """Drop the env's per-session state. Idempotent, runs exactly once. Owned by the
        finalizer (after evidence); close() routes through it too. The MCP client sessions are
        disposed by :meth:`close` in its own task (anyio requires the client to be exited in
        the task that entered it), so this touches only pure-Python env state and is safe to
        run from the finalization background task."""
        # Drain any still-running evaluator BEFORE dropping env state. On the deadline path the
        # verdict was already committed fail-closed while the (shielded) evaluator kept running;
        # it must finish before `end_session` clears the per-session state it may still touch, so
        # a timed-out evaluator can never use-after-free or run a second time.
        eval_task = self._eval_task
        if eval_task is not None and not eval_task.done():
            try:
                await asyncio.shield(eval_task)
            except BaseException:  # incl. the evaluator's own CancelledError — never block teardown
                pass
        async with self._lock:
            if self._torn_down:
                return
            self._torn_down = True
            self._teardown_runs += 1
            self._state = LifecycleState.TEARING_DOWN
        await self._release_session()
        self._state = LifecycleState.CLOSED

    def _release(self) -> "concurrent.futures.Future[None]":
        """This session's **one** release, issued here and never issued again.

        The session is claimed on the caller's loop and the hook runs on the lifecycle's.
        Claiming first is what makes the later ``env.close()`` in :meth:`close` safe: the id is
        gone from the env the instant
        this returns, so a close that arrives while the hook is still running finds nothing of
        this session left and cannot enter the hook a second time. That is the whole failure a
        bounded teardown used to create: abandoning the wait left the first hook running, and
        `close` went straight into the second.

        A session somebody else has already claimed (a rollback that ran, an earlier close) gets
        an already-finished future, because the release it names has an owner."""
        if self._released is None:
            release = self._env.claim_session(self._session_id)
            if release is None:
                self._released = _settled()
            else:
                try:
                    self._released = self._lifecycle.call_in(self._opened_context, release)
                except RuntimeError:
                    # The lifecycle has stopped (a second close), so this caller runs the release.
                    release()
                    self._released = _settled()
        return self._released

    def _evaluated(self, req: "FinalizeRequest") -> "asyncio.Future[Any]":
        """Start the env's ``finalize`` where the env's other work runs, and hand back a future.

        An env built on this episode's lifecycle loop was told that loop is where its work
        happens, and it is allowed to have bound it in its constructor. ``finalize`` is env work
        like any other, so running it on the serving caller's loop breaks exactly the promise
        that made building it off that loop safe, and the affinity error it raises comes back as
        a fail-closed verdict: a valid episode scored zero because the serve layer used the wrong
        loop. An env the caller built keeps the caller's loop here, as it does everywhere else.

        The future is this loop's either way, so the deadline, the shield and the drain around it
        are unchanged."""
        finalize = self._finalize
        assert finalize is not None
        if self._cleanup.owned_by_caller:
            return asyncio.ensure_future(finalize(req))  # type: ignore[misc]
        return asyncio.wrap_future(self._lifecycle.run(finalize(req)))  # type: ignore[misc]

    async def _env_verify(self, **kwargs: Any) -> Any:
        """The env's own `verify`, on the loop that owns the env."""
        return await self._env_call(
            functools.partial(self._env.verify, self._trajectory, self._task, **kwargs)
        )

    def _scoring_budget(self) -> float:
        """What the verifier gets, which is not always what is left of the answer.

        While the answer is still owed, the verifier is inside it and gets what remains of it.
        Once the deadline has already answered fail-closed, the caller has its result and what
        the verifier is still doing is labelling the evidence for the record: that belongs to the
        teardown's bound, not to a promise that has already been kept."""
        remaining = self._verify_budget()
        if remaining is None:
            return self._teardown_budget()
        return remaining if remaining > 0 else self._teardown_budget()

    def _start_answer_clock(self) -> None:
        """Start the one clock the whole terminal transaction shares."""
        if self._finalize_deadline is not None and self._answer_deadline is None:
            self._answer_deadline = time.monotonic() + self._finalize_deadline

    async def _env_call(self, fn: "Callable[..., Any]", *args: Any) -> Any:
        """One synchronous env method, on the loop that owns the env.

        A flagged env was told its work happens on its episode's lifecycle loop, and it is
        allowed to have bound that loop in its constructor. `load_task`, `describe`, `verify` and
        the rest are env work like any other, and running them on the serving caller's loop broke
        the same promise `finalize` was breaking: a fixture that checked its loop in `_load_task`
        failed on the first dispense. An env the caller built keeps the caller's loop here, where
        this is a direct call and costs nothing."""
        if self._cleanup.owned_by_caller:
            # Off the serving loop even here. These are the env's own synchronous hooks, and one
            # that blocks blocks everything this process is serving; run inline, a `wait_for`
            # around it could not preempt it either, so a bound over it was not a bound. The
            # context travels, as it does for every other hook.
            context = contextvars.copy_context()
            return await asyncio.to_thread(context.run, fn, *args)
        return await asyncio.wrap_future(self._lifecycle.call(fn, *args))

    def _verify_budget(self) -> Optional[float]:
        """What is left of the caller's answer deadline, not another whole one.

        The deadline is a promise about when an answer arrives, and the evaluator and the verifier
        are both inside it. Given the full value each, the promise was worth twice what it said,
        and a terminal call took the evaluator's share plus the verifier's on top."""
        if self._finalize_deadline is None:
            return None
        if self._answer_deadline is None:
            return self._finalize_deadline
        return max(0.0, self._answer_deadline - time.monotonic())

    def _teardown_budget(self) -> float:
        """What is left of the one bound this episode's teardown gets, in seconds.

        One bound, not one per step. The release and the env close behind it are two halves of
        the same operation, so waiting ``_END_SESSION_SECONDS`` for each turns a stated sixty
        seconds into a hundred and twenty for exactly the env that needs it least: the one whose
        release never comes back."""
        if self._teardown_deadline is None:
            self._teardown_deadline = time.monotonic() + _END_SESSION_SECONDS
        return max(0.0, self._teardown_deadline - time.monotonic())

    async def _release_session(self) -> None:
        """Issue the session's one release and wait for it, within the teardown budget.

        Off the caller's loop and bounded, for the reason `begin_session` is: this runs on the
        shared loop and an env that waits on a wedged child would hold every other episode with
        it. At the bound the wait is abandoned, never the release: an env whose own hook carries
        no timeout leaves this the only bound there is, as frontier_bench's release does when it
        reaches Docker cleanup, so an env that needs its own release to be certain has to bound
        that hook itself.

        **What a second caller joins is the release, not the fact that somebody started a
        wait.** A flag meaning "a wait has been made" let a second close skip to the env close
        over a release still running, which is the overlap this whole design exists to prevent.
        The release future is the shared state, so a second caller waits exactly as long as the
        first one has left."""
        release = self._release()
        if release.done():
            return
        try:
            # Shielded so the timeout cancels this wait and not the release: cancelling the
            # wrapper would try to cancel the underlying job, and a job still queued behind a
            # slow hook would be cancelled outright, abandoning the release rather than the wait.
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(release)),
                timeout=self._teardown_budget(),
            )
        except asyncio.CancelledError:
            # Two unrelated things arrive here as the same exception. One is this caller being
            # cancelled, which is theirs and goes back to them. The other is the env's hook
            # raising `CancelledError` from its own thread, which is ordinary third-party code
            # failing and is contained like any other hook failure: let out, it would leave the
            # MCP sessions and the env unclosed, and answer a terminating call with a traceback
            # instead of the constant, since teardown runs inside it.
            if _caller_cancelled():
                raise
        except Exception:
            pass

    async def close(self) -> None:
        # close() participates in the lifecycle for a seal env. If a finalization is in flight,
        # WAIT for it to commit evidence + tear down before disposing anything (so the
        # evaluator's live session isn't reclaimed mid-finalize); otherwise atomically claim an
        # abort and own teardown. Then dispose the MCP client sessions and let the env drop
        # residual state. For a non-seal env none of this engages (state stays OPEN with no
        # finalization), so close() is just the plain teardown below.
        #
        # **One `finally`, from the first await.** Everything below can be cancelled: a
        # finalization that is shielded but joined, an MCP session's own close, the wait for the
        # release. A cancellation at any of them used to leave this method before it had arranged
        # the env close or the lifecycle's shutdown, so the env stayed open and its thread stayed
        # alive. What the `finally` runs is arrangement rather than waiting, so a cancelled close
        # still hands both to the lifecycle, and the lifecycle finishes them.
        try:
            # An accepted call is using the session everything below is about to release. This
            # is not the deadline's overtake: an ordinary close waits for it, within the same
            # bound the rest of teardown gets, and only then claims an abort. Released under it,
            # `_end_session` took the env's per-call state away while the call was still inside
            # the tool, and the call came back tombstoned or with a dead transport.
            inflight = self._inflight
            if inflight is not None and not inflight.done():
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(
                        asyncio.shield(inflight), timeout=self._teardown_budget()
                    )
                if not inflight.done():
                    # It has outrun the bound. From here the episode ends against it, exactly as
                    # the deadline does, and the call is tombstoned when it lands. For a non-seal
                    # env that gate is `_terminated` and nothing else sets it: without this the
                    # close tore the session down, returned, and the call then committed step 1
                    # and ran `verify` against an env that was gone.
                    async with self._lock:
                        self._preempted = self._inflight_tool
                        if not self._seal_enabled:
                            self._terminated = True
            finalization: Optional["asyncio.Future[CallResult]"] = None
            if self._seal_enabled:
                async with self._lock:
                    finalization = self._finalization
                    if finalization is None and self._state is LifecycleState.OPEN:
                        # No seal happened (a score env closed before submitting): claim an abort
                        # finalization so this close owns teardown + records a no-score verdict.
                        finalization = self._begin_finalization("abort", None, None)
                if finalization is not None:
                    # Shielded: close() must not cancel the single in-flight finalization.
                    try:
                        await asyncio.shield(finalization)
                    except Exception:
                        pass
                # A deadline-path finalization returns before its background drain+teardown
                # finishes; wait for it so the evaluator has drained and env state is released
                # before we dispose the MCP sessions below.
                if self._drain_task is not None:
                    try:
                        await asyncio.shield(self._drain_task)
                    except Exception:
                        pass
                await self._teardown()  # idempotent: a no-op if the finalizer already ran it

            # Close every MCP session opened for this episode, then let the env tear down its own
            # per-session state. Disposal is one retained operation rather than a loop this
            # caller owns: cancelled halfway it used to leave the rest open, and a session marks
            # itself closed before it awaits the transport, so a later close is not a retry.
            # Shielded, so this caller's cancellation abandons the wait and not the work.
            disposing = self._dispose_sessions()
            if disposing is not None:
                try:
                    await asyncio.shield(disposing)
                except asyncio.CancelledError:
                    # Retained and shielded, so the disposal finishes; but a caller that asked to
                    # be cancelled gets its cancellation back after the arrangement below, rather
                    # than a close that returns as though nothing had been asked.
                    if _caller_cancelled():
                        raise
                except Exception:
                    pass
            # This episode's session is released here, through the one owned release, and
            # bounded. For a non-seal env this is where it happens at all, since nothing above
            # tears down. The env close below is then a close over what is left of the env, not
            # a second entry into this session's hook: the session was claimed the moment the
            # release was issued, so even a release still running in its thread leaves
            # `Env.close` with nothing of this episode's to do.
            await self._release_session()
            await self._close_env()
        finally:
            # Arrangement, not waiting, so this runs the same way on a cancellation as on an
            # ordinary return.
            self._arrange_teardown()


    def _arrange_teardown(self) -> None:
        """Hand the MCP sessions, the env close and this lifecycle's shutdown to whatever still
        owns this episode, in that order, and get out of the way.

        **Behind every owner, not only the finalization.** An env is owned in turn by its
        finalization while that is grading, by the drain that is still holding the evaluator after
        a deadline committed the verdict early, and by the MCP clients that are still talking to
        it. Arranged in front of any of them, a cancelled `close()` released the session and ran
        `_close` underneath: a correct submission came back `correct=false` because its gold
        answer had been taken away, and a deadline path left the evaluator running against state
        that was already gone. So this waits for each of them in turn, by hanging off the one
        ahead rather than by anyone standing there, and only then closes.

        The sessions are one of those owners and are disposed here rather than left to the next
        caller. A cancelled close used to skip them entirely, and `ClientMCPSession.close` marks
        itself closed before it awaits the transport, so a second close is not a retry: the
        subprocess it was going to reap stays."""
        if self._teardown_arranged:
            return
        self._teardown_arranged = True
        self._teardown_behind()

    def _teardown_behind(self) -> None:
        """Run the teardown once nothing owns the env any more, re-asking each time.

        **Lazily, and in order, and never from a list.** The owners are created by each other: a
        deadline's finalization completes and only *then* makes the drain and the evaluator that
        outlive it. A list built when the teardown was arranged captured those as ``None`` and
        skipped them, and closed the env under an evaluator that had been created since. Worse,
        naming the session disposal in that list *started* it, so the transport was being closed
        underneath the finalizer the ordering exists to protect.

        So this asks who owns the env now, waits for that one, and asks again. The disposal is
        started last, when it is genuinely the last owner, and the dispatch is asked about first,
        because a call this episode accepted is using the session everything else is about to
        release."""
        for gate in (
            self._inflight,
            self._finalization,
            self._drain_task,
            self._eval_task,
        ):
            if gate is not None and not gate.done():
                gate.add_done_callback(lambda _f: self._teardown_behind())
                return
        disposing = self._dispose_sessions()
        if disposing is not None and not disposing.done():
            disposing.add_done_callback(lambda _f: self._teardown_behind())
            return
        self._teardown_now()

    def _dispose_sessions(self) -> "Optional[asyncio.Future[None]]":
        """Close every MCP session this episode opened, once, as an operation of its own.

        Retained rather than awaited inline, for the reason the env hooks are: a cancellation
        between two of them left the rest open, and a session's own close is not retryable
        because it marks itself closed before it awaits the transport it is reaping. A caller
        that stays awaits this; a caller that is cancelled leaves it running and the teardown
        waits behind it."""
        if self._disposing is None:
            loop = _running_loop()
            if loop is None:
                return None
            self._disposing = loop.create_task(self._dispose())
            _PENDING_CLOSES.add(self._disposing)
            self._disposing.add_done_callback(_PENDING_CLOSES.discard)
        return self._disposing

    async def _dispose(self) -> None:
        for session in self._opened:
            try:
                await session.close()
            except asyncio.CancelledError:
                # A session's own cancellation is third-party code failing, and containing it is
                # what keeps the sessions after it from being skipped. This task is nobody's
                # caller, so a cancellation arriving here is never a caller withdrawing.
                pass
            except Exception:  # noqa: BLE001 - best effort; one session may not block the rest
                pass

    def _teardown_now(self) -> None:
        """The arrangement itself: the release, then one close, then the shutdown behind it.

        The release is issued *here* and not only on the path that awaited it. A cancellation at
        an earlier await jumped straight to the arrangement with the session still unclaimed, and
        the base `Env.close` this ends in releases any session it still finds, synchronously, on
        whichever loop it was posted to: a three-hundred-millisecond hook ran on the serving loop
        and stopped everything else on it. Claimed first, that close has nothing of this
        episode's left to do."""
        self._release()
        self._lifecycle.stop_when(self._cleanup.watching())

    async def _close_env(self) -> None:
        """Close the env, after this session's release and never beside it.

        ``Env.close`` states the order: release the sessions, then hand off to ``_close``. The
        claim keeps a second caller out of the release hook, and this keeps ``_close`` from
        running while the first one is still inside it: a ``_close`` that tears down what the
        release is using is the same use-after-free by a different route.

        Where it runs was decided when the env was built and is not decided again here. Built on
        this caller's loop, it is closed on this caller's loop, awaited, which is the only loop a
        loop-affine env's ``close`` may use. Built on the lifecycle loop, it is closed there, and
        this waits for it. Either way one close happens and every caller that joins is told what
        it did, including a second ``close()`` and including the stream holding the slot."""
        release = self._release()
        self._cleanup.close_env()
        if self._cleanup.owned_by_caller and release.done():
            # The release is out and this is the loop that built the env, so this caller closes
            # it here and now, and is told what that did. Nothing is revoked from anyone: a
            # second caller arriving mid-close joins this one rather than moving the env to a
            # loop it never met.
            await self._cleanup.here()
            return
        # Either the env belongs to the lifecycle loop, or its release has outrun the bound this
        # caller was promised. Both are the same arrangement: the lifecycle runs the close behind
        # the release it is already holding, and posts it back to the owning loop when there is
        # one. What is left here is a wait, inside whatever the one budget still allows.
        await self._cleanup.joined(self._teardown_budget())
