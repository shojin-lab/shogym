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
import concurrent.futures
import contextvars
import json
import os
import threading
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


#: How long a failed or cancelled ``open_env`` waits for the rollback of a setup hook it
#: abandoned. The release does not need this wait to happen (the hook thread performs it either
#: way), so this only decides how long the caller is held before its own error reaches it.
_ROLLBACK_SECONDS = 60.0


def _session_hooks() -> "concurrent.futures.ThreadPoolExecutor":
    """The one thread an episode runs its env's session hooks on.

    One per episode, one thread in it, built at the moment the first hook is submitted and shut
    down when the episode closes. Three properties come out of that shape and every one of them
    is load-bearing:

    * **It outlives the caller and the loop.** What a submit hands back is a plain
      :class:`concurrent.futures.Future` whose completion callbacks run *in the worker thread*.
      So a rollback arranged for an abandoned setup finishes in the thread that ran the setup,
      whether or not the caller's task still exists and whether or not the event loop it was
      running on is still open.
    * **One thread means the hooks cannot overlap.** A release submitted while the setup hook is
      still running waits in the queue rather than beside it.
    * **Nothing is created at import.** A pool built at import gives every process that imports
      this module threads it never asked for, and a fork inherits the pool's locks without the
      threads that would release them, so the first submit in the child waits forever. This
      module is imported by a stream that has a test which forks, so that is not hypothetical.
    """
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="shogym-session"
    )


def _running_loop() -> "Optional[asyncio.AbstractEventLoop]":
    """The loop this call is on, or ``None`` when it is not on one."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


#: How long a caller waits to join an env close somebody else owns. A bound rather than a promise,
#: for the reason the release bound is one: the close it is joining may be inside a hook that is
#: not coming back.
_ENV_CLOSE_SECONDS = 60.0

#: Env closes scheduled onto an owning loop from another thread. A task nothing holds may be
#: collected before it runs, so each one is kept until it is done.
_PENDING_CLOSES: "set[asyncio.Task[None]]" = set()


class _EnvClose:
    """The one close of one env, taken by one caller and run on the loop that built it.

    Two things have to be true at once and neither of them is free.

    **It has to happen exactly once.** ``close`` is not ``end_session``: it hands off to the env's
    ``_close``, which releases what the *constructor* made, and running that twice tears down the
    same resource twice. The paths that want it are a caller that closed an episode, a second
    caller that closed it again, and the hook thread finishing a release the first caller stopped
    waiting for. All three come here and exactly one of them runs it; the others join.

    **It has to happen on the right loop.** ``TaskStream`` promises that its envs are closed on
    the loop that built them and on no other, and a factory is explicitly allowed to bind
    loop-affine resources, so the throwaway loop a worker thread could give it is not a safe
    generalisation. The owning loop is captured when this is made, and a close arranged from a
    thread is scheduled back onto it.

    **When that loop is gone.** An env whose caller tore its loop down still has a constructor's
    worth of resources and nobody but this to release them, so the close runs on a temporary loop
    rather than not at all. That is the one case where the loop is not the env's own, so it is
    also the one case that is said out loud: an env that refuses it raises, and the refusal is
    warned about rather than swallowed."""

    def __init__(self, env: Any, owner: "Optional[asyncio.AbstractEventLoop]") -> None:
        self._env = env
        self._owner = owner
        self._lock = threading.Lock()
        self._taken = False
        self._arranged = False
        self._done = threading.Event()
        #: What the close raised, if it raised. Read by tests and by the warning below.
        self.failure: Optional[BaseException] = None
        #: True if the close had to run somewhere other than the loop that built the env.
        self.orphaned = False

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def _take(self) -> bool:
        with self._lock:
            if self._taken:
                return False
            self._taken = True
            return True

    async def here(self) -> None:
        """Close on the caller's loop, and join whoever is already closing if that is not this."""
        if self._owner is not None and _running_loop() is not self._owner:
            # Somebody else's loop. Hand it to the one that owns the env rather than closing an
            # env's constructor state from a loop it never met.
            self.from_thread()
            await self._joined()
            return
        if not self._take():
            await self._joined()
            return
        await self._run()

    def arrange(self, hooks: "concurrent.futures.ThreadPoolExecutor") -> None:
        """Queue the close behind whatever this episode's hook thread is already doing.

        Arranged rather than awaited, and arranged *early*: the thread starts it once the release
        ahead of it in this one thread's queue has finished, so the ordering ``Env.close`` states
        holds without the caller being held for a hook that has outrun its bound, and without the
        decision depending on a coroutine that may never be resumed. A loop that closes while its
        caller is parked cannot orphan a close that was already queued."""
        with self._lock:
            if self._arranged or self._taken:
                return
            self._arranged = True
        try:
            hooks.submit(contextvars.copy_context().run, self.from_thread)
        except RuntimeError:
            # The hook thread is gone, so there is nothing left for this to queue behind.
            self.from_thread()

    def from_thread(self) -> None:
        """Run the close from a thread, on the owning loop while that loop can still take work."""
        if not self._take():
            return
        loop = self._owner
        if loop is not None and not loop.is_closed() and loop.is_running():
            try:
                loop.call_soon_threadsafe(self._spawn)
                return
            except RuntimeError:
                pass
        self._on_a_loop_of_its_own()

    def _spawn(self) -> None:
        """On the owning loop: start the close and keep the task alive until it finishes."""
        loop = self._owner
        try:
            assert loop is not None
            task = loop.create_task(self._run())
        except BaseException as exc:  # noqa: BLE001 - reported, never raised into the loop
            self.failure = exc
            self._done.set()
            self._warn()
            return
        _PENDING_CLOSES.add(task)
        task.add_done_callback(_PENDING_CLOSES.discard)

    def _on_a_loop_of_its_own(self) -> None:
        self.orphaned = True
        try:
            asyncio.run(self._run())
        except BaseException as exc:  # noqa: BLE001 - reported, never raised into a worker thread
            self.failure = exc
            self._done.set()
        self._warn()

    async def _run(self) -> None:
        try:
            await self._env.close()
        except BaseException as exc:  # noqa: BLE001 - recorded; there is no caller to raise to
            self.failure = exc
        finally:
            self._done.set()

    def _warn(self) -> None:
        if self.failure is None:
            return
        warnings.warn(
            f"an env could not be closed after its episode failed: "
            f"{type(self.failure).__name__}: {self.failure}. Its loop was gone by the time the "
            "session release finished, so the close was attempted on a temporary one; an env "
            "whose resources belong to the loop that built it cannot be closed this way and "
            "whatever its constructor made is still held.",
            RuntimeWarning,
            stacklevel=2,
        )

    async def _joined(self) -> None:
        """Wait for the owner, off the loop, so a close scheduled onto this loop can run."""
        if self._done.is_set():
            return
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._done.wait, _ENV_CLOSE_SECONDS),
                timeout=_ENV_CLOSE_SECONDS + 1.0,
            )
        except BaseException:
            pass


class _SetupRollback:
    """The single owner of the rollback for a setup hook that was abandoned mid-flight.

    A thread cannot be cancelled, so a caller that gives up on ``begin_session`` leaves the hook
    running and whatever it goes on to create, a process or a port or a directory, with nobody
    left to release it. This queues the rollback behind the hook on the episode's own hook thread:
    the hook lands, and the rollback runs next, in that same thread. No loop is consulted and none
    has to still exist.

    **One owner, and no second caller beside it.** ``Env.begin_session`` records the session id
    before entering the hook, so an ``env.close()`` on the failure path would end a session the
    hook is still inside. The failure path therefore waits on :meth:`settled` instead of closing
    the env itself, and :meth:`shogym.core.Env.claim_session` makes any later attempt a no-op.

    **Both halves are queued, not decided later.** Releasing the session is half of the cleanup
    ``open_env`` promised; closing the env is the other half, and it used to be decided by the
    coroutine after the wait. A coroutine parked on a loop that then closes never decides
    anything, so the env was released and left open. The whole rollback is one job now, queued
    when this is made."""

    def __init__(
        self,
        hooks: "concurrent.futures.ThreadPoolExecutor",
        env: Any,
        session_id: str,
        cleanup: _EnvClose,
    ) -> None:
        self._cleanup = cleanup
        self._released = threading.Event()

        def rollback() -> None:
            try:
                env.end_session(session_id)
            except Exception:
                pass
            finally:
                self._released.set()
            # Then the env, on the loop that built it if that loop is still there to take it.
            cleanup.from_thread()

        try:
            hooks.submit(contextvars.copy_context().run, rollback)
        except RuntimeError:
            # The executor is already shutting down, which leaves this caller as the only one
            # who can still run the rollback. Here rather than nowhere.
            rollback()

    @property
    def released(self) -> bool:
        """Has the session release actually finished? Not "was it issued", and not "did the wait
        return": whether the hook is out."""
        return self._released.is_set()

    async def settled(self, timeout: Optional[float] = None) -> bool:
        """Wait for the whole rollback, if there is still a loop to wait on, and say whether the
        session release landed.

        A bound rather than a promise: the caller is already carrying an error and must not be
        held by a hook that is not coming back. What is guaranteed without any waiting at all is
        that both halves happen, because the hook thread performs them; the wait is what lets the
        env close, which is scheduled back onto this loop, actually run before the error reaches
        the caller.

        The default is read here rather than bound as a default argument, so the module constant
        is one value a caller can see and a test can move."""
        bound = _ROLLBACK_SECONDS if timeout is None else timeout
        try:
            await asyncio.wait_for(
                asyncio.to_thread(self._cleanup._done.wait, bound), timeout=bound + 1.0
            )
        except BaseException:
            # Including a `CancelledError`: this runs inside a handler that re-raises the failure
            # it was called for, so the wait giving way never loses it. Neither half depends on
            # this wait.
            pass
        return self.released


def _settled() -> "concurrent.futures.Future[None]":
    """A future that is already done: the release for a session somebody else has claimed."""
    done: "concurrent.futures.Future[None]" = concurrent.futures.Future()
    done.set_result(None)
    return done


def _discard_when_built(
    building: "concurrent.futures.Future[Any]",
    owner: "Optional[asyncio.AbstractEventLoop]",
) -> None:
    """Close an env whose caller stopped waiting for it to be built, once it exists.

    A constructed env holds whatever its constructor made (for an out-of-process world, another
    process), and the caller that asked for it has gone. The close goes back to ``owner``, the
    loop the caller was on, while that loop can still take it, and to a temporary one otherwise;
    :class:`_EnvClose` is what decides which and what says so when the second is not enough.

    Not a daemon thread: this is the only thing that will ever close this env, and a process that
    exits before it runs leaks whatever the constructor made."""

    def discard(finished: "concurrent.futures.Future[Any]") -> None:
        if finished.cancelled() or finished.exception() is not None:
            return
        cleanup = _EnvClose(finished.result(), owner)
        threading.Thread(
            target=cleanup.from_thread, name="shogym-discard", daemon=False
        ).start()

    building.add_done_callback(discard)


def _swallow(finished: "asyncio.Future[Any]") -> None:
    """Read an abandoned wrapper's outcome so asyncio does not report it as never retrieved.

    The failure it carries is the one already on its way to the caller through the shield, and
    the work it describes has an owner of its own."""
    if not finished.cancelled():
        finished.exception()


async def _built(factory: "Callable[[], Any]") -> Any:
    """Construct one env off the event loop, and let go of it cleanly if the caller stops waiting.

    Constructing an env is ordinary blocking work, and for some envs it is real work: provisioning
    a corpus, walking and copying two views of it, taking a file lock. Run on the shared loop it
    stops every other episode the process is serving, along with their watchdogs and deadlines,
    which is the whole reason the session hooks are off the loop too.

    Offloading introduces one thing the synchronous call did not have: a moment where the env
    exists and its caller no longer does. A thread cannot be cancelled, so a caller that gives up
    on this await still gets an env built, holding whatever its constructor made. The wait is
    therefore shielded and an abandoned build is closed by :func:`_discard`, arranged on the
    builder's own thread so it does not depend on the loop the caller just left.

    **Only for a factory whose caller has said it may be.** There is no running loop in the
    thread this runs on, so a constructor that binds one raises there; callers declare the
    difference rather than this guessing it (see ``TaskStream``'s ``off_loop_factory``). What is
    carried across is the caller's context, so a factory that reads a context variable reads the
    value the caller set; what is not carried is the loop, which is the whole point."""
    builder = concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="shogym-build"
    )
    try:
        building = builder.submit(contextvars.copy_context().run, factory)
    finally:
        # Immediately: the thread runs the job it already has and then exits. One construction,
        # one thread, no pool left behind and none created at import.
        builder.shutdown(wait=False)
    waiter = asyncio.wrap_future(building)
    try:
        return await asyncio.shield(waiter)
    except BaseException:
        waiter.add_done_callback(_swallow)
        _discard_when_built(building, _running_loop())
        raise


#: How long teardown waits for an env to release one episode's resources before going on without
#: it. Teardown runs on the shared loop, so this is a bound on the wait rather than a kill: an env
#: whose release has to be certain makes its own hook bounded. Spent once per session: whichever
#: of teardown and close reaches the release first waits, and the other observes.
_END_SESSION_SECONDS = 60.0


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
        hooks: Optional["concurrent.futures.ThreadPoolExecutor"] = None,
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
        # The thread this episode's env session hooks run on, and the ONE release future for its
        # session. Both are per-episode: `_hooks` is shut down by `close()`, and `_released` is
        # issued by whichever of teardown and close reaches it first and then only observed
        # (see `_release`). `_release_waited` records that the bound has been spent, so the two
        # paths cannot wait for one operation twice.
        self._hooks = hooks if hooks is not None else _session_hooks()
        self._released: Optional["concurrent.futures.Future[None]"] = None
        self._release_waited = False
        # The one close of this env, and the loop it was built on. Every path that wants the env
        # closed goes through this object, so exactly one of them runs `_close` and the rest join
        # (see `_EnvClose` and `_close_env`).
        self._cleanup = _EnvClose(env, _running_loop())
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

        ``off_loop_factory`` declares that this env's constructor is safe to run in a worker
        thread: no running loop bound in it, no thread-affine resources. Constructing an env is
        blocking work and some envs make it real work (provisioning a corpus, walking and
        copying a cache, taking a file lock), so a cold or contended construction on the shared
        loop stops every other episode this process is serving, and their watchdogs and deadlines
        with them. Off by default, because an env is allowed to bind loop-affine resources in its
        constructor and a caller that has not said this one does not is a caller whose env this
        may not move."""
        if off_loop_factory:
            env = await _built(lambda: make(env_name, config=env_config))
        else:
            env = make(env_name, config=env_config)
        return await cls.open_env(
            env,
            env_name=env_name,
            task=task,
            trace_path=trace_path,
            finalize_deadline=finalize_deadline,
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
        # The episode's own hook thread, built here so the setup hook and its rollback are the
        # same thread's work, and handed to the episode below. On every path out of this method
        # it belongs to exactly one owner: the episode that was returned, or the failure handler.
        hooks = _session_hooks()
        # The one close of this env, made before anything can fail so that every failure path
        # below hands it to the same owner, and carrying the loop this call is on so a close
        # arranged from the hook thread goes back to the loop that built the env.
        cleanup = _EnvClose(env, _running_loop())
        rollback: Optional[_SetupRollback] = None
        # The session id, once `begin_session` has returned and the env therefore holds one. It
        # is what the failure handler below needs to release, and it is a value rather than a
        # flag so that "there is a session to release" and "here is which one" are one answer.
        began: Optional[str] = None
        handed_over = False
        env_name = env_name if env_name is not None else env.name
        try:
            task_idx = int(task) if task is not None else None
            task_data = env.load_task(task_idx)
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
            for spec in env.essential_specs():
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
            beginning = hooks.submit(
                contextvars.copy_context().run, env.begin_session, session_id, task_data
            )
            waiter = asyncio.wrap_future(beginning)
            try:
                await asyncio.shield(waiter)
            except BaseException:
                waiter.add_done_callback(_swallow)
                rollback = _SetupRollback(hooks, env, session_id, cleanup)
                raise
            # From here the env holds a session, so every failure below owes it a release, and
            # that release goes the same way this one would have: through the hook thread, never
            # on the loop. `env.close()` is what used to do it, and `Env.close` runs the hook
            # inline, so a slow release on a `describe` that raised froze the whole server.
            began = session_id
            # The one description this episode ever asks for. Everything published about the task
            # and everything enforced on it comes off this single answer — see the snapshot the
            # constructor takes of it.
            spec = env.describe(resolved_task)
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
                hooks=hooks,
            )
            handed_over = True
            return episode
        except BaseException:
            # Setup failed, so no ServedEpisode is returned for the caller to close: release
            # everything here. Every step is best-effort so the original setup error propagates.
            for session in opened:
                try:
                    await session.close()
                except Exception:
                    pass
            if rollback is None and began is not None:
                # Setup got past `begin_session` and failed after it: `describe` raised, or the
                # constructor's own fail-loud guard did. The env holds a session either way, and
                # it is released the same way a cancelled one is.
                rollback = _SetupRollback(hooks, env, began, cleanup)
            if rollback is not None:
                # The setup hook may still be running and its rollback is already queued behind
                # it on the hook thread. Closing the env here would be a second release on one
                # episode's resources, because `Env.begin_session` records the session id before
                # entering the hook, so `Env.close` would find it and enter `_end_session` while
                # `_begin_session` is still inside. Wait for the one owner instead, for as long
                # as there is a loop to wait on; a caller that is tearing the loop down gets a
                # thread that finishes the release without it.
                #
                # The other half of what this method promised, the env close, was queued with the
                # release rather than decided here. That is the difference between a promise and
                # a hope: a coroutine parked on a loop that then closes never decides anything,
                # and the env was released and left open. This wait is only what lets a close
                # scheduled back onto this loop finish before the error reaches the caller.
                await rollback.settled()
            else:
                # No session was ever begun, so there is nothing for the close to follow.
                await cleanup.here()
            if not handed_over:
                hooks.shutdown(wait=False)
            raise

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
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
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
        if not terminal_call:
            await self._settled()

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
                    tool_name, args, write_trace=not is_horizon
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
            async with self._lock:
                # The world may have been sealed under this call while it ran: the deadline
                # forces a terminal without waiting for it. Then there is no horizon terminal to
                # begin, and this caller reads the same tombstone any post-seal call reads.
                if self._state is not LifecycleState.OPEN or self._terminated:
                    return self._sealed_tombstone()
                # Horizon has no submission: finalize with source=horizon and no args, but keep
                # the real tool name so the terminal trace row is labelled with the call that hit
                # the budget.
                finalization = self._begin_finalization("horizon", tool_name, None)

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

        feedback = self._env.verify(self._trajectory, self._task, terminated=terminated)
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

        Shielded, and it never re-raises: what a waiter needs from a previous operation is that it
        is over, not that it succeeded."""
        while True:
            inflight = self._inflight
            if inflight is None or inflight.done():
                return
            try:
                await asyncio.shield(inflight)
            except BaseException:  # noqa: BLE001 (a previous caller's failure is not this one's)
                return

    def _begin_dispatch(
        self,
        tool_name: str,
        args: Dict[str, Any],
        *,
        write_trace: bool = True,
        legacy: bool = False,
    ) -> "asyncio.Future[tuple[CallResult, int]]":
        """Start the one ordinary dispatch this episode owns. **Called under the lock.**"""
        runner = (
            self._legacy_step(tool_name, args)
            if legacy
            else self._dispatch_step(
                tool_name, args, terminated=False, write_trace=write_trace
            )
        )
        dispatch: "asyncio.Future[tuple[CallResult, int]]" = asyncio.ensure_future(runner)
        # Read whatever it ends with, so an operation whose caller went away does not leave an
        # unretrieved exception for asyncio to complain about at collection. Nothing acts on it:
        # the failure belongs to the caller that is still waiting, if there is one.
        dispatch.add_done_callback(lambda done: done.cancelled() or done.exception())
        self._inflight = dispatch
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
        feedback = self._env.verify(self._trajectory, self._task, terminated=terminated)
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
                eval_task: "asyncio.Future[Any]" = asyncio.ensure_future(
                    self._finalize(req)  # type: ignore[misc]
                )
                self._eval_task = eval_task
                if self._finalize_deadline is not None:
                    # shield inside wait_for: the deadline must not cancel the evaluator
                    # mid-flight (that could wedge the in-process client); it only bounds how
                    # long we *await* the verdict before failing closed. The evaluator keeps
                    # running and is drained by `_teardown`.
                    evidence = await asyncio.wait_for(
                        asyncio.shield(eval_task), timeout=self._finalize_deadline
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
                try:
                    feedback = self._env.verify(
                        self._trajectory, self._task, terminated=True, evidence=evidence
                    )
                    items = [*feedback.inference, *feedback.episode]
                except Exception as exc:  # noqa: BLE001 (verifier failure => fail closed)
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
                    diagnostic=diagnostic,
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

        The session is claimed on the loop and the hook runs off it. Claiming first is what makes
        the later ``env.close()`` in :meth:`close` safe: the id is gone from the env the instant
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
                    self._released = self._hooks.submit(
                        contextvars.copy_context().run, release
                    )
                except RuntimeError:
                    # The hook thread is gone (a second close), so this caller runs the release.
                    release()
                    self._released = _settled()
        return self._released

    async def _release_session(self) -> None:
        """Issue the session's one release and wait for it, once, up to the bound.

        Off the loop and bounded, for the reason `begin_session` is: this runs on the shared loop
        and an env that waits on a wedged child would hold every other episode with it. At the
        bound the wait is abandoned, never the release: an env that needs its own release to be
        certain has to make its hook bounded too, which is why the appworld worker's close signals
        and reaps rather than asking politely over a socket.

        The bound is spent once per session. Whichever of teardown and close reaches this first
        does the waiting; the other returns immediately, because waiting a second time for one
        operation is how a 60-second bound becomes a 120-second one."""
        release = self._release()
        if self._release_waited:
            return
        self._release_waited = True
        try:
            # Shielded so the timeout cancels this wait and not the release: cancelling the
            # wrapper would try to cancel the underlying job, and a job still queued behind a
            # slow hook would be cancelled outright, abandoning the release rather than the
            # wait.
            await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(release)),
                timeout=_END_SESSION_SECONDS,
            )
        except Exception:
            pass

    async def close(self) -> None:
        # close() participates in the lifecycle for a seal env. If a finalization is in flight,
        # WAIT for it to commit evidence + tear down before disposing anything (so the
        # evaluator's live session isn't reclaimed mid-finalize); otherwise atomically claim an
        # abort and own teardown. Then dispose the MCP client sessions and let the env drop
        # residual state. For a non-seal env none of this engages (state stays OPEN with no
        # finalization), so close() is just the plain teardown below.
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
            # A deadline-path finalization returns before its background drain+teardown finishes;
            # wait for it so the evaluator has drained and env state is released before we dispose
            # the MCP sessions below.
            if self._drain_task is not None:
                try:
                    await asyncio.shield(self._drain_task)
                except Exception:
                    pass
            await self._teardown()  # idempotent: a no-op if the finalizer already ran it

        # Close every MCP session opened for this episode, then let the env tear down its own
        # per-session state. Out-of-process sessions are reaped by their `close()` above (one
        # subprocess/session).
        for session in self._opened:
            try:
                await session.close()
            except Exception:
                pass
        # This episode's session is released here, through the one owned release, and bounded.
        # For a non-seal env this is where it happens at all, since nothing above tears down. The
        # `env.close()` below is then a close over what is left of the env, not a second entry
        # into this session's hook: the session was claimed the moment the release was issued, so
        # even a release still running in its thread leaves `Env.close` with nothing of this
        # episode's to do.
        await self._release_session()
        await self._close_env()
        # The hook thread has no work left that anything is waiting for. `wait=False`: a release
        # still inside a wedged hook, and the env close queued behind it, finish in their own
        # time and this close is not held by them, which is the same bound the wait above
        # already declared.
        self._hooks.shutdown(wait=False)

    async def _close_env(self) -> None:
        """Close the env, after this session's release and never beside it.

        ``Env.close`` states the order: release the sessions, then hand off to ``_close``. The
        claim keeps the second caller out of the release hook, but it does not by itself keep
        ``_close`` from running while the first one is still inside it, and a ``_close`` that
        tears down what the release is using is the same use-after-free by a different route.

        So the release decides where this runs. Landed, and it is awaited here, on the loop that
        built the env and the only loop a loop-affine env's ``close`` may use. Still inside the
        hook, and it is queued behind the release on the hook thread instead: the caller is not
        held past the bound it was promised, ``_close`` still does not start until the release is
        out, and the close is scheduled back onto the owning loop when it does.

        Both routes go through the same owner, so the second caller of ``close()`` joins the
        close the first one arranged rather than starting another beside it. Checking whether the
        release had landed was not that check: a first close that timed out and queued the close
        left a second one, arriving after the release finished, walking straight into ``_close``
        while the queued one was already inside it."""
        release = self._release()
        if release.done():
            await self._cleanup.here()
            return
        self._cleanup.arrange(self._hooks)
