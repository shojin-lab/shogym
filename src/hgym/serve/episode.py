"""The episode-serving engine (RFC 008): drive a served env one tool call at a time.

A tool call *is* the step. :class:`ServedEpisode` opens the env's essential MCP sessions
directly (the terminate server + the env-mandatory servers), dispatches each incoming call
to the right session, records a flat :class:`~hgym.trajectory.Trajectory`, gates the
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
:class:`~hgym.serve.lifecycle.TerminalEvidence`; (4) records the evidence in the **durable
finalization store** and appends a versioned ``terminal`` trace event; (5) tears down; (6)
returns a public-safe, sanitized payload. After the seal every ``tools/call`` is tombstoned
with no inward dispatch. The state machine is
``OPEN -> SEALED -> FINALIZING -> FINALIZED -> TEARING_DOWN -> CLOSED``.

An env that marks nothing ``score`` never enters this path: ``_seal_enabled`` is False, so
:meth:`call` runs *only* the legacy single-step body — its behaviour is unchanged.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema

from hgym.envs import make
from hgym.feedback.wire import build_meta, dump_item, select_inband
from hgym.mcp.session import MCPSession
from hgym.mcp.toolset import _open_session_for_spec
from hgym.serve.lifecycle import (
    FinalizationRecord,
    FinalizationStore,
    FinalizeRequest,
    LifecycleState,
    TerminalEvidence,
    args_digest,
    fail_closed_verdict,
)
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from hgym.task import TaskSpec
from hgym.trace import (
    append_terminal_event,
    append_trace,
    step_record,
    terminal_event_record,
)
from hgym.trajectory import Step, Trajectory
from hgym.utils.uuid7 import uuid7


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
        # through — not only in ToolUsingEnv's construction check. A non-ToolUsingEnv env that
        # builds its TaskSpec/manifest directly would otherwise slip a score terminal past the
        # serve layer with no callable finalize, silently leaving `_seal_enabled` False and
        # routing its advertised, authoritative scoring through the legacy marker/trajectory
        # path — reopening the grade->read->fix->grade exploit for an env that expected the seal
        # to protect it. Refuse to run, loudly, rather than downgrade.
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
        # Set if a durable-record write ever failed (best-effort persistence — never fatal).
        self._persist_degraded = False
        # The in-flight evaluator task (retained so teardown drains it — see `_run_finalize`).
        self._eval_task: Optional["asyncio.Future[Any]"] = None
        # A background drain+teardown task, used only on the deadline path so the caller gets
        # the fail-closed result AT the deadline while resource cleanup waits for the evaluator.
        self._drain_task: Optional["asyncio.Future[None]"] = None
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
            try:
                self._store.recover()
            except OSError:
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
    ) -> "ServedEpisode":
        """Build the env, load the task instance, open the essential MCP sessions, and push
        per-episode state into the (in-process) tool servers."""
        return await cls.open_env(
            make(env_name, config=env_config),
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
        than a shared one (``Env.close`` on a ``ToolUsingEnv`` ends **every** session it
        tracks, which would tear down any sibling episode sharing the instance).
        """
        opened: List[MCPSession] = []
        env_name = env_name if env_name is not None else env.name
        try:
            task_idx = int(task) if task is not None else None
            task_data = env.load_task(task_idx)
            # Publish the *resolved* task identity so a random-default episode (task
            # omitted) is still attributable: an env that indexes tasks records the
            # chosen index in task_data (Wordle: "task_idx"), so a `hgym serve wordle_v1`
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
            env.begin_session(session_id, task_data)
            # The one description this episode ever asks for. Everything published about the task
            # and everything enforced on it comes off this single answer — see the snapshot the
            # constructor takes of it.
            spec = env.describe(resolved_task)
            # Construct inside the try so the cleanup below also covers the constructor's own
            # fail-loud guard (a `score` manifest with no callable finalize): the sessions
            # opened above are released before the error propagates.
            return cls(
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
            )
        except BaseException:
            # Setup failed, so no ServedEpisode is returned for the caller to close:
            # release everything here. Close any opened MCP sessions, then close the
            # env (drops per-episode state begin_session may have pushed before
            # raising). Both are best-effort so the original setup error propagates.
            for session in opened:
                try:
                    await session.close()
                except Exception:
                    pass
            try:
                await env.close()
            except Exception:
                pass
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
                # Non-seal env: the single-step path, entirely under the lock.
                return await self._legacy_step(tool_name, args)

            # ----- seal-enabled env -----
            if tool_name in self._score_schemas:
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
                result, _ = await self._dispatch_step(
                    tool_name, args, terminated=False, write_trace=not is_horizon
                )
                if not is_horizon:
                    return result
                # Horizon has no submission: finalize with source=horizon and no args, but keep
                # the real tool name so the terminal trace row is labelled with the call that hit
                # the budget.
                finalization = self._begin_finalization("horizon", tool_name, None)

        # Phase 2 (lock released): await the single in-flight finalization, *shielded* so a
        # cancellation/disconnect of THIS request never cancels the evaluator or re-dispatches
        # it. If we are cancelled the finalization keeps running to completion in the
        # background; a later close() awaits the same future. Exactly one evaluation.
        await asyncio.shield(finalization)
        return finalization.result()

    # ----- legacy (non-seal) step: dispatch, record, verify -----

    async def _legacy_step(self, tool_name: str, args: Dict[str, Any]) -> CallResult:
        # Prospective step: don't advance `self._step` until the call actually completes. If
        # `call_tool` is cancelled (harness timeout) or raises, the counter stays put so the
        # next call reuses this number — the trajectory stays contiguous, one Step per
        # completed call.
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
        return CallResult(
            content=content,
            meta=build_meta(inband, terminate=terminated),
            terminated=terminated,
        )

    # ----- seal-enabled ordinary step (mid-episode, non-terminal) -----

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
        :meth:`hgym.serve.stream.TaskStream.dispatch`). What does not propagate is the *name* of
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
            "core": "hgym-serve",
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
                except Exception:  # noqa: BLE001 — verifier failure => fail closed
                    evidence = TerminalEvidence(
                        source=source,  # type: ignore[arg-type]
                        status="finalize_error",
                        verdict=fail_closed_verdict(confidence),
                        provenance=evidence.provenance,
                        finalization_id=finalization_id,
                        diagnostic="verify() raised while scoring the terminal evidence",
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
        try:
            self._env.end_session(self._session_id)
        except Exception:
            pass
        self._state = LifecycleState.CLOSED

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
        # per-session state. `env.close()` drops in-process server state via `end_session`;
        # out-of-process sessions are reaped by their `close()` above (one subprocess/session).
        for session in self._opened:
            try:
                await session.close()
            except Exception:
                pass
        await self._env.close()
