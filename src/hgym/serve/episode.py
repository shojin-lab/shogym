"""The episode-serving engine (RFC 008): drive a served env one tool call at a time.

A tool call *is* the step. :class:`ServedEpisode` opens the env's essential MCP sessions
directly (the terminate server + the env-mandatory servers), dispatches each incoming call
to the right session, records a flat :class:`~hgym.trajectory.Trajectory`, gates the
horizon env-side, and runs the env's pure ``verify`` on each call. Feedback rides back on
the ``_meta`` sidecar (episode-level hidden until the terminal result) and every step is
appended to the JSONL trace. No gym ``step``/``Observation`` anywhere.

**RFC 009 (HLE-only prototype) — seal-before-verdict.** When a tool the env marks
``terminal_kind="score"`` is called, ``call`` runs a *terminal transaction* instead of an
ordinary step: it (1) validates the args against the advertised outer schema — an invalid
terminal request is a normal validation error while the episode is still OPEN, never a
sealed finalizer with unvalidated evidence; (2) atomically seals the episode
(``OPEN -> SEALED``); (3) runs the evaluator (for HLE, the judge inside ``submit_answer``)
as a single, cancellation-safe finalization; (4) records the in-memory verdict; (5) tears
down; (6) returns a public-safe, sanitized payload. After the seal every ``tools/call`` is
tombstoned with no inward dispatch. The state machine is
``OPEN -> SEALED -> FINALIZING -> FINALIZED -> TEARING_DOWN -> CLOSED`` and is **explicitly
in-memory / non-durable** — no crash/restart/exactly-once-durability claim is made (that is
Phase 1). Envs that mark nothing ``score`` never enter this path and are byte-identical to
the pre-RFC-009 engine.
"""

from __future__ import annotations

import asyncio
import enum
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import jsonschema

from hgym.envs import make
from hgym.feedback.wire import build_meta, dump_item, select_inband
from hgym.mcp.session import MCPSession
from hgym.mcp.toolset import _open_session_for_spec
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from hgym.task import TaskSpec
from hgym.trace import append_trace, step_record
from hgym.trajectory import Step, Trajectory
from hgym.utils.uuid7 import uuid7


class LifecycleState(enum.Enum):
    """The RFC-009 per-episode lifecycle (in-memory, non-durable). Only a ``score``-terminal
    call drives it past ``OPEN``; ordinary/abort tools never touch it (they use the legacy
    ``terminated`` flag), so non-score envs stay on the ``OPEN`` state throughout."""

    OPEN = "open"
    SEALED = "sealed"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    TEARING_DOWN = "tearing_down"
    CLOSED = "closed"


@dataclass
class CallResult:
    """The outcome of one tool call: the tool's functional ``content`` (the observation the
    harness needs), the ``meta`` sidecar (feedback + terminate flag), and whether the episode
    is now over."""

    content: str
    meta: Dict[str, Any]
    terminated: bool


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
        score_schemas: Optional[Dict[str, Dict[str, Any]]] = None,
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
        # concurrent MCP requests on this session must not interleave. Under RFC 009 this
        # same lock is the lifecycle-state lock (`_state` below): seal, finalize-evidence
        # commit, close-race arbitration, and teardown all take it.
        self._lock = asyncio.Lock()

        # ----- RFC 009 seal-before-verdict lifecycle (in-memory, non-durable) -----
        # The score-terminal tool name -> its advertised outer schema, from the manifest.
        # Empty for every non-score env, so `call()` never leaves its legacy path there.
        self._score_schemas: Dict[str, Dict[str, Any]] = dict(score_schemas or {})
        self._state = LifecycleState.OPEN
        # The single in-flight finalization (the judge run). Created exactly once, at seal;
        # shielded by every awaiter so a client cancellation/close can never abandon or
        # re-dispatch it. There is at most one per episode — never a second judge call.
        self._finalization: Optional["asyncio.Future[CallResult]"] = None
        self._finalization_id: Optional[str] = None
        # The committed in-memory verdict (public-safe). None until FINALIZED.
        self._evidence: Optional[Dict[str, Any]] = None
        # Teardown is idempotent and runs exactly once (owned by the finalizer, after
        # evidence; close() routes through the same guard). `_teardown_runs` is a test hook.
        self._torn_down = False
        self._teardown_runs = 0
        # Optional finalize deadline (seconds): the judge run is awaited up to this bound;
        # on timeout the episode fails closed to a `finalize_error` verdict. None disables
        # it (the default in normal serve; tests set a small value to exercise the rule).
        self._finalize_deadline = finalize_deadline

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
        env = make(env_name, config=env_config)
        opened: List[MCPSession] = []
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
            # RFC 009: read the score-terminal marker off the *published contract* (the
            # manifest is the enforcement point, not a marker scan). At most one tool is
            # `score`; capture its advertised schema so the seal path can validate args.
            score_schemas = {
                m.name: m.input_schema
                for m in env.describe(resolved_task).tools
                if m.terminal_kind == "score"
            }
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

        return cls(
            env,
            env_name,
            resolved_task,
            session_id,
            task_data,
            sessions,
            opened,
            trace_path,
            score_schemas=score_schemas,
            finalize_deadline=finalize_deadline,
        )

    @property
    def terminated(self) -> bool:
        return self._terminated

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def terminal_feedback(self) -> List[Dict[str, Any]]:
        """The terminal step's feedback (wire form), or ``[]`` until the episode ends."""
        return self._terminal_feedback

    def describe(self) -> TaskSpec:
        return self._env.describe(self._task_id)

    async def call(
        self, tool_name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> CallResult:
        """Execute one tool call as one step; return its result + feedback sidecar.

        For an ordinary or ``abort`` tool this is unchanged: dispatch, record, verify. For a
        ``score``-terminal tool it runs the RFC-009 transaction (validate -> seal ->
        evaluate). After termination (legacy terminate/horizon) or after a seal, further
        calls are tombstoned with no inward dispatch."""
        # Phase 1: decide and, for a score-terminal call, seal + spawn the finalization —
        # all under the lock so state transitions are atomic against concurrent ingress.
        async with self._lock:
            # Post-seal tombstone: once the lifecycle has left OPEN, no `tools/call`
            # reaches task/env state. Read-only `describe`/`tools/list` bypass `call()`
            # (they hit `episode.describe()` directly), so they stay readable.
            if self._state is not LifecycleState.OPEN:
                return self._sealed_tombstone()
            # Legacy terminate/horizon tombstone (byte-identical to the pre-RFC-009 engine).
            # Only reachable for non-score envs (a score env leaves OPEN via a seal above).
            if self._terminated:
                return CallResult(
                    content="<episode already terminated>",
                    meta=build_meta(terminate=True),
                    terminated=True,
                )

            args = dict(arguments or {})
            # `_session_id` is a reserved hidden field the transport injects with the
            # real id. Strip any caller-supplied value before *both* dispatch and Step
            # construction, so a forged id can't run against the real session nor land
            # in the trajectory a verifier reads.
            args.pop("_session_id", None)

            if tool_name in self._score_schemas:
                # --- validate -> seal (evaluate happens after the lock is released) ---
                invalid = self._validate_terminal_args(tool_name, args)
                if invalid is not None:
                    # A malformed terminal request is a NORMAL validation error while the
                    # episode is still OPEN — no seal, no verdict, no evidence. The harness
                    # may correct and re-submit.
                    return invalid
                # Atomically seal. From here the episode is un-mutable and un-continuable;
                # the verdict that follows is only ever produced for a sealed episode.
                self._finalization_id = str(uuid7())
                self._state = LifecycleState.SEALED
                finalization: "asyncio.Future[CallResult]" = asyncio.ensure_future(
                    self._run_finalize(tool_name, args)
                )
                self._finalization = finalization
                self._state = LifecycleState.FINALIZING
            else:
                # Ordinary / abort tool: the legacy single-step path, still fully under the
                # lock (identical to the pre-RFC-009 engine).
                return await self._ordinary_step(tool_name, args)

        # Phase 2 (lock released): await the single in-flight finalization, *shielded* so a
        # cancellation/disconnect of THIS request never cancels the judge or re-dispatches
        # it. If we are cancelled the finalization keeps running to completion in the
        # background; a later close() awaits the same future. Exactly one judge invocation.
        await asyncio.shield(finalization)
        return finalization.result()

    # ----- ordinary (non-score) step: the legacy path, unchanged -----

    async def _ordinary_step(self, tool_name: str, args: Dict[str, Any]) -> CallResult:
        # Prospective step: don't advance `self._step` until the call actually
        # completes. If `call_tool` is cancelled (harness timeout) or raises, the
        # counter stays put so the next call reuses this number — the trajectory
        # stays contiguous, one Step per completed call.
        step = self._step + 1
        session = self._sessions.get(tool_name)
        if session is None:
            content = f"<unknown tool {tool_name!r}>"
        else:
            result = await session.call_tool(tool_name, args, tool_call_id=f"call-{step}")
            content = result.result
        # The await completed; commit the step atomically with its Step. Everything
        # from here on is synchronous, so no cancellation point can split them.
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
            # Retain the terminal feedback so the no-trace `evaluate()` path can
            # report the score directly off the episode (not only via the trace).
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

        # Eval-safe default: dense inference feedback is recorded (above) but not
        # surfaced in-band. v0 exposes no per-tool opt-in, so surface_inference
        # stays False; episode feedback rides out only on the terminal result.
        inband = select_inband(items, terminal=terminated, surface_inference=False)
        return CallResult(
            content=content,
            meta=build_meta(inband, terminate=terminated),
            terminated=terminated,
        )

    # ----- RFC 009 score-terminal transaction -----

    def _validate_terminal_args(
        self, tool_name: str, args: Dict[str, Any]
    ) -> Optional[CallResult]:
        """Validate a ``score``-terminal call's args against its advertised outer schema.

        Returns a normal validation-error :class:`CallResult` (episode stays OPEN, not
        sealed, no verdict) when the request is malformed, else ``None``. Validates the
        **complete** advertised JSON Schema — types, ``required``, and
        ``additionalProperties`` — BEFORE any lifecycle mutation, so a request FastMCP would
        reject downstream (a non-integer ``confidence``, an unknown extra field, a missing
        ``answer``) is a normal error the harness can correct and re-submit, never a sealed
        finalizer that irrevocably scores 0. On top of the schema, a required string must be
        non-blank (schema ``type: string`` accepts ``""``), so a whitespace-only ``answer``
        can't seal either."""
        schema = self._score_schemas.get(tool_name, {})
        # `args` already had the transport-injected `_session_id` stripped (see `call`), and
        # the advertised schema doesn't list it — so this validates exactly what the harness
        # is allowed to send, mirroring FastMCP's own downstream check.
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
                        f"argument {key!r} must be a non-blank string"
                    )
        return None

    def _validation_error(self, message: str) -> CallResult:
        """A terminal-arg validation failure: a plain error result while still OPEN. Carries
        no seal, no `terminate` flag, and does not advance the step — the call was rejected
        at ingress before any dispatch."""
        return CallResult(
            content=json.dumps({"error": message, "validation_error": True}),
            meta=build_meta(),
            terminated=False,
        )

    async def _run_finalize(self, tool_name: str, args: Dict[str, Any]) -> CallResult:
        """The single finalization: run the evaluator on the *already-sealed* episode,
        commit the in-memory verdict, tear down, and return the sanitized public payload.

        Runs as its own task (created at seal), shielded by every awaiter, so a cancellation
        of the awaiting request or a racing close() cannot abandon it or spawn a second judge
        call. Fail-closed: an evaluator timeout/crash yields a `finalize_error` verdict
        (``correct=False``) rather than propagating."""
        step = self._step + 1
        session = self._sessions.get(tool_name)
        finalize_error = False
        try:
            if session is None:  # unreachable for a real score tool; fail closed anyway
                raise RuntimeError(f"no session for score-terminal tool {tool_name!r}")
            dispatch = session.call_tool(tool_name, args, tool_call_id=f"call-{step}")
            if self._finalize_deadline is not None:
                # `asyncio.shield` inside wait_for: the judge dispatch must not be cancelled
                # by the deadline mid-flight (that could leave the in-process client wedged);
                # the deadline only bounds how long we *await* it before failing closed.
                result = await asyncio.wait_for(
                    asyncio.shield(dispatch), timeout=self._finalize_deadline
                )
            else:
                result = await dispatch
            content = result.result
        except (asyncio.TimeoutError, Exception):
            # Evaluator timeout or crash: synthesize a core-owned, marked fail-closed grade
            # so the (unchanged, marker-based) verifier scores `correct=False` + judge_error.
            finalize_error = True
            content = json.dumps(
                {
                    "hle_grade": True,
                    "correct": False,
                    "confidence": args.get("confidence", 100),
                    "judged_by": "llm_judge_error",
                }
            )

        # Commit evidence atomically under the lifecycle lock (FINALIZING -> FINALIZED).
        async with self._lock:
            self._step = step
            self._trajectory.append(
                Step(index=step, tool=tool_name, arguments=args, result=content)
            )
            self._terminated = True
            feedback = self._env.verify(self._trajectory, self._task, terminated=True)
            items = [*feedback.inference, *feedback.episode]
            self._terminal_feedback = [dump_item(item) for item in items]
            if self._trace_path is not None:
                # Old trace format (RFC 009 prototype keeps it): one terminal step row.
                append_trace(
                    self._trace_path,
                    step_record(
                        session_id=self._session_id,
                        env_name=self._env_name,
                        task_id=self._task_id,
                        step=step,
                        tool=tool_name,
                        feedback=items,
                        terminated=True,
                    ),
                )
            public = self._sanitize_terminal(items, finalize_error=finalize_error)
            self._evidence = public
            self._state = LifecycleState.FINALIZED

        # Teardown is owned by the finalizer and runs after evidence is committed.
        await self._teardown()

        inband = select_inband(items, terminal=True, surface_inference=False)
        return CallResult(
            content=json.dumps(public),
            meta=build_meta(inband, terminate=True),
            terminated=True,
        )

    def _sanitize_terminal(
        self, items: List[Any], *, finalize_error: bool
    ) -> Dict[str, Any]:
        """Build the public-safe terminal payload: the score + a ``judge_error`` flag only.

        Deliberately drops the judge's ``reasoning`` / ``extracted_answer`` / ``judged_by``
        and any exception text — those are answer oracles. ``judge_error`` unions a judge
        failure (surfaced by the verifier as ``judge_error``) with a finalize-level
        timeout/crash, so a fail-closed zero is distinguishable in audit data from an honest
        wrong answer without exposing an oracle.

        This payload is a **read-only projection** for the agent — NOT the trust source.
        Credit derives from the authoritative, server-produced marked grade recorded on the
        sealed ``submit_answer`` step of the trajectory (which the agent never sees and
        cannot forge — the agent controls only args, results are server-owned, and post-seal
        calls record no step); ``_verify`` reads that, never this. (Phase 1 replaces the
        marker-JSON trust source with core-owned protected ``TerminalEvidence``; the
        prototype keeps HLE's marker-based verifier but keeps the authoritative object
        distinct from this sanitized view.)"""
        by_name = {getattr(i, "name", None): getattr(i, "value", None) for i in items}
        judge_error = bool(finalize_error or by_name.get("judge_error"))
        return {"correct": bool(by_name.get("correct")), "judge_error": judge_error}

    def _sealed_tombstone(self) -> CallResult:
        """The generic post-seal tombstone: no inward dispatch, no verdict re-exposed. Used
        for every `tools/call` (repeat score tool, terminate, unknown tool) after the seal."""
        return CallResult(
            content="<episode sealed; no further tool calls are dispatched>",
            meta=build_meta(terminate=True),
            terminated=True,
        )

    async def _teardown(self) -> None:
        """Drop the env's per-session state (HLE's judge/question). Idempotent, runs exactly
        once. Owned by the finalizer (after evidence); close() routes through it too. The MCP
        client sessions are disposed by :meth:`close` in its own task (anyio requires the
        client to be exited in the task that entered it), so this touches only pure-Python
        env state and is safe to run from the finalization background task."""
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
        # RFC 009 (B3): close() participates in the lifecycle. If a score-terminal
        # finalization is in flight, WAIT for it to commit evidence + tear down before
        # disposing anything (so the judge's live session isn't reclaimed mid-finalize);
        # otherwise atomically claim an abort and own teardown. Only then dispose the MCP
        # client sessions and let the env drop any residual state.
        async with self._lock:
            finalization = self._finalization
            if finalization is None and self._state is LifecycleState.OPEN:
                # No seal happened (ordinary env, or a score env closed before submitting):
                # claim an abort finalization so this close owns teardown.
                self._state = LifecycleState.SEALED
        if finalization is not None:
            # Shielded: close() must not cancel the single in-flight judge run.
            try:
                await asyncio.shield(finalization)
            except Exception:
                pass
        await self._teardown()  # idempotent: a no-op if the finalizer already ran it

        # Dispose the MCP client sessions in close()'s own task, then let the env release
        # any remaining per-session state (idempotent with the teardown above).
        for session in self._opened:
            try:
                await session.close()
            except Exception:
                pass
        await self._env.close()
