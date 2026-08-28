# tau2's source is provisioned at runtime into a cache dir (see `shogym.envs.tau2.adapter`); it is
# intentionally absent from the base type-check / offline environment, so its imports are expected
# to be unresolved there.
# pyright: reportMissingImports=false
"""The tau2 control-inversion bridge, hosted as an shogym in-process MCP server.

tau2's Orchestrator *drives the agent*: it asks the agent for its next action, executes
tools, invokes the user simulator, and checks termination. shogym's harness *drives tool
calls*. This module bridges the two by hosting tau2's Orchestrator on a **background thread**
(issue #31) and reusing tau2's own ``GymAgent`` — a ``HalfDuplexAgent`` whose
``generate_next_message`` blocks until an external ``set_action``. Each incoming MCP tool
call becomes the agent's next action:

  - a tau2 **domain tool** → an ``AssistantMessage`` with one tool call; the tool output (or,
    after ``send_message``, the **user-simulator's** reply) is returned as the MCP result;
  - **``send_message``** (non-solo domains) → an ``AssistantMessage`` with text content routed
    to the user simulator; its result is the user's reply;
  - **``done``** → the env's ``score`` terminal. The serve layer **seals** the episode first and
    then runs :meth:`~shogym.envs.tau2.env_v1.Tau2Env.finalize`, which drives tau2's agent-stop,
    lets the Orchestrator finalize, and runs tau2's **evaluator** over tau2's final state exactly
    once (:meth:`_Tau2Session.finalize_once`). The verdict comes back as core-owned
    ``TerminalEvidence`` and the env's pure ``verify`` reads it from there; the trajectory scan is
    kept only as a legacy defensive fallback.

Only the *agent* is replaced; tau2's Orchestrator, user simulator, domains/tools/tasks, and
evaluator are reused verbatim. State is keyed by ``_session_id`` (shogym injects it for tools
that declare it), so several episodes can share this module safely. This is the one module in
shogym that imports ``tau2``; it provisions the pinned upstream source first (see
:mod:`shogym.envs.tau2.adapter` — shogym declares no ``pip`` dependency on tau2-bench), so
importing it needs the ``tau2`` extra and, on a cold cache, the network — but it is *only ever
imported when a tau2 env is constructed or served*, never by ``import shogym``.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.tools import Tool, ToolResult
from loguru import logger
from pydantic import PrivateAttr

from shogym.envs.tau2.adapter import ensure_source

# Bind the pinned upstream source into `sys.modules` before any `tau2` import below resolves.
ensure_source()

from tau2.config import DEFAULT_LLM_ARGS_USER, DEFAULT_LLM_USER  # noqa: E402
from tau2.data_model.message import AssistantMessage, ToolCall  # noqa: E402
from tau2.environment.tool import as_tool  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation  # noqa: E402
from tau2.gym.gym_agent import GymAgent, done as _done_fn  # noqa: E402
from tau2.orchestrator.orchestrator import Orchestrator  # noqa: E402
from tau2.registry import registry  # noqa: E402
from tau2.user.user_simulator import DummyUser, UserSimulator  # noqa: E402

# tau2's registry is dynamically typed (its loader/constructor annotations don't reflect the
# actual call surface — e.g. the `solo_mode`/`retrieval_variant` kwargs), so treat it as `Any`
# at the call sites rather than fighting upstream types. Runtime is exercised by the tests.
_reg: Any = registry

# The stop tool tau2's GymAgent appends; calling it ends the simulation and triggers scoring.
DONE_TOOL_NAME = "done"
# Tool the harness uses to talk to the user simulator (non-solo domains).
SEND_MESSAGE_TOOL_NAME = "send_message"
# Marker key stamped on the `done` verdict so the pure verifier can find it in the recorded
# trajectory without trusting arbitrary tool output.
VERDICT_MARKER = "tau2_verdict"
DEFAULT_EVALUATION_TYPE = "all"

# session_id -> _Tau2Session
_sessions: Dict[str, "_Tau2Session"] = {}
# Guards the `_sessions` map (each session's Orchestrator has its own thread and is driven
# sequentially by one episode).
_lock = threading.RLock()


# ----- tau2 loaders (offline given a tau2 data checkout; banking needs an offline variant) -----


def load_tasks(domain: str, split: Optional[str] = None) -> List[Any]:
    """Return tau2's task objects for ``domain``.

    When ``split`` is given, tau2's **declared** named split is used
    (``get_tasks(task_split_name=...)``), so the benchmark population and held-out set are
    honored exactly — an *unsupported* split raises ``ValueError`` (no silent fall-back to the
    full, leaky set). ``split=None`` returns the full set. Callers that must degrade to the
    full set for domains without a train/test holdout do so explicitly (see
    ``Tau2Env.has_canonical_split``)."""
    loader = _reg.get_tasks_loader(domain)
    if split is None:
        return list(loader())
    return list(loader(task_split_name=split))


def _make_env(domain: str, *, solo_mode: bool, env_kwargs: Dict[str, Any]) -> Any:
    return _reg.get_env_constructor(domain)(solo_mode=solo_mode, **env_kwargs)


def get_policy(domain: str, solo_mode: bool, env_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Return the domain policy text (tau2's agent system prompt)."""
    return _make_env(domain, solo_mode=solo_mode, env_kwargs=env_kwargs or {}).get_policy()


def get_ticket(domain: str, task_id: str) -> Optional[str]:
    """Return the task's solo-mode ticket / user request, if any."""
    for task in load_tasks(domain):
        if task.id == task_id:
            ticket = getattr(task, "ticket", None)
            if ticket:
                return str(ticket)
            scenario = getattr(task, "user_scenario", None)
            instr = getattr(scenario, "instructions", None) if scenario else None
            return str(instr) if instr else None
    return None


def _find_task(domain: str, task_id: str) -> Any:
    for task in load_tasks(domain):
        if task.id == task_id:
            return task
    raise ValueError(f"tau2 task {task_id!r} not found in domain {domain!r}")


# ----- the hosted Orchestrator session -----


class _Tau2Session:
    """One tau2 simulation: a GymAgent + user + environment wired into an Orchestrator that
    runs on a background thread, driven one action at a time by MCP tool calls."""

    def __init__(
        self,
        *,
        domain: str,
        task: Any,
        solo_mode: bool,
        max_steps: int,
        user_llm: str,
        user_llm_args: Optional[Dict[str, Any]],
        evaluation_type: str,
        env_kwargs: Dict[str, Any],
    ) -> None:
        self.domain = domain
        self.task = task
        self.solo_mode = solo_mode
        self.evaluation_type = evaluation_type
        self.env_kwargs = env_kwargs
        self.terminated = False
        self.verdict: Optional[Dict[str, Any]] = None
        # The evaluator's private diagnostic (exception text on an evaluator failure), kept OFF
        # the public verdict — the shogym `finalize` hook forwards it only to the durable store.
        self.eval_error: Optional[str] = None

        self._done = threading.Event()
        self._sim_run: Any = None
        # Serialize (and once-guard) every stop+evaluate against the background Orchestrator.
        # The autonomous-stop detector (an ordinary tool/user turn that drove tau2 to stop) and
        # the shogym seal finalizer both drive termination; this lock + the `terminated` guard make
        # the Orchestrator impossible to double-stop and `evaluate_simulation` run at most once.
        # `abort()` shares it, so an end_session teardown can never race an in-flight finalize.
        self._finalize_lock = threading.Lock()

        environment = _make_env(domain, solo_mode=solo_mode, env_kwargs=env_kwargs)
        agent_tools = list(environment.get_tools())
        user_tools = _safe_user_tools(environment, task)
        if solo_mode:
            # Solo agent operates the user's tools itself; no user simulator.
            agent_tools += list(user_tools or [])
            user: Any = DummyUser()
        else:
            # Match upstream AgentGymEnv's default: when no override is given, use a *copy* of
            # tau2's DEFAULT_LLM_ARGS_USER (which sets temperature=0.0), not an empty dict —
            # otherwise the default user policy would silently diverge from `tau2 run`.
            if user_llm_args is None:
                user_llm_args = deepcopy(DEFAULT_LLM_ARGS_USER)
            user = UserSimulator(
                llm=user_llm,
                instructions=getattr(task, "user_scenario", None),
                tools=user_tools,
                llm_args=user_llm_args,
            )
        self._agent = GymAgent(tools=agent_tools, domain_policy=environment.get_policy())
        self._orch = Orchestrator(
            domain=domain,
            agent=self._agent,
            user=user,
            environment=environment,
            task=task,
            max_steps=max_steps,
            solo_mode=solo_mode,
        )
        self._thread = threading.Thread(target=self._run, daemon=True)

    # -- lifecycle --

    @property
    def orchestrator_done(self) -> bool:
        """True once the Orchestrator thread has finished (terminated or errored)."""
        return self._done.is_set()

    def start(self) -> None:
        self._thread.start()
        self._wait_for_turn()

    def _run(self) -> None:
        try:
            self._sim_run = self._orch.run()
        except Exception as exc:  # keep the thread from dying silently
            logger.error(f"tau2 orchestrator error ({self.domain}): {exc}")
        finally:
            self._done.set()

    def _wait_for_turn(self) -> None:
        # Block until the GymAgent is parked awaiting the next action, or the sim ended.
        while not self._done.is_set() and not self._agent.is_agent_turn:
            self._done.wait(0.01)

    # -- driving actions --

    def act_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Feed a domain tool call as the agent's action; return the tool result string."""
        if self._done.is_set():
            return json.dumps({"error": "episode already terminated"})
        self._agent.set_action(_tool_call_message(tool_name, arguments))
        self._wait_for_turn()
        return self._latest_content()

    def act_message(self, content: str) -> str:
        """Feed a text message to the user simulator; return the user's reply string."""
        if self._done.is_set():
            return json.dumps({"error": "episode already terminated"})
        self._agent.set_action(AssistantMessage(role="assistant", content=content))
        self._wait_for_turn()
        return self._latest_content()

    def finalize_once(self) -> Dict[str, Any]:
        """Atomically finalize tau2 and return the verdict. Idempotent + once-guarded.

        The single stop+score transaction, safe to call from BOTH the autonomous-stop detector
        (an ordinary tool/user turn that already drove tau2 to a stop) and the shogym seal
        finalizer. Under ``_finalize_lock``: if the run is already terminated (an autonomous stop
        stashed a verdict, or a prior finalize ran) return the **stored** outcome — never
        re-stop, never re-evaluate. Otherwise deliver ``done`` to the Orchestrator exactly once
        (unless it already finished), wait for it to unwind, run ``evaluate_simulation`` exactly
        once, and stash the verdict (+ any private diagnostic)."""
        with self._finalize_lock:
            if self.terminated:
                return self.verdict if self.verdict is not None else _empty_verdict()
            if not self._done.is_set() and self._agent.is_agent_turn:
                self._agent.set_action(_tool_call_message(DONE_TOOL_NAME, {}))
            while not self._done.is_set():
                self._done.wait(0.05)
            verdict, diagnostic = self._evaluate()
            self.verdict = verdict
            self.eval_error = diagnostic
            self.terminated = True
            return verdict

    def stop_and_evaluate(self) -> Dict[str, Any]:
        """Signal tau2 agent-stop, wait for the Orchestrator to finalize, and score.

        Retained as the autonomous-stop entry point (see ``dispatch``); delegates to the
        once-guarded :meth:`finalize_once` so a stop detected on the background thread and the
        shogym finalizer can never double-stop the Orchestrator or double-run the evaluator."""
        return self.finalize_once()

    def abort(self) -> None:
        """Best-effort teardown of a still-running simulation (no scoring).

        Shares ``_finalize_lock`` with :meth:`finalize_once`, so an ``end_session`` teardown can
        never drive the agent concurrently with an in-flight finalize (which would double-stop
        the Orchestrator). A run that already terminated/scored is left untouched."""
        with self._finalize_lock:
            if self.terminated or self._done.is_set():
                return
            try:
                if self._agent.is_agent_turn:
                    self._agent.set_action(_tool_call_message(DONE_TOOL_NAME, {}))
            except Exception:
                pass
        # Give the thread a moment to unwind; it's a daemon, so don't block forever. Waited
        # OUTSIDE the lock so it can't stall a (defensively concurrent) finalize.
        self._done.wait(2.0)

    # -- helpers --

    def _latest_content(self) -> str:
        obs = self._agent.observation
        if not obs:
            return ""
        content = getattr(obs[-1], "content", None)
        if isinstance(content, str):
            return content
        return json.dumps(content) if content is not None else ""

    def _evaluate(self) -> "tuple[Dict[str, Any], Optional[str]]":
        """Run tau2's evaluator over the completed simulation.

        Returns ``(public-safe verdict, private diagnostic)``. The exception text of an evaluator
        failure goes ONLY into the diagnostic (a fail-closed reward-0 verdict is returned) — it
        is never written into the public verdict the agent can read, so an evaluator crash leaks
        no oracle. A clean run returns ``(verdict, None)``."""
        if self._sim_run is None:
            return _empty_verdict(), "no completed simulation to evaluate"
        try:
            reward_info = evaluate_simulation(
                simulation=self._sim_run,
                task=self.task,
                evaluation_type=EvaluationType(self.evaluation_type),
                solo_mode=self.solo_mode,
                domain=self.domain,
                env_kwargs=self.env_kwargs,
            )
        except Exception as exc:
            return _empty_verdict(), f"tau2 evaluate_simulation failed: {exc}"
        return _verdict_from_reward_info(reward_info), None


def _safe_user_tools(environment: Any, task: Any) -> Optional[List[Any]]:
    try:
        return environment.get_user_tools(include=getattr(task, "user_tools", None)) or None
    except Exception:
        return None


def _tool_call_message(tool_name: str, arguments: Dict[str, Any]) -> AssistantMessage:
    return AssistantMessage(
        role="assistant",
        tool_calls=[
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:12]}",
                name=tool_name,
                arguments=arguments,
                requestor="assistant",
            )
        ],
    )


def _empty_verdict() -> Dict[str, Any]:
    return {
        VERDICT_MARKER: True,
        "reward": 0.0,
        "db_match": None,
        "action_match_proportion": None,
        "reward_basis": None,
    }


def _verdict_from_reward_info(reward_info: Any) -> Dict[str, Any]:
    """Distil tau2's ``RewardInfo`` into a compact, self-describing verdict."""
    data = reward_info.model_dump() if hasattr(reward_info, "model_dump") else dict(reward_info)
    reward = data.get("reward")
    db_check = data.get("db_check") or {}
    db_match = db_check.get("db_match") if isinstance(db_check, dict) else None
    action_checks = data.get("action_checks") or []
    if action_checks:
        matched = sum(1 for a in action_checks if a.get("action_match"))
        action_match_proportion: Optional[float] = matched / len(action_checks)
    else:
        action_match_proportion = None
    basis = data.get("reward_basis")
    if basis is not None:
        basis = [getattr(b, "value", b) for b in basis]
    return {
        VERDICT_MARKER: True,
        "reward": reward,
        "db_match": db_match if isinstance(db_match, bool) else None,
        "action_match_proportion": action_match_proportion,
        "reward_basis": basis,
    }


# ----- MCP tool surface (built once per domain server module, at import) -----


def _tool_parameters(tau2_tool: Any) -> Dict[str, Any]:
    """MCP input schema for a tau2 tool: its params plus the hidden ``_session_id``."""
    schema = tau2_tool.openai_schema["function"]["parameters"]
    properties = dict(schema.get("properties") or {})
    properties["_session_id"] = {
        "type": "string",
        "description": "Reserved shogym session id (injected by the harness).",
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(schema.get("required") or []),
        "additionalProperties": False,
    }


class _BridgeTool(Tool):
    """An MCP tool that forwards a call to the session's tau2 Orchestrator as the agent's next
    action. Overriding ``run`` lets us advertise tau2's exact JSON-Schema and forward only the
    keys the caller sent."""

    _tau2_name: str = PrivateAttr()
    _kind: str = PrivateAttr()

    def __init__(self, *, tau2_name: str, kind: str, **data: Any) -> None:
        super().__init__(**data)
        self._tau2_name = tau2_name
        self._kind = kind

    async def run(self, arguments: Dict[str, Any]) -> ToolResult:
        # `dispatch` blocks on the Orchestrator thread; run it off the event loop.
        text = await asyncio.to_thread(dispatch, self._tau2_name, self._kind, dict(arguments))
        return ToolResult(content=text)


def _agent_tool_objects(domain: str, solo_mode: bool, env_kwargs: Dict[str, Any]) -> List[Any]:
    """The tau2 ``Tool`` objects the harness may call as domain tools, plus ``done``.

    Mirrors the session's GymAgent tool set (unfiltered by task, since the manifest is
    published once at construction): domain agent tools, plus the user's tools in solo mode
    (the solo agent operates them itself), plus the ``done`` stop tool.
    """
    env = _make_env(domain, solo_mode=solo_mode, env_kwargs=env_kwargs)
    tools: List[Any] = list(env.get_tools())
    if solo_mode:
        try:
            tools += list(env.get_user_tools() or [])
        except Exception:
            pass
    tools.append(as_tool(_done_fn))
    seen: set[str] = set()
    unique: List[Any] = []
    for tool in tools:
        if tool.name not in seen:
            seen.add(tool.name)
            unique.append(tool)
    return unique


def build_domain_server(
    domain: str, *, solo_mode: bool, env_kwargs: Optional[Dict[str, Any]] = None
) -> FastMCP:
    """Build the FastMCP server exposing ``domain``'s tau2 tools (+ ``send_message`` for
    non-solo domains). Called once, at import of a per-domain server module."""
    env_kwargs = env_kwargs or {}
    server: FastMCP = FastMCP(name=f"tau2_{domain}")
    for tau2_tool in _agent_tool_objects(domain, solo_mode, env_kwargs):
        server.add_tool(
            _BridgeTool(
                tau2_name=tau2_tool.name,
                kind="tool",
                name=tau2_tool.name,
                description=tau2_tool.openai_schema["function"].get("description") or "",
                parameters=_tool_parameters(tau2_tool),
            )
        )
    if not solo_mode:
        server.add_tool(
            _BridgeTool(
                tau2_name=SEND_MESSAGE_TOOL_NAME,
                kind="message",
                name=SEND_MESSAGE_TOOL_NAME,
                description=(
                    "Send a natural-language message to the user and return their reply. "
                    "Use this to ask for information or confirm actions with the user."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Message to the user."},
                        "_session_id": {
                            "type": "string",
                            "description": "Reserved shogym session id (injected by the harness).",
                        },
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
            )
        )
    return server


# ----- dispatch + session lifecycle -----


def dispatch(tool_name: str, kind: str, arguments: Dict[str, Any]) -> str:
    """Execute one MCP tool call as the agent's next tau2 action; return the result string."""
    session_id = arguments.pop("_session_id", None)
    with _lock:
        session = _sessions.get(session_id) if session_id is not None else None
    if session is None:
        return json.dumps({"error": "session not initialized; env did not call begin_session"})
    if session.terminated:
        # The run is already scored (an autonomous stop stashed the verdict, or a prior
        # finalize ran). `done` never reaches here on the served path — it's the env's `score`
        # terminal, intercepted and finalized by the serve layer, not dispatched inward — but
        # keep the retrieve-the-stored-verdict branch as a defensive fallback for any direct
        # dispatch. Every other call after termination is a no-op.
        if tool_name == DONE_TOOL_NAME and session.verdict is not None:
            return json.dumps(session.verdict)
        return json.dumps({"error": "episode already terminated"})

    if tool_name == DONE_TOOL_NAME:
        # Defensive: on the served path the serve layer handles `done` via the seal + `finalize`,
        # so this is unreachable there. Retained so a direct in-process dispatch still scores.
        return json.dumps(session.stop_and_evaluate())

    if kind == "message":
        result = session.act_message(str(arguments.get("content", "")))
    else:
        result = session.act_tool(tool_name, arguments)
    # A domain tool call or user message can also drive tau2 to termination on its own
    # (max_steps, or the user simulator ending the conversation). Score it now and STASH the
    # verdict on the background-thread session, but return the tool/user result here. The shogym
    # `finalize` hook (run when the harness reaches a terminal, or at the horizon) later retrieves
    # this stashed verdict via `finalize_once` — so tau2's evaluator score over the completed run
    # is preserved without double-stopping the Orchestrator.
    if session.orchestrator_done and not session.terminated:
        session.stop_and_evaluate()
    return result


def begin_session(
    session_id: str,
    *,
    domain: str,
    task_id: str,
    solo_mode: bool,
    max_steps: int,
    user_llm: Optional[str] = None,
    user_llm_args: Optional[Dict[str, Any]] = None,
    evaluation_type: str = DEFAULT_EVALUATION_TYPE,
    env_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Start a tau2 simulation for ``task_id`` on a background thread and block until the
    agent's first turn."""
    session = _Tau2Session(
        domain=domain,
        task=_find_task(domain, task_id),
        solo_mode=solo_mode,
        max_steps=max_steps,
        user_llm=user_llm or DEFAULT_LLM_USER,
        user_llm_args=user_llm_args,
        evaluation_type=evaluation_type,
        env_kwargs=env_kwargs or {},
    )
    session.start()
    with _lock:
        _sessions[session_id] = session


def finalize_once(session_id: str) -> "tuple[Dict[str, Any], Optional[str]]":
    """Run tau2's atomic finalize for ``session_id`` and return ``(verdict, private diagnostic)``.

    The shogym-side terminal wiring: the env's ``finalize`` hook calls this on the already-sealed
    episode to drive tau2 to a stop (or reuse a stashed autonomous-stop verdict) and score it —
    exactly once (see :meth:`_Tau2Session.finalize_once`). If the session is gone (never begun,
    or already torn down), there is no tau2 run to score: return a premature reward-0 verdict
    with a diagnostic, never raise."""
    with _lock:
        session = _sessions.get(session_id)
    if session is None:
        return (
            _empty_verdict(),
            "tau2 session not found at finalize (never begun or already ended)",
        )
    verdict = session.finalize_once()
    return verdict, session.eval_error


def end_session(session_id: str) -> None:
    """Tear down a session's tau2 simulation. Idempotent.

    Runs *after* the episode is sealed and finalized (``finalize_once`` already scored the run,
    or synthesized no-score abort evidence), so aborting a still-running simulation here cannot
    change the episode's score — it just unblocks the parked Orchestrator thread instead of
    leaking a daemon. A finalized session is already ``terminated`` so ``abort()`` is a no-op;
    the shared ``_finalize_lock`` guarantees this teardown never races an in-flight finalize."""
    with _lock:
        session = _sessions.pop(session_id, None)
    if session is None:
        return
    if not session.terminated:
        session.abort()


def reset_state() -> None:
    """Drop all sessions (test hygiene)."""
    with _lock:
        for session_id in list(_sessions):
            end_session(session_id)
