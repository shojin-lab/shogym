"""``automationbench`` on the env-as-center core (RFC 008): a faithful wrap of AutomationBench.

AutomationBench (Zapier, MIT) drops an agent into a fully **simulated** world of ~47 SaaS apps
and asks it to carry out a realistic cross-application business workflow described in natural
language; scoring checks — programmatically, **end-state only** — whether the right data landed
in the right systems. No LLM judge, no live SaaS, no network.

Upstream ships its own agent loop (a Prime Intellect ``verifiers`` ``StatefulToolEnv``). This
port throws that away and keeps the three deterministic, offline pieces verbatim (see
:mod:`shogym.envs.automationbench.adapter`): the simulated tools + ``WorldState`` engine, the typed
task defs, and the pure rubric. shogym's harness *is* the agent loop:

  - **describe** — a :class:`TaskSpec` from the task dict: instructions from the task ``prompt``,
    the pinned ``api`` tool surface, and the horizon (upstream max-steps default **50**).
  - **serve** — the ``api`` toolset (``api_search`` / ``api_fetch`` / ``base64_encode``) over MCP
    against a per-session ``WorldState``, plus the ``done`` **score terminal**.
  - **finalize + verify** — ``done`` is the ``score`` terminal: the serve layer validates it,
    atomically **seals** the episode, then runs ``finalize``, which scores the sealed
    ``WorldState`` server-side with the reused rubric (``partial_credit`` /
    ``task_completed_correctly``) and returns core-owned :class:`TerminalEvidence` carrying only
    the score numbers. The pure ``_verify`` reads those numbers off the trusted evidence — never
    the trajectory, never an oracle. At the horizon (max-steps with no ``done``) the same
    finalizer scores the *current* partial state; an explicit ``terminate`` is a no-score abort.

This module imports **nothing** from the upstream package at load time, so ``import shogym`` (which
imports this module to register the env) stays offline. The upstream source (provisioned lazily
into a cache — see :mod:`shogym.envs.automationbench.adapter`) + ``datasets`` are pulled in only
when the env is *constructed* (tasks load) or *served*.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from shogym.envs.registration import register
from shogym.envs.tool_using_env import ToolUsingEnv
from shogym.mcp import MCPServerSpec
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

if TYPE_CHECKING:
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence

# The env's `score` terminal. Calling it seals + finalizes the episode (validate -> seal ->
# finalize), so scoring only ever happens on an already-sealed, un-continuable world.
DONE_TOOL_NAME = "done"

# Upstream's max-steps default (auto-bench `--max-steps`, and the "~50 tool-using turns" budget
# the task system prompts advertise). +2 leaves room past the budget so a run can still call
# `done` explicitly rather than being forced to the horizon.
DEFAULT_MAX_STEPS = 50

# Default task selection: the public benchmark's 6 domains (the `public` alias upstream expands
# to sales/marketing/operations/support/finance/hr — the 600 distributed tasks).
DEFAULT_DOMAIN = "public"

AUTOMATIONBENCH_SPEC = MCPServerSpec(
    name="automationbench",
    transport="in_process",
    module="shogym.envs.automationbench.mcp_server",
)

_TOOL_GUIDE = """\
# Tools
You operate the workspace through a small REST-style tool surface (no per-app tools):
- `api_search(query, top_k=5)` — find the endpoint to use. Search by API-native keywords
  ("messages" not "emails", "trash" not "delete"). Returns endpoints with a `url`, `method`,
  parameters, and request-body shape.
- `api_fetch(method, url, params, body)` — call an endpoint by its full `url` (from a search
  result). `params` and `body` are JSON strings. This is the only tool that changes state.
- `base64_encode(text)` — encode an email body to the base64url form Gmail endpoints require.

Always `api_search` for an endpoint before you `api_fetch` it. When the workflow is complete,
call `done` — that ends the episode and scores the final state (there is no second submission
and no need to call `terminate` afterward)."""


@register("automationbench")
class AutomationBenchEnv(ToolUsingEnv):
    """AutomationBench wrapped as an shogym env (the pinned ``api`` toolset).

    Config (all optional, via ``shogym.make("automationbench", config=...)`` / ``env_config``):
      - ``domain``: a domain name (``sales`` / ``marketing`` / ``operations`` / ``support`` /
        ``finance`` / ``hr`` / ``simple``) or the ``public`` alias (default) that expands to the
        six public domains — the 600 distributed tasks.
      - ``tasks``: an explicit list of raw upstream task rows (each ``{"prompt", "info", ...}``,
        ``info`` a dict or JSON string). When given, the domain datasets are **not** loaded —
        this is how offline tests construct the env without ``datasets``.
      - ``max_steps``: the shogym horizon (default 50, upstream's default).
    """

    mcp_servers = (AUTOMATIONBENCH_SPEC,)
    function_name = "agent"
    # `done` is the env's single `score` terminal. The serve layer validates its (empty) args,
    # atomically seals the episode, then runs `finalize` (seal-before-verdict), so a graded
    # verdict only ever exists for an already-sealed, un-continuable world — no read-score-then-fix
    # exploit. Every non-scoring env leaves this `None` and never enters the seal transaction.
    score_terminal_tool = DONE_TOOL_NAME

    def __init__(
        self,
        domain: str = DEFAULT_DOMAIN,
        tasks: Optional[List[Dict[str, Any]]] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        self._domain = domain
        self._max_steps = max_steps
        self._tasks: List[Dict[str, Any]] = (
            [_normalize_row(t) for t in tasks] if tasks is not None else _load_domain(domain)
        )
        self.function = FunctionConfig(example_system_template=_static_instructions())
        # +2 keeps the horizon a hair above the advertised budget so a run that means to submit
        # can still call `done` explicitly; a run that never does hits the horizon, where the same
        # finalizer scores its current (partial) state.
        super().__init__(horizon=max_steps + 2, num_tasks=len(self._tasks))

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if not self._tasks:
            raise ValueError("automationbench env has no tasks loaded")
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._tasks)))
        if not 0 <= task_idx < len(self._tasks):
            raise ValueError(f"Task index {task_idx} is out of range for {len(self._tasks)} tasks")
        row = self._tasks[task_idx]
        return {
            "task_idx": task_idx,
            "example_id": row.get("example_id"),
            "name": str(row.get("task", task_idx)),
            "prompt": row.get("prompt", []),
            "info": row["info"],  # already a dict (normalized on load)
            "domain": self._domain,
        }

    # ----- session lifecycle -----

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        from shogym.envs.automationbench import mcp_server  # lazy: provisions the upstream source

        mcp_server.begin_session(session_id, info=task["info"])

    def _end_session(self, session_id: str) -> None:
        from shogym.envs.automationbench import mcp_server

        mcp_server.end_session(session_id)

    # ----- describe -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        spec = super().describe(task_id)
        row = self._resolve_row(task_id)
        if row is None:
            return spec
        return spec.model_copy(update={"instructions": _render_instructions(row)})

    def _resolve_row(self, task_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if task_id is None:
            return None
        try:
            idx = int(task_id)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(self._tasks):
            return self._tasks[idx]
        return None

    # ----- finalize (seal-before-verdict) -----

    # `Env.finalize` is declared as an opt-in attribute (`Optional[Callable[[FinalizeRequest],
    # Awaitable[TerminalEvidence]]] = None`); a `score`-terminal env opts in by overriding it with
    # this coroutine method — the documented contract (see `shogym.core.Env`). pyright's
    # variable-vs-method override check flags the intentional attribute->method swap, so suppress
    # it here (the same pattern the `_fixture_score` test env uses).
    async def finalize(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, req: "FinalizeRequest"
    ) -> "TerminalEvidence":
        """Score the **sealed** episode's current ``WorldState`` and return core-owned evidence.

        Runs after the serve layer has atomically sealed the episode (on a ``done`` call or at the
        horizon), so the world it scores can no longer be mutated — the read-score-then-fix exploit
        the old one-shot ``done`` guard defended against is now closed *structurally* by the seal.

        Scoring is delegated to :func:`mcp_server.score_session`, which runs AutomationBench's
        reused rubric (``partial_credit`` then ``task_completed_correctly``, including the
        free/negative-assertion "must not shotgun" gate) against this session's private world. The
        horizon path scores the current *partial* state identically to an explicit ``done`` — a run
        that hits max-steps mid-task still earns partial credit for the assertions it satisfied.

        The returned verdict carries **only** the score numbers (``partial_credit`` / ``success``)
        — never the assertions, target values, or world dump, which are answer oracles. The private
        ``diagnostic`` (durable store / server logs only) never reaches the agent. An explicit
        ``terminate`` never reaches here: the serve layer synthesizes a no-score abort verdict for
        it directly.
        """
        from shogym.envs.automationbench import mcp_server
        from shogym.serve.lifecycle import TerminalEvidence

        pc, success = mcp_server.score_session(req.session_id)
        success_pass = _as_unit(success) == 1.0
        return TerminalEvidence(
            source=req.source,
            status="ok",
            # Public-safe: score numbers only. No assertions / targets / world — nothing an agent
            # could act on. `partial_credit` doubles as `reward` when `_verify` reads it back.
            verdict={"partial_credit": _as_unit(pc), "success": success_pass},
            diagnostic=f"scored source={req.source} partial_credit={pc} success={success}",
        )

    # ----- verify -----

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: "Optional[TerminalEvidence]" = None,
    ) -> FeedbackCollection:
        """Score the episode off the core-owned terminal ``evidence`` (never the trajectory).

        The trusted verdict is produced by ``finalize`` on the sealed world and handed here as
        immutable :class:`TerminalEvidence`; this only reads its numbers. Emits ``reward``
        (== ``partial_credit``), ``partial_credit``, and ``success`` (``task_completed_correctly``:
        True iff every scored assertion passed). A missing/abort verdict (no ``partial_credit``,
        e.g. an explicit ``terminate``) coerces to a clean zero. When the finalizer failed closed,
        an extra ``finalize_error`` flag is emitted so infra failures are filterable from honest
        zeros."""
        fb = FeedbackCollection()
        if not terminated:
            return fb

        verdict = evidence.verdict if evidence is not None else {}
        pc = _as_unit(verdict.get("partial_credit"))
        fb.episode.append(EpisodeFeedback(name="reward", value=pc))
        fb.episode.append(EpisodeFeedback(name="partial_credit", value=pc))
        fb.episode.append(
            EpisodeFeedback(name="success", value=_as_unit(verdict.get("success")) == 1.0)
        )
        if evidence is not None and evidence.finalize_error:
            fb.episode.append(EpisodeFeedback(name="finalize_error", value=True))
        return fb


# ----- pure scoring helpers (module-level so they are unit-testable without the upstream package) -----


def _as_unit(value: Any) -> float:
    """Coerce a score to a finite [0, 1] float; junk/absent defaults to 0.0."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if out != out or out in (float("inf"), float("-inf")):  # NaN / inf
        return 0.0
    return max(0.0, min(1.0, out))


# ----- module-level helpers -----


def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a task row so ``info`` is a dict (dataset rows store ``info`` as a JSON string)."""
    out = dict(row)
    info = out.get("info", {})
    if isinstance(info, str):
        info = json.loads(info)
    out["info"] = info
    return out


def _load_domain(domain: str) -> List[Dict[str, Any]]:
    from shogym.envs.automationbench import adapter  # lazy: provisions the upstream source + datasets

    return [_normalize_row(row) for row in adapter.load_domain_tasks(domain)]


def _static_instructions() -> str:
    """The durable, task-independent framing published by ``describe(task_id=None)``."""
    return (
        "You are a workflow automation agent operating a simulated business workspace "
        "(email, CRM, spreadsheets, chat, and other SaaS apps). Carry out the requested "
        "workflow end to end, making reasonable assumptions rather than asking questions.\n\n"
        + _TOOL_GUIDE
    )


def _render_instructions(row: Dict[str, Any]) -> str:
    """Build per-task instructions from the task's chat ``prompt`` + the tool guide."""
    parts: List[str] = []
    for message in row.get("prompt", []) or []:
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        role = str(message.get("role", "")).lower()
        if role == "user":
            parts.append("# Request\n" + content)
        else:
            parts.append(content)
    parts.append(_TOOL_GUIDE)
    return "\n\n".join(parts)
