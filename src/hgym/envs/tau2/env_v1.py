"""``tau2_<domain>`` on the env-as-center core (RFC 008): a faithful wrap of tau2-bench.

The env declares one in-process MCP server (a per-domain module that hosts tau2's
Orchestrator via its ``GymAgent`` bridge), loads a tau2 task per episode, and verifies the
recorded trajectory by parsing tau2's evaluator verdict off the terminal ``done`` step.

Fidelity: tau2's Orchestrator, user simulator, domain tools/tasks, and evaluator are reused
verbatim; only the *agent* is replaced (by the harness, through the bridge). See
``mcp_server`` for the control-inversion details.

This module imports **nothing** from ``tau2`` at load time, so ``import hgym`` (which imports
this module to register the envs) stays offline. tau2 is imported lazily inside the methods
that need it — all of which run only when the env is *constructed* or *served*.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from hgym.envs.registration import register
from hgym.envs.tool_using_env import ToolUsingEnv
from hgym.mcp import MCPServerSpec
from hgym.task import TaskSpec
from hgym.trajectory import Trajectory
from hgym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

# Kept in sync with mcp_server (duplicated here so this module needn't import tau2). The
# verdict is only trusted off the `done` step — the sole tool that runs tau2's evaluator.
VERDICT_MARKER = "tau2_verdict"
DONE_TOOL_NAME = "done"

# Match upstream tau2.gym.AgentGymEnv's default step budget (a default hgym episode should
# not hit MAX_STEPS earlier than a default `tau2 run`).
DEFAULT_MAX_STEPS = 100

_BASE_INSTRUCTIONS_SOLO = (
    "You are an agent operating a tau2-bench domain through its tools.\n"
    "Follow the domain policy exactly. Take one tool action per step.\n"
    "When the task is complete, call `done` to finish (its result reports the score), "
    "then call `terminate` to end the episode."
)

_BASE_INSTRUCTIONS_USER = (
    "You are a customer-service agent operating a tau2-bench domain through its tools.\n"
    "Follow the domain policy exactly. Take one action per step: either call a tool, or call "
    "`send_message` to talk to the user (its result is the user's reply).\n"
    "When the task is complete, call `done` to finish (its result reports the score), "
    "then call `terminate` to end the episode."
)


class Tau2Env(ToolUsingEnv):
    """Base for tau2 domain envs. Subclasses set ``domain``, ``server_module``, and (for
    non-solo domains) ``solo_mode = False``.

    Config (all optional, via ``hgym.make(..., config=...)`` / ``env_config``):
      - ``task_split``: ``"train"`` (default) or ``"test"``.
      - ``max_steps``: tau2 Orchestrator step budget.
      - ``user_llm`` / ``user_llm_args``: the user-simulator model + kwargs (non-solo only).
        Pass ``user_llm_args={"mock_response": "..."}`` for a deterministic, **offline**
        user simulator (litellm returns the fixed text without a network call).
      - ``evaluation_type``: which tau2 evaluator to run at ``done`` (default ``"all"``).
        Use an offline-safe type (e.g. ``"env"``) for domains whose ``reward_basis`` includes
        NL assertions (retail, banking) when no judge LLM / key is available.
    """

    domain: str = ""
    solo_mode: bool = True
    server_module: str = ""
    # Fixed per-domain env kwargs passed to tau2's env constructor + evaluator (e.g. banking's
    # offline retrieval variant). Not user-configurable — must match the published manifest.
    env_kwargs: Dict[str, Any] = {}
    # Whether the domain declares tau2 train/test splits. True: `task_split` selects the
    # declared named split, and an unsupported value is rejected. False: the domain has no
    # holdout (mock declares only `base`; banking ships no split file), so both splits are the
    # full task set.
    has_canonical_split: bool = True
    function_name = "agent"

    def __init__(
        self,
        task_split: str = "train",
        max_steps: int = DEFAULT_MAX_STEPS,
        user_llm: Optional[str] = None,
        user_llm_args: Optional[Dict[str, Any]] = None,
        evaluation_type: str = "all",
    ) -> None:
        if not self.domain or not self.server_module:
            raise ValueError("Tau2Env subclasses must set `domain` and `server_module`")
        self._task_split = task_split
        self._max_steps = max_steps
        self._user_llm = user_llm
        self._user_llm_args = user_llm_args
        self._evaluation_type = evaluation_type
        self.function = FunctionConfig(example_system_template=self._base_instructions())
        # Instance-level so `essential_specs()` / `_probe_manifest()` (called by super's
        # __init__) see the right in-process server for this domain.
        self.mcp_servers = (
            MCPServerSpec(
                name=f"tau2_{self.domain}",
                transport="in_process",
                module=self.server_module,
            ),
        )
        self._task_ids: List[str] = self._load_task_ids(task_split)
        # Horizon: one hgym step per tau2 agent turn, plus room for `done` + `terminate`.
        super().__init__(horizon=max_steps + 2, num_tasks=len(self._task_ids))

    def _base_instructions(self) -> str:
        return _BASE_INSTRUCTIONS_SOLO if self.solo_mode else _BASE_INSTRUCTIONS_USER

    # ----- task loading -----

    def _load_task_ids(self, split: str) -> List[str]:
        from hgym.envs.tau2 import mcp_server  # lazy: pulls in tau2

        # Honor tau2's *declared* train/test split (no positional slicing): this keeps the
        # benchmark population and held-out set exact. Domains without a declared holdout
        # (mock, banking) use the full task set for both splits.
        if not self.has_canonical_split:
            return [task.id for task in mcp_server.load_tasks(self.domain)]
        if split not in ("train", "test"):
            raise ValueError(
                f"unknown task_split {split!r} for {self.domain!r}; expected 'train' or 'test'"
            )
        # Raises if tau2 doesn't declare this split — no silent fall-back to the leaky full set.
        return [task.id for task in mcp_server.load_tasks(self.domain, split)]

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._task_ids)))
        if task_idx >= len(self._task_ids):
            raise ValueError(
                f"Task index {task_idx} is out of range for {len(self._task_ids)} tasks"
            )
        return {
            "task_idx": task_idx,
            "task_id": self._task_ids[task_idx],
            "domain": self.domain,
            "split": self._task_split,
            "solo_mode": self.solo_mode,
        }

    # ----- session lifecycle -----

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        from hgym.envs.tau2 import mcp_server

        mcp_server.begin_session(
            session_id,
            domain=task["domain"],
            task_id=task["task_id"],
            solo_mode=self.solo_mode,
            max_steps=self._max_steps,
            user_llm=self._user_llm,
            user_llm_args=self._user_llm_args,
            evaluation_type=self._evaluation_type,
            env_kwargs=dict(self.env_kwargs),
        )

    def _end_session(self, session_id: str) -> None:
        from hgym.envs.tau2 import mcp_server

        mcp_server.end_session(session_id)

    # ----- describe: surface the domain policy + this task's ticket -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        spec = super().describe(task_id)
        from hgym.envs.tau2 import mcp_server

        policy = mcp_server.get_policy(self.domain, self.solo_mode, dict(self.env_kwargs))
        parts = [self._base_instructions(), "", "# Domain policy", policy]
        resolved = self._resolve_task_id(task_id)
        if resolved is not None:
            ticket = mcp_server.get_ticket(self.domain, resolved)
            if ticket:
                parts += ["", "# Task", ticket]
        return spec.model_copy(update={"instructions": "\n".join(parts)})

    def _resolve_task_id(self, task_id: Optional[str]) -> Optional[str]:
        """Map the (stringified index) ``task_id`` the serve layer passes to a tau2 task id."""
        if task_id is None:
            return None
        try:
            idx = int(task_id)
        except (TypeError, ValueError):
            return task_id if task_id in self._task_ids else None
        if 0 <= idx < len(self._task_ids):
            return self._task_ids[idx]
        return None

    # ----- verify: parse tau2's evaluator verdict off the recorded trajectory -----

    def _verify(
        self, trajectory: Trajectory, task: Dict[str, Any], *, terminated: bool
    ) -> FeedbackCollection:
        """Score the episode from tau2's verdict, recorded on the ``done`` step.

        Unlike wordle (which re-derives the score from the recorded arguments), the
        authoritative score here is tau2's own evaluator, run server-side against tau2's
        final environment state — hgym cannot re-run it from the flat trajectory alone. So
        ``_verify`` *parses* the verdict the ``done`` tool emitted, exactly as the issue
        specifies. Parsing is defensive: a missing, malformed, or forged verdict yields a
        zero/premature score rather than raising (mirroring wordle's untrusted-result
        handling)."""
        return score_trajectory(trajectory, terminated=terminated)


# ----- pure scoring (module-level so it is unit-testable without tau2 installed) -----


def score_trajectory(
    trajectory: Trajectory, *, terminated: bool
) -> FeedbackCollection:
    """Build episode feedback from tau2's verdict recorded on the ``done`` step. Pure."""
    fb = FeedbackCollection()
    if not terminated:
        return fb

    verdict = _find_verdict(trajectory)
    if verdict is None:
        # No `done` verdict recorded — the harness ended the episode before tau2
        # completed. tau2 scores a premature termination as reward 0.
        fb.episode.append(EpisodeFeedback(name="reward", value=0.0))
        fb.episode.append(EpisodeFeedback(name="success", value=False))
        return fb

    reward = _as_float(verdict.get("reward"))
    fb.episode.append(EpisodeFeedback(name="reward", value=reward))
    fb.episode.append(EpisodeFeedback(name="success", value=reward >= 1.0))
    db_match = verdict.get("db_match")
    if isinstance(db_match, bool):
        fb.episode.append(EpisodeFeedback(name="db_match", value=db_match))
    amp = verdict.get("action_match_proportion")
    if isinstance(amp, (int, float)) and not isinstance(amp, bool):
        fb.episode.append(
            EpisodeFeedback(name="action_match_proportion", value=float(amp))
        )
    return fb


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _find_verdict(trajectory: Trajectory) -> Optional[Dict[str, Any]]:
    """Return the most recent tau2 verdict recorded in the trajectory, or None.

    Only a ``done`` step is trusted: ``done`` is the one tool whose body runs tau2's
    evaluator, so a marked verdict on any *other* recorded result (e.g. a forged
    ``create_task`` output) must not grant terminal credit. Scans results from the end for a
    ``done`` step carrying a JSON object with ``VERDICT_MARKER``; any non-JSON / non-object /
    unmarked / non-``done`` result is skipped, never raised on."""
    for step in reversed(trajectory):
        if step.tool != DONE_TOOL_NAME:
            continue
        try:
            payload = json.loads(step.result)
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get(VERDICT_MARKER) is True:
            return payload
    return None


@register("tau2_mock")
class Tau2MockEnv(Tau2Env):
    """tau2's ``mock`` domain (solo mode) — the smallest faithful, fully-offline slice."""

    domain = "mock"
    solo_mode = True
    server_module = "hgym.envs.tau2.mock_server"
    has_canonical_split = False  # mock declares only a `base` split — no train/test holdout


@register("tau2_airline")
class Tau2AirlineEnv(Tau2Env):
    """tau2's ``airline`` domain (non-solo; user simulator). reward_basis = DB + COMMUNICATE,
    both scored offline — so a ``mock_response`` user makes the whole slice offline."""

    domain = "airline"
    solo_mode = False
    server_module = "hgym.envs.tau2.airline_server"


@register("tau2_retail")
class Tau2RetailEnv(Tau2Env):
    """tau2's ``retail`` domain (non-solo). reward_basis includes NL_ASSERTION, so the *full*
    reward needs a judge LLM (keyed). The DB component is scored offline — pass
    ``evaluation_type="env"`` for an offline served run."""

    domain = "retail"
    solo_mode = False
    server_module = "hgym.envs.tau2.retail_server"


@register("tau2_telecom")
class Tau2TelecomEnv(Tau2Env):
    """tau2's ``telecom`` domain (non-solo; user operates device tools). reward_basis =
    ACTION + ENV_ASSERTION, both offline — the primary keyed-fidelity target."""

    domain = "telecom"
    solo_mode = False
    server_module = "hgym.envs.tau2.telecom_server"


@register("tau2_banking_knowledge")
class Tau2BankingKnowledgeEnv(Tau2Env):
    """tau2's ``banking_knowledge`` domain (non-solo). Pinned to the offline ``bm25_grep``
    retrieval variant so the env constructs/serves without OpenAI embeddings; the benchmark
    default (``alltools``, dense embeddings) is a keyed follow-up. reward_basis includes
    NL_ASSERTION — use ``evaluation_type="env"`` for an offline served run."""

    domain = "banking_knowledge"
    solo_mode = False
    server_module = "hgym.envs.tau2.banking_knowledge_server"
    env_kwargs = {"retrieval_variant": "bm25_grep"}
    has_canonical_split = False  # banking ships no split file — no train/test holdout
