"""``yc_bench`` on the env-as-center core (RFC 008): a wrap of YC-Bench.

YC-Bench puts an agent in charge of a simulated AI startup for one year. Starting with
$200,000, the agent issues ``yc-bench`` CLI commands against a deterministic, SQLite-backed
discrete-event simulation — accepting tasks from a marketplace, assigning employees,
advancing the clock with ``sim resume``, and managing cash flow — until bankruptcy (funds < 0)
or the one-year horizon. The score is how the company ends up.

YC-Bench ships its own LLM agent loop (``agent/loop.py``, driven by ``runner/main.py``) over a
sim built to take an external driver: something has to advance the clock, feed CLI results
back, and collect the next commands. shogym's harness is that driver. The sim engine, CLI entry
point and command validation, SQLite state/ORM, and ``_init_simulation`` seeding are reused
verbatim (via :mod:`shogym.envs.yc_bench.adapter`); the agent loop is replaced, and shogym
supplies the command, terminal and scoring layers around them (see :func:`score_verdict`).

This module imports **nothing** from ``yc_bench`` at load time, so ``import shogym`` (which
imports this module to register the env) stays offline. yc-bench is imported lazily — only
when the env is *constructed* (its in-process MCP server is probed) or *served*.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from shogym.core import Env
from shogym.envs.registration import register
from shogym.mcp import MCPServerSpec
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

if TYPE_CHECKING:
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence

# The `submit` score terminal (duplicated so this module needn't import yc_bench). Its call is
# sealed by the serve layer and scored from core-owned evidence, never from tool output.
SUBMIT_TOOL_NAME = "submit"

# One shogym step per yc-bench command. A full solvent year is a few hundred commands; the
# default budget is generous so a default episode won't hit the cap before the sim terminates.
DEFAULT_MAX_COMMANDS = 4000

# Fidelity defaults — match a plain ``yc-bench run`` so a fixed seed reproduces upstream.
DEFAULT_CONFIG = "default"
DEFAULT_START_DATE = "2025-01-01"
DEFAULT_COMPANY_NAME = "BenchCo"

# Deterministic, disjoint train/test seed banks. A task *is* a seed: it selects the market
# tasks generated for the world (employees/clients are fixed across seeds upstream), so a seed
# fully reproduces an instance. Seeds are small positive ints (upstream uses 1, 2, 3, …); the
# test bank is offset far away so the two never overlap.
_TRAIN_SEEDS: List[int] = list(range(1, 17))  # 16 tasks
_TEST_SEEDS: List[int] = list(range(9001, 9017))  # 16 held-out tasks

_INSTRUCTIONS = """\
You are the CEO of an AI startup in YC-Bench, a one-year business simulation. You begin with \
$200,000. Your goal: maximize the company's funds while avoiding bankruptcy (funds < 0) before \
the one-year horizon ends.

You act by issuing YC-Bench CLI commands through the `run_command` tool — pass the full command \
string, e.g. `run_command(command="yc-bench company status")`. Every command returns JSON.

# Core loop (repeat)
1. `yc-bench market browse` — see available tasks (client, reward, domain, work required).
2. `yc-bench task accept --task-id Task-42` — accept a task (starts its deadline).
3. `yc-bench task assign --task-id Task-42 --employees Emp_1,Emp_4` — assign employees \
(check `yc-bench employee list` for per-domain skill rates).
4. `yc-bench task dispatch --task-id Task-42` — begin work (requires an assignment).
5. `yc-bench sim resume` — advance the clock to the next event. Requires at least one active \
task. The result reports events processed, payroll deducted, funds delta, and whether the run \
hit bankruptcy or the horizon.

Run several tasks concurrently. Employees split their rate across their active tasks.

# Key mechanics
- Completing a task before its deadline pays its reward and raises prestige; missing a deadline \
costs a prestige penalty and 35% of the reward.
- Every completed task bumps the salary of each assigned employee, so payroll grows over time — \
assigning more employees grows payroll faster.
- Payroll is deducted monthly; if funds go below 0 you are bankrupt and the run ends.
- Clients build trust as you complete their tasks (less work required, better tasks unlocked), \
but some clients are adversarial: after acceptance they inflate the work so deadlines fail. \
Check `yc-bench client history` for failure patterns.
- Your conversation history is limited; use `yc-bench scratchpad write --content "..."` to \
persist strategy notes.

# Observe commands
`yc-bench company status` · `yc-bench employee list` · `yc-bench market browse` · \
`yc-bench task list` · `yc-bench task inspect --task-id T` · `yc-bench client list` · \
`yc-bench client history` · `yc-bench finance ledger`

# Finishing
When the run is over — `sim resume` reports bankruptcy or horizon end, or you choose to stop — \
call `submit`. That ends the episode and records your final result (funds, survival, task \
outcomes) read straight off the sim; there is no separate stop step. Submitting while the \
company is still solvent *before* the horizon scores zero, so only submit once the run has \
actually ended."""


class YcBenchEnv(Env):
    """YC-Bench wrapped as an shogym env.

    Config (all optional, via ``shogym.make("yc_bench", config=...)`` / ``env_config``):
      - ``task_split``: ``"train"`` (default) or ``"test"`` — selects the seed bank.
      - ``config_name``: YC-Bench preset name or ``.toml`` path (default ``"default"``).
      - ``max_commands``: the *advertised* command budget per episode. The shogym horizon is
        ``max_commands + 1`` and the command that reaches it is dispatched before it seals, so
        ``max_commands + 1`` ordinary commands can execute; ``submit`` is handled before the
        horizon is consulted and is never preempted.
      - ``horizon_years``: sim horizon (default: the preset's ``sim.horizon_years``).
      - ``start_date`` / ``company_name``: seeding parameters (defaults match ``yc-bench run``).
      - ``command_timeout_seconds``: per-command wall-clock budget.
    """

    function_name = "run_command"
    # `submit` is the env's single `score` terminal: the serve layer validates its call, seals
    # the episode, then runs `finalize` (below) to score off the sim's frozen end-state — so an
    # authoritative verdict only ever exists for an already-sealed, un-continuable episode.
    score_terminal_tool = SUBMIT_TOOL_NAME

    def __init__(
        self,
        task_split: str = "train",
        config_name: str = DEFAULT_CONFIG,
        max_commands: int = DEFAULT_MAX_COMMANDS,
        horizon_years: Optional[int] = None,
        start_date: str = DEFAULT_START_DATE,
        company_name: str = DEFAULT_COMPANY_NAME,
        command_timeout_seconds: float = 60.0,
    ) -> None:
        if task_split not in ("train", "test"):
            raise ValueError(
                f"unknown task_split {task_split!r}; expected 'train' or 'test'"
            )
        self._task_split = task_split
        self._config_name = config_name
        self._horizon_years = horizon_years
        self._start_date = start_date
        self._company_name = company_name
        self._command_timeout_seconds = command_timeout_seconds
        self._seeds = _TRAIN_SEEDS if task_split == "train" else _TEST_SEEDS
        self.function = FunctionConfig(example_system_template=_INSTRUCTIONS)
        self.mcp_servers = (
            MCPServerSpec(
                name="yc_bench",
                transport="in_process",
                module="shogym.envs.yc_bench.mcp_server",
            ),
        )
        # Horizon: one step per command, plus a single reserved slot for the terminal call.
        # `submit` (the `score` terminal) and `terminate` (the `abort` terminal) are intercepted
        # by the serve layer *before* the horizon check, so neither consumes a counted step — the
        # budget is exactly `max_commands` non-terminal `run_command` steps, after which the agent
        # can still `submit`. The `+ 1` keeps a policy that runs the full command budget from
        # having its terminal `submit` preempted by the horizon; a policy that instead keeps
        # issuing commands hits the horizon on its `max_commands + 1`-th call.
        super().__init__(horizon=max_commands + 1, num_tasks=len(self._seeds))

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._seeds)))
        if task_idx >= len(self._seeds):
            raise ValueError(
                f"Task index {task_idx} is out of range for {len(self._seeds)} tasks"
            )
        return {
            "task_idx": task_idx,
            "seed": self._seeds[task_idx],
            "config_name": self._config_name,
            "start_date": self._start_date,
            "horizon_years": self._horizon_years,
            "company_name": self._company_name,
            "split": self._task_split,
        }

    # ----- session lifecycle -----

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        from shogym.envs.yc_bench import mcp_server  # lazy: pulls in yc_bench

        mcp_server.begin_session(
            session_id,
            seed=task["seed"],
            config_name=task["config_name"],
            start_date=task["start_date"],
            horizon_years=task["horizon_years"],
            company_name=task["company_name"],
            command_timeout_seconds=self._command_timeout_seconds,
        )

    def _end_session(self, session_id: str) -> None:
        from shogym.envs.yc_bench import mcp_server

        mcp_server.end_session(session_id)

    # ----- describe -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        spec = super().describe(task_id)
        seed = self._resolve_seed(task_id)
        if seed is None:
            return spec
        parts = [
            _INSTRUCTIONS,
            "",
            "# This run",
            f"- seed: {seed}",
            f"- config: {self._config_name}",
            f"- split: {self._task_split}",
        ]
        return spec.model_copy(update={"instructions": "\n".join(parts)})

    def _resolve_seed(self, task_id: Optional[str]) -> Optional[int]:
        if task_id is None:
            return None
        try:
            idx = int(task_id)
        except (TypeError, ValueError):
            return None
        if 0 <= idx < len(self._seeds):
            return self._seeds[idx]
        return None

    # ----- finalize + verify (seal-before-verdict) -----

    # The base declares `finalize` as an `Optional[Callable]` attribute (default None) precisely
    # so a score-terminal env overrides it with this coroutine; the variable→method override is
    # the intended opt-in, so silence pyright's structural mismatch.
    async def finalize(  # type: ignore[reportIncompatibleVariableOverride]
        self, req: "FinalizeRequest"
    ) -> "TerminalEvidence":
        """Score the sealed episode off YC-Bench's authoritative sim end-state.

        Runs on the already-sealed episode (the serve layer froze it before this hook fires) and
        reads the sim's final metrics straight off the **live** SQLite DB through the session,
        returning them as core-owned :class:`TerminalEvidence`. The serve layer disposes the
        session (``engine.dispose()`` / DB teardown) only *after* this returns, so the read here
        always sees an open DB. The verdict is the sim's own end-state — the agent's company
        metrics, public-safe with no oracle; ``verify`` applies the terminal-state gate when it
        scores it, and core stamps the provenance. Keyless and in-process. Determinism is
        narrower than it looks: the seeded *business attributes* follow the seed, and the tested
        property is that one seed and one command sequence end on the same funds. Row ids are
        ``uuid4``, they reach the agent through ``sim resume`` wake events, and they break ties
        between simultaneous events, so the trajectory itself is not fixed by the seed.
        """
        from shogym.envs.yc_bench import mcp_server  # lazy: pulls in yc_bench
        from shogym.serve.lifecycle import TerminalEvidence

        verdict = mcp_server.read_verdict(req.session_id)
        return TerminalEvidence(source=req.source, status="ok", verdict=verdict)

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: "Optional[TerminalEvidence]" = None,
    ) -> FeedbackCollection:
        """Score the episode from the core-owned terminal ``evidence`` (never from tool output).

        The authoritative metrics are read server-side by ``finalize`` against the sim's frozen
        final state and handed here as trusted evidence; this scores from ``evidence.verdict``.
        Scoring is defensive — a missing / non-terminal verdict scores a premature, non-surviving
        zero rather than raising — and preserves YC-Bench's terminal-state gate: a solvent state
        is credited ONLY on a genuine ``horizon_end`` / bankruptcy, never on a manual or
        pre-horizon submission."""
        verdict = evidence.verdict if evidence is not None else None
        return score_verdict(verdict, terminated=terminated)


# ----- pure scoring (module-level so it is unit-testable without yc_bench installed) -----


# The sim's terminal states, from ``read_final_state`` / ``sim resume``. Only these two are a
# *genuine* end of the one-year run — anything else means the agent stopped early.
_TERMINAL_REASONS = ("horizon_end", "bankruptcy")


def score_verdict(
    verdict: Optional[Dict[str, Any]], *, terminated: bool
) -> FeedbackCollection:
    """Build episode feedback from the core-owned terminal verdict. Pure.

    ``verdict`` is the sim's final end-state (``finalize`` read it off the DB), or ``None`` when
    the episode ended with no scoring submission (an abort, or a fail-closed finalize)."""
    fb = FeedbackCollection()
    if not terminated:
        return fb

    # The terminal-state gate (YC-Bench fidelity): `submit` is callable at any time, so a
    # solvent end-state alone is not enough — an agent could seal on turn one and bank the
    # starting $200k without operating the company. Credit the final funds *only* when the sim
    # actually reached a terminal state — the horizon (a legitimate completion) or bankruptcy (a
    # genuine, negative-funds outcome). A solvent, pre-horizon seal is premature and scores zero,
    # exactly like a missing verdict.
    if verdict is None or verdict.get("terminal_reason") not in _TERMINAL_REASONS:
        fb.episode.append(EpisodeFeedback(name="reward", value=0.0))
        fb.episode.append(EpisodeFeedback(name="survived", value=False))
        fb.episode.append(EpisodeFeedback(name="success", value=False))
        fb.episode.append(EpisodeFeedback(name="final_funds_cents", value=0.0))
        fb.episode.append(EpisodeFeedback(name="horizon_reached", value=False))
        return fb

    funds = _as_float(verdict.get("final_funds_cents"))
    survived = bool(verdict.get("survived"))
    horizon_reached = bool(verdict.get("horizon_reached"))
    # The benchmark objective: finish the year solvent with as much cash as possible.
    fb.episode.append(EpisodeFeedback(name="reward", value=funds))
    fb.episode.append(EpisodeFeedback(name="final_funds_cents", value=funds))
    fb.episode.append(EpisodeFeedback(name="survived", value=survived))
    fb.episode.append(EpisodeFeedback(name="horizon_reached", value=horizon_reached))
    fb.episode.append(
        EpisodeFeedback(name="success", value=survived and horizon_reached)
    )
    fb.episode.append(
        EpisodeFeedback(
            name="tasks_succeeded", value=_as_float(verdict.get("tasks_succeeded"))
        )
    )
    fb.episode.append(
        EpisodeFeedback(
            name="tasks_failed", value=_as_float(verdict.get("tasks_failed"))
        )
    )
    return fb


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


@register("yc_bench")
class YcBenchDefault(YcBenchEnv):
    """YC-Bench's canonical ``default`` preset (1-year horizon)."""
