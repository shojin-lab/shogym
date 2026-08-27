"""``frontier_bench`` on the env-as-center core (RFC 008): a Docker-backed port of Frontier-Bench.

Frontier-Bench (the Harbor team's maintained successor to Terminal-Bench) is the **heaviest**
shogym port type: each task is a Docker *environment* an agent operates through a shell, plus a
programmatic *verifier* that scores the container end-state (never the transcript). Harbor
already draws the boundary shogym wants — task/verifier data vs. agent runner — so this port
drives the served shell as the agent and reuses the task's own verifier for the verdict.

This slice vendors a first batch of the CPU-only, single-container, ``gpus=0``, pytest-verified,
small-bundled-data tasks (issue #48) at upstream commit ``eb4af26c``, plus each task's
``dataset.toml`` digest. More upstream tasks meet those criteria and are deferred; see the env
README. The env exposes them as ``num_tasks``
selectable by **index** (``0..N-1``) or **name**:

  - **describe(task_id)** — return the selected task's ``instruction.md``.
  - **serve** — build+start that task's ``environment/Dockerfile`` container and expose its
    shell as MCP tools (``exec`` / ``read_file`` / ``write_file``) + a ``done`` **score
    terminal**. The external harness is the agent.
  - **finalize (sealed)** — ``done`` is the ``score`` terminal: the serve layer *seals* the
    episode, then this env's :meth:`FrontierBenchEnv.finalize` collects the declared
    ``artifacts`` off the (still-live) agent container, runs the task's verifier in a second
    container, reads ``/logs/verifier/reward.txt``, and returns the core-owned
    :class:`TerminalEvidence`. Only *then* is the container torn down.
  - **verify (pure)** — scores the 0/1 reward straight off ``evidence.verdict`` (no marker
    scanning): the seal makes the reward non-forgeable and one-shot.

This module imports **nothing** Docker-related (nor the serve layer) at load time, so
``import shogym`` (which imports it to register the env) stays offline. Docker is touched only
when an episode is *served* (``begin_session`` builds the container; ``finalize`` runs the
verifier). :func:`score_evidence` is Docker-free and unit-tested without a daemon.
"""

from __future__ import annotations

import asyncio
import math
from typing import TYPE_CHECKING, Any, Dict, Optional

from shogym.core import Env
from shogym.envs.registration import register
from shogym.mcp import MCPServerSpec
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

if TYPE_CHECKING:
    # Only for annotations — imported lazily at runtime inside `finalize` so `import shogym`
    # (which imports this module to register the env) never pulls in the serve layer, avoiding
    # the envs<->serve import cycle (mirrors `shogym.core`).
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence

# The score terminal: calling `done` seals the episode and runs the verifier in `finalize`.
DONE_TOOL_NAME = "done"

# The step budget: shell tasks are long. One shogym step per tool call; the default is generous
# so a real run won't hit the cap before the agent calls `done`.
DEFAULT_MAX_STEPS = 400

_INSTRUCTIONS = """\
You are operating inside a Frontier-Bench task container over a served shell. Call `describe` \
first to read the full task instruction (it specifies the inputs under `/app/inputs/`, the \
exact outputs to produce, and their format).

Act on the container through these tools:
- `exec(command)` — run a shell command in the container (inputs are under `/app/inputs/`; \
write outputs under `/app` as the instruction directs). Returns JSON with `ok`, `exit_code`, \
`stdout`, `stderr`.
- `read_file(path)` / `write_file(path, content)` — read or write a file in the container.

When you have produced the required outputs, call `done`. This **ends the episode**: it seals \
the run, then the task's own verifier runs over the container's final state (collecting the \
declared output artifacts and running the pytest checks in a separate container) and returns \
the recorded 0/1 reward. `done` is one-shot and authoritative — you cannot inspect the verdict \
and re-grade, so make sure the outputs are complete before calling it. No further tool calls \
run after `done`.

The score is the task verifier's 0/1 verdict read off the container end-state — not your \
transcript. Do not fetch online solutions or task-specific hints."""


class FrontierBenchEnv(Env):
    """A Frontier-Bench task suite wrapped as an shogym env (Docker-backed, sealed verdict).

    Exposes the N vendored CPU tasks as ``num_tasks``, selectable by index (``0..N-1``) or name.
    ``done`` is the ``score`` terminal: the serve layer seals the episode and calls
    :meth:`finalize`, which runs the task verifier over the container end-state.

    Config (all optional, via ``shogym.make("frontier_bench", config=...)`` / ``env_config``):
      - ``task``: the *default* vendored task, by name or index — used when a serve/describe call
        omits the task selector (default ``"fin-saccr-rwa"``, index 0).
      - ``max_steps``: the tool-call budget per episode (the shogym horizon is ``max_steps + 2``).
      - ``command_timeout_seconds``: per-``exec`` wall-clock budget inside the container.
      - ``keep_container``: leave the container + images up after the episode (debugging).
    """

    function_name = "exec"
    # `done` is the score terminal: a call seals the episode (validate -> seal -> finalize).
    score_terminal_tool = DONE_TOOL_NAME

    def __init__(
        self,
        task: Optional[str] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        command_timeout_seconds: float = 600.0,
        keep_container: bool = False,
    ) -> None:
        # Validate the default selector eagerly (offline) so a bad config fails at construction,
        # not mid-serve.
        from shogym.envs.frontier_bench import manifest

        self._default_task = manifest.resolve_name(task)
        self._max_steps = max_steps
        self._command_timeout_seconds = command_timeout_seconds
        self._keep_container = keep_container
        self.function = FunctionConfig(example_system_template=_INSTRUCTIONS)
        self.mcp_servers = (
            MCPServerSpec(
                name="frontier_bench",
                transport="in_process",
                module="shogym.envs.frontier_bench.mcp_server",
            ),
        )
        # Horizon: one step per tool call, plus room for `done`.
        super().__init__(horizon=max_steps + 2, num_tasks=manifest.num_tasks())

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        # Resolve the index (or the configured default when None) to a vendored task. An
        # out-of-range index raises rather than silently serving another task under a bogus
        # public id: the serve layer would record the episode as e.g. task "9" while running
        # a different task, and `describe("9")` would then refuse to resolve it — a
        # misattributed run. (resolve_name rejects indices outside 0..N-1, negatives included.)
        from shogym.envs.frontier_bench.manifest import load_task

        selector = self._default_task if task_idx is None else task_idx
        meta = load_task(selector)
        return {
            "task_idx": meta.index,
            "task_name": meta.name,
            "dataset_name": meta.dataset_name,
            "digest": meta.digest,
            "environment_mode": meta.environment_mode,
        }

    # ----- session lifecycle -----

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        from shogym.envs.frontier_bench import mcp_server  # lazy: touches Docker on begin

        mcp_server.begin_session(
            session_id,
            task_name=task["task_name"],
            command_timeout_seconds=self._command_timeout_seconds,
            keep_container=self._keep_container,
        )

    def _end_session(self, session_id: str) -> None:
        from shogym.envs.frontier_bench import mcp_server

        mcp_server.end_session(session_id)

    # ----- describe -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        from shogym.envs.frontier_bench.manifest import load_task

        spec = super().describe(task_id)
        # Resolve the selector (name or index string; None -> configured default) and publish
        # that task's real instruction.md (canary comment included) plus a short provenance
        # footer, so a harness reads the concrete task it will run.
        selector = self._default_task if task_id is None else task_id
        meta = load_task(selector)
        parts = [
            meta.instructions().rstrip(),
            "",
            "# This run",
            f"- task: {meta.dataset_name} (index {meta.index})",
            f"- pinned: frontier-bench {self._upstream_tag()} · {meta.digest}",
            f"- environment_mode: {meta.environment_mode}",
        ]
        return spec.model_copy(update={"instructions": "\n".join(parts)})

    @staticmethod
    def _upstream_tag() -> str:
        from shogym.envs.frontier_bench.manifest import UPSTREAM_TAG

        return UPSTREAM_TAG

    # ----- finalize: the sealed terminal transaction -----

    async def finalize(  # type: ignore[reportIncompatibleVariableOverride]
        self, req: "FinalizeRequest"
    ) -> "TerminalEvidence":
        """Run the task verifier over the container end-state and return the reward verdict.

        (``Env.finalize`` is declared a typed *attribute* — ``Optional[Callable[...]]`` — that the
        serve layer reads via ``getattr``/``callable``; a subclass opts in by overriding it with
        this method, which is the documented contract, so the variable-vs-method override warning
        is suppressed here.)

        Invoked by the serve layer AFTER the episode is sealed (a ``done`` call or the horizon)
        and BEFORE teardown, so the agent container is **still live**: this hook owns it through
        artifact-collection -> verifier build/run -> reward read -> **evidence commit**, and only
        then does the base call ``end_session`` (which stops the container). The blocking Docker
        work runs in a worker thread so the event loop stays responsive — post-seal ingress is
        tombstoned promptly, and a racing close() waits for this finalization to commit before it
        disposes anything.

        Fail-closed: a crash/cancel mid-verify propagates to the serve layer, which commits a
        ``finalize_error`` verdict (reward 0) and — via the durable finalization record — never
        re-invokes this hook. The agent thus cannot read a failing verdict and re-run the verifier
        (the read-and-retry exploit): the seal makes ``done`` structurally one-shot.

        ``req.source`` is ``explicit_tool`` (a ``done`` call) or ``horizon`` (the step budget was
        reached — ``on_horizon`` scores the current container state). ``terminate`` is the ``abort``
        terminal and never reaches here — the base synthesizes a no-score abort verdict directly.
        """
        from shogym.envs.frontier_bench import mcp_server
        from shogym.serve.lifecycle import TerminalEvidence, fail_closed_verdict

        outcome = await asyncio.to_thread(mcp_server.finalize_session, req.session_id)
        if outcome is None:
            # No live session/container to score. This should not happen on the sealed path
            # (finalize runs before teardown), so treat it as an infra failure and fail closed
            # rather than award an honest reward 0.
            return TerminalEvidence(
                source=req.source,
                status="finalize_error",
                verdict=fail_closed_verdict(),
                diagnostic="finalize: no live session/container to verify",
            )
        reward = _as_float(outcome.reward)
        verdict: Dict[str, Any] = {
            "reward": reward,
            "success": reward >= 1.0,
            "reward_found": bool(outcome.reward_found),
            # path -> bool map (which declared artifacts crossed the boundary); benign provenance.
            "artifacts_collected": dict(outcome.artifacts_collected),
        }
        # The verifier's stdout / exit code are PRIVATE — recorded only in the durable store's
        # diagnostic, NEVER surfaced to the agent (surfacing the grader's expected-vs-actual is
        # the read-and-retry leak). `verdict` above carries no grader output.
        diagnostic = (
            f"verifier reward={reward} exit={outcome.test_exit_code} "
            f"tail={outcome.test_stdout_tail[-800:]}"
        )
        return TerminalEvidence(
            source=req.source, status="ok", verdict=verdict, diagnostic=diagnostic
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
        """Score exclusively from the core-owned terminal evidence (never marker JSON).

        The authoritative reward is the task verifier's 0/1 over the container end-state, computed
        in :meth:`finalize` and handed here as ``evidence.verdict``. Parsing is defensive: a
        missing/malformed reward, an abort (``terminate``), or a fail-closed finalize scores 0
        rather than raising."""
        fb = FeedbackCollection()
        if not terminated:
            return fb
        return score_evidence(evidence)


# ----- pure scoring (module-level so it is unit-testable without Docker) -----


def score_evidence(evidence: "Optional[TerminalEvidence]") -> FeedbackCollection:
    """Build episode feedback from the core-owned terminal evidence. Pure.

    A verified episode (``done`` or horizon ran the verifier) emits ``reward`` (0/1),
    ``success``, ``verified=True`` and ``reward_found``. An unverified end — an ``abort``
    (``terminate``/close before ``done``) or a fail-closed finalize — emits ``reward=0``,
    ``success=False``, ``verified=False`` (plus ``finalize_error`` when the finalize itself
    failed, so an analyst can filter infra failures from honest zeros)."""
    fb = FeedbackCollection()
    verdict: Dict[str, Any] = (
        evidence.verdict
        if (evidence is not None and isinstance(evidence.verdict, dict))
        else {}
    )
    if "reward" not in verdict:
        # No verifier verdict: an abort (`terminate`/close, no `done`) or a fail-closed finalize.
        fb.episode.append(EpisodeFeedback(name="reward", value=0.0))
        fb.episode.append(EpisodeFeedback(name="success", value=False))
        fb.episode.append(EpisodeFeedback(name="verified", value=False))
        if evidence is not None and evidence.finalize_error:
            fb.episode.append(EpisodeFeedback(name="finalize_error", value=True))
        return fb

    reward = _as_float(verdict.get("reward"))
    fb.episode.append(EpisodeFeedback(name="reward", value=reward))
    fb.episode.append(EpisodeFeedback(name="success", value=reward >= 1.0))
    fb.episode.append(EpisodeFeedback(name="verified", value=True))
    fb.episode.append(
        EpisodeFeedback(name="reward_found", value=bool(verdict.get("reward_found")))
    )
    return fb


def _as_float(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


@register("frontier_bench")
class FrontierBenchDefault(FrontierBenchEnv):
    """Frontier-Bench's CPU-only task suite (default task ``fin-saccr-rwa``, index 0)."""


__all__ = [
    "FrontierBenchEnv",
    "FrontierBenchDefault",
    "score_evidence",
    "DONE_TOOL_NAME",
]
