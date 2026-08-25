"""``latentgym`` on the env-as-center core (RFC 008): one LatentGym episode is one shogym task.

LatentGym (MIT, © Namkoong Lab) asks whether a model can infer a hidden rule from a run of small
games and then exploit it. A *family* is one ``(core env, latent)`` pair; a run under it is a
sequence of episodes the latent generates, and everything interesting about the benchmark lives in
what an agent carries from the early episodes into the later ones.

That maps onto a task stream exactly, and onto nothing smaller. A task here is **one episode of a
family**, addressed by its position in it, so a queue of ``[0, 1, 2, ...]`` is the trajectory a
LatentGym run would have played, one dispensed task at a time, with the harness as the loop rather
than an inline agent driver. Positions are stable and repeats are legal: task 3 is the same
starting world every time it is asked for (see :func:`~shogym.envs.latentgym.adapter.episode_seed`),
which is what lets one task be served twice to two agents and compared.

  - **describe**: a :class:`TaskSpec` whose instructions are upstream's own composition, the core
    env's game rules, the prompt template's multi-episode framing, this episode's header, and its
    opening observation. Nothing about the position is added and nothing about the latent is
    revealed beyond what the chosen prompt template says.
  - **serve**: ``act`` plays one turn; ``done`` is the ``score`` terminal.
  - **finalize + verify**: ``done`` seals the episode, then ``finalize`` reads the reward the core
    env returned when the game ended (never a number parsed back out of feedback text) and
    ``_verify`` publishes it, along with the two report channels below.

**The two report channels are the point of the port.** Upstream concatenates its end-of-episode
write-up onto the last turn's observation, so every agent reads it whatever regime the run claims
to be serving. Here the write-up is published as episode feedback under
:data:`~shogym.feedback.wire.REPORT_FEEDBACK_NAME`, beside an inert, length-matched stand-in under
:data:`~shogym.feedback.wire.NOTICE_FEEDBACK_NAME`, so a stream serving
:class:`~shogym.serve.stream.Information` hands the agent the report, one serving
:class:`~shogym.serve.stream.Placebo` hands it the same-sized nothing, and one serving
:class:`~shogym.serve.stream.Never` hands it neither. What an agent is told about its ending is a
property of the run's construction, and the record says which.

This module imports **nothing** from upstream at load time, so ``import shogym`` (which imports it
to register the env) stays offline. The pinned upstream source is provisioned lazily when a
``latentgym`` env is *constructed*; see :mod:`shogym.envs.latentgym.adapter`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from shogym.core import Env
from shogym.envs.registration import register
from shogym.feedback.wire import NOTICE_FEEDBACK_NAME, REPORT_FEEDBACK_NAME
from shogym.mcp import MCPServerSpec
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection, FunctionConfig

if TYPE_CHECKING:
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence

ACT_TOOL_NAME = "act"

# The env's `score` terminal. Calling it seals the episode and only then scores it, so a verdict
# never exists for a game that can still be played.
DONE_TOOL_NAME = "done"

# Defaults, chosen to be the least surprising rather than the most interesting: the first core env
# upstream registers, its first latent, and the prompt variant whose framing says a pattern may
# exist without naming it. Which family is worth measuring is a question for whoever is measuring;
# the README says what is known about that.
DEFAULT_CORE_ENV = "bandits"
DEFAULT_LATENT = "loyal_favorite_0"
DEFAULT_PROMPT = "some_info"
DEFAULT_SEED = 0
DEFAULT_LENGTH = 32

# Turns past the game's own budget, so a run that means to end the task explicitly can still reach
# `done` after the last turn instead of being forced to the horizon. Two: the ending turn, and the
# `done` that records it.
_TERMINAL_SLACK = 2

LATENTGYM_SPEC = MCPServerSpec(
    name="latentgym",
    transport="in_process",
    module="shogym.envs.latentgym.mcp_server",
)

_TOOL_GUIDE = """\
# Tools
- `act(action)`: take one turn. Put your action in square brackets exactly as the rules above
  describe, and put square brackets nowhere else. It answers with what the game said back and
  whether the episode is over.
- `done()`: end the task, once `act` reports the episode is over. It records the outcome; there
  is no second submission and no need to call `terminate` afterward."""

# The sentence the inert channel is built out of. Deliberately about the mechanics of the record
# and not about the episode: it names no outcome, no score and no ground truth, so an agent that
# reads it learns that its task was recorded and nothing else. See `notice_for`.
_NOTICE_SENTENCE = "This episode has been recorded. The next task is drawn the same way. "


@register("latentgym")
class LatentGymEnv(Env):
    """One family of LatentGym episodes, served one episode per task.

    Config (all optional, via ``shogym.make("latentgym", config=...)`` / ``env_config``):
      - ``core_env``: which game. ``bandits`` (default), ``secretary``, ``number_guessing`` or
        ``wordladder``. The other three registered upstream draw their episodes from candidate
        pools the public repository does not ship.
      - ``latent``: the hidden rule the family is drawn under. Must be a ``generator`` latent of
        ``core_env``; ``adapter.generator_latents(core_env)`` lists them.
      - ``prompt``: ``no_info`` / ``some_info`` (default) / ``full_info``, i.e. how much the framing
        gives away about there being a pattern at all.
      - ``seed``: the family's seed. The seed and the length together decide every episode, so two
        runs that share them share their worlds.
      - ``length``: how many episodes the family has, which is also how many tasks it offers. A
        latent generates its trajectory as a whole and an episode may depend on the ones before
        it, so changing this changes the episodes, including the early ones.
    """

    mcp_servers = (LATENTGYM_SPEC,)
    function_name = "agent"
    score_terminal_tool = DONE_TOOL_NAME

    def __init__(
        self,
        core_env: str = DEFAULT_CORE_ENV,
        latent: str = DEFAULT_LATENT,
        prompt: str = DEFAULT_PROMPT,
        seed: int = DEFAULT_SEED,
        length: int = DEFAULT_LENGTH,
    ) -> None:
        from shogym.envs.latentgym import adapter  # lazy: provisions the upstream source

        self._family = adapter.load_family(
            core_env=core_env, latent=latent, prompt=prompt, seed=seed, length=length
        )
        self.function = FunctionConfig(example_system_template=self._static_instructions())
        super().__init__(
            horizon=self._family.max_turns + _TERMINAL_SLACK, num_tasks=self._family.length
        )

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, self._family.length))
        if not 0 <= task_idx < self._family.length:
            # Negatives too: Python would index backwards into a real episode while the record
            # said `-1`, so a task that ran would be filed as one that does not exist.
            raise ValueError(
                f"Task index {task_idx} is out of range for {self._family.length} episodes"
            )
        return {
            "task_idx": task_idx,
            "core_env": self._family.core_env,
            "latent": self._family.latent,
            "config": self._family.config(task_idx),
        }

    # ----- session lifecycle -----

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        from shogym.envs.latentgym import mcp_server

        episode, _ = self._open(int(task["task_idx"]))
        mcp_server.begin_session(session_id, episode)

    def _end_session(self, session_id: str) -> None:
        from shogym.envs.latentgym import mcp_server

        mcp_server.end_session(session_id)

    def _open(self, task_idx: int) -> Tuple[Any, str]:
        """Build the episode at ``task_idx`` and its opening text.

        The one place an episode is started, used both to serve one and to render the opening an
        agent is shown, so what ``describe`` publishes is what ``act`` is played against, rather
        than a second rendering that could drift from it."""
        from shogym.envs.latentgym import adapter, mcp_server

        return mcp_server.open_episode(
            core=self._family.new_core_env(),
            step_handler=self._family.step_handler(),
            report_handler=self._family.report_handler(),
            episode_idx=task_idx,
            config=self._family.config(task_idx),
            seed=adapter.episode_seed(self._family, task_idx),
        )

    # ----- describe -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        spec = super().describe(task_id)
        task_idx = self._resolve_idx(task_id)
        if task_idx is None:
            return spec
        return spec.model_copy(update={"instructions": self._instructions(task_idx)})

    def _resolve_idx(self, task_id: Optional[str]) -> Optional[int]:
        if task_id is None:
            return None
        try:
            idx = int(task_id)
        except (TypeError, ValueError):
            return None
        return idx if 0 <= idx < self._family.length else None

    def _static_instructions(self) -> str:
        """The durable, episode-independent framing published by ``describe(task_id=None)``: the
        game's rules and the family's framing, with no episode in them."""
        core = self._family.new_core_env()
        try:
            framing = self._family.prompt_template().initial_system_prompt(
                game_rules=core.get_game_rules(),
                env_params=self._family.env_params,
                num_episodes=self._family.length,
            )
        finally:
            core.close()
        return f"{framing}\n\n{_TOOL_GUIDE}"

    def _instructions(self, task_idx: int) -> str:
        """One episode's instructions: upstream's own composition, plus the tool guide.

        Assembled the way upstream assembles the opening of a trajectory (framing, then the
        episode header, then the opening observation), with the header naming this episode's
        place in the family rather than always the first. An agent picking up episode 8 is told it
        is on episode 8, which is true, and is the only way the cross-episode framing above it
        means anything."""
        episode, opening = self._open(task_idx)
        try:
            framing = self._family.prompt_template().initial_system_prompt(
                game_rules=episode.core.get_game_rules(),
                env_params=self._family.env_params,
                num_episodes=self._family.length,
            )
        finally:
            episode.core.close()
        header = f"\n\n--- Episode {task_idx + 1} of {self._family.length} ---\n"
        return f"{framing}{header}{opening}\n\n{_TOOL_GUIDE}"

    # ----- finalize (seal-before-verdict) -----

    async def finalize(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, req: "FinalizeRequest"
    ) -> "TerminalEvidence":
        """Score the **sealed** episode and return core-owned evidence.

        Reads the outcome the core env returned when the game ended, held on the session since
        that turn, never a score parsed back out of feedback text, which upstream's own handlers
        are not always honest about. An episode that never ended (``done`` called early, or the
        horizon reached mid-game) has no outcome, and is scored as one: no reward, not a zero the
        agent played for.

        The verdict carries the report, because publishing it is the whole point of the port: it
        becomes the episode feedback a feedback policy decides the fate of. It is public-safe in
        exactly the sense every other published verdict here is: a stream answers a terminating
        call from its policy and never from this dict, so what reaches an agent is the run's
        construction rather than this env's generosity."""
        from shogym.envs.latentgym import mcp_server
        from shogym.serve.lifecycle import TerminalEvidence

        outcome = mcp_server.score_session(req.session_id)
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict={
                "reward": _as_reward(outcome.get("reward")),
                "success": _succeeded(outcome),
                "turns": float(outcome.get("turns") or 0.0),
                "finished": bool(outcome.get("finished")),
                REPORT_FEEDBACK_NAME: str(outcome.get("report") or ""),
            },
            diagnostic=(
                f"scored source={req.source} finished={outcome.get('finished')} "
                f"reward={outcome.get('reward')} turns={outcome.get('turns')}"
            ),
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
        """Publish the episode's outcome off the core-owned terminal ``evidence``.

        ``reward`` is the core env's own number and the record's headline; ``success`` is the
        env's own verdict where it publishes one; ``turns`` is how many turns the episode took,
        which on this benchmark is the more sensitive of the two readouts. An agent that has
        inferred the latent stops exploring, and that shows up in the turn count before it shows
        up in a reward already near its ceiling.

        ``report`` and ``notice`` are the matched pair a feedback policy chooses between. Both are
        always published, whatever regime the run is serving, because the env does not know the
        regime and may not: an env that published only the one its run was going to reveal would
        have made the record depend on the treatment."""
        fb = FeedbackCollection()
        if not terminated:
            return fb
        verdict = evidence.verdict if evidence is not None else {}
        fb.episode.append(EpisodeFeedback(name="reward", value=_as_reward(verdict.get("reward"))))
        fb.episode.append(EpisodeFeedback(name="success", value=bool(verdict.get("success"))))
        fb.episode.append(EpisodeFeedback(name="turns", value=float(verdict.get("turns") or 0.0)))
        report = str(verdict.get(REPORT_FEEDBACK_NAME) or "")
        fb.episode.append(EpisodeFeedback(name=REPORT_FEEDBACK_NAME, value=report))
        fb.episode.append(EpisodeFeedback(name=NOTICE_FEEDBACK_NAME, value=notice_for(report)))
        if evidence is not None and evidence.finalize_error:
            fb.episode.append(EpisodeFeedback(name="finalize_error", value=True))
        return fb


# ----- pure helpers (module-level so they are unit-testable without the upstream package) -----


def notice_for(report: str) -> str:
    """The inert stand-in for ``report``: the same length on the wire, none of it evaluative.

    Length is what makes it a stand-in rather than a different treatment. The two arms of a paired
    design differ in one thing only if the channel they ride is the same size, and an agent that
    reads five words in one arm and forty in the other has been handed a cue about which arm it is
    in before it has read a word of either.

    **Matched as the wire carries it, not as Python holds it.** A terminating call is answered with
    a JSON document and the agent reads that document, so what has to match is the *encoded* length,
    and the two are not the same thing here. Upstream writes its reports with a dash character the
    default encoder escapes to six characters, so a notice built to the report's character count
    would arrive five characters short of it. The report's own encoding decides the target
    (:func:`_wire_length`); this text is plain ASCII, which encodes as itself.

    Built by repeating one neutral sentence and cutting on a word boundary, with the shortfall
    taken up in trailing spaces, so the text reads as text rather than as a mangled word, and the
    count still matches exactly. An empty report gets an empty notice: there was no channel to
    match."""
    target = _wire_length(report)
    if target <= 0:
        return ""
    filled: List[str] = []
    used = 0
    while True:
        for word in _NOTICE_SENTENCE.split():
            if used + len(word) + (1 if filled else 0) > target:
                return (" ".join(filled)).ljust(target)
            used += len(word) + (1 if filled else 0)
            filled.append(word)
        if used == target:
            return " ".join(filled)


def _wire_length(report: str) -> int:
    """How much room ``report`` takes inside the JSON document a terminating call is answered with.

    The encoded string minus its two quotes, so a plain-ASCII notice of this many characters
    encodes to exactly as many bytes as the report does. Encoded with the encoder the answer is
    actually composed with, ASCII-escaping and all, rather than with an assumption about which
    characters upstream happens to use."""
    return len(json.dumps(report)) - 2


def _as_reward(value: Any) -> float:
    """Coerce a reward to a finite float; junk/absent is no reward.

    Not clamped to [0, 1]: this is the core env's own number and a port that quietly rescaled it
    would be reporting a benchmark it had adjusted. Only a value that cannot be a reward at all
    (absent, non-numeric, NaN, infinite) becomes zero, because the wire contract admits no other
    answer for one."""
    if isinstance(value, bool):
        return 0.0
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    if out != out or out in (float("inf"), float("-inf")):
        return 0.0
    return out


# The keys a core env records its own boolean verdict under. Read in this order, and only when the
# value really is a bool: these say "solved" in the env's own words, which is what a success rate
# should count, and no arithmetic on the reward reproduces them (a bandits episode won on turn 2
# scores 0.97, never 1.0).
_VERDICT_KEYS = ("is_correct", "solved")


def _succeeded(outcome: Dict[str, Any]) -> bool:
    """Whether the episode was a success: the core env's own verdict, or full credit.

    Falling back to ``reward >= 1.0`` rather than to ``reward > 0`` because full credit is the one
    thing every core env here agrees means solved: secretary pays partial credit for a miss, so
    any positive reward would count near-misses as wins."""
    if not outcome.get("finished"):
        return False
    info = outcome.get("info") or {}
    for key in _VERDICT_KEYS:
        value = info.get(key)
        if isinstance(value, bool):
            return value
    return _as_reward(outcome.get("reward")) >= 1.0
