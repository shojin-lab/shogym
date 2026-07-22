"""``wordle_v1`` on the minimal env-as-center core (RFC 008).

The env declares one MCP server (`guess`), advisory templates, and a horizon; it loads a
target word per task, pushes it into the in-process server on session start, and verifies
the recorded trajectory of guesses. No agent loop, no observations — a harness drives the
`guess`/`terminate` tools and this scores what happened.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from hgym.envs.registration import register
from hgym.envs.tool_using_env import ToolUsingEnv
from hgym.envs.wordle import mcp_server as wordle_mcp_server
from hgym.envs.wordle.functions.guess.system_schema import WordleGuessSystemSchema
from hgym.envs.wordle.functions.guess.user_schema import WordleGuessUserSchema
from hgym.envs.wordle.utils import load_words, score_guess
from hgym.mcp import MCPServerSpec
from hgym.trajectory import Step, Trajectory
from hgym.types import (
    EpisodeFeedback,
    FeedbackCollection,
    FunctionConfig,
    InferenceFeedback,
)
from hgym.utils import load_template

MAX_GUESSES = 6

WORDLE_SPEC = MCPServerSpec(
    name="wordle",
    transport="in_process",
    module="hgym.envs.wordle.mcp_server",
)


class WordleV1Env(ToolUsingEnv):
    mcp_servers = (WORDLE_SPEC,)
    function_name = "guess"
    function = FunctionConfig(
        system_schema=WordleGuessSystemSchema,
        user_schema=WordleGuessUserSchema,
        example_system_template=load_template(
            "envs/wordle/functions/guess_v1/example/system.minijinja"
        ),
        example_user_template=load_template(
            "envs/wordle/functions/guess_v1/example/user.minijinja"
        ),
    )

    def __init__(self, words: List[str], task_split: str = "train") -> None:
        super().__init__(horizon=MAX_GUESSES, num_tasks=len(words))
        self._words = words
        self._task_split = task_split

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._words)))
        if task_idx >= len(self._words):
            raise ValueError(
                f"Task index {task_idx} is out of range for {len(self._words)} tasks"
            )
        return {
            "task_idx": task_idx,
            "answer": self._words[task_idx],
            "split": self._task_split,
        }

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        # In-process: push the target word into the server's per-session state.
        wordle_mcp_server.begin_session(session_id, task["answer"])

    def _end_session(self, session_id: str) -> None:
        # In-process: drop the per-session state pushed in `_begin_session`.
        wordle_mcp_server.end_session(session_id)

    def _verify(
        self, trajectory: Trajectory, task: Dict[str, Any], *, terminated: bool
    ) -> FeedbackCollection:
        fb = FeedbackCollection()
        target = str(task["answer"]).lower()

        # Per-step inference signal: was the most recent guess well-formed?
        last = trajectory[-1] if trajectory else None
        if last is not None and last.tool == "guess":
            valid, _, _ = _score_guess_step(last, target)
            fb.inference.append(
                InferenceFeedback(name="format_reward", step=last.index, value=valid)
            )

        # Terminal episode scoring, derived from the recorded guess arguments and
        # the task answer. The tool result is untrusted diagnostic data — a forged
        # or malformed result can neither grant credit nor crash scoring here.
        if terminated:
            guesses = [s for s in trajectory if s.tool == "guess"]
            solved = False
            best_green = 0
            consumed = 0
            for step in guesses[:MAX_GUESSES]:
                consumed += 1
                valid, step_solved, score = _score_guess_step(step, target)
                if valid and score is not None:
                    best_green = max(best_green, score.count("G"))
                if step_solved:
                    solved = True
                    break

            fb.episode.append(EpisodeFeedback(name="check_answer", value=solved))
            partial = 1.0 if solved else best_green / 5.0
            fb.episode.append(EpisodeFeedback(name="partial_credit", value=partial))
            fb.episode.append(
                EpisodeFeedback(name="count_turns", value=float(consumed))
            )
        return fb


def _score_guess_step(step: Step, target: str) -> Tuple[bool, bool, Optional[str]]:
    """Score a recorded guess against the task answer.

    The verifier is authoritative: validity, the G/Y/X mask, and solved status are
    derived from the recorded ``word`` argument and the target, never from the
    (untrusted) tool result. A malformed ``word`` scores as an invalid guess rather
    than raising."""
    word = step.arguments.get("word")
    if not (isinstance(word, str) and len(word) == 5 and word.isalpha()):
        return (False, False, None)
    score = score_guess(word.lower(), target)
    return (True, score == "GGGGG", score)


@register("wordle_v1")
class WordleV1Default(WordleV1Env):
    def __init__(self, task_split: str = "train") -> None:
        all_words = load_words()
        split_idx = int(len(all_words) * 0.8)
        words = all_words[split_idx:] if task_split == "test" else all_words[:split_idx]
        super().__init__(words=words, task_split=task_split)
