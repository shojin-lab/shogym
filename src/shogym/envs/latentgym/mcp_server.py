"""FastMCP server for the latentgym env.

Two tools. `act` plays one turn of the episode the env pushed into this session and answers with
what the game said back; `done` is the score terminal, which the serve layer turns into a
validate -> seal -> finalize transaction. State is keyed by ``_session_id`` so one env instance
can back several concurrent episodes.

Lifecycle (in-process only):
  - ``begin_session(session_id, ...)``: the env resets a core env onto this session's episode
    configuration before any tool is called, and hands over the handlers the episode is played
    with.
  - ``act(action, _session_id)``: one turn.
  - ``score_session(session_id)``: what the episode earned, read from the core env's own return
    value at the moment it ended. The env's ``finalize`` calls this on the sealed episode.
  - ``end_session(session_id)``: the env drops the state on teardown.

**The end-of-episode report never rides out through `act`.** Upstream concatenates it onto the
final turn's observation, which would hand every agent the ground truth regardless of what regime
the run is serving. Here it is held on the session and published as episode feedback, where the
stream's feedback policy decides whether it reaches the agent at all.
"""

from __future__ import annotations

import random
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from fastmcp import FastMCP

server: FastMCP = FastMCP(name="latentgym")

# Per-episode state keyed by session_id.
sessions: Dict[str, "_Episode"] = {}

# Serializes both the session table and the global-`random` swap below. Upstream's core envs draw
# from the module-level `random` API while a turn is being played, so a session's RNG state has to
# be installed, used and taken back out as one indivisible step: two concurrent episodes that
# interleaved on it would deal each other's pulls, and neither would replay.
_state_lock = threading.RLock()

_T = TypeVar("_T")


@dataclass
class _Episode:
    """One LatentGym episode, mid-play."""

    core: Any
    step_handler: Any
    report_handler: Any
    episode_idx: int
    # The RNG state this episode plays under, advanced turn by turn (see `_under`).
    rng_state: Tuple[Any, ...]
    turns: int = 0
    over: bool = False
    reward: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)
    report: Optional[str] = None


def _under(episode: "_Episode", call: Callable[[], _T]) -> _T:
    """Run ``call`` with this episode's RNG installed as the global one, then take it back out.

    The swap is upstream's own idiom (its factory seeds the global module and restores the
    caller's state afterwards) applied per turn instead of per trajectory, which is what makes an
    episode's realized outcomes a function of the episode rather than of whatever else in the
    process has drawn from ``random`` since. Held under the lock, so the installed state is
    always this episode's."""
    with _state_lock:
        caller_state = random.getstate()
        random.setstate(episode.rng_state)
        try:
            return call()
        finally:
            episode.rng_state = random.getstate()
            random.setstate(caller_state)


def open_episode(
    *,
    core: Any,
    step_handler: Any,
    report_handler: Any,
    episode_idx: int,
    config: Dict[str, Any],
    seed: Any,
) -> Tuple["_Episode", str]:
    """Build an episode and reset its core env onto ``config``; return it and the opening text.

    Not a session yet. :func:`begin_session` files one, and the env also calls this with a
    throwaway core env to render the opening text for ``describe()``. Both go through here so the
    text an agent is *told* and the game it is *served* come from the same call under the same
    seed."""
    prepared = _Episode(
        core=core,
        step_handler=step_handler,
        report_handler=report_handler,
        episode_idx=episode_idx,
        rng_state=random.Random(seed).getstate(),
    )
    opening = _under(prepared, lambda: core.reset(dict(config)))
    return prepared, str(opening)


def begin_session(session_id: str, episode: "_Episode") -> None:
    """Register a prepared episode as this session's game. Idempotent within a session_id."""
    with _state_lock:
        sessions[session_id] = episode


def end_session(session_id: str) -> None:
    """Drop a finished episode's per-session state. Idempotent."""
    with _state_lock:
        episode = sessions.pop(session_id, None)
    if episode is not None:
        episode.core.close()


def score_session(session_id: str) -> Dict[str, Any]:
    """What the episode earned: the core env's own reward, plus the report written from it.

    Read from what ``step`` returned at the moment the game ended, never parsed back out of any
    feedback text. Upstream's secretary handlers, for one, print a hardcoded zero on every miss
    while the env awards partial credit, so the text and the reward genuinely disagree.

    An episode that never ended earns nothing: ``done`` called early, or a budget spent without
    finishing, is a task with no outcome rather than a task with a zero the agent played for."""
    with _state_lock:
        episode = sessions.get(session_id)
    if episode is None:
        return {"finished": False, "reward": 0.0, "turns": 0, "report": ""}
    return {
        "finished": episode.over,
        "reward": float(episode.reward) if episode.over else 0.0,
        "turns": float(episode.turns),
        "report": episode.report or "",
        "info": dict(episode.info),
    }


@server.tool
def act(action: str, _session_id: str) -> Dict[str, Any]:
    """Take one turn in the current episode. Put your action in square brackets, exactly as the
    task instructions describe, and send nothing else in brackets.

    Returns a dict with:
      - ``observation``: what the game said back
      - ``turn``: how many turns this episode has taken
      - ``episode_over``: True once the game has ended; call `done` when it is
    """
    # The whole turn is one critical section, re-entrant so `_under` can take the same lock for
    # the RNG swap. A read-then-play split would let two calls fired in parallel by one action
    # both observe a live game and both step it: one turn counted as two, and an episode whose
    # ending is decided by whichever finished last. Serializing costs nothing here, because a
    # turn is microseconds of pure Python.
    with _state_lock:
        episode = sessions.get(_session_id)
        if episode is None:
            return {
                "observation": "<error: session not initialized; env did not call begin_session>",
                "turn": 0,
                "episode_over": True,
            }
        if episode.over:
            # A turn past the ending is not a turn: the game is finished, and re-entering a
            # finished core env would either raise inside upstream or start scoring a second
            # ending over the first. Answering plainly lets an agent that lost count reach `done`.
            return {
                "observation": "The episode is over. Call `done` to end the task.",
                "turn": episode.turns,
                "episode_over": True,
            }

        episode.turns += 1
        raw, reward, ended, info = _under(episode, lambda: episode.core.step(str(action)))
        observation = episode.step_handler.format_step_feedback(
            raw, episode.episode_idx, episode.turns, info
        )
        if ended:
            episode.over = True
            episode.reward = float(reward)
            episode.info = dict(info)
            # Written now, while the ending is in hand, and held for `score_session`. Deliberately
            # not appended to `observation`, which is where upstream puts it: what an agent is told
            # about its ending is the feedback policy's call, and a report concatenated onto a tool
            # result would have been told regardless.
            episode.report = _report(episode, float(reward), info)
        return {
            "observation": str(observation),
            "turn": episode.turns,
            "episode_over": episode.over,
        }


@server.tool
def done() -> str:
    """Finish the task: end the episode and record what it scored.

    Call this once the episode is over (`act` says so). ``done`` is the terminal action: there is
    no second submission, no ``done``-then-fix loop, and no need to call ``terminate`` after it.
    It takes no arguments and reveals nothing about the grade.
    """
    # The serve layer intercepts `done` as the env's `score` terminal (validate -> seal ->
    # finalize), so this body is never dispatched for a served episode. Scoring happens in
    # `env_v1.LatentGymEnv.finalize` via `score_session`. It answers at all only for a direct,
    # unsealed call against this server on its own.
    return "episode ended"


def _report(episode: "_Episode", reward: float, info: Dict[str, Any]) -> str:
    """The end-of-episode report, written by upstream's richest handler and corrected to agree
    with the reward this task is actually scored on.

    The correction is one substitution and it is not cosmetic. Upstream's secretary handlers print
    ``Score: 0.000`` on every miss while the core env awards ``0.5 * accepted / max``, so an agent
    is told it scored nothing on an episode worth 0.4, a report that contradicts the number in
    the record beside it. Applied to every env rather than to the one known to need it: what the
    report says a task scored has to be what the record says it scored, and an env whose handler
    already agrees is left byte-identical."""
    text = str(
        episode.report_handler.format_episode_end_feedback(episode.episode_idx, reward, info)
    )
    return _agree_on_score(text, reward)


def _agree_on_score(text: str, reward: float) -> str:
    """Rewrite a ``Score: <number>`` claim in ``text`` to the reward the task is scored on.

    Matches upstream's own rendering (three decimals, the only form its handlers emit), so a
    report that already agrees is rewritten to the bytes it already had, and one with no such
    claim is left alone."""
    return re.sub(r"(?<=Score: )\d+\.\d+", f"{reward:.3f}", text)


__all__: List[str] = [
    "act",
    "begin_session",
    "done",
    "end_session",
    "open_episode",
    "score_session",
    "server",
]
