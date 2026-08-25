"""End-to-end: drive served ``latentgym`` episodes through shogym's seal-before-verdict serve.

The whole path (resolve a family, publish one episode's instructions, play it a turn at a time
with ``act``, then call ``done``, the ``score`` terminal, which seals and scores in one step)
runs in-process, offline and keyless once the pinned upstream source is provisioned.

The properties worth defending here are the ones that make the port an instrument rather than a
wrapper: an episode addressed by index is the *same* episode every time it is served, the
end-of-episode report never reaches the observation stream, the score is the core env's own number
rather than something read back out of the report's prose, and the inert channel really is the
same size as the channel it stands in for.

The upstream source is provisioned lazily on first use; the module skips if it can't be fetched
(offline + cold cache), like the tau2, yc_bench and automationbench tests.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from tests._fixtures.upstream_gate import gate

gate("shogym.envs.latentgym.adapter", package="latentgym", extra="latentgym")

import shogym  # noqa: E402
from shogym.envs.latentgym import adapter  # noqa: E402
from shogym.feedback.wire import NOTICE_FEEDBACK_NAME, REPORT_FEEDBACK_NAME  # noqa: E402
from shogym.serve import ServedEpisode  # noqa: E402

BANDITS = {"core_env": "bandits", "latent": "loyal_favorite_0", "seed": 7, "length": 12}
SECRETARY = {"core_env": "secretary", "latent": "position_cycle_258", "seed": 7, "length": 12}


async def _play(config: Dict[str, Any], task: int, actions: List[str]) -> Dict[str, Any]:
    """Serve one episode, take ``actions``, end it, and report everything it produced."""
    env = shogym.make("latentgym", config=config)
    episode = await ServedEpisode.open_env(env, env_name="latentgym", task=task)
    try:
        spec = episode.describe()
        observations = []
        for action in actions:
            result = await episode.call("act", {"action": action})
            observations.append(json.loads(result.content))
        terminal = await episode.call("done", {})
        return {
            "instructions": spec.instructions,
            "horizon": spec.horizon,
            "observations": observations,
            "verdict": json.loads(terminal.content),
            "feedback": {
                item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
            },
        }
    finally:
        await episode.close()


# ----- one episode of a family, served -----


async def test_a_served_episode_plays_and_scores() -> None:
    played = await _play(BANDITS, 0, ["[red]", "[blue]", "[select red]"])

    assert played["observations"][-1]["episode_over"] is True
    assert "Correct!" in played["observations"][-1]["observation"]
    # 1.0 - 3 * 0.015, the core env's own decay. The record's number is upstream's arithmetic.
    assert played["verdict"]["reward"] == pytest.approx(0.955)
    assert played["verdict"]["success"] is True
    assert played["verdict"]["turns"] == 3.0
    assert set(played["feedback"]) == {
        "reward",
        "success",
        "turns",
        REPORT_FEEDBACK_NAME,
        NOTICE_FEEDBACK_NAME,
    }


async def test_the_instructions_carry_the_family_framing_and_this_episodes_place_in_it() -> None:
    # The cross-episode framing is the benchmark. An episode told it was "1 of 1" twelve times
    # would be twelve unrelated games, which is a different instrument entirely.
    played = await _play(BANDITS, 7, [])
    instructions = played["instructions"]

    assert "You will play 12 rounds of this game sequentially." in instructions
    assert "--- Episode 8 of 12 ---" in instructions
    # Upstream's own rules and opening observation, not a paraphrase.
    assert "Multi-Armed Bandit game with 5 buttons" in instructions
    assert "Explore with [button_name]" in instructions
    # ...and the tool surface the harness will actually drive.
    assert "`act(action)`" in instructions and "`done()`" in instructions


async def test_the_horizon_leaves_room_to_end_the_task() -> None:
    # The game's own budget plus the ending turn plus the `done` that records it.
    played = await _play(BANDITS, 0, [])
    assert played["horizon"] == 32


# ----- the same index is the same episode, twice -----


async def test_an_episode_served_twice_is_the_same_episode() -> None:
    # The property the whole design rests on: one task dispensed to two agents has to be one
    # task. Upstream draws its outcomes from the global `random` module, so this is only true
    # because the port installs a per-episode seed around every call into it.
    actions = ["[green]", "[green]", "[purple]", "[select green]"]
    first = await _play(BANDITS, 5, actions)
    second = await _play(BANDITS, 5, actions)

    assert first["instructions"] == second["instructions"]
    assert first["observations"] == second["observations"]
    assert first["verdict"] == second["verdict"]


async def test_different_episodes_of_a_family_are_different() -> None:
    # ...and the guarantee above is not just "everything is identical".
    played = [await _play(BANDITS, idx, []) for idx in (0, 5)]
    assert played[0]["instructions"] != played[1]["instructions"]


# ----- the report is feedback, never observation -----


async def test_the_report_never_reaches_the_observation_stream() -> None:
    # Upstream concatenates the end-of-episode write-up onto the final turn's text, which would
    # hand every agent the ground truth whatever regime the run claims to serve. Here the turn
    # that ends the game says what the game said and no more.
    played = await _play(BANDITS, 0, ["[red]", "[select red]"])
    report = played["feedback"][REPORT_FEEDBACK_NAME]

    assert "Probabilities:" in report  # the ground truth really is in the report
    for observation in played["observations"]:
        assert "Probabilities:" not in observation["observation"]
        assert "Score:" not in observation["observation"]


async def test_the_report_names_the_ground_truth_win_or_lose() -> None:
    # What makes this channel a treatment rather than an echo of the transcript.
    lost = await _play(BANDITS, 0, ["[select yellow]"])
    report = lost["feedback"][REPORT_FEEDBACK_NAME]

    assert lost["verdict"]["success"] is False
    assert "The best was 'red'" in report
    assert "Probabilities: red=" in report


async def test_the_inert_channel_matches_the_report_it_stands_in_for() -> None:
    played = await _play(BANDITS, 0, ["[red]", "[select red]"])
    report = played["feedback"][REPORT_FEEDBACK_NAME]
    notice = played["feedback"][NOTICE_FEEDBACK_NAME]

    # Matched as the answer encodes it, which is the form the agent reads. Upstream writes this
    # report with a dash character, so the encoded length and the character count are not the same
    # number and a notice built to the second would arrive five characters short of the first.
    assert "\u2014" in report
    assert len(json.dumps(notice)) == len(json.dumps(report))
    assert "Probabilities" not in notice and "Score" not in notice


# ----- the score is the core env's own number -----


async def test_a_report_that_disagrees_with_the_reward_is_corrected() -> None:
    # Upstream's secretary handlers print `Score: 0.000` on every miss while the core env awards
    # `0.5 * accepted / max`, so an agent is told it scored nothing on an episode worth 0.4. The
    # number the record scores on and the number the report states have to be one number.
    played = await _play(SECRETARY, 0, ["[continue]", "[continue]", "[continue]", "[accept]"])
    reward = played["verdict"]["reward"]
    report = played["feedback"][REPORT_FEEDBACK_NAME]

    assert 0.0 < reward < 1.0  # partial credit: the case upstream misreports
    assert f"Score: {reward:.3f}" in report
    assert "Score: 0.000" not in report
    assert played["feedback"]["reward"] == reward


async def test_a_report_that_already_agrees_is_left_alone() -> None:
    played = await _play(SECRETARY, 0, ["[continue]"] * 9 + ["[accept]"])
    if played["verdict"]["reward"] == 1.0:
        assert "Score: 1.000" in played["feedback"][REPORT_FEEDBACK_NAME]


# ----- endings the agent did not play for -----


async def test_ending_the_task_early_records_no_outcome() -> None:
    # `done` on a game still in play is a task with no outcome, not a zero the agent earned.
    played = await _play(BANDITS, 0, ["[red]"])
    assert played["verdict"]["finished"] is False
    assert played["verdict"]["reward"] == 0.0
    assert played["feedback"][REPORT_FEEDBACK_NAME] == ""
    assert played["feedback"][NOTICE_FEEDBACK_NAME] == ""


async def test_a_turn_after_the_ending_is_answered_not_played() -> None:
    played = await _play(BANDITS, 0, ["[select red]", "[blue]", "[green]"])
    assert [o["turn"] for o in played["observations"]] == [1, 1, 1]
    assert played["observations"][-1]["observation"].startswith("The episode is over")
    # The ending the agent did play still scores.
    assert played["verdict"]["reward"] == pytest.approx(0.985)


async def test_the_horizon_scores_what_the_game_recorded() -> None:
    # A run that never calls `done` is finalized at the horizon on the ending it did reach.
    env = shogym.make("latentgym", config=BANDITS)
    episode = await ServedEpisode.open_env(env, env_name="latentgym", task=0)
    try:
        for _ in range(40):
            result = await episode.call("act", {"action": "[blue]"})
            if result.terminated:
                break
        assert result.terminated
        verdict = json.loads(result.content)
        assert verdict["finished"] is True
        assert verdict["turns"] == 30.0  # the core env's own per-episode budget
    finally:
        await episode.close()


# ----- what the port refuses, and when -----


def test_a_filter_latent_is_refused_at_construction() -> None:
    # It would draw its episodes from a candidate pool the public repository does not ship. The
    # refusal belongs here rather than three tasks into a run.
    with pytest.raises(ValueError, match="candidate pool"):
        shogym.make("latentgym", config={"core_env": "wordladder", "latent": "order_left_to_right"})


def test_an_unknown_name_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown LatentGym core env"):
        shogym.make("latentgym", config={"core_env": "chess"})
    with pytest.raises(ValueError, match="not found"):
        shogym.make("latentgym", config={"latent": "no_such_latent"})
    with pytest.raises(ValueError, match="not found"):
        shogym.make("latentgym", config={"prompt": "no_such_prompt"})


def test_a_task_outside_the_family_is_refused() -> None:
    env = shogym.make("latentgym", config=BANDITS)
    assert env.num_tasks == 12
    with pytest.raises(ValueError, match="out of range"):
        env.load_task(12)
    # A negative index would index backwards into a real episode while the record said `-1`. It
    # never reaches the env's own guard (the base seeds its generator from the index first and
    # refuses a negative seed), but it is refused, which is what matters.
    with pytest.raises(Exception):
        env.load_task(-1)


# ----- the family itself -----


def test_a_family_is_a_function_of_its_seed_and_its_length() -> None:
    first = adapter.load_family(
        core_env="bandits", latent="cold_hand", prompt="some_info", seed=3, length=8
    )
    same = adapter.load_family(
        core_env="bandits", latent="cold_hand", prompt="some_info", seed=3, length=8
    )
    other_seed = adapter.load_family(
        core_env="bandits", latent="cold_hand", prompt="some_info", seed=4, length=8
    )

    assert first.configs == same.configs
    assert first.configs != other_seed.configs


def test_the_executable_core_envs_are_the_ones_that_run() -> None:
    # The census the port's scope rests on: 83 generator latents across four core envs is what is
    # reachable without the absent `latentgym.data` pools. A change upstream should move this.
    counts = {name: len(adapter.generator_latents(name)) for name in adapter.EXECUTABLE_CORE_ENVS}
    assert counts == {"bandits": 29, "secretary": 41, "number_guessing": 7, "wordladder": 6}
