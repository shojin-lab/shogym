"""Who ends an episode when its step budget runs out.

An episode that owns its own ending seals and grades when the budget is spent, which is the
only ending a caller driving it directly has. An episode served through the durable stream does
not: the stream is what ends an attempt there, and an ending decided inside the episode would
seal the environment and take a verdict off it while the stream still held the attempt open,
with nothing in the stream's record saying it had happened.

So the flag is checked from both sides. The episode is asked what it does with a spent budget,
and the gateway is asked what it does with an episode that would end itself.
"""

from __future__ import annotations

import pytest

from shogym.serve import ServedEpisode

from tests._fixtures import score_env, score_mcp
from tests._fixtures.score_env import ENV_NAME, HORIZON


@pytest.fixture(autouse=True)
def _clean_fixture_state() -> None:
    score_mcp.reset_state()
    score_mcp.reset_block()


async def test_an_episode_that_owns_its_ending_seals_when_the_budget_runs_out() -> None:
    """The behaviour a caller driving an episode directly relies on."""
    env = score_env._FixtureScoreEnv()
    episode = await ServedEpisode.open_env(env, env_name=ENV_NAME, task=0)
    try:
        assert episode.ends_on_horizon is True
        for _ in range(HORIZON):
            result = await episode.call("noop", {})
        assert result.terminated is True
        assert env.finalize_calls == 1
        assert episode.sealed is True
    finally:
        await episode.close()


async def test_an_episode_whose_caller_ends_it_does_not_seal_on_a_spent_budget() -> None:
    """The budget runs out and nothing is sealed, nothing is graded, nothing is ended.

    This is the path that could fire inside a durable generation: an ordinary environment call
    reaching the budget would run the finalizer and take a verdict, which is a whole ending
    made by a layer the stream is not reading. The step is still committed, so the trajectory
    says what happened; only the decision about the episode is gone.
    """
    env = score_env._FixtureScoreEnv()
    episode = await ServedEpisode.open_env(
        env, env_name=ENV_NAME, task=0, ends_on_horizon=False
    )
    try:
        assert episode.ends_on_horizon is False
        for _ in range(HORIZON):
            result = await episode.call("noop", {})
        assert result.terminated is False
        assert env.finalize_calls == 0
        assert episode.sealed is False

        # Past the budget the episode is still the one that was open, and it still answers.
        beyond = await episode.call("noop", {})
        assert beyond.terminated is False
        assert beyond.tombstoned is False
        assert env.finalize_calls == 0

        # The terminal is still the only thing that ends it, and it still works.
        ended = await episode.call("submit", {"answer": "4"})
        assert ended.terminated is True
        assert env.finalize_calls == 1
    finally:
        await episode.close()


async def test_the_gateway_refuses_an_episode_that_would_end_itself() -> None:
    """A generation cannot be opened on an episode whose budget seals behind the stream.

    Refused where the generation is composed rather than noticed when a budget runs out, so no
    run can reach the state at all. The refusal is made before the client is touched, which is
    why this needs no service.
    """
    pytest.importorskip("temporalio")
    from shogym.serve.protocol_v2.gateway import open_gateway

    episode = await ServedEpisode.start(ENV_NAME, task=0)
    try:
        with pytest.raises(ValueError, match="ends_on_horizon=False"):
            await open_gateway(None, episode)  # type: ignore[arg-type]
    finally:
        await episode.close()
