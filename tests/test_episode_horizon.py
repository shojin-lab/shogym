"""Who ends an episode when its step budget runs out.

An episode that owns its own ending seals and grades when the budget is spent, which is the
only ending a caller driving it directly has. An episode served through the durable stream does
not: the stream is what ends an attempt there, and an ending decided inside the episode would
seal the environment and take a verdict off it while the stream still held the attempt open,
with nothing in the stream's record saying it had happened.

So the flag is checked from both sides. The episode is asked what it does with a spent budget,
and the gateway is asked what it does with an episode that would end itself.

The budget is still spent and it is still enforced. The gateway counts the calls that spend it,
because they never reach the stream, and when it is gone the attempt is ended through the stream
that owns it. The env is not sealed and not graded: what ends is the attempt, in the one record
that knows what an attempt is.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, List

import pytest

from shogym.serve import ServedEpisode

from tests._fixtures import score_env, score_mcp
from tests._fixtures.score_env import ENV_NAME, HORIZON

ATTEMPT = "a" * 32
MESSAGE = "b" * 32
CURSOR = "c" * 32


@dataclass
class _Ack:
    """What a presentation answers with. The gateway reads the cursor and nothing else."""

    cursor: str


class _ScriptedStream:
    """A stream that offers one task and records what the gateway asked it to do.

    It stands in for the durable stream so this file stays about the budget. What the real one
    does with a finalization is what the kernel tests drive.
    """

    def __init__(self, task: Any) -> None:
        self._task = task
        self.finalized: List[Any] = []
        self.attempts: dict = {}
        self.cursor = CURSOR
        self.held: Any = None

    async def pull(self, request: Any) -> Any:
        return self._task

    async def present(self, message: Any, **blobs: Any) -> _Ack:
        return _Ack(cursor=message.message_id)

    async def finalize(self, request: Any) -> Any:
        self.finalized.append(request)
        # The attempt is over, which is the fact the gateway routes by and reads back from here.
        self.attempts[request.attempt_id] = "final_failed"
        return request

    async def begin_environment_call(self, call: Any) -> Any:
        """Decide one environment call, which is the one call a stream never sees.

        An attempt this stream has ended is one it does not hold the generation for, so the
        call that would reach a world after the budget is spent is refused here. The refusal is
        imported where it is used, because this file also tests an episode that needs no
        durable service and the kernel is where that service lives.
        """
        from shogym.serve.protocol_v2.kernel import StreamProtocolError

        if self.attempts.get(call.attempt_id) != "active":
            raise StreamProtocolError("invalid_attempt")
        self.held = call.call_id
        return SimpleNamespace(call_id=call.call_id, attempt_id=call.attempt_id, held=True)

    async def end_environment_call(self, call: Any) -> Any:
        held, self.held = self.held == call.call_id, None
        return SimpleNamespace(call_id=call.call_id, attempt_id=call.attempt_id, held=held)

    async def stream_state(self) -> Any:
        return SimpleNamespace(
            cursor=self.cursor,
            generation_state="open",
            attempts=dict(self.attempts),
            stream_state_sha256="0" * 64,
            # Every message it offers is handed over, so it is never holding one for a request
            # that has not collected it.
            pending_message_id=None,
            pending_kind=None,
        )

    async def commit_presentation(self, commit: Any) -> _Ack:
        self.cursor = commit.message_id
        if commit.task_start_checkpoint_blob is not None:
            self.attempts[self._task.attempt_id] = "active"
        return _Ack(cursor=commit.message_id)


async def _refused(awaitable: Any) -> str:
    """Return the protocol error code a refused tool call carries."""
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001 - the code is the assertion
        return json.loads(str(error))["code"]
    raise AssertionError("the call was accepted")


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


async def test_an_episode_served_by_the_stream_is_ended_when_its_budget_runs_out() -> None:
    """The budget is spent, and the attempt ends, in the record that holds attempts.

    This is the other half of the test above. The episode decides nothing when its budget runs
    out, and what happens instead is that the transport counting those calls tells the stream to
    end the attempt. Nothing is sealed here and nothing is graded, so the env's world is left
    exactly as the last call left it, and the model reads that last call's own observation.
    """
    pytest.importorskip("temporalio")
    from shogym.serve.protocol_v2.gateway import StreamGateway, terminal_manifest
    from shogym.serve.protocol_v2.kernel import OfferedMessage

    env = score_env._FixtureScoreEnv()
    episode = await ServedEpisode.open_env(
        env, env_name=ENV_NAME, task=0, ends_on_horizon=False
    )
    try:
        spec = episode.describe()
        assert spec.horizon == HORIZON
        stream = _ScriptedStream(
            OfferedMessage(
                message_id=MESSAGE, kind="task", visible_text="{}", attempt_id=ATTEMPT
            )
        )
        gateway = StreamGateway(
            stream,  # type: ignore[arg-type]
            episode,
            spec,
            terminal_manifest(spec),
            initial_cursor=CURSOR,
        )
        await gateway.pull({})
        for _ in range(HORIZON):
            assert stream.finalized == []
            await gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}})

        [request] = stream.finalized
        assert request.attempt_id == ATTEMPT
        assert request.reason == "step_cap"
        # The env sealed nothing and graded nothing: the ending is not this episode's to make.
        assert env.finalize_calls == 0
        assert episode.sealed is False
        # And the attempt is not one this transport routes to any more.
        code = await _refused(
            gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}})
        )
        assert code == "invalid_attempt"
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
