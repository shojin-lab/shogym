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
from hashlib import sha256
from types import SimpleNamespace
from dataclasses import dataclass
from typing import Any, List

import pytest

from shogym.serve import ServedEpisode
from shogym.serve.protocol_v2.gateway import stream_start

from tests._fixtures import score_env, score_mcp
from tests._fixtures.score_env import ENV_NAME, HORIZON

ATTEMPT = "a" * 32
MESSAGE = "b" * 32
CURSOR = "c" * 32
ACK = "d" * 32


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
        self.sealed: List[Any] = []
        self.attempts: dict = {}
        # The grants it made per attempt, which is what the real stream counts and what a
        # gateway reads the spent budget out of.
        self.environment_calls: dict = {}
        self.cursor = CURSOR
        self.held: Any = None

    async def pull(self, request: Any) -> Any:
        return self._task

    async def seal(self, request: Any) -> Any:
        """Take one filing and offer its acknowledgement, the way the real stream does.

        What the seal and the grade behind it come to is the kernel's, and it is driven where the
        kernel is. What this has to be faithful about is the shape: a filing is accepted, an
        attempt stops being one the transport routes to, and bytes come back to be presented.
        """
        from shogym.serve.protocol_v2 import SealAck, visible_bytes
        from shogym.serve.protocol_v2.kernel import OfferedMessage

        self.sealed.append(request)
        attempt_id = request.metadata.attempt_id
        ack = SealAck(
            message_id=ACK,
            attempt_id=attempt_id,
            submission_digest="d" * 64,
            canonicalization_version="shogym.fixture.1",
        )
        return OfferedMessage(
            message_id=ACK,
            kind="seal_ack",
            visible_text=visible_bytes(ack).decode("utf-8"),
            attempt_id=attempt_id,
        )

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
        # The grant is the whole of what this stream ever learns about that world, so it is
        # counted here the way the durable one counts it.
        self.environment_calls[call.attempt_id] = self.environment_calls.get(call.attempt_id, 0) + 1
        return SimpleNamespace(call_id=call.call_id, attempt_id=call.attempt_id, held=True)

    async def end_environment_call(self, call: Any) -> Any:
        held, self.held = self.held == call.call_id, None
        return SimpleNamespace(call_id=call.call_id, attempt_id=call.attempt_id, held=held)

    async def stream_state(self) -> Any:
        return SimpleNamespace(
            cursor=self.cursor,
            generation_state="open",
            attempts=dict(self.attempts),
            environment_calls=dict(self.environment_calls),
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
            self.environment_calls[self._task.attempt_id] = 0
        if commit.message_id == ACK:
            self.attempts[self._task.attempt_id] = "ack_presented"
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
    episode = await ServedEpisode.open_env(env, env_name=ENV_NAME, task=0, ends_on_horizon=False)
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
    """The budget runs out, and the attempt ends, in the record that holds attempts.

    This is the other half of the test above. The episode decides nothing about its budget, and
    what happens instead is that the transport counting those calls tells the stream to end the
    attempt. It says so on the call that has nothing left to spend rather than on the one that
    spends the last of it: an attempt out of world calls can still be filed, and filing under
    this protocol is a call to the stream. Nothing is sealed here and nothing is graded, so the
    env's world is left exactly as the last call left it.
    """
    pytest.importorskip("temporalio")
    from shogym.serve.protocol_v2.gateway import StreamGateway, terminal_manifest
    from shogym.serve.protocol_v2.kernel import OfferedMessage

    env = score_env._FixtureScoreEnv()
    episode = await ServedEpisode.open_env(env, env_name=ENV_NAME, task=0, ends_on_horizon=False)
    try:
        spec = episode.describe()
        assert spec.horizon == HORIZON
        stream = _ScriptedStream(
            OfferedMessage(message_id=MESSAGE, kind="task", visible_text="{}", attempt_id=ATTEMPT)
        )
        start = stream_start(
            spec,
            terminal_manifest(spec),
            claim_hash=sha256(b"a claim").hexdigest(),
            evaluation_only=True,
        )
        gateway = StreamGateway(
            stream,  # type: ignore[arg-type]
            episode,
            spec,
            terminal_manifest(spec),
            initial_cursor=CURSOR,
            generation=start,
        )
        await gateway.pull({})
        for _ in range(HORIZON):
            assert stream.finalized == []
            await gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}})

        # The call with nothing left to spend is where the ending is. It reaches no world and is
        # refused, and the attempt is not one this transport routes to afterwards either.
        assert stream.finalized == []
        assert len(episode._trajectory) == HORIZON
        code = await _refused(gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}}))
        assert code == "invalid_attempt"
        assert len(episode._trajectory) == HORIZON

        [request] = stream.finalized
        assert request.attempt_id == ATTEMPT
        assert request.reason == "step_cap"
        # The env sealed nothing and graded nothing: the ending is not this episode's to make.
        assert env.finalize_calls == 0
        assert episode.sealed is False

        # And one ending, whatever the model does next.
        code = await _refused(gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}}))
        assert code == "invalid_attempt"
        assert len(stream.finalized) == 1
        assert env.finalize_calls == 0
    finally:
        await episode.close()


async def test_a_graded_horizon_files_the_environments_terminal_on_the_last_call() -> None:
    """The other budget rule: the call that spends the last step is the one that ends the attempt.

    An environment whose horizon is an ending its own scorer answers for is not served the floor.
    There is no call after the last one, because the attempt was filed as that call's step
    committed: the filing says the horizon made it, it carries nothing the agent authored, and
    the acknowledgement comes back in the same result as the observation. What is not here is a
    finalization: nothing floored, nothing ended without a filing.
    """
    pytest.importorskip("temporalio")
    from shogym.serve.protocol_v2.gateway import (
        CANONICALIZATION_VERSION,
        EnvironmentTerminal,
        StreamGateway,
        WorldRoute,
        terminal_manifest,
    )
    from shogym.serve.protocol_v2.kernel import OfferedMessage
    from shogym.serve.protocol_v2.policy import KERNEL_STAND_IN_GRADE

    env = score_env._FixtureScoreEnv()
    episode = await ServedEpisode.open_env(env, env_name=ENV_NAME, task=0, ends_on_horizon=False)
    try:
        spec = episode.describe()
        stream = _ScriptedStream(
            OfferedMessage(message_id=MESSAGE, kind="task", visible_text="{}", attempt_id=ATTEMPT)
        )
        start = stream_start(
            spec,
            terminal_manifest(spec),
            claim_hash=sha256(b"a claim").hexdigest(),
            evaluation_only=True,
        )
        gateway = StreamGateway(
            stream,  # type: ignore[arg-type]
            episode,
            spec,
            terminal_manifest(spec),
            initial_cursor=CURSOR,
            generation=start,
            environment=EnvironmentTerminal(
                CANONICALIZATION_VERSION,
                [],
                None,
                WorldRoute(),
                KERNEL_STAND_IN_GRADE,
                "graded",
            ),
        )
        await gateway.pull({})
        for _ in range(HORIZON - 1):
            result = await gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}})
            assert stream.sealed == []
            assert len(result.content) == 1

        # The last step. Its own observation comes back first, and the acknowledgement of the
        # filing that step ended is the second item of the same result.
        result = await gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}})
        assert len(episode._trajectory) == HORIZON
        assert len(result.content) == 2
        assert json.loads(result.content[1].text)["kind"] == "seal_ack"

        [filing] = stream.sealed
        assert filing.metadata.attempt_id == ATTEMPT
        assert filing.terminal_source == "horizon"
        assert filing.native_arguments == {}
        assert stream.finalized == []

        # And the attempt is over: nothing this transport routes to, and one filing however
        # many calls the model makes afterwards.
        code = await _refused(gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}}))
        assert code == "invalid_attempt"
        assert len(stream.sealed) == 1
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
