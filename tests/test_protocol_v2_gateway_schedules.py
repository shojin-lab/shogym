"""A scheduled generation, served to a model through the gateway.

The stream's own tests drive the workflow. These drive what a model would actually call: one
``pull`` tool, one terminal tool, and nothing else. That is the difference worth testing here,
because the schedule is invisible from the model's side and has to stay that way: a twelve task
generation under Immediate and the same one under Never differ only in what arrives between the
tasks, and neither shows a tool, a field, or a count that says so.

The gateway is built in process against the durable service. They are marked ``network``
because that service downloads a test server on first use, and they skip when it is not there.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Sequence, Tuple

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from fastmcp import Client  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    IMMEDIATE,
    NEVER,
    PAYLOAD_FIRST,
    RELEASE_AT_SEAL,
    EligibilityGate,
    ReleasePlan,
    WireFormatError,
    check_release,
)
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    PULL_TOOL,
    StreamGateway,
    build_gateway_server,
    open_gateway,
    stream_start,
    terminal_manifest,
)
from shogym.serve.protocol_v2.kernel import OfferedMessage, TaskItem, stream_worker  # noqa: E402

from tests._fixtures import score_env, score_mcp  # noqa: E402

TEST_ENV = "wordle_v1"
# The env whose terminal takes an argument and whose per-session state can be read from out
# here, which is what lets a test say which world a task was worked in.
FIXTURE_ENV = score_env.ENV_NAME
DOSE = 12
CLAIM_HASH = "d" * 64


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await WorkflowEnvironment.start_time_skipping()
    except Exception as error:  # noqa: BLE001 - an absent test server is a skip, not a failure
        pytest.skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


@pytest_asyncio.fixture
async def serving(env: WorkflowEnvironment) -> AsyncIterator[WorkflowEnvironment]:
    """The environment with a Worker on it, which is what lets a generation make progress."""
    async with stream_worker(env.client):
        yield env


@pytest_asyncio.fixture
async def episode() -> AsyncIterator[ServedEpisode]:
    started = await ServedEpisode.start(TEST_ENV, task=0)
    try:
        yield started
    finally:
        await started.close()


async def refused(awaitable: Any) -> str:
    """Return the protocol error code a refused call carries."""
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001 - the code is the assertion
        record = json.loads(str(error))
        assert record["kind"] == "protocol_error"
        return record["code"]
    raise AssertionError("the call was accepted")


async def wordle_world(attempt_id: str) -> ServedEpisode:
    """One task's world. A generation of twelve tasks is twelve of these, one after another."""
    return await ServedEpisode.start(TEST_ENV, task=0)


async def score_world() -> ServedEpisode:
    """One task's world in the env whose per-session state can be read from out here."""
    return await ServedEpisode.start(FIXTURE_ENV, task=0)


async def opened(
    environment: WorkflowEnvironment,
    episode: ServedEpisode,
    *,
    workflow_id: str,
    bodies: Tuple[str, ...],
    release: Any,
    open_episode: Any = wordle_world,
) -> StreamGateway:
    """Compose a generation, bind this transport to it, and close its manifest."""
    spec = episode.describe()
    start = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        bodies=list(bodies),
        release=release,
    )
    gateway = await open_gateway(
        environment.client,
        episode,
        workflow_id=workflow_id,
        start=start,
        open_episode=open_episode,
    )
    # The controller closes the queue. A transport connecting is not what makes a run stop
    # accepting work, so the queue is open until this call, which is what is read here: a
    # gateway that closed it at open would serve every one of these tests just as well.
    assert (await gateway.stream_state()).queue_closed is False
    await gateway.close_queue()
    assert (await gateway.stream_state()).queue_closed is True
    return gateway


async def served(gateway: StreamGateway, *, limit: int = 200) -> List[Dict[str, Any]]:
    """Drive the generation to Done through the model's own tools, and report what came back.

    The loop is the whole of the model's behavior: pull, file when the result is a task, and
    pull again. Nothing in it knows what schedule is running.
    """
    seen: List[Dict[str, Any]] = []
    for _ in range(limit):
        record = json.loads(await gateway.pull({}))
        seen.append(record)
        if record["kind"] == "done":
            return seen
        if record["kind"] == "task":
            seen.append(
                json.loads(
                    await gateway.terminal(
                        {"attempt_id": record["attempt_id"], "arguments": {}}
                    )
                )
            )
    raise AssertionError("the generation never reached Done")


@pytest.mark.network
async def test_a_dose_of_twelve_tasks_and_their_payloads(serving, episode) -> None:
    """Task, acknowledgement, payload, twelve times, and then Done, as a model would see it."""
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-immediate/1",
        bodies=tuple(f"Round {index}." for index in range(DOSE)),
        release=IMMEDIATE,
    )
    seen = await served(gateway)
    assert [record["kind"] for record in seen] == ["task", "seal_ack", "payload"] * DOSE + ["done"]
    for position in range(DOSE):
        trio = seen[position * 3 : position * 3 + 3]
        assert len({record["attempt_id"] for record in trio}) == 1
    # Nothing the model read named a schedule, a position, or a plan.
    for record in seen:
        assert set(record) <= {
            "protocol_version",
            "kind",
            "message_id",
            "attempt_id",
            "body",
            "submission_digest",
            "canonicalization_version",
        }

    state = await gateway.stream_state()
    assert state.payload_delivery_count == DOSE
    assert state.assignment_count == DOSE
    assert await refused(gateway.pull({})) == "closed_stream"


@pytest.mark.network
async def test_the_same_dose_under_never_delivers_nothing(serving, episode) -> None:
    """The same twelve tasks with no payload between them, through the same tools."""
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-never/1",
        bodies=tuple(f"Round {index}." for index in range(DOSE)),
        release=NEVER,
    )
    seen = await served(gateway)
    assert [record["kind"] for record in seen] == ["task", "seal_ack"] * DOSE + ["done"]

    state = await gateway.stream_state()
    assert state.obligations == {}
    assert state.payload_delivery_count == 0
    assert state.assignment_count == DOSE


@pytest.mark.network
async def test_every_task_is_worked_in_a_world_of_its_own(serving) -> None:
    """Three tasks, three worlds. What a task inherits from the task before it is nothing.

    The seal captures what an attempt left behind, so an attempt that started in the world its
    predecessor was sealed in would file that predecessor's work a second time, and an
    environment that stops its own world at the seal would have nothing left to serve at all.
    Each task opens a world of its own when it is presented and it is closed when the attempt
    is sealed, which is what the fixture's session state shows from out here.
    """
    worlds: List[ServedEpisode] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        """The opener, and the one call that makes the world the generation is opened on."""
        started = await score_world()
        worlds.append(started)
        return started

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-per-task-world/1",
        bodies=("Round 0.", "Round 1.", "Round 2."),
        release=IMMEDIATE,
        open_episode=open_world,
    )
    for position in range(3):
        task = json.loads(await gateway.pull({}))
        assert task["kind"] == "task"
        # The world this task was given is the one its ordinary calls reach, and it is the only
        # one still open: everything before it was closed as its attempt was sealed.
        assert len(worlds) == position + 1
        live = worlds[position]
        assert score_mcp.gold(live.session_id) == "4"
        assert [score_mcp.gold(world.session_id) for world in worlds[:position]] == [""] * position

        wrapper = {"attempt_id": task["attempt_id"], "arguments": {}}
        assert json.loads((await gateway.environment("noop", wrapper)).content[0].text)["ok"]
        assert len(live._trajectory) == 1

        filing = {"attempt_id": task["attempt_id"], "arguments": {"answer": "4"}}
        assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
        assert json.loads(await gateway.pull({}))["kind"] == "payload"

    assert json.loads(await gateway.pull({}))["kind"] == "done"
    # One world per task, and none of them left open. What each one was asked to do is checked
    # while it is open, above: closing an episode runs the env's own end of it, which for a
    # score env is a finalization, and that leaves a mark on the trajectory of its own.
    assert len(worlds) == 3
    assert len({world.session_id for world in worlds}) == 3
    assert [score_mcp.gold(world.session_id) for world in worlds] == ["", "", ""]


@pytest.mark.network
async def test_a_task_is_sealed_in_the_world_its_own_calls_reached(serving) -> None:
    """The attempt a world was opened for is said out loud, so a terminal can find that world.

    Sealing an attempt belongs to the environment rather than to this transport, and an
    environment that seals by stopping the world it started is bound to one world when its
    Activities are registered: the world the generation was opened on, which is the first
    attempt's. Every attempt after that one works in a world this gateway opened, and the
    opener is the one moment anything outside can hear which attempt it was opened for. So a
    route built there has to name, for each attempt, the world that attempt's own calls
    reached, and a seal sent anywhere else would file a world its attempt never worked in.
    """
    worlds: Dict[str, ServedEpisode] = {}
    route: Dict[str, str] = {}

    generation_world = await score_world()
    worlds[generation_world.session_id] = generation_world

    async def open_world(attempt_id: str) -> ServedEpisode:
        started = await score_world()
        worlds[started.session_id] = started
        route[attempt_id] = started.session_id
        return started

    def sealed_in(attempt_id: str) -> str:
        """Where a terminal bound at registration would send this attempt's seal."""
        session_id = route.get(attempt_id, generation_world.session_id)
        if score_mcp.gold(session_id) == "":
            raise AssertionError(f"session {session_id} has no open world left to seal")
        return session_id

    gateway = await opened(
        serving,
        generation_world,
        workflow_id="stream/gateway-seal-route/1",
        bodies=("Round 0.", "Round 1.", "Round 2."),
        release=IMMEDIATE,
        open_episode=open_world,
    )
    for position in range(3):
        task = json.loads(await gateway.pull({}))
        attempt = task["attempt_id"]
        # The generation's own world is the first attempt's and the opener names the rest.
        assert (attempt in route) is (position > 0)

        wrapper = {"attempt_id": attempt, "arguments": {}}
        assert json.loads((await gateway.environment("noop", wrapper)).content[0].text)["ok"]
        # The world that seal would reach is the world this attempt's ordinary call reached.
        assert len(worlds[sealed_in(attempt)]._trajectory) == 1

        filing = {"attempt_id": attempt, "arguments": {"answer": "4"}}
        assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
        assert json.loads(await gateway.pull({}))["kind"] == "payload"

    assert json.loads(await gateway.pull({}))["kind"] == "done"
    assert len(worlds) == 3
    assert len(route) == 2


@pytest.mark.network
async def test_a_world_that_would_not_open_leaves_the_task_where_it_was(serving, episode) -> None:
    """A world is opened before the Presentation that reports the task, not after it.

    Starting one reaches something outside this process and can fail. If it failed after the
    commit the task would have been presented to nobody: the durable record would say the model
    is working position two while no world exists and no bytes ever reached it, and the call
    that came back for them would present that task a second time.
    """
    failures = ["the world would not start"]

    async def open_world(attempt_id: str) -> ServedEpisode:
        if failures:
            raise RuntimeError(failures.pop())
        return await wordle_world(attempt_id)

    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-world-will-not-open/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
    )
    first = json.loads(await gateway.pull({}))
    await gateway.terminal({"attempt_id": first["attempt_id"], "arguments": {}})
    assert json.loads(await gateway.pull({}))["kind"] == "payload"

    with pytest.raises(RuntimeError, match="would not start"):
        await gateway.pull({})
    # The stream is still holding the task, so nothing was presented and no attempt was started
    # in a world that does not exist.
    state = await gateway.stream_state()
    assert state.pending_kind == "task"
    assert state.pending_message_id is not None

    second = json.loads(await gateway.pull({}))
    assert second["kind"] == "task"
    assert second["body"] == "Round 1."
    await gateway.terminal({"attempt_id": second["attempt_id"], "arguments": {}})
    assert json.loads(await gateway.pull({}))["kind"] == "payload"
    assert json.loads(await gateway.pull({}))["kind"] == "done"
    await gateway.aclose()


@pytest.mark.network
async def test_a_world_that_would_not_close_leaves_the_acknowledgement_where_it_was(
    serving, episode
) -> None:
    """The world an attempt sealed is let go of before the acknowledgement is committed.

    The same rule from the other side. A seal that could not release its world is a filing this
    transport has not finished, and the model reads it when it has: the acknowledgement is the
    message the stream is still holding, and the filing that comes back for it is the one that
    gets it.
    """
    closes: List[int] = []

    class WillNotStopOnce:
        """A world whose cleanup fails once. Everything else about it is the episode it wraps."""

        def __init__(self, wrapped: ServedEpisode) -> None:
            self._wrapped = wrapped

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

        async def close(self) -> None:
            closes.append(len(closes))
            if len(closes) == 1:
                raise RuntimeError("the world would not stop")
            await self._wrapped.close()

    async def open_world(attempt_id: str) -> Any:
        return WillNotStopOnce(await wordle_world(attempt_id))

    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-world-will-not-close/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
    )
    first = json.loads(await gateway.pull({}))
    await gateway.terminal({"attempt_id": first["attempt_id"], "arguments": {}})
    assert json.loads(await gateway.pull({}))["kind"] == "payload"
    second = json.loads(await gateway.pull({}))
    filing = {"attempt_id": second["attempt_id"], "arguments": {}}

    with pytest.raises(RuntimeError, match="would not stop"):
        await gateway.terminal(filing)
    state = await gateway.stream_state()
    assert state.pending_kind == "seal_ack"

    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    assert closes == [0, 1]
    assert json.loads(await gateway.pull({}))["kind"] == "payload"
    assert json.loads(await gateway.pull({}))["kind"] == "done"
    await gateway.aclose()


async def test_a_world_is_retired_when_its_attempt_ends_without_a_seal() -> None:
    """A world belongs to the attempt it was opened for, and a seal is not the only ending.

    An attempt the generation finalized for itself, on a step cap or on a deadline, presents no
    acknowledgement: nothing about that ending arrives here as a message, and the first thing
    this transport sees of it is the next task arriving under a different attempt. That task
    still gets a world of its own, because the one in hand is not its.
    """
    worlds: List[ServedEpisode] = []
    opened_for: List[str] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        started = await score_world()
        worlds.append(started)
        opened_for.append(attempt_id)
        return started

    first = await open_world()
    spec = first.describe()
    gateway = StreamGateway(
        None,  # type: ignore[arg-type]
        first,
        spec,
        terminal_manifest(spec),
        initial_cursor="0" * 32,
        open_episode=open_world,
    )
    try:
        await gateway._prepared(task_presentation("a" * 32))
        assert len(worlds) == 1
        # Nothing sealed that attempt, so nothing closed its world.
        await gateway._prepared(task_presentation("b" * 32))
        assert len(worlds) == 2
        # The world it opened was opened for the attempt that is now in front of the model,
        # which is the only place anything out here can learn that.
        assert opened_for == ["", "b" * 32]
        assert score_mcp.gold(worlds[0].session_id) == ""
        assert score_mcp.gold(worlds[1].session_id) == "4"
    finally:
        await gateway.aclose()


def task_presentation(attempt_id: str) -> OfferedMessage:
    """The message a task is presented as, which is all this transport learns about one."""
    return OfferedMessage(
        message_id="m" * 32, kind="task", visible_text="{}", attempt_id=attempt_id
    )


async def test_a_generation_of_more_than_one_task_needs_a_world_for_each(
    episode: ServedEpisode,
) -> None:
    """One episode is one task's world, so composing twelve tasks against one is refused.

    The refusal is at composition, before a stream exists. A run that served the second task in
    the world the first one sealed would report a successful dose whose every position after
    the first was worked and scored in somebody else's world.
    """
    spec = episode.describe()
    start = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        bodies=[f"Round {index}." for index in range(DOSE)],
    )
    with pytest.raises(ValueError, match="a world of its own"):
        await open_gateway(None, episode, start=start)  # type: ignore[arg-type]


async def test_an_evaluation_only_generation_is_pinned_to_never(episode: ServedEpisode) -> None:
    """A generation that scores without delivering cannot be composed with an outbox."""
    spec = episode.describe()
    terminal = terminal_manifest(spec)
    pinned = stream_start(spec, terminal, claim_hash=CLAIM_HASH, evaluation_only=True)
    assert pinned.evaluation_only
    assert pinned.release.release_plan_id == NEVER.release_plan_id
    with pytest.raises(ValueError, match="must be "):
        stream_start(
            spec, terminal, claim_hash=CLAIM_HASH, release=IMMEDIATE, evaluation_only=True
        )


async def test_the_roster_a_composed_generation_carries(episode: ServedEpisode) -> None:
    """The queue and the roster are built together, so no row names a position the queue lacks."""
    spec = episode.describe()
    start = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        bodies=[f"Round {index}." for index in range(DOSE)],
    )
    assert [row.task_position for row in start.assignments] == list(range(DOSE))
    assert [row.payload_position for row in start.assignments] == list(range(DOSE))
    assert all(row.release_plan_id == IMMEDIATE.release_plan_id for row in start.assignments)
    assert [row.attempt_id for row in start.assignments] == [
        item.attempt_id for item in start.tasks
    ]
    assert len({row.assignment_id for row in start.assignments}) == DOSE


async def test_a_leg_is_composed_with_the_attempts_its_plan_gates(episode: ServedEpisode) -> None:
    """A plan gates a task by attempt ID, and the attempt IDs are minted here.

    So a composer that gates anything hands over a plan for the queue rather than a finished
    one: it is called with the tasks, in order, once they exist. This is the delayed transfer
    leg. A carries the payload, the filler waits for it, B waits for the filler to seal, and
    neither the filler nor B is a position anything is delivered against.
    """

    def leg(tasks: Sequence[TaskItem]) -> ReleasePlan:
        return ReleasePlan(
            RELEASE_AT_SEAL,
            PAYLOAD_FIRST,
            BY_POSITION,
            gates=[
                EligibilityGate(tasks[2].attempt_id, after_payload_position=0),
                EligibilityGate(
                    tasks[1].attempt_id, after_sealed_attempt_id=tasks[2].attempt_id
                ),
            ],
        )

    spec = episode.describe()
    start = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        bodies=["A.", "B.", "The filler."],
        release=leg,
        without_payload=(1, 2),
    )
    minted = [item.attempt_id for item in start.tasks]
    assert [gate.attempt_id for gate in start.release.gates] == [minted[2], minted[1]]
    assert start.release.gates[1].after_sealed_attempt_id == minted[2]
    assert [row.creates_payload_obligation for row in start.assignments] == [True, False, False]
    # The generation the stream would accept is the generation this returned.
    check_release(start.release, start.assignments, evaluation_only=False)


async def test_a_generation_is_refused_where_it_is_built(episode: ServedEpisode) -> None:
    """A plan the stream would refuse at start is refused here, where the composer reads it."""
    spec = episode.describe()
    terminal = terminal_manifest(spec)
    with pytest.raises(WireFormatError, match="a gate names a task"):
        stream_start(
            spec,
            terminal,
            claim_hash=CLAIM_HASH,
            bodies=["A.", "B."],
            release=lambda tasks: ReleasePlan(
                RELEASE_AT_SEAL,
                PAYLOAD_FIRST,
                BY_POSITION,
                gates=[EligibilityGate("f" * 32, after_payload_position=0)],
            ),
        )
    with pytest.raises(ValueError, match="this queue has 2"):
        stream_start(
            spec, terminal, claim_hash=CLAIM_HASH, bodies=["A.", "B."], without_payload=(2,)
        )


async def test_the_controller_calls_are_not_tools(episode: ServedEpisode) -> None:
    """Closing the queue and reading the counts are the controller's. The model has neither."""
    spec = episode.describe()
    gateway = StreamGateway(
        None,  # type: ignore[arg-type]
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor="0" * 32,
    )
    async with Client(build_gateway_server(gateway)) as client:
        assert {tool.name for tool in await client.list_tools()} == {
            PULL_TOOL,
            "guess",
            "terminate",
        }
