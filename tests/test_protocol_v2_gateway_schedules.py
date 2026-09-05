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

import asyncio
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Set, Tuple

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from fastmcp import Client  # noqa: E402
from temporalio.client import WorkflowExecutionStatus  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import (  # noqa: E402
    BY_POSITION,
    IMMEDIATE,
    NEVER,
    PAYLOAD_FIRST,
    RELEASE_AT_SEAL,
    TASK_FIRST,
    Done,
    EligibilityGate,
    Info,
    Payload,
    ReleasePlan,
    SealAck,
    SealReject,
    Task,
    Wait,
    WireFormatError,
    check_release,
)
from shogym.serve.protocol_v2 import gateway as gateway_module  # noqa: E402
from shogym.serve.protocol_v2 import records as records_module  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    CANONICALIZATION_VERSION,
    INFO_TOOL,
    PULL_TOOL,
    EnvironmentTerminal,
    StreamGateway,
    WorldRoute,
    build_gateway_server,
    environment_terminal,
    open_gateway,
    served_manifest,
    stream_start,
    terminal_manifest,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    BLINDED_RECEIPT_V1,
    DELIVER,
    EXPERIMENT,
    KERNEL_STAND_IN_GRADE,
    ORDINARY,
    WITHHOLD,
    PayloadDisposition,
    policy_digest,
)
from shogym.envs.wordle.protocol_v2 import WORDLE_GRADE  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    DEADLINE,
    STREAM_TASK_QUEUE,
    OfferedMessage,
    StreamWorkflow,
    TaskItem,
    configuration_hash,
    kernel_activities,
    resume_run_directory,
    start_stream,
    stream_worker,
)
from shogym.serve.protocol_v2.rundir import (  # noqa: E402
    MANIFEST_FILE,
    ResumeRefused,
    create_run_directory,
    open_run_directory,
    staged_generation,
)

from tests._fixtures import score_env, score_mcp  # noqa: E402

TEST_ENV = "wordle_v1"
# The env whose terminal takes an argument and whose per-session state can be read from out
# here, which is what lets a test say which world a task was worked in.
FIXTURE_ENV = score_env.ENV_NAME
DOSE = 12
CLAIM_HASH = "d" * 64

# What each kind of record the model reads is made of, written down per kind rather than as one
# union of every name any of them may carry. A union grows every time one record gains a field
# and stops saying anything about the others, which is the opposite of what it is here to say.
# `budget` is the one name a task may add to its entry, and no other kind may add anything.
# The counts are on the info record and nowhere else, which is what a per-kind table says and a
# union of every name would not: a task admitting `remaining` would be a task carrying a queue
# position, and this is the assertion that would fail.
PUBLIC_KEYS: Dict[str, Set[str]] = {
    "task": {"protocol_version", "kind", "message_id", "attempt_id", "body"},
    "payload": {"protocol_version", "kind", "message_id", "attempt_id", "body"},
    "wait": {"protocol_version", "kind", "message_id", "retry_after_ms"},
    "done": {"protocol_version", "kind", "message_id"},
    "seal_ack": {
        "protocol_version",
        "kind",
        "message_id",
        "attempt_id",
        "submission_digest",
        "canonicalization_version",
    },
    "seal_reject": {"protocol_version", "kind", "message_id", "attempt_id", "body", "code"},
    "info": {"protocol_version", "kind", "message_id", "remaining", "consumed", "in_flight"},
}
OPTIONAL_KEYS: Dict[str, Set[str]] = {"task": {"budget"}}

# One of every record the model can be shown, so the table above is checked against the records
# themselves and not only against the kinds one flow happens to produce. A kind that fell out of
# the table, or a record that grew a field, fails here rather than going unnoticed until some
# later flow serves it.
ONE_OF_EVERY_KIND: Tuple[Any, ...] = (
    Task(message_id="0" * 32, attempt_id="1" * 32, body="work"),
    Task(message_id="0" * 32, attempt_id="1" * 32, body="work", budget=52),
    Payload(message_id="0" * 32, attempt_id="1" * 32, body="a receipt"),
    Wait(message_id="0" * 32, retry_after_ms=1000),
    Done(message_id="0" * 32),
    SealAck(
        message_id="0" * 32,
        attempt_id="1" * 32,
        submission_digest="a" * 64,
        canonicalization_version="world.1",
    ),
    SealReject(message_id="0" * 32, attempt_id="1" * 32, body="arg is required"),
    Info(message_id="0" * 32, remaining=7, consumed=3, in_flight=1),
)


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
async def serving_wordle(
    env: WorkflowEnvironment,
) -> AsyncIterator[Tuple[WorkflowEnvironment, EnvironmentTerminal]]:
    """A service running wordle's own terminal, which is what a real wordle generation runs on.

    The Activities and the route come back together and both go to one place: the Worker
    registers the Activities and the gateway is opened on the terminal that holds their route,
    because a seal that cannot find the world an attempt was played in has nothing to capture.
    """
    episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        terminal = environment_terminal(episode)
    finally:
        await episode.close()
    async with stream_worker(env.client, activities=terminal.activities):
        yield env, terminal


@pytest_asyncio.fixture
async def episode() -> AsyncIterator[ServedEpisode]:
    started = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
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
    return await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)


async def score_world() -> ServedEpisode:
    """One task's world in the env whose per-session state can be read from out here."""
    return await ServedEpisode.start(FIXTURE_ENV, task=0, ends_on_horizon=False)


def kernel_environment() -> EnvironmentTerminal:
    """The stream's own stand-in terminal, declared as what these generations are served on.

    What is under test here is the schedule, so these generations keep the stand-ins: they
    compute from the shape of a filing, reach no world, and are the same for every task. The
    declaration says so rather than leaving it to be inferred, which is what lets the composition
    below register a body that reports nothing about the work.
    """
    return EnvironmentTerminal(
        CANONICALIZATION_VERSION,
        list(kernel_activities()),
        None,
        WorldRoute(),
        KERNEL_STAND_IN_GRADE,
    )


def blinded(rows: Sequence[Any]) -> List[PayloadDisposition]:
    """Register a body that says a filing was answered and nothing about how good it was.

    A generation over the stand-in grade has no grade to publish, so a fixture that wants
    payloads registers the blinded receipt for every position it owes one against and the reason
    it owes none for the rest. Both are rows in the record: this is an experiment saying what it
    delivers, which is the only way a body that conceals is served at all.
    """
    return [
        PayloadDisposition(
            attempt_id=row.attempt_id,
            payload_position=row.payload_position,
            kind=DELIVER,
            policy_digest=policy_digest(BLINDED_RECEIPT_V1),
            cell=BLINDED_RECEIPT_V1.cells[0],
        )
        if row.creates_payload_obligation
        else PayloadDisposition(
            attempt_id=row.attempt_id,
            payload_position=row.payload_position,
            kind=WITHHOLD,
            reason="the schedule under test delivers nothing here",
        )
        for row in rows
    ]


async def opened(
    environment: WorkflowEnvironment,
    episode: ServedEpisode,
    *,
    workflow_id: str,
    bodies: Tuple[str, ...],
    release: Any,
    open_episode: Any = wordle_world,
    terminal: Optional[EnvironmentTerminal] = None,
    budget: Optional[int] = None,
    capacity: int = 1,
    info: bool = False,
    attempt_deadline_ms: int = 0,
) -> StreamGateway:
    """Compose a generation, bind this transport to it, and close its manifest.

    A generation over an environment with a grader of its own is an ordinary run and delivers
    the honest body. One over the stand-ins has no grade to publish, so it registers the blinded
    receipt for the positions it owes and the reason for the ones it does not. Both are composed
    here rather than in each test, because what a schedule does is the same either way.
    """
    served_on = terminal if terminal is not None else kernel_environment()
    spec = episode.describe()
    start = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        bodies=list(bodies),
        release=release,
        grade=served_on.grade,
        profile=ORDINARY if not served_on.grade.stand_in else EXPERIMENT,
        dispositions=None if not served_on.grade.stand_in else blinded,
        experiment="" if not served_on.grade.stand_in else "what_a_schedule_does",
        budget=budget,
        capacity=capacity,
        info=info,
        attempt_deadline_ms=attempt_deadline_ms,
    )
    gateway = await open_gateway(
        environment.client,
        episode,
        workflow_id=workflow_id,
        start=start,
        open_episode=open_episode,
        environment=served_on,
    )
    # The controller closes the queue. A transport connecting is not what makes a run stop
    # accepting work, so the queue is open until this call, which is what is read here: a
    # gateway that closed it at open would serve every one of these tests just as well.
    assert (await gateway.stream_state()).queue_closed is False
    await gateway.close_queue()
    assert (await gateway.stream_state()).queue_closed is True
    return gateway


def test_the_public_key_set_is_pinned_for_every_record_the_model_can_read() -> None:
    """The table above is a claim about the wire and not about one flow's output.

    A served generation shows whichever kinds its schedule produces, so a kind that dropped out
    of the table, or one that grew a field, would go unnoticed until some later flow served it.
    Here every presentable record is built and compared to its entry, both task encodings among
    them, and the table is required to name exactly the kinds the wire can present.
    """
    presentable = {
        cls.__dataclass_fields__["kind"].default for cls in records_module._PRESENTABLE
    }
    assert set(PUBLIC_KEYS) == presentable
    assert set(OPTIONAL_KEYS) <= presentable
    assert {record.kind for record in ONE_OF_EVERY_KIND} == presentable
    for record in ONE_OF_EVERY_KIND:
        allowed = PUBLIC_KEYS[record.kind] | OPTIONAL_KEYS.get(record.kind, set())
        assert PUBLIC_KEYS[record.kind] <= set(record.to_wire()) <= allowed
    # And the optional half is exercised rather than merely allowed: one task carries it.
    assert any(set(record.to_wire()) > PUBLIC_KEYS[record.kind] for record in ONE_OF_EVERY_KIND)


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
async def test_a_dose_of_twelve_tasks_and_their_payloads(serving_wordle, episode) -> None:
    """Task, acknowledgement, payload, twelve times, and then Done, as a model would see it."""
    serving, terminal = serving_wordle
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-immediate/1",
        bodies=tuple(f"Round {index}." for index in range(DOSE)),
        release=IMMEDIATE,
        terminal=terminal,
    )
    seen = await served(gateway)
    assert [record["kind"] for record in seen] == ["task", "seal_ack", "payload"] * DOSE + ["done"]
    for position in range(DOSE):
        trio = seen[position * 3 : position * 3 + 3]
        assert len({record["attempt_id"] for record in trio}) == 1
    # Nothing the model read named a schedule, a position, or a plan. The set is pinned per kind
    # rather than over the union of them, so a key one record may carry is not thereby admitted on
    # every other one. The single key a task may add is `budget`, which this generation does not
    # declare: it is what an attempt may spend, one number for the whole generation and the same
    # on every task it serves, so nothing about where a task sits in a line can be read off it.
    for record in seen:
        assert set(record) == PUBLIC_KEYS[record["kind"]]

    state = await gateway.stream_state()
    assert state.payload_delivery_count == DOSE
    assert state.assignment_count == DOSE
    assert await refused(gateway.pull({})) == "closed_stream"


@pytest.mark.network
async def test_a_declared_budget_reaches_the_model_on_every_task(serving_wordle, episode) -> None:
    """The number the agent reads is the step cap this transport enforces.

    A generation that declares one hands it over with the work rather than making the agent
    discover it by running out, and it hands the same number over every time, because it is the
    environment's own step budget and not a fact about any one task.
    """
    serving, terminal = serving_wordle
    horizon = episode.describe().horizon
    assert horizon is not None
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-budget/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        terminal=terminal,
        budget=horizon,
    )
    seen = await served(gateway)
    tasks = [record for record in seen if record["kind"] == "task"]
    assert len(tasks) == 2
    for record in tasks:
        assert set(record) == PUBLIC_KEYS["task"] | OPTIONAL_KEYS["task"]
        assert record["budget"] == horizon
    # And nothing else the model read grew a key because a task did.
    for record in seen:
        if record["kind"] != "task":
            assert set(record) == PUBLIC_KEYS[record["kind"]]


@pytest.mark.network
async def test_a_declared_info_tool_answers_the_model_through_the_whole_stack(
    serving_wordle, episode
) -> None:
    """One real generation, one real transport, and the tool the model actually calls.

    Everything below this is real: a durable stream, the gateway that serves it, the MCP server
    the model reaches it through, and a wordle world per attempt. So what is checked is the whole
    path rather than any one layer's account of it: the tool is advertised, its answer arrives as
    the record the stream minted, the counts move as the work does, and every answer is in the
    ledger of what this generation committed, in the order the model was shown them.
    """
    serving, terminal = serving_wordle
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-info/1",
        bodies=("Round 0.", "Round 1."),
        release=NEVER,
        terminal=terminal,
        info=True,
    )
    server = build_gateway_server(gateway)
    async with Client(server) as client:
        assert INFO_TOOL in {tool.name for tool in await client.list_tools()}

        async def asked() -> Dict[str, Any]:
            answer = await client.call_tool(INFO_TOOL, {})
            assert len(answer.content) == 1
            return json.loads(answer.content[0].text)

        async def pulled() -> Dict[str, Any]:
            return json.loads((await client.call_tool(PULL_TOOL, {})).content[0].text)

        shown = [await asked()]
        first = await pulled()
        shown.append(first)
        shown.append(await asked())
        # A call into the world is work on an attempt and not a change to the queue, so the
        # same three numbers come back after it.
        await client.call_tool(
            "guess", {"attempt_id": first["attempt_id"], "arguments": {"word": "crane"}}
        )
        shown.append(await asked())
        filed = json.loads(
            (
                await client.call_tool(
                    gateway.terminal_tool, {"attempt_id": first["attempt_id"], "arguments": {}}
                )
            ).content[0].text
        )
        shown.append(filed)
        shown.append(await asked())

    assert [record["kind"] for record in shown] == [
        "info",
        "task",
        "info",
        "info",
        "seal_ack",
        "info",
    ]
    counts = [
        (record["remaining"], record["consumed"], record["in_flight"])
        for record in shown
        if record["kind"] == "info"
    ]
    # Nothing handed out, one being worked on, the same one twice because asking changes
    # nothing, and one handed out and ended after the filing.
    assert counts == [(2, 0, 0), (1, 1, 1), (1, 1, 1), (1, 1, 0)]
    for record in shown:
        assert set(record) == PUBLIC_KEYS[record["kind"]]

    # And every one of them is in the generation's own record of what it committed, in order.
    presentations = await gateway._stream.handle.query(StreamWorkflow.presented_messages)
    assert [row.kind for row in presentations] == [record["kind"] for record in shown]
    assert [row.message_id for row in presentations] == [record["message_id"] for record in shown]


@pytest.mark.network
async def test_the_same_dose_under_never_delivers_nothing(serving_wordle, episode) -> None:
    """The same twelve tasks with no payload between them, through the same tools."""
    serving, terminal = serving_wordle
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-never/1",
        bodies=tuple(f"Round {index}." for index in range(DOSE)),
        release=NEVER,
        terminal=terminal,
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
async def test_two_tasks_are_held_at_once_and_each_call_lands_in_its_own_world(serving) -> None:
    """A capacity above one is several live attempts, each working in a world of its own.

    The model holds two tasks and works them in whatever order it likes, so the world of the task
    that arrived last is not the world of the next call. Every call names its attempt and lands in
    that attempt's world, the seal of the older one closes that world and leaves the newer one
    working in its own, and the generation goes on to the task after them.
    """
    worlds: Dict[str, ServedEpisode] = {}
    order: List[ServedEpisode] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        started = await score_world()
        order.append(started)
        if attempt_id:
            worlds[attempt_id] = started
        return started

    generation_world = await open_world()
    gateway = await opened(
        serving,
        generation_world,
        workflow_id="stream/gateway-two-at-once/1",
        bodies=("Round 0.", "Round 1.", "Round 2."),
        release=IMMEDIATE,
        open_episode=open_world,
        capacity=2,
    )

    async def noop(attempt_id: str) -> None:
        wrapper = {"attempt_id": attempt_id, "arguments": {}}
        assert json.loads((await gateway.environment("noop", wrapper)).content[0].text)["ok"]

    first = json.loads(await gateway.pull({}))
    # The second task arrives with the first attempt still live, which is what the capacity buys.
    second = json.loads(await gateway.pull({}))
    assert first["kind"] == second["kind"] == "task"
    assert first["attempt_id"] != second["attempt_id"]
    worlds[first["attempt_id"]] = generation_world
    older, newer = worlds[first["attempt_id"]], worlds[second["attempt_id"]]
    assert older is not newer
    assert (await gateway.stream_state()).capacity_in_use == 2

    await noop(first["attempt_id"])
    await noop(second["attempt_id"])
    await noop(first["attempt_id"])
    assert len(older._trajectory) == 2
    assert len(newer._trajectory) == 1

    filing = {"attempt_id": first["attempt_id"], "arguments": {"answer": "4"}}
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    # The sealed attempt's world is let go of and the live attempt's is not.
    assert score_mcp.gold(older.session_id) == ""
    assert score_mcp.gold(newer.session_id) == "4"
    await noop(second["attempt_id"])
    assert len(newer._trajectory) == 2

    assert json.loads(await gateway.pull({}))["kind"] == "payload"
    filing = {"attempt_id": second["attempt_id"], "arguments": {"answer": "4"}}
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    assert score_mcp.gold(newer.session_id) == ""

    # And the third task is served like any other, in a world of its own, through to Done.
    assert json.loads(await gateway.pull({}))["kind"] == "payload"
    third = json.loads(await gateway.pull({}))
    assert third["kind"] == "task"
    filing = {"attempt_id": third["attempt_id"], "arguments": {"answer": "4"}}
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    assert json.loads(await gateway.pull({}))["kind"] == "payload"
    assert json.loads(await gateway.pull({}))["kind"] == "done"
    assert len(order) == 3
    assert len({world.session_id for world in order}) == 3


@pytest.mark.network
async def test_two_live_attempts_are_sealed_and_paid_out_from_their_own_worlds(
    serving_wordle,
) -> None:
    """The environment's own terminal ends each of two live attempts in the world it worked in.

    A stand-in terminal reaches no world, so a generation served on one cannot tell a right route
    from a wrong one. This one is served on wordle's, which seals by stopping the world an attempt
    played in: what it captures, what it grades and what the agent is paid out all come from that
    world. Two attempts are live at once, they are played differently, and they are sealed in the
    reverse of the order they were served, so a transport with one current world would file the
    wrong play against both of them.
    """
    serving, terminal = serving_wordle
    worlds: Dict[str, ServedEpisode] = {}

    async def open_world(attempt_id: str) -> ServedEpisode:
        started = await wordle_world(attempt_id)
        worlds[attempt_id] = started
        return started

    seed = await wordle_world("")
    gateway = await opened(
        serving,
        seed,
        workflow_id="stream/gateway-two-live-wordle/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        terminal=terminal,
        capacity=2,
    )

    async def guessed(attempt_id: str, word: str) -> None:
        wrapper = {"attempt_id": attempt_id, "arguments": {"word": word}}
        assert json.loads((await gateway.environment("guess", wrapper)).content[0].text)["score"]

    older = json.loads(await gateway.pull({}))["attempt_id"]
    newer = json.loads(await gateway.pull({}))["attempt_id"]
    worlds[older] = seed
    assert worlds[older] is not worlds[newer]
    # Played differently, so what each world holds is what says which one was read.
    await guessed(older, "crane")
    await guessed(newer, "slate")
    await guessed(newer, "adieu")
    assert (len(worlds[older]._trajectory), len(worlds[newer]._trajectory)) == (1, 2)
    # Both are where a seal would find them, and each is its own.
    assert terminal.route(older) is not None
    assert terminal.route(newer) is not None
    assert terminal.route(older) != terminal.route(newer)

    # Sealed in the reverse of the order they were served.
    sealed = {}
    for attempt_id in (newer, older):
        ack = json.loads(await gateway.terminal({"attempt_id": attempt_id, "arguments": {}}))
        assert ack["kind"] == "seal_ack"
        sealed[attempt_id] = ack["submission_digest"]
        payload = json.loads(await gateway.pull({}))
        assert payload["kind"] == "payload"
        assert payload["attempt_id"] == attempt_id
        # The receipt reports the play this attempt made in the world it made it in.
        assert f"guesses_used {len(worlds[attempt_id]._trajectory)}" in payload["body"]
        # And the world it was sealed in is gone, while anything still live is not.
        assert terminal.route(attempt_id) is None
    assert sealed[older] != sealed[newer]
    assert json.loads(await gateway.pull({}))["kind"] == "done"


@pytest.mark.network
async def test_a_world_handed_to_a_replacement_is_the_world_its_seal_captures(
    serving_wordle,
) -> None:
    """A transport handed a world for an attempt seals in that world, not in the one before it.

    A replacement is given the world its attempt goes on working in, restored where the process
    before it left off. Every way of asking which world that attempt is in has to answer with that
    one: an ordinary call routes through the map this transport keeps, and an environment that
    seals by stopping the world an attempt played in resolves the route it was registered with.
    Two answers is the failure this is here for, and it is a silent one: the calls land in the
    restored world while the seal captures the predecessor's, so the submission, the score and the
    receipt all describe a world the agent was never working in.

    The predecessor letting go of its own world afterwards does not undo the pairing either. It is
    clearing up after the world it was holding, and the attempt has a newer one.
    """
    serving, terminal = serving_wordle
    seed = await wordle_world("")
    gateway = await opened(
        serving,
        seed,
        workflow_id="stream/gateway-restored-world/1",
        bodies=("Round 0.",),
        release=IMMEDIATE,
        terminal=terminal,
    )
    attempt = json.loads(await gateway.pull({}))["attempt_id"]
    wrapper = {"attempt_id": attempt, "arguments": {"word": "crane"}}
    assert json.loads((await gateway.environment("guess", wrapper)).content[0].text)["score"]
    assert len(seed._trajectory) == 1

    # The world a new owner restored for that attempt, played on in where the record said it was.
    restored = await wordle_world(attempt)
    for word in ("slate", "adieu"):
        await restored.call("guess", {"word": word})
    state = await gateway.stream_state()
    replacement = StreamGateway(
        gateway._stream,
        restored,
        gateway.spec,
        terminal_manifest(gateway.spec),
        initial_cursor=state.cursor,
        generation=gateway.generation,
        world_attempt=attempt,
        environment=terminal,
    )
    assert terminal.route(attempt) == (restored.env, restored.session_id)
    # And the transport it replaced lets go of the world it was holding, which is not the world
    # this attempt is in any more.
    await gateway.aclose()
    assert terminal.route(attempt) == (restored.env, restored.session_id)

    filing = {"attempt_id": attempt, "arguments": {}}
    assert json.loads(await replacement.terminal(filing))["kind"] == "seal_ack"
    payload = json.loads(await replacement.pull({}))
    assert payload["kind"] == "payload"
    # Two guesses is the restored world. The world the attempt started in had one.
    assert "guesses_used 2" in payload["body"]
    assert json.loads(await replacement.pull({}))["kind"] == "done"
    await replacement.aclose()


@pytest.mark.network
async def test_a_pull_past_the_capacity_waits_and_says_nothing_about_why(serving) -> None:
    """The tasks a generation holds out at once are its capacity, and the next pull is a Wait.

    Nothing is displaced and nothing is forfeited: the attempts in hand stay live, no world is
    opened for the task that was not served, and why the agent is waiting stays where the model
    cannot read it.
    """
    order: List[ServedEpisode] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        started = await score_world()
        order.append(started)
        return started

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-pull-past-capacity/1",
        bodies=("Round 0.", "Round 1.", "Round 2."),
        release=IMMEDIATE,
        open_episode=open_world,
        capacity=2,
    )
    held = [json.loads(await gateway.pull({})) for _ in range(2)]
    assert [record["kind"] for record in held] == ["task", "task"]

    waited = json.loads(await gateway.pull({}))
    assert set(waited) == {"protocol_version", "kind", "message_id", "retry_after_ms"}
    assert waited["kind"] == "wait"
    assert "capacity" not in json.dumps(waited)

    state = await gateway.stream_state()
    assert state.capacity == 2
    assert state.capacity_in_use == 2
    # The reason is the generation's own record of it, and it is not on the wire.
    assert state.wait_reasons == {"capacity": 1}
    assert [state.attempts[record["attempt_id"]] for record in held] == ["active", "active"]
    # The task the wait was answered instead of is still where it was, unserved and unstarted.
    assert state.tasks_remaining == 1
    # No world was opened for a task nothing served.
    assert len(order) == 2
    await gateway.aclose()


@pytest.mark.network
async def test_a_pull_at_a_full_capacity_still_answers_what_is_ready(serving) -> None:
    """A full capacity stops tasks being offered and stops nothing else, as the words say.

    The description a wider capacity serves says a pull cannot return another task while that
    many are held, and that it answers a wait when nothing else is ready. Both halves are the
    stream's own rule rather than a flat wait: capacity takes tasks out of what a pull may be
    answered with and leaves everything else in, so a payload owed to an attempt already sealed is
    served while the agent is holding its limit. A schedule that puts tasks first is where the two
    can be told apart, because it is what leaves a payload owed while the capacity fills.
    """
    order: List[ServedEpisode] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        started = await score_world()
        order.append(started)
        return started

    async def sealed(record: Dict[str, Any]) -> None:
        filing = {"attempt_id": record["attempt_id"], "arguments": {"answer": "4"}}
        assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-full-capacity-payload/1",
        bodies=("Round 0.", "Round 1.", "Round 2.", "Round 3."),
        release=ReleasePlan(RELEASE_AT_SEAL, TASK_FIRST, BY_POSITION),
        open_episode=open_world,
        capacity=2,
    )
    first = json.loads(await gateway.pull({}))
    assert json.loads(await gateway.pull({}))["kind"] == "task"
    # One of the two is sealed, so a payload is owed, and a schedule that puts tasks first fills
    # the capacity again before delivering it.
    await sealed(first)
    assert json.loads(await gateway.pull({}))["kind"] == "task"
    state = await gateway.stream_state()
    assert (state.capacity, state.capacity_in_use) == (2, 2)

    # The capacity is full and the payload is what is ready, so the payload is what comes back.
    owed = json.loads(await gateway.pull({}))
    assert owed["kind"] == "payload"
    assert owed["attempt_id"] == first["attempt_id"]
    # And with nothing else ready, the same full capacity is a wait.
    assert json.loads(await gateway.pull({}))["kind"] == "wait"
    state = await gateway.stream_state()
    assert state.wait_reasons == {"capacity": 1}
    assert state.tasks_remaining == 1
    await gateway.aclose()


#: What two servings of the same queue cannot agree on, whatever else about them is the same.
#: A generation mints its own attempt and message identifiers, and the digest a seal captures is
#: taken over a world of that attempt's own, so all three are values one run has and the other
#: does not. What they say about the order things happened in is kept.
MINTED = {"message_id", "attempt_id", "submission_digest"}


def named_identifiers(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The records with every minted value replaced by the order it was first seen in.

    The same value in the same place is the same name here, so a record that named another
    attempt, or repeated a message, is a record this does not turn into its counterpart. The
    substitution reaches inside a body as well, because a body may quote what was minted: a
    receipt says which attempt it is the receipt for.
    """
    names: Dict[str, str] = {}
    for record in records:
        for key, value in record.items():
            if key in MINTED:
                names.setdefault(value, f"#{len(names)}")

    def named(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        for minted, name in names.items():
            value = value.replace(minted, name)
        return value

    return [{key: named(value) for key, value in record.items()} for record in records]


def how_it_went(state: Any) -> Dict[str, Any]:
    """The counts and outcomes the generation ends with, without the names it minted.

    A selection rather than the whole state, and these are the fields two servings of one queue
    have to agree on: what became of every attempt and every obligation, and how many of each
    thing the schedule did. What is left out is what a generation cannot help differing in, its
    identifiers and its digests, and its capacity, which is the thing under test.
    """
    return {
        "generation_state": state.generation_state,
        "attempts": sorted(state.attempts.values()),
        "obligations": sorted(state.obligations.values()),
        "assignment_count": state.assignment_count,
        "materialization_count": state.materialization_count,
        "eligibility_count": state.eligibility_count,
        "offer_count": state.offer_count,
        "presentation_count": state.presentation_count,
        "payload_delivery_count": state.payload_delivery_count,
        "wait_count": state.wait_count,
        "wait_reasons": state.wait_reasons,
        "final_failures": state.final_failures,
    }


@pytest.mark.network
async def test_a_generation_at_eight_serves_a_one_at_a_time_agent_the_way_one_does(
    serving_wordle,
) -> None:
    """A capacity is a bound and not an instruction, so an agent that holds one is served as ever.

    The same queue under the same plan, driven by the same loop, at a capacity of one and at a
    capacity of eight. What the model reads is the same records in the same order, task bodies and
    payload bodies included, and the generation ends with the same counts and the same outcome for
    every attempt. What the wider capacity changes for this agent is only what it would have been
    allowed to do, and the tests above are where holding several is exercised: this loop holds one
    at a time, so what it says about routing is nothing.
    """
    serving, terminal = serving_wordle

    async def run(capacity: int, name: str) -> Tuple[List[Dict[str, Any]], Any]:
        gateway = await opened(
            serving,
            await wordle_world(""),
            workflow_id=f"stream/gateway-capacity-{name}/1",
            bodies=tuple(f"Round {index}." for index in range(3)),
            release=IMMEDIATE,
            terminal=terminal,
            capacity=capacity,
        )
        try:
            return await served(gateway), await gateway.stream_state()
        finally:
            await gateway.aclose()

    at_one, after_one = await run(1, "one")
    at_eight, after_eight = await run(8, "eight")
    assert [record["kind"] for record in at_one] == [
        "task",
        "seal_ack",
        "payload",
        "task",
        "seal_ack",
        "payload",
        "task",
        "seal_ack",
        "payload",
        "done",
    ]
    assert named_identifiers(at_eight) == named_identifiers(at_one)
    assert how_it_went(after_eight) == how_it_went(after_one)
    # The capacity is the one thing about the two that differs, and it is on the record.
    assert (after_one.capacity, after_eight.capacity) == (1, 8)


@pytest.mark.network
async def test_a_world_that_would_not_open_leaves_the_task_where_it_was(
    serving_wordle, episode
) -> None:
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

    serving, terminal = serving_wordle
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-world-will-not-open/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        terminal=terminal,
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
    serving_wordle, episode
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

        async def close(self, *, finalize: bool = True) -> None:
            closes.append(len(closes))
            if len(closes) == 1:
                raise RuntimeError("the world would not stop")
            await self._wrapped.close(finalize=finalize)

    async def open_world(attempt_id: str) -> Any:
        return WillNotStopOnce(await wordle_world(attempt_id))

    serving, terminal = serving_wordle
    gateway = await opened(
        serving,
        episode,
        workflow_id="stream/gateway-world-will-not-close/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        terminal=terminal,
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
    acknowledgement: nothing about that ending arrives here as a message, so the first this
    transport hears of one is the state it reads at the top of the next call. Until then the
    world stays open, because a task presented under another attempt says nothing about the
    attempts already live: at a capacity above one they are still being worked. The attempt the
    state reports as ended is the one whose world is let go of, and the others keep theirs.
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
        generation=stream_start(
            spec, terminal_manifest(spec), claim_hash=CLAIM_HASH, evaluation_only=True
        ),
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
        # Both attempts are live, so both worlds are.
        assert [score_mcp.gold(world.session_id) for world in worlds] == ["4", "4"]

        # The generation ended the first attempt itself, and the state is where that is said.
        await gateway._retired(
            SimpleNamespace(attempts={"a" * 32: "final_failed", "b" * 32: "active"})
        )
        assert score_mcp.gold(worlds[0].session_id) == ""
        assert score_mcp.gold(worlds[1].session_id) == "4"
    finally:
        await gateway.aclose()


@pytest.mark.network
async def test_a_deadline_closes_the_world_of_the_attempt_it_ended(serving) -> None:
    """A deadline reaches an attempt that is still working, and the world goes with the attempt.

    The generation ends the attempt itself when its time is up, and nothing about that ending is a
    message: the next call this transport makes is where it reads it. That call is where the world
    is let go of, before anything new is asked for, and a call naming the attempt whose world has
    gone is refused the way a call for an attempt that is over is.
    """
    order: List[ServedEpisode] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        started = await score_world()
        order.append(started)
        return started

    deadline_ms = 600_000
    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-deadline-world/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        attempt_deadline_ms=deadline_ms,
    )
    task = json.loads(await gateway.pull({}))
    assert task["kind"] == "task"
    world = order[0]
    assert score_mcp.gold(world.session_id) == "4"

    await serving.sleep(timedelta(milliseconds=deadline_ms + 1000))
    wrapper = {"attempt_id": task["attempt_id"], "arguments": {}}
    assert await refused(gateway.environment("noop", wrapper)) == "invalid_attempt"
    assert score_mcp.gold(world.session_id) == ""

    state = await gateway.stream_state()
    assert state.attempts[task["attempt_id"]] == "final_failed"
    assert state.final_failures == {task["attempt_id"]: DEADLINE}
    assert state.capacity_in_use == 0

    # And the generation goes on: the next task is served in a world of its own.
    following = json.loads(await gateway.pull({}))
    assert following["kind"] == "task"
    assert len(order) == 2
    assert score_mcp.gold(order[1].session_id) == "4"
    await gateway.aclose()


@pytest.mark.network
async def test_a_deadline_that_reaches_several_live_attempts_closes_every_one_of_their_worlds(
    serving,
) -> None:
    """A deadline is a fact about one attempt, and a generation may have several when it fires.

    Two attempts are live and neither is sealed when their time runs out. The next call is where
    this transport hears of both endings, and what it lets go of is every world whose attempt the
    generation ended rather than the one the call happens to name. A transport that retired one
    would leave the other running with nothing left to close it.
    """
    order: List[ServedEpisode] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        started = await score_world()
        order.append(started)
        return started

    deadline_ms = 600_000
    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-deadline-two-live/1",
        bodies=("Round 0.", "Round 1.", "Round 2."),
        release=IMMEDIATE,
        open_episode=open_world,
        capacity=2,
        attempt_deadline_ms=deadline_ms,
    )
    held = [json.loads(await gateway.pull({})) for _ in range(2)]
    assert [record["kind"] for record in held] == ["task", "task"]
    assert [score_mcp.gold(world.session_id) for world in order] == ["4", "4"]

    await serving.sleep(timedelta(milliseconds=deadline_ms + 1000))
    wrapper = {"attempt_id": held[0]["attempt_id"], "arguments": {}}
    assert await refused(gateway.environment("noop", wrapper)) == "invalid_attempt"
    # Both worlds went with their attempts, and not only the one this call named.
    assert [score_mcp.gold(world.session_id) for world in order] == ["", ""]

    state = await gateway.stream_state()
    assert state.final_failures == {record["attempt_id"]: DEADLINE for record in held}
    assert state.capacity_in_use == 0

    # And the capacity the endings gave back is capacity the generation goes on serving with.
    assert json.loads(await gateway.pull({}))["kind"] == "task"
    assert len(order) == 3
    assert score_mcp.gold(order[2].session_id) == "4"
    await gateway.aclose()


def ending_is_visible(
    stream: Any, attempt_id: str, *, lose_first_release: Optional[str] = None
) -> Any:
    """The stream, answering a release only once the generation has acted on the deadline it frees.

    The generation holds an expired attempt's ending back while a call is in that attempt's world,
    because an ending cannot cancel what a world is already doing, and makes it when the grant
    comes back. Nothing says how soon. This waits for it, so what these tests read is what the
    transport does about an ending the release has already made rather than which of two things
    happened first, and the residual case where the generation has not got to it yet is the one
    the next call retires.

    ``lose_first_release`` is how a release's answer goes missing, which is what leaves a call
    holding a grant it has to give back again. ``"after"`` is a release the stream took and
    answered into nothing, so the ending is made on the first call. ``"before"`` is one the stream
    never saw, so the generation is still waiting for it and the ending is made on the retry.
    """

    class Waited:
        def __init__(self) -> None:
            self._lose = lose_first_release

        def __getattr__(self, name: str) -> Any:
            return getattr(stream, name)

        async def end_environment_call(self, call: Any) -> Any:
            if self._lose == "before":
                self._lose = None
                raise RuntimeError("the release never reached the stream")
            answer = await stream.end_environment_call(call)
            if self._lose == "after":
                self._lose = None
                raise RuntimeError("the release never came back")
            for _ in range(200):
                state = await stream.stream_state()
                if state.attempts.get(attempt_id) == "final_failed":
                    return answer
                await asyncio.sleep(0.01)
            raise AssertionError("the generation never ended the attempt whose deadline passed")

    return Waited()


@pytest.mark.network
async def test_a_call_that_spans_its_deadline_lets_the_world_go_before_it_answers(serving) -> None:
    """The deadline of the attempt this call is in falls due while the call is in its world.

    The generation cannot end an attempt while a call is inside it, so it waits for the grant to
    come back and ends it then, which is inside this call. By the time the observation is handed
    to the model the attempt is over, and the world it was working in is still running: that
    observation may be the last thing the agent ever asks this transport for, so a world left for
    the call after it is a world nothing lets go of until the transport stops.
    """
    order: List[ServedEpisode] = []
    deadline_ms = 600_000

    async def open_world(attempt_id: str = "") -> Any:
        started = await score_world()
        order.append(started)
        return started

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-deadline-inside-a-call/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        attempt_deadline_ms=deadline_ms,
    )
    task = json.loads(await gateway.pull({}))
    attempt = task["attempt_id"]
    world = order[0]

    # The clock runs out while the call is in the world, which is where the generation holds the
    # ending back, and the release is what lets it be made.
    async def call_that_outlasts_the_attempt(tool_name: str, arguments: Dict[str, Any]) -> Any:
        answer = await ServedEpisode.call(world, tool_name, arguments)
        await serving.sleep(timedelta(milliseconds=deadline_ms + 1000))
        return answer

    world.call = call_that_outlasts_the_attempt  # type: ignore[method-assign]
    gateway._stream = ending_is_visible(gateway._stream, attempt)

    played = await gateway.environment("noop", {"attempt_id": attempt, "arguments": {}})
    # The observation the call landed with is still what it answers.
    assert json.loads(played.content[0].text)["ok"]
    # And the world it was made in is already gone, with the attempt it belonged to.
    assert score_mcp.gold(world.session_id) == ""
    assert gateway._worlds == {}
    state = await gateway.stream_state()
    assert state.final_failures == {attempt: DEADLINE}

    # The generation goes on: the next task is served in a world of its own.
    assert json.loads(await gateway.pull({}))["kind"] == "task"
    assert len(order) == 2
    await gateway.aclose()


@pytest.mark.network
async def test_a_landed_observation_collected_after_its_deadline_lets_the_world_go(serving) -> None:
    """The same ending, reached by the call that comes back for an observation it never got.

    A release whose answer is lost leaves the call holding what it landed with and the generation
    still holding the grant. The call that comes back gives the grant back, and the ending that
    was waiting on it is made then, so the observation and the ending arrive together again. What
    this adds is where the ending is read: the state this call arrived with is older than the
    question it hands the observation over through, and the ending is in the answer to that
    question and not in what it arrived with.
    """
    order: List[ServedEpisode] = []
    deadline_ms = 600_000

    async def open_world(attempt_id: str = "") -> Any:
        started = await score_world()
        order.append(started)
        return started

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-deadline-under-a-lost-release/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        attempt_deadline_ms=deadline_ms,
    )
    task = json.loads(await gateway.pull({}))
    attempt = task["attempt_id"]
    world = order[0]

    async def call_that_outlasts_the_attempt(tool_name: str, arguments: Dict[str, Any]) -> Any:
        answer = await ServedEpisode.call(world, tool_name, arguments)
        await serving.sleep(timedelta(milliseconds=deadline_ms + 1000))
        return answer

    world.call = call_that_outlasts_the_attempt  # type: ignore[method-assign]
    gateway._stream = ending_is_visible(gateway._stream, attempt, lose_first_release="after")

    wrapper = {"attempt_id": attempt, "arguments": {}}
    with pytest.raises(RuntimeError, match="release never came back"):
        await gateway.environment("noop", wrapper)
    # The call is holding what it landed with, and nothing has ended yet: the grant is still out.
    assert isinstance(gateway._recovery, gateway_module._LeaseHeld)
    assert score_mcp.gold(world.session_id) == "4"

    played = await gateway.environment("noop", wrapper)
    assert json.loads(played.content[0].text)["ok"]
    assert score_mcp.gold(world.session_id) == ""
    assert gateway._worlds == {}
    state = await gateway.stream_state()
    assert state.final_failures == {attempt: DEADLINE}
    await gateway.aclose()


@pytest.mark.network
async def test_a_call_that_fails_after_its_deadline_lets_the_world_go_all_the_same(serving) -> None:
    """The call the ending was made inside failed, and the world it was in goes anyway.

    What the caller is told about is the world's own failure, because that is the one thing this
    call has to say and the only thing it can say truthfully. The ending is not a thing it says at
    all: the grant went back on the way out, the generation made the ending it had been holding,
    and an attempt that is over is no more this call's to walk away from for having failed than it
    would have been for having worked.
    """
    order: List[ServedEpisode] = []
    deadline_ms = 600_000

    async def open_world(attempt_id: str = "") -> Any:
        started = await score_world()
        order.append(started)
        return started

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-deadline-under-a-failed-call/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        attempt_deadline_ms=deadline_ms,
    )
    task = json.loads(await gateway.pull({}))
    attempt = task["attempt_id"]
    world = order[0]

    async def call_that_fails_after_the_attempt(tool_name: str, arguments: Dict[str, Any]) -> Any:
        await serving.sleep(timedelta(milliseconds=deadline_ms + 1000))
        raise RuntimeError("the world failed after the deadline")

    world.call = call_that_fails_after_the_attempt  # type: ignore[method-assign]
    gateway._stream = ending_is_visible(gateway._stream, attempt)

    with pytest.raises(RuntimeError, match="world failed after the deadline"):
        await gateway.environment("noop", {"attempt_id": attempt, "arguments": {}})
    assert score_mcp.gold(world.session_id) == ""
    assert gateway._worlds == {}
    state = await gateway.stream_state()
    assert state.final_failures == {attempt: DEADLINE}

    # And the generation goes on, in a world of its own.
    assert json.loads(await gateway.pull({}))["kind"] == "task"
    assert len(order) == 2
    await gateway.aclose()


@pytest.mark.network
async def test_a_lease_given_back_a_second_time_lets_go_of_what_that_release_ended(serving) -> None:
    """A call that landed nothing comes back, gives its grant back, and finds its attempt over.

    The release the generation was waiting for is the one that never reached it, so the ending is
    made on the retry, and there is no observation to carry it out with: this call has nothing to
    hand over and nothing to say except that the attempt it named is gone. The world that attempt
    was working in still has to go, and it goes before this call is allowed to ask for a world
    again, because the call that starts over is the one the ending happened under.
    """
    order: List[ServedEpisode] = []
    deadline_ms = 600_000

    async def open_world(attempt_id: str = "") -> Any:
        started = await score_world()
        order.append(started)
        return started

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-deadline-under-a-resultless-lease/1",
        bodies=("Round 0.", "Round 1."),
        release=IMMEDIATE,
        open_episode=open_world,
        attempt_deadline_ms=deadline_ms,
    )
    task = json.loads(await gateway.pull({}))
    attempt = task["attempt_id"]
    world = order[0]

    async def call_that_fails_after_the_attempt(tool_name: str, arguments: Dict[str, Any]) -> Any:
        await serving.sleep(timedelta(milliseconds=deadline_ms + 1000))
        raise RuntimeError("the world failed after the deadline")

    world.call = call_that_fails_after_the_attempt  # type: ignore[method-assign]
    gateway._stream = ending_is_visible(gateway._stream, attempt, lose_first_release="before")

    wrapper = {"attempt_id": attempt, "arguments": {}}
    with pytest.raises(RuntimeError, match="world failed after the deadline"):
        await gateway.environment("noop", wrapper)
    # The grant never went back, so the generation is still waiting on it and nothing has ended.
    held = gateway._recovery
    assert isinstance(held, gateway_module._LeaseHeld)
    assert held.result is None
    assert score_mcp.gold(world.session_id) == "4"

    # The same call again gives the grant back, and the ending that was waiting on it is made.
    assert await refused(gateway.environment("noop", wrapper)) == "invalid_attempt"
    assert score_mcp.gold(world.session_id) == ""
    assert gateway._worlds == {}
    state = await gateway.stream_state()
    assert state.final_failures == {attempt: DEADLINE}
    await gateway.aclose()


@pytest.mark.network
async def test_a_retirement_that_cannot_close_one_world_still_closes_the_others(serving) -> None:
    """A world that will not close is one world, and the ended ones beside it are still retired.

    Retirement runs at the top of every call, so a cleanup that failed is asked for again by the
    call that comes back. Stopping at the first failure would mean the ended worlds behind it were
    never tried at all: the same one would fail on every entry, and every other ended world would
    stay running for as long as this transport did, with nothing that would ever reach them.
    """
    order: List[ServedEpisode] = []

    class WillNotStopOnce:
        """A world whose first cleanup fails. Everything else about it is the episode it wraps."""

        def __init__(self, wrapped: ServedEpisode) -> None:
            self._wrapped = wrapped
            self._refused = False

        def __getattr__(self, name: str) -> Any:
            return getattr(self._wrapped, name)

        async def close(self, *, finalize: bool = True) -> None:
            if not self._refused:
                self._refused = True
                raise RuntimeError("the first world would not stop")
            await self._wrapped.close(finalize=finalize)

    async def open_world(attempt_id: str = "") -> Any:
        started = await score_world()
        order.append(started)
        return WillNotStopOnce(started) if len(order) == 1 else started

    deadline_ms = 600_000
    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-retirement-that-fails/1",
        bodies=("Round 0.", "Round 1.", "Round 2."),
        release=IMMEDIATE,
        open_episode=open_world,
        capacity=2,
        attempt_deadline_ms=deadline_ms,
    )
    held = [json.loads(await gateway.pull({})) for _ in range(2)]
    assert [record["kind"] for record in held] == ["task", "task"]
    await serving.sleep(timedelta(milliseconds=deadline_ms + 1000))

    # Both attempts have ended, and the world of the first will not go.
    with pytest.raises(RuntimeError, match="first world would not stop"):
        await gateway.pull({})
    # The other was retired anyway, and nothing was offered while the first was still running.
    assert [score_mcp.gold(world.session_id) for world in order] == ["4", ""]
    assert (await gateway.stream_state()).pending_message_id is None

    # And the call that comes back asks for the cleanup again, and then serves the next task.
    assert json.loads(await gateway.pull({}))["kind"] == "task"
    assert [score_mcp.gold(world.session_id) for world in order[:2]] == ["", ""]
    assert len(order) == 3
    await gateway.aclose()


@pytest.mark.network
async def test_eight_tasks_are_held_at_once_and_each_of_the_eight_worlds_stands_up(
    serving,
) -> None:
    """The capacity the cell serves, with every task it allows in hand at the same time.

    Eight worlds are open together, each is called into, and each call lands in the world of the
    attempt it names. What this adds to the two-attempt tests is the number: nothing here is a
    pair that a transport holding the newest world could get right by accident, and the ninth pull
    is the wait that says the capacity is what it says it is.

    The attempts are called different numbers of times, in an order that is neither the order they
    were served in nor the reverse of it, and then sealed in a third order one at a time. Each seal
    closes the world of the attempt it named and no other, so a shuffle of eight attempts onto
    eight worlds is a shuffle this reads back, whatever it is.
    """
    order: List[ServedEpisode] = []

    async def open_world(attempt_id: str = "") -> ServedEpisode:
        started = await score_world()
        order.append(started)
        return started

    gateway = await opened(
        serving,
        await open_world(),
        workflow_id="stream/gateway-eight-live/1",
        bodies=tuple(f"Round {index}." for index in range(9)),
        release=IMMEDIATE,
        open_episode=open_world,
        capacity=8,
    )
    held = [json.loads(await gateway.pull({})) for _ in range(8)]
    assert [record["kind"] for record in held] == ["task"] * 8
    assert len(order) == 8
    assert len({world.session_id for world in order}) == 8
    assert (await gateway.stream_state()).capacity_in_use == 8
    assert json.loads(await gateway.pull({}))["kind"] == "wait"

    # How many calls each of the eight gets, interleaved so that no world is reached by being the
    # one opened or worked in last. This env gives an attempt three world calls, so the counts
    # repeat across eight of them; the seals below are what tell every world from every other.
    apiece = (1, 3, 2, 3, 1, 2, 3, 1)
    rounds = [
        held[position]
        for turn in range(max(apiece))
        for position, count in enumerate(apiece)
        if count > turn
    ]
    for record in rounds:
        wrapper = {"attempt_id": record["attempt_id"], "arguments": {}}
        assert json.loads((await gateway.environment("noop", wrapper)).content[0].text)["ok"]
    assert [len(world._trajectory) for world in order] == list(apiece)

    # And each of the eight is sealed in the world it was worked in, in a third order: after each
    # filing exactly the worlds of the attempts filed so far are closed.
    sealed: List[int] = []
    for position in (5, 0, 7, 2, 6, 1, 4, 3):
        filing = {"attempt_id": held[position]["attempt_id"], "arguments": {"answer": "4"}}
        assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
        sealed.append(position)
        gone = [index for index, world in enumerate(order) if not score_mcp.gold(world.session_id)]
        assert gone == sorted(sealed)
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
        grade=WORDLE_GRADE,
    )
    with pytest.raises(ValueError, match="a world of its own"):
        await open_gateway(None, episode, start=start)  # type: ignore[arg-type]


async def test_a_stream_that_never_started_leaves_no_generation_to_resume(
    episode: ServedEpisode, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a run is a directory and then a stream, and the manifest says the stream is there.

    So the manifest is written once it is. A manifest written first would name a workflow nobody
    had started: no owner could resume that generation, and the identical retry would be refused
    by the manifest the dead attempt left behind. What a failed start leaves is a directory the
    next attempt can still use.
    """
    root = tmp_path / "run"

    async def never(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the service went away before the stream existed")

    monkeypatch.setattr("shogym.serve.protocol_v2.gateway.start_stream", never)
    with pytest.raises(RuntimeError, match="before the stream existed"):
        await open_gateway(None, episode, run_directory=root)  # type: ignore[arg-type]

    assert (root / MANIFEST_FILE).exists() is False
    with pytest.raises(ResumeRefused) as caught:
        open_run_directory(root)
    assert caught.value.code == "configuration_mismatch"

    # The next attempt runs out of the same directory rather than being refused by it.
    run = create_run_directory(
        root,
        workflow_id="stream/retried/1",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash="a" * 64,
    )
    assert open_run_directory(root).manifest == run.manifest


@pytest.mark.network
async def test_a_generation_no_manifest_ever_named_does_not_stay_running(
    serving: WorkflowEnvironment,
    episode: ServedEpisode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream is started and then recorded, and a run that dies between leaves neither behind.

    The identifier is minted in this call and nowhere else. A manifest that never landed
    therefore means nothing on the disk names the authority that did start: no owner can resume
    it, and the next attempt out of the directory mints another identifier and leaves the first
    running with no consumer for as long as the service lives. So the name goes down before the
    stream does, and the attempt that finds one ends what it names before starting its own.
    """
    root = tmp_path / "run"
    started: List[str] = []

    async def watched(*args: Any, **kwargs: Any) -> Any:
        stream = await start_stream(*args, **kwargs)
        started.append(kwargs["workflow_id"])
        return stream

    def dies(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the machine went away before the manifest landed")

    async def never(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("the service went away before the stream existed")

    # A name written down for a stream that then failed to start. What it names does not
    # exist, which is the state the next attempt wants it in, so it is not an error.
    monkeypatch.setattr("shogym.serve.protocol_v2.gateway.start_stream", never)
    with pytest.raises(RuntimeError, match="before the stream existed"):
        await open_gateway(serving.client, episode, run_directory=root)
    monkeypatch.undo()
    unstarted = staged_generation(root)
    assert unstarted is not None

    monkeypatch.setattr("shogym.serve.protocol_v2.gateway.start_stream", watched)
    monkeypatch.setattr("shogym.serve.protocol_v2.gateway.create_run_directory", dies)
    with pytest.raises(RuntimeError, match="before the manifest landed"):
        await open_gateway(serving.client, episode, run_directory=root)
    monkeypatch.undo()

    abandoned = started[0]
    assert abandoned != unstarted.workflow_id
    assert (root / MANIFEST_FILE).exists() is False
    left = staged_generation(root)
    assert left is not None and left.workflow_id == abandoned
    running = await serving.client.get_workflow_handle(abandoned).describe()
    assert running.status == WorkflowExecutionStatus.RUNNING

    # The next attempt runs out of the same directory, and what the cut left is over.
    gateway = await open_gateway(serving.client, episode, run_directory=root)
    try:
        manifest = open_run_directory(root).manifest
        assert manifest.workflow_id != abandoned
        assert staged_generation(root) is None
        ended = await serving.client.get_workflow_handle(abandoned).describe()
        assert ended.status == WorkflowExecutionStatus.TERMINATED
    finally:
        await gateway.aclose()


async def test_a_manifest_arrives_whole_or_leaves_the_directory_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A publication that dies half way leaves no manifest, rather than an unusable one.

    A file that is there is what says this directory holds a generation, so a torn one is the
    worst of both: nothing can resume what it names, and the next attempt is refused by it
    instead of running out of the directory. So the bytes are written beside the manifest and
    renamed onto it, and this cuts the run at the rename.
    """
    root = tmp_path / "run"

    def dies(*args: Any, **kwargs: Any) -> None:
        raise OSError("the disk went away between the write and the rename")

    monkeypatch.setattr("shogym.serve.protocol_v2.rundir.os.replace", dies)
    with pytest.raises(OSError, match="between the write and the rename"):
        create_run_directory(
            root,
            workflow_id="stream/torn/1",
            task_queue=STREAM_TASK_QUEUE,
            configuration_hash="b" * 64,
        )
    assert (root / MANIFEST_FILE).exists() is False
    assert sorted(path.name for path in root.iterdir()) == ["blobs"]

    monkeypatch.undo()
    run = create_run_directory(
        root,
        workflow_id="stream/torn/1",
        task_queue=STREAM_TASK_QUEUE,
        configuration_hash="b" * 64,
    )
    assert open_run_directory(root).manifest == run.manifest


@pytest.mark.network
async def test_a_run_this_call_composed_can_still_be_taken_over(
    serving: WorkflowEnvironment, episode: ServedEpisode, tmp_path: Path
) -> None:
    """A resume presents a composition, so the composition this call made is on what it returns.

    A generation mints its own identifiers, and they are part of what a resume is held to. A
    replacement that composed the same environment and task afresh would therefore be composed
    for a different generation, which is what makes the composition itself the thing a later
    owner needs. A caller that composed the run holds it already; one that let this call compose
    reads it back from here.
    """
    root = tmp_path / "run"
    gateway = await opened_run(serving, episode, root)
    composed = gateway.generation
    assert composed is not None
    manifest = open_run_directory(root).manifest
    assert configuration_hash(composed) == manifest.configuration_hash

    # Composing the same environment and task again is composing another generation.
    spec = episode.describe()
    afresh = stream_start(spec, terminal_manifest(spec), claim_hash=CLAIM_HASH, grade=WORDLE_GRADE)
    assert configuration_hash(afresh) != manifest.configuration_hash

    taken = await resume_run_directory(
        serving.client, root, start=composed, claimant_id="the-next-owner"
    )
    assert (await taken.stream_state()).ownership_epoch == 2


async def opened_run(
    environment: WorkflowEnvironment, episode: ServedEpisode, root: Path
) -> StreamGateway:
    """Open a generation the way a caller with no composition of its own does."""
    return await open_gateway(
        environment.client, episode, workflow_id="stream/composed/1", run_directory=root
    )


@pytest.mark.network
async def test_a_budgeted_generation_is_taken_over_with_the_number_it_declared(
    serving_wordle: Tuple[WorkflowEnvironment, EnvironmentTerminal],
    episode: ServedEpisode,
    tmp_path: Path,
) -> None:
    """A replacement serves the budget the generation it resumed declared, or it is not built.

    This is the whole of a takeover: the generation is composed with a number, started, served
    once, and then resumed by a second owner holding the exact composition. That owner builds the
    transport the model talks to, and what the transport advertises and what it counts calls
    against both come from that composition rather than from whatever episode it happens to be
    holding. A replacement pointed at an episode the generation was not composed over is refused
    instead of serving records that say one number while it enforces another.
    """
    serving, terminal = serving_wordle
    spec = episode.describe()
    horizon = spec.horizon
    assert horizon is not None
    composed = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        budget=horizon,
        grade=terminal.grade,
    )
    root = tmp_path / "run"
    gateway = await open_gateway(
        serving.client,
        episode,
        workflow_id="stream/budget-resume/1",
        start=composed,
        run_directory=root,
        environment=terminal,
    )
    await gateway.close_queue()
    assert json.loads(await gateway.pull({}))["budget"] == horizon

    # A second owner, holding the exact composition, which is what the identity is derived from.
    resumed = await resume_run_directory(
        serving.client, root, start=gateway.generation, claimant_id="the-next-owner"
    )
    state = await resumed.stream_state()
    assert state.ownership_epoch == 2

    replacement = StreamGateway(
        resumed,
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor=state.cursor,
        generation=gateway.generation,
        environment=terminal,
    )
    assert replacement._step_cap == horizon
    async with Client(build_gateway_server(replacement)) as client:
        described = {tool.name: tool.description for tool in await client.list_tools()}
    assert (
        described[PULL_TOOL]
        == served_manifest(spec, terminal_manifest(spec), budget=horizon)["control_tool"][
            "description"
        ]
    )

    elsewhere = spec.model_copy(update={"horizon": horizon + 1})
    with pytest.raises(ValueError, match="step cap this transport enforces"):
        StreamGateway(
            resumed,
            episode,
            elsewhere,
            terminal_manifest(elsewhere),
            initial_cursor=state.cursor,
            generation=gateway.generation,
            environment=terminal,
        )


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
        grade=WORDLE_GRADE,
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
        grade=WORDLE_GRADE,
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
        generation=stream_start(
            spec, terminal_manifest(spec), claim_hash=CLAIM_HASH, evaluation_only=True
        ),
    )
    async with Client(build_gateway_server(gateway)) as client:
        assert {tool.name for tool in await client.list_tools()} == {
            PULL_TOOL,
            "guess",
            "terminate",
        }
