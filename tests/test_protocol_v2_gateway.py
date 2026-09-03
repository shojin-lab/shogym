"""The gateway: what the model can call, and what it gets back.

Most of what the gateway owns is decided before an Update is sent, so most of it is tested
against a scripted stream: whether a wrapper is well formed, which attempt a call routes to,
whether two calls may overlap, and what a refusal looks like on the wire. The stream itself is
element two's and is not re-tested here.

Two tests need the durable stream itself. One runs the whole arc for real: it spawns ``shogym
serve`` as a stdio MCP server, drives it through Task, terminal, SealAck, Payload, and Done with
a scripted client, and reads the bytes the model would have read. The other is the one recovery
a double cannot vouch for, because what it turns on is the stream answering a completed call
rather than running it again. Both are marked ``network`` because the durable service downloads
its binary the first time, and both skip when that service cannot start.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from fastmcp import Client  # noqa: E402
from fastmcp.client.transports import StdioTransport  # noqa: E402
from mcp.shared.exceptions import McpError  # noqa: E402
from temporalio.service import RPCError, RPCStatusCode  # noqa: E402

from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import gateway as gateway_module  # noqa: E402
from shogym.serve.protocol_v2 import (  # noqa: E402
    Done,
    Payload,
    PresentationAck,
    PresentationCommit,
    SealAck,
    SealReject,
    Task,
    Wait,
    visible_bytes,
)
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    CANONICALIZATION_VERSION,
    PULL_TOOL,
    GatewayClosed,
    StreamGateway,
    build_gateway_server,
    declared_argument_names,
    durable_client,
    environment_grade,
    environment_terminal,
    open_gateway,
    stream_start,
    stream_worker,
    terminal_manifest,
    wrapped_manifests,
)
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    _configuration_hash,
    _Idle,
    _LeaseHeld,
    _PresentationRefused,
    _PresentationUncertain,
    _RequestUncertain,
    _ResultOwed,
)
from shogym.serve.protocol_v2.kernel import OfferedMessage, StreamProtocolError  # noqa: E402
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    HONEST_V1,
    GradeIdentity,
    policy_digest,
    policy_preimage,
)
from shogym.serve.protocol_v2.rundir import open_run_directory  # noqa: E402
from shogym.task import TaskSpec, ToolManifest  # noqa: E402
from shogym.trace import load_traces  # noqa: E402

from tests._fixtures import score_env, score_mcp  # noqa: E402

TEST_ENV = "wordle_v1"
# The env that scores its own terminal. It is what brings a seal lifecycle and a durable
# finalization store, and a test that reads what a v2 ending leaves behind needs an env that has
# them: an env without them has nothing to end twice.
SCORING_ENV = score_env.ENV_NAME
# What the doubles below say their grader is. A composition that delivers the honest body is one
# over an environment whose score is its own, so a double that stands in for such an environment
# says so rather than being taken for the stream's stand-in.
DOUBLE_GRADE = GradeIdentity(
    grader_id="double-grade", grader_version="1", stand_in=False, score_component="answer"
)
ATTEMPT = "00000000000000000000000000000100"
TASK_ID = "00000000000000000000000000000101"
ACK_ID = "00000000000000000000000000000102"
SECOND_ATTEMPT = "00000000000000000000000000000200"
SECOND_TASK_ID = "00000000000000000000000000000201"
DONE_ID = "00000000000000000000000000000002"
CURSOR = "00000000000000000000000000000001"


def offered(record: Any, attempt_id: Optional[str] = None) -> OfferedMessage:
    """One offered message carrying the exact bytes of ``record``."""
    return OfferedMessage(
        message_id=record.message_id,
        kind=record.kind,
        visible_text=visible_bytes(record).decode("utf-8"),
        attempt_id=attempt_id,
    )


TASK_OFFER = offered(
    Task(message_id=TASK_ID, attempt_id=ATTEMPT, body="Guess the word."), ATTEMPT
)
ACK_OFFER = offered(
    SealAck(
        message_id=ACK_ID,
        attempt_id=ATTEMPT,
        submission_digest="a" * 64,
        canonicalization_version="shogym.gateway.1",
    ),
    ATTEMPT,
)
SECOND_TASK_OFFER = offered(
    Task(message_id=SECOND_TASK_ID, attempt_id=SECOND_ATTEMPT, body="Guess the next word."),
    SECOND_ATTEMPT,
)
DONE_OFFER = offered(Done(message_id=DONE_ID))


class ScriptedProjection:
    """What the gateway reads out of the stream at the top of every call.

    The state hash moves with every read, so a commit built a second time is visibly not the
    one that was built the first time. Everything else is the authority's own answer about
    where the generation is, which is what the gateway routes by, including the message it is
    holding for a request that has not collected it.
    """

    def __init__(self, stream: "ScriptedStream", reads: int) -> None:
        self.cursor = stream.cursor
        self.generation_state = stream.generation_state
        self.attempts = dict(stream.attempts)
        # The grants it has made per attempt, which is the count a transport reads its spent
        # step budget out of rather than keeping one of its own.
        self.environment_calls = dict(stream.environment_calls)
        self.stream_state_sha256 = format(reads, "064x")
        pending = stream.pending
        self.pending_message_id = None if pending is None else pending.message_id
        self.pending_kind = None if pending is None else pending.kind


class ScriptedStream:
    """A stream that answers from a script and records what it was asked for.

    It exists so a test can tell a call the gateway refused from a call it sent: nothing
    reaches ``calls`` unless the gateway decided the request was one worth sending. Its
    presentation half keeps the two rules the gateway's recovery rests on: an attestation
    applies once and the same attestation asked again is answered with what it was answered
    with, refusal included, and applying one is what moves the cursor and the attempt it names.
    Its offering half keeps the other two: a message is reserved for the request that was given
    it and is reachable through no other, and offering one is what takes the seal it carries.
    That is why this is an authority rather than a script: the gateway asks it where the
    generation is before every call, and takes its answer over anything it remembers.

    An attestation is answered from what it was answered with because a durable stream reaches
    one by an identity its own call ID is derived from. A refusal completes that call, so the
    same attestation sent afterwards collects the refusal rather than being verified again, and
    a double that quietly ran it a second time would hide exactly the case a repair needs.

    It is also the authority for an ordinary environment call, which is the one call that never
    reaches a stream. It decides that call by the same rules it decides an Update by, and it
    stays held for it: while one is out, every Update it could be racing against is refused.
    """

    def __init__(self, *offers: Any) -> None:
        self.offers: List[Any] = list(offers)
        self.calls: List[str] = []
        self.requests: List[Any] = []
        self.commits: List[Any] = []
        self.blobs: List[Dict[str, Optional[str]]] = []
        self.attestations: Dict[str, PresentationAck] = {}
        self.refused_attestations: Dict[str, str] = {}
        self.cursor = CURSOR
        self.generation_state = "open"
        self.attempts: Dict[str, str] = {}
        self.environment_calls: Dict[str, int] = {}
        self.offered: Dict[str, OfferedMessage] = {}
        self.reserved: Dict[str, OfferedMessage] = {}
        self.pending: Optional[OfferedMessage] = None
        self.state_reads = 0
        self.queue_closed = False
        self.lose_next_ack = False
        self.lose_next_result = False
        self.fault_before_offer = False
        self.fault_before_commit = False
        self.refuse_next_commit = False
        self.gate: Optional[asyncio.Event] = None
        self.commit_gate: Optional[asyncio.Event] = None
        self.state_gate: Optional[asyncio.Event] = None
        # Whether this generation now has a different writer, and how often it has been asked
        # about that. A query answers either way, so the asking is what a test watches.
        self.fenced = False
        self.confirmations = 0
        # The environment call this stream is currently held for, and a decision it has already
        # made about the next one, which is how a test says the writer has been replaced.
        self.held: Optional[str] = None
        self.environment_refusal: Optional[BaseException] = None
        self.leases: List[Any] = []
        self.releases: List[Any] = []
        self.lose_next_grant = False
        self.fault_before_release = False
        self.lose_next_release = False

    def _hold(self) -> None:
        """Refuse an Update while an environment call is holding this generation."""
        if self.held is not None:
            raise StreamProtocolError("overlapping_call")

    async def _next(self, label: str, request_id: str) -> Any:
        held = self.reserved.get(request_id)
        if held is not None:
            return held
        self._hold()
        self.calls.append(label)
        if self.gate is not None:
            await self.gate.wait()
        if self.fault_before_offer:
            self.fault_before_offer = False
            raise RuntimeError("the response channel failed before anything was offered")
        if self.pending is not None:
            raise StreamProtocolError("outstanding_response")
        answer = self.offers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        if isinstance(answer, OfferedMessage):
            self.offered[answer.message_id] = answer
            self.reserved[request_id] = answer
            self.pending = answer
            self._take(answer)
        if self.lose_next_result:
            self.lose_next_result = False
            raise RuntimeError("the result never arrived")
        return answer

    async def pull(self, request: Any) -> Any:
        self.requests.append(request)
        return await self._next("pull", request.request_id)

    async def seal(self, request: Any) -> Any:
        self.requests.append(request)
        return await self._next("seal", request.metadata.request_id)

    async def stream_state(self) -> ScriptedProjection:
        self.state_reads += 1
        answer = ScriptedProjection(self, self.state_reads)
        if self.state_gate is not None:
            await self.state_gate.wait()
        return answer

    async def confirm_state(self) -> ScriptedProjection:
        """Answer the same state, asked the way a write is asked for and refusable like one.

        A query has no writer to check, so it answers a transport this generation has been
        taken from exactly as it answers the one holding it. This is the path that can tell
        them apart, and ``fenced`` is how a test says which one is asking.

        A generation that is done answers nothing here, because there is nothing left to answer
        with. Done ends the stream: it accepts no call after it, and it stops running once the
        calls it had already taken have finished, so this path is gone rather than refusing on
        it. A double that went on answering would let a gateway hand over the last thing it is
        owed on a promise the stream it names is in no position to keep.
        """
        self.confirmations += 1
        if self.generation_state != "open":
            raise StreamProtocolError("closed_stream")
        if self.fenced:
            raise StreamProtocolError("fenced_writer")
        self.state_reads += 1
        return ScriptedProjection(self, self.state_reads)

    async def close_queue(self) -> Any:
        self._hold()
        self.calls.append("close_queue")
        self.queue_closed = True
        return SimpleNamespace(task_count=1, closed=True)

    async def begin_environment_call(self, call: Any) -> Any:
        """Decide one call to a world this stream cannot see, and stay held while it happens."""
        self.calls.append("environment")
        self.leases.append(call)
        if self.environment_refusal is not None:
            raise self.environment_refusal
        self._hold()
        if self.generation_state != "open":
            raise StreamProtocolError("closed_stream")
        if self.pending is not None:
            raise StreamProtocolError("outstanding_response")
        if self.attempts.get(call.attempt_id) != "active":
            raise StreamProtocolError("invalid_attempt")
        self.held = call.call_id
        # The grant is the last thing this stream learns about that world, so it is counted as
        # the change it authorized whatever became of the call.
        self.environment_calls[call.attempt_id] = (
            self.environment_calls.get(call.attempt_id, 0) + 1
        )
        if self.lose_next_grant:
            self.lose_next_grant = False
            raise RuntimeError("the grant never came back")
        return SimpleNamespace(
            call_id=call.call_id, attempt_id=call.attempt_id, cursor=self.cursor, held=True
        )

    async def end_environment_call(self, call: Any) -> Any:
        """Give the generation back, or lose the answer on one side of doing it or the other."""
        self.releases.append(call)
        if self.fault_before_release:
            self.fault_before_release = False
            raise RuntimeError("the release never reached the stream")
        held = self.held == call.call_id
        if held:
            self.held = None
        if self.lose_next_release:
            self.lose_next_release = False
            raise RuntimeError("the release never came back")
        return SimpleNamespace(
            call_id=call.call_id, attempt_id=call.attempt_id, cursor=self.cursor, held=held
        )

    async def commit_presentation(self, commit: PresentationCommit) -> PresentationAck:
        if self.fault_before_commit:
            self.fault_before_commit = False
            raise RuntimeError("the presentation request never arrived")
        self.commits.append(commit)
        known = self.attestations.get(commit.attestation_id)
        if known is not None:
            return known
        answered = self.refused_attestations.get(commit.attestation_id)
        if answered is not None:
            # This attestation has been answered, and the answer was a refusal. A durable
            # stream reaches it by an identity its call ID is derived from, so this is that
            # completed call being collected rather than a second verification of it.
            raise StreamProtocolError(answered)
        if self.refuse_next_commit:
            # The shape a storage refusal has: the attestation reached the stream and was
            # verified against something outside it, and nothing was applied. The message it
            # names is still the one being held, and the cursor is where it was.
            self.refuse_next_commit = False
            self.refused_attestations[commit.attestation_id] = "invalid_message"
            raise StreamProtocolError("invalid_message")
        self._hold()
        self.calls.append("present")
        if self.commit_gate is not None:
            await self.commit_gate.wait()
        self.blobs.append(
            {
                "transcript": commit.transcript_blob,
                "provider_turn": commit.provider_turn_blob,
                "checkpoint": commit.task_start_checkpoint_blob,
            }
        )
        self.cursor = commit.message_id
        self.pending = None
        self._apply(self.offered[commit.message_id])
        ack = PresentationAck(
            attestation_id=commit.attestation_id,
            cursor=commit.message_id,
            stream_state_sha256="b" * 64,
        )
        self.attestations[commit.attestation_id] = ack
        if self.lose_next_ack:
            self.lose_next_ack = False
            raise RuntimeError("the acknowledgement never arrived")
        return ack

    def _take(self, message: OfferedMessage) -> None:
        """Take the seal an acknowledgement carries, which happens as it is offered.

        Sealing is the terminal's work and the acknowledgement is the report of it, so the
        attempt is over the moment the stream has one to give, whether or not anything ever
        collects it.
        """
        if message.kind == "seal_ack" and message.attempt_id is not None:
            self.attempts[message.attempt_id] = "sealed"

    def _apply(self, message: OfferedMessage) -> None:
        """Move the generation the way presenting this message moves it, and no other way."""
        if message.kind == "task" and message.attempt_id is not None:
            self.attempts[message.attempt_id] = "active"
            self.environment_calls[message.attempt_id] = 0
        elif message.kind == "seal_ack" and message.attempt_id is not None:
            self.attempts[message.attempt_id] = "ack_presented"
        elif message.kind == "done":
            self.generation_state = "done"


def stream_writes(stream: ScriptedStream) -> Dict[str, Any]:
    """Everything a call writes into this double, which is what a read must leave alone.

    A durable generation would carry each of these in its history: the requests it was sent,
    the attestations it applied, where its cursor is, and what its attempts and its offers are.
    The confirmation counter is here for the same reason. A confirmation is asked for the way a
    write is asked for and is refusable like one, so a path that took it to answer a read has
    reached the generation on a writer's route, and a snapshot that skipped it would call that
    untouched. The plain read counter is deliberately not here. A Query moves none of this, so
    counting one beside the writes would let a claim about writing hide what the reading costs.
    """
    return {
        "calls": list(stream.calls),
        "requests": list(stream.requests),
        "confirmations": stream.confirmations,
        "commits": list(stream.commits),
        "blobs": list(stream.blobs),
        "attestations": dict(stream.attestations),
        "refused_attestations": dict(stream.refused_attestations),
        "leases": list(stream.leases),
        "releases": list(stream.releases),
        "cursor": stream.cursor,
        "generation_state": stream.generation_state,
        "queue_closed": stream.queue_closed,
        "attempts": dict(stream.attempts),
        "offered": dict(stream.offered),
        "reserved": dict(stream.reserved),
        "pending": stream.pending,
    }


@pytest_asyncio.fixture
async def episode() -> AsyncIterator[ServedEpisode]:
    started = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
    try:
        yield started
    finally:
        await started.close()


async def scoring_world(trace_path: Optional[Path] = None) -> ServedEpisode:
    """One world of the env that scores its own terminal, started the way this protocol serves.

    In one place because it is what the cross layer tests below open, and what a world of theirs
    is started with is a property of this protocol rather than of any one of them.
    """
    return await ServedEpisode.start(
        SCORING_ENV, task=0, trace_path=trace_path, ends_on_horizon=False
    )


def make_gateway(episode: ServedEpisode, stream: ScriptedStream) -> StreamGateway:
    spec = episode.describe()
    return StreamGateway(
        stream,  # type: ignore[arg-type]
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )


async def refused(awaitable: Any) -> str:
    """Return the protocol error code a refused call carries."""
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001 - the code is the assertion
        record = json.loads(str(error))
        assert record["kind"] == "protocol_error"
        assert record["protocol_version"] == 2
        return record["code"]
    raise AssertionError("the call was accepted")


def test_the_terminal_is_the_one_tool_that_can_end_an_attempt(episode: ServedEpisode) -> None:
    """The env's score terminal, or its abort. Wordle scores on the reserved abort."""
    manifest = terminal_manifest(episode.describe())
    assert manifest.name == "terminate"
    assert manifest.terminal_kind == "abort"
    # The stream holds a terminal call to a set of names, so a schema is read down to one.
    assert declared_argument_names(manifest.input_schema) == []
    assert declared_argument_names({"properties": {"a": {}, "b": {}}}) == ["a", "b"]
    assert declared_argument_names({"properties": {"a": {}, "b": {}}, "required": ["b"]}) == ["b"]


def test_an_abort_beside_a_score_terminal_is_not_served() -> None:
    """One terminal, and it is the one the stream knows about.

    An env that scores also advertises the reserved abort, and a call to it would end that
    env's episode with no terminal request, leaving the stream holding an attempt nothing can
    still seal.
    """
    spec = TaskSpec(
        env_name="scoring",
        instructions="File it.",
        tools=[
            ToolManifest(name="act", description="d", input_schema={"type": "object"}),
            ToolManifest(
                name="submit",
                description="d",
                input_schema={"type": "object"},
                terminal_kind="score",
            ),
            ToolManifest(
                name="terminate",
                description="d",
                input_schema={"type": "object"},
                terminal_kind="abort",
            ),
        ],
    )
    terminal = terminal_manifest(spec)
    assert terminal.name == "submit"
    assert [tool.name for tool in wrapped_manifests(spec, terminal)] == ["act", "submit"]
    # Wordle's terminal is the abort itself, so nothing is dropped from it.
    wordle = TaskSpec(
        env_name="wordle",
        instructions="Guess.",
        tools=[
            ToolManifest(
                name="terminate",
                description="d",
                input_schema={"type": "object"},
                terminal_kind="abort",
            )
        ],
    )
    assert [t.name for t in wrapped_manifests(wordle, terminal_manifest(wordle))] == ["terminate"]


def hashing_spec() -> TaskSpec:
    """An env with one of everything the served surface is made of."""
    return TaskSpec(
        env_name="filing",
        task_id="7",
        instructions="File the bands.",
        horizon=10,
        tools=[
            ToolManifest(
                name="act",
                description="Do one thing.",
                input_schema={"type": "object", "properties": {"note": {"type": "string"}}},
            ),
            ToolManifest(
                name="file_bands",
                description="File one record per band.",
                input_schema={
                    "type": "object",
                    "properties": {"entries": {"type": "array"}},
                    "required": ["entries"],
                },
                terminal_kind="score",
            ),
            ToolManifest(
                name="terminate",
                description="Give up.",
                input_schema={"type": "object"},
                terminal_kind="abort",
            ),
        ],
    )


def configuration_of(spec: TaskSpec) -> str:
    """The digest a generation serving ``spec`` is started under."""
    return _configuration_hash(spec, terminal_manifest(spec))


def with_tool(spec: TaskSpec, called: str, **changes: Any) -> TaskSpec:
    """The same env with one thing about the tool named ``called`` changed."""
    return spec.model_copy(
        update={
            "tools": [
                tool.model_copy(update=changes) if tool.name == called else tool
                for tool in spec.tools
            ]
        }
    )


MUTATIONS = {
    "env name": lambda spec: spec.model_copy(update={"env_name": "other"}),
    "task id": lambda spec: spec.model_copy(update={"task_id": "8"}),
    "contract version": lambda spec: spec.model_copy(update={"contract_version": 3}),
    "instructions": lambda spec: spec.model_copy(update={"instructions": "File the songs."}),
    "horizon": lambda spec: spec.model_copy(update={"horizon": 11}),
    "a tool's name": lambda spec: with_tool(spec, "act", name="act_twice"),
    "a tool's description": lambda spec: with_tool(spec, "act", description="Do one other."),
    "a tool's schema": lambda spec: with_tool(
        spec, "act", input_schema={"type": "object", "properties": {"note": {"type": "integer"}}}
    ),
    "the terminal's description": lambda spec: with_tool(
        spec, "file_bands", description="File one record per song."
    ),
    "the terminal's arguments": lambda spec: with_tool(
        spec,
        "file_bands",
        input_schema={
            "type": "object",
            "properties": {"songs": {"type": "array"}},
            "required": ["songs"],
        },
    ),
    "which tool is the terminal": lambda spec: with_tool(spec, "file_bands", terminal_kind="none"),
}


@pytest.mark.parametrize("changed", sorted(MUTATIONS))
def test_a_change_the_model_could_see_is_a_changed_configuration(changed: str) -> None:
    """The digest is what a resume has to serve the same thing under, so it covers all of it.

    Everything the model reads is in here, a tool's description as much as its name, and so is
    everything that decides how the served episode behaves, the horizon among it. A digest that
    left one of them out would let a resume pass verification while serving other instructions
    or allowing a different number of environment actions.
    """
    base = hashing_spec()
    changed_spec = MUTATIONS[changed](base)
    assert changed_spec != base
    assert configuration_of(changed_spec) != configuration_of(base)


@pytest.mark.parametrize(
    "attribute, value",
    [
        ("WRAPPER_VERSION", "shogym.gateway.wrapper.2"),
        ("_PULL_DESCRIPTION", "Ask the stream for something."),
        ("_WRAPPER_NOTE", "\n\nCall this tool some other way."),
    ],
)
def test_what_this_gateway_adds_to_an_environment_is_part_of_the_configuration(
    monkeypatch: pytest.MonkeyPatch, attribute: str, value: str
) -> None:
    """The surface is the environment's manifest and this renderer, so the renderer is in it.

    The model reads a description this gateway appends to, calls a control tool the environment
    never declared, and sends its arguments in a wrapper this gateway shaped. A resume has the
    digest and nothing else to tell one renderer from another.
    """
    spec = hashing_spec()
    before = configuration_of(spec)
    monkeypatch.setattr(f"shogym.serve.protocol_v2.gateway.{attribute}", value)
    assert configuration_of(spec) != before


def test_the_configuration_is_what_this_gateway_serves() -> None:
    """A tool nothing serves is not part of what the generation serves.

    An env that scores advertises the reserved abort as well, and this gateway does not serve
    it: the model never reads its description and no call can reach it. Two generations that
    differ only there are serving the same surface, and the digest says so.
    """
    spec = hashing_spec()
    served = [tool.name for tool in wrapped_manifests(spec, terminal_manifest(spec))]
    assert served == ["act", "file_bands"]
    changed = with_tool(spec, "terminate", description="Give up already.")
    assert configuration_of(changed) == configuration_of(spec)


def test_what_the_environment_is_configured_as_is_part_of_the_configuration() -> None:
    """A setting the model cannot see can still decide what its filing is worth.

    An environment may draw a hidden key, grade against one corpus rather than another, or hand
    an episode a different machine, and none of that appears in a tool description. So a
    generation carries the digest the environment publishes of itself, and a resume under a
    changed one is refused rather than worked and scored against a key nobody drew for it. An
    environment that publishes nothing hashes exactly what it hashed before.
    """
    spec = hashing_spec()
    terminal = terminal_manifest(spec)
    plain = stream_start(spec, terminal, claim_hash="a" * 64, grade=DOUBLE_GRADE)
    assert plain.configuration_hash == configuration_of(spec)
    first = stream_start(
        spec, terminal, claim_hash="a" * 64, environment_digest="pulse-0", grade=DOUBLE_GRADE
    )
    second = stream_start(
        spec, terminal, claim_hash="a" * 64, environment_digest="pulse-1", grade=DOUBLE_GRADE
    )
    assert first.configuration_hash != second.configuration_hash
    assert first.configuration_hash != plain.configuration_hash


def test_an_environment_is_asked_how_its_attempts_end_and_answers_with_its_own_route() -> None:
    """One that brings its own terminal replaces the version, the Activities and the identity.

    The route is what the answer is built over rather than one world, because the Activities are
    registered once and a generation may serve a task after this one. An environment that brings
    nothing keeps the stand-ins, declares this gateway's version, and adds nothing to the
    identity, which is what leaves every other environment's generation byte for byte as it was.
    """
    asked: List[Any] = []

    class Own:
        def protocol_v2_terminal(self, route: Any) -> Any:
            asked.append(route)
            return "world.1", ["seal", "grade"], "config-7"

    own = environment_terminal(SimpleNamespace(env=Own(), session_id="session-1"))
    assert own.canonicalization_version == "world.1"
    assert own.activities == ["seal", "grade"]
    assert own.configuration_digest == "config-7"
    assert asked == [own.route]
    assert own.route("00000000000000000000000000000100") is None

    plain = environment_terminal(SimpleNamespace(env=object(), session_id="session-2"))
    assert plain.canonicalization_version == CANONICALIZATION_VERSION
    assert plain.configuration_digest is None
    assert len(plain.activities) == 4


def test_a_grader_an_environment_declares_without_a_terminal_of_its_own_is_refused() -> None:
    """The grade and the terminal are one fact, so half of it is not a composition.

    An environment that declares a grader and brings no terminal is sealed and graded by the
    kernel's Activities, whose number is a fact about the shape of a filing however the
    environment describes itself. A generation composed over that claim would stamp the honest
    policy, pass every check that reads the declaration, and then fail every seal when the
    stand-in's identity arrived with the score instead. The refusal is where the two halves meet
    and it names the one that is missing.
    """

    class Half:
        def protocol_v2_grade(self) -> Any:
            return DOUBLE_GRADE

    episode = SimpleNamespace(env=Half(), session_id="session-3")
    with pytest.raises(ValueError, match="brings no terminal of its own"):
        environment_grade(episode)
    with pytest.raises(ValueError, match="protocol_v2_terminal"):
        environment_terminal(episode)

    # And the stand-in an environment that says nothing about its grader still gets.
    assert environment_grade(SimpleNamespace(env=object(), session_id="s")).stand_in is True


def test_a_world_belongs_to_the_attempt_it_was_opened_for() -> None:
    """The route says which world each attempt filed in, and answers nothing for the others."""
    route = environment_terminal(SimpleNamespace(env=object(), session_id="session-1")).route
    world = SimpleNamespace(env="env-a", session_id="session-a")
    route.record("00000000000000000000000000000100", world)
    assert route("00000000000000000000000000000100") == ("env-a", "session-a")
    assert route("00000000000000000000000000000200") is None


async def test_a_generation_a_controller_composed_ends_the_way_its_environment_does(
    episode: ServedEpisode, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment says how its attempts end, whoever composed the queue they end in.

    A run whose manifest and schedule are decided above one episode hands this call the
    generation it composed, and that is the ordinary shape for more than one task. The
    environment is still what the acknowledgements declare their digests were taken under and
    still the other half of what the generation is, so both are applied here rather than only
    where this call composes the generation itself. A composed one that kept this gateway's own
    version would have its first terminal refused as a mismatch, by the environment, after the
    world was worked.
    """
    spec = episode.describe()
    terminal = terminal_manifest(spec)
    monkeypatch.setattr(
        episode.env,
        "protocol_v2_terminal",
        lambda route: ("world.1", [], "pulse-7"),
        raising=False,
    )
    environment = environment_terminal(episode)
    # Composed over the grader this environment declares. That claim is inside what the
    # generation is and its honest bodies publish that grader's numbers, so a composition naming
    # another one is not a composition this environment can be opened over.
    composed = stream_start(
        spec, terminal, claim_hash="a" * 64, bodies=["one", "two"], grade=environment.grade
    )
    assert composed.canonicalization_version == CANONICALIZATION_VERSION
    started: List[Any] = []

    class Started:
        async def claim_consumer(self, claim: Any) -> Any:
            return SimpleNamespace(initial_cursor=CURSOR)

    async def capture(client: Any, start: Any, *, workflow_id: str) -> Any:
        started.append(start)
        return Started()

    async def opener(attempt_id: str) -> ServedEpisode:
        raise AssertionError("no world is opened by composing a generation")

    monkeypatch.setattr(gateway_module, "start_stream", capture)
    await open_gateway(
        None,  # type: ignore[arg-type]
        episode,
        start=composed,
        open_episode=opener,
        environment=environment,
    )
    assert started[0].canonicalization_version == "world.1"
    assert started[0].configuration_hash == _configuration_hash(spec, terminal, "pulse-7")
    assert started[0].tasks == composed.tasks
    # The controller's own object is left as it composed it.
    assert composed.canonicalization_version == CANONICALIZATION_VERSION


class ClaimedStream(ScriptedStream):
    """The scripted stream, reachable through the call that starts a generation on one."""

    async def claim_consumer(self, claim: Any) -> Any:
        return SimpleNamespace(initial_cursor=CURSOR)


async def test_a_world_that_is_not_the_environment_the_generation_declared_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A later task is worked in the environment the generation committed to, or in none.

    The generation's identity carries what its environment is configured as, taken from the
    episode it was opened on, and that is what a resume is held to. Every task after the first
    is worked in a world this gateway opens, and an opener that answered with a differently
    configured environment would have that task scored under a hidden rule the generation never
    committed to, while its own hash still named the first one. So the answer is checked before
    the world is routed or the task presented, and a world that is not what the generation
    declared is let go of rather than served.
    """
    first = await scoring_world()
    later = await scoring_world()
    for world, digest in ((first, "pulse-0"), (later, "pulse-1")):
        monkeypatch.setattr(
            world.env,
            "protocol_v2_terminal",
            lambda route, digest=digest: ("world.1", [], digest),
            raising=False,
        )
        monkeypatch.setattr(
            world.env, "protocol_v2_grade", lambda: DOUBLE_GRADE, raising=False
        )
    environment = environment_terminal(first)
    spec = first.describe()
    composed = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash="a" * 64,
        bodies=["one", "two"],
        grade=DOUBLE_GRADE,
    )
    stream = ClaimedStream(TASK_OFFER, ACK_OFFER, SECOND_TASK_OFFER)

    async def started(client: Any, start: Any, *, workflow_id: str) -> Any:
        return stream

    opened: List[str] = []

    async def opener(attempt_id: str) -> ServedEpisode:
        opened.append(attempt_id)
        return later

    monkeypatch.setattr(gateway_module, "start_stream", started)
    gateway = await open_gateway(
        None,  # type: ignore[arg-type]
        first,
        start=composed,
        open_episode=opener,
        environment=environment,
    )
    assert json.loads(await gateway.pull({}))["kind"] == "task"
    filing = {"attempt_id": ATTEMPT, "arguments": {"answer": "4"}}
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"

    with pytest.raises(RuntimeError):
        await gateway.pull({})
    assert opened == [SECOND_ATTEMPT]
    # The world was let go of, so nothing is left running in a configuration nothing serves.
    assert score_mcp.gold(later.session_id) == ""
    # It was never routed, so no seal can reach it, and the task was never presented, so the
    # model never saw work it would have had scored under it.
    assert environment.route(SECOND_ATTEMPT) is None
    assert gateway.cursor == ACK_ID
    assert [commit.message_id for commit in stream.commits] == [TASK_ID, ACK_ID]
    assert stream.pending is SECOND_TASK_OFFER
    await gateway.aclose()


async def test_a_world_the_stream_sealed_is_let_go_of_without_a_second_ending(
    tmp_path: Path,
) -> None:
    """The stream sealed and scored this attempt, so nothing here may end it a second time.

    An environment that scores its own terminal brings a seal lifecycle and a durable
    finalization store with it, and under this protocol a filing reaches neither: it becomes the
    stream's terminal request, and what it was worth comes back in the acknowledgement. The
    world is still let go of as that acknowledgement is presented, and an ordinary close reads
    the untouched lifecycle as an episode that ended without a seal and claims an abort for it.
    That abort is a second result for a scored attempt, in a durable record and in the trace,
    and it says the attempt was aborted and worth nothing.
    """
    trace = tmp_path / "run.jsonl"
    episode = await scoring_world(trace)
    session = episode.session_id
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    try:
        assert json.loads(await gateway.pull({}))["kind"] == "task"
        played = await gateway.environment("noop", {"attempt_id": ATTEMPT, "arguments": {}})
        assert json.loads(played.content[0].text)["ok"]
        filing = {"attempt_id": ATTEMPT, "arguments": {"answer": "4"}}
        assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    finally:
        await gateway.aclose()

    # The world is let go of the way any other is: the session it was worked in is released.
    assert score_mcp.gold(session) == ""
    # And nothing ended it a second time. The durable store holds no record, the episode has no
    # verdict of its own, and the trace ends at the last call that really happened.
    assert episode._store is not None
    assert episode._store.load_all() == []
    assert episode._evidence is None
    assert episode._terminal_feedback == []
    rows = load_traces(trace)
    assert [row["tool"] for row in rows] == ["noop"]
    assert [row for row in rows if row.get("terminated") or "verdict" in row] == []


async def test_pull_takes_nothing_and_every_environment_tool_is_wrapped(
    episode: ServedEpisode,
) -> None:
    """The advertised surface: one control tool with a closed empty schema, and wrappers."""
    server = build_gateway_server(make_gateway(episode, ScriptedStream()))
    async with Client(server) as client:
        schemas = {tool.name: tool.inputSchema for tool in await client.list_tools()}
    assert set(schemas) == {PULL_TOOL, "guess", "terminate"}
    assert schemas[PULL_TOOL] == {"type": "object", "properties": {}, "additionalProperties": False}
    spec = episode.describe()
    served = wrapped_manifests(spec, terminal_manifest(spec))
    assert {manifest.name for manifest in served} == {"guess", "terminate"}
    # Every one of them, the terminal included. The model builds its terminal call from what
    # this schema says, so a terminal advertising its own arguments rather than the wrapper
    # would have it omit the attempt_id the filing is routed by.
    for manifest in served:
        wrapper = schemas[manifest.name]
        assert wrapper["type"] == "object"
        assert wrapper["required"] == ["attempt_id", "arguments"]
        assert wrapper["additionalProperties"] is False
        assert set(wrapper["properties"]) == {"attempt_id", "arguments"}
        assert wrapper["properties"]["attempt_id"]["type"] == "string"
        # The native schema is nested rather than dropped: the model still has to know what a
        # `guess` takes, and the wrapper is what it takes it in.
        assert wrapper["properties"]["arguments"] == manifest.input_schema
    assert schemas["guess"]["properties"]["arguments"]["required"] == ["word"]


async def test_a_native_tool_named_pull_is_refused_at_construction(
    episode: ServedEpisode,
) -> None:
    """Two tools of one name cannot both be reachable, and the control tool would lose."""
    gateway = make_gateway(episode, ScriptedStream())
    spec = gateway.spec
    spec.tools[0] = spec.tools[0].model_copy(update={"name": PULL_TOOL})
    with pytest.raises(ValueError, match="collides with the stream control tool"):
        build_gateway_server(gateway)


async def test_a_malformed_call_never_reaches_the_stream(episode: ServedEpisode) -> None:
    """Every wrapper the protocol does not define is refused before a request is built."""
    stream = ScriptedStream(TASK_OFFER)
    gateway = make_gateway(episode, stream)
    assert await refused(gateway.pull({"cursor": CURSOR})) == "invalid_message"
    for wrapper in (
        {},
        {"attempt_id": ATTEMPT},
        {"arguments": {}},
        {"attempt_id": ATTEMPT, "arguments": {}, "hint": 1},
        {"attempt_id": "not-an-opaque-id", "arguments": {}},
        {"attempt_id": ATTEMPT, "arguments": "word=crane"},
    ):
        assert await refused(gateway.terminal(wrapper)) == "invalid_message"
        assert await refused(gateway.environment("guess", wrapper)) == "invalid_message"
    assert stream.calls == []


async def test_a_malformed_native_call_never_reaches_the_environment(
    episode: ServedEpisode,
) -> None:
    """The wrapper nests the native schema, so the server that advertised it holds calls to it.

    A call the tool never accepted must not be answered by the environment's own framework, and
    must not spend a step of the episode on its way to being told so.
    """
    server = build_gateway_server(make_gateway(episode, ScriptedStream(TASK_OFFER)))
    async with Client(server) as client:
        await client.call_tool(PULL_TOOL, {})
        for native in (
            {},  # a required argument is missing
            {"word": 5},  # here it is the wrong type
            {"word": "crane", "hint": 1},  # here there is one the schema does not declare
            {"word": "crane", "attempt_id": ATTEMPT},  # including the wrapper's own field name
        ):
            answer = await client.call_tool(
                "guess", {"attempt_id": ATTEMPT, "arguments": native}, raise_on_error=False
            )
            assert answer.is_error
            assert json.loads(answer.content[0].text)["code"] == "invalid_message"
        assert episode._trajectory == []
        # A call the schema does accept is dispatched, and it is the episode's first step.
        played = await client.call_tool(
            "guess", {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
        )
        assert json.loads(played.content[0].text)["valid"] is True
    assert [step.arguments for step in episode._trajectory] == [{"word": "crane"}]


async def test_the_attempt_id_is_the_routing_handle(episode: ServedEpisode) -> None:
    """A call names an attempt this transport is serving, or it is not routed anywhere.

    Both kinds of call are asked, because they are routed in two places. An ordinary call is
    routed on its way to a world, and a terminal is routed as its filing is built, which is a
    separate path with a separate reason to be reached: the one filing a retry repeats is not
    routed at all, so the routing a first filing does is not something the retry rule can be
    read as covering.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    wrapper = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    # No task has been presented, so no attempt is being served yet.
    assert await refused(gateway.environment("guess", wrapper)) == "invalid_attempt"

    await gateway.pull({})
    other = {"attempt_id": "0" * 32, "arguments": {"word": "crane"}}
    assert await refused(gateway.environment("guess", other)) == "invalid_attempt"

    # A well formed identifier this transport was never given, through the tool that ends an
    # attempt. Nothing was filed and nothing was left owed, so the generation is where it was.
    sent = list(stream.requests)
    unserved = {"attempt_id": "0" * 32, "arguments": {}}
    assert await refused(gateway.terminal(unserved)) == "invalid_attempt"
    assert stream.requests == sent
    assert isinstance(gateway._recovery, _Idle)
    assert episode._trajectory == []

    played = await gateway.environment("guess", wrapper)
    # `attempt_id` is stripped: the env sees the arguments it declared and answers about them.
    assert json.loads(played.content[0].text)["score"]
    # And only the tool's own observation comes back. Feedback under this protocol is a
    # presented Payload, never a sidecar on an ordinary result.
    assert played.structured_content is None
    assert not (played.meta or {})

    await gateway.terminal({"attempt_id": ATTEMPT, "arguments": {}})
    # The attempt is sealed, so nothing more can be done to it, by either route.
    assert await refused(gateway.environment("guess", wrapper)) == "invalid_attempt"
    filed = list(stream.requests)
    assert await refused(gateway.terminal({"attempt_id": ATTEMPT, "arguments": {}})) == (
        "invalid_attempt"
    )
    assert stream.requests == filed


async def test_a_result_is_delivered_verbatim_and_attested(episode: ServedEpisode) -> None:
    """The bytes the stream offered are the bytes the model reads, in one text item."""
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    server = build_gateway_server(gateway)
    async with Client(server) as client:
        result = await client.call_tool(PULL_TOOL, {})
        assert len(result.content) == 1
        assert result.content[0].text == TASK_OFFER.visible_text
        assert result.structured_content is None
        assert gateway.cursor == TASK_ID

        ack = await client.call_tool("terminate", {"attempt_id": ATTEMPT, "arguments": {}})
        assert ack.content[0].text == ACK_OFFER.visible_text
        assert gateway.cursor == ACK_ID

    assert [commit.message_id for commit in stream.commits] == [TASK_ID, ACK_ID]
    # A Task presentation carries the state a crash restores from, and an acknowledgement
    # carries the provider turn it is the last result of. Neither carries the other's.
    task_blobs, ack_blobs = stream.blobs
    assert task_blobs["checkpoint"] is not None and task_blobs["provider_turn"] is None
    assert ack_blobs["provider_turn"] is not None and ack_blobs["checkpoint"] is None
    # The transcript hash covers what has been presented, so it moves with each delivery.
    assert task_blobs["transcript"] != ack_blobs["transcript"]


async def test_every_result_crosses_the_boundary_as_one_text_item(
    episode: ServedEpisode,
) -> None:
    """Task, Wait, Payload, Done, SealAck, SealReject: one text item each, and nothing beside.

    The rule is about what the transport delivers, so every kind goes through the real server
    rather than through a return value the server would still have to render.
    """
    wait = offered(Wait(message_id="0" * 31 + "3", retry_after_ms=1000))
    reject = offered(
        SealReject(message_id="0" * 31 + "4", attempt_id=ATTEMPT, body="missing word"), ATTEMPT
    )
    payload = offered(
        Payload(message_id="0" * 31 + "5", attempt_id=ATTEMPT, body="receipt 0"), ATTEMPT
    )
    order = (TASK_OFFER, wait, reject, ACK_OFFER, payload, DONE_OFFER)
    gateway = make_gateway(episode, ScriptedStream(*order))
    server = build_gateway_server(gateway)
    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    async with Client(server) as client:
        answers = [
            await client.call_tool(PULL_TOOL, {}),
            await client.call_tool(PULL_TOOL, {}),
            await client.call_tool("terminate", filing),
            await client.call_tool("terminate", filing),
            await client.call_tool(PULL_TOOL, {}),
            await client.call_tool(PULL_TOOL, {}),
        ]
    for answer, message in zip(answers, order):
        assert len(answer.content) == 1
        assert answer.content[0].type == "text"
        assert answer.content[0].text == message.visible_text
        assert answer.structured_content is None


async def test_a_refused_filing_leaves_the_attempt_where_it_was(episode: ServedEpisode) -> None:
    """A SealReject is a result like any other, and the attempt is still there to file again."""
    reject = offered(
        SealReject(message_id="0" * 31 + "7", attempt_id=ATTEMPT, body="missing word"), ATTEMPT
    )
    stream = ScriptedStream(TASK_OFFER, reject, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})
    answer = await gateway.terminal({"attempt_id": ATTEMPT, "arguments": {}})
    assert json.loads(answer)["kind"] == "seal_reject"
    # It is not the last result of a completed provider turn, so it carries neither blob.
    assert stream.blobs[-1]["provider_turn"] is None
    assert stream.blobs[-1]["checkpoint"] is None
    # And the attempt is still the one this transport routes to, so a later filing lands. It is
    # the same call again by every visible measure, and it reaches the stream rather than being
    # answered from what the first one was told: nothing here keeps results under a call's name.
    again = await gateway.terminal({"attempt_id": ATTEMPT, "arguments": {}})
    assert json.loads(again)["kind"] == "seal_ack"
    assert stream.calls.count("seal") == 2
    assert stream.requests[-1] is not stream.requests[-2]


GUESS = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
FILING = {"attempt_id": ATTEMPT, "arguments": {}}


async def every_kind_overlaps(gateway: StreamGateway) -> None:
    """Every kind of call, including a repeat of the running one, is refused while it runs."""
    assert await refused(gateway.pull({})) == "overlapping_call"
    assert await refused(gateway.terminal(FILING)) == "overlapping_call"
    assert await refused(gateway.environment("guess", GUESS)) == "overlapping_call"
    assert await refused(gateway.environment("guess", {**GUESS, "arguments": {"word": "stack"}}))


async def test_one_call_is_in_flight_at_a_time(episode: ServedEpisode) -> None:
    """A second call while one is running is refused, not queued behind it.

    Each kind of call is gated in turn, and while it is running every other kind, another call
    of its own kind, and the exact retry of the call that is running are all refused. What that
    costs is a refusal. What it buys is that nothing runs afterwards out of a backlog, because a
    refused call was answered and is over rather than kept for later.
    """
    # A pull holds it.
    stream = ScriptedStream(TASK_OFFER)
    stream.gate = asyncio.Event()
    gateway = make_gateway(episode, stream)
    pulling = asyncio.create_task(gateway.pull({}))
    await asyncio.sleep(0.01)
    await every_kind_overlaps(gateway)
    stream.gate.set()
    assert (await pulling) == TASK_OFFER.visible_text
    assert stream.calls == ["pull", "present"]

    # A terminal holds it, from the moment the seal is sent.
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})
    stream.gate = asyncio.Event()
    sealing = asyncio.create_task(gateway.terminal(FILING))
    await asyncio.sleep(0.01)
    await every_kind_overlaps(gateway)
    stream.gate.set()
    assert json.loads(await sealing)["kind"] == "seal_ack"
    assert stream.calls == ["pull", "present", "seal", "present"]

    # And an ordinary call holds it while a world this transport does not own is changing.
    world = BlockingEpisode()
    spec = episode.describe()
    stream = ScriptedStream(TASK_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    await gateway.pull({})
    playing = asyncio.create_task(gateway.environment("guess", GUESS))
    await asyncio.sleep(0.01)
    await every_kind_overlaps(gateway)
    world.gate.set()
    assert (await playing).content[0].text == "landed"
    # None of the refused calls ran later: one guess reached the world, and no request that was
    # turned away ever reached the stream.
    assert world.landed == ["guess"]
    assert stream.calls == ["pull", "present", "environment"]


async def test_nothing_is_served_after_done(episode: ServedEpisode) -> None:
    """Done ends the generation, and it ends this transport's part in it too."""
    gateway = make_gateway(episode, ScriptedStream(DONE_OFFER))
    assert (await gateway.pull({})) == DONE_OFFER.visible_text
    assert await refused(gateway.pull({})) == "closed_stream"
    assert (
        await refused(gateway.environment("guess", {"attempt_id": ATTEMPT, "arguments": {}}))
        == "closed_stream"
    )


async def test_a_finished_execution_reads_as_a_closed_stream(episode: ServedEpisode) -> None:
    """The transport says the execution is over; the protocol says the stream is closed."""
    finished = RPCError(
        "workflow execution already completed", RPCStatusCode.NOT_FOUND, b""
    )
    gateway = make_gateway(episode, ScriptedStream(finished))
    assert await refused(gateway.pull({})) == "closed_stream"


async def test_a_fault_is_not_a_protocol_answer(episode: ServedEpisode) -> None:
    """A failure with no code is raised as itself, so nothing invents an answer for it."""
    gateway = make_gateway(episode, ScriptedStream(RuntimeError("the worker died")))
    with pytest.raises(RuntimeError, match="the worker died"):
        await gateway.pull({})


async def test_an_answer_that_never_arrived_is_recovered_by_the_same_request(
    episode: ServedEpisode,
) -> None:
    """A call is repeated under the identity it was sent with, so a lost answer is reachable.

    The stream reserves a result for the request that asked for it and refuses a second ask
    while that result is outstanding. A fresh identity would therefore strand the generation
    the first ask left waiting.
    """
    lost = RuntimeError("the answer never arrived")
    stream = ScriptedStream(lost, TASK_OFFER, lost, ACK_OFFER, DONE_OFFER)
    gateway = make_gateway(episode, stream)

    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.pull({})
    assert (await gateway.pull({})) == TASK_OFFER.visible_text
    first, second = stream.requests[:2]
    assert first.request_id == second.request_id

    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.terminal(filing)
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    third, fourth = stream.requests[2:4]
    assert third.metadata.request_id == fourth.metadata.request_id

    # And a call made after an answer was delivered is a new request, because the cursor it
    # would carry has moved past everything the outstanding one could have been given.
    await gateway.pull({})
    assert stream.requests[-1].request_id != second.request_id


async def test_a_result_the_stream_is_holding_lets_nothing_else_through(
    episode: ServedEpisode,
) -> None:
    """A message the stream reserved and this transport never saw still holds the generation.

    The other side of a lost answer: the request reached the stream, the stream took a message
    out of the queue for it, and only the answer went missing. The stream refuses every other
    request while it holds that message, so the generation is waiting on one call. An ordinary
    environment call never reaches the stream to be told so, and the world it would change is
    one the stream cannot see, so a Wait the model has not read must not be the moment a guess
    is played. The exact pull is what collects it, and the episode is untouched until it does.
    """
    wait = offered(Wait(message_id="0" * 31 + "a", retry_after_ms=1000))
    stream = ScriptedStream(TASK_OFFER, wait)
    gateway = make_gateway(episode, stream)
    assert (await gateway.pull({})) == TASK_OFFER.visible_text

    stream.lose_next_result = True
    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.pull({})
    # The stream is holding the Wait it offered, and nothing here knows what it says. What this
    # gateway does know is which request it is being held for, and that it is unfinished.
    assert stream.pending is not None and stream.pending.kind == "wait"
    record = gateway._recovery
    assert isinstance(record, _RequestUncertain)
    assert record.request is stream.requests[-1]
    sent = len(stream.requests)

    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    assert await refused(gateway.environment("guess", guess)) == "outstanding_response"
    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    assert await refused(gateway.terminal(filing)) == "outstanding_response"
    # Neither of them allocated a request, reached the stream, or moved what is unfinished.
    assert len(stream.requests) == sent
    assert gateway._recovery is record
    assert episode._trajectory == []

    # A gateway that has forgotten which request the message is reserved for cannot collect it
    # either, and the stream would refuse it too, so it refuses rather than reaching the world
    # first and asking afterwards.
    gateway._recovery = _Idle()
    assert await refused(gateway.environment("guess", guess)) == "outstanding_response"
    assert await refused(gateway.pull({})) == "outstanding_response"
    gateway._recovery = record

    assert (await gateway.pull({})) == wait.visible_text
    assert stream.requests[-1] is record.request
    assert episode._trajectory == []


def filing_spec() -> TaskSpec:
    """An env whose terminal declares typed arguments, as several in this catalogue do."""
    return TaskSpec(
        env_name="filing",
        instructions="File the bands.",
        tools=[
            ToolManifest(
                name="terminate",
                description="d",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                terminal_kind="abort",
            ),
            ToolManifest(
                name="file_bands",
                description="File one record per band.",
                input_schema={
                    "type": "object",
                    "properties": {"entries": {"type": "array", "items": {"type": "object"}}},
                    "required": ["entries"],
                    "additionalProperties": False,
                },
                terminal_kind="score",
            ),
        ],
    )


async def test_a_terminal_filing_its_own_schema_refuses_is_never_sealed(
    episode: ServedEpisode,
) -> None:
    """The terminal is wrapped like every other tool, so it is held to what it advertised.

    The stream holds a filing to the names its tool declares and not to their types, and a seal
    is the one answer nothing later can take back. So a filing the advertised schema refuses is
    refused here, with the attempt still active and a well formed filing still to come.
    """
    spec = filing_spec()
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    server = build_gateway_server(gateway)
    async with Client(server) as client:
        await client.call_tool(PULL_TOOL, {})
        for native in (
            {},  # the one required argument is missing
            {"entries": 5},  # the name is there, and it is not what the schema says it is
            {"entries": [1, 2]},  # here the items are not
            {"entries": [], "note": "x"},  # and here there is one the schema does not declare
        ):
            answer = await client.call_tool(
                "file_bands",
                {"attempt_id": ATTEMPT, "arguments": native},
                raise_on_error=False,
            )
            assert answer.is_error
            assert json.loads(answer.content[0].text)["code"] == "invalid_message"
        # None of them became a terminal request, so the attempt is still there to file for.
        assert "seal" not in stream.calls
        filed = await client.call_tool(
            "file_bands", {"attempt_id": ATTEMPT, "arguments": {"entries": [{"band": "a"}]}}
        )
        assert json.loads(filed.content[0].text)["kind"] == "seal_ack"
    assert stream.requests[-1].native_arguments == {"entries": [{"band": "a"}]}


async def test_a_typed_terminal_advertises_the_wrapper_and_not_its_own_arguments(
    episode: ServedEpisode,
) -> None:
    """The other half of the same rule, where the terminal's own arguments are worth confusing.

    A terminal that scores declares real arguments, so a wrapper regression that reached only it
    would advertise those arguments at the top level. Nothing below would notice: the tool
    forwards whatever it is given. What notices is the model, which would file without the
    attempt_id and have the filing refused instead of sealing the attempt.
    """
    spec = filing_spec()
    gateway = StreamGateway(
        ScriptedStream(),  # type: ignore[arg-type]
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    async with Client(build_gateway_server(gateway)) as client:
        schemas = {tool.name: tool.inputSchema for tool in await client.list_tools()}
    # The reserved abort is not served beside a scoring terminal, so the terminal is the only
    # environment tool here and the only one whose wrapper there is to get wrong.
    assert set(schemas) == {PULL_TOOL, "file_bands"}
    for manifest in wrapped_manifests(spec, terminal_manifest(spec)):
        wrapper = schemas[manifest.name]
        assert wrapper["required"] == ["attempt_id", "arguments"]
        assert wrapper["additionalProperties"] is False
        assert set(wrapper["properties"]) == {"attempt_id", "arguments"}
        assert wrapper["properties"]["arguments"] == manifest.input_schema
    # The terminal's own arguments are where the wrapper says they are, and nowhere else.
    assert "entries" not in schemas["file_bands"]["properties"]
    assert schemas["file_bands"]["properties"]["arguments"]["required"] == ["entries"]


async def test_a_presentation_whose_answer_was_lost_is_finished_by_the_same_commit(
    episode: ServedEpisode,
) -> None:
    """A presentation that commits and loses its answer is repeated, never replaced.

    The stream has advanced past it, so every request this gateway could build from its own
    stale cursor is one the stream refuses. The attestation is what the stream answers a second
    time, so the gateway keeps the exact commit it made and sends that one again. And then it
    returns the message that commit attested to, because that is the message the stream has
    counted as read and nothing later can put it in front of the model again.
    """
    payload = offered(
        Payload(message_id="0" * 31 + "8", attempt_id=ATTEMPT, body="receipt 0"), ATTEMPT
    )
    stream = ScriptedStream(TASK_OFFER, payload)
    stream.lose_next_ack = True
    gateway = make_gateway(episode, stream)

    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.pull({})
    # The stream is past the task and this gateway does not know it yet.
    assert stream.cursor == TASK_ID
    assert gateway.cursor == CURSOR

    assert (await gateway.pull({})) == TASK_OFFER.visible_text
    assert stream.commits[1] is stream.commits[0]
    # Repeating it was not a second presentation, and it was not a second pull either: the task
    # was applied once, and the message it was applied to is the one that came back.
    assert stream.calls == ["pull", "present"]
    assert gateway.cursor == TASK_ID

    # Only then is there anything else to ask for.
    assert (await gateway.pull({})) == payload.visible_text
    assert stream.calls == ["pull", "present", "pull", "present"]


async def test_a_refused_attestation_is_repaired_by_attesting_the_message_again(
    episode: ServedEpisode,
) -> None:
    """A refusal decides the attestation, and it decides nothing about the offer.

    A stream verifies an attestation against things that are not in it, the blobs its references
    name among them, and it refuses one it cannot verify without applying anything. So the
    message is still reserved for the request that asked for it, and a gateway that dropped its
    record there would have thrown away the only handle to it: every later call would read a
    message pending, have nothing to present it with, and be refused for a response that nothing
    left could produce.

    What is kept is the message, not the attestation. The attestation is the one thing the
    refusal settled, and a stream reaches it by an identity its own call ID is derived from, so
    sending it again collects that refusal rather than verifying anything a second time. Putting
    the reference right would then change nothing at all. The retry attests the same message
    afresh, which is honest for the same reason the exact replay is honest after a lost answer:
    the stream applied nothing, so it is still the stream this new attestation describes.
    """
    stream = ScriptedStream(TASK_OFFER, DONE_OFFER)
    gateway = make_gateway(episode, stream)

    stream.refuse_next_commit = True
    assert await refused(gateway.pull({})) == "invalid_message"
    # The refusal applied nothing: the offer is still reserved and the cursor has not moved.
    assert stream.pending is TASK_OFFER
    assert stream.cursor == CURSOR
    assert stream.calls == ["pull"]
    record = gateway._recovery
    assert isinstance(record, _PresentationRefused)

    # Those bytes are owed to the call that asked for them, so nothing else may attest to them.
    assert await refused(gateway.terminal(FILING)) == "outstanding_response"
    assert gateway._recovery is record
    assert stream.calls == ["pull"]

    # The same call attests the same message under a new attestation, built from a stream that
    # has not moved, and the bytes it was holding are delivered.
    assert (await gateway.pull({})) == TASK_OFFER.visible_text
    assert stream.commits[-1] is not stream.commits[0]
    assert stream.commits[-1].attestation_id != stream.commits[0].attestation_id
    assert stream.commits[-1].message_id == TASK_ID
    assert stream.commits[-1].cursor_before == CURSOR
    # One delivery, however many attestations it took to make it.
    assert stream.calls == ["pull", "present"]
    assert gateway.cursor == TASK_ID
    # And with the message delivered, the generation serves the next one.
    assert (await gateway.pull({})) == DONE_OFFER.visible_text


async def test_a_refused_attestation_for_a_message_the_stream_let_go_of_is_not_pretended_away(
    episode: ServedEpisode,
) -> None:
    """The other end of a refusal: the offer this gateway was keeping is no longer there.

    Reconciling with the stream is what says which of the two happened. When the message it was
    holding is gone, those bytes were offered and never presented, and this gateway cannot say
    what became of them. It says so, rather than clearing its record and serving the next call
    as though nothing had been offered at all.
    """
    stream = ScriptedStream(TASK_OFFER)
    gateway = make_gateway(episode, stream)

    stream.refuse_next_commit = True
    assert await refused(gateway.pull({})) == "invalid_message"
    assert isinstance(gateway._recovery, _PresentationRefused)

    stream.pending = None
    with pytest.raises(RuntimeError, match="no longer holding"):
        await gateway.pull({})
    assert isinstance(gateway._recovery, _PresentationRefused)


async def test_a_delivery_the_transport_never_returned_is_what_the_next_call_gets(
    episode: ServedEpisode,
) -> None:
    """A message the stream has counted as read has to reach the model, before anything newer.

    The presentation commits and its acknowledgement goes missing, so the tool call fails with
    no result to return and those bytes never crossed the transport. The stream counted the
    delivery anyway. What recovers it is a retry of the call that asked: the commit is repeated
    and the exact message it attested to comes back, rather than the message after it.
    """
    payload = offered(
        Payload(message_id="0" * 31 + "9", attempt_id=ATTEMPT, body="receipt 0"), ATTEMPT
    )
    stream = ScriptedStream(TASK_OFFER, payload, DONE_OFFER)
    gateway = make_gateway(episode, stream)
    assert (await gateway.pull({})) == TASK_OFFER.visible_text

    stream.lose_next_ack = True
    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.pull({})

    # Nothing else is served while those bytes are owed, and the episode is not touched.
    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    committed = len(stream.commits)
    assert await refused(gateway.environment("guess", guess)) == "outstanding_response"
    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    assert await refused(gateway.terminal(filing)) == "outstanding_response"
    assert episode._trajectory == []
    # A call that is not the owner commits nothing and changes nothing on its way to be refused.
    assert len(stream.commits) == committed
    assert stream.cursor == payload.message_id

    assert (await gateway.pull({})) == payload.visible_text
    assert (await gateway.pull({})) == DONE_OFFER.visible_text


async def test_an_acknowledgement_the_transport_never_returned_still_ended_the_attempt(
    episode: ServedEpisode,
) -> None:
    """The other half of the same cut: a seal whose acknowledgement was lost is still a seal.

    The attempt is over the moment the stream applies that presentation, so an ordinary call
    must not reach the episode afterwards. The acknowledgement is owed to the filing that asked
    for it, and until that filing comes back for it nothing else is served either.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})

    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    stream.lose_next_ack = True
    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.terminal(filing)
    assert stream.attempts[ATTEMPT] == "ack_presented"

    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    assert await refused(gateway.environment("guess", guess)) == "outstanding_response"
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    # And with the acknowledgement collected, the attempt is one nothing routes to.
    assert await refused(gateway.environment("guess", guess)) == "invalid_attempt"
    assert episode._trajectory == []


async def test_a_changed_filing_never_displaces_the_one_that_may_have_sealed(
    episode: ServedEpisode,
) -> None:
    """A filing whose answer was lost may be the one that sealed, so it is kept until it lands.

    Its acknowledgement is reachable through that request and through no other. A model that
    revises a valid filing after an ambiguous failure is refused, the first request is left
    where it was, and the retry that repeats it reaches the acknowledgement the stream minted.
    The two are told apart by their canonical bytes, which is what the stream computes the
    request's identity from.
    """
    spec = filing_spec()
    lost = RuntimeError("the answer never arrived")
    stream = ScriptedStream(TASK_OFFER, lost, ACK_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    await gateway.pull({})

    filed = {"attempt_id": ATTEMPT, "arguments": {"entries": [{"band": "a"}]}}
    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.terminal(filed)
    sent = len(stream.requests)

    revised = {"attempt_id": ATTEMPT, "arguments": {"entries": [{"band": "b"}]}}
    assert await refused(gateway.terminal(revised)) == "outstanding_response"
    assert len(stream.requests) == sent
    assert json.loads(await gateway.terminal(filed))["kind"] == "seal_ack"
    # The retry was the first filing, under the identity the first filing was sent with.
    assert stream.requests[-1] is stream.requests[-2]


async def test_a_filing_that_sealed_is_what_reaches_the_acknowledgement_it_earned(
    episode: ServedEpisode,
) -> None:
    """A seal that landed and lost its answer left an attempt nothing routes to any more.

    The stream took the seal when it minted the acknowledgement, so the attempt it names is over
    and the acknowledgement is reserved for the request that asked for it. Routing the retry by
    the attempt would turn away the one call that can still reach it, and every other call is
    answered with the outstanding result it cannot have.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})

    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    stream.lose_next_result = True
    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.terminal(filing)
    assert stream.attempts[ATTEMPT] == "sealed"
    filed = stream.requests[-1]
    assert await refused(gateway.pull({})) == "outstanding_response"

    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    # It was the first filing again, under the identity the acknowledgement is reserved for.
    assert stream.requests[-1] is filed
    assert stream.attempts[ATTEMPT] == "ack_presented"


async def test_a_changed_filing_never_collects_what_the_first_one_was_owed(
    episode: ServedEpisode,
) -> None:
    """The bytes a delivery owes are owed to the call that asked, not to the next one like it.

    A filing commits, its acknowledgement is applied, and the answer never crosses the
    transport. A revised filing is a different submission: it never reached the stream, so what
    it must not be given is the acknowledgement of the submission that did.
    """
    spec = filing_spec()
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        episode,
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    await gateway.pull({})

    filed = {"attempt_id": ATTEMPT, "arguments": {"entries": [{"band": "a"}]}}
    stream.lose_next_ack = True
    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.terminal(filed)

    revised = {"attempt_id": ATTEMPT, "arguments": {"entries": [{"band": "b"}]}}
    assert await refused(gateway.terminal(revised)) == "outstanding_response"
    assert json.loads(await gateway.terminal(filed))["kind"] == "seal_ack"


async def test_the_stream_is_the_authority_when_the_gateway_forgets(
    episode: ServedEpisode,
) -> None:
    """What this gateway keeps between calls is a cache, so losing it costs a query.

    The scenario runs a task, discards everything the gateway believes about the generation,
    and carries on. The cursor its next request carries, the attempt an ordinary call routes
    to, and whether anything may still be served all come back out of the stream itself.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER, DONE_OFFER)
    gateway = make_gateway(episode, stream)
    assert (await gateway.pull({})) == TASK_OFFER.visible_text

    gateway._cursor = CURSOR
    gateway._active = frozenset()
    gateway._closed = False
    gateway._transcript = []

    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    played = await gateway.environment("guess", guess)
    assert json.loads(played.content[0].text)["valid"] is True

    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    # The filing was made from the cursor the stream said it was at, not the forgotten one.
    assert stream.requests[-1].metadata.last_presented_cursor == TASK_ID

    assert (await gateway.pull({})) == DONE_OFFER.visible_text
    assert await refused(gateway.pull({})) == "closed_stream"


class Observation:
    """What an episode call answers with, of which the gateway returns the content."""

    content = "landed"


class BlockingEpisode:
    """An episode whose call has been handed to the world and has not come back yet."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.landed: List[str] = []
        self.fault: Optional[BaseException] = None
        self.closed_after: Optional[List[str]] = None
        self.finalized: Optional[bool] = None

    async def call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        await self.gate.wait()
        if self.fault is not None:
            raise self.fault
        self.landed.append(tool_name)
        return Observation()

    async def close(self, *, finalize: bool = True) -> None:
        """Record what had landed by the time this episode was closed, and how.

        The keyword is the real episode's: a world let go of by this transport is not a world
        whose attempt this transport ends, so what a stream-owned cleanup passes is read here.
        """
        self.closed_after = list(self.landed)
        self.finalized = finalize


async def test_a_cancelled_call_holds_the_gateway_until_the_environment_lands(
    episode: ServedEpisode,
) -> None:
    """A caller that goes away does not take back a call the environment is already running.

    The operation mutates a world this transport does not own, so what holds the gateway is the
    operation and not the caller waiting on it. Until it lands, a pull, a filing and another
    environment call are all refused, and a seal cannot get in front of it.
    """
    world = BlockingEpisode()
    spec = episode.describe()
    stream = ScriptedStream(TASK_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    tool = await build_gateway_server(gateway).get_tool("guess")
    await gateway.pull({})

    handler = asyncio.create_task(
        tool.run({"attempt_id": ATTEMPT, "arguments": {"word": "crane"}})
    )
    await asyncio.sleep(0.05)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler

    assert world.landed == []
    assert await refused(gateway.pull({})) == "overlapping_call"
    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    assert await refused(gateway.terminal(filing)) == "overlapping_call"
    stack = {"attempt_id": ATTEMPT, "arguments": {"word": "stack"}}
    assert await refused(tool.run(stack)) == "overlapping_call"

    world.gate.set()
    await asyncio.sleep(0.05)
    assert world.landed == ["guess"]
    # And the gateway is no longer held by it: the same call is refused now for the observation
    # still owed to the caller that went away, and not for anything being in flight.
    assert await refused(tool.run(stack)) == "outstanding_response"


async def test_a_cancelled_terminal_holds_the_gateway_until_the_seal_is_settled(
    episode: ServedEpisode,
) -> None:
    """A caller that goes away does not take back a seal the stream has already accepted.

    Cancelling the call does not cancel the Update it is waiting on: the seal runs its
    Activities, records its score, and mints its acknowledgement with nobody listening. So the
    gateway is held by that operation and not by the waiter, all the way through the
    acknowledgement's presentation, and an environment call cannot reach the world in between.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    tool = await build_gateway_server(gateway).get_tool("terminate")
    await gateway.pull({})

    stream.gate = asyncio.Event()
    handler = asyncio.create_task(tool.run({"attempt_id": ATTEMPT, "arguments": {}}))
    await asyncio.sleep(0.05)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler

    # The seal is in progress, and everything is refused while it is.
    assert stream.attempts[ATTEMPT] == "active"
    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    filing = {"attempt_id": ATTEMPT, "arguments": {}}
    assert await refused(gateway.pull({})) == "overlapping_call"
    assert await refused(gateway.terminal(filing)) == "overlapping_call"
    assert await refused(gateway.environment("guess", guess)) == "overlapping_call"

    stream.gate.set()
    await asyncio.sleep(0.05)
    # It landed and its acknowledgement was presented, both with no waiter left to see either.
    assert stream.attempts[ATTEMPT] == "ack_presented"
    assert episode._trajectory == []
    assert await refused(gateway.environment("guess", guess)) == "outstanding_response"
    assert json.loads(await gateway.terminal(filing))["kind"] == "seal_ack"
    assert await refused(gateway.environment("guess", guess)) == "invalid_attempt"


async def test_a_cancelled_environment_call_is_answered_rather_than_run_again(
    episode: ServedEpisode,
) -> None:
    """The call landed, so the retry of it is a call for its result and not for a second one.

    A world this transport does not own has already changed by the time the caller goes away.
    Dispatching the same call again would change it twice for one thing the model asked for, so
    what the retry gets is the observation the first one landed with, and nothing else is served
    until it does.
    """
    world = BlockingEpisode()
    spec = episode.describe()
    stream = ScriptedStream(TASK_OFFER, DONE_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    tool = await build_gateway_server(gateway).get_tool("guess")
    await gateway.pull({})

    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    handler = asyncio.create_task(tool.run(guess))
    await asyncio.sleep(0.05)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    world.gate.set()
    await asyncio.sleep(0.05)
    assert world.landed == ["guess"]

    # The observation is owed to the call that asked for it, and to nothing else.
    assert await refused(gateway.pull({})) == "outstanding_response"
    retried = await tool.run(guess)
    assert retried.content[0].text == "landed"
    assert world.landed == ["guess"]
    # And with it collected, the gateway is free again.
    assert (await gateway.pull({})) == DONE_OFFER.visible_text


async def test_a_client_that_stopped_waiting_is_not_something_this_boundary_can_see(
    episode: ServedEpisode,
) -> None:
    """A client's own timeout is a decision it makes alone, and it tells the server nothing.

    The test above is the cut where the handler is cancelled, which is what a client asking for
    a call to stop produces. A client that simply stops waiting produces the other one: the
    handler is never told, so it runs to the end and returns into a response nobody reads.

    What that leaves guaranteed is what this gateway can see. The call runs once and holds the
    gateway while it does, so the repeat that arrives during it is refused rather than sent to
    the world a second time. What it leaves unguaranteed is the repeat that arrives after it.
    Two identical calls are two calls the model could have meant, and a tool call carries no
    identity of its own to tell one from the other, so the second is dispatched. Closing that
    needs an identity the caller supplies and keeps across its own retry, which is a thing
    neither this gateway nor the client on the other end of it has.
    """
    world = BlockingEpisode()
    spec = episode.describe()
    stream = ScriptedStream(TASK_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    server = build_gateway_server(gateway)
    await gateway.pull({})

    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    async with Client(server) as client:
        with pytest.raises(McpError):
            await client.call_tool("guess", guess, timeout=0.05)
        # Nothing reached the handler, so the call is still on its way to the world.
        assert world.landed == []
        during = await client.call_tool("guess", guess, raise_on_error=False)
        assert json.loads(during.content[0].text)["code"] == "overlapping_call"

        world.gate.set()
        await asyncio.sleep(0.05)
        assert world.landed == ["guess"]
        # It landed and its result was returned, so nothing is owed and the world is reached
        # again. This is the boundary's limit rather than the loop's: the retry and a second
        # guess the model meant to play are the same call arriving twice.
        retried = await client.call_tool("guess", guess)
        assert retried.content[0].text == "landed"
        assert world.landed == ["guess", "guess"]


async def test_a_cancelled_final_pull_still_hands_over_the_done_it_asked_for(
    episode: ServedEpisode,
) -> None:
    """Done closed the generation, and the call that asked for it has not read it yet.

    The pull is what the stream counted as read, so those bytes are owed to a pull. Refusing the
    retry because the generation is closed would leave the harness with a protocol error where
    the record telling it to stop should have been.
    """
    stream = ScriptedStream(TASK_OFFER, DONE_OFFER)
    gateway = make_gateway(episode, stream)
    tool = await build_gateway_server(gateway).get_tool(PULL_TOOL)
    await gateway.pull({})

    stream.gate = asyncio.Event()
    handler = asyncio.create_task(tool.run({}))
    await asyncio.sleep(0.05)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    stream.gate.set()
    await asyncio.sleep(0.05)
    assert stream.generation_state == "done"

    guess = {"attempt_id": ATTEMPT, "arguments": {"word": "crane"}}
    assert await refused(gateway.environment("guess", guess)) == "outstanding_response"
    assert (await gateway.pull({})) == DONE_OFFER.visible_text
    # Collected, and now the closed generation is all there is to say.
    assert await refused(gateway.pull({})) == "closed_stream"


# The stream decides an environment call, because nothing else can.


def blocking_gateway(
    episode: ServedEpisode, stream: ScriptedStream, world: BlockingEpisode
) -> StreamGateway:
    """A gateway serving ``episode``'s contract against a world that has not answered yet."""
    spec = episode.describe()
    return StreamGateway(
        stream,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )


async def test_an_environment_call_is_the_streams_decision_and_not_a_remembered_one(
    episode: ServedEpisode,
) -> None:
    """The one call that never reaches the stream is still the stream's to allow.

    A state read followed by a dispatch is two moments, and everything that can take an attempt
    away happens in the second one. So the stream is asked at the moment of the call rather than
    before it, it answers about the attempt it is holding now, and it stays held until the call
    settles. A refusal is the stream's answer and this gateway forwards it; a world it did not
    allow is one this gateway never touched.
    """
    stream = ScriptedStream(TASK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})

    # The stream has decided this writer may not change anything. Nothing here argues with it.
    stream.environment_refusal = StreamProtocolError("fenced_writer")
    assert await refused(gateway.environment("guess", GUESS)) == "fenced_writer"
    assert episode._trajectory == []
    stream.environment_refusal = None

    # And the answer is about the attempt at the moment of the call. Here the attempt ends in
    # the window between this gateway reading the stream and asking it to hold the generation,
    # which is exactly the window a read followed by a dispatch leaves open.
    stream.state_gate = asyncio.Event()
    playing = asyncio.create_task(gateway.environment("guess", GUESS))
    await asyncio.sleep(0.01)
    stream.attempts[ATTEMPT] = "ack_presented"
    stream.state_gate.set()
    assert await refused(playing) == "invalid_attempt"
    assert episode._trajectory == []


async def test_the_stream_stays_held_while_a_world_it_cannot_see_is_changing(
    episode: ServedEpisode,
) -> None:
    """The grant is not a permission slip, it is the generation itself, held until the call ends.

    While an ordinary call is out, every Update that could race it is refused by the stream,
    which is the part a local guard cannot do: a second writer, a resume, or a controller does
    not go through this gateway's memory. The lease is given back when the call settles, however
    it settles, and the stream serves again.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    playing = asyncio.create_task(gateway.environment("guess", GUESS))
    await asyncio.sleep(0.01)
    assert stream.held is not None
    # Everything that reaches the stream while it is held is refused by the stream itself,
    # whether or not it came through this gateway.
    for update in (
        stream.pull(SimpleNamespace(request_id=ACK_ID)),
        stream.close_queue(),
    ):
        with pytest.raises(StreamProtocolError) as raised:
            await update
        assert raised.value.code == "overlapping_call"
    assert stream.queue_closed is False

    world.gate.set()
    assert (await playing).content[0].text == "landed"
    assert stream.held is None
    assert len(stream.leases) == 1

    # A call the world refuses gives the generation back the same way.
    world.gate = asyncio.Event()
    world.fault = RuntimeError("the world went away")
    world.gate.set()
    with pytest.raises(RuntimeError, match="the world went away"):
        await gateway.environment("guess", GUESS)
    assert stream.held is None
    assert isinstance(gateway._recovery, _Idle)


async def test_a_result_this_gateway_kept_is_still_handed_over_by_asking_the_stream(
    episode: ServedEpisode,
) -> None:
    """Nothing retained here is answered from memory alone: the stream is read first, always.

    A retained result is bytes the stream offered and counted as read, and the call collecting
    it reads the stream's state before it gets them and asks the stream again, through a path
    it could be refused on, before they leave. That is what keeps a gateway that has been
    replaced from answering out of its own pocket rather than out of the authority.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    handler = asyncio.create_task(gateway.environment("guess", GUESS))
    await asyncio.sleep(0.01)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    world.gate.set()
    await asyncio.sleep(0.05)
    assert isinstance(gateway._recovery, _ResultOwed)

    reads = stream.state_reads
    confirmations = stream.confirmations
    assert (await gateway.environment("guess", GUESS)).content[0].text == "landed"
    # Read once on the way in, and asked once more on the way out, where a refusal would still
    # have stopped the bytes.
    assert stream.state_reads == reads + 2
    assert stream.confirmations == confirmations + 1
    assert world.landed == ["guess"]


async def test_a_result_kept_by_a_transport_the_generation_left_behind_is_not_handed_over(
    episode: ServedEpisode,
) -> None:
    """A transport that has been replaced does not finish the delivery it was in the middle of.

    The acknowledgement landed, the caller that asked for it was gone before it could be read,
    and it waits under the name of that call. Then the generation gets another owner. Those
    bytes are still bytes the stream counted as read, but the transcript they would reach is now
    somebody else's to write, and a terminal record put there from here is the second consumer
    the generation exists to prevent.

    Reading the state does not catch it: a query has no writer to check and answers both
    transports alike. So the handover asks on the path that does, which is the same path this
    gateway's next real call would be refused on, and it forwards that refusal rather than the
    acknowledgement.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    tool = await build_gateway_server(gateway).get_tool("terminate")
    await gateway.pull({})

    stream.commit_gate = asyncio.Event()
    handler = asyncio.create_task(tool.run(FILING))
    await asyncio.sleep(0.05)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    stream.commit_gate.set()
    await asyncio.sleep(0.05)
    record = gateway._recovery
    assert isinstance(record, _ResultOwed)
    assert json.loads(record.result)["kind"] == "seal_ack"

    # Another owner has the generation now, and this transport is no longer the one writing.
    stream.fenced = True
    assert await refused(gateway.terminal(FILING)) == "fenced_writer"
    assert stream.confirmations == 1
    # And nothing was thrown away over it: what is owed is still owed, to the same call.
    assert gateway._recovery is record
    assert await refused(gateway.pull({})) == "outstanding_response"


async def test_a_call_that_arrives_during_an_environment_call_waits_for_nothing(
    episode: ServedEpisode,
) -> None:
    """A refusal arrives at once, so a caller's own deadline is never spent queueing.

    A call that had to wait its turn would burn its client's timeout behind work it has nothing
    to do with, and then run after that client stopped listening. The refusal is immediate
    instead, and nothing about the running call is disturbed by it.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    server = build_gateway_server(gateway)
    await gateway.pull({})

    async with Client(server) as client:
        playing = asyncio.create_task(client.call_tool("guess", GUESS))
        await asyncio.sleep(0.05)
        loop = asyncio.get_running_loop()
        started = loop.time()
        refusal = await client.call_tool(PULL_TOOL, {}, timeout=10, raise_on_error=False)
        assert loop.time() - started < 5
        assert json.loads(refusal.content[0].text)["code"] == "overlapping_call"
        # It was refused rather than sent, so nothing was offered, presented, or advanced.
        assert stream.calls == ["pull", "present", "environment"]
        assert gateway.cursor == TASK_ID

        world.gate.set()
        assert (await playing).content[0].text == "landed"
    # And nothing ran late once the environment call landed either.
    assert stream.calls == ["pull", "present", "environment"]
    assert world.landed == ["guess"]


# What a lost answer leaves behind, and who may pick it up.


async def test_a_presentation_that_never_reached_the_stream_is_nobody_elses_to_commit(
    episode: ServedEpisode,
) -> None:
    """A call that is not the owner may not commit the owner's presentation, even to fail after.

    The bytes are owed to the pull that asked for them. If an unrelated call commits that
    presentation on its way to being refused, the stream counts the message as read, the cursor
    moves and the delivery is recorded, while neither call has put a single byte in front of the
    model. So the owner is compared first, and a call that is not the owner changes nothing.
    """
    payload = offered(
        Payload(message_id="0" * 31 + "b", attempt_id=ATTEMPT, body="receipt 0"), ATTEMPT
    )
    stream = ScriptedStream(TASK_OFFER, payload)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})

    stream.fault_before_commit = True
    with pytest.raises(RuntimeError, match="the presentation request never arrived"):
        await gateway.pull({})
    record = gateway._recovery
    assert isinstance(record, _PresentationUncertain)
    assert stream.pending is not None and stream.pending.kind == "payload"
    presentations = stream.calls.count("present")
    commits = len(stream.commits)

    for call in (
        gateway.environment("guess", GUESS),
        gateway.environment("guess", {**GUESS, "arguments": {"word": "stack"}}),
        gateway.terminal(FILING),
    ):
        assert await refused(call) == "outstanding_response"
    # Nothing moved: not the presentation, not the cursor, not the delivery the stream counts,
    # not the trajectory, and not the record that says whose presentation this is.
    assert stream.calls.count("present") == presentations
    assert len(stream.commits) == commits
    assert stream.cursor == TASK_ID and gateway.cursor == TASK_ID
    assert stream.pending is not None and stream.pending.kind == "payload"
    assert episode._trajectory == []
    assert gateway._recovery is record

    # The owner comes back, and it is the same attestation that finishes the trip.
    assert (await gateway.pull({})) == payload.visible_text
    assert stream.commits[-1] is record.commit
    assert stream.calls.count("present") == presentations + 1
    assert stream.cursor == payload.message_id


async def test_a_seal_whose_answer_was_lost_before_any_offer_holds_the_generation(
    episode: ServedEpisode,
) -> None:
    """A filing the stream may still be running is the only call that may go on.

    The response channel fails while the stream is sealing, so there is nothing pending for a
    later pull to be refused by and nothing here knows whether an acknowledgement is coming. If
    a pull may allocate a request in that window, it takes the recovery slot from the filing,
    and the acknowledgement the stream later mints is one nothing can ask for again. So the
    filing keeps the slot until its own retry resolves it.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})

    stream.fault_before_offer = True
    with pytest.raises(RuntimeError, match="the response channel failed"):
        await gateway.terminal(FILING)
    # The stream is sealing: nothing is pending, and the attempt is nobody's to work on.
    stream.attempts[ATTEMPT] = "sealing"
    assert stream.pending is None
    record = gateway._recovery
    assert isinstance(record, _RequestUncertain)
    filed = record.request
    sent = len(stream.requests)

    changed = {"attempt_id": ATTEMPT, "arguments": {"note": "revised"}}
    assert await refused(gateway.pull({})) == "outstanding_response"
    assert await refused(gateway.terminal(changed)) == "outstanding_response"
    assert await refused(gateway.environment("guess", GUESS)) == "outstanding_response"
    # None of them allocated a request, sent one, or took the slot the filing is holding.
    assert len(stream.requests) == sent
    assert gateway._recovery is record
    assert episode._trajectory == []

    # The filing comes back under the identity it was sent with, and collects the one Ack.
    assert json.loads(await gateway.terminal(FILING))["kind"] == "seal_ack"
    assert stream.requests[-1] is filed
    assert stream.attempts[ATTEMPT] == "ack_presented"


@pytest.mark.parametrize("lost", ["before the stream", "after the stream"])
async def test_a_release_whose_answer_was_lost_is_the_same_calls_to_send_again(
    episode: ServedEpisode, lost: str
) -> None:
    """The generation is held for one call, so that call is the only one that can give it back.

    A release that faults leaves nobody able to say whether the stream is still holding the
    grant, and no other call can end a hold it does not name. A gateway that walked away from it
    would hand back an observation and leave the generation held against every later call, with
    the one release that could free it forgotten. So the grant stays where the call that made it
    will find it, the observation it landed with stays there too, and the same call again gives
    the generation back and collects what it is owed.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER, DONE_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    if lost == "before the stream":
        stream.fault_before_release = True
    else:
        stream.lose_next_release = True
    world.gate.set()
    with pytest.raises(RuntimeError, match="the release never"):
        await gateway.environment("guess", GUESS)
    assert world.landed == ["guess"]

    # Nothing else is served meanwhile. This gateway is the only thing that knows the
    # observation exists, and the only thing that can end the hold it was landed under.
    record = gateway._recovery
    assert isinstance(record, _LeaseHeld)
    assert await refused(gateway.pull({})) == "outstanding_response"
    assert await refused(gateway.terminal(FILING)) == "outstanding_response"
    other = {"attempt_id": ATTEMPT, "arguments": {"word": "stack"}}
    assert await refused(gateway.environment("guess", other)) == "outstanding_response"
    assert gateway._recovery is record
    assert stream.calls == ["pull", "present", "environment"]

    # The same call again ends the hold and hands over the one observation it landed with.
    assert (await gateway.environment("guess", GUESS)).content[0].text == "landed"
    assert world.landed == ["guess"]
    assert stream.held is None
    assert stream.releases[-1].call_id == record.call.call_id
    assert isinstance(gateway._recovery, _Idle)
    # And with the grant given back, the generation serves again.
    assert (await gateway.pull({})) == DONE_OFFER.visible_text


async def test_a_grant_whose_answer_was_lost_is_given_back_by_the_call_that_asked_for_it(
    episode: ServedEpisode,
) -> None:
    """A grant this gateway never learned it had holds the generation just the same.

    The record is written before the grant is asked for, for the case where no answer comes:
    the release goes by name straight away, and when that release does not land either, the
    grant waits for the call that made it rather than being forgotten. Nothing reached the
    world, so the retry gives the old grant back and then makes the call for the first time.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    stream.lose_next_grant = True
    stream.fault_before_release = True
    world.gate.set()
    with pytest.raises(RuntimeError, match="the grant never came back"):
        await gateway.environment("guess", GUESS)
    assert world.landed == []
    record = gateway._recovery
    assert isinstance(record, _LeaseHeld)
    assert record.result is None
    assert stream.held == record.call.call_id
    assert await refused(gateway.pull({})) == "outstanding_response"

    # The same call gives that grant back, and asks for the one it is making now after it.
    assert (await gateway.environment("guess", GUESS)).content[0].text == "landed"
    assert world.landed == ["guess"]
    assert stream.held is None
    assert [lease.call_id for lease in stream.leases] == [
        record.call.call_id,
        stream.releases[-1].call_id,
    ]


async def test_every_retry_carries_the_identity_the_first_attempt_was_sent_under(
    episode: ServedEpisode,
) -> None:
    """A request, a filing, and an attestation each survive the fault that lost their answer.

    The stream reaches the same durable call from the same identity, so a retry that minted a
    new one would ask a question nobody had asked before and leave the first one unanswered
    forever. Nothing a mismatched call does may change which identity that is.
    """
    payload = offered(
        Payload(message_id="0" * 31 + "c", attempt_id=ATTEMPT, body="receipt 0"), ATTEMPT
    )
    lost = RuntimeError("the answer never arrived")
    stream = ScriptedStream(lost, TASK_OFFER, payload)
    gateway = make_gateway(episode, stream)

    with pytest.raises(RuntimeError, match="never arrived"):
        await gateway.pull({})
    asked = stream.requests[-1]
    assert await refused(gateway.environment("guess", GUESS)) == "outstanding_response"
    assert (await gateway.pull({})) == TASK_OFFER.visible_text
    assert stream.requests[-1] is asked

    stream.fault_before_offer = True
    with pytest.raises(RuntimeError, match="the response channel failed"):
        await gateway.terminal(FILING)
    filed = stream.requests[-1]
    assert await refused(gateway.pull({})) == "outstanding_response"
    assert stream.requests[-1] is filed

    # And the attestation. The seal has no more offers to give, so the filing's retry raises
    # from the script; what matters is that the presentation kept its own identity.
    gateway._recovery = _Idle()
    stream.fault_before_commit = True
    with pytest.raises(RuntimeError, match="the presentation request never arrived"):
        await gateway.pull({})
    uncertain = gateway._recovery
    assert isinstance(uncertain, _PresentationUncertain)
    assert await refused(gateway.terminal(FILING)) == "outstanding_response"
    assert (await gateway.pull({})) == payload.visible_text
    assert stream.commits[-1] is uncertain.commit
    attested = [commit for commit in stream.commits if commit.message_id == payload.message_id]
    assert attested == [uncertain.commit]


# Where a call was cut, and what it left.


MESSAGES = {
    "task": TASK_OFFER,
    "wait": offered(Wait(message_id="0" * 31 + "d", retry_after_ms=1000)),
    "payload": offered(
        Payload(message_id="0" * 31 + "e", attempt_id=ATTEMPT, body="receipt 0"), ATTEMPT
    ),
    "done": DONE_OFFER,
    "seal_ack": ACK_OFFER,
    "seal_reject": offered(
        SealReject(message_id="0" * 31 + "f", attempt_id=ATTEMPT, body="missing word"), ATTEMPT
    ),
}


@pytest.mark.parametrize("kind", sorted(MESSAGES))
async def test_a_call_cut_between_the_offer_and_the_presentation_still_delivers(
    episode: ServedEpisode, kind: str
) -> None:
    """The stream offered it, so it is presented and then owed, whatever became of the caller.

    An offer is a reservation the stream will not give to anything else, so abandoning it there
    would strand the generation on a message nobody may collect. The operation finishes instead:
    the attestation goes, the bytes become what the calling call is owed, and this gateway is
    held for that operation until both are done rather than until its waiter went away.
    """
    message = MESSAGES[kind]
    filing = kind in ("seal_ack", "seal_reject")
    stream = ScriptedStream(*(([TASK_OFFER] if filing else []) + [message]))
    gateway = make_gateway(episode, stream)
    if filing:
        await gateway.pull({})
    presentations = stream.calls.count("present")

    stream.commit_gate = asyncio.Event()
    call = gateway.terminal(FILING) if filing else gateway.pull({})
    handler = asyncio.create_task(call)
    await asyncio.sleep(0.01)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler

    # The attestation is out and nobody knows whether it landed, so nothing else may go.
    assert isinstance(gateway._recovery, _PresentationUncertain)
    assert await refused(gateway.environment("guess", GUESS)) == "overlapping_call"

    stream.commit_gate.set()
    await asyncio.sleep(0.05)
    assert stream.calls.count("present") == presentations + 1
    record = gateway._recovery
    assert isinstance(record, _ResultOwed)
    assert record.result == message.visible_text
    assert await refused(gateway.environment("guess", GUESS)) == "outstanding_response"

    again = await (gateway.terminal(FILING) if filing else gateway.pull({}))
    assert again == message.visible_text
    assert stream.calls.count("present") == presentations + 1
    assert isinstance(gateway._recovery, _Idle)


async def test_a_call_cut_before_its_request_was_built_leaves_the_generation_alone(
    episode: ServedEpisode,
) -> None:
    """Cut early enough and there is nothing to recover, because nothing was asked for yet.

    The operation still runs to the end, because this gateway does not know at the cut what the
    stream has been asked, and it still holds the gateway while it does. What comes back is the
    message the stream offered, owed to the call that asked for it.
    """
    stream = ScriptedStream(TASK_OFFER)
    stream.state_gate = asyncio.Event()
    gateway = make_gateway(episode, stream)

    handler = asyncio.create_task(gateway.pull({}))
    await asyncio.sleep(0.01)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    assert stream.calls == []
    assert isinstance(gateway._recovery, _Idle)
    assert await refused(gateway.pull({})) == "overlapping_call"

    stream.state_gate.set()
    await asyncio.sleep(0.05)
    assert isinstance(gateway._recovery, _ResultOwed)
    assert (await gateway.pull({})) == TASK_OFFER.visible_text
    assert stream.calls == ["pull", "present"]


async def test_a_call_that_fails_leaves_this_gateway_serving(
    episode: ServedEpisode,
) -> None:
    """One call failing is one call failing. It is not the transport giving up.

    A fault carries no protocol code, so nothing here turns it into a claim about the durable
    generation, which is neither failed nor closed because a local call went wrong. The gateway
    is given back, the stream is given back, and the next call is served.
    """
    world = BlockingEpisode()
    world.fault = RuntimeError("the world raised")
    world.gate.set()
    stream = ScriptedStream(TASK_OFFER, DONE_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    with pytest.raises(RuntimeError, match="the world raised"):
        await gateway.environment("guess", GUESS)
    assert gateway._operation is None
    assert stream.held is None
    assert isinstance(gateway._recovery, _Idle)
    assert (await gateway.pull({})) == DONE_OFFER.visible_text


async def test_a_call_the_episode_makes_back_into_this_gateway_is_refused(
    episode: ServedEpisode,
) -> None:
    """A call that reaches back in gets an answer rather than a gateway waiting on itself."""

    class ReentrantEpisode:
        def __init__(self) -> None:
            self.answer: Optional[str] = None
            self.gateway: Optional[StreamGateway] = None

        async def call(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
            assert self.gateway is not None
            self.answer = await refused(self.gateway.pull({}))
            return Observation()

    world = ReentrantEpisode()
    spec = episode.describe()
    stream = ScriptedStream(TASK_OFFER)
    gateway = StreamGateway(
        stream,  # type: ignore[arg-type]
        world,  # type: ignore[arg-type]
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
    )
    world.gateway = gateway
    await gateway.pull({})
    played = await asyncio.wait_for(gateway.environment("guess", GUESS), timeout=5)
    assert played.content[0].text == "landed"
    assert world.answer == "overlapping_call"


async def test_two_calls_the_model_meant_are_two_calls(episode: ServedEpisode) -> None:
    """Nothing here caches a result by what the call looked like, so a repeat is a repeat.

    Two identical guesses are two guesses the model chose to play. Answering the second from the
    first one's result would drop a real step of the episode, which is worse than the duplicate
    it would be preventing: the duplicate is in the trajectory and the dropped step is nowhere.
    """
    stream = ScriptedStream(TASK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})
    first = await gateway.environment("guess", GUESS)
    second = await gateway.environment("guess", GUESS)
    assert json.loads(first.content[0].text)["valid"] is True
    assert json.loads(second.content[0].text)["valid"] is True
    assert [step.arguments for step in episode._trajectory] == [
        {"word": "crane"},
        {"word": "crane"},
    ]
    assert stream.calls.count("environment") == 2


async def test_recovery_is_one_record_however_many_calls_are_refused(
    episode: ServedEpisode,
) -> None:
    """There is one thing owed at a time, and being asked a hundred other things does not add.

    Nothing accumulates per refused call: no result kept under its name, no request allocated
    for it, and nothing dispatched for it later. The one record stays the one record, and it
    stays the same one.

    The gateway does keep a tally of what it refused, which is the only place a refusal is
    counted. What the stream is spared is written down here as the three facts it is: a hundred
    refusals dispatch no request and no Update, leave the generation no row saying it was
    asked, and move nothing it would carry in its history.

    They are not silence on the wire, and the boundary is worth stating exactly. Each refusal
    is decided against a fresh reading of where the generation is, so a hundred of them are a
    hundred read-only Queries, and that count is asserted here rather than left out of a claim
    the numbers would not support.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    handler = asyncio.create_task(gateway.environment("guess", GUESS))
    await asyncio.sleep(0.01)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    world.gate.set()
    await asyncio.sleep(0.05)
    record = gateway._recovery
    assert isinstance(record, _ResultOwed)

    refusals = gateway.refusals
    written = stream_writes(stream)
    reads = stream.state_reads

    for index in range(50):
        word = f"w{index:04d}"
        assert (
            await refused(gateway.environment("guess", {**GUESS, "arguments": {"word": word}}))
            == "outstanding_response"
        )
        assert await refused(gateway.terminal(FILING)) == "outstanding_response"
    assert gateway._recovery is record
    assert world.landed == ["guess"]
    assert stream.calls.count("environment") == 1
    assert gateway.refusals == refusals + 100
    # No dispatch, no row, and nothing the generation would carry in its history.
    assert stream_writes(stream) == written
    # And what a hundred locally decided refusals do cost the stream: one read each.
    assert stream.state_reads == reads + 100

    assert (await gateway.environment("guess", GUESS)).content[0].text == "landed"
    assert world.landed == ["guess"]


# Stopping.


async def test_closing_this_gateway_is_idempotent_and_closes_the_episode_last(
    episode: ServedEpisode,
) -> None:
    """Closing twice closes once, and a caller that goes away does not take the close with it."""
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)

    await gateway.aclose()
    assert world.closed_after == []
    # Released rather than ended. What became of the attempt this world was opened for is the
    # generation's to say, and a transport stopping is not it saying so.
    assert world.finalized is False
    shutdown = gateway._shutdown
    await gateway.aclose()
    assert gateway._shutdown is shutdown

    # And nothing is served afterwards. It is not a protocol refusal: this transport stopping
    # says nothing about the generation, which may well still be open.
    with pytest.raises(GatewayClosed):
        await gateway.pull({})
    with pytest.raises(GatewayClosed):
        await gateway.environment("guess", GUESS)


async def test_closing_waits_for_the_call_it_found_running(
    episode: ServedEpisode,
) -> None:
    """A call that is changing a world is not something a clean stop may walk away from.

    Admission stops at once, so nothing new is taken. The one call that was already accepted is
    left to settle, the stream is given its generation back, and the episode is closed after
    that and not before: closing it first would take the world out from under a call that is
    still in it.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    playing = asyncio.create_task(gateway.environment("guess", GUESS))
    await asyncio.sleep(0.01)
    closing = asyncio.create_task(gateway.aclose())
    await asyncio.sleep(0.01)
    # Admission has stopped, and the call that was running has not.
    with pytest.raises(GatewayClosed):
        await gateway.pull({})
    assert not closing.done()
    assert world.closed_after is None

    world.gate.set()
    assert (await playing).content[0].text == "landed"
    await closing
    assert world.closed_after == ["guess"]
    assert stream.held is None


@pytest.mark.parametrize("gated", ["seal", "presentation"])
async def test_closing_waits_for_a_stream_operation_too(
    episode: ServedEpisode, gated: str
) -> None:
    """A seal the stream accepted, and an attestation already sent, are the same kind of thing.

    Both are running whether or not this transport is still listening, and both end in a result
    that has to be recoverable afterwards. So the stop waits for the one it found rather than
    leaving the generation with a result nothing here could ever name again.
    """
    stream = ScriptedStream(TASK_OFFER, ACK_OFFER)
    gateway = make_gateway(episode, stream)
    await gateway.pull({})
    if gated == "seal":
        stream.gate = asyncio.Event()
        release = stream.gate
    else:
        stream.commit_gate = asyncio.Event()
        release = stream.commit_gate

    sealing = asyncio.create_task(gateway.terminal(FILING))
    await asyncio.sleep(0.01)
    closing = asyncio.create_task(gateway.aclose())
    await asyncio.sleep(0.01)
    with pytest.raises(GatewayClosed):
        await gateway.pull({})
    assert not closing.done()

    release.set()
    assert json.loads(await sealing)["kind"] == "seal_ack"
    await closing
    assert stream.attempts[ATTEMPT] == "ack_presented"


async def test_a_close_whose_caller_goes_away_still_closes(episode: ServedEpisode) -> None:
    """Stopping belongs to the gateway, not to whoever asked for it first."""
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    playing = asyncio.create_task(gateway.environment("guess", GUESS))
    await asyncio.sleep(0.01)
    first = asyncio.create_task(gateway.aclose())
    await asyncio.sleep(0.01)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    # A second caller joins the shutdown that is already under way rather than starting another.
    second = asyncio.create_task(gateway.aclose())
    await asyncio.sleep(0.01)
    assert not second.done()
    world.gate.set()
    await playing
    await second
    assert world.closed_after == ["guess"]


@pytest.mark.parametrize("owed", ["done", "seal_ack", "payload", "observation"])
async def test_closing_keeps_what_is_still_owed(episode: ServedEpisode, owed: str) -> None:
    """A clean stop is not a reason to throw away the only copy of what has not been read.

    What is owed is a message the stream has counted as read, or an observation of a world that
    has already changed. Presenting it, collecting it, or discarding it on the way out would
    each be this transport deciding something on nobody's behalf, so the record is left exactly
    where the call that is owed it will find it.
    """
    world = BlockingEpisode()
    ordinary = owed == "observation"
    offers = [TASK_OFFER] if ordinary else [TASK_OFFER, MESSAGES[owed]]
    stream = ScriptedStream(*offers)
    gateway = blocking_gateway(episode, stream, world) if ordinary else make_gateway(episode, stream)
    await gateway.pull({})

    if ordinary:
        call = gateway.environment("guess", GUESS)
        expected: Any = "landed"
    else:
        gate = stream.gate = asyncio.Event()
        call = gateway.terminal(FILING) if owed == "seal_ack" else gateway.pull({})
        expected = MESSAGES[owed].visible_text
    handler = asyncio.create_task(call)
    await asyncio.sleep(0.01)
    handler.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handler
    (world.gate if ordinary else gate).set()
    await asyncio.sleep(0.05)
    record = gateway._recovery
    assert isinstance(record, _ResultOwed)

    presented = stream.calls.count("present")
    await gateway.aclose()
    # The record is where it was, and closing neither presented nor collected anything.
    assert gateway._recovery is record
    assert stream.calls.count("present") == presented
    if ordinary:
        assert record.result.content[0].text == expected
    else:
        assert record.result == expected


@pytest.mark.parametrize("landed", ["the call landed", "the world refused it"])
async def test_closing_gives_back_a_grant_this_gateway_is_still_holding(
    episode: ServedEpisode, landed: str
) -> None:
    """A stop that walks away from a live grant leaves the generation held by nothing.

    The release is the one thing a stop can finish on its own. It names the exact call this
    gateway wrote down before asking for the grant, it is the same Update sent again rather than
    a second decision, and it dispatches nothing into the world. So it goes before this
    transport stops serving and before the world is closed. What the call landed with is a
    different thing, and it is kept: giving the generation back is not collecting an observation
    on behalf of a caller that never read it.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    if landed == "the world refused it":
        world.fault = RuntimeError("the world refused the call")
    stream.fault_before_release = True
    world.gate.set()
    with pytest.raises(RuntimeError):
        await gateway.environment("guess", GUESS)
    held = gateway._recovery
    assert isinstance(held, _LeaseHeld)
    assert stream.held == held.call.call_id
    assert len(stream.releases) == 1

    await gateway.aclose()
    # The grant went back under the identity it was made under, and it went back before the
    # world it was granted for was closed.
    assert stream.held is None
    assert len(stream.releases) == 2
    assert stream.releases[-1].call_id == held.call.call_id
    assert world.closed_after == world.landed
    if landed == "the call landed":
        owed = gateway._recovery
        assert isinstance(owed, _ResultOwed)
        assert owed.owner == held.owner
        assert owed.result.content[0].text == "landed"
    else:
        assert isinstance(gateway._recovery, _Idle)


async def test_a_stop_that_cannot_give_the_grant_back_says_so(episode: ServedEpisode) -> None:
    """A stop reports what it could not do rather than recording a cleanup that never happened.

    Marking itself closed over a release the stream never answered would leave the generation
    held with nothing left that knows the call's name. So the fault comes out of the stop, the
    record stays where it was, and what becomes of that grant is for whatever claims the
    generation next. Nothing is served here either way: admission stopped when the stop began.
    """
    world = BlockingEpisode()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    stream.fault_before_release = True
    world.gate.set()
    with pytest.raises(RuntimeError, match="the release never reached"):
        await gateway.environment("guess", GUESS)
    held = gateway._recovery
    assert isinstance(held, _LeaseHeld)

    stream.fault_before_release = True
    with pytest.raises(RuntimeError, match="the release never reached"):
        await gateway.aclose()
    assert gateway._recovery is held
    assert stream.held == held.call.call_id
    assert world.closed_after is None
    with pytest.raises(GatewayClosed):
        await gateway.pull({})


async def test_a_call_racing_the_close_is_taken_whole_or_refused_whole(
    episode: ServedEpisode,
) -> None:
    """There is no third answer: a call is accepted before the stop or told there is nothing.

    The claim and the stop are both taken without awaiting, so they cannot interleave. Nothing
    can be left half admitted in a queue that is about to be abandoned, because there is no
    queue for it to be left in.
    """
    world = BlockingEpisode()
    world.gate.set()
    stream = ScriptedStream(TASK_OFFER)
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})

    async def call() -> str:
        try:
            await gateway.environment("guess", GUESS)
        except GatewayClosed:
            return "refused"
        return "taken"

    racing = [asyncio.create_task(call()) for _ in range(8)]
    await gateway.aclose()
    answers = await asyncio.gather(*racing)
    assert set(answers) <= {"taken", "refused"}
    assert answers.count("taken") == len(world.landed)
    assert stream.held is None


async def test_the_queue_closing_is_not_this_gateway_closing(
    episode: ServedEpisode,
) -> None:
    """Closing the queue to insertion is what makes Done reachable, and it stops nothing here."""
    stream = ScriptedStream(TASK_OFFER, DONE_OFFER)
    gateway = make_gateway(episode, stream)
    await stream.close_queue()
    assert stream.queue_closed is True

    assert (await gateway.pull({})) == TASK_OFFER.visible_text
    assert (await gateway.environment("guess", GUESS)).content[0].text is not None
    assert (await gateway.pull({})) == DONE_OFFER.visible_text


async def test_a_gateway_built_directly_closes_the_episode_it_was_given(
    episode: ServedEpisode,
) -> None:
    """A gateway constructed without a runner around it still has a way to be stopped."""
    world = BlockingEpisode()
    world.gate.set()
    stream = ScriptedStream(TASK_OFFER)
    running = asyncio.all_tasks()
    gateway = blocking_gateway(episode, stream, world)
    await gateway.pull({})
    assert (await gateway.environment("guess", GUESS)).content[0].text == "landed"
    await gateway.aclose()
    assert world.closed_after == ["guess"]
    assert gateway._operation is None
    # Nothing of this gateway's is still running: it starts no task of its own, so serving two
    # calls and stopping leaves the loop with exactly what it had before.
    assert asyncio.all_tasks() == running


# The whole arc, over a real stdio server against a real durable service.


@pytest.mark.network
async def test_the_whole_arc_over_stdio(tmp_path) -> None:
    """Task, terminal, SealAck, Payload, Done: what a harness spawning the CLI would see."""
    # `running` separates a service that could not start, which is a skip, from a failure the
    # test itself found, which is a failure. Without it an offline machine and a broken gateway
    # would report the same thing.
    running = False
    try:
        async with durable_client() as client:
            running = True
            address = client.service_client.config.target_host
            # No Worker here. The served process registers the Activities its own environment
            # brings, and a second Worker on that queue with different implementations would be
            # answering for an environment nobody asked it about.
            await _drive_stdio(address, tmp_path)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")


async def _drive_stdio(address: str, tmp_path: Any) -> None:
    """Spawn ``shogym serve`` and run one episode through it."""
    environment = dict(os.environ)
    # The served process joins the service this test already started rather than starting a
    # second one on the same database.
    environment["SHOGYM_TEMPORAL_ADDRESS"] = address
    transport = StdioTransport(
        command=sys.executable,
        args=[
            "-m",
            "shogym.cli",
            "serve",
            TEST_ENV,
            "--task",
            "0",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        env=environment,
        cwd=str(tmp_path),
    )
    async with Client(transport) as client:
        names = {tool.name for tool in await client.list_tools()}
        assert names == {PULL_TOOL, "guess", "terminate"}

        task = await _record(client, PULL_TOOL, {})
        assert task["kind"] == "task"
        attempt = task["attempt_id"]
        assert "Wordle" in task["body"]

        guess = {"attempt_id": attempt, "arguments": {"word": "crane"}}
        played = await client.call_tool("guess", guess)
        assert json.loads(played.content[0].text)["valid"] is True

        ack = await _record(client, "terminate", {"attempt_id": attempt, "arguments": {}})
        assert ack["kind"] == "seal_ack"
        assert ack["attempt_id"] == attempt
        assert len(ack["submission_digest"]) == 64
        # Wordle brings its own terminal, so the digest was taken under the version that env
        # declares rather than under this gateway's stand-in.
        assert ack["canonicalization_version"] == "shogym.wordle.1"

        payload = await _record(client, PULL_TOOL, {})
        assert payload["kind"] == "payload"
        assert payload["attempt_id"] == attempt
        # The default is honest: the agent is told the attempt, the score the stream committed,
        # and the numbers the env published beside it. One guess, and it was not the word.
        assert payload["body"] == f"attempt {attempt}\nscore 0\nguesses_used 1"

        done = await _record(client, PULL_TOOL, {})
        assert done["kind"] == "done"

        closed = await client.call_tool(PULL_TOOL, {}, raise_on_error=False)
        assert closed.is_error
        assert json.loads(closed.content[0].text)["code"] == "closed_stream"

    # The run directory holds what a later owner would resume from, and every presentation in
    # that arc referenced a transcript blob the stream read before it committed anything.
    run = open_run_directory(Path(tmp_path) / "run")
    assert run.manifest.workflow_id.startswith("stream/")
    assert [path for path in run.blobs.root.rglob("*") if path.is_file()]
    # What the body was allowed to say is in there as bytes rather than as a digest naming
    # bytes nobody kept, so the directory alone says what this run's payloads meant.
    assert run.blobs.read(policy_digest(HONEST_V1)) == policy_preimage(HONEST_V1)


async def _record(client: Client, tool: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Call one tool and return the single protocol record it answered with."""
    result = await client.call_tool(tool, arguments)
    assert len(result.content) == 1
    return json.loads(result.content[0].text)


@pytest.mark.network
async def test_a_refused_presentation_is_repaired_against_a_real_stream(
    episode: ServedEpisode,
) -> None:
    """The repair, against a stream that really does answer an attestation only once.

    A stream verifies an attestation against objects it does not hold, and refuses one it cannot
    verify without applying anything. Putting the object back is supposed to let the message it
    was holding through. Whether it does is not a question about this gateway's memory: the
    stream reaches an attestation by an identity its durable call ID is derived from, a refusal
    completes that call, and the same attestation sent afterwards collects the refusal instead
    of being verified again. So a repair only reaches verification under an attestation the
    stream has not answered yet, and only a real stream can show that.

    The object store arrives with a later element, so what stands in for a reference the stream
    cannot verify is a pre-event state hash it cannot match. It is the same refusal in the same
    place: decisive, and nothing applied.
    """
    running = False
    try:
        async with durable_client() as client:
            running = True
            async with stream_worker(client):
                await _repair_a_refusal(client, episode)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")


async def _repair_a_refusal(client: Client, episode: ServedEpisode) -> None:
    """Refuse the first attestation of the first Task, repair, and retry the exact call."""
    gateway = await open_gateway(client, episode)
    built = gateway._attestation
    unrepaired = [True]

    async def attestation(message: OfferedMessage) -> PresentationCommit:
        commit = await built(message)
        if unrepaired:
            unrepaired.pop()
            return replace(commit, stream_state_before_sha256="0" * 64)
        return commit

    setattr(gateway, "_attestation", attestation)

    assert await refused(gateway.pull({})) == "invalid_message"
    refused_state = await gateway._stream.stream_state()
    assert refused_state.presentation_count == 0
    assert refused_state.pending_kind == "task"

    # The exact call again, with what the refusal was about put right.
    task = json.loads(await gateway.pull({}))
    assert task["kind"] == "task"
    delivered = await gateway._stream.stream_state()
    assert delivered.presentation_count == 1
    assert delivered.pending_message_id is None
    await gateway.aclose()


@pytest.mark.network
async def test_a_cancelled_final_pull_collects_its_done_from_a_real_stream(
    episode: ServedEpisode,
) -> None:
    """The last thing a generation owes, collected from a stream that has finished.

    The double above can be told that a done generation answers nothing, and it is, but what
    makes that the right thing to tell it is only visible here: a stream ends at Done, so the
    write path a kept result is normally handed over on is gone by the time the retry asks. A
    gateway that asked anyway would answer the harness's last pull with a protocol error where
    the record telling it to stop should have been, and every prior outcome durable behind it.
    """
    running = False
    try:
        async with durable_client() as client:
            running = True
            terminal = environment_terminal(episode)
            async with stream_worker(client, activities=terminal.activities):
                await _collect_a_cancelled_done(client, episode, terminal)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")


async def _collect_a_cancelled_done(
    client: Client, episode: ServedEpisode, terminal: Any
) -> None:
    """Run the arc to Done, drop the caller after it lands, and ask for it again."""
    gateway = await open_gateway(client, episode, environment=terminal)
    # Done is what a generation says when it has nothing left and nothing more can arrive, so
    # the queue is closed here rather than assumed closed.
    await gateway._stream.close_queue()
    task = json.loads(await gateway.pull({}))
    attempt = task["attempt_id"]
    await gateway.environment("guess", {"attempt_id": attempt, "arguments": {"word": "crane"}})
    assert json.loads(await gateway.terminal({"attempt_id": attempt, "arguments": {}}))[
        "kind"
    ] == "seal_ack"
    assert json.loads(await gateway.pull({}))["kind"] == "payload"

    # The presentation is what the stream counts, so the caller is dropped after that has
    # landed and before the bytes it landed with have been returned to anyone.
    presented = asyncio.Event()
    collected = asyncio.Event()
    present = gateway._present

    async def held(record: Any) -> str:
        text = await present(record)
        presented.set()
        await collected.wait()
        return text

    setattr(gateway, "_present", held)
    final = asyncio.ensure_future(gateway.pull({}))
    await presented.wait()
    final.cancel()
    with pytest.raises(asyncio.CancelledError):
        await final
    collected.set()
    await asyncio.sleep(0.1)
    setattr(gateway, "_present", present)

    assert isinstance(gateway._recovery, _ResultOwed)
    closed = await gateway._stream.stream_state()
    assert closed.generation_state == "done"
    presentations = closed.presentation_count

    # The stream is over, and this is what it is still owed to. The retry hands the bytes over
    # and presents nothing, because there is nothing left to present them to.
    assert json.loads(await gateway.pull({}))["kind"] == "done"
    after = await gateway._stream.stream_state()
    assert after.presentation_count == presentations
    # Collected, and now the closed generation is all there is to say.
    assert await refused(gateway.pull({})) == "closed_stream"
    await gateway.aclose()
