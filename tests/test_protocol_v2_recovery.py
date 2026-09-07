"""The public channel of a repair: one refused pull, retained, sent again, collected by the agent.

A delivery whose committed bytes have gone is refused with a bare token and nothing else. What
this file is about is the shape of what happens next, at the transport: the refused pull is
retained where it can be sent again, the controller repairs the bytes and asks this gateway to
send that exact request under the new epoch, the recovery stops at the offer because a
presentation is a commitment, and the agent's own next pull collects those bytes under an
attestation built for that call.

The stream here is a double, and it keeps the two rules the whole design rests on: a message is
reserved for the request that was given it and is reachable through no other, and the same request
sent again reaches the answer it already got rather than a second one. Everything else about a
durable stream is the kernel's and is tested where it lives.
"""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Any, Dict, List, Optional

import pytest

pytest.importorskip("temporalio")

from shogym.serve.protocol_v2 import (  # noqa: E402
    Payload,
    PresentationAck,
    PresentationCommit,
    PullRequest,
    Wait,
    pull_request_identity,
    visible_bytes,
)
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    StreamGateway,
    stream_start,
    terminal_manifest,
)
from shogym.serve.protocol_v2.gateway import _Idle, _PullRecovered  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    OfferedMessage,
    StreamProtocolError,
)
from shogym.task import TaskSpec, ToolManifest  # noqa: E402

ATTEMPT = "b" * 32
PAYLOAD_ID = "c" * 32
CURSOR = "1" * 32
BODY = "the committed cell of a source this run captured"


def a_spec() -> TaskSpec:
    """One environment with an ordinary tool and a terminal, which is all a gateway needs."""
    return TaskSpec(
        env_name="a_world",
        instructions="file the report",
        tools=[
            ToolManifest(
                name="look",
                description="look at the table",
                input_schema={"type": "object", "properties": {}},
            ),
            ToolManifest(
                name="submit_filing",
                description="file the report",
                input_schema={
                    "type": "object",
                    "properties": {"filing": {"type": "string"}},
                    "required": ["filing"],
                },
                terminal_kind="score",
            ),
        ],
    )


class Projection:
    """What the gateway reads out of the stream at the top of every call."""

    def __init__(self, stream: "Reserving") -> None:
        self.cursor = stream.cursor
        self.generation_state = "open"
        self.attempts: Dict[str, str] = {ATTEMPT: "sealed"}
        self.environment_calls: Dict[str, int] = {}
        self.stream_state_sha256 = format(stream.reads, "064x")
        self.pending_message_id = None if stream.pending is None else stream.pending.message_id
        self.pending_kind = None if stream.pending is None else stream.pending.kind


class Reserving:
    """A stream that reserves what it offers for the request that asked, and nothing else.

    The reservation is the point. A durable generation derives the identifier of the Update a
    request reaches from the request itself, so the same request sent again reaches the answer it
    already got: that is what makes a lost reply recoverable and what a recovery under a stepped
    epoch relies on. A refusal reserves nothing, which is the other half of the same rule.
    """

    def __init__(self, *offers: Any) -> None:
        self.offers: List[Any] = list(offers)
        self.reserved: Dict[str, OfferedMessage] = {}
        self.requests: List[PullRequest] = []
        self.commits: List[PresentationCommit] = []
        self.cursor = CURSOR
        self.pending: Optional[OfferedMessage] = None
        self.reads = 0
        self.lose_next_result = False

    async def pull(self, request: PullRequest) -> OfferedMessage:
        self.requests.append(request)
        held = self.reserved.get(request.request_id)
        if held is not None:
            return held
        answer = self.offers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        self.reserved[request.request_id] = answer
        self.pending = answer
        if self.lose_next_result:
            self.lose_next_result = False
            raise RuntimeError("the reply never arrived")
        return answer

    async def stream_state(self) -> Projection:
        self.reads += 1
        return Projection(self)

    async def confirm_state(self) -> Projection:
        return await self.stream_state()

    async def commit_presentation(self, commit: PresentationCommit) -> PresentationAck:
        self.commits.append(commit)
        self.pending = None
        self.cursor = commit.attestation_id
        return PresentationAck(
            attestation_id=commit.attestation_id,
            cursor=self.cursor,
            stream_state_sha256=format(self.reads, "064x"),
        )


def offered(record: Any) -> OfferedMessage:
    return OfferedMessage(
        message_id=record.message_id,
        kind=record.kind,
        visible_text=visible_bytes(record).decode("utf-8"),
        attempt_id=getattr(record, "attempt_id", None),
    )


PAYLOAD = offered(Payload(message_id=PAYLOAD_ID, attempt_id=ATTEMPT, body=BODY))
WAITING = offered(Wait(message_id="d" * 32, retry_after_ms=1000))


def a_gateway(stream: Reserving) -> StreamGateway:
    """One transport over that stream, with no world anything here has to open."""
    spec = a_spec()
    return StreamGateway(
        stream,  # type: ignore[arg-type]
        None,  # type: ignore[arg-type]
        spec,
        terminal_manifest(spec),
        initial_cursor=CURSOR,
        generation=stream_start(
            spec, terminal_manifest(spec), claim_hash="c" * 64, evaluation_only=True
        ),
    )


async def refused(awaitable: Any) -> Dict[str, Any]:
    """Return the protocol error record a refused call carries, as its whole text."""
    try:
        await awaitable
    except Exception as error:  # noqa: BLE001 - the record is the assertion
        return json.loads(str(error))
    raise AssertionError("the call was accepted")


async def test_a_pull_refused_over_missing_evidence_is_kept_where_it_can_be_sent_again() -> None:
    """One refusal leaves a record behind, and it holds the request rather than a digest of it.

    The canonical identity is what names the operation, and the request is what a recovery has to
    send: a digest cannot be sent to anything. Both are kept, with the call they were made under.
    """
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD)
    gateway = a_gateway(stream)

    record = await refused(gateway.pull({}))
    assert record == {
        "code": "evidence_unavailable",
        "kind": "protocol_error",
        "protocol_version": 2,
    }
    [identity] = gateway.refused_pulls
    assert identity == pull_request_identity(stream.requests[0])
    # And it gates nothing: the refusal reserved no message, so this transport is idle.
    assert isinstance(gateway._recovery, _Idle)


async def test_every_other_refusal_still_clears_this_transport_to_idle() -> None:
    """A refusal that settles the request it answered leaves nothing to send again."""
    stream = Reserving(StreamProtocolError("invalid_cursor"))
    gateway = a_gateway(stream)

    assert (await refused(gateway.pull({})))["code"] == "invalid_cursor"
    assert gateway.refused_pulls == ()
    assert isinstance(gateway._recovery, _Idle)


async def test_an_ordinary_next_pull_mints_a_fresh_request_as_it_always_did() -> None:
    """The record reserves nothing and refuses nothing, so the next call is served as usual."""
    stream = Reserving(StreamProtocolError("evidence_unavailable"), WAITING)
    gateway = a_gateway(stream)

    await refused(gateway.pull({}))
    answer = json.loads(await gateway.pull({}))

    assert answer["kind"] == "wait"
    assert stream.requests[1].request_id != stream.requests[0].request_id
    # The refused pull is still retained: a later call answered says nothing about the operation
    # that was refused, and only that operation being answered recovers it.
    assert gateway.refused_pulls == (pull_request_identity(stream.requests[0]),)


async def test_a_recovery_sends_that_exact_request_again_and_stops_at_the_offer() -> None:
    """A presentation is a commitment, and the recovery makes none.

    One committed outside a call the agent made would be committed and never observed, so what
    the recovery does is get the generation to reserve the bytes again and then leave them
    reserved.
    """
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD)
    gateway = a_gateway(stream)
    await refused(gateway.pull({}))
    [identity] = gateway.refused_pulls

    assert await gateway.recover_refused_pull(identity) is True

    assert stream.requests[1] == stream.requests[0]
    assert stream.commits == []
    assert stream.pending == PAYLOAD
    # The retained record and the recovered one belong to the recovery now.
    assert gateway.refused_pulls == ()
    assert isinstance(gateway._recovery, _PullRecovered)


async def test_the_agents_own_next_pull_delivers_the_recovered_bytes_under_its_own_attestation(
) -> None:
    """The pull adopts what the recovery left rather than being refused against it.

    It sends that same request once more, receives the message the generation reserved for it,
    and presents it under an attestation built for this call: the one before it was built by the
    call that was refused and describes a stream that has moved.
    """
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD)
    gateway = a_gateway(stream)
    await refused(gateway.pull({}))
    [identity] = gateway.refused_pulls
    await gateway.recover_refused_pull(identity)

    delivered = json.loads(await gateway.pull({}))

    assert delivered == json.loads(PAYLOAD.visible_text)
    assert delivered["body"] == BODY
    # One request, sent three times: by the call that was refused, by the recovery, and by the
    # pull that collected it.
    assert [request.request_id for request in stream.requests] == [
        stream.requests[0].request_id
    ] * 3
    # One presentation, of those bytes, built by the call that delivered them.
    [commit] = stream.commits
    assert commit.message_id == PAYLOAD_ID
    assert commit.visible_bytes_sha256 == sha256(PAYLOAD.visible_text.encode()).hexdigest()
    # And this transport is idle again: the delivery it was owed has been made.
    assert isinstance(gateway._recovery, _Idle)


async def test_a_recovery_whose_reply_never_arrives_is_finished_by_the_next_pull() -> None:
    """A lost reply needs nothing further, being what an uncertain record already covers.

    The record is written before the request goes, so a recovery that never heard back leaves the
    same uncertainty a lost pull leaves. The agent's next pull adopts it, sends the same request
    under the same epoch, reaches the same Update and delivers those bytes once.
    """
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD)
    gateway = a_gateway(stream)
    await refused(gateway.pull({}))
    [identity] = gateway.refused_pulls

    stream.lose_next_result = True
    with pytest.raises(RuntimeError):
        await gateway.recover_refused_pull(identity)
    assert isinstance(gateway._recovery, _PullRecovered)

    delivered = json.loads(await gateway.pull({}))

    assert delivered["body"] == BODY
    assert [request.request_id for request in stream.requests] == [
        stream.requests[0].request_id
    ] * 3
    # Once. The generation reserved one message and one presentation was committed for it.
    assert len(stream.commits) == 1


async def test_a_recovery_the_generation_refuses_again_leaves_the_pull_retained() -> None:
    """Repairing bytes that are still not there is not a recovery, and says so.

    The generation reserved nothing, so the record goes back where it was: a controller with
    another verified copy repairs and asks again, and nothing about the transport has moved.
    """
    stream = Reserving(
        StreamProtocolError("evidence_unavailable"),
        StreamProtocolError("evidence_unavailable"),
        PAYLOAD,
    )
    gateway = a_gateway(stream)
    await refused(gateway.pull({}))
    [identity] = gateway.refused_pulls

    assert await gateway.recover_refused_pull(identity) is False
    assert gateway.refused_pulls == (identity,)
    assert isinstance(gateway._recovery, _Idle)
    assert stream.commits == []

    # And the next attempt, over bytes that are back, recovers it.
    assert await gateway.recover_refused_pull(identity) is True
    assert json.loads(await gateway.pull({}))["body"] == BODY


async def test_a_recovery_names_a_pull_that_was_refused_rather_than_a_fresh_one() -> None:
    """The identity is the whole of what a controller may name, and an unknown one is an error."""
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD)
    gateway = a_gateway(stream)
    await refused(gateway.pull({}))

    with pytest.raises(ValueError):
        await gateway.recover_refused_pull("f" * 64)
    assert len(gateway.refused_pulls) == 1
    assert len(stream.requests) == 1


async def test_a_recovery_takes_nothing_another_call_is_owed() -> None:
    """One call at a time here too, and a recovery is not an exception to it."""
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD, WAITING)
    gateway = a_gateway(stream)
    await refused(gateway.pull({}))
    [identity] = gateway.refused_pulls
    await gateway.recover_refused_pull(identity)

    # The record now belongs to the recovery, and a second recovery of the same pull finds
    # nothing retained to send.
    with pytest.raises(ValueError):
        await gateway.recover_refused_pull(identity)


async def test_no_public_message_changes_for_any_of_this() -> None:
    """The one new token, the closed payload schema, and no resolver detail anywhere.

    What a refusal says is that the evidence is unavailable. Which object was missing, which cell
    it belonged to and which resolver went looking are facts about the run's evidence and none of
    them is a thing to hand the agent.
    """
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD)
    gateway = a_gateway(stream)

    record = await refused(gateway.pull({}))
    assert set(record) == {"code", "kind", "protocol_version"}

    [identity] = gateway.refused_pulls
    await gateway.recover_refused_pull(identity)
    delivered = json.loads(await gateway.pull({}))
    assert set(delivered) == {"message_id", "attempt_id", "body", "kind", "protocol_version"}
    assert delivered["kind"] == "payload"
    assert delivered["body"] == BODY


async def test_a_controller_turned_away_at_this_door_is_not_a_refusal_an_agent_was_given() -> None:
    """The count is reconciled against what the model saw, so only what it saw is in it.

    A recovery holds this transport the way a tool call does, and a controller that asks while
    another call is in flight is turned away with the same code. That refusal reaches a
    controller and never an agent, so there is no entry in the harness's transcript for it to be
    checked against: counting it would make the two records disagree by exactly the errors nobody
    was ever told about.

    The same door counts an agent's own overlapping call, which is what says the count was
    narrowed rather than turned off.
    """
    stream = Reserving(StreamProtocolError("evidence_unavailable"), PAYLOAD)
    gateway = a_gateway(stream)
    await refused(gateway.pull({}))
    [identity] = gateway.refused_pulls
    assert gateway.refusals == 1

    # Another call is holding this transport, the way one in flight does.
    gateway._claim(b"another call")
    record = await refused(gateway.recover_refused_pull(identity))
    assert record["code"] == "overlapping_call"
    assert gateway.refusals == 1
    # The pull was not sent and the record is still retained for the next repair.
    assert gateway.refused_pulls == (identity,)
    assert len(stream.requests) == 1

    # And an agent's own call at the same door is counted, because the model is told about it.
    assert (await refused(gateway.pull({})))["code"] == "overlapping_call"
    assert gateway.refusals == 2
