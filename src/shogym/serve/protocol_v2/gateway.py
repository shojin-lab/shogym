"""The protocol v2 gateway: one environment, served to a model over MCP.

The kernel owns the stream and the gateway owns the transport. Everything here is a
consequence of that split. The kernel cannot see who is calling, so this is where one
authenticated consumer is bound and where a second one is refused. The kernel cannot see the
model transcript, so this is where an offered result is delivered and attested. And the kernel
answers Updates rather than tool calls, so this is where a tool call is turned into a request
that is either well formed or refused before it is sent.

Two kinds of tool reach the model. ``pull`` takes no arguments and returns one protocol record.
Every environment tool is wrapped in the closed ``{attempt_id, arguments}`` object, which makes
the attempt the routing handle and keeps a native argument name from colliding with a protocol
field. The environment's terminal tool is wrapped like the others and then intercepted: it never
reaches the environment from here, it becomes the stream's terminal request, and it answers with
the acknowledgement or the refusal the stream minted.

A result the model sees is the exact bytes the stream offered, in one MCP text item, never
re-rendered. A refusal is not a result: it carries the canonical ProtocolError record in the
transport's error, which is the channel MCP has for one and is not where a result travels.
Whether a harness puts the bytes of that error in front of the model is the harness's to say,
and this server has no way to reach into a transcript it cannot see.

One rule holds the delivery loop together: this gateway does not trust what it remembers. Every
call the model can make begins by reading the stream's own state, finishing whatever that read
says is unfinished, and only then accepting new work. What is kept between calls is a cache of
that read, and losing it costs a query rather than a generation.

Two things kept here are not caches. One is the call that is running: it is claimed before
anything can await, so a second call is refused rather than queued, and it is given back where
the operation settles rather than where its caller stops waiting. The other is one recovery
record, which says what this gateway has left open and who may close it. A request that was
sent and never answered, a grant the stream has not been told is over, a message offered, an
attestation sent, a message whose attestation was refused, and a result that exists and has not
reached anyone are six states of that one record, each with exactly one owner. Every other call
is refused with `outstanding_response` until the owner comes back for what is owed.

An environment call is the one thing here that changes a world the stream cannot see, so it is
not dispatched on this gateway's own reading of the stream. The stream decides it and stays
held while it happens, which is what keeps the decision and the change from being two moments
with room between them.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Coroutine,
    Dict,
    FrozenSet,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import jsonschema
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from temporalio.client import Client
from temporalio.service import RPCError

from shogym.serve.episode import ServedEpisode
from shogym.serve.protocol_v2 import (
    ProtocolError,
    PullRequest,
    TerminalMetadata,
    WireFormatError,
    PresentationCommit,
    canonical_bytes,
    canonical_json,
    length_prefixed,
    require_opaque_id,
)
from shogym.serve.protocol_v2.kernel import (
    ConsumerClaim,
    EnvironmentCall,
    OfferedMessage,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamState,
    TaskItem,
    TerminalTool,
    durable_client,
    protocol_error_code,
    start_stream,
    stream_worker,
)
from shogym.serve.server import build_tool
from shogym.task import TaskSpec, ToolManifest

PULL_TOOL = "pull"

_LOG = logging.getLogger(__name__)

# The stream's own words for the facts this gateway routes by. They are read out of the stream
# rather than remembered, so nothing here is a second opinion about any of them.
_OPEN = "open"
_DONE = "done"
_ACTIVE = "active"

# What this transport is doing, which is a different question from what the generation is.
_SERVING = "serving"
_CLOSING = "closing"
_CLOSED = "closed"

_Result = TypeVar("_Result")

# The version this gateway declares for the canonical submission its terminal captures. The
# capture itself belongs to the environment, so the name says which gateway made the promise.
CANONICALIZATION_VERSION = "shogym.gateway.1"

# The version of the surface this gateway renders around an environment: the control tool it
# adds, the wrapper every environment tool is advertised behind, and the note appended to a
# description. It is part of what is served, so a change to it is a changed configuration even
# where the environment's own manifest is untouched.
WRAPPER_VERSION = "shogym.gateway.wrapper.1"

_PULL_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

_PULL_DESCRIPTION = (
    "Ask the stream for your next message. Takes no arguments. The result is one JSON record: "
    "a task to work on, a payload, a wait, or done. Work only on the task you were given, and "
    "pull again when you have finished with it."
)

_WRAPPER_NOTE = (
    '\n\nCall this tool as {"attempt_id": <the attempt_id of your current task>, '
    '"arguments": {...the tool\'s own arguments...}}.'
)


def _opaque() -> str:
    """Mint one opaque 128-bit identifier."""
    return secrets.token_hex(16)


def _refusal(code: str) -> ToolError:
    """Return the transport error that carries one protocol refusal.

    The canonical record is the error's whole text. It travels on the transport's error
    channel, so a caller that reads results and a caller that reads errors never confuse a
    refusal with a message the stream offered.
    """
    return ToolError(canonical_bytes(ProtocolError(code=code)).decode("utf-8"))


def _wrapper_schema(native: Dict[str, Any]) -> Dict[str, Any]:
    """Return the closed wrapper schema for one environment tool.

    The native schema is nested rather than dropped, because the model has to know what the
    tool takes. Nothing here validates it: the outer shape is what this schema declares, and
    the native arguments are checked where their failure has a protocol answer.
    """
    return {
        "type": "object",
        "properties": {
            "attempt_id": {
                "type": "string",
                "description": "The attempt_id from the task you are working on.",
            },
            "arguments": native,
        },
        "required": ["attempt_id", "arguments"],
        "additionalProperties": False,
    }


def terminal_manifest(spec: TaskSpec) -> ToolManifest:
    """Return the one tool that ends an attempt.

    An environment that scores declares a score terminal, and one that does not ends on the
    reserved abort. Either way there is exactly one, which is what the stream needs: a second
    tool that could seal would be a second way to end an attempt the ledger cannot tell apart.
    """
    for kind in ("score", "abort"):
        for manifest in spec.tools:
            if manifest.terminal_kind == kind:
                return manifest
    raise ValueError(
        f"env {spec.env_name!r} advertises no terminal tool, so no call could end an attempt"
    )


def wrapped_manifests(spec: TaskSpec, terminal: ToolManifest) -> List[ToolManifest]:
    """Return the environment tools the model may call.

    Everything is here except a reserved abort that is not the terminal. An environment that
    scores has one, and under this protocol a call to it would end the environment's episode
    without a terminal request, so the stream would hold an attempt that nothing can still
    seal. The one terminal the stream knows about is the only way out of an attempt.
    """
    return [
        manifest
        for manifest in spec.tools
        if manifest.name == terminal.name or manifest.terminal_kind != "abort"
    ]


def declared_argument_names(schema: Dict[str, Any]) -> List[str]:
    """Return the argument names the stream holds a terminal call to.

    The stream's terminal schema is a set of names, so a JSON Schema has to be read down to
    one. The required names are that set when the schema declares any, and the property names
    are it otherwise.
    """
    properties = schema.get("properties") or {}
    required = schema.get("required") or []
    return sorted(required) if required else sorted(properties)


def served_manifest(spec: TaskSpec, terminal: ToolManifest) -> Dict[str, Any]:
    """Return everything this gateway serves, in the words it serves it in.

    What a resume has to serve identically is what the model could read and call, and what the
    episode behind it is allowed to do. So the tools are here as they are advertised, with the
    description the model reads and the wrapper it sends its arguments in rather than the
    environment's own schema; the horizon is here, because it is how many environment actions
    the attempt gets; and the terminal is here as the stream knows it, by name, by kind, and by
    the argument names a filing is held to.

    Two things that are not the environment's are here too. This gateway adds a control tool the
    environment never declared and renders every other one behind its wrapper, so the surface is
    the manifest and this renderer together, and the renderer says which version of itself made
    it. A tool this gateway does not serve is not here at all: nothing about it reaches the
    model or the stream, so two generations that differ only there are serving the same thing.
    """
    return {
        "env_name": spec.env_name,
        "task_id": spec.task_id,
        "contract_version": spec.contract_version,
        "instructions": spec.instructions,
        "horizon": spec.horizon,
        "wrapper_version": WRAPPER_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "control_tool": {
            "name": PULL_TOOL,
            "description": _PULL_DESCRIPTION,
            "schema": _PULL_SCHEMA,
        },
        "terminal": {
            "name": terminal.name,
            "kind": terminal.terminal_kind,
            "argument_names": declared_argument_names(terminal.input_schema),
        },
        "tools": [
            {
                "name": tool.name,
                "description": tool.description + _WRAPPER_NOTE,
                "schema": _wrapper_schema(tool.input_schema),
            }
            for tool in wrapped_manifests(spec, terminal)
        ],
    }


def _configuration_hash(spec: TaskSpec, terminal: ToolManifest) -> str:
    """Hash what the generation is serving, so a resume can refuse a changed configuration."""
    return sha256(canonical_json(served_manifest(spec, terminal))).hexdigest()


def stream_start(spec: TaskSpec, terminal: ToolManifest, *, claim_hash: str) -> StreamStart:
    """Return the generation that serves ``spec``: one task, one attempt, one obligation."""
    item = TaskItem(
        task_position=0,
        attempt_id=_opaque(),
        task_message_id=_opaque(),
        ack_message_id=_opaque(),
        payload_position=0,
        payload_message_id=_opaque(),
        body=spec.instructions,
    )
    return StreamStart(
        configuration_hash=_configuration_hash(spec, terminal),
        consumer_claim_hash=claim_hash,
        initial_cursor=_opaque(),
        done_message_id=_opaque(),
        id_key_hex=secrets.token_hex(32),
        hidden_execution_id=_opaque(),
        canonicalization_version=CANONICALIZATION_VERSION,
        terminal_tool=TerminalTool(
            public_tool_name=terminal.name,
            native_terminal_name=terminal.name,
            argument_names=declared_argument_names(terminal.input_schema),
        ),
        tasks=[item],
    )


def _call_key(tool_name: str, arguments: Dict[str, Any]) -> bytes:
    """Name one call by exactly what it asked for.

    What a result is owed to is the call that asked for it, and two calls of the same kind are
    not the same call: a revised filing is a different submission from the one the stream may
    already have sealed. So the name is the tool and its arguments, canonically, which is what
    the stream computes a request's identity from too.

    Arguments with no canonical encoding have no name here, and they do not need one: such a
    call is refused before it reaches anything that could land a result, so the name it would
    have been kept under is never asked for.
    """
    try:
        return canonical_json({"tool": tool_name, "arguments": arguments})
    except WireFormatError:
        return canonical_json({"tool": tool_name, "arguments": None})


# The recovery record: one at a time, and every state of it but the first has exactly one
# owner. The owner is the call that made the thing this gateway has left open, named by what
# that call asked for, and it is the only call that may advance it.


@dataclass(frozen=True)
class _Idle:
    """Nothing is left open, so a call this gateway accepts may start new work."""

    owner: None = None


@dataclass(frozen=True)
class _RequestUncertain:
    """A request that was sent and never answered, so nothing here knows what it did.

    The request is kept by value rather than rebuilt. The stream reserves what it offers for
    the request that asked for it, derives the durable call's identity from the same bytes, and
    refuses every other request while it is holding a result, so the one call that can still
    reach that result is this one sending this request again.
    """

    owner: bytes
    request: Union[PullRequest, SealRequest]


@dataclass(frozen=True)
class _Offered:
    """A message the stream offered, and the attestation that will present it.

    A commit names the stream as it was before the event, so one built a second time describes
    a stream that has already moved and attests nothing about the delivery it was meant to
    attest. It is fixed once here and repeated afterwards.
    """

    owner: bytes
    message: OfferedMessage
    commit: PresentationCommit


@dataclass(frozen=True)
class _PresentationUncertain:
    """An attestation that was sent and never acknowledged.

    The stream answers a repeated attestation with the acknowledgement it applied and with no
    other, so what finishes this is sending the same commit again rather than making a new one.
    """

    owner: bytes
    message: OfferedMessage
    commit: PresentationCommit


@dataclass(frozen=True)
class _PresentationRefused:
    """A message the stream is still holding, whose attestation it refused.

    A refusal is decisive about the attestation and about nothing else. The stream verifies one
    against things that are not in it, the objects its references name among them, and refuses
    the ones it cannot verify without applying anything: the cursor stays where it was and the
    message stays reserved for the request that was given it. So the message is kept, because
    it is still owed and nothing else could present it.

    The attestation is not kept. It is the one thing the refusal did settle, and it settled it
    as failed: the stream reaches an attestation by an identity its own ID is derived from, so
    the same one sent again is answered with the refusal it already gave rather than verified a
    second time. Fixing what it was refused about would then change nothing. What the retry
    sends is a new attestation for the same message, built from a stream that has not moved.
    """

    owner: bytes
    message: OfferedMessage


@dataclass(frozen=True)
class _LeaseHeld:
    """A grant this gateway asked the stream for and has not given back.

    The stream holds the generation for the exact call it granted, so no other call can end that
    hold: a release whose answer never arrives is not something to walk away from, because
    nothing else can make it land. The grant is written down before it is asked for, so one
    whose answer was lost is still given back by name, and it keeps the observation once the
    world has changed, because that observation exists nowhere else.
    """

    owner: bytes
    call: EnvironmentCall
    result: Optional[ToolResult] = None


@dataclass(frozen=True)
class _ResultOwed:
    """A result that exists and has not reached the call that asked for it.

    Either the stream has counted its message as read, or a world the stream cannot see has
    already changed. Neither is something anything can produce a second time, so it waits under
    the name of the call that asked, and nothing else is served until that call comes back.

    One result is the end of the generation rather than a step in it, and it says so by keeping
    the cursor it left the stream at. Done is the message that closes a stream: nothing is
    written after it, and the stream that answered the attestation for it stops existing once
    the calls it accepted have finished. So the question every other kept result is handed over
    on, whether this transport may still write, is one the last result cannot be asked, and
    ``closed_at`` is what the answer is read from instead.
    """

    owner: bytes
    result: Any
    closed_at: Optional[str] = None


_Recovery = Union[
    _Idle,
    _RequestUncertain,
    _LeaseHeld,
    _Offered,
    _PresentationUncertain,
    _PresentationRefused,
    _ResultOwed,
]


@dataclass
class _Operation:
    """The one public call this gateway is running, named by what it asked for."""

    key: bytes


class GatewayClosed(RuntimeError):
    """Raised when a call arrives after this transport stopped serving its generation.

    It is deliberately not a protocol refusal. A transport that has stopped says nothing about
    the durable generation, which may well still be open, and answering `closed_stream` would
    assert a fact only the stream is in a position to state.
    """


class StreamGateway:
    """One transport's view of one stream generation.

    The gateway holds what the stream deliberately does not: who the caller is, which attempt
    that caller is currently working on, and what has been put in front of the model. It sends
    an Update only when the request it would send is already well formed, so a malformed call
    is a protocol refusal here rather than a transport fault somewhere below.

    What it does not hold is an opinion about the stream. Every call the model can make reads
    the stream's state first, finishes any delivery that read says is unfinished, and only then
    asks for new work. The cached fields below are what that read last said: losing them costs
    one query, and disagreeing with the stream is not something they are given the chance to do.

    Two fields are not that. One call is running at a time, claimed before the first await and
    released where the operation settles, and one recovery record says what has been left open
    and which call may close it.
    """

    def __init__(
        self,
        stream: StreamHandle,
        episode: ServedEpisode,
        spec: TaskSpec,
        terminal: ToolManifest,
        *,
        initial_cursor: str,
    ) -> None:
        self._stream = stream
        self._episode = episode
        self._spec = spec
        self._terminal = terminal.name
        self._cursor = initial_cursor
        self._active: FrozenSet[str] = frozenset()
        self._transcript: List[bytes] = []
        self._closed = False
        # What each tool declares it takes, which is what a call to it is held to. The wrapper
        # carries the native object where the transport's own schema does not reach it.
        self._schemas: Dict[str, Dict[str, Any]] = {
            manifest.name: manifest.input_schema for manifest in spec.tools
        }
        # The one call this gateway is running, and the operation it handed the work to. The
        # operation holds the gateway, so a caller that goes away does not release it.
        self._operation: Optional[_Operation] = None
        self._landing: Optional["asyncio.Future[Any]"] = None
        self._recovery: _Recovery = _Idle()
        # Whether this transport is still serving, which the stream neither knows nor decides.
        self._serving = _SERVING
        self._shutdown: Optional["asyncio.Future[None]"] = None
        self._refusals = 0

    @property
    def spec(self) -> TaskSpec:
        """The contract this gateway serves."""
        return self._spec

    @property
    def terminal_tool(self) -> str:
        """The tool whose call is intercepted into the stream's terminal request."""
        return self._terminal

    @property
    def cursor(self) -> str:
        """The last presented message, which every request this gateway sends carries."""
        return self._cursor

    @property
    def refusals(self) -> int:
        """How many refusals this transport has issued.

        A refusal leaves nothing in the generation to count, so the count is kept here. It is a
        cross-check against the harness transcript, which is where a refusal the model saw is
        recorded, and nothing this gateway decides reads it.
        """
        return self._refusals

    def _refuse(self, code: str) -> ToolError:
        """Count one refusal, write its code to the log, and return the error carrying it."""
        self._refusals += 1
        _LOG.info("protocol v2 refusal: %s", code)
        return _refusal(code)

    def check_native_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> None:
        """Hold a wrapped call's native object to the schema its tool declares.

        The transport's own schema reaches the wrapper and stops there, and the wrapper carries
        the native object nested inside itself. So the server that advertises a wrapper checks
        what it advertised, before the call reaches the environment: a call the tool never
        accepted would otherwise be answered by whatever that environment's framework says
        about it, having already spent a step of the episode and written itself into the
        trajectory.

        The terminal is held to it too. Its filing never reaches the environment, it becomes the
        stream's terminal request, and the stream holds it to the names the tool declares rather
        than to their types: a filing the advertised schema refuses would otherwise be sealed,
        and a seal is the one answer no later call can take back.

        Only the native object is judged here. A closed stream, a call already in flight, and a
        wrapper that is not the shape this protocol declares are all refused by the call itself,
        and each of those refusals stays the one the protocol names for it.
        """
        native = arguments.get("arguments")
        schema = self._schemas.get(tool_name)
        busy = self._operation is not None
        if self._closed or busy or schema is None or not isinstance(native, dict):
            return
        try:
            jsonschema.validate(instance=native, schema=schema)
        except (jsonschema.ValidationError, jsonschema.SchemaError) as error:
            # A schema that cannot be read is refused along with the call it would have judged.
            # Dispatching one anyway is exactly the unchecked call this is here to stop.
            raise self._refuse("invalid_message") from error

    async def pull(self, arguments: Dict[str, Any]) -> str:
        """Return the exact bytes of the next protocol result."""
        key = _call_key(PULL_TOOL, arguments)
        operation = self._claim(key)
        return await self._accepted(operation, self._pull(key, arguments))

    async def terminal(self, arguments: Dict[str, Any]) -> str:
        """End the attempt named in the wrapper, and return the stream's answer."""
        key = _call_key(self._terminal, arguments)
        operation = self._claim(key)
        return await self._accepted(operation, self._terminal_call(key, arguments))

    async def environment(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Dispatch one ordinary environment tool against the routed attempt."""
        key = _call_key(tool_name, arguments)
        operation = self._claim(key)
        return await self._accepted(operation, self._environment(key, tool_name, arguments))

    async def aclose(self) -> None:
        """Stop serving, let the one accepted call settle, and close the episode.

        Admission stops first and stops synchronously, so a call either was already accepted or
        is refused, and there is no backlog to run during teardown: an overlapping call was
        refused when it arrived rather than queued for later. What was accepted is left to
        settle, because a call that has changed a world the stream cannot see is not something
        a shutdown may walk away from, and the episode is closed only after it has.

        A grant the stream is still holding for this transport is the one thing the stop
        finishes. It is the same Update under the exact call this gateway wrote down before
        asking for it, so sending it again is the stream's own idempotency rather than a second
        decision, and it puts nothing into the world. Leaving it would hold the generation
        behind a transport that is no longer here to give it back.

        Anything the recovery record still owes is left where it is rather than presented,
        collected, or discarded. This gateway is the only thing that knows what it is owed to,
        and a clean stop is not a reason to throw away the one copy of it. That includes what
        the released call landed with: the grant going back is not the observation being read.

        Closing twice closes once. The second caller waits on the same shutdown, and a caller
        that goes away does not take it with them.
        """
        if self._shutdown is None:
            self._serving = _CLOSING
            self._shutdown = asyncio.ensure_future(self._stopped())
        await asyncio.shield(self._shutdown)

    async def _stopped(self) -> None:
        """Run the shutdown once, whoever asked for it and however many are waiting.

        The grant goes back after the accepted call has settled, because until then it is that
        call's to give back, and before anything is marked closed, because a stop that recorded
        a cleanup it did not do would leave the generation held by nobody. A stream that will
        not take it fails the stop and leaves the record where the next owner of the generation
        will find it.
        """
        landing = self._landing
        if landing is not None:
            await asyncio.wait([landing])
        await self._given_back_on_stop()
        self._serving = _CLOSED
        await self._episode.close()

    async def _given_back_on_stop(self) -> None:
        """Give back a grant this transport is still holding, and keep what it landed with."""
        record = self._recovery
        if not isinstance(record, _LeaseHeld):
            return
        await self._given_back(record)
        if record.result is not None:
            self._recovery = _ResultOwed(owner=record.owner, result=record.result)

    async def _pull(self, key: bytes, arguments: Dict[str, Any]) -> str:
        """Ask the stream for the next message, once it has nothing older to give.

        The model's own call carries nothing, so an argument is not a value to ignore: it is a
        call this protocol does not define, and it is refused before a request is built.
        """
        if arguments:
            raise self._refuse("invalid_message")
        owed = await self._resumed(key)
        if owed is not None:
            return owed
        request = self._pull_request(key)
        message = await self._decisive(key, self._stream.pull(request))
        return await self._deliver(message, key)

    async def _terminal_call(self, key: bytes, arguments: Dict[str, Any]) -> str:
        """End the attempt named in the wrapper, and return the stream's answer.

        The call never reaches the environment from here. Sealing has to record its prepared
        state before any finalizer runs, so the environment's half of the terminal belongs
        inside the stream's transaction and not in front of it.
        """
        attempt_id, native = self._unwrap(arguments)
        owed = await self._resumed(key)
        if owed is not None:
            return owed
        request = self._seal_request(attempt_id, native, key)
        result = await self._decisive(key, self._stream.seal(request))
        return await self._deliver(result, key)

    async def _environment(
        self, key: bytes, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Dispatch one ordinary environment tool against the routed attempt.

        ``attempt_id`` is stripped before dispatch, so the environment sees the arguments it
        declared and a native argument of that name would still be its own. What makes it the
        routed attempt is the stream saying so, and saying so at the moment of the call rather
        than at some earlier read: the stream decides this call and holds the generation until
        it settles, so an attempt that ended in between is not one it still reaches.
        """
        attempt_id, native = self._unwrap(arguments)
        owed = await self._resumed(key)
        if owed is not None:
            return owed
        held = _LeaseHeld(
            owner=key, call=EnvironmentCall(call_id=_opaque(), attempt_id=attempt_id)
        )
        self._recovery = held
        await self._granted(held)
        try:
            observation = await self._episode.call(tool_name, native)
        except BaseException:
            # The world's own failure is what its caller is told about. The release still goes,
            # and one that faults too leaves the grant where this call will find it again.
            await self._given_back_quietly(held)
            raise
        # Only the tool's own observation goes back. Feedback under this protocol is a
        # presented Payload, so a sidecar carrying it here would be a second channel that
        # no offer, presentation, or delivery count could see.
        result = ToolResult(content=observation.content)
        landed = _LeaseHeld(owner=key, call=held.call, result=result)
        self._recovery = landed
        await self._given_back(landed)
        return result

    async def _granted(self, held: _LeaseHeld) -> None:
        """Ask the stream to hold the generation while one environment call happens.

        The stream cannot serialize a call it never sees and cannot take one back afterwards,
        so what it does instead is decide this one before it happens and stay held while it
        does. A refusal is that decision, and it says nothing was granted, so the record this
        call wrote before asking closes with it.

        A fault leaves the grant unknown, and a grant that was made and lost holds the
        generation against every later call, so the release goes by name at once. When that
        release does not land either, the record is what keeps the grant reachable: the same
        call is the only one that can still give it back.
        """
        try:
            await self._sent(self._stream.begin_environment_call(held.call))
        except ToolError:
            self._recovery = _Idle()
            raise
        except BaseException:
            await self._given_back_quietly(held)
            raise

    async def _given_back(self, held: _LeaseHeld) -> None:
        """Give the generation back for one call, or leave the record holding it.

        The stream holds the generation for the exact call it granted, so nothing but that call
        can end the hold. A release whose answer never comes is therefore kept rather than
        dropped, and the fault is raised rather than swallowed: a caller told its call is over
        would never send the one release that can free the generation, and the observation it
        landed with would be the last thing this stream ever served.

        An answer settles it either way. A refusal is the stream saying it is holding nothing
        for this call, which is as decisive as the lease it hands back when it was.
        """
        try:
            await self._sent(self._stream.end_environment_call(held.call))
        except ToolError:
            pass
        self._recovery = _Idle()

    async def _given_back_quietly(self, held: _LeaseHeld) -> None:
        """Give the generation back where the caller is already being told about something else."""
        try:
            await self._given_back(held)
        except Exception:
            pass

    # The entry discipline: the stream first, then what it says is unfinished, then new work.

    def _claim(self, key: bytes) -> _Operation:
        """Take this gateway for one call, before anything has had the chance to await.

        The check and the claim are both synchronous, because a call that waited its turn would
        be queued where the protocol says it is refused. What is claimed is the operation and
        not its waiter, so it is given back where the operation settles.

        A transport that has stopped serving refuses without saying anything about the stream.
        A closed generation is the stream's own answer, and it still owes what it has not handed
        over: Done is what closes a generation, so the pull that asked for Done may be the one
        caller that has not read it. A generation with something owed is entered rather than
        refused, and the recovery record decides who gets what.
        """
        if self._serving != _SERVING:
            raise GatewayClosed("this transport has stopped serving its generation")
        if self._closed and isinstance(self._recovery, _Idle):
            raise self._refuse("closed_stream")
        if self._operation is not None:
            raise self._refuse("overlapping_call")
        operation = _Operation(key=key)
        self._operation = operation
        return operation

    async def _accepted(
        self, operation: _Operation, call: Coroutine[Any, Any, _Result]
    ) -> _Result:
        """Run one accepted call to the end, whatever becomes of the caller waiting on it.

        What holds this gateway is the operation, not its waiter. A call handed to a world this
        transport does not own is already running and will still land, and a terminal the stream
        accepted runs its Activities and writes its acknowledgement whether or not anyone is
        listening for it. So the operation is shielded from its caller's cancellation, and the
        gateway stays held against overlap until the operation settles rather than until the
        waiter is dropped. Without that, an MCP call cancelled mid-seal leaves the way open for
        an environment call to reach a world the seal is in the middle of capturing.

        What is owed is let go of here and nowhere earlier, because this returning is the last
        thing that can tell that the result reached the caller that asked for it. A waiter that
        went away in the meantime leaves it where the same call, asked again, will find it. The
        gateway is given back after that and never before, so the window between an operation
        settling and its result being written down is one no other call can enter.
        """
        landing = asyncio.ensure_future(call)
        self._landing = landing
        # Nobody may be left to read a failure, and a failure nobody reads is reported as an error.
        landing.add_done_callback(_read_failure)
        try:
            result = await asyncio.shield(landing)
        except asyncio.CancelledError:
            landing.add_done_callback(partial(self._abandoned, operation))
            raise
        except BaseException:
            # A failure settles the operation and owes nothing. What the record holds open, if
            # anything, is the request or the attestation whose fate nobody knows.
            self._released(operation)
            raise
        self._recovery = _Idle()
        self._released(operation)
        return result

    def _released(self, operation: _Operation) -> None:
        """Give this gateway back once the operation that held it has nothing left to do."""
        if self._operation is operation:
            self._operation = None
            self._landing = None

    def _abandoned(self, operation: _Operation, landing: "asyncio.Future[Any]") -> None:
        """Keep what an operation landed with once the call that asked for it has gone.

        A delivery has already written itself down: the stream applied its attestation and this
        gateway watched that happen. An ordinary result has nothing of the kind, because the
        world changed, the observation of that change exists nowhere else, and dispatching the
        same call again would change the world a second time. So it waits under the name of the
        call that asked, and the gateway is given back after it is written and not before.
        """
        if not landing.cancelled() and landing.exception() is None:
            if isinstance(self._recovery, _Idle):
                self._recovery = _ResultOwed(owner=operation.key, result=landing.result())
        self._released(operation)

    async def _resumed(self, key: bytes) -> Optional[Any]:
        """Take the stream's word for where this generation is, and finish what it left open.

        This is the first thing every call does. The stream is the only authority on where the
        cursor is, which attempts are being worked on, and whether anything may still be served,
        so the notes this gateway keeps between calls are refreshed here from the thing they are
        notes about.

        Then the owner is compared, and that comparison comes before everything else. Nothing an
        unrelated call does may advance what another call left open: not a presentation retried
        on its behalf, not a request allocated in its place, not a cursor moved under it, and
        not a dispatch into a world it is waiting on. A call that is not the owner is refused
        with `outstanding_response` and changes nothing, which leaves what is open exactly where
        the owner will find it.

        The owner is what settles it. A message the stream offered has its fixed attestation
        sent, or sent again, and then its bytes go back to the call that asked for them. A
        refusal is the one case where the attestation is not the one that goes again: it
        applied nothing, so the message is still the stream's to be holding and the stream is
        still where the refused one described, but that attestation has been answered and would
        be answered the same way for ever, so a new one is built for the same message once the
        stream has said it is still holding it. When the stream is holding something else, or
        nothing, this gateway cannot say what became of those bytes, and it says that rather
        than deciding it. A request whose answer was lost is sent again as itself. A grant the
        stream may still be holding is given back by name before anything else happens, and
        whatever that call landed with is handed over once it is. A result that already exists
        is handed over, once the stream has been asked whether this transport may still hand
        anything over at all.

        A result the stream is holding is covered by the same rule even when this transport
        never learned that it exists. The stream reserves an offered message for the request
        that asked for it and refuses every other request while it holds one, so a lost answer
        leaves the generation waiting on one call. An ordinary environment call never reaches
        the stream to be refused by it, so that refusal is made here. A gateway that has
        forgotten whose message it is refuses everything, the exact call included: the stream
        would answer the same way, and this way no world is changed before it does.
        """
        state = await self._sent(self._stream.stream_state())
        self._adopt(state)
        record = self._recovery
        if record.owner is not None and record.owner != key:
            raise self._refuse("outstanding_response")
        if isinstance(record, _Idle):
            if state.pending_message_id is not None:
                raise self._refuse("outstanding_response")
            if self._closed:
                raise self._refuse("closed_stream")
            return None
        if isinstance(record, _RequestUncertain):
            return None
        if isinstance(record, _LeaseHeld):
            # The grant goes back first, because the stream is holding the generation for this
            # exact call and refuses everything else meanwhile. A call that landed hands over
            # what it landed with; one that never reached the world starts over.
            await self._given_back(record)
            if record.result is None:
                return None
            return await self._owed(record.result, state)
        if isinstance(record, _ResultOwed):
            return await self._owed(record.result, state, closed_at=record.closed_at)
        if isinstance(record, _PresentationRefused):
            if state.pending_message_id != record.message.message_id:
                # The message this gateway was keeping is not the one the stream is holding, so
                # the bytes it owes were offered and never presented and nothing here can say
                # what became of them. Saying so is the honest answer; clearing the record and
                # serving the next call would be this transport deciding that on the stream's
                # behalf.
                raise RuntimeError(
                    "the stream is no longer holding the message this call was offered, so "
                    "the presentation it owes cannot be made"
                )
            record = _Offered(
                owner=record.owner,
                message=record.message,
                commit=await self._attestation(record.message),
            )
            self._recovery = record
        return await self._present(record)

    async def _owed(
        self, result: Any, state: StreamState, closed_at: Optional[str] = None
    ) -> Any:
        """Hand over a result this gateway kept, once the stream says it may still be handed over.

        A kept result was decided while this transport was the one writing to the generation,
        and every call since has read that generation with a query. A query answers a
        transport the generation has been taken from exactly as it answers the one that holds
        it, so nothing read that way can tell the two apart, and a terminal acknowledgement
        handed over on the strength of it would reach a model whose generation somebody else is
        now serving.

        So the last thing before the bytes go is a question the stream can refuse: the same
        state, asked through the path a write takes rather than around it. It is asked here and
        not earlier, because a takeover is not something this gateway can hold still: anything
        checked further back could stop being true while the rest of the call ran.

        The result that closed the generation is the one exception, and it is an exception
        because the question has no one left to ask. A stream ends at Done: it accepts nothing
        after it, and it stops running once the calls it had already taken are finished, so the
        write path this asks through is gone rather than refusing. What that question was for is
        gone with it. There is no later writer to be holding a generation that has finished, and
        these bytes put nothing into it: they are the stream's own last word, and the harness
        that asked for them is being told to stop.

        The stream is still what says so. The record kept the cursor Done left it at, and the
        authority is taken at its word that the generation is closed and standing there. A
        stream saying anything else is one whose Done these are not, and that is asked the same
        question every other kept result is.
        """
        if closed_at is not None and state.generation_state == _DONE:
            if state.cursor == closed_at:
                return result
        self._adopt(await self._sent(self._stream.confirm_state()))
        return result

    def _adopt(self, state: StreamState) -> None:
        """Take the three facts this gateway routes by out of the stream that owns them.

        The cursor every request carries, the attempts an ordinary call may reach, and whether
        the generation is still open. An attempt is reachable exactly while the stream says it
        is active, which is the same condition under which the stream would accept work on it.
        """
        self._cursor = state.cursor
        self._closed = state.generation_state != _OPEN
        self._active = frozenset(
            attempt for attempt, value in state.attempts.items() if value == _ACTIVE
        )

    def _routed(self, attempt_id: str) -> None:
        """Refuse a call that does not name an attempt this transport is serving."""
        if attempt_id not in self._active:
            raise self._refuse("invalid_attempt")

    # Requests, built before they are sent.

    def _pull_request(self, key: bytes) -> PullRequest:
        """Return the request this pull is made under, and write it down before it is sent.

        A pull whose answer never arrives may have left a result reserved for the request that
        asked for it, and the stream hands that result back to that request and to no other. The
        record is therefore installed here rather than after an answer, because the case it
        exists for is the one where no answer comes: from this point on this call owns the
        recovery, every other call is refused, and the retry that reaches the reserved result is
        this request sent again.
        """
        record = self._recovery
        if isinstance(record, _RequestUncertain) and isinstance(record.request, PullRequest):
            return record.request
        try:
            request = PullRequest(request_id=_opaque(), last_presented_cursor=self._cursor)
        except WireFormatError as error:
            raise self._refuse("invalid_message") from error
        self._recovery = _RequestUncertain(owner=key, request=request)
        return request

    def _seal_request(
        self, attempt_id: str, native: Dict[str, Any], key: bytes
    ) -> SealRequest:
        """Return the request this filing is made under, and write it down before it is sent.

        A filing whose answer never arrives may be the one that sealed the attempt, and the
        acknowledgement it minted is reachable through that request and through no other. So the
        request is frozen before it goes: an exact retry is sent again under the same identity,
        and every other call is refused before it reaches here. Losing the first one to a second
        filing would leave a sealed generation with an acknowledgement nothing can still ask
        for, and losing it to a pull would do the same.

        A retry is not routed by the attempt it names, because the seal it is repeating may be
        what ended that attempt: the stream takes the seal when it mints the acknowledgement,
        and a retry turned away for naming a sealed attempt is the one call that could still
        collect it. A filing that is not a retry is routed like every other call.
        """
        record = self._recovery
        if isinstance(record, _RequestUncertain) and isinstance(record.request, SealRequest):
            # The owner was compared before anything here ran, so this is that filing again.
            return record.request
        self._routed(attempt_id)
        request = SealRequest(
            metadata=self._terminal_metadata(attempt_id),
            public_tool_name=self._terminal,
            native_terminal_name=self._terminal,
            native_arguments=native,
        )
        self._recovery = _RequestUncertain(owner=key, request=request)
        return request

    def _terminal_metadata(self, attempt_id: str) -> TerminalMetadata:
        try:
            return TerminalMetadata(
                request_id=_opaque(),
                last_presented_cursor=self._cursor,
                attempt_id=attempt_id,
            )
        except WireFormatError as error:
            raise self._refuse("invalid_message") from error

    def _unwrap(self, arguments: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Read the wrapper, or refuse it.

        The wrapper is closed, so a missing field and an extra one are the same kind of
        mistake. Only its shape is judged here. Whether the attempt it names is one this
        transport is serving is the stream's answer, and it is asked after the stream has been
        read rather than against whatever this gateway last remembered.
        """
        if set(arguments) != {"attempt_id", "arguments"}:
            raise self._refuse("invalid_message")
        attempt_id = arguments["attempt_id"]
        try:
            require_opaque_id("attempt_id", attempt_id)
        except WireFormatError as error:
            raise self._refuse("invalid_message") from error
        native = arguments["arguments"]
        if not isinstance(native, dict):
            raise self._refuse("invalid_message")
        try:
            canonical_json(native)
        except WireFormatError as error:
            raise self._refuse("invalid_message") from error
        return attempt_id, dict(native)


    # Sending, delivering, attesting.

    async def _sent(self, call: Any) -> Any:
        """Await one Update and turn the stream's refusal into this transport's error.

        A failure that carries no protocol code is a fault, and it is raised as one. The single
        exception is a stream that has already finished: Temporal answers that its execution is
        complete, which under this protocol is what closed means.
        """
        try:
            return await call
        except Exception as error:
            code = protocol_error_code(error)
            if code is None and _stream_finished(error):
                code = "closed_stream"
            if code is None:
                raise
            if code == "closed_stream":
                self._closed = True
            raise self._refuse(code) from error

    async def _decisive(self, key: bytes, call: Any) -> Any:
        """Send one request whose refusal, if it comes, says what that request did.

        The request was written into the recovery record before this, because the case that
        matters is a fault, where nobody can say whether the stream accepted it. A refusal is
        the stream saying that it did not: nothing was reserved, nothing is owed, and the record
        that was held open for an answer closes here.
        """
        try:
            return await self._sent(call)
        except ToolError:
            record = self._recovery
            if isinstance(record, _RequestUncertain) and record.owner == key:
                self._recovery = _Idle()
            raise

    async def _deliver(self, message: OfferedMessage, key: bytes) -> str:
        """Fix the attestation ``message`` travels under, then present it.

        A commit names the stream as it was before the event, so one built a second time
        describes a stream that has already moved and attests nothing about the delivery it was
        supposed to attest. It is built once, kept, and repeated.
        """
        record = self._recovery
        if not isinstance(record, (_Offered, _PresentationUncertain)) or (
            record.message.message_id != message.message_id
        ):
            record = _Offered(
                owner=key, message=message, commit=await self._attestation(message)
            )
            self._recovery = record
        return await self._present(record)

    async def _attestation(self, message: OfferedMessage) -> PresentationCommit:
        """Build the one attestation that will present ``message``, from the stream itself."""
        state = await self._sent(self._stream.stream_state())
        transcript_blob = self._transcript_hash(message.visible_text)
        completed_turn = message.kind == "seal_ack"
        return PresentationCommit(
            attestation_id=_opaque(),
            cursor_before=state.cursor,
            message_id=message.message_id,
            visible_bytes_sha256=sha256(message.visible_text.encode("utf-8")).hexdigest(),
            transcript_blob=transcript_blob,
            provider_turn_blob=transcript_blob if completed_turn else None,
            task_start_checkpoint_blob=transcript_blob if message.kind == "task" else None,
            completed_turn=completed_turn,
            stream_state_before_sha256=state.stream_state_sha256,
        )

    async def _present(self, record: Union[_Offered, _PresentationUncertain]) -> str:
        """Attest that the exact offered bytes were delivered, and advance the cursor.

        The attestation is made as the bytes are handed to the transport, and that is the whole
        of what it says. A harness that owns the model's transcript attests after writing it,
        which is a stronger claim than this one and belongs to the runner rather than to a
        server.

        The uncertainty is installed before the attestation goes, for the same reason a request
        is: an answer that never comes leaves nobody able to say whether it was applied, and
        the stream answers a repeated attestation with the acknowledgement of the one it did
        apply. What the message owes survives the acknowledgement, because the bytes are not
        read until the call that asked for them returns them.
        """
        self._recovery = _PresentationUncertain(
            owner=record.owner, message=record.message, commit=record.commit
        )
        try:
            ack = await self._sent(self._stream.commit_presentation(record.commit))
        except ToolError:
            # The stream answered, and what it answered about is the attestation: it applied
            # nothing, so it is still holding the message this one names and still at the cursor
            # this one was built from. The refusal is a reason to fix what it was refused about
            # and attest that message again, not a reason to walk away from it. The attestation
            # goes with the refusal, because it is the one thing that was decided. A fault is
            # the other case, and it keeps the commit: nobody knows yet whether that one was
            # applied, and the same one sent again is what finds out.
            self._recovery = _PresentationRefused(
                owner=record.owner, message=record.message
            )
            raise
        self._applied(record.message, ack.cursor)
        self._recovery = _ResultOwed(
            owner=record.owner,
            result=record.message.visible_text,
            # Done is the one message whose presentation ends the stream it was presented to, so
            # what it is owed to is recorded with the cursor it ended at rather than with a
            # promise that the stream can still be asked about it.
            closed_at=ack.cursor if record.message.kind == "done" else None,
        )
        return record.message.visible_text

    def _applied(self, message: OfferedMessage, cursor: str) -> None:
        """Move this transport the way presenting ``message`` moved the stream."""
        self._transcript.append(message.visible_text.encode("utf-8"))
        self._cursor = cursor
        if message.kind == "task" and message.attempt_id is not None:
            self._active = self._active | {message.attempt_id}
        elif message.kind == "seal_ack" and message.attempt_id is not None:
            # The attempt is sealed, so nothing may still be done to it. A SealReject leaves
            # the attempt where it was and deliberately does not reach this branch.
            self._active = self._active - {message.attempt_id}
        elif message.kind == "done":
            self._closed = True

    def _transcript_hash(self, text: str) -> str:
        """Hash everything presented so far, with ``text`` appended.

        Each entry is length prefixed, so no two transcripts of different messages hash alike
        by running one message's bytes into the next.
        """
        digest = sha256()
        for entry in self._transcript:
            digest.update(length_prefixed(entry))
        digest.update(length_prefixed(text.encode("utf-8")))
        return digest.hexdigest()


def _read_failure(landing: "asyncio.Future[Any]") -> None:
    """Read the outcome of a call whose caller may be gone, so a failure is not left unhandled."""
    if not landing.cancelled():
        landing.exception()


def _stream_finished(error: BaseException) -> bool:
    """True when the transport is saying the stream's execution is already over."""
    if isinstance(error, RPCError):
        return "already completed" in error.message.lower()
    cause = error.__cause__
    return cause is not None and _stream_finished(cause)


def build_gateway_server(gateway: StreamGateway, *, name: Optional[str] = None) -> FastMCP:
    """Build the MCP server the model talks to: ``pull``, and every environment tool wrapped.

    A native tool named ``pull`` is refused here rather than served: the control tool and an
    environment tool of the same name cannot both be reached, and the one that would lose is
    the one the whole protocol runs through.

    Every wrapped call is held to the native schema this server advertised for it, the terminal
    filing included, because the wrapper is where that schema is nested and nothing below it
    looks there. What happens to a call the gateway accepts is the gateway's: it runs each one
    to the end whatever becomes of the caller waiting on it.
    """
    spec = gateway.spec
    served = wrapped_manifests(spec, terminal_manifest(spec))
    for manifest in spec.tools:
        if manifest.name == PULL_TOOL:
            raise ValueError(
                f"env tool name {PULL_TOOL!r} collides with the stream control tool; an env "
                f"served under protocol v2 may not expose a tool named {PULL_TOOL!r}"
            )

    async def dispatch(tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == PULL_TOOL:
            return ToolResult(content=await gateway.pull(arguments))
        gateway.check_native_arguments(tool_name, arguments)
        if tool_name == gateway.terminal_tool:
            return ToolResult(content=await gateway.terminal(arguments))
        return await gateway.environment(tool_name, arguments)

    server: FastMCP = FastMCP(name=name or f"shogym:{spec.env_name}")
    server.add_tool(
        build_tool(
            ToolManifest(
                name=PULL_TOOL, description=_PULL_DESCRIPTION, input_schema=_PULL_SCHEMA
            ),
            dispatch,
        )
    )
    for manifest in served:
        server.add_tool(
            build_tool(
                manifest.model_copy(update={"description": manifest.description + _WRAPPER_NOTE}),
                dispatch,
                parameters=_wrapper_schema(manifest.input_schema),
            )
        )
    return server


async def open_gateway(
    client: Client,
    episode: ServedEpisode,
    *,
    workflow_id: Optional[str] = None,
    consumer_id: Optional[str] = None,
) -> StreamGateway:
    """Start a generation for ``episode`` and bind this transport as its one consumer.

    The claim secret is minted here and never leaves. It is what authentication amounts to at
    this layer: the stream binds whoever presents it first, and a second transport presenting
    anything else is refused before a message has been offered.
    """
    spec = episode.describe()
    terminal = terminal_manifest(spec)
    start = stream_start(spec, terminal, claim_hash=sha256(secrets.token_bytes(32)).hexdigest())
    stream = await start_stream(
        client, start, workflow_id=workflow_id or f"stream/{_opaque()}/1"
    )
    receipt = await stream.claim_consumer(
        ConsumerClaim(
            consumer_id=consumer_id or _opaque(), claim_hash=start.consumer_claim_hash
        )
    )
    # The queue is closed at the start because it is complete at the start: this gateway serves
    # the one episode it was opened on, so nothing can be inserted later and Done becomes
    # reachable once that episode's task has been sealed, acknowledged, and paid out.
    await stream.close_queue()
    return StreamGateway(stream, episode, spec, terminal, initial_cursor=receipt.initial_cursor)


async def run_stdio_v2(
    env_name: str,
    *,
    task: Optional[Union[int, str]] = None,
    trace_path: Optional[Union[str, Path]] = None,
) -> None:
    """Serve one environment under protocol v2 over stdio, durably.

    The service, the Worker, and the stream all belong to this process, so a harness spawns one
    command and gets a durable stream without installing or starting anything.

    The gateway is stopped before the Worker and the service are, because stopping it settles
    whatever call was accepted when the transport went away, and that call may still need the
    stream. Stopping it closes the episode, so the episode is closed here only when there was
    no gateway to do it or its stop did not finish.
    """
    episode = await ServedEpisode.start(env_name, task=task, trace_path=trace_path)
    stopped = False
    try:
        async with durable_client() as client:
            async with stream_worker(client):
                gateway = await open_gateway(client, episode)
                try:
                    await build_gateway_server(gateway).run_async(transport="stdio")
                finally:
                    await gateway.aclose()
                    stopped = True
    finally:
        if not stopped:
            await episode.close()
