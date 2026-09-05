"""The protocol v2 gateway: one environment, served to a model over MCP.

The kernel owns the stream and the gateway owns the transport. Everything here is a
consequence of that split. The kernel cannot see who is calling, so this is where one
authenticated consumer is bound and where a second one is refused. The kernel cannot see the
model transcript, so this is where an offered result is delivered and attested. And the kernel
answers Updates rather than tool calls, so this is where a tool call is turned into a request
that is either well formed or refused before it is sent.

Two kinds of tool reach the model. ``pull`` takes no arguments and returns one protocol record,
and a generation that declares it serves ``info`` beside it, which takes no arguments either and
returns how many of that generation's attempts have been handed out, how many of those have not
ended, and how many are still to be handed out.
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
from dataclasses import dataclass, replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Coroutine,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import jsonschema
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from temporalio.client import Client
from temporalio.service import RPCError

from shogym.serve.episode import ServedEpisode
from shogym.serve.protocol_v2 import (
    FLOOR_HORIZON,
    GRADED_HORIZON,
    HORIZON_ENDINGS,
    HORIZON_FILED,
    IMMEDIATE,
    NEVER,
    Assignment,
    BlobStore,
    FilesystemBlobStore,
    InfoRequest,
    ProtocolError,
    PullRequest,
    ReleasePlan,
    TerminalMetadata,
    WireFormatError,
    PresentationCommit,
    canonical_bytes,
    canonical_json,
    check_release,
    length_prefixed,
    require_declaration,
    require_opaque_id,
    require_step_budget,
)
from shogym.serve.protocol_v2.kernel import (
    STEP_CAP,
    STREAM_TASK_QUEUE,
    ConsumerClaim,
    EnvironmentCall,
    FinalizeRequest,
    OfferedMessage,
    QueueClosed,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamState,
    TaskItem,
    TerminalTool,
    assignments_for,
    configuration_hash,
    discard_stream,
    durable_client,
    kernel_activities,
    protocol_error_code,
    start_stream,
    stream_worker,
)
from shogym.serve.protocol_v2.policy import (
    DELIVER,
    EXPERIMENT,
    HONEST_V1,
    HONEST,
    KERNEL_STAND_IN_GRADE,
    LEGACY,
    NO_OBLIGATION,
    NO_RELEASE,
    HONEST_V1_DIGEST,
    ORDINARY,
    PLATFORM_DEFAULT,
    POLICIES,
    REGISTERED,
    WITHHOLD,
    GradeIdentity,
    MatchedFamily,
    PayloadDisposition,
    PolicyProvenance,
    PolicyViolation,
    check_dispositions,
    descriptor_digests,
    policy_digest,
    policy_preimage,
    roster_digest,
)
from shogym.serve.protocol_v2.rundir import (
    create_run_directory,
    prepare_run_directory,
    stage_run_directory,
    staged_generation,
)
from shogym.serve.server import build_tool
from shogym.task import TaskSpec, ToolManifest

PULL_TOOL = "pull"
INFO_TOOL = "info"

_LOG = logging.getLogger(__name__)

# The stream's own words for the facts this gateway routes by. They are read out of the stream
# rather than remembered, so nothing here is a second opinion about any of them.
_OPEN = "open"
_DONE = "done"
_ACTIVE = "active"
_FINAL_FAILED = "final_failed"

# What this transport is doing, which is a different question from what the generation is.
_SERVING = "serving"
_CLOSING = "closing"
_CLOSED = "closed"

_Result = TypeVar("_Result")

#: How a generation gets a world for the task it is about to serve. One episode is one task's
#: world: it is opened when that task is presented and closed when the attempt is over. It is
#: told which attempt it is opening for, because sealing an attempt belongs to the environment
#: rather than to this transport, and an environment that seals by stopping its own world has to
#: be told which world that attempt filed in. This gateway is the only thing that knows.
EpisodeOpener = Callable[[str], Awaitable[ServedEpisode]]

#: Where a harness keeps this transport's count of the refusals it has issued. A refusal advances
#: no protocol state, so the generation has nothing to count and the model's transcript is the
#: refusal's whole record; this count is the cross-check on that record, and it is only useful to
#: a harness that still holds it when the run is read. It is called with the new count inside the
#: call that issues the refusal and before the error goes back, so a transport killed the instant
#: after a refusal has already told whoever keeps the number. Nothing this gateway decides reads
#: it, and what it does with the number is the harness's own business.
RefusalSink = Callable[[int], None]

#: The dispositions an experiment registers, built from the roster they resolve. A row names an
#: attempt and a payload position, and both are minted where the generation is, so a registration
#: is given as a function of the roster rather than as a finished value: it is called with the
#: rows, in order, once they exist and before anything has been served.
DispositionsFor = Callable[[Sequence[Assignment]], Sequence[PayloadDisposition]]

#: A release plan built from the queue it releases. A gate names an attempt and the attempts are
#: minted where the generation is, so a composer that gates one is handed the tasks it gates.
ReleaseFor = Callable[[Sequence[TaskItem]], ReleasePlan]


class WorldRoute:
    """Which world each attempt filed in, for an environment that ends an attempt in its own.

    The pairing exists nowhere else. A generation works each task in a world of its own, this
    gateway is what opens them, and the environment's Activities are registered on a Worker
    before the first of those worlds exists. An environment asked for its terminal is therefore
    handed this rather than one world, and it resolves an attempt when it seals rather than when
    it is registered.

    An attempt this process never opened a world for resolves to nothing, which is what every
    attempt resolves to in a process that took the generation over. That is the answer the
    environment needs: there is no world here, so either the evidence was already captured under
    the seal id or this owner cannot capture it.

    An attempt whose world has been let go of resolves to nothing for the same reason, so the
    pairing is forgotten as the world closes. A world is closed once the attempt that worked in
    it is over, and by then the seal has read whatever it was going to read, so what an entry
    left behind would answer with is a torn down environment presented as somewhere to capture
    evidence from.
    """

    def __init__(self) -> None:
        self._worlds: Dict[str, ServedEpisode] = {}

    def record(self, attempt_id: str, episode: ServedEpisode) -> None:
        """Say that this attempt is working in this episode's world."""
        self._worlds[attempt_id] = episode

    def forget(self, attempt_id: str, episode: ServedEpisode) -> None:
        """Say that this world of this attempt's is gone, if it is still the one recorded.

        The episode is named rather than only the attempt, because an attempt may have been
        handed to a newer owner in this process: a replacement records the world it restored for
        the attempt it took over, and the transport it replaced still holds the world it was
        working in and lets go of it afterwards. That later cleanup is about its own world, so it
        clears the pairing only where the pairing is still that world's.
        """
        if self._worlds.get(attempt_id) is episode:
            del self._worlds[attempt_id]

    def __call__(self, attempt_id: str) -> Optional[Tuple[Any, str]]:
        """The environment and session this attempt filed in, or ``None`` if not this process."""
        episode = self._worlds.get(attempt_id)
        return None if episode is None else (episode.env, episode.session_id)


class EnvironmentTerminal(NamedTuple):
    """What an environment answers when a generation asks how its attempts end.

    ``configuration_digest`` is the environment's half of what the generation is. The served
    manifest carries the instructions and the tools, and an environment may have more than that
    deciding what a filing is worth, so what it publishes here is folded into the identity a
    resume checks itself against.

    ``grade`` is what its grader is. An environment that brings its own terminal brings its own
    verdict with it, and one that does not is scored by a stand-in that reaches no world, so this
    is the fact a generation checks before it may promise an agent the score.

    ``horizon_ending`` is what spending the environment's step budget does. The default is the
    floor: the world work is over, nothing was filed, and the attempt ends with the reason on it.
    An environment whose horizon is an ending its own scorer answers for declares that instead,
    and then running out of steps is a filing of the world as it stands rather than an attempt
    nobody finished.
    """

    canonicalization_version: str
    activities: List[Any]
    configuration_digest: Optional[str]
    route: WorldRoute
    grade: GradeIdentity = KERNEL_STAND_IN_GRADE
    horizon_ending: str = FLOOR_HORIZON

# The version a generation declares for the canonical submission its terminal captures when the
# environment does not declare one of its own. The capture belongs to the environment, so the name
# says which gateway made the promise, and an environment that brings its own terminal replaces
# both the promise and the Activities that keep it (see :func:`environment_terminal`).
CANONICALIZATION_VERSION = "shogym.gateway.1"

# The version of the surface this gateway renders around an environment: the control tool it
# adds, the wrapper every environment tool is advertised behind, and the note appended to a
# description. It is part of what is served, so a change to it is a changed configuration even
# where the environment's own manifest is untouched.
WRAPPER_VERSION = "shogym.gateway.wrapper.1"

_PULL_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

# What every generation says about the control tool, up to the sentence that says how many tasks
# the agent may be working on. That sentence is the one part of the text a capacity decides, so
# the two are written as one literal per capacity below rather than assembled from pieces: the
# text a generation at one serves is the text it served before a capacity could be declared, and
# it stays a single literal a reader can check byte for byte.
_PULL_OPENING = (
    "Ask the stream for your next message. Takes no arguments. The result is one JSON record: "
    "a task to work on, a payload, a wait, or done. "
)

_PULL_DESCRIPTION = _PULL_OPENING + (
    "Work only on the task you were given, and pull again when you have finished with it."
)

# What the control tool says about a budget, where the generation declares one. It is appended
# to the description rather than written into it, because the description a generation that
# declares nothing serves is the one it served before a budget could be declared, and the model
# reads that text out of the same manifest a resume is held to.
_BUDGET_NOTE = (
    " A task also carries `budget`, the whole allowance of environment tool calls that task "
    "gets rather than what is left of it, and neither pulling nor filing spends it."
)

_INFO_SCHEMA: Dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

# What a generation that declares the info tool says about it. It is a literal of its own rather
# than a sentence spliced into the control tool's, because this is a second tool with a second
# description, and a generation that declares no such tool serves neither.
_INFO_DESCRIPTION = (
    "Ask how much work there is. Takes no arguments, and is answered between delivered results, "
    "while the run is open and nothing is owed: asking while an earlier call is still owed a "
    "result, or after the run is done, is refused. The result is one JSON record with three "
    "counts: `consumed`, how many tasks have been handed out so far; `in_flight`, how many of "
    "those have not ended yet; and `remaining`, how many are still to be handed out. `in_flight` "
    "counts tasks `consumed` counts too, and a run can end a task it never handed out, so the "
    "three need not add up to the whole run. They are totals for the run and name no task. "
    "`remaining` reaching zero does not mean the run is over: a payload may still be owed and a "
    "task may still be running, so stop when `pull` answers with done."
)

_WRAPPER_NOTE = (
    '\n\nCall this tool as {"attempt_id": <the attempt_id of your current task>, '
    '"arguments": {...the tool\'s own arguments...}}.'
)

# The refusals a finalization the gateway made for itself may be answered with. Both say the
# attempt is already over, which is the state the finalization was asking for: the deadline can
# have reached it first, and either way there is nothing left to end. Any other code is a fault
# and is raised, because then the attempt is still open and nothing ended it.
_ALREADY_OVER = ("invalid_attempt", "conflicting_seal")

# What a call that reached no world landed with. A horizon filing carries back whatever the call
# that owed it landed with, and a call this transport refused a step to landed with nothing.
_EMPTY_RESULT = ToolResult(content=[])


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


def _capacity_sentence(capacity: int) -> str:
    """Return what a generation serving more than one task at a time says about that.

    Three facts, because three is what an agent needs to work several tasks without asking: how
    many it may hold, what a pull gets while it holds that many, and how it ends one of them. The
    last is the one a reader of the single-task sentence would get wrong: a terminal call names
    the attempt it ends, and an agent holding several has to name the right one.

    What the middle one says is what the stream does, which is narrower than a wait. A capacity
    that is full stops tasks being offered and stops nothing else, so a pull can still answer with
    a payload or with anything else the schedule has ready, and the wait is what is left when
    there is none. Saying it as a flat wait would be telling the agent to stop pulling for
    messages it is owed. The ending is written the same way: a terminal is how the agent ends a
    task, and it is not the only way a task ends, since a deadline or a spent budget ends one
    without the agent doing anything.
    """
    return (
        f"You may hold up to {capacity} tasks at once, so you may pull again before you have "
        "finished the one you are working on. While you hold that many, a pull cannot return "
        "another task, and it answers a wait when nothing else is ready for you. To end a task "
        "yourself, call its terminal with that task's attempt_id."
    )


def _pull_description(budget: Optional[int] = None, capacity: int = 1) -> str:
    """Return the words the control tool is served under.

    A generation that declares a budget says so here, in one sentence appended to the text every
    generation serves. Appending is the whole of the change: the description a generation that
    declares nothing serves is the one it served before there was a budget to declare, byte for
    byte, so its configuration hash has not moved and its resume is not refused.

    A capacity is the other way round. What it changes is the sentence about how many tasks the
    agent works at once, because a generation that serves several would otherwise be telling the
    agent to finish before it pulls again: the text would forbid the one thing the capacity is
    there to allow. So the sentence is replaced rather than appended to, and a generation at one
    serves the whole text unchanged.
    """
    if capacity <= 1:
        described = _PULL_DESCRIPTION
    else:
        described = _PULL_OPENING + _capacity_sentence(capacity)
    return described if budget is None else described + _BUDGET_NOTE


def served_manifest(
    spec: TaskSpec,
    terminal: ToolManifest,
    horizon_ending: str = FLOOR_HORIZON,
    budget: Optional[int] = None,
    capacity: int = 1,
    info: bool = False,
) -> Dict[str, Any]:
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

    What the horizon does is here where an environment declares something other than the floor.
    A budget that ends an attempt at the floor and one that files the world as it stands are two
    different rules to be scored under, so the identity has to cover which was in force. It is
    written only where it is declared, because a generation recorded before an environment could
    say hashed exactly the keys below and reopening one of those under an added key would refuse
    every resume of it over a rule that has not changed.

    ``budget`` is the number this generation hands the agent on every task, and what it changes
    here is the control tool's own description, on the same terms: the sentence is appended where
    a generation declares a number and the text is untouched where it declares none.

    ``capacity`` is how many tasks this generation lets the agent hold at once, and it changes the
    same description: how many tasks to work on is a rule of the generation the model reads out of
    that text and nowhere else, and a generation serving several under the words for one would be
    telling the agent not to do what it is being served. A capacity of one leaves the text alone.

    ``info`` is whether this generation answers how much of its queue is left. It is a second tool
    rather than a sentence, so it is written here as one, under the rule the horizon's ending is
    written under: a generation that declares it names the tool it serves and the words it serves
    it in, and a generation that declares none has nothing here at all, which is what "a tool this
    gateway does not serve is not here" means for the only other tool there is.
    """
    declared: Dict[str, Any] = (
        {} if horizon_ending == FLOOR_HORIZON else {"horizon_ending": horizon_ending}
    )
    if info:
        declared["info_tool"] = {
            "name": INFO_TOOL,
            "description": _INFO_DESCRIPTION,
            "schema": _INFO_SCHEMA,
        }
    return {
        **declared,
        "env_name": spec.env_name,
        "task_id": spec.task_id,
        "contract_version": spec.contract_version,
        "instructions": spec.instructions,
        "horizon": spec.horizon,
        "wrapper_version": WRAPPER_VERSION,
        "canonicalization_version": CANONICALIZATION_VERSION,
        "control_tool": {
            "name": PULL_TOOL,
            "description": _pull_description(budget, capacity),
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


def _configuration_hash(
    spec: TaskSpec,
    terminal: ToolManifest,
    environment_digest: Optional[str] = None,
    horizon_ending: str = FLOOR_HORIZON,
    budget: Optional[int] = None,
    capacity: int = 1,
    info: bool = False,
) -> str:
    """Hash what the generation is serving, so a resume can refuse a changed configuration.

    The served manifest is what a model can see: the instructions, the tools, the schemas. An
    environment may also have settings that decide what a filing is worth without appearing in
    any of that, and one that publishes a digest of them has it folded in here. One that
    publishes nothing hashes exactly what it hashed before.
    """
    manifest = served_manifest(spec, terminal, horizon_ending, budget, capacity, info)
    if environment_digest is not None:
        manifest = {**manifest, "environment": environment_digest}
    return sha256(canonical_json(manifest)).hexdigest()


def stream_start(
    spec: TaskSpec,
    terminal: ToolManifest,
    *,
    claim_hash: str,
    bodies: Optional[Sequence[str]] = None,
    release: Optional[Union[ReleasePlan, ReleaseFor]] = None,
    without_payload: Sequence[int] = (),
    evaluation_only: bool = False,
    attempt_deadline_ms: int = 0,
    canonicalization_version: str = CANONICALIZATION_VERSION,
    environment_digest: Optional[str] = None,
    profile: str = ORDINARY,
    grade: GradeIdentity = KERNEL_STAND_IN_GRADE,
    dispositions: Optional[Union[Sequence[PayloadDisposition], DispositionsFor]] = None,
    experiment: str = "",
    families: Sequence[MatchedFamily] = (),
    budget: Optional[int] = None,
    capacity: int = 1,
    info: bool = False,
) -> StreamStart:
    """Return a generation that serves ``spec``.

    This is the one place a generation is built, so it is the one place the manifest, the
    preallocated public identifiers, the assignment roster, and the release plan are decided
    together. They have to be: a roster names positions the manifest declares, and a plan can
    gate a task only by the attempt ID this function minted for it. So a plan that gates
    anything is given as a function of the queue rather than as a finished value: it is called
    with the tasks, in order, once they exist and before anything has been served.

    ``bodies`` is the closed queue, one work order to an entry, and it defaults to the episode's
    own instructions as a single task. ``without_payload`` names the positions this generation
    delivers nothing against: they are served, worked, and scored like every other task, and a
    leg's filler is what needs them. An evaluation-only generation is one that scores without
    delivering: it pins Never, and a plan that would produce an outbox is refused here rather
    than left to be noticed when a payload turns up in a transcript.

    ``canonicalization_version`` and ``environment_digest`` are what the environment answered
    when it was asked how its attempts end. The first is what the acknowledgements declare their
    digests were taken under; the second is the environment's half of what this generation is,
    and it goes into the configuration hash a resume checks itself against, so a generation
    restarted under a differently configured environment is refused before it is worked rather
    than scored against a key nobody drew for it.

    ``attempt_deadline_ms`` is how long an attempt may stay active before the generation ends it.
    It is off by default, because how long a run is willing to wait on one attempt is a decision
    about that run and not about this transport.

    ``profile`` is what kind of run this is, and it decides how an unspecified payload policy is
    read. An ordinary generation stamps every obligation with the honest policy before it is
    created and every silent position with the reason it delivers nothing, so an ordinary run
    cannot be blinded by an omission: there is no omission left by the time the generation
    exists. An experiment has no default at all. It is created from the dispositions it
    registered and refused where one of its positions is uncovered, so an experiment cannot be
    unblinded by an omission either. Both mistakes are refusals rather than quiet resolutions.

    ``grade`` is what the environment said its grader is, and honesty is a claim about that
    number. A generation over an environment scored by a stand-in cannot deliver the honest
    policy, and asking for one is refused here rather than discovered in a receipt.

    ``dispositions`` is what an experiment registered. A row names an attempt, and the attempts
    are minted here, so a registration that names them is given as a function of the roster and
    called with the rows once they exist, exactly as a plan that gates anything is.

    ``experiment`` is which experiment registered them, and an experiment generation is not
    built without one. A profile is a word, and this is what makes it a fact: the name goes into
    the generation with the digest of the rows it registered, so a run cannot call itself an
    experiment with nothing behind the label and cannot carry a registration made over other
    answers. An ordinary generation names none and is recorded as stamped from the platform
    default instead.

    ``families`` are the matched arms those rows are cells of, where an experiment declares
    them. Each fixes the group its cells are built in and the byte count they come to, and a
    cell that comes back as anything else ends the attempt rather than being served.

    ``budget`` is what this generation tells the agent it may spend, on every task it serves. It
    is off by default: handing the number over is a decision about the run, and a generation that
    makes no such decision serves the task record it served before the number could be handed
    over. Where it is declared it has to be the budget this transport enforces, and a generation
    that says anything else is refused here.

    ``capacity`` is how many tasks this generation may have active at once, and one is the
    default. It is one number in two places: the stream stops offering tasks while that many are
    live, and the control tool's own description says so, because how many tasks the agent may
    hold is a rule of the generation and the model reads the rules of the generation there. It is
    held to the same rule as the number a task record carries, an exact whole count of at least
    one, and a generation that says anything else is refused here rather than at start.

    ``info`` is whether this generation serves the tool that says how much of its queue is left,
    and it is off by default. It says yes or no and is held to that here, for the reason the
    capacity is held to being a count: everything downstream reads it as one of two, so anything
    else would be turned into one of them by each of them separately rather than by a decision
    anyone made. What that tool answers is three counts of this generation's own
    attempts, so what a generation declares here is a decision to tell the agent how much work
    there is; one that declares nothing serves the surface it served before there was such a tool,
    and there is no word about it in what that generation hashes.

    What comes back is a generation the stream would accept, because the same check the stream
    makes at start is made here, where the caller composing a run is the one who reads it.
    """
    _check_declared_budget(spec, budget)
    # A capacity is a count of tasks and it is held to the rule the number on a task record is
    # held to: an exact integer of at least one, inside the range a JSON reader can carry. The
    # check is here, before anything is minted, because every value this refuses composes a
    # generation that either cannot start or serves the words of a generation at one while
    # calling itself something else.
    require_step_budget("capacity", capacity)
    require_declaration("info", info)
    items = [
        TaskItem(
            task_position=position,
            attempt_id=_opaque(),
            task_message_id=_opaque(),
            ack_message_id=_opaque(),
            payload_position=position,
            payload_message_id=_opaque(),
            body=body,
        )
        for position, body in enumerate(bodies if bodies is not None else [spec.instructions])
    ]
    if callable(release):
        plan = release(items)
    else:
        plan = release if release is not None else (NEVER if evaluation_only else IMMEDIATE)
    if evaluation_only and plan.creates_obligations:
        raise ValueError(
            "an evaluation-only generation delivers no payload, so its release plan must be "
            f"Never, and this one releases {plan.predicate!r}"
        )
    if any(position not in range(len(items)) for position in without_payload):
        raise ValueError(
            "a position that carries no payload is one of this generation's own tasks, and "
            f"this queue has {len(items)}"
        )
    roster = assignments_for(
        items, plan, without_payload=[items[position].attempt_id for position in without_payload]
    )
    check_release(plan, roster, evaluation_only=evaluation_only)
    registered = dispositions(roster) if callable(dispositions) else dispositions
    resolved = _resolved_dispositions(
        profile, roster, plan.creates_obligations, registered, grade, experiment, list(families)
    )
    return StreamStart(
        configuration_hash=_configuration_hash(
            spec, terminal, environment_digest, budget=budget, capacity=capacity, info=info
        ),
        consumer_claim_hash=claim_hash,
        initial_cursor=_opaque(),
        done_message_id=_opaque(),
        id_key_hex=secrets.token_hex(32),
        hidden_execution_id=_opaque(),
        canonicalization_version=canonicalization_version,
        terminal_tool=TerminalTool(
            public_tool_name=terminal.name,
            native_terminal_name=terminal.name,
            argument_names=declared_argument_names(terminal.input_schema),
        ),
        tasks=items,
        capacity=capacity,
        release=plan,
        assignments=roster,
        evaluation_only=evaluation_only,
        attempt_deadline_ms=attempt_deadline_ms,
        profile=profile,
        grade=grade,
        dispositions=resolved,
        provenance=_provenance(profile, resolved, experiment),
        families=list(families),
        budget=budget,
        info=info,
    )


def _check_declared_budget(spec: TaskSpec, budget: Optional[int]) -> None:
    """Refuse a declared budget that is not the one this transport enforces.

    The number on a task record is what an agent paces its work against, and the step cap this
    transport enforces is the environment's own horizon, counted here call by call. A generation
    that advertised one and enforced the other would be handing the agent a figure nothing keeps,
    and the agent would find out by running short of world where it expected room. So the two are
    the same number or the generation is not composed.

    What a number has to be comes first, and it is the record's own rule rather than one this
    layer keeps beside it. Equality alone would let ``True``, a float that compares equal, zero
    and a negative through, because an environment's published horizon carries no bound of its
    own; each of those composes a generation whose first offer cannot build the record it owes.
    """
    if budget is None:
        return
    require_step_budget("budget", budget)
    if spec.horizon is None:
        raise ValueError(
            f"env {spec.env_name!r} publishes no step budget, so a generation over it has no "
            f"number to declare on the tasks it serves, and this one declares {budget}"
        )
    if budget != spec.horizon:
        raise ValueError(
            f"this generation declares a budget of {budget} on every task it serves and env "
            f"{spec.env_name!r} is served with {spec.horizon} environment actions to the "
            "attempt; the number an agent reads is the step cap this transport enforces"
        )


def _provenance(
    profile: str, rows: Sequence[PayloadDisposition], experiment: str
) -> Optional[PolicyProvenance]:
    """Return what entitles this generation to the profile it is being built under.

    An experiment names itself and the rows it registered. An ordinary run names the platform
    default it was stamped from and the same rows. Both are the record of who decided, and the
    digest is what stops that record from being moved onto another set of answers.
    """
    if profile == EXPERIMENT:
        return PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(list(rows)),
            experiment_id=experiment,
        )
    return PolicyProvenance(
        authority=PLATFORM_DEFAULT,
        roster_digest=roster_digest(list(rows)),
        descriptor_digest=HONEST_V1_DIGEST,
    )


def _resolved_dispositions(
    profile: str,
    roster: Sequence[Assignment],
    creates_obligations: bool,
    registered: Optional[Sequence[PayloadDisposition]],
    grade: GradeIdentity,
    experiment: str,
    families: List[MatchedFamily],
) -> List[PayloadDisposition]:
    """Return one resolved disposition per roster row, or refuse to build the generation.

    The two profiles differ in one way and it is the whole of the design. Under the ordinary
    profile there is nothing to register: every row that owes a payload is stamped with the
    honest policy and every row that owes none is stamped with the reason its roster gave, both
    marked as the platform's conversion so a reader can see they were not chosen. A caller that
    hands registered rows to an ordinary run is refused, because concealment is something a run
    declares itself to be doing rather than something it passes in.

    Under the experiment profile there is nothing to convert. The rows are the ones the
    experiment registered, they are checked against the roster and against the environment's
    grader, and a position nobody covered is a generation that does not get created.
    """
    if profile == ORDINARY:
        if registered:
            raise ValueError(
                "an ordinary generation delivers the honest policy against every payload it "
                "owes, so a registered disposition belongs to an experiment run rather than to "
                "this one"
            )
        if experiment:
            raise ValueError(
                f"this generation is being built as {ORDINARY} under the experiment "
                f"{experiment!r}, and a run an experiment registered is an {EXPERIMENT} run"
            )
        rows = [_stamped(row, creates_obligations) for row in roster]
    elif profile == EXPERIMENT:
        if registered is None:
            raise ValueError(
                "an experiment generation has no default policy, so it is created from the "
                "dispositions it registered and not from an omission"
            )
        if not experiment:
            raise ValueError(
                "an experiment generation says which experiment registered it, because a "
                "profile nothing stands behind is a word rather than a fact about the run"
            )
        rows = list(registered)
    else:
        raise ValueError(f"a generation is created as {ORDINARY} or {EXPERIMENT}, not {profile!r}")
    obligations = {
        row.attempt_id: row.payload_position
        for row in roster
        if row.creates_payload_obligation and creates_obligations
    }
    silent = {
        row.attempt_id: row.payload_position
        for row in roster
        if row.attempt_id not in obligations
    }
    try:
        check_dispositions(
            rows,
            profile=profile,
            obligations=obligations,
            silent=silent,
            grade=grade,
            provenance=_provenance(profile, rows, experiment),
            families=families,
        )
    except PolicyViolation as error:
        raise ValueError(str(error)) from error
    return rows


def _stamped(row: Assignment, creates_obligations: bool) -> PayloadDisposition:
    """Return the disposition an ordinary run converts one roster row's silence into.

    A row that owes a payload delivers the honest policy in its one cell. A row that owes none
    withholds, under the reason the roster already implies: a plan that releases nothing, or a
    position this generation was composed to deliver nothing against. Neither is an absence in
    the record. What separates an ordinary run that delivers nothing from an experiment that
    conceals is that this row says which of the two it is and who decided.
    """
    if creates_obligations and row.creates_payload_obligation:
        return PayloadDisposition(
            attempt_id=row.attempt_id,
            payload_position=row.payload_position,
            kind=DELIVER,
            policy_digest=policy_digest(HONEST_V1),
            cell=HONEST_V1.cells[0],
            resolution_source=PLATFORM_DEFAULT,
        )
    return PayloadDisposition(
        attempt_id=row.attempt_id,
        payload_position=row.payload_position,
        kind=WITHHOLD,
        reason=NO_RELEASE if not creates_obligations else NO_OBLIGATION,
        resolution_source=PLATFORM_DEFAULT,
    )


def _landed(record: Union["_Offered", "_PresentationUncertain"]) -> Any:
    """Return what the call that produced this delivery hands back.

    A message the model asked for is its own answer, and the bytes go back as they are. A
    message that came of some other call goes back behind what that call landed with, in one
    result with the observation first: what happened in the world, and then the record saying the
    attempt it happened in has been filed. The bytes are the offered ones either way, in one text
    item, never re-rendered.
    """
    if record.alongside is None:
        return record.message.visible_text
    return ToolResult(
        content=[
            *record.alongside.content,
            TextContent(type="text", text=record.message.visible_text),
        ]
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
    request: Union[PullRequest, InfoRequest, SealRequest]


@dataclass(frozen=True)
class _Offered:
    """A message the stream offered, and the attestation that will present it.

    A commit names the stream as it was before the event, so one built a second time describes
    a stream that has already moved and attests nothing about the delivery it was meant to
    attest. It is fixed once here and repeated afterwards.

    ``alongside`` is what this delivery travels with, where the message is one this gateway
    asked for on the model's behalf rather than one the model called for. The horizon filing is
    the case: the call that made it was an environment call, the observation it landed with
    exists nowhere else, and the acknowledgement goes back in the same result. So the two are
    carried together from here to the commit, and a delivery interrupted anywhere between still
    hands back both halves.
    """

    owner: bytes
    message: OfferedMessage
    commit: PresentationCommit
    alongside: Optional[ToolResult] = None


@dataclass(frozen=True)
class _PresentationUncertain:
    """An attestation that was sent and never acknowledged.

    The stream answers a repeated attestation with the acknowledgement it applied and with no
    other, so what finishes this is sending the same commit again rather than making a new one.
    """

    owner: bytes
    message: OfferedMessage
    commit: PresentationCommit
    alongside: Optional[ToolResult] = None


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
    alongside: Optional[ToolResult] = None


@dataclass(frozen=True)
class _HorizonOwed:
    """A filing this gateway owes on an attempt's behalf, and what the call that owes it landed
    with.

    The call is the environment call that spent the last of the budget. It changed a world, its
    observation exists nowhere else, and the attempt has no world work left, so the filing and
    the observation are one thing this call still has to finish. The request is frozen here
    before it is sent, for the reason every other request is: the acknowledgement it may mint is
    reachable through that request and through no other, and the same one sent again is a retry
    rather than a second filing.
    """

    owner: bytes
    request: SealRequest
    result: ToolResult


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
    _HorizonOwed,
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

    It also enforces the environment's step budget. The budget belongs to the environment, and
    the calls that spend it never reach the stream, so this is the only layer that can see the
    budget run out. When it does, the attempt is ended through the stream that owns it rather
    than inside the episode, which is what keeps the ending a fact the ledger holds. What has
    been spent of the budget is not this gateway's own count, though: the stream authorizes
    every one of those calls and counts them, and a gateway built over a generation somebody
    else was serving reads that count with the rest of what it routes by. A budget a process
    keeps is a budget the next process starts over.

    Which ending that is, the environment decides. Its default is the floor: the attempt is
    ended with the reason on it and nothing is filed, which is the honest answer for a budget
    that is a guard rather than a rule of the task. An environment whose horizon is an ending
    its own scorer answers for declares that instead, and then the call that spends the last
    step is the last call: this gateway files that environment's terminal for the attempt, the
    stream seals and grades it, and the acknowledgement goes back with the observation the call
    landed with. Nothing waits for a terminal the agent has no step left to reach.
    """

    def __init__(
        self,
        stream: StreamHandle,
        episode: ServedEpisode,
        spec: TaskSpec,
        terminal: ToolManifest,
        *,
        initial_cursor: str,
        generation: StreamStart,
        world_attempt: Optional[str] = None,
        open_episode: Optional[EpisodeOpener] = None,
        blobs: Optional[BlobStore] = None,
        environment: Optional[EnvironmentTerminal] = None,
        on_refusal: Optional[RefusalSink] = None,
    ) -> None:
        # The composition is required rather than optional, and this is why. A transport that
        # was never told what its generation declared cannot tell that from a generation which
        # declared nothing, and the two are served differently: the stream mints its records
        # from the composition and this transport counts calls against the episode, so a
        # replacement that guessed would advertise one number, enforce a second, and hand the
        # agent a third. Every way of getting one of these has the composition already, because
        # composing a generation returns it and resuming one is held to it, so the argument
        # costs a caller nothing and closes the gap for all of them.
        _check_declared_budget(spec, generation.budget)
        self._stream = stream
        # The composition this transport is serving, whether it started that generation or took
        # one over. What it declares is what the stream's own records carry.
        self._generation = generation
        # The world each live attempt is working in, and how an attempt gets one of its own. A
        # task is a fresh world: the seal captures what an attempt left behind, so a second
        # attempt in a world its predecessor worked in would file that predecessor's work. There
        # is one of these per live attempt rather than one for the transport, because a
        # generation may have several attempts live at once and a call names the attempt it is
        # for: a single current world would run an older attempt's calls in the newest world.
        #
        # ``episode`` is one of two things and the caller says which. Named by an attempt, it is
        # that attempt's world and nothing else's from here: a caller handing one over knows
        # which attempt it opened it for, and an unnamed world is one no rule here could pair
        # correctly, so the naming is asked for rather than guessed at. Unnamed, it is the seed
        # of a generation this transport is starting: it belongs to nobody until the first task
        # is presented, and it is that attempt's world from then on. The named one is paired
        # below, once the route it is paired in exists.
        self._worlds: Dict[str, ServedEpisode] = {}
        self._unclaimed: Optional[ServedEpisode] = episode if world_attempt is None else None
        self._open_episode = open_episode
        # What the environment answered when this generation was composed: the version its
        # acknowledgements declare their digests under, and the digest of what it is configured
        # as. The generation's identity was built from that answer, so it is what every world
        # opened afterwards is held to. It is ``None`` for a gateway that was never given one,
        # which committed to nothing about an environment and so holds nothing to anything.
        self._declared_environment = environment
        # Where the pairing of attempt to world is written down, for the environment that seals
        # by stopping the world an attempt worked in. It is recorded as each world opens, which
        # is the only moment both halves of the pair are in one place.
        self._route = WorldRoute() if environment is None else environment.route
        # A world handed over is written into both at once. The attempt it belongs to is a fact
        # the caller states, and every way of asking which world an attempt is in has to get the
        # same answer from it: the ordinary call routes through the map and the environment's own
        # terminal resolves the route, so a pairing written in one and not the other would work
        # the restored world and seal whatever the process before this one left behind.
        if world_attempt is not None:
            self._worlds[world_attempt] = episode
            self._route.record(world_attempt, episode)
        self._spec = spec
        self._terminal = terminal.name
        self._cursor = initial_cursor
        self._active: FrozenSet[str] = frozenset()
        # The env's step budget, and what each attempt has spent of it. The budget is the one
        # the contract publishes, so an env that declares none is served without a cap. Where the
        # generation declared a number of its own, that is the cap: it is the number every task
        # record hands the agent, the check above has already established that the two agree, and
        # reading it from the generation is what keeps them agreeing. The spending is the stream's
        # own count of the calls it granted, refreshed from it on every call like the cursor and
        # the active attempts are, and it starts empty for the same reason they start empty: what
        # this gateway has not been told yet, it does not know.
        self._step_cap = spec.horizon if generation.budget is None else generation.budget
        self._spent: Dict[str, int] = {}
        # What running the budget out does. The environment says, and it says it once, when the
        # generation is composed; a gateway given no answer serves the floor, which is what an
        # environment that never declared one gets.
        self._horizon_ending = (
            FLOOR_HORIZON if environment is None else environment.horizon_ending
        )
        # The ending each attempt that ran out of steps owes, composed by the call that found no
        # budget left and kept until that ending has landed.
        self._endings: Dict[str, FinalizeRequest] = {}
        self._transcript: List[bytes] = []
        self._blobs = blobs
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
        # Where the count goes as it changes, for a harness that wants it somewhere that
        # survives this process. Without one the count lives here and nowhere else.
        self._on_refusal = on_refusal

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
        """Count one refusal, tell whoever is keeping the count, and return the error.

        The sink is called here rather than anywhere later because here is the only place the
        count is certainly known to somebody: a transport is not always asked to stop, and one
        that is killed after answering a call runs nothing on its way out. So the number reaches
        the harness inside the call that made it, before the error the model will see goes back.
        """
        self._refusals += 1
        _LOG.info("protocol v2 refusal: %s", code)
        if self._on_refusal is not None:
            self._on_refusal(self._refusals)
        return _refusal(code)

    @property
    def generation(self) -> StreamStart:
        """The composition this transport is serving.

        A resume is held to what its new owner serves, so the value a later owner presents is a
        composition rather than anything the run directory recorded. Most of that composition is
        reproducible from the environment and the task, and the identifiers this generation
        minted are not, so a controller that let this call compose the generation for it needs
        the composition back to be able to take the generation over later.

        Every gateway has one. A transport handed a stream somebody else composed is handed that
        somebody's composition with it, because what the generation declares decides what this
        gateway advertises and what it enforces, and a transport that had to guess at either
        would serve a surface the generation never hashed.
        """
        return self._generation

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

    async def info(self, arguments: Dict[str, Any]) -> str:
        """Return the exact bytes of the stream's answer about its own queue."""
        key = _call_key(INFO_TOOL, arguments)
        operation = self._claim(key)
        return await self._accepted(operation, self._info(key, arguments))

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

        A stop that failed is the exception, and it is asked for again rather than answered with
        what went wrong the first time. What made it fail is a world this transport is still
        holding, and a replayed failure would mean nothing here ever released it: the record of
        the failed stop is dropped, so the next call runs the stop again over what is left. A
        caller that was cancelled leaves the stop where it is, because that stop is still running.
        """
        if self._shutdown is None:
            self._serving = _CLOSING
            self._shutdown = asyncio.ensure_future(self._stopped())
        try:
            await asyncio.shield(self._shutdown)
        except BaseException:
            shutdown = self._shutdown
            if shutdown is not None and shutdown.done() and not shutdown.cancelled():
                if shutdown.exception() is not None:
                    self._shutdown = None
            raise

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
        # A generation that sealed an attempt has already let go of the world it was in, so
        # stopping closes the worlds there are and not the ones there were.
        await self._close_worlds()

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

    async def _info(self, key: bytes, arguments: Dict[str, Any]) -> str:
        """Ask the stream how much of its queue there is, once it has nothing older to give.

        It is the pull's own path, step for step, because the answer is the same kind of thing: a
        record the stream minted, reserved for the request that asked, and delivered under an
        attestation. So an answer left owed by a lost call is collected here rather than asked for
        again, and a call that arrives while something else is owed is refused instead of taking
        it. The model's own call carries nothing, and an argument is a call this protocol does not
        define rather than a value to ignore.
        """
        if arguments:
            raise self._refuse("invalid_message")
        owed = await self._resumed(key)
        if owed is not None:
            return owed
        request = self._info_request(key)
        message = await self._decisive(key, self._stream.info(request))
        return await self._deliver(message, key)

    async def _terminal_call(self, key: bytes, arguments: Dict[str, Any]) -> str:
        """End the attempt named in the wrapper, and return the stream's answer.

        The call never reaches the environment from here. Sealing has to record its prepared
        state before any finalizer runs, so the environment's half of the terminal belongs
        inside the stream's transaction and not in front of it.

        A filing for an attempt whose world is not here is refused before it is built, which is
        where the routing says it belongs. Sending it would end the attempt: the environment's
        own terminal would find no world to read, its Activity would fail, and the stream would
        record the attempt as finally failed on a filing this transport should never have made.
        """
        attempt_id, native = self._unwrap(arguments)
        owed = await self._resumed(key)
        if owed is not None:
            return owed
        request = self._seal_request(attempt_id, native, key)
        result = await self._decisive(key, self._stream.seal(request))
        return await self._deliver(result, key)

    async def stream_state(self) -> StreamState:
        """Read the generation's state without changing it.

        Harness-only, like the Query it forwards. No tool reaches this, and the one tool that
        reaches anything like it reaches three counts the stream mints a record for: a model that
        could read the schedule's own counts would learn from them what a Wait is shaped to
        withhold.
        """
        return await self._stream.stream_state()

    async def close_queue(self) -> QueueClosed:
        """Close the generation's queue to insertion.

        This is a controller call and not a tool. It is reachable from the process that composed
        the generation and from nothing the model can reach. Closing is what makes Done
        reachable, so who does it and when is a decision about the run rather than about this
        transport.
        """
        return await self._sent(self._stream.close_queue())

    async def _environment(
        self, key: bytes, tool_name: str, arguments: Dict[str, Any]
    ) -> ToolResult:
        """Dispatch one ordinary environment tool against the routed attempt.

        ``attempt_id`` is stripped before dispatch, so the environment sees the arguments it
        declared and a native argument of that name would still be its own. What makes it the
        routed attempt is the stream saying so, and saying so at the moment of the call rather
        than at some earlier read: the stream decides this call and holds the generation until
        it settles, so an attempt that ended in between is not one it still reaches.

        The world is the named attempt's own, looked up rather than assumed. A generation may
        have several attempts live at once and the model works them in whatever order it likes,
        so the current world is not this call's world: dispatching into it would run an older
        attempt's action in a newer attempt's environment and file it under the wrong seal. An
        attempt with no world here is refused the way an attempt that is over is, because that is
        what it is: every world this transport holds is the world of an attempt it is serving.

        A call for an attempt already being worked when this transport was built is the exception,
        and it is where a replacement takes over the world it was handed. Such an attempt was
        started by the transport before this one, so no task presentation here ever paired it
        with a world, and this call is what says which attempt the handed world is for.

        The call that spends the last of the env's step budget is still an ordinary call: it is
        dispatched, it is committed, and its observation comes back, and the attempt is still
        the attempt. The budget is how many environment actions an attempt gets, and filing is
        not one of them: an environment that promises six moves and asks for a terminal after
        them would otherwise have every full-length play ended without a filing, unsealed and
        ungraded, on the move its own contract told the agent to make. So spending the last of
        the budget leaves the attempt terminal-call-only, and the call after that is the one with
        nothing behind it: it reaches no world, ends the attempt, and is refused.

        That is the floor, and it is what an environment gets by saying nothing. Where the
        environment declared its horizon a graded ending there is no call after: this one files
        the environment's terminal for the attempt as soon as its own step is committed, and the
        acknowledgement the stream mints comes back beside the observation.
        """
        attempt_id, native = self._unwrap(arguments)
        owed = await self._resumed(key)
        if owed is not None:
            return owed
        overspent = self._overspent_request(attempt_id)
        if overspent is not None:
            await self._budget_spent(overspent)
            raise self._refuse("invalid_attempt")
        stranded = self._horizon_request(attempt_id)
        if stranded is not None:
            # Nothing this transport was serving reaches here: the filing goes out with the call
            # that spends the last step. What does is a generation taken over from a process that
            # spent the budget and did not file, where the count came back from the stream and
            # the attempt is still open. It is finished before anything else, and this call gets
            # the acknowledgement, because there is no world work left for it to have done.
            return await self._horizon_sealed(_HorizonOwed(key, stranded, _EMPTY_RESULT))
        episode = self._world_of(attempt_id)
        held = _LeaseHeld(
            owner=key, call=EnvironmentCall(call_id=_opaque(), attempt_id=attempt_id)
        )
        self._recovery = held
        await self._granted(held)
        try:
            observation = await episode.call(tool_name, native)
        except BaseException:
            # The world's own failure is what its caller is told about. The release still goes,
            # and one that faults too leaves the grant where this call will find it again. The
            # release is also what lets a deadline this call outlasted be made, and the world of
            # an attempt that ended is let go of here as surely as on the way back with an
            # observation: a call that failed is still the call the ending happened inside.
            await self._given_back_quietly(held)
            await self._released_quietly()
            raise
        # Only the tool's own observation goes back. Feedback under this protocol is a
        # presented Payload, so a sidecar carrying it here would be a second channel that
        # no offer, presentation, or delivery count could see.
        result = ToolResult(content=observation.content)
        # The step is spent where the world changed. The stream counted the grant, so this is the
        # same step written down twice; keeping it here means the call after this one does not
        # have to ask again before it can refuse.
        self._spent[attempt_id] = self._spent.get(attempt_id, 0) + 1
        # The release is a durable operation and its answer can be lost, so the record holds the
        # observation until it has settled: a call that got no answer to its release is one
        # nothing else can free the generation for, and the observation exists nowhere else.
        landed = _LeaseHeld(owner=key, call=held.call, result=result)
        self._recovery = landed
        await self._given_back(landed)
        # And the filing this call owes, if that step was the last one. It goes after the grant
        # is back, because the stream holds the generation for the call it granted and would
        # refuse a filing sent while it does.
        spent = self._horizon_request(attempt_id)
        if spent is None:
            return await self._settled(key, result)
        return await self._horizon_sealed(_HorizonOwed(key, spent, result))

    async def _released_worlds(self) -> None:
        """Let go of what the generation ended while this transport was holding its grant.

        The generation holds an attempt's deadline back while a call is in that attempt's world:
        an ending cannot cancel what a world is already doing, so it waits for the grant to come
        back and is made as soon as it does. That is inside the call that was holding it. The
        attempt is over before that call answers, and the world it was working in is still
        running, so this is where it goes: the answer may be the last thing the agent ever asks
        for, and a world left for a call that never comes is a world nothing releases until this
        transport stops.

        Only a generation that declares a deadline is asked about. It is the one ending that can
        be made while this transport is inside a call: the step cap is this transport's own and
        closes its world where it makes it, and a seal is a call the model has not made yet. A
        generation that declares no deadline is served exactly as it was, with no read here.

        The read is a query. What it decides is whether a world may be let go of, which is the
        same question every other retirement asks, and the answer a generation gives about its own
        attempts is not one a writer has to be asked for. It is one read, so the ending the
        generation has not applied yet is not in it: that world is retired by the next call, the
        way every ending this transport is not inside is.
        """
        if not self._generation.attempt_deadline_ms:
            return
        await self._retired(await self._sent(self._stream.stream_state()))

    async def _released_quietly(self) -> None:
        """Let go of those worlds where the caller is already being told about something else.

        A call whose world failed is told about that failure and about nothing else, so a cleanup
        that fails here does not become the answer: what it could not close stays mapped, and the
        call that comes back retires it before it asks for anything.
        """
        try:
            await self._released_worlds()
        except Exception:
            pass

    async def _settled(self, key: bytes, result: ToolResult) -> ToolResult:
        """Hand back what an ordinary call landed with, once what it ended has been let go of.

        The observation is written down before the question is asked, so a cleanup that fails
        leaves it for the call that comes back rather than losing it to a world that would not
        stop. A generation with no deadline writes nothing down and asks nothing, which is the
        call it made before there were worlds to let go of here.
        """
        if not self._generation.attempt_deadline_ms:
            return result
        self._recovery = _ResultOwed(owner=key, result=result)
        await self._released_worlds()
        return result

    def _overspent_request(self, attempt_id: str) -> Optional[FinalizeRequest]:
        """The ending this call owes, if there is no step left for it to spend.

        The ending belongs to the call that had no budget rather than to the one that spent the
        last of it. An attempt out of environment steps is not an attempt that is over: it can
        still be filed, and under this protocol filing is a call to the stream rather than a step
        in the world. What is over is the world work, so the first call that would have taken a
        step it does not have is where the attempt ends, and it takes no step to do it.

        It is composed once and kept. One logical request is one ending however many times it
        has to be sent, so the call whose ending got no answer sends that ending again rather
        than composing a second one the stream would read as another. An attempt this transport
        has stopped serving is owed no ending at all: the one it owed has landed, and a call
        naming it now is refused by the stream the way every other call to an attempt that is
        over is.

        What has been spent is the stream's count and not this gateway's own, and the call
        reading it here has just refreshed it. So a replacement transport over a generation that
        already spent its budget refuses on its first call rather than on its horizon-plus-first,
        and the declared horizon stays a bound on the attempt rather than on the process.

        None of this is the answer where the environment declared its horizon a graded ending.
        Such an attempt was filed as its last step committed, so there is no call with nothing
        behind it to refuse, and nothing to floor.
        """
        if self._horizon_ending != FLOOR_HORIZON or not self._out_of_budget(attempt_id):
            return None
        ending = self._endings.get(attempt_id)
        if ending is None:
            ending = FinalizeRequest(
                request_id=_opaque(), attempt_id=attempt_id, reason=STEP_CAP
            )
            self._endings[attempt_id] = ending
        return ending

    def _out_of_budget(self, attempt_id: str) -> bool:
        """Whether this attempt is one this transport is serving with no step left to spend."""
        if self._step_cap is None or self._spent.get(attempt_id, 0) < self._step_cap:
            return False
        return attempt_id in self._active

    def _horizon_request(self, attempt_id: str) -> Optional[SealRequest]:
        """The filing this attempt owes because its world work is over, under a graded horizon.

        The filing carries no arguments. What the environment reads at its own horizon is the
        world the attempt left, which is where it stands when the last step is committed, and a
        generation whose terminal wanted arguments was refused a graded horizon before it served
        anything. So there is nothing here that an agent authored and nothing this gateway had
        to invent: the call names the attempt, says the horizon made it, and stops.

        It is composed once per call that owes one and written into the recovery record by the
        caller, which is what fixes the cursor it carries and what a retry of that call resends.
        A filing composed now and sent after something else moved the stream would be refused for
        a cursor it no longer has, so nothing composes one while a call still owes the last.
        """
        if self._horizon_ending != GRADED_HORIZON or not self._out_of_budget(attempt_id):
            return None
        return SealRequest(
            metadata=self._terminal_metadata(attempt_id),
            public_tool_name=self._terminal,
            native_terminal_name=self._terminal,
            terminal_source=HORIZON_FILED,
        )

    async def _horizon_sealed(self, record: _HorizonOwed) -> ToolResult:
        """File the terminal this attempt's horizon owes, and hand back both halves of the call.

        The record is installed before the filing goes, for the reason every other request is
        written down before it is sent: the acknowledgement it may mint is reachable through
        that request and through no other, the observation the call landed with exists nowhere
        else, and a fault anywhere in here leaves the one call that can still finish both.

        A refusal is different from a fault, and it is not raised as one. It says the stream did
        not take this filing and never will: the generation's own deadline reached the attempt
        first, or a filing under this request already landed and was presented. Either way the
        attempt is not this transport's to end any more, and the call that made it is still owed
        the observation it landed with.

        Which is not the same as being free to hand it over. A refusal is also what a transport
        the generation has been taken from is answered with, and a replacement that put the world
        back to the checkpoint made this observation the report of a mutation that no longer
        happened. Reading the refusal itself cannot tell those apart: an ending that won and a
        restoration that won both refuse this filing. So what is owed is written down and handed
        over the way every other kept result is, through a question this stream refuses a writer
        it has replaced.
        """
        self._recovery = record
        try:
            offered = await self._sent(self._stream.seal(record.request))
        except ToolError:
            owed = _ResultOwed(owner=record.owner, result=record.result)
            self._recovery = owed
            # The refusal says the attempt is over, so the world it was working in is over with
            # it. The question this hands over through is the one that answers where the
            # generation is, and the world goes on the strength of that answer.
            return await self._handed_over(owed.result)
        return await self._deliver(offered, record.owner, alongside=record.result)

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
        """Give the generation back for one call, and finish what that call still owes.

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

        One thing left open is closed by the stream rather than by the call that opened it. A
        filing is kept until its answer is collected, and a filing whose attempt has ended has
        no answer to collect, so it is closed here before the owner is compared. Leaving it
        would hold this transport against a message that does not exist, and every call, the
        owner's own retry included, would be refused for as long as the generation ran.
        """
        state = await self._sent(self._stream.stream_state())
        self._adopt(state)
        record = self._recovery
        if self._nothing_to_collect(record, state):
            record = self._recovery = _Idle()
        if record.owner is not None and record.owner != key:
            raise self._refuse("outstanding_response")
        if isinstance(record, _Idle):
            if state.pending_message_id is not None:
                raise self._refuse("outstanding_response")
            if self._closed:
                raise self._refuse("closed_stream")
            await self._retired(state)
            return None
        if isinstance(record, _RequestUncertain):
            return None
        if isinstance(record, _LeaseHeld):
            # The grant goes back first, because the stream is holding the generation for this
            # exact call and refuses everything else meanwhile. A call that landed hands over
            # what it landed with; one that never reached the world starts over.
            await self._given_back(record)
            if record.result is None:
                # A call that never reached the world starts over, and it starts over against a
                # generation the release may just have moved: the ending an attempt's deadline
                # owed is made as the grant goes back. So what that ending freed is let go of
                # before this call is allowed to ask for a world again, out of the state this
                # call arrived with and out of the one the release left behind.
                await self._retired(state)
                await self._released_worlds()
                return None
            # Giving the grant back is what lets the generation end an attempt whose deadline
            # fell due while the world was being called, so what this hands over is handed over
            # the way every other kept result is: the observation is written down first, so a
            # cleanup that fails leaves it for the call that comes back rather than losing it.
            self._recovery = _ResultOwed(owner=record.owner, result=record.result)
            return await self._handed_over(record.result, state)
        if isinstance(record, _HorizonOwed):
            if self._filing_settled(record.request, state):
                # The filing did not fail to be decided: it was decided, and what it committed
                # was the ending rather than an acknowledgement. Sending it again would ask the
                # stream a question it has already answered for good, and every call that is not
                # this one would go on being refused for as long as the generation ran. So what
                # is left is the observation the call landed with, kept the way every other
                # result this gateway holds is kept.
                self._recovery = _ResultOwed(owner=record.owner, result=record.result)
                return await self._handed_over(record.result, state)
            # A filing this gateway owes on the attempt's behalf, sent again as itself. The
            # stream reserves what a request offered for that request, so the retry reaches the
            # acknowledgement the first one may already have minted rather than filing twice.
            return await self._horizon_sealed(record)
        if isinstance(record, _ResultOwed):
            return await self._handed_over(record.result, state, closed_at=record.closed_at)
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
                alongside=record.alongside,
            )
            self._recovery = record
        return await self._present(record)

    def _nothing_to_collect(self, record: _Recovery, state: StreamState) -> bool:
        """Whether a filing whose answer never arrived has nothing left to come back for.

        Such a record is held open against the answer, and this is the case where the answer
        will not come and the record owes nothing else: it is dropped, and the next call is
        served. A filing kept beside an observation is the other case and is not this one, so
        it is asked about where it is handled rather than here.
        """
        if not isinstance(record, _RequestUncertain):
            return False
        return self._filing_settled(record.request, state)

    def _filing_settled(self, request: Any, state: StreamState) -> bool:
        """Whether the attempt this filing names has ended with no answer left to collect.

        A filing is held open because the acknowledgement it may have minted is reachable
        through that request and through no other. An attempt reported as ended minted none:
        what the seal committed was the ending, or something else ended the attempt before the
        filing reached the stream, and from then on every filing for it is refused, the exact
        one included.

        A message the stream is holding is the case where there is something, so that is asked
        first: an acknowledgement is offered against an attempt the seal took and a rejection
        against one that is still active, and neither of those is an attempt that has ended.
        """
        if not isinstance(request, SealRequest) or state.pending_message_id is not None:
            return False
        return state.attempts.get(request.metadata.attempt_id) == _FINAL_FAILED

    async def _retired(self, state: StreamState) -> None:
        """Close the worlds of the attempts the generation ended, before new work is asked for.

        A sealed attempt's world is retired as its acknowledgement is presented. An attempt the
        generation ended instead presents nothing, so that moment never comes: the ending is a
        fact about the generation, and the first this transport can hear of one is the state it
        reads at the top of every call. A deadline is that ending, and so is the step cap.

        The world is retired here, before the pull that could reserve anything. The ended
        attempt's capacity is free the moment it ends, so a pull made first would reserve the
        next task while the old world was still running, and a cleanup that fails would leave the
        stream holding a task no cleanup will ever release. A cleanup that fails here raises
        before a request is built, which leaves nothing offered and the same world still to
        close, and the call that comes back asks for it again.

        Every world this transport holds is asked about rather than one, because more than one
        attempt may be live and a deadline reaches whichever of them ran out of time. Each of
        them is tried whatever the others do, for the reason a stop tries each of them: a world
        whose close keeps failing would otherwise be the last one ever attempted, and every ended
        world behind it would stay running for as long as this transport did.
        """
        ended = [
            attempt_id
            for attempt_id in self._worlds
            if state.attempts.get(attempt_id) == _FINAL_FAILED
        ]
        self._raise_failures(await self._closed_each(ended))

    async def _handed_over(
        self,
        result: Any,
        state: Optional[StreamState] = None,
        closed_at: Optional[str] = None,
    ) -> Any:
        """Hand over a kept result, and retire what the generation ended while it was owed.

        A kept result is handed over by the call that comes back for it, and that call is the
        only entry this transport is certain to get: an agent given the observation it was owed
        may be finished with the attempt it came from. So the ending that happened while the
        result was outstanding is acted on here as well as where a call finds nothing owed. A
        graded horizon whose filing finally failed is the case this exists for: the attempt is
        over, the world it was working in is not, and only this call knows the observation exists
        to be handed over at all.

        The order is the one the ownership question fixes. The result goes through the check that
        a replacement has not taken the generation over, and the worlds are let go of after it
        answers: a transport that has been replaced hands nothing over, and it does not tear down
        a world on the way to finding out. What that check answers with is what the retirement
        reads, rather than the older state the calling path arrived with: an ending that lands
        while the confirmation is in flight is in the one and not the other, and it is exactly the
        ending this is here to act on.

        A cleanup that fails raises instead of the result, and what is kept keeps it: every path
        that reaches here has written the observation into the recovery record first, so the call
        that comes back asks for the cleanup again and hands over the same observation when it
        works.
        """
        confirmed = await self._confirmed(state, closed_at)
        settled = confirmed if confirmed is not None else state
        if settled is not None:
            await self._retired(settled)
        return result

    async def _owed(
        self,
        result: Any,
        state: Optional[StreamState] = None,
        closed_at: Optional[str] = None,
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

        ``state`` is the read the calling path already made, and a path that made none passes
        nothing. The only thing read out of it here is whether the generation closed at the
        cursor a Done was owed to, and a result that is not a Done has nothing to ask of it.
        """
        await self._confirmed(state, closed_at)
        return result

    async def _confirmed(
        self, state: Optional[StreamState], closed_at: Optional[str]
    ) -> Optional[StreamState]:
        """Ask the stream whether this transport may still hand anything over, and where it is.

        The question and its answer are one thing, so the answer is handed back rather than
        thrown away: it is the freshest word this transport will get about the generation, and
        what is decided after a hand-over, which world is over and may be let go of, has to be
        decided from it. The read the calling path made before it is older by exactly the time
        the confirmation took, which is the window an ending falls in.

        A Done that closed the generation is answered by nobody, for the reason :meth:`_owed`
        gives, and there is nothing to hand back.
        """
        if closed_at is not None and state is not None and state.generation_state == _DONE:
            if state.cursor == closed_at:
                return None
        confirmed = await self._sent(self._stream.confirm_state())
        self._adopt(confirmed)
        return confirmed

    def _adopt(self, state: StreamState) -> None:
        """Take the four facts this gateway routes by out of the stream that owns them.

        The cursor every request carries, the attempts an ordinary call may reach, and whether
        the generation is still open. An attempt is reachable exactly while the stream says it
        is active, which is the same condition under which the stream would accept work on it.

        The fourth is how much of the environment's budget each attempt has spent. The stream
        grants every call that spends one and counts the grants, which makes its count the one
        that survives this process: a gateway built over a live generation adopts what has been
        spent rather than deciding the attempt has spent nothing. The count is taken whole and
        not merged, because the stream is also where a restored checkpoint says the world is
        back where it started and the budget with it.
        """
        self._cursor = state.cursor
        self._closed = state.generation_state != _OPEN
        self._active = frozenset(
            attempt for attempt, value in state.attempts.items() if value == _ACTIVE
        )
        self._spent = dict(state.environment_calls)

    def _routed(self, attempt_id: str) -> None:
        """Refuse a call that does not name an attempt this transport is serving.

        Two things make it one. The generation says the attempt is active, and this transport
        holds the world that attempt is working in. The second is as much a part of it as the
        first: a filing ends an attempt, and one sent for an attempt whose world is somewhere
        else would end it here, on the strength of an environment that never read the world it
        was ending. So the refusal is made here rather than left to the seal to fail on.
        """
        if attempt_id not in self._active:
            raise self._refuse("invalid_attempt")
        self._world_of(attempt_id)

    def _world_of(self, attempt_id: str) -> ServedEpisode:
        """The world this attempt is being worked in here, or the refusal that there is none.

        Every world this transport holds is the world of an attempt it is serving: one it opened
        for a task it presented, or one it was handed under that attempt's name. An attempt with
        neither is one whose world is somewhere this process cannot reach, and the answer to it
        is the answer an attempt that is over gets.
        """
        episode = self._worlds.get(attempt_id)
        if episode is None:
            raise self._refuse("invalid_attempt")
        return episode

    async def _budget_spent(self, request: FinalizeRequest) -> None:
        """End the attempt that has no step left, through the stream that owns it.

        A refusal saying it is already over is the answer this was asking for and is not raised:
        it means the generation's own deadline reached the attempt first, or this exact request
        landed and its answer was lost, and neither is something the model may read as a tool
        failure.

        The world goes with the attempt, here rather than at whatever call comes next. The
        attempt is over the moment the ending lands, so what is left running is a world nothing
        may still do anything to, and a generation whose agent stops calling would hold it for as
        long as this transport stood. A cleanup that fails keeps the world where the next call
        will find it, exactly as every other cleanup here does.

        The routing handle is dropped after the ending has settled, and not before. An ending is
        a durable operation whose answer can go missing, and the call that owes one is the call
        that will be made again: an attempt this transport had already stopped serving would
        refuse that retry over the routing rather than finishing what it left open, and the
        ending would be owed by nobody. The request is the one the first call composed, so the
        retry is the same ending arriving again rather than a second one.
        """
        try:
            await self._stream.finalize(request)
        except Exception as error:  # noqa: BLE001 - the code decides, and anything else is a fault
            if protocol_error_code(error) not in _ALREADY_OVER:
                raise
        await self._close_world(request.attempt_id)
        self._active = self._active - {request.attempt_id}

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

    def _info_request(self, key: bytes) -> InfoRequest:
        """Return the request this info call is made under, and write it down before it is sent.

        It is written down for the reason a pull's is: the stream reserves its answer for the
        request that asked, so a call whose answer never arrives is finished by sending that same
        request again, and every other call is refused until it is.
        """
        record = self._recovery
        if isinstance(record, _RequestUncertain) and isinstance(record.request, InfoRequest):
            return record.request
        try:
            request = InfoRequest(request_id=_opaque(), last_presented_cursor=self._cursor)
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

    async def _deliver(
        self, message: OfferedMessage, key: bytes, *, alongside: Optional[ToolResult] = None
    ) -> Any:
        """Fix the attestation ``message`` travels under, then present it.

        A commit names the stream as it was before the event, so one built a second time
        describes a stream that has already moved and attests nothing about the delivery it was
        supposed to attest. It is built once, kept, and repeated.

        ``alongside`` is what these bytes are delivered with, where the call being answered
        asked for something else and this message came of it. The two travel together from here,
        so what the attestation says is still true of exactly one delivery: the bytes were handed
        to the transport, in the result of the call that produced them.
        """
        record = self._recovery
        if not isinstance(record, (_Offered, _PresentationUncertain)) or (
            record.message.message_id != message.message_id
        ):
            record = _Offered(
                owner=key,
                message=message,
                commit=await self._attestation(message),
                alongside=alongside,
            )
            self._recovery = record
        return await self._present(record)

    async def _attestation(self, message: OfferedMessage) -> PresentationCommit:
        """Build the one attestation that will present ``message``, from the stream itself.

        The transcript is installed in the blob store before it is referenced, because the
        stream verifies the reference and a hash of bytes nobody stored is a reference to
        nothing. Without a store the reference is still the hash of the same bytes, and the
        stream reports that it verifies nothing.
        """
        state = await self._sent(self._stream.stream_state())
        transcript_blob = self._install_transcript(message.visible_text)
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

    async def _present(self, record: Union[_Offered, _PresentationUncertain]) -> Any:
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

        The world this message opens or closes changes before the attestation rather than after
        it. A Presentation is durable the moment it is committed, so anything that can fail is on
        the side of that commit where failing costs nothing: the message stays where the stream
        is holding it, and the call that comes back for it starts over. What is left after the
        commit is bookkeeping that cannot fail.
        """
        await self._prepared(record.message)
        self._recovery = _PresentationUncertain(
            owner=record.owner,
            message=record.message,
            commit=record.commit,
            # What these bytes are delivered with is carried into the uncertainty too. The
            # delivery this record finishes is the same delivery, and the observation it goes
            # behind exists nowhere else, so a record that dropped it would answer the retry
            # with half of what the call landed with.
            alongside=record.alongside,
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
                owner=record.owner, message=record.message, alongside=record.alongside
            )
            raise
        self._applied(record.message, ack.cursor)
        landed = _landed(record)
        self._recovery = _ResultOwed(
            owner=record.owner,
            result=landed,
            # Done is the one message whose presentation ends the stream it was presented to, so
            # what it is owed to is recorded with the cursor it ended at rather than with a
            # promise that the stream can still be asked about it.
            closed_at=ack.cursor if record.message.kind == "done" else None,
        )
        return landed

    async def _prepared(self, message: OfferedMessage) -> None:
        """Finish the world this message moves before the Presentation that reports it.

        A presented task and a presented acknowledgement are the two moments a world opens and
        closes, and both reach something outside this process, so both can fail. They happen
        here, before the commit, and each of them repeats: a message this call could not
        prepare is still the message the stream is holding, and the call that comes back for it
        finds the world already opened or already closed and goes on to the commit.
        """
        if message.kind == "task" and message.attempt_id is not None:
            await self._open_world(message.attempt_id)
        elif message.kind == "seal_ack" and message.attempt_id is not None:
            # The attempt is sealed, so nothing may still be done to it, and the world it was
            # working in is what was sealed. A SealReject leaves the attempt where it was and
            # deliberately does not reach this branch. What closes is the sealed attempt's own
            # world: another attempt may be live beside it and still working in its own.
            await self._close_world(message.attempt_id)

    def _applied(self, message: OfferedMessage, cursor: str) -> None:
        """Move this transport the way presenting ``message`` moved the stream.

        Nothing here reaches anything outside this process, which is what lets it run after the
        Presentation is already durable: there is no state in which it half happened.
        """
        self._transcript.append(message.visible_text.encode("utf-8"))
        self._cursor = cursor
        if message.kind == "task" and message.attempt_id is not None:
            self._active = self._active | {message.attempt_id}
            # The budget is per attempt, so the count starts where the attempt does. The stream
            # starts it there too, on the same Presentation, and this is that fact arriving one
            # call early rather than a second opinion about it.
            self._spent[message.attempt_id] = 0
        elif message.kind == "seal_ack" and message.attempt_id is not None:
            self._active = self._active - {message.attempt_id}
        elif message.kind == "done":
            self._closed = True

    async def _open_world(self, attempt_id: str) -> None:
        """Give the attempt being presented a world of its own.

        A world belongs to the attempt it was opened for. The episode a generation was opened on
        belongs to the first attempt it serves, and every attempt after that one gets a world
        nothing has filed in yet: the seal captures what an attempt left behind, and a second
        attempt in the same world would be filing the first attempt's work a second time.

        Nothing is closed here. A task being presented says a new attempt has started and says
        nothing about the attempts already live: a generation at a capacity above one may have
        several, each still working in a world of its own, and closing the last one opened would
        take the world out from under whichever of them is still using it. A world is closed
        where its own attempt ends, which is the seal it is acknowledged by, the ending the
        generation made for it, or this transport stopping.

        The opener is told the attempt its world is for, and this is the only moment anything
        outside this gateway can learn it. Ordinary calls are routed here, but sealing is the
        environment's own, and an environment that seals by stopping a world it started is
        handed an attempt ID and has to find that attempt's world from it. So the pair is
        written into the route here, in both branches: the episode this generation was opened on
        belongs to the first attempt as surely as an opened one belongs to the next.

        A world that is not the environment this generation declared is refused instead. The
        declaration was taken from the episode this generation was opened on and folded into
        what the generation is, and an opener is free to answer with anything: an environment
        configured differently scores a filing against a different hidden rule, so a task worked
        in one would be graded under a configuration nothing committed to while the generation's
        own identity still named the first world's. The refusal comes before the world is routed
        or the task presented, because a world nothing can seal is better than a world sealed
        against the wrong rule, and what was opened is let go of rather than left running.
        """
        if attempt_id in self._worlds:
            return
        if self._claimed(attempt_id) is not None:
            return
        if self._open_episode is None:
            raise RuntimeError(
                "this generation has a task after the one its episode was opened for, and no "
                "way to open a world for it"
            )
        opened = await self._open_episode(attempt_id)
        _ended_by_the_stream(opened)
        started_as = self._declared_environment
        if started_as is not None:
            declared = environment_terminal(opened)
            if (
                declared.canonicalization_version != started_as.canonicalization_version
                or declared.configuration_digest != started_as.configuration_digest
            ):
                await _let_go(opened)
                raise RuntimeError(
                    "the world opened for this task is not the environment this generation was "
                    f"started as: it declares {declared.canonicalization_version!r} configured "
                    f"as {declared.configuration_digest!r}, and this generation was started as "
                    f"{started_as.canonicalization_version!r} configured as "
                    f"{started_as.configuration_digest!r}"
                )
        self._worlds[attempt_id] = opened
        self._route.record(attempt_id, opened)

    def _claimed(self, attempt_id: str) -> Optional[ServedEpisode]:
        """Give the attempt whose task is being presented the seed this transport was built on.

        The seed of a generation belongs to nobody: a generation that has served no task has no
        attempt for a world to have been opened for. The first task presented here is what claims
        it, and the pair is written into the route with it, exactly as an opened world's is: an
        environment that seals by stopping the world an attempt worked in has to be able to find
        that world, and the generation's own episode is the first attempt's world as surely as an
        opened one is the next attempt's.

        A presentation is the only thing that claims it. A world handed over for an attempt that
        was already being worked arrives named by that attempt and is never here to be claimed, so
        nothing this transport serves can take a world that is another attempt's, and no task
        starting here can inherit one an attempt before it worked in.
        """
        claimed = self._unclaimed
        if claimed is None:
            return None
        self._worlds[attempt_id] = claimed
        self._route.record(attempt_id, claimed)
        self._unclaimed = None
        return claimed

    async def _close_world(self, attempt_id: str) -> None:
        """Let go of the world one attempt is done with.

        Sealing is what makes the submission, so nothing this transport can still do to that
        world would be part of it. An environment that seals by stopping its own world has
        already stopped it; closing here is what releases everything the serving process was
        holding for it, and it is the same call the process that opened the episode would make.

        The world is named by its attempt, so what closes is that attempt's own and nothing
        else's. A generation may have several live at once, and the one being let go of is over
        while the others are still being worked.

        The reference is dropped after the close returns rather than before it. A cleanup that
        failed is one something can still reach, and closing an episode twice is the episode's
        own business, so the call that comes back for this message asks for it again. What is
        dropped is the routing as well as the reference: an attempt whose world is gone is an
        attempt with no world here, and that is the answer a seal and an ordinary call both need.
        """
        episode = self._worlds.get(attempt_id)
        if episode is not None:
            await _let_go(episode)
            self._route.forget(attempt_id, episode)
        self._worlds.pop(attempt_id, None)

    async def _closed_each(self, attempt_ids: Sequence[str]) -> List[Exception]:
        """Let go of each of these worlds, whatever the others do, and report what failed.

        A world is a process outside this one and any of them may refuse to stop. Stopping at the
        first refusal would strand every world behind it: the one that failed is kept for the call
        that comes back, and a world that never had a close attempted is a world nothing is coming
        back for, because the caller that would have retried it never learns it is there.
        """
        failures: List[Exception] = []
        for attempt_id in attempt_ids:
            try:
                await self._close_world(attempt_id)
            except Exception as error:  # noqa: BLE001 - every world is tried and all of it raised
                failures.append(error)
        return failures

    def _raise_failures(self, failures: Sequence[Exception]) -> None:
        """Raise what a best-effort cleanup could not do, all of it.

        One failure is raised as itself, so a caller waiting for the reason a world would not stop
        reads that reason. Several are raised together: a cleanup that reported one and dropped
        the rest would be the same silence in a smaller place.
        """
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise ExceptionGroup(
                "this transport could not let go of every world it was holding", list(failures)
            )

    async def _close_worlds(self) -> None:
        """Let go of every world this transport still holds, in the order they were opened.

        The seed this generation was opened on is one of them even where no task was ever
        presented: it belongs to nobody until an attempt claims it, and a stop that walked away
        from it would leave the world of a generation that never served a task running.
        """
        failures: List[Exception] = []
        unclaimed = self._unclaimed
        if unclaimed is not None:
            try:
                await _let_go(unclaimed)
                self._unclaimed = None
            except Exception as error:  # noqa: BLE001 - every world is tried and all of it raised
                failures.append(error)
        self._raise_failures(failures + await self._closed_each(list(self._worlds)))

    def _install_transcript(self, text: str) -> str:
        """Install everything presented so far, with ``text`` appended, and return its hash.

        Each entry is length prefixed, so no two transcripts of different messages hash alike
        by running one message's bytes into the next. The reference is the hash of exactly the
        bytes installed, which is what lets the stream check it by reading the store.
        """
        transcript = b"".join(
            length_prefixed(entry) for entry in self._transcript + [text.encode("utf-8")]
        )
        if self._blobs is None:
            return sha256(transcript).hexdigest()
        return self._blobs.put(transcript, media_type="application/octet-stream").sha256


async def _let_go(episode: ServedEpisode) -> None:
    """Release one world without ending the attempt that worked in it.

    An episode's own close decides an ending: a world that reaches it with its lifecycle still
    open is read as an episode that stopped without a seal, and an abort verdict is claimed,
    recorded durably and appended to the trace. Under this protocol that reading is never the
    right one. A terminal becomes the stream's terminal request and reaches no lifecycle here,
    so the attempt this world was sealed, scored and acknowledged for would get a second result
    saying it was aborted and worth nothing, beside the one the generation committed; and an
    attempt that was not sealed is ended by the generation rather than by the transport that was
    serving it. Everything else the close does, from waiting for what is in flight to releasing
    the session and tearing the env down, is what this is here for.
    """
    await episode.close(finalize=False)


def _ended_by_the_stream(episode: ServedEpisode) -> None:
    """Refuse a world that would end itself, whichever task this generation opened it for.

    The rule is the one :func:`open_gateway` states, and a generation opens a world for every
    task rather than only for its first, so every one of them is held to it here.
    """
    if episode.ends_on_horizon:
        raise ValueError(
            "this episode ends itself when its step budget runs out, and under this protocol "
            "the stream is what ends an attempt; start it with ends_on_horizon=False"
        )


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

    A native tool named ``pull`` or ``info`` is refused here rather than served: a control tool
    and an environment tool of the same name cannot both be reached, and the one that would lose
    is the one the protocol runs through. Both names are refused whatever this generation
    declares, because which names this protocol keeps for itself is a fact about the protocol,
    and an environment that could be served under one generation and not another would be an
    environment whose surface depended on a decision about the run.

    ``info`` is served only where the generation declares it. What is advertised is what was
    hashed, so a generation that declared nothing offers the tool set it offered before there
    was a second tool to offer.

    Every wrapped call is held to the native schema this server advertised for it, the terminal
    filing included, because the wrapper is where that schema is nested and nothing below it
    looks there. What happens to a call the gateway accepts is the gateway's: it runs each one
    to the end whatever becomes of the caller waiting on it.

    The control tool is advertised in the words the generation hashed, which is why the
    description is taken from the composition the gateway holds rather than from the constant.
    Every gateway holds one, and what it holds was checked against the episode it serves over
    before it was built, so the surface served here is the surface that generation hashed.
    """
    spec = gateway.spec
    served = wrapped_manifests(spec, terminal_manifest(spec))
    for manifest in spec.tools:
        if manifest.name in (PULL_TOOL, INFO_TOOL):
            raise ValueError(
                f"env tool name {manifest.name!r} collides with a stream control tool; an env "
                f"served under protocol v2 may not expose a tool named {PULL_TOOL!r} or "
                f"{INFO_TOOL!r}"
            )

    async def dispatch(tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        if tool_name == PULL_TOOL:
            return ToolResult(content=await gateway.pull(arguments))
        if tool_name == INFO_TOOL:
            return ToolResult(content=await gateway.info(arguments))
        gateway.check_native_arguments(tool_name, arguments)
        if tool_name == gateway.terminal_tool:
            return ToolResult(content=await gateway.terminal(arguments))
        return await gateway.environment(tool_name, arguments)

    server: FastMCP = FastMCP(name=name or f"shogym:{spec.env_name}")
    server.add_tool(
        build_tool(
            ToolManifest(
                name=PULL_TOOL,
                description=_pull_description(
                    gateway.generation.budget, gateway.generation.capacity
                ),
                input_schema=_PULL_SCHEMA,
            ),
            dispatch,
        )
    )
    if gateway.generation.info:
        server.add_tool(
            build_tool(
                ToolManifest(
                    name=INFO_TOOL,
                    description=_INFO_DESCRIPTION,
                    input_schema=_INFO_SCHEMA,
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
    start: Optional[StreamStart] = None,
    open_episode: Optional[EpisodeOpener] = None,
    run_directory: Optional[Union[str, Path]] = None,
    environment: Optional[EnvironmentTerminal] = None,
    on_refusal: Optional[RefusalSink] = None,
) -> StreamGateway:
    """Start a generation for ``episode`` and bind this transport as its one consumer.

    The claim secret is minted here and never leaves. It is what authentication amounts to at
    this layer: the stream binds whoever presents it first, and a second transport presenting
    anything else is refused before a message has been offered.

    ``start`` is the generation a controller composed, for a run whose manifest and schedule are
    decided above one episode. Without it this opens the one-task generation the episode
    implies. Either way the queue is left open: closing it is a controller's call rather than a
    consequence of a transport connecting.

    ``open_episode`` is how each task after the first gets its own world, and a generation with
    more than one task is refused without it. The alternative is a run that reports success
    while every task after the first was worked and scored in the world its predecessor sealed.
    It is called with the attempt its world is for, so a caller serving an environment that
    seals its own worlds can record which world each attempt filed in and end that attempt in
    the world it worked in. ``episode`` is the first attempt's world, so the route a caller
    builds from the opener covers every attempt after that one.

    ``environment`` is what the environment answered when it was asked how its attempts end
    (see :func:`environment_terminal`). It carries the version this generation's acknowledgements
    declare their digests were taken under, the digest of what the environment is, and the route
    the worlds this gateway opens are recorded into. An environment that seals by stopping its
    own world is refused a generation of more than one task without it, because the seal would
    otherwise have no way to find the world each task after the first was worked in.

    That answer is also what every world opened after the first is held to. It is folded into
    what this generation is, so a later world configured differently would have its task scored
    against a hidden rule this generation never committed to, under an identity that still names
    the first world's configuration. The gateway refuses such a world rather than serving it.

    That answer is applied to the generation a controller composed as well as to the one this
    call composes. Asking the environment is this gateway's own job and it is done here, so a
    composition made where the environment was not is completed rather than served as it
    arrived: one that kept this gateway's version would have its first terminal refused by the
    environment as a mismatch, after the world was worked, and one that carried no environment
    digest would let a resume under a different configuration pass as the same generation.

    ``run_directory`` is where this generation keeps the blobs its presentations reference and
    the manifest a later owner resumes it from. The directory is made before the stream starts,
    because a directory that already holds another generation refuses this one before there is
    a generation to refuse it for. The manifest goes in afterwards, once the stream it names
    exists: a manifest is what makes a directory a generation somebody can resume, and one
    written first would say that about a stream a crash left unstarted.

    Between those two the name is still written down. A run that dies after starting a stream
    and before recording it would otherwise leave an authority running that nothing points at:
    its identifier was minted here and here only, and the next attempt mints another. So the
    identifier goes into the directory first, and the attempt that finds one there ends the
    generation it names before starting its own. What it ends never had a consumer, because
    the manifest is written before this binds one.

    The composition this opened is on the gateway it returns. A resume presents what its new
    owner serves rather than what the directory recorded, so the directory alone is not enough
    to take a generation over: the identifiers a generation mints are its own, and a controller
    that let this call compose one has nowhere else to read them back from.

    ``on_refusal`` is where this transport's count of its own refusals goes as it changes. A
    refusal advances no protocol state, so the generation counts none of them and the model's
    transcript is the whole record of one; the count is the cross-check on that record, and it is
    worth nothing to a harness that no longer has it by the time the run is read. Given a sink,
    the number reaches it inside the call that issued the refusal, which is the only moment a
    transport that is killed rather than stopped is certain to reach.

    A composed generation whose declared budget is not this episode's is refused here, for the
    reason a composition is refused at the moment it is made: a controller composes where the
    episode is not, this call is where the two meet, and a number the agent reads that nothing
    enforces is worse than no number at all.

    Whether that generation declares the info tool is carried into the hash this call recomputes,
    for the reason the budget is: the surface a resume is held to has to be the surface being
    served, and a tool the model can call that the hash did not name would be a generation
    serving one thing and identified as another.

    An episode that ends on its own horizon is refused here. Under this protocol the stream is
    what ends an attempt, and an episode that seals and grades itself when a step budget runs
    out would end one behind the stream's back: the attempt would still be active, nothing the
    ledger holds would say the environment had been graded, and the terminal the model went on
    to call would seal a world that was already gone. The budget is still enforced: the gateway
    counts the calls that spend it and ends the attempt through the stream when it is gone.
    """
    _ended_by_the_stream(episode)
    spec = episode.describe()
    terminal = terminal_manifest(spec)
    grade = environment.grade if environment is not None else environment_grade(episode)
    composed = start or stream_start(
        spec,
        terminal,
        claim_hash=sha256(secrets.token_bytes(32)).hexdigest(),
        grade=grade,
    )
    _check_honest_over(composed, grade)
    _check_declared_budget(spec, composed.budget)
    if environment is not None:
        _check_graded_horizon(spec, terminal, environment)
        composed = replace(
            composed,
            canonicalization_version=environment.canonicalization_version,
            configuration_hash=_configuration_hash(
                spec,
                terminal,
                environment.configuration_digest,
                environment.horizon_ending,
                composed.budget,
                composed.capacity,
                composed.info,
            ),
        )
    if len(composed.tasks) > 1 and open_episode is None:
        raise ValueError(
            f"this generation has {len(composed.tasks)} tasks and one episode to serve them "
            "with; each task is worked in a world of its own, so composing more than one needs "
            "a way to open the next"
        )
    if len(composed.tasks) > 1 and environment is None and _seals_its_own_worlds(episode):
        raise ValueError(
            f"this generation has {len(composed.tasks)} tasks and its environment ends an "
            "attempt in the world that attempt worked in; without the terminal this gateway "
            "records those worlds into, every task after the first would be sealed against the "
            "first task's world"
        )
    identifier = workflow_id or f"stream/{_opaque()}/1"
    blobs: Optional[FilesystemBlobStore] = None
    if run_directory is not None:
        blobs = FilesystemBlobStore.under(run_directory)
        composed = replace(composed, blob_root=str(blobs.root))
        prepare_run_directory(run_directory)
        install_policies(blobs, composed)
        abandoned = staged_generation(run_directory)
        if abandoned is not None:
            await discard_stream(client, workflow_id=abandoned.workflow_id)
        stage_run_directory(
            run_directory,
            workflow_id=identifier,
            task_queue=STREAM_TASK_QUEUE,
            configuration_hash=configuration_hash(composed),
        )
    stream = await start_stream(client, composed, workflow_id=identifier)
    if run_directory is not None:
        create_run_directory(
            run_directory,
            workflow_id=identifier,
            task_queue=STREAM_TASK_QUEUE,
            configuration_hash=configuration_hash(composed),
        )
    receipt = await stream.claim_consumer(
        ConsumerClaim(
            consumer_id=consumer_id or _opaque(), claim_hash=composed.consumer_claim_hash
        )
    )
    return StreamGateway(
        stream,
        episode,
        spec,
        terminal,
        initial_cursor=receipt.initial_cursor,
        open_episode=open_episode,
        blobs=blobs,
        generation=composed,
        environment=environment,
        on_refusal=on_refusal,
    )


def _check_graded_horizon(
    spec: TaskSpec, terminal: ToolManifest, environment: EnvironmentTerminal
) -> None:
    """Refuse a graded horizon this gateway could not file for the attempt.

    Two things have to hold. The environment has to publish a budget, because a horizon that
    grades is an ending at a step count and an environment that declares none never reaches one:
    such a generation would say its attempts are graded at the horizon and floor nothing, seal
    nothing, and wait for a filing that only the agent can make.

    And its terminal has to be one this gateway can fill in. The filing made at the horizon is
    the attempt's world as it stands, so there is nothing for the gateway to put in the call, and
    a terminal that declares arguments is one whose filing the agent authors. Such a call composed
    here would carry names with nothing behind them, and the stream would reject it after the
    budget was already spent.
    """
    if environment.horizon_ending != GRADED_HORIZON:
        return
    if spec.horizon is None:
        raise ValueError(
            f"env {spec.env_name!r} says its horizon is a graded ending and publishes no step "
            "budget, so there is no horizon for an attempt over it to be graded at"
        )
    declared = declared_argument_names(terminal.input_schema)
    if declared:
        raise ValueError(
            f"env {spec.env_name!r} says its horizon is a graded ending and its terminal "
            f"{terminal.name!r} takes {declared}, and the filing made at a horizon is the world "
            "the attempt left rather than arguments this gateway could write for it"
        )


def _check_honest_over(composed: StreamStart, grade: GradeIdentity) -> None:
    """Refuse a composition whose declared grader is not the one it is being opened over.

    A controller composes a generation where the environment is not, and this call is where the
    two meet. The composition carries the grader it was built over, that claim is inside the
    generation's identity, and the honest body publishes what that grader produced. So the claim
    has to be the environment's own, whole: a generation composed for one grader and opened over
    another commits the second grader's numbers under the first one's name, and its record and
    its bodies then describe measurements nobody took.

    Equality is the check rather than compatibility, because every field of the identity is a
    fact the record depends on: which grader, which version of it, whether its number is the
    environment's own, which measure the headline is, and which numbers it may publish beside it.
    A generation that resolved nothing to a policy publishing the grade is making no claim, and
    the stand-in it was composed over is caught here as the stand-in it is.
    """
    if composed.profile == LEGACY:
        # A start carrying no profile is a decoded history rather than a composition, and it
        # claims no grader to be held to. What refuses to create one is the call that starts a
        # generation, which is where a run created now is told to say what it delivers.
        return
    declared = composed.grade or KERNEL_STAND_IN_GRADE
    if declared != grade:
        raise ValueError(
            f"this generation was composed over {declared.grader_id}/{declared.grader_version} "
            f"and is being opened over {grade.grader_id}/{grade.grader_version}, and what an "
            "honest body publishes is the grade the environment it ran in produced"
        )
    if not grade.stand_in:
        return
    for row in composed.dispositions:
        policy = POLICIES.get(row.policy_digest or "")
        if row.kind == DELIVER and policy is not None and policy.exposure == HONEST:
            raise ValueError(
                f"this generation delivers {policy.policy_name}, which publishes the "
                f"environment's grade, and this environment is scored by {grade.grader_id}, "
                "which is a stand-in"
            )


def install_policies(blobs: FilesystemBlobStore, composed: StreamStart) -> None:
    """Install the preimage of every policy this generation names, and prove it went in.

    A digest says that something was hashed. What an audit needs years later is the something,
    so the bytes the digest names go into the run's own store beside the blobs its presentations
    reference, and a reader with the directory can say what a body was allowed to contain
    without this package being installed.

    They are required objects rather than a convenience. The generation counts each of them
    among the references its history cites, so a later owner reads them back before it may
    continue the run, and a store that cannot produce one is a directory whose record says what
    its bodies were allowed to contain and cannot produce the saying. So the write is checked
    here as well: an object installed under a name other than the one this generation resolved
    would satisfy nothing that reads it back.

    The set covers every cell of every matched family this generation declares as well as the
    rows it serves. A leg builds one cell of an arm and never renders the counterpart it is
    matched against, so a directory holding only the served descriptor can say what this body
    was allowed to contain and not what the comparison was between.
    """
    for digest in descriptor_digests(composed.dispositions, composed.families):
        policy = POLICIES.get(digest)
        if policy is None:
            continue
        installed = blobs.put(policy_preimage(policy), media_type="application/json")
        if installed.sha256 != digest:
            raise ValueError(
                f"the preimage of {policy.policy_name} installed under {installed.sha256[:16]} "
                f"and this generation resolved to {digest[:16]}"
            )


def _seals_its_own_worlds(episode: ServedEpisode) -> bool:
    """True iff this episode's environment brings its own terminal."""
    return getattr(episode.env, "protocol_v2_terminal", None) is not None


def environment_terminal(episode: ServedEpisode) -> EnvironmentTerminal:
    """Ask this episode's environment how the attempts of a generation over it end.

    The environment is asked rather than looked up by name. One that seals and grades for itself
    answers with its own canonicalization version, its own Activities and a digest of what it is
    configured as; every other one keeps the kernel's stand-ins, which compute from the
    terminal's arguments and reach nothing. The stream is the same either way: it is the
    environment's half of the terminal that is replaced, and the seal transaction around it that
    is not.

    The route is minted here and handed to both sides. The environment holds it because it is
    what a seal resolves an attempt through, and the gateway holds it because opening a world is
    the only moment an attempt and a world are in one place. Nothing is in it yet: a generation
    that has served no task has opened no world for anything to be sealed against.
    """
    route = WorldRoute()
    brings_own = getattr(episode.env, "protocol_v2_terminal", None)
    # Asked on both branches, because the answer is refused where it does not go with a terminal
    # of the environment's own: an environment graded by the kernel's stand-in cannot come to
    # claim a grader by declaring one on the half of the boundary that does not produce numbers.
    grade = environment_grade(episode)
    ending = environment_horizon_ending(episode)
    if brings_own is None:
        return EnvironmentTerminal(
            CANONICALIZATION_VERSION, list(kernel_activities()), None, route, grade, ending
        )
    version, activities, digest = brings_own(route)
    return EnvironmentTerminal(version, list(activities), digest, route, grade, ending)


def environment_horizon_ending(episode: ServedEpisode) -> str:
    """Ask this episode's environment what running out of steps does to an attempt.

    An environment that says nothing gets the floor, and the default is the conservative one for
    the reason the grade's is: a horizon that files the world as it stands commits a number
    against a world nobody said was finished, and an environment cannot come to have its
    unfinished attempts scored by forgetting to answer.

    An environment that says its horizon is a graded ending and brings no terminal of its own is
    refused here. The filing such a horizon makes is the environment's own seal reading the world
    as it stands, and an environment with no seal has nothing to read it with: what would be
    committed is the kernel's stand-in over an empty filing, which is the floor with a score
    written on it. The refusal names the half that is missing.
    """
    declared = getattr(episode.env, "protocol_v2_horizon_ending", None)
    if declared is None:
        return FLOOR_HORIZON
    ending = declared()
    if ending not in HORIZON_ENDINGS:
        raise ValueError(
            f"this environment says its horizon ends an attempt {ending!r}, and what a "
            f"generation is served under is one of {HORIZON_ENDINGS}"
        )
    if ending == GRADED_HORIZON and not _seals_its_own_worlds(episode):
        raise ValueError(
            "this environment says its horizon is a graded ending and brings no terminal of "
            "its own, so the filing made at that horizon would be the kernel's stand-in over "
            "an empty submission rather than this environment reading the world as it stands: "
            "an environment whose horizon grades declares protocol_v2_terminal beside "
            "protocol_v2_horizon_ending"
        )
    return ending


def environment_grade(episode: ServedEpisode) -> GradeIdentity:
    """Ask this episode's environment what its grader is.

    An environment that says nothing is scored by the kernel's stand-in, and that is what comes
    back. The default is the conservative one on purpose: an environment cannot come to publish
    its score to an agent by forgetting to answer, only by saying what its grader is.

    An environment that says its grader is its own and brings no terminal of its own is refused
    here. The two halves are one fact: an attempt over such an environment is sealed and graded
    by the kernel's Activities, so the number a generation over it commits is a fact about the
    shape of a filing whatever the environment declares. A generation composed over that claim
    would be honest by its record and would fail every seal, because the identity arriving with
    the score is the stand-in's. The refusal names the half that is missing.
    """
    declared = getattr(episode.env, "protocol_v2_grade", None)
    if declared is None:
        return KERNEL_STAND_IN_GRADE
    grade = declared()
    if not isinstance(grade, GradeIdentity):
        raise ValueError(
            f"this environment answered its grader with {type(grade).__name__}, and what a "
            "generation is built over is a declared grade identity"
        )
    if not grade.stand_in and not _seals_its_own_worlds(episode):
        raise ValueError(
            f"this environment says it is scored by {grade.grader_id}/{grade.grader_version} "
            "and brings no terminal of its own, so its attempts are sealed and graded by the "
            "kernel's stand-in and no number this environment took would reach a seal: an "
            "environment whose grade is its own declares protocol_v2_terminal beside "
            "protocol_v2_grade"
        )
    return grade


async def run_stdio_v2(
    env_name: str,
    *,
    task: Optional[Union[int, str]] = None,
    trace_path: Optional[Union[str, Path]] = None,
    run_directory: Optional[Union[str, Path]] = None,
) -> None:
    """Serve one environment under protocol v2 over stdio, durably.

    The service, the Worker, and the stream all belong to this process, so a harness spawns one
    command and gets a durable stream without installing or starting anything. Given a run
    directory it also leaves behind what a later owner needs to take the generation over: the
    blobs its events reference, the manifest saying which generation this was, and the history
    the service wrote, all three in that directory. Given none, the history lives and dies with
    this process, along with everything else there would have been to resume.

    The world's durable finalization store is rooted in that directory as well. A record an
    episode leaves about how it ended is then inside the run somebody reads rather than in a
    store shared with every session this machine has served, and the startup pass that resolves
    what a crash left dangling reads the run's own records rather than that shared store.

    The gateway is stopped before the Worker and the service are, because stopping it settles
    whatever call was accepted when the transport went away, and that call may still need the
    stream. Stopping it closes the episode, so the episode is closed here only when there was
    no gateway to do it or its stop did not finish.
    """
    episode = await ServedEpisode.start(
        env_name,
        task=task,
        trace_path=trace_path,
        ends_on_horizon=False,
        run_directory=run_directory,
    )
    stopped = False
    try:
        environment = environment_terminal(episode)
        async with durable_client(run_directory=run_directory) as client:
            async with stream_worker(client, activities=environment.activities):
                gateway = await open_gateway(
                    client,
                    episode,
                    run_directory=run_directory,
                    environment=environment,
                )
                # This command is the controller as well as the transport, and its manifest is
                # complete the moment it is built: one episode, one task. So it closes the
                # queue before the model can pull, which is what makes Done reachable once
                # that task has been sealed, acknowledged, and paid out.
                await gateway.close_queue()
                try:
                    await build_gateway_server(gateway).run_async(transport="stdio")
                finally:
                    await gateway.aclose()
                    stopped = True
    finally:
        if not stopped:
            # Released rather than ended, like every other world this command lets go of. What
            # became of an attempt is the generation's to say, and this is the path where there
            # was no generation to say it or the gateway's own stop did not finish.
            await _let_go(episode)
