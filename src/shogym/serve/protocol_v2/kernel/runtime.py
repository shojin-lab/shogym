"""Reaching the stream from outside it: the service, the Worker, and the caller's handle.

The service is embedded. A user installs the package and runs; nothing here asks them to
install a server, start one, or keep one alive. The dev service is downloaded once into
``~/.cache/shogym/temporal`` and every embedded service since runs that same binary.

Its database is not shared. One SQLite file under the cache root would make two serving
processes on one machine contend for one write lock, and the second one to ask would fail to
start. So the file belongs to the run: a generation given a run directory keeps its history in
that directory, beside the blobs and the manifest naming the same generation, and a generation
given no directory keeps it in a directory this process owns and removes when it exits.
``SHOGYM_TEMPORAL_ADDRESS`` points at a server someone else runs, and then nothing is
downloaded or started here at all.

:class:`StreamHandle` is the surface a gateway calls. It derives each Update ID from the
logical request and its canonical identity, which is what makes a transport retry reach the
same Update instead of a second one, and it turns an offered message plus its blobs into the
presentation the workflow will verify.

A handle is also an ownership claim. :func:`start_stream` claims the generation it starts and
:func:`resume_stream` claims one that is already running, fencing whoever held it; the token
each of them mints lives in the handle and travels with every call that can change the stream.
:func:`resume_run_directory` is the same thing from a directory, held to the composition the
resuming process serves rather than to the one the directory recorded, and it is where a
version one run is refused before anything is claimed.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from google.protobuf.duration_pb2 import Duration
from temporalio.api.common.v1 import Payload
from temporalio.api.namespace.v1 import NamespaceConfig
from temporalio.api.workflowservice.v1 import (
    DescribeNamespaceRequest,
    UpdateNamespaceRequest,
)
from temporalio.client import Client, WorkflowHandle, WorkflowUpdateFailedError
from temporalio.converter import default as default_converter
from temporalio.exceptions import ApplicationError
from temporalio.service import RPCError, RPCStatusCode
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from shogym.serve.protocol_v2 import (
    InfoRequest,
    PresentationAck,
    PresentationCommit,
    PullRequest,
    info_request_identity,
    presentation_request_identity,
    pull_request_identity,
    terminal_request_identity,
)
from shogym.serve.protocol_v2.kernel.activities import fork_activities, kernel_activities
from shogym.serve.protocol_v2.kernel.messages import (
    AnsweredUpdate,
    AttemptFinalized,
    CheckpointEvidenceAnswer,
    ChildReady,
    ConsumerClaim,
    ConsumerReceipt,
    EnvironmentCall,
    EnvironmentLease,
    FORK_ABANDONED,
    FORK_EXPIRED_AUTHORITY,
    FORK_IN_FLIGHT,
    FORK_REFUSALS,
    FORK_REQUEST_CONFLICT,
    FORK_UNREADABLE_PARENT,
    FinalizeRequest,
    finalize_request_identity,
    ForkReceipt,
    ForkRequest,
    ForkStatusAnswer,
    ForkStatusQuestion,
    OfferedMessage,
    OwnershipClaim,
    OwnershipReceipt,
    QueueClosed,
    RETRYABLE_FORK_REFUSALS,
    SealRequest,
    StreamStart,
    StreamState,
    Writer,
    answer_window_ends,
    check_child_ready,
    check_fork_receipt,
    check_fork_request,
    check_prepared_fork,
    check_transmitted_size,
    configuration_hash,
    fork_request_digest,
)
from shogym.serve.protocol_v2.kernel.workflow import (
    FORK_BARRIER,
    FORK_RETENTION_FLOOR_MS,
    ORIGIN_UNVERIFIED,
    TURNOVER_PENDING,
    TURNOVER_PAYLOAD_CEILING_BYTES,
    ForkRefused,
    StreamProtocolError,
    StreamWorkflow,
)
from shogym.serve.protocol_v2.policy import LEGACY
from shogym.serve.protocol_v2.rundir import RunDirectory, ResumeRefused, open_run_directory

TEMPORAL_ADDRESS_ENV = "SHOGYM_TEMPORAL_ADDRESS"

# How a call waits out a generation that is between executions. The generation rejects an
# arriving Update once it has decided to continue as new and until the boundary is behind it,
# and the wait here is what makes that decision invisible: the same request goes again, under
# the same Update ID, and reaches whichever execution is current.
#
# The boundary is one activation away in the ordinary case, so the first wait is short and the
# backoff is gentle.
_TURNOVER_FIRST_WAIT = 0.02
_TURNOVER_LONGEST_WAIT = 0.5

# What the wait is bounded by, which is not the clock. A boundary can legitimately take a long
# time: the grant a world call holds blocks it until that call comes back, and an agent's tool
# call runs for as long as the agent's work does, minutes at a stretch; a claim reading the
# store blocks it for as long as the read and its retries take. A wait bounded by elapsed time
# would give up on all of those and hand the caller a fault where there was only work in
# progress, which is the wedge this change exists to remove rather than a smaller version of it.
#
# So the bound is the boundary's own liveness. While the generation says it is latched and names
# something it is waiting for, or while its projection is moving, the call keeps waiting: the
# generation is making progress towards the boundary and the request will be admitted when it
# gets there. The clock below starts only when the generation says it is latched, names nothing
# it is waiting for, and has not moved. That is a generation that has stopped, and a call that
# waited on it for ever would be a hang with nothing recorded anywhere.
#
# What a caller gets when that clock runs out is the failure it already had: the rejection, as
# the fault it arrives as, carrying no protocol code. The transport raises it rather than mapping
# it to anything, so nothing new reaches the agent. What a run then gets recorded as is not this
# module's to say: the rule that reads a fault and decides whether a run completed lives with the
# launcher, and the change that makes it require a delivered Done is a separate one.
_TURNOVER_STALLED_AFTER = 300.0

# How long the journal read is given. It is asked only by a caller that already holds a failure,
# and what it can add is an answer that failure hid. So it is bounded rather than open ended: a
# deployment whose Workers cannot answer a Query leaves the caller with the failure it had, at the
# speed of this timeout, instead of holding it for as long as the read takes to be given up on.
_ANSWER_READ_TIMEOUT = timedelta(seconds=10)

# And how many times one of the fork's reads is asked before the failure it met is the answer. A
# Query is served by a Worker replaying the generation, so a question that arrives while that
# generation is closing is left with the Worker that has just evicted it: nothing answers it and
# the deadline above is what ends it, while the very next ask is routed afresh and answered at
# once. A read that got no answer is therefore asked again rather than reported as a parent nobody
# can read, which is the difference between a lost question and an absent Worker.
_ANSWER_READ_TRIES = 3
STREAM_TASK_QUEUE = "shogym-stream-v2"

#: The file one run's embedded service keeps that run's history in.
STREAM_DATABASE_FILE = "stream.sqlite"


def temporal_home() -> Path:
    """The directory the embedded service's downloaded binary lives in.

    The binary is the shared part: it is fetched once and every run after that starts it again.
    A run's database is per run, and :func:`_stream_database` says where.
    """
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "shogym"
    return root / "temporal"


@contextmanager
def _stream_database(run_directory: Optional[Union[str, Path]]) -> Iterator[Path]:
    """Yield the file this run's embedded service writes its history to.

    A run directory holds it, so the history sits beside the blobs and the manifest that name
    the same generation and a later owner finds all three together. Without a run directory
    there is nothing to take over later, so the file goes in a temporary directory this process
    owns and goes away with it.
    """
    if run_directory is not None:
        root = Path(run_directory)
        root.mkdir(parents=True, exist_ok=True)
        yield root / STREAM_DATABASE_FILE
        return
    with TemporaryDirectory(prefix="shogym-stream-") as scratch:
        yield Path(scratch) / STREAM_DATABASE_FILE


@contextmanager
def _service_output_off_the_wire() -> Iterator[None]:
    """Point this process's descriptor 1 at its standard error while the service is spawned.

    The embedded service is another program, and it prints a banner naming its address and its
    database when it starts. It inherits this process's descriptors, so on a server whose
    standard output is the protocol wire that banner arrives in the middle of the transport's
    framing and a strict client rejects it before the handshake.

    The descriptor is swapped rather than ``sys.stdout``, because the writer never sees a Python
    object. The child keeps whatever descriptor 1 named when it was spawned, so restoring this
    one afterwards leaves the service talking to standard error for the rest of its life and
    leaves this process talking to the wire.
    """
    sys.stdout.flush()
    try:
        saved = os.dup(1)
    except OSError:  # pragma: no cover - a process with no descriptor 1 has no wire to protect
        yield
        return
    try:
        os.dup2(2, 1)
        yield
    finally:
        os.dup2(saved, 1)
        os.close(saved)


@asynccontextmanager
async def durable_client(
    *, namespace: str = "default", run_directory: Optional[Union[str, Path]] = None
) -> AsyncIterator[Client]:
    """Yield a client for the durable service, starting an embedded one if needed.

    ``run_directory`` is the run this service serves. It is where the history goes, and passing
    the same directory the gateway is given is what keeps two serving processes on one machine
    out of each other's database.

    It is also what a resume is opened against. A client opened without one starts an empty
    database of its own, so it reaches no generation any other process served, and a resume it
    is handed to answers that the workflow does not exist. Taking a run over means opening its
    directory here first, and then resuming out of it.

    The download directory has to exist before the service is asked for, so it is created
    here rather than left to the first user who has never run this.
    """
    address = os.environ.get(TEMPORAL_ADDRESS_ENV)
    if address:
        yield await Client.connect(address, namespace=namespace)
        return
    home = temporal_home()
    home.mkdir(parents=True, exist_ok=True)
    with _stream_database(run_directory) as database:
        with _service_output_off_the_wire():
            environment = await WorkflowEnvironment.start_local(
                namespace=namespace,
                dev_server_database_filename=str(database),
                download_dest_dir=str(home),
            )
        try:
            await _the_retention_a_fork_is_answerable_for(environment.client)
            yield environment.client
        finally:
            await environment.shutdown()


async def _the_retention_a_fork_is_answerable_for(client: Client) -> None:
    """Keep this run's own histories for as long as a fork can still be asked about them.

    A fork's answers stand until an absolute horizon the barrier recorded, and answering a
    question inside that window needs the history the question is about: the parent's, and each
    child's own original execution, which the service schedules for deletion from that execution's
    own close time rather than from the parent's. The service starts with less than that, so a
    child that stopped early would be deleted while its parent was still prepared and still
    answerable, and a run would lose the evidence its own contract says it can be asked for.

    So the floor is configured here, where the run's service is started, rather than declared and
    hoped for. It is a floor and never a ceiling: a service already keeping more keeps it, because
    what a deployment keeps beyond this is a deployment's business and lengthening it changes no
    answer this build gives.

    Only the embedded service is configured. An address someone else runs is that operator's, and
    a package that rewrote a namespace it was merely pointed at would be changing a deployment
    nobody asked it to change.
    """
    service = client.service_client.workflow_service
    described = await service.describe_namespace(
        DescribeNamespaceRequest(namespace=client.namespace)
    )
    owed = timedelta(milliseconds=FORK_RETENTION_FLOOR_MS)
    if described.config.workflow_execution_retention_ttl.ToTimedelta() >= owed:
        return
    keeping = Duration()
    keeping.FromTimedelta(owed)
    await service.update_namespace(
        UpdateNamespaceRequest(
            namespace=client.namespace,
            config=NamespaceConfig(workflow_execution_retention_ttl=keeping),
        )
    )


# The workflow sandbox reimports every module a workflow reaches, which for `shogym` means
# reimporting the whole package and everything the env registry pulls in behind it. That is
# both expensive and fragile: an unrelated dependency's import hook does not survive being run
# a second time inside the sandbox. `shogym` is passed through instead. What the sandbox
# defends against is nondeterminism, and the protocol code it would be guarding has no clock,
# no randomness, no I/O, and no mutable module state.
_RESTRICTIONS = SandboxRestrictions.default.with_passthrough_modules("shogym")


def stream_worker(
    client: Client,
    *,
    task_queue: str = STREAM_TASK_QUEUE,
    activities: Optional[Sequence[Any]] = None,
    cached_workflows: Optional[int] = None,
) -> Worker:
    """Return a Worker that serves stream workflows on ``task_queue``.

    ``cached_workflows`` sizes the sticky cache. Zero turns it off, which makes the server
    route every task to whichever Worker is free and makes every task replay the history from
    the beginning. That costs throughput and buys a stream that keeps serving the moment a
    Worker is replaced, which is the trade a short run wants.
    """
    served = _registered(activities)
    runner = SandboxedWorkflowRunner(restrictions=_RESTRICTIONS)
    if cached_workflows is None:
        return Worker(
            client,
            task_queue=task_queue,
            workflows=[StreamWorkflow],
            activities=served,
            workflow_runner=runner,
        )
    return Worker(
        client,
        task_queue=task_queue,
        workflows=[StreamWorkflow],
        activities=served,
        workflow_runner=runner,
        max_cached_workflows=cached_workflows,
    )


def _registered(activities: Optional[Sequence[Any]]) -> list:
    """Return what a Worker serves: what it was handed, and the fork's own beside it.

    A Worker takes whatever Activity list it is given wholesale, and an environment that brings its
    own terminal hands over only its seal, its grade, the payload bundle Activity and the blob
    verification Activity. The fork's four are none of those and a caller composing a list has no
    reason to know about them, so they are added here rather than left to be remembered. A caller
    that supplied one of them itself keeps its own, because the SDK refuses two Activities of one
    name in a Worker.

    A caller handing an empty list is saying the other thing: this Worker serves no Activity at
    all. That is a read, replaying a generation to answer a Query beside the Worker already
    serving the run, and an Activity registered there would make it a second Worker offering to
    work the run's task queue, which the SDK refuses.
    """
    if activities is not None and not activities:
        return []
    served = list(activities) if activities is not None else kernel_activities()
    named = {getattr(one, "__temporal_activity_definition").name for one in served}
    return served + [
        one
        for one in fork_activities()
        if getattr(one, "__temporal_activity_definition").name not in named
    ]


def stream_replayer() -> Replayer:
    """Return a Replayer that runs saved histories through the current workflow code.

    A history that no longer replays is a deployment that would lose the streams already
    open, so this belongs in a check that runs before one, not only in a test.
    """
    return Replayer(
        workflows=[StreamWorkflow],
        workflow_runner=SandboxedWorkflowRunner(restrictions=_RESTRICTIONS),
    )


async def run_stream_worker(*, task_queue: str = STREAM_TASK_QUEUE) -> None:
    """Serve stream workflows until cancelled, on an embedded service or a named one."""
    async with durable_client() as client:
        await stream_worker(client, task_queue=task_queue).run()


def refuse_a_carried_projection(start: StreamStart) -> None:
    """Refuse a generation somebody is trying to create already holding a projection.

    A carried projection is how one execution of a generation hands itself to the next, and it
    is legal exactly there. A caller composing one into a start would be creating a generation
    that has already served: a cursor past its own beginning, scores nothing filed, messages
    nothing presented, and an ownership epoch fencing whoever tries to use it. The generation
    refuses that pairing itself, from inside, because the carrier is legal only where the
    service says an execution continued another. This is the same refusal one step earlier, at
    the two doors a caller comes in through, so the answer is a plain error the caller can read
    rather than a workflow that fails on its first activation.
    """
    if start.carry is not None:
        raise ValueError(
            "a carried projection is how one execution of a generation hands itself to the "
            "next, and a generation being created has no earlier execution to be handed from"
        )


async def start_stream(
    client: Client,
    start: StreamStart,
    *,
    workflow_id: str,
    task_queue: str = STREAM_TASK_QUEUE,
    claimant_id: Optional[str] = None,
) -> "StreamHandle":
    """Start one generation, take ownership of it, and return the handle that holds it.

    Creation is a claim like a resume is: the returned handle carries the first epoch and the
    token that goes with it, and no call can change the stream without them.

    A generation created here says what it delivers. The legacy profile is not a third answer to
    that question: it is how a start recorded before the question existed decodes, and a run
    created under it would resolve nothing, serve the placeholder receipt, and hold no record of
    having decided to. The stream itself has to keep accepting it, because replaying one of those
    histories replays its start; creating one is refused here, which is the boundary a replay
    never crosses.
    """
    if start.profile == LEGACY:
        raise ValueError(
            "a generation created now says what each of its payloads may contain, and the "
            "legacy profile is how a history recorded before that reads rather than a shape a "
            "new run may be created in"
        )
    refuse_a_carried_projection(start)
    handle = await client.start_workflow(
        StreamWorkflow.run,
        start,
        id=workflow_id,
        task_queue=task_queue,
    )
    stream = StreamHandle(handle)
    await stream.claim_ownership(
        configuration_hash=configuration_hash(start),
        previous_epoch=0,
        claimant_id=claimant_id or workflow_id,
        reason="fresh",
    )
    return stream


async def discard_stream(client: Client, *, workflow_id: str) -> None:
    """End a generation that was started and never recorded, so it stops being live.

    This is for the one generation nobody can reach: a run that died between starting the
    stream and writing down its name leaves an authority with no consumer, no served message,
    and no record anywhere pointing at it. Ending it is what the caller that found its name
    does instead of leaving it running for the life of the service.

    A generation that is not there, or is already over, is the state this asks for, so it is
    not an error. Anything else the service says is.
    """
    handle = client.get_workflow_handle(workflow_id)
    try:
        await handle.terminate("this generation was started and never recorded")
    except RPCError as error:
        if error.status is not RPCStatusCode.NOT_FOUND:
            raise


async def resume_stream(
    client: Client,
    *,
    workflow_id: str,
    configuration_hash: str,
    claimant_id: Optional[str] = None,
    restored_checkpoints: Optional[Dict[str, str]] = None,
) -> "StreamHandle":
    """Take ownership of a generation that is already running, and fence its last writer.

    The epoch is read first and then swapped, so the claim carries the witness that makes two
    would-be owners resolve to one. ``configuration_hash`` is what this owner believes it is
    resuming: the generation refuses a claim that believes something else, before any state
    moves, which is what stops a changed queue, roster, plan, or capacity from continuing a
    history that was serving the old one.

    ``restored_checkpoints`` is what this owner put back before it claimed, per attempt. A
    generation holding an active attempt whose world it has authorized a change to refuses a
    claim that does not name that attempt and the exact checkpoint it retained for it, because
    continuing there means continuing in a world nobody restored. A caller that cannot restore
    one passes nothing and is refused, which is the answer that leaves the world alone. What has
    to be named, and the checkpoint to name it under, are both in :class:`StreamState`.

    A generation nobody owns yet is taken here too. Creation is two round trips, the Workflow
    and then the first claim, and a process that died between them left a generation that exists
    and has never had a writer. Nothing was served under it, because no call passes the ownership
    check while no token is installed, so the claim that takes it is a first claim rather than a
    replacement and says so.
    """
    handle = client.get_workflow_handle_for(StreamWorkflow.run, workflow_id)
    stream = StreamHandle(handle)
    state = await stream.stream_state()
    await stream.claim_ownership(
        configuration_hash=configuration_hash,
        previous_epoch=state.ownership_epoch,
        claimant_id=claimant_id or workflow_id,
        reason="fresh" if state.ownership_epoch == 0 else "resume",
        restored_checkpoints=restored_checkpoints,
    )
    return stream


async def resume_run_directory(
    client: Client,
    root: Union[str, Path],
    *,
    start: StreamStart,
    claimant_id: Optional[str] = None,
    restored_checkpoints: Optional[Dict[str, str]] = None,
) -> "StreamHandle":
    """Resume the generation a run directory holds, as the composition ``start`` describes it.

    ``client`` is opened against ``root``, because on an embedded service that directory is where
    the generation's history is and a client opened anywhere else does not hold it.

    The directory is read before the authority is: a directory with no protocol version, with
    version one, or with a version one log beside a version two manifest is refused here, and
    nothing is claimed. What the manifest supplies is the generation's identity.

    What it does not supply is the configuration. ``start`` is the generation this process was
    composed to serve, and the hash is derived from it here: a resume that handed the manifest's
    own recorded hash to the authority would compare one value with a copy of itself, and a
    replacement whose tasks, tools, prompts, plan, capacity, renderer or environment had changed
    would pass that comparison. The manifest is checked against the derived value first, so a
    directory this process was not composed for is refused before anything is claimed, and the
    authority is then given the derived value rather than the one the directory holds.
    """
    run: RunDirectory = open_run_directory(root)
    expected = configuration_hash(start)
    if expected != run.manifest.configuration_hash:
        raise ResumeRefused(
            "configuration_mismatch",
            f"{Path(root)} holds a generation composed against "
            f"{run.manifest.configuration_hash}, and this process is composed against {expected}",
        )
    return await resume_stream(
        client,
        workflow_id=run.manifest.workflow_id,
        configuration_hash=expected,
        claimant_id=claimant_id,
        restored_checkpoints=restored_checkpoints,
    )


def _nothing_took_it(error: BaseException) -> bool:
    """Whether this failure says the request never reached a generation that could answer it.

    A generation that has finished accepts no Update: the service refuses it before any
    validator or handler runs, and says so in the transport's own words rather than in a
    protocol code. That is not an answer to the request, and a caller holding an Update the
    generation did answer before it finished should be given that answer instead.
    """
    if isinstance(error, RPCError):
        if error.status is RPCStatusCode.NOT_FOUND:
            return True
        return "already completed" in str(error).lower()
    return False


def _decoded(answer: AnsweredUpdate, result_type: Any) -> Any:
    """The value a successful answer holds, as the type its handler returns.

    A refusal and a failure carry no value, and reading one is not decoding anything: what those
    hold is a code or a type and a message, and they are raised rather than returned.
    """
    if answer.kind in ("protocol", "failure"):
        return None
    return default_converter().payload_converter.from_payload(
        Payload(
            metadata={"encoding": b"json/plain"},
            data=json.dumps(
                answer.value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            ).encode("utf-8"),
        ),
        result_type,
    )


@dataclass(frozen=True)
class _Answered:
    """One answer read back out of a generation's journal, ready to be given to its caller."""

    answer: AnsweredUpdate
    value: Any

    def give_back(self) -> Any:
        """Hand the answer over the way the handler would have, raising what it raised."""
        if self.answer.kind == "protocol":
            raise StreamProtocolError(self.answer.code)
        if self.answer.kind == "failure":
            raise ApplicationError(
                self.answer.message, type=self.answer.type_name, non_retryable=True
            )
        return self.value


def _still_reaching_for_a_boundary(state: StreamState, moved: Optional[Tuple[str, int]]) -> bool:
    """Whether this generation is still working towards the boundary it decided on.

    Five things say yes, and the first of them is the one that matters most often. A generation
    that is no longer holding traffic back has either crossed its boundary, refused it, or found
    it was never available, and in every case the request that was rejected will be admitted the
    moment it is sent again, so waiting is right.

    A generation holding a message it owes, a grant for a world call, or a seal it has prepared
    is naming the thing the boundary is waiting for, and every one of those is cleared by work
    somebody else is doing rather than by time passing. A named item is progress only while the
    generation says it is still holding traffic back, and that is the whole of the guard: a
    generation whose gate can no longer admit the work that would clear what it named stops
    saying so, records why, and goes back to serving, so the first line below has already
    answered before a named item is ever consulted.

    A generation reading its store is the one that names nothing and is still working. An
    ownership claim reads the store before it may swap the epoch, and that read is an Activity
    with a timeout of its own and retries of its own: a batch can take minutes, and a second
    batch can follow it, while the projection stands still and nothing is pending, held or
    prepared. Without this a healthy claim following its ordinary retry policy would be read as
    a generation that had stopped.

    And a projection that has moved since the last look, or a store read that has finished since
    the last look, is a generation doing something, whatever it is.

    What is left is a generation that is holding traffic back, names nothing, is reading nothing,
    and is not moving. That is the only shape a caller should stop waiting on.
    """
    if not state.turnover_requested:
        return True
    if state.pending_message_id is not None or state.environment_call is not None:
        return True
    if state.prepared_seals or state.verifying > 0:
        return True
    if state.unfinished_handlers > 0:
        return True
    return moved is None or _where_it_stands(state) != moved


def _where_it_stands(state: StreamState) -> Tuple[str, int]:
    """The mark a caller compares one look at a generation with the next by.

    The projection hash alone is not enough. A store read that finished and a second one that
    started move nothing a projection covers, and a caller comparing hashes would call that
    standing still. The count of finished reads goes with it, so work that leaves the projection
    where it was still reads as work.
    """
    return state.stream_state_sha256, state.verification_batches


def turnover_pending(error: BaseException) -> bool:
    """Whether this failure says the generation was between executions when the Update arrived.

    It is deliberately not something :func:`protocol_error_code` answers for. A protocol code is
    a closed-set statement about the caller's request, and this says nothing about the request:
    it says the generation had decided to continue as new and had not got there yet. A transport
    that read it as a refusal would hand the agent an error where there is only latency.
    """
    cause = error.__cause__ if isinstance(error, WorkflowUpdateFailedError) else error
    return isinstance(cause, ApplicationError) and cause.type == TURNOVER_PENDING


def origin_unverified(error: BaseException) -> bool:
    """Whether this failure says a child had not yet authorized the lineage it carries.

    It is the gate rather than a refusal, and it is not something :func:`protocol_error_code`
    answers for. Nothing the caller sent is wrong: a child owns nothing and serves nothing until
    its own history has read the parent's record of it, so the same request under the same
    identifier reaches a child that has done that first work.
    """
    cause = error.__cause__ if isinstance(error, WorkflowUpdateFailedError) else error
    return isinstance(cause, ApplicationError) and cause.type == ORIGIN_UNVERIFIED


def fork_refusal(error: BaseException) -> Optional[str]:
    """Return the reason one fork was refused for, or ``None`` for anything else.

    A fork refusal is a controller refusal and never a protocol answer: it says which clause of the
    boundary, the witnesses or the plans did not hold, and none of it is a code the agent has ever
    seen. Anything that is not one is a fault, and a caller must not read it as a decision.

    The reason is the failure's own type, which is what makes it survive the outcome journal: an
    exact identifier answered out of that journal keeps a type and a message and carries no
    details, and a refusal read back after a boundary or from a closed parent has to name the same
    clause the live one named.
    """
    cause = error.__cause__ if isinstance(error, WorkflowUpdateFailedError) else error
    if not isinstance(cause, ApplicationError) or cause.type not in FORK_REFUSALS:
        return None
    return cause.type


def fork_can_be_retried(error: BaseException) -> bool:
    """Whether this refusal is infrastructure rather than a decision.

    The experiment retries infrastructure and never retries a decision, so this is the line it
    reads: a decision refusal is recorded with its reason rather than attempted again.
    """
    reason = fork_refusal(error)
    return reason is not None and reason in RETRYABLE_FORK_REFUSALS


def fork_preparation_epoch(error: BaseException) -> int:
    """Return the epoch the preparation this refusal came from was frozen at, or zero.

    Zero is a refusal raised before any episode existed, which is a different fact from a refusal
    recorded inside one: the first has no operation to be kept under, and the second has exactly
    one and keeps it for ever, whatever epoch the child holds by the time a controller comes back.

    What travels is that number and never the operation's own identity, so a caller recomputes the
    name it keeps the outcome under rather than being handed one by a transport.
    """
    cause = error.__cause__ if isinstance(error, WorkflowUpdateFailedError) else error
    if not isinstance(cause, ApplicationError) or cause.type not in FORK_REFUSALS:
        return 0
    details = list(cause.details)
    if len(details) < 3 or not isinstance(details[2], int) or details[2] < 1:
        return 0
    return details[2]


def fork_barrier_stands(error: BaseException) -> bool:
    """Whether this failure says the parent has forked and admits no further work."""
    cause = error.__cause__ if isinstance(error, WorkflowUpdateFailedError) else error
    return isinstance(cause, ApplicationError) and cause.type == FORK_BARRIER


async def fork_stream(
    client: Client, request: ForkRequest, *, attempt: int = 1
) -> ForkReceipt:
    """Fork one generation into the two children ``request`` plans, and return its typed evidence.

    The request is preflighted here before it is sent, because parent-side validation cannot
    produce a typed refusal for bytes the service would never admit: a request over the limit would
    come back as a transport fault naming nothing.

    ``attempt`` is which try at this fork is being sent, and it is the only thing that moves
    between two tries of one logical fork. An Update that completed keeps returning what it was
    answered with under its exact identifier for ever, refusals included, so a fork refused over a
    body the store had lost goes on being refused under the identifier that met the loss however
    thoroughly the object is repaired; progress is a fresh identifier carrying the same fork id and
    the same checkpoint. Repeating one attempt is the other half of that contract: the same number
    reaches the same identifier, so a caller whose reply was lost reads its own outcome rather than
    opening a second try.

    The request is measured as this client's own converter would encode it rather than as the
    default one would, because what the service admits is what this client sends: a caller that
    configured a codec of its own would otherwise be preflighted against bytes nobody transmits.

    Where the Update did not dispatch at all, the fork status Query is consulted rather than the
    request being sent again. Two things stop a dispatch, and both of them are places where the
    answer exists and only the route to it is missing: a parent that has forked and admits no
    further work, and a parent that has closed and accepts no Update at all. The typed contract is
    that Query rather than raw Update behaviour, and it charges the parent nothing.

    That reading comes first on the recovery path, because it is the one that says whether the
    parent still answers at all. Then the exact identifier's own outcome, which is immutable and
    stands however thoroughly the thing it refused over was repaired. Then the recorded receipt,
    which is what a fresh identifier for a fork that already completed is owed: a completed fork
    has an answer, and reporting it as still in flight would send a controller back to a parent
    that accepts nothing.

    The last of those is reached only where the parent said it holds no outcome under this
    identifier. A reading that could not be served says nothing about what the identifier was
    answered with, and taking it for nothing is how one try's caller ends up holding another
    try's success: a failure the parent accepted crosses a boundary in the carried journal, a
    fresh identifier goes on to complete the same logical fork, and the try that failed would
    then be answered with the fork that went through.
    """
    check_fork_request(request)
    converter = client.data_converter.payload_converter
    check_transmitted_size(
        "the fork request", request, converter, ceiling=TURNOVER_PAYLOAD_CEILING_BYTES
    )
    handle = client.get_workflow_handle_for(
        StreamWorkflow.run, request.parent_workflow_id
    )
    update_id = f"fork-{request.fork_id}-{fork_request_digest(request)[:32]}-{attempt}"
    try:
        return _admitted_receipt(
            await handle.execute_update(
                StreamWorkflow.fork_generation, request, id=update_id
            )
        )
    except Exception as error:
        if fork_refusal(error) is not None:
            raise
        if not fork_barrier_stands(error) and not _nothing_took_it(error):
            raise
        recorded = await fork_status(client, request)
        answered = await _answered_fork(handle, update_id, request.fork_id)
        if answered is not None:
            return _admitted_receipt(answered)
        if recorded.receipt is not None:
            return _admitted_receipt(recorded.receipt)
        raise _what_the_record_says(recorded, request) from error


def _admitted_receipt(receipt: ForkReceipt) -> ForkReceipt:
    """Return one received receipt, having refused a version this build does not read.

    The check is made where the receipt is received rather than inside whatever goes on to consume
    it, and it covers the recovered routes as well as the live one: a receipt read back out of a
    record or off a status answer is a received receipt too, and a consumer that skipped the check
    for those would decide over a shape nothing admitted.
    """
    check_fork_receipt(receipt)
    return receipt


async def fork_status(
    client: Client, request: ForkRequest, *, now_ms: Optional[int] = None
) -> ForkStatusAnswer:
    """Read where one fork stands, from a parent that may already have closed.

    It is a Query, so it costs the generation nothing, writes nothing, and can be asked of an
    execution that has closed. What it needs is a Worker able to replay the parent on its task
    queue and a history the service still retains, which together are the fork's answer window.

    A parent that never answers is the retryable refusal that says so rather than the transport
    fault the read arrived as, because a controller reads this to decide what to do next and a
    fault names no clause.

    A history the service no longer holds is the other answer, and it is not a fault to retry: the
    question was asked outside the window this parent's answers stand in, so it comes back as
    expired authority and the run is recorded incomplete rather than resolved by guessing.

    A history the service still holds is not the window either. Retention bounds what can be read
    and grants no permission, so a deployment keeping more history than it owes must not lengthen
    the answers this parent stands behind: the window is the one the barrier recorded and the
    parent's own close, and a question past it is expired authority whether or not the bytes are
    still there. ``now_ms`` is the observation of the moment that question is being asked at, the
    process clock where a caller gives none.

    Both declared shapes the answer can carry are admitted where the answer is received, before
    anything reads a field of either and before either is returned. A prepared record and a
    receipt are each a versioned value a caller decides over, and a route that returned one at a
    version this build does not read would hand that decision a shape nothing admitted.
    """
    handle = client.get_workflow_handle_for(
        StreamWorkflow.run, request.parent_workflow_id, run_id=request.parent_run_id
    )
    question = ForkStatusQuestion(
        fork_id=request.fork_id, request_digest=fork_request_digest(request)
    )
    try:
        answer: ForkStatusAnswer = await _asked_again(
            lambda: handle.query(
                StreamWorkflow.fork_status, question, rpc_timeout=_ANSWER_READ_TIMEOUT
            )
        )
    except RPCError as error:
        if error.status is RPCStatusCode.NOT_FOUND:
            raise ForkRefused(
                FORK_EXPIRED_AUTHORITY,
                f"the execution that prepared {request.fork_id} can no longer be read, so its "
                "answer window has closed",
            ) from error
        raise ForkRefused(
            FORK_UNREADABLE_PARENT,
            f"the parent of {request.fork_id} answered none of {_ANSWER_READ_TRIES} reads of "
            "where its fork stands",
        ) from error
    _admitted_answer(answer)
    await _inside_the_answer_window(handle, request, answer, now_ms)
    return answer


def _admitted_answer(answer: ForkStatusAnswer) -> ForkStatusAnswer:
    """Return one status answer, having refused the versions this build does not read.

    Both members are checked, because both are declared shapes a caller acts on: the record says
    which children a fork prepared and where it stands, and the receipt is the fork's own outcome.
    Neither is admitted by the other's check, and the answer is where both arrive.
    """
    if answer.record is not None:
        check_prepared_fork(answer.record)
    if answer.receipt is not None:
        check_fork_receipt(answer.receipt)
    return answer


async def _inside_the_answer_window(
    handle: WorkflowHandle,
    request: ForkRequest,
    answer: ForkStatusAnswer,
    now_ms: Optional[int],
) -> None:
    """Refuse an answer read after the window this parent's answers stand in had closed.

    The window is the earlier of the horizon recorded at the barrier and the parent's actual close
    plus the answer window, so a late close shortens it rather than moving it and a parent that
    closed after the horizon has none at all. The close is read from the service rather than
    guessed, and a parent still running has no close to cap anything with.

    A fork this parent never accepted has no window to be outside of, so nothing is asked about
    one: what that answer says is that there is no record, which is the answer either way.

    The close is read under the bound and the classification the status Query is read under, and
    for the same reason: this is one read operation in two parts, and a transport fault let out of
    the second half would replace the typed result a controller acts on with an exception naming
    no clause. A parent that answers and then cannot be described is unreadable and stays a retry;
    one whose history the service can no longer produce is expired authority, which is what the
    Query's own missing history already is.
    """
    if answer.record is None or answer.record.authority_horizon_at <= 0:
        return
    try:
        described = await _asked_again(lambda: handle.describe(rpc_timeout=_ANSWER_READ_TIMEOUT))
    except RPCError as error:
        if error.status is RPCStatusCode.NOT_FOUND:
            raise ForkRefused(
                FORK_EXPIRED_AUTHORITY,
                f"the execution that prepared {request.fork_id} can no longer be read, so its "
                "answer window has closed",
            ) from error
        raise ForkRefused(
            FORK_UNREADABLE_PARENT,
            f"the parent of {request.fork_id} answered none of {_ANSWER_READ_TRIES} reads of "
            "when it closed",
        ) from error
    closed = 0 if described.close_time is None else int(described.close_time.timestamp() * 1000)
    window = answer_window_ends(answer.record.authority_horizon_at, closed)
    asked = int(time.time() * 1000) if now_ms is None else now_ms
    if asked > window:
        raise ForkRefused(
            FORK_EXPIRED_AUTHORITY,
            f"the answers the parent of {request.fork_id} owes stood until {window} and this "
            f"question was asked at {asked}",
        )


async def _asked_again(read: Callable[[], Awaitable[Any]]) -> Any:
    """Ask one read again where it got no answer, because a lost question is not an answer.

    The last try is the one that answers or raises, so a read that cannot be served fails with
    what the service said rather than with a count of what was tried.
    """
    for _try in range(_ANSWER_READ_TRIES - 1):
        try:
            return await read()
        except RPCError:
            continue
    return await read()


async def _answered_fork(
    handle: WorkflowHandle, update_id: str, fork_id: str
) -> Optional[ForkReceipt]:
    """The receipt this exact identifier was answered with, if the parent still holds one.

    An Update that completed as a failure keeps returning that failure under its exact identifier
    for ever, so what is raised here is what the parent answered rather than a fresh judgement of
    it, and progress under a fresh identifier is the caller's next move.

    Nothing recorded and nothing readable are two different facts and only the first of them is
    ``None``. The parent saying it holds no outcome under this identifier is an answer, and it is
    the one that lets a fresh identifier be given the fork's own receipt. A read that could not be
    served is not an answer at all, so it comes back as the typed refusal every other read of this
    parent is classified under: a history the service can no longer produce is expired authority,
    and a parent that answered none of the tries is unreadable and stays a retry.

    An outcome this build cannot decode is unread for the same reason. What came back is an
    answer, and reporting the identifier as unanswered because these bytes could not be read would
    put a caller on the route reserved for one the parent never answered.
    """
    try:
        answer = await _asked_again(
            lambda: handle.query(
                StreamWorkflow.answered_update, update_id, rpc_timeout=_ANSWER_READ_TIMEOUT
            )
        )
    except Exception as error:  # noqa: BLE001 - a read that failed is not an answer
        if isinstance(error, RPCError) and error.status is RPCStatusCode.NOT_FOUND:
            raise ForkRefused(
                FORK_EXPIRED_AUTHORITY,
                f"the execution that answered this try at {fork_id} can no longer be read, so "
                "its answer window has closed",
            ) from error
        raise ForkRefused(
            FORK_UNREADABLE_PARENT,
            f"the parent of {fork_id} answered none of {_ANSWER_READ_TRIES} reads of what this "
            "try was answered with",
        ) from error
    if not answer.found:
        return None
    if answer.kind == "protocol":
        raise StreamProtocolError(answer.code)
    if answer.kind == "failure":
        raise ApplicationError(answer.message, type=answer.type_name, non_retryable=True)
    try:
        return _decoded(answer, ForkReceipt)
    except Exception as error:  # noqa: BLE001 - an answer this code cannot read is not an answer
        raise ForkRefused(
            FORK_UNREADABLE_PARENT,
            f"the parent of {fork_id} answered this try with an outcome this build cannot read",
        ) from error


def _what_the_record_says(
    answer: ForkStatusAnswer, request: ForkRequest
) -> ApplicationError:
    """Turn a status reading into the typed refusal a controller acts on.

    A fork the parent never accepted, one bound to another request, one that ended on a decision,
    one the parent has given up on and one whose parent cannot be asked are five different answers,
    and none of them is the transport fault the caller arrived with.

    A fork that ended on a decision answers with that decision, which is what the record retains it
    for. An abandoned fork answers with the reason it was abandoned for, which is a decision too:
    the reserve is spent or the preparation bound expired, the status stands, and the experiment
    records the link incomplete rather than sending the same fork again. The experiment retries
    infrastructure and never retries a decision, so what a controller reads here is the reason this
    fork ended rather than a state to send the same request into again.
    """
    if answer.conflict:
        return ForkRefused(
            FORK_REQUEST_CONFLICT,
            f"the fork {request.fork_id} is bound to another checkpoint or other plans",
        )
    if not answer.found:
        return ForkRefused(
            FORK_EXPIRED_AUTHORITY,
            f"the parent of {request.fork_id} accepts no Update and holds no record of it",
        )
    if answer.record is not None and answer.record.conflict_reason is not None:
        return ForkRefused(answer.record.conflict_reason, answer.record.conflict_clause)
    if answer.record is not None and answer.record.status == FORK_ABANDONED:
        return ForkRefused(
            answer.record.abandoned_reason or FORK_IN_FLIGHT,
            f"the fork {request.fork_id} was abandoned and every child it created stands as it "
            "was left",
        )
    return ForkRefused(
        FORK_IN_FLIGHT,
        f"the fork {request.fork_id} stands {answer.record.status if answer.record else ''} and "
        "the parent accepts no further work under this identifier",
    )


def protocol_error_code(error: BaseException) -> Optional[str]:
    """Return the protocol error code an Update failure carries, or ``None``.

    A refusal crosses the transport as an application failure whose type says it is one of
    ours. Anything else is a fault, and a caller must not read it as a protocol answer.
    """
    cause = error.__cause__ if isinstance(error, WorkflowUpdateFailedError) else error
    if isinstance(cause, ApplicationError) and cause.type == "ProtocolError":
        return cause.message
    return None


@dataclass
class StreamHandle:
    """The Python API a gateway calls: claim, pull, seal, present, close, inspect.

    The handle holds the writer this owner claimed, and every stream-affecting call carries it.
    A handle whose epoch has been superseded keeps working exactly as far as the transport: its
    calls reach the generation and are refused there, which is what fencing means.
    """

    handle: WorkflowHandle
    _writer: Optional[Writer] = None

    @property
    def writer(self) -> Writer:
        """The epoch and token this handle speaks for."""
        if self._writer is None:
            raise ValueError("this handle has not claimed the generation, so it cannot write")
        return self._writer

    async def _answered(self, update_id: str, result_type: Any) -> Optional["_Answered"]:
        """What this generation already answered this exact Update with, if it can be read.

        This is a Query, so it costs the generation nothing and can be asked of an execution that
        has closed, which is the point: the two places a request cannot be sent again are a
        generation holding traffic back for a boundary and a generation that has finished, and
        both of them already hold the answer.

        A Query is answered by this package's own code rather than by the service alone, so it
        needs a Worker able to replay the generation on its task queue. A deployment with none
        cannot answer one, and a caller that cannot get an answer is left with the failure it
        already had. Nothing here writes: an Update that was never answered stays unanswered.

        A successful answer is decoded here rather than at the point it is handed over, for the
        same reason. Decoding is the last thing that can fail, and a caller must not be given a
        decoding fault in place of the failure that sent it here: an answer this code cannot read
        is no answer, and the original failure stands.
        """
        if not update_id:
            return None
        try:
            answer = await self.handle.query(
                StreamWorkflow.answered_update, update_id, rpc_timeout=_ANSWER_READ_TIMEOUT
            )
        except Exception:  # noqa: BLE001 - a generation that cannot be read answers nothing
            return None
        if not answer.found:
            return None
        try:
            return _Answered(answer, _decoded(answer, result_type))
        except Exception:  # noqa: BLE001 - an answer this code cannot read is not an answer
            return None

    async def _sent(
        self, update_id: str, result_type: Any, send: Callable[[], Awaitable[Any]]
    ) -> Any:
        """Send one Update, waiting out a generation that is between executions.

        The durable service bounds one execution, and a generation longer than that bound
        continues as new before it reaches the bound. While it is waiting for the quiet point
        to do that, it rejects every arriving Update that cannot bring it there. That rejection
        is not a protocol answer and it is never the caller's: it says the generation is between
        executions, and the same request sent again reaches the one that comes next.

        A rejection costs the generation nothing, because the service does not count an Update
        it never accepted, so sending again is free of the very limit the generation is avoiding.
        The Update ID does not change, which is what keeps a retry a retry: within an execution
        the service answers a repeated ID from its own record, and across the boundary the
        generation answers it from what it carried over.

        How long this waits is the generation's business rather than a clock's. Between each
        send it reads the state, which is a Query and therefore free, and it keeps waiting for
        as long as that state says the generation is still working towards its boundary. A
        boundary can take a long time and still be healthy: a grant held for a world call blocks
        it until the agent's own tool call comes back, and a claim reading the store blocks it
        for as long as that read and its retries take. Giving up on those would hand a caller a
        fault where there was only work in progress.

        What is not healthy is a generation that is holding traffic back, names nothing it is
        waiting for, is reading nothing, and is not moving. That is the only thing the clock
        below measures, and when it runs out the caller gets the rejection as the fault it
        already arrived as. The transport raises it rather than mapping it to anything, so the
        agent is shown no code it has never seen and the failure reaches whatever reads a fault.
        """
        stalled_since: Optional[float] = None
        moved: Optional[Tuple[str, int]] = None
        wait = _TURNOVER_FIRST_WAIT
        while True:
            try:
                return await send()
            except Exception as error:
                # An Update this generation has already answered is answered from what it
                # answered, but only where the request could not be dispatched at all. Two
                # things stop a dispatch: a generation holding traffic back for a boundary, and
                # one that has finished and accepts no Update. In both the answer exists and
                # only the route to it is missing, so it is read rather than sent. Every other
                # failure is the generation's own answer to this call and is left alone.
                gated = turnover_pending(error)
                if gated or _nothing_took_it(error):
                    answered = await self._answered(update_id, result_type)
                    if answered is not None:
                        return answered.give_back()
                if not gated:
                    raise
                # A generation that cannot be read is one that is not visibly making progress,
                # and it is answered that way rather than by handing the caller the read's own
                # failure in place of the one it actually got.
                try:
                    state: Optional[StreamState] = await self.stream_state()
                except Exception:  # noqa: BLE001 - an unreadable generation is not progress
                    state = None
                if state is not None and _still_reaching_for_a_boundary(state, moved):
                    stalled_since, moved = None, _where_it_stands(state)
                else:
                    if stalled_since is None:
                        stalled_since = time.monotonic()
                    if time.monotonic() - stalled_since >= _TURNOVER_STALLED_AFTER:
                        raise
            await asyncio.sleep(wait)
            wait = min(wait * 2, _TURNOVER_LONGEST_WAIT)

    async def claim_ownership(
        self,
        *,
        configuration_hash: str,
        previous_epoch: int,
        claimant_id: str,
        reason: str,
        restored_checkpoints: Optional[Dict[str, str]] = None,
    ) -> OwnershipReceipt:
        """Claim the generation for this owner, and hold the token the claim installed.

        The token is minted here and the generation keeps only its hash, so the value that
        proves ownership exists in this process and in the calls it makes.

        ``restored_checkpoints`` says which active attempts this owner put back, and from which
        checkpoint. It is the claim's own account of itself, so an owner that restored nothing
        sends nothing and the generation decides what that costs it.
        """
        token = secrets.token_hex(32)
        claim = OwnershipClaim(
            claimant_id=claimant_id,
            previous_epoch=previous_epoch,
            fencing_token=token,
            configuration_hash=configuration_hash,
            reason=reason,
            restored_checkpoints=dict(restored_checkpoints or {}),
        )
        update_id = f"own-{previous_epoch}-{claimant_id}-{sha256(token.encode()).hexdigest()[:16]}"
        receipt = await self._sent(
            update_id,
            OwnershipReceipt,
            lambda: self.handle.execute_update(
                StreamWorkflow.claim_ownership, claim, id=update_id
            )
        )
        self._writer = Writer(
            ownership_epoch=receipt.ownership_epoch, fencing_token=token
        )
        return receipt

    async def claim_consumer(self, claim: ConsumerClaim) -> ConsumerReceipt:
        """Bind this caller as the generation's one consumer."""
        writer = self.writer
        return await self._sent(
            f"claim-{writer.ownership_epoch}-{claim.consumer_id}",
            ConsumerReceipt,
            lambda: self.handle.execute_update(
                StreamWorkflow.claim_consumer,
                args=[claim, writer],
                id=f"claim-{writer.ownership_epoch}-{claim.consumer_id}",
            )
        )

    async def pull(self, request: PullRequest) -> OfferedMessage:
        """Ask for the next message. A retry of the same request reaches the same Update."""
        writer = self.writer
        return await self._sent(
            _update_id("pull", request.request_id, request, writer),
            OfferedMessage,
            lambda: self.handle.execute_update(
                StreamWorkflow.pull,
                args=[request, writer],
                id=_update_id("pull", request.request_id, request, writer),
            )
        )

    async def info(self, request: InfoRequest) -> OfferedMessage:
        """Ask how much of the queue there is. A retry of the same request reaches the same
        Update, and a generation that declares no info tool refuses it."""
        writer = self.writer
        return await self._sent(
            _update_id("info", request.request_id, request, writer),
            OfferedMessage,
            lambda: self.handle.execute_update(
                StreamWorkflow.info,
                args=[request, writer],
                id=_update_id("info", request.request_id, request, writer),
            )
        )

    async def seal(self, request: SealRequest) -> OfferedMessage:
        """End an attempt with a terminal call, and get its acknowledgement or refusal."""
        identity = terminal_request_identity(
            request.metadata,
            request.public_tool_name,
            request.native_terminal_name,
            request.native_arguments,
            request.terminal_source,
        )
        writer = self.writer
        return await self._sent(
            f"seal-{writer.ownership_epoch}-{request.metadata.request_id}-{identity[:32]}",
            OfferedMessage,
            lambda: self.handle.execute_update(
                StreamWorkflow.seal_attempt,
                args=[request, writer],
                id=f"seal-{writer.ownership_epoch}-{request.metadata.request_id}-{identity[:32]}",
            )
        )

    async def commit_presentation(self, commit: PresentationCommit) -> PresentationAck:
        """Attest that the exact offered bytes were handed to the transport."""
        writer = self.writer
        return await self._sent(
            _update_id("present", commit.attestation_id, commit, writer),
            PresentationAck,
            lambda: self.handle.execute_update(
                StreamWorkflow.commit_presentation,
                args=[commit, writer],
                id=_update_id("present", commit.attestation_id, commit, writer),
            )
        )

    async def begin_environment_call(self, call: EnvironmentCall) -> EnvironmentLease:
        """Take the generation for one environment call. A retry reaches the same Update."""
        writer = self.writer
        return await self._sent(
            f"environment-{writer.ownership_epoch}-{call.call_id}",
            EnvironmentLease,
            lambda: self.handle.execute_update(
                StreamWorkflow.begin_environment_call,
                args=[call, writer],
                id=f"environment-{writer.ownership_epoch}-{call.call_id}",
            )
        )

    async def end_environment_call(self, call: EnvironmentCall) -> EnvironmentLease:
        """Give the generation back. Releasing twice releases once."""
        writer = self.writer
        return await self._sent(
            f"environment-end-{writer.ownership_epoch}-{call.call_id}",
            EnvironmentLease,
            lambda: self.handle.execute_update(
                StreamWorkflow.end_environment_call,
                args=[call, writer],
                id=f"environment-end-{writer.ownership_epoch}-{call.call_id}",
            )
        )

    async def finalize(self, request: FinalizeRequest) -> AttemptFinalized:
        """End one attempt that nothing is going to finish.

        The Update ID is the logical request and the whole of what it asked, so a controller
        that lost the answer reaches the same Update rather than a second ending, while the
        same logical ID carrying anything else reaches the workflow and is judged there. The
        identity is hashed rather than concatenated from the fields the ending reads, so a
        request that differs only in a field this ID forgot cannot be answered from the cache.
        """
        writer = self.writer
        identity = finalize_request_identity(request)
        return await self._sent(
            f"finalize-{writer.ownership_epoch}-{request.request_id}-{identity[:32]}",
            AttemptFinalized,
            lambda: self.handle.execute_update(
                StreamWorkflow.finalize_attempt,
                args=[request, writer],
                id=f"finalize-{writer.ownership_epoch}-{request.request_id}-{identity[:32]}",
            )
        )


    async def close_queue(self) -> QueueClosed:
        """Close the queue to insertion."""
        writer = self.writer
        return await self._sent(
            f"close-queue-{writer.ownership_epoch}",
            QueueClosed,
            lambda: self.handle.execute_update(
                StreamWorkflow.close_queue,
                args=[writer],
                id=f"close-queue-{writer.ownership_epoch}",
            )
        )

    async def confirm_state(self) -> StreamState:
        """Read the generation's state through the path a write takes, so it can be refused.

        Every other Update here is deduplicated by an identity its caller can repeat, because
        a lost answer has to reach the same Update rather than a second one. This one is the
        opposite. It changes nothing, and a caller asking whether the stream still admits it
        must not be answered with what the stream said the last time it asked, so every ask is
        its own Update.
        """
        writer = self.writer
        update_id = f"confirm-{writer.ownership_epoch}-{secrets.token_hex(16)}"
        return await self._sent(
            update_id,
            StreamState,
            lambda: self.handle.execute_update(
                StreamWorkflow.confirm_state, args=[writer], id=update_id
            )
        )

    async def prepare_child(self, *, fork_id: str) -> ChildReady:
        """Build this child's own body for the obligation it inherited, and read its readiness.

        The Update identifier carries the epoch this owner holds, which is the one thing that
        moves between two attempts at one logical preparation. A refusal keeps coming back under
        the identifier that met it, however thoroughly the object behind it is repaired, so a
        repair is submitted by the owner that follows and reaches a handler rather than replaying
        an answer. The operation the record joins them by is frozen at the first attempt and is
        inside neither identifier.
        """
        writer = self.writer
        update_id = f"prepare-{fork_id}-{writer.ownership_epoch}"
        ready = await self._sent(
            update_id,
            ChildReady,
            lambda: self.handle.execute_update(
                StreamWorkflow.prepare_child, args=[writer], id=update_id
            ),
        )
        check_child_ready(ready)
        return ready

    async def stream_state(self) -> StreamState:
        """Read the generation's state without changing it."""
        return await self.handle.query(StreamWorkflow.stream_state)

    async def checkpoint_evidence(self) -> CheckpointEvidenceAnswer:
        """Read the stream's half of a freeze, which is what a fork is prepared against.

        This is the documented way to it, and there is no other: the cursor, the visible digest and
        the attestation a request carries are the generation's own record of what it committed, and
        a controller reading them off a transport would be handing the parent back a value that
        transport already believed.

        It is a Query, so it charges nothing, writes nothing, and can be asked as often as a
        controller likes while it waits for a generation to reach the quiet point.
        """
        return await self.handle.query(StreamWorkflow.checkpoint_evidence)

    async def present(
        self,
        message: OfferedMessage,
        *,
        attestation_id: str,
        transcript_blob: str,
        provider_turn_blob: Optional[str] = None,
        task_start_checkpoint_blob: Optional[str] = None,
    ) -> PresentationAck:
        """Build and commit the presentation for ``message``.

        The cursor and the pre-event state hash are read from the stream itself, and the
        visible hash is taken from the offered bytes, so the attestation covers what was
        offered rather than what the caller believes was offered.
        """
        state = await self.stream_state()
        commit = PresentationCommit(
            attestation_id=attestation_id,
            cursor_before=state.cursor,
            message_id=message.message_id,
            visible_bytes_sha256=sha256(message.visible_text.encode("utf-8")).hexdigest(),
            transcript_blob=transcript_blob,
            provider_turn_blob=provider_turn_blob,
            task_start_checkpoint_blob=task_start_checkpoint_blob,
            completed_turn=message.kind == "seal_ack",
            stream_state_before_sha256=state.stream_state_sha256,
        )
        return await self.commit_presentation(commit)


def _update_id(prefix: str, request_id: str, value: Any, writer: Writer) -> str:
    """The Temporal Update ID one logical request reaches its own Update under.

    The epoch is part of it because Temporal deduplicates by this ID: two owners sending the
    same logical request must not have the second one answered with the first one's refusal.
    """
    if isinstance(value, PullRequest):
        identity = pull_request_identity(value)
    elif isinstance(value, InfoRequest):
        identity = info_request_identity(value)
    else:
        identity = presentation_request_identity(value)
    return f"{prefix}-{writer.ownership_epoch}-{request_id}-{identity[:32]}"
