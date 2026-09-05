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
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
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
    Union,
)

from temporalio.client import Client, WorkflowHandle, WorkflowUpdateFailedError
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
from shogym.serve.protocol_v2.kernel.activities import kernel_activities
from shogym.serve.protocol_v2.kernel.messages import (
    AttemptFinalized,
    ConsumerClaim,
    ConsumerReceipt,
    EnvironmentCall,
    EnvironmentLease,
    FinalizeRequest,
    finalize_request_identity,
    OfferedMessage,
    OwnershipClaim,
    OwnershipReceipt,
    QueueClosed,
    SealRequest,
    StreamStart,
    StreamState,
    Writer,
    configuration_hash,
)
from shogym.serve.protocol_v2.kernel.workflow import TURNOVER_PENDING, StreamWorkflow
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
# the fault it arrives as, carrying no protocol code. The transport raises it rather than
# mapping it, and the launcher records the run incomplete, which is what a generation that
# stopped should produce. There is no new code for the agent to see.
_TURNOVER_STALLED_AFTER = 300.0
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
            yield environment.client
        finally:
            await environment.shutdown()


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
    served = list(activities) if activities is not None else kernel_activities()
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


def _still_reaching_for_a_boundary(state: StreamState, moved: Optional[str]) -> bool:
    """Whether this generation is still working towards the boundary it decided on.

    Four things say yes, and the first of them is the one that matters most often. A generation
    that is no longer latched has either crossed its boundary or refused it, and either way the
    request that was rejected will be admitted the moment it is sent again, so waiting is right.
    A generation holding a message it owes, a grant for a world call, or a seal it has prepared
    is naming the thing the boundary is waiting for, and every one of those is cleared by work
    somebody else is doing rather than by time passing. And a projection that has moved since
    the last look is a generation doing something, whatever it is.

    What is left is a generation that says it is latched, names nothing, and is not moving. That
    is the only shape a caller should stop waiting on.
    """
    if not state.turnover_requested:
        return True
    if state.pending_message_id is not None or state.environment_call is not None:
        return True
    if state.prepared_seals:
        return True
    return moved is None or state.stream_state_sha256 != moved


def turnover_pending(error: BaseException) -> bool:
    """Whether this failure says the generation was between executions when the Update arrived.

    It is deliberately not something :func:`protocol_error_code` answers for. A protocol code is
    a closed-set statement about the caller's request, and this says nothing about the request:
    it says the generation had decided to continue as new and had not got there yet. A transport
    that read it as a refusal would hand the agent an error where there is only latency.
    """
    cause = error.__cause__ if isinstance(error, WorkflowUpdateFailedError) else error
    return isinstance(cause, ApplicationError) and cause.type == TURNOVER_PENDING


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

    async def _sent(self, send: Callable[[], Awaitable[Any]]) -> Any:
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

        What is not healthy is a generation that says it is latched, names nothing it is waiting
        for, and is not moving. That is the only thing the clock below measures, and when it
        runs out the caller gets the rejection as the fault it already arrived as. The transport
        raises it rather than mapping it to anything, so the run is recorded incomplete and the
        agent is shown no code it has never seen.
        """
        stalled_since: Optional[float] = None
        moved: Optional[str] = None
        wait = _TURNOVER_FIRST_WAIT
        while True:
            try:
                return await send()
            except Exception as error:
                if not turnover_pending(error):
                    raise
                # A generation that cannot be read is one that is not visibly making progress,
                # and it is answered that way rather than by handing the caller the read's own
                # failure in place of the one it actually got.
                try:
                    state: Optional[StreamState] = await self.stream_state()
                except Exception:  # noqa: BLE001 - an unreadable generation is not progress
                    state = None
                if state is not None and _still_reaching_for_a_boundary(state, moved):
                    stalled_since, moved = None, state.stream_state_sha256
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
            lambda: self.handle.execute_update(
                StreamWorkflow.confirm_state, args=[writer], id=update_id
            )
        )

    async def stream_state(self) -> StreamState:
        """Read the generation's state without changing it."""
        return await self.handle.query(StreamWorkflow.stream_state)

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
