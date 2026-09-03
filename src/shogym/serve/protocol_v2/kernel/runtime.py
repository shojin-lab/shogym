"""Reaching the stream from outside it: the service, the Worker, and the caller's handle.

The service is embedded. A user installs the extra and runs; nothing here asks them to
install a server, start one, or keep one alive. The dev service is downloaded once into
``~/.cache/shogym/temporal`` and its SQLite file lives beside it, so a stream survives the
process that served it. ``SHOGYM_TEMPORAL_ADDRESS`` points at a server someone else runs, and
then nothing is downloaded or started here at all.

:class:`StreamHandle` is the surface a gateway calls. It derives each Update ID from the
logical request and its canonical identity, which is what makes a transport retry reach the
same Update instead of a second one, and it turns an offered message plus its blobs into the
presentation the workflow will verify.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncIterator, Optional, Sequence

from temporalio.client import Client, WorkflowHandle, WorkflowUpdateFailedError
from temporalio.exceptions import ApplicationError
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Replayer, Worker
from temporalio.worker.workflow_sandbox import SandboxedWorkflowRunner, SandboxRestrictions

from shogym.serve.protocol_v2 import (
    PresentationAck,
    PresentationCommit,
    PullRequest,
    presentation_request_identity,
    pull_request_identity,
    terminal_request_identity,
)
from shogym.serve.protocol_v2.kernel.activities import kernel_activities
from shogym.serve.protocol_v2.kernel.messages import (
    ConsumerClaim,
    ConsumerReceipt,
    OfferedMessage,
    QueueClosed,
    SealRequest,
    StreamStart,
    StreamState,
)
from shogym.serve.protocol_v2.kernel.workflow import StreamWorkflow

TEMPORAL_ADDRESS_ENV = "SHOGYM_TEMPORAL_ADDRESS"
STREAM_TASK_QUEUE = "shogym-stream-v2"
_DATABASE_FILE = "stream.sqlite"


def temporal_home() -> Path:
    """The directory the embedded service and its database live in."""
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "shogym"
    return root / "temporal"


@asynccontextmanager
async def durable_client(*, namespace: str = "default") -> AsyncIterator[Client]:
    """Yield a client for the durable service, starting an embedded one if needed.

    The download directory has to exist before the service is asked for, so it is created
    here rather than left to the first user who has never run this.
    """
    address = os.environ.get(TEMPORAL_ADDRESS_ENV)
    if address:
        yield await Client.connect(address, namespace=namespace)
        return
    home = temporal_home()
    home.mkdir(parents=True, exist_ok=True)
    environment = await WorkflowEnvironment.start_local(
        namespace=namespace,
        dev_server_database_filename=str(home / _DATABASE_FILE),
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


async def start_stream(
    client: Client,
    start: StreamStart,
    *,
    workflow_id: str,
    task_queue: str = STREAM_TASK_QUEUE,
) -> "StreamHandle":
    """Start one generation and return the handle a gateway drives it through."""
    handle = await client.start_workflow(
        StreamWorkflow.run,
        start,
        id=workflow_id,
        task_queue=task_queue,
    )
    return StreamHandle(handle)


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
    """The Python API a gateway calls: claim, pull, seal, present, close, inspect."""

    handle: WorkflowHandle

    async def claim_consumer(self, claim: ConsumerClaim) -> ConsumerReceipt:
        """Bind this caller as the generation's one consumer."""
        return await self.handle.execute_update(
            StreamWorkflow.claim_consumer, claim, id=f"claim-{claim.consumer_id}"
        )

    async def pull(self, request: PullRequest) -> OfferedMessage:
        """Ask for the next message. A retry of the same request reaches the same Update."""
        return await self.handle.execute_update(
            StreamWorkflow.pull, request, id=_update_id("pull", request.request_id, request)
        )

    async def seal(self, request: SealRequest) -> OfferedMessage:
        """End an attempt with a terminal call, and get its acknowledgement or refusal."""
        identity = terminal_request_identity(
            request.metadata,
            request.public_tool_name,
            request.native_terminal_name,
            request.native_arguments,
        )
        return await self.handle.execute_update(
            StreamWorkflow.seal_attempt,
            request,
            id=f"seal-{request.metadata.request_id}-{identity[:32]}",
        )

    async def commit_presentation(self, commit: PresentationCommit) -> PresentationAck:
        """Attest that the exact offered bytes were handed to the transport."""
        return await self.handle.execute_update(
            StreamWorkflow.commit_presentation,
            commit,
            id=_update_id("present", commit.attestation_id, commit),
        )

    async def close_queue(self) -> QueueClosed:
        """Close the queue to insertion."""
        return await self.handle.execute_update(StreamWorkflow.close_queue, id="close-queue")

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


def _update_id(prefix: str, request_id: str, value: Any) -> str:
    identity = (
        pull_request_identity(value)
        if isinstance(value, PullRequest)
        else presentation_request_identity(value)
    )
    return f"{prefix}-{request_id}-{identity[:32]}"
