"""The durable protocol v2 kernel: one stream generation as a Temporal workflow.

This package is where Temporal is imported, and nothing outside it imports Temporal. A
quickstart install has it, because serving is what the package is for, and the runner starts
its own embedded service, so a user never installs or runs a server.

What is here is the generic kernel: a typed pull, a durable seal before any acknowledgement,
offers and presentations against a harness cursor, a schedule model, and a monotonic Done. A
generation carries an immutable assignment roster and one release plan. Immediate makes an
obligation eligible at the seal of the attempt it belongs to, Never creates no obligation at
all, and a plan may gate a named task on a payload being presented or on another task sealing.
The experimental profile is not here: no leg automata, payload families, fork, or retry scopes,
and no delayed release, blocked obligations, or end-of-queue tail policies.

The Python API a gateway calls
------------------------------

Start a service and a Worker. ``run_directory`` is the run this service is for, and the history
it writes goes there, beside that run's blobs and manifest. A client opened without one gets a
database that goes when the process does, which is the right answer for a stream nobody will
take over and the wrong one for every other::

    async with durable_client(run_directory=run) as client:
        async with stream_worker(client):
            ...

Start a stream. :class:`~shogym.serve.protocol_v2.kernel.messages.StreamStart` carries the
closed queue with every public identifier preallocated, the capacity, the consumer claim hash,
the initial cursor, and the terminal tool::

    stream = await start_stream(client, start, workflow_id="stream/run-1/gen-1")
    await stream.claim_consumer(ConsumerClaim(consumer_id=..., claim_hash=...))

Starting is a claim of ownership, and so is taking a generation over after the process that was
serving it went away. The client that takes it over is opened against that run's directory,
because the history the last process wrote is what is being resumed and that is where it is::

    async with durable_client(run_directory=run) as client:
        stream = await resume_stream(client, workflow_id="stream/run-1/gen-1",
                                     configuration_hash=configuration_hash(start))

That fences the previous writer. Its handle keeps the epoch it had, and every call it makes
from then on is refused without touching the stream, including one that was already in flight.
A claim that presents a different configuration hash is refused instead, and nothing moves.
:func:`resume_run_directory` does the same from a directory, deriving that hash from the
composition the resuming process serves rather than reading it back out of the manifest. It is
also where a protocol v1 run is refused before anything is claimed at all.

An active attempt the generation has since authorized a call to a world for is one whose world
has moved past the checkpoint it would come back from, and taking it over means saying so. The
claim carries ``restored_checkpoints``, the attempts this owner put back and the checkpoint it
put each of them back from, and a claim that says nothing about such an attempt is refused with
``invalid_attempt`` and nothing touched. What has to be named, and what to name it under, are
``restoration_required`` and ``task_checkpoints`` in the state Query.

Then the loop. A pull returns one
:class:`~shogym.serve.protocol_v2.kernel.messages.OfferedMessage`, whose ``visible_text`` is
the exact canonical bytes to put in front of the model. Nothing advances until the harness has
written those bytes and said so::

    message = await stream.pull(PullRequest(request_id=..., last_presented_cursor=cursor))
    ack = await stream.present(message, attestation_id=..., transcript_blob=...,
                               task_start_checkpoint_blob=...)
    cursor = ack.cursor

A Task presentation must carry the task-start checkpoint it can be restored from. When the
model files, the gateway intercepts the terminal call and seals::

    result = await stream.seal(SealRequest(metadata=..., public_tool_name=...,
                                           native_terminal_name=..., native_arguments=...))

That returns a SealAck, or a SealReject when the native arguments do not match the declared
schema, which leaves the attempt exactly where it was. An acknowledgement is presented like
any other result, but it is the last result of a completed provider turn, so its presentation
carries the provider-turn blob and says the turn completed. The next pull is refused until it
does. When the queue is closed and everything has been sealed, acknowledged, and delivered, a
pull returns Done, and presenting Done ends the generation.

A refusal arrives as an Update failure carrying one code from the protocol's closed set. Read
it with :func:`protocol_error_code`; anything that function returns ``None`` for is a fault
and not a protocol answer. :meth:`StreamHandle.stream_state` is harness-only and writes
nothing.
"""

from shogym.serve.protocol_v2 import BlobRef, blob_ref
from shogym.serve.protocol_v2.kernel.activities import (
    generate_payload_bundle_activity,
    grade_attempt_activity,
    kernel_activities,
    seal_attempt_activity,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel.messages import (
    BlobsVerified,
    ConsumerClaim,
    ConsumerReceipt,
    EnvironmentCall,
    EnvironmentLease,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    GradeAttemptResult,
    OfferedMessage,
    OwnershipClaim,
    OwnershipReceipt,
    PayloadBundle,
    PayloadCandidate,
    QueueClosed,
    SealAttemptInput,
    SealAttemptResult,
    SealRequest,
    StreamOutcome,
    StreamStart,
    StreamState,
    TaskItem,
    TerminalTool,
    VerifyBlobsInput,
    Writer,
    assignments_for,
    configuration_hash,
    hidden_seal_id,
)
from shogym.serve.protocol_v2.kernel.runtime import (
    STREAM_TASK_QUEUE,
    TEMPORAL_ADDRESS_ENV,
    StreamHandle,
    discard_stream,
    durable_client,
    protocol_error_code,
    resume_run_directory,
    resume_stream,
    run_stream_worker,
    start_stream,
    stream_replayer,
    stream_worker,
    temporal_home,
)
from shogym.serve.protocol_v2.kernel.workflow import StreamProtocolError, StreamWorkflow

__all__ = [
    "STREAM_TASK_QUEUE",
    "TEMPORAL_ADDRESS_ENV",
    "BlobRef",
    "BlobsVerified",
    "ConsumerClaim",
    "ConsumerReceipt",
    "EnvironmentCall",
    "EnvironmentLease",
    "GeneratePayloadBundleInput",
    "GradeAttemptInput",
    "GradeAttemptResult",
    "OfferedMessage",
    "OwnershipClaim",
    "OwnershipReceipt",
    "PayloadBundle",
    "PayloadCandidate",
    "QueueClosed",
    "SealAttemptInput",
    "SealAttemptResult",
    "SealRequest",
    "StreamHandle",
    "StreamOutcome",
    "StreamProtocolError",
    "StreamStart",
    "StreamState",
    "StreamWorkflow",
    "TaskItem",
    "TerminalTool",
    "VerifyBlobsInput",
    "Writer",
    "assignments_for",
    "blob_ref",
    "configuration_hash",
    "discard_stream",
    "durable_client",
    "generate_payload_bundle_activity",
    "grade_attempt_activity",
    "hidden_seal_id",
    "kernel_activities",
    "protocol_error_code",
    "resume_run_directory",
    "resume_stream",
    "run_stream_worker",
    "seal_attempt_activity",
    "start_stream",
    "stream_replayer",
    "stream_worker",
    "temporal_home",
    "verify_blobs_activity",
]
