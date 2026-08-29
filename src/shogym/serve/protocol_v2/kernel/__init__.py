"""The durable protocol v2 kernel: one stream generation as a Temporal workflow.

This package is behind the ``shogym[durable]`` extra and nothing outside it imports Temporal.
A quickstart install still runs an episode; a durable stream needs the extra, and then the
runner starts its own embedded service, so a user never installs or runs a server.

What is here is the generic kernel: a typed pull, a durable seal before any acknowledgement,
offers and presentations against a harness cursor, and a monotonic Done. The experimental
profile is not here. There are no assignments, release plans, leg automata, payload families,
fork, or retry scopes, and the schedule is the compatibility Immediate one: every task carries
exactly one payload obligation, which becomes eligible when that task seals.

The Python API a gateway calls
------------------------------

Start a service and a Worker::

    async with durable_client() as client:
        async with stream_worker(client):
            ...

Start a stream. :class:`~shogym.serve.protocol_v2.kernel.messages.StreamStart` carries the
closed queue with every public identifier preallocated, the capacity, the consumer claim hash,
the initial cursor, and the terminal tool::

    stream = await start_stream(client, start, workflow_id="stream/run-1/gen-1")
    await stream.claim_consumer(ConsumerClaim(consumer_id=..., claim_hash=...))

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

from shogym.serve.protocol_v2.kernel.activities import (
    generate_payload_bundle_activity,
    grade_attempt_activity,
    kernel_activities,
    seal_attempt_activity,
)
from shogym.serve.protocol_v2.kernel.messages import (
    BlobRef,
    ConsumerClaim,
    ConsumerReceipt,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    GradeAttemptResult,
    OfferedMessage,
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
    blob_ref,
    hidden_seal_id,
)
from shogym.serve.protocol_v2.kernel.runtime import (
    STREAM_TASK_QUEUE,
    TEMPORAL_ADDRESS_ENV,
    StreamHandle,
    durable_client,
    protocol_error_code,
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
    "ConsumerClaim",
    "ConsumerReceipt",
    "GeneratePayloadBundleInput",
    "GradeAttemptInput",
    "GradeAttemptResult",
    "OfferedMessage",
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
    "blob_ref",
    "durable_client",
    "generate_payload_bundle_activity",
    "grade_attempt_activity",
    "hidden_seal_id",
    "kernel_activities",
    "protocol_error_code",
    "run_stream_worker",
    "seal_attempt_activity",
    "start_stream",
    "stream_replayer",
    "stream_worker",
    "temporal_home",
]
