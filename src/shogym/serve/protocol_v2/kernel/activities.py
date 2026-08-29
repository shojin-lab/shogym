"""The Activities the stream depends on: three stand-ins, and one that is not.

The seal transaction's three compute deterministically from their inputs, hold no state between
calls, and reach no environment or grader. What is not a stand-in is their shape: each already
carries the attempt ID, the seal ID, the hashes, the protocol version, and the blob references
a real implementation needs, so replacing a body here does not move a boundary.

:func:`verify_blobs_activity` is real. Verifying a reference means reading the object and
hashing it, and the workflow may not open a file, so the read lives here and the decision the
read supports lives there.

Everything that will one day be I/O is already on this side of the line. The workflow computes
the submission digest from what :func:`seal_attempt_activity` returns and never opens a file,
a socket, or a clock of its own.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from temporalio import activity

from shogym.serve.protocol_v2 import FilesystemBlobStore, Payload, blob_ref, visible_bytes
from shogym.serve.protocol_v2.kernel.messages import (
    BlobsVerified,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    GradeAttemptResult,
    PayloadBundle,
    PayloadCandidate,
    SealAttemptInput,
    SealAttemptResult,
    VerifyBlobsInput,
)

SEAL_ATTEMPT = "shogym.protocol_v2.SealAttemptActivity"
GRADE_ATTEMPT = "shogym.protocol_v2.GradeAttemptActivity"
GENERATE_PAYLOAD_BUNDLE = "shogym.protocol_v2.GeneratePayloadBundleActivity"
VERIFY_BLOBS = "shogym.protocol_v2.VerifyBlobsActivity"

KERNEL_CELL = "graded"
KERNEL_MATCH_GROUP = "kernel"
KERNEL_RENDERER = "kernel-receipt-1"


@activity.defn(name=SEAL_ATTEMPT)
async def seal_attempt_activity(request: SealAttemptInput) -> SealAttemptResult:
    """Capture the canonical submission and seal the environment under ``seal_id``.

    A real environment deduplicates on ``seal_id``, so an Activity retry finds the seal it
    already made rather than making a second one. This one is a pure function of its input,
    which deduplicates for the same reason and by the same key.
    """
    submission = "\n".join(
        f"{name}={request.native_arguments[name]!r}" for name in sorted(request.native_arguments)
    )
    return SealAttemptResult(
        attempt_id=request.attempt_id,
        seal_id=request.seal_id,
        canonicalization_version=request.canonicalization_version,
        canonical_submission_text=submission,
        canonical_submission=blob_ref(submission),
        environment_recovery_token=sha256(request.seal_id.encode("utf-8")).hexdigest(),
    )


@activity.defn(name=GRADE_ATTEMPT)
async def grade_attempt_activity(request: GradeAttemptInput) -> GradeAttemptResult:
    """Score the sealed evidence. An unreadable submission is a result, not a failure."""
    decode_state = "decoded" if request.canonical_submission_text else "ambiguous_zero"
    score = 1 if decode_state == "decoded" else 0
    return GradeAttemptResult(
        attempt_id=request.attempt_id,
        seal_id=request.seal_id,
        score=score,
        decode_state=decode_state,
        evidence=blob_ref(f"{request.seal_id}:{decode_state}"),
    )


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def generate_payload_bundle_activity(request: GeneratePayloadBundleInput) -> PayloadBundle:
    """Build every candidate for one obligation and measure what was built.

    The measurements are the point. A family gate compares complete model-visible byte counts
    across cells, so the bundle reports the serialized result's hash and length, not the body's.
    """
    body = f"receipt {request.payload_position} for {request.submission_digest[:16]}"
    result = Payload(
        message_id=request.payload_message_id,
        attempt_id=request.attempt_id,
        body=body,
    )
    serialized = visible_bytes(result)
    candidate = PayloadCandidate(
        cell=KERNEL_CELL,
        renderer_id=KERNEL_RENDERER,
        match_group=KERNEL_MATCH_GROUP,
        body=body,
        inner_sha256=sha256(body.encode("utf-8")).hexdigest(),
        visible_sha256=sha256(serialized).hexdigest(),
        visible_byte_count=len(serialized),
    )
    return PayloadBundle(
        attempt_id=request.attempt_id,
        payload_position=request.payload_position,
        submission_digest=request.submission_digest,
        candidates=[candidate],
    )


@activity.defn(name=VERIFY_BLOBS)
async def verify_blobs_activity(request: VerifyBlobsInput) -> BlobsVerified:
    """Read the store and say which references it can produce the exact bytes for.

    This one is not a stand-in. Reading a blob is the verification, so the answer is a fact
    about installed bytes rather than a claim about them, and it is computed here because the
    workflow that acts on it may not open a file.
    """
    store = FilesystemBlobStore(Path(request.blob_root))
    unverified = store.unverified(request.references)
    return BlobsVerified(
        verified=[digest for digest in request.references if digest not in unverified],
        unverified=unverified,
    )


def kernel_activities() -> list:
    """Return the Activities a stream Worker registers."""
    return [
        seal_attempt_activity,
        grade_attempt_activity,
        generate_payload_bundle_activity,
        verify_blobs_activity,
    ]
