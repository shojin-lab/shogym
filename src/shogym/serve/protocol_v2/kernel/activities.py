"""The Activities the stream depends on: two stand-ins, and two that are not.

The seal and the grade compute deterministically from their inputs, hold no state between calls,
and reach no environment. What is not a stand-in is their shape: each already carries the attempt
ID, the seal ID, the hashes, the protocol version, and the blob references a real implementation
needs, so replacing a body here does not move a boundary. The grade also says what it is, because
a generation that publishes its number to an agent has to be able to tell that the number is a
fact about the shape of a filing rather than about the work in it.

:func:`generate_payload_bundle_activity` is real. What a body may contain is a registered policy
rather than a convention, so the request names the policy and this renders what that policy
declares, echoing the descriptor and the renderer back with the candidate. Under an artifact
policy there is nothing to render: the body is an environment's own committed bytes for one
declared cell, so the request carries that single reference, the store is read here because the
workflow may not open a file, and the object comes back exactly as it was installed or not at
all.

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
from typing import Optional

from temporalio import activity
from temporalio.exceptions import ApplicationError

from shogym.serve.protocol_v2 import (
    BlobRef,
    FilesystemBlobStore,
    Payload,
    blob_ref,
    visible_bytes,
)
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.policy import (
    ARTIFACT,
    KERNEL_MATCH_GROUP,
    KERNEL_STAND_IN_GRADE,
    LEGACY_PLACEHOLDER_V1,
    POLICIES,
    PayloadPolicy,
    PolicyViolation,
    render_body,
)
from shogym.serve.protocol_v2.kernel.messages import (
    BlobsVerified,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    GradeAttemptResult,
    PayloadBundle,
    PayloadCandidateResult,
    SealAttemptInput,
    SealAttemptResult,
    SelectedSourceReference,
    VerifyBlobsInput,
)

SEAL_ATTEMPT = "shogym.protocol_v2.SealAttemptActivity"
GRADE_ATTEMPT = "shogym.protocol_v2.GradeAttemptActivity"
GENERATE_PAYLOAD_BUNDLE = "shogym.protocol_v2.GeneratePayloadBundleActivity"
VERIFY_BLOBS = "shogym.protocol_v2.VerifyBlobsActivity"

KERNEL_CELL = "graded"
# The renderer a request with no policy in it gets, which is the one a legacy history recorded.
# It is read off the policy rather than repeated, so the name a build serves under and the name
# the policy declares cannot drift apart.
KERNEL_RENDERER = LEGACY_PLACEHOLDER_V1.renderer_id


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
    """Score the sealed evidence. An unreadable submission is a result, not a failure.

    The verdict goes into the run's store before the reference naming it comes back, which a
    real grader has to do as well: the generation commits that reference beside the score, and a
    name the store cannot produce the bytes for is not evidence of anything. A generation given
    no store gets the reference and nothing to resolve it in.
    """
    decode_state = "decoded" if request.canonical_submission_text else "ambiguous_zero"
    score = 1 if decode_state == "decoded" else 0
    return GradeAttemptResult(
        attempt_id=request.attempt_id,
        seal_id=request.seal_id,
        score=score,
        decode_state=decode_state,
        evidence=_installed(request.blob_root, f"{request.seal_id}:{decode_state}"),
        grade=KERNEL_STAND_IN_GRADE,
    )


def _installed(blob_root: Optional[str], text: str) -> BlobRef:
    """Return the reference that names ``text``, with those bytes where the name resolves."""
    if blob_root is None:
        return blob_ref(text)
    return FilesystemBlobStore(Path(blob_root)).put(text.encode("utf-8"), media_type="text/plain")


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def generate_payload_bundle_activity(request: GeneratePayloadBundleInput) -> PayloadBundle:
    """Build every candidate for one obligation and measure what was built.

    The measurements are the point. A family gate compares complete model-visible byte counts
    across cells, so the bundle reports the serialized result's hash and length, not the body's.

    Which body is built is the request's to say and never this function's. The policy the
    obligation was resolved to names the renderer, and a digest this build does not implement is
    a failure it will not be retried on: a Worker that cannot render what a generation asked for
    must not serve what it can render instead.

    An artifact policy is resolved rather than rendered. Its body is the environment's own
    committed bytes for the one cell this obligation was selected for, so this reads that entry
    out of the store and returns it exactly, and there is no body it could build instead.
    """
    policy = _renderer_for(request.policy_digest)
    if policy.exposure == ARTIFACT:
        return _resolved_bundle(request, policy)
    try:
        body = render_body(
            policy,
            grade=request.public_grade,
            payload_position=request.payload_position,
            submission_digest=request.submission_digest,
        )
    except PolicyViolation as violation:
        raise ApplicationError(
            str(violation), type="PolicyViolation", non_retryable=True
        ) from violation
    result = Payload(
        message_id=request.payload_message_id,
        attempt_id=request.attempt_id,
        body=body,
    )
    serialized = visible_bytes(result)
    candidate = PayloadCandidateResult(
        cell=request.cell or KERNEL_CELL,
        renderer_id=policy.renderer_id,
        match_group=KERNEL_MATCH_GROUP,
        body=body,
        inner_sha256=sha256(body.encode("utf-8")).hexdigest(),
        visible_sha256=sha256(serialized).hexdigest(),
        visible_byte_count=len(serialized),
        renderer_version="" if not request.policy_digest else policy.renderer_version,
        policy_digest=request.policy_digest,
    )
    return PayloadBundle(
        attempt_id=request.attempt_id,
        payload_position=request.payload_position,
        submission_digest=request.submission_digest,
        candidates=[candidate],
    )


def _resolved_bundle(
    request: GeneratePayloadBundleInput, policy: PayloadPolicy
) -> PayloadBundle:
    """Return the one committed cell this obligation was selected for, exactly as it was stored.

    Nothing here decides anything. The reference is the request's, the store is the run's, and
    what comes back is the object under that name or a refusal: a missing cell never licenses a
    search for another one, and a body this function assembled would be a receipt nobody
    committed.

    The grade and the filing's text are refused rather than ignored. An artifact request carries
    neither, so a call that arrived with one was composed by something that thinks this route
    renders, and reading past that would be reading a request this build does not answer.
    """
    selected = request.selected
    if selected is None:
        raise ApplicationError(
            f"{policy.policy_name} delivers a committed source cell and this request selected "
            "none, and there is no body to render in its place",
            type="SelectedSourceRequired",
            non_retryable=True,
        )
    if request.public_grade is not None or request.canonical_submission_text:
        raise ApplicationError(
            f"{policy.policy_name} copies an environment's own bytes, and this request carries "
            "the filing or the grade a renderer would have been given",
            type="PolicyViolation",
            non_retryable=True,
        )
    if selected.cell != request.cell or selected.cell not in policy.cells:
        raise ApplicationError(
            f"{policy.policy_name} declares the cells {list(policy.cells)}, this obligation was "
            f"assigned {request.cell!r} and the selected reference is for {selected.cell!r}",
            type="SelectedSourceMismatch",
            non_retryable=True,
        )
    body = _decoded(selected)
    result = Payload(
        message_id=request.payload_message_id,
        attempt_id=request.attempt_id,
        body=body,
    )
    serialized = visible_bytes(result)
    return PayloadBundle(
        attempt_id=request.attempt_id,
        payload_position=request.payload_position,
        submission_digest=request.submission_digest,
        candidates=[
            PayloadCandidateResult(
                cell=selected.cell,
                renderer_id=policy.renderer_id,
                match_group=KERNEL_MATCH_GROUP,
                body=body,
                inner_sha256=sha256(body.encode("utf-8")).hexdigest(),
                visible_sha256=sha256(serialized).hexdigest(),
                visible_byte_count=len(serialized),
                renderer_version=policy.renderer_version,
                policy_digest=request.policy_digest,
                source_commitment=selected.source_commitment,
                body_reference=selected.body_sha256,
                resolver_id=policy.resolver_id,
                resolver_version=policy.resolver_version,
            )
        ],
    )


def _decoded(selected: SelectedSourceReference) -> str:
    """Return the exact text one committed cell holds, or say why there is none to return.

    A name the store cannot produce the bytes for is unavailable evidence, which is a fact about
    the store rather than about this obligation. Bytes that are not the size or the encoding the
    contract registered are corrupt evidence, which is a fact about the object: the store
    verifies a digest without knowing what the object was, so the shape is checked where the
    contract that fixed it is known.
    """
    store = FilesystemBlobStore(Path(selected.blob_root))
    try:
        raw = store.read(selected.body_sha256)
    except WireFormatError as error:
        raise ApplicationError(
            str(error), type="UnavailableEvidence", non_retryable=True
        ) from error
    if len(raw) != selected.body_size:
        raise ApplicationError(
            f"the contract {selected.contract_id} fixes a body at {selected.body_size} bytes and "
            f"the object under this reference is {len(raw)}",
            type="CorruptEvidence",
            non_retryable=True,
        )
    try:
        return raw.decode(selected.body_encoding)
    except (LookupError, UnicodeDecodeError) as error:
        raise ApplicationError(
            f"the contract {selected.contract_id} publishes {selected.body_encoding!r} and this "
            "object is not that",
            type="CorruptEvidence",
            non_retryable=True,
        ) from error


def _renderer_for(digest: str) -> PayloadPolicy:
    """Return the policy a request names, and refuse one this build cannot render.

    An empty digest is a request written before a generation carried a policy. It renders the
    placeholder, which is the body that history recorded, so an Activity a stopped generation
    left scheduled comes back with the bytes its own run was going to serve.
    """
    if not digest:
        return LEGACY_PLACEHOLDER_V1
    policy = POLICIES.get(digest)
    if policy is None:
        raise ApplicationError(
            f"this build implements no payload policy under {digest[:16]}, and a renderer it "
            "could substitute is not a renderer this generation asked for",
            type="UnknownPayloadPolicy",
            non_retryable=True,
        )
    return policy


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
