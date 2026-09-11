"""The Activities the stream depends on: two stand-ins, two that are not, and the fork's four.

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

The fork's four are real and none of them is a stand-in. One reads the objects a fork requires
before its barrier commits, one creates a child under its derived identity and never a second one,
one asks a parent for the row it committed about a child, and one reads a child's own two objects
and resolves the cell that child was selected for. Two of them reach the service rather than the
store, which is the other reason they are here: a workflow may open no file and it may call no
client either.

Everything that will one day be I/O is already on this side of the line. The workflow computes
the submission digest from what :func:`seal_attempt_activity` returns and never opens a file,
a socket, or a clock of its own.
"""

from __future__ import annotations

from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, List, Optional

from temporalio import activity
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import ApplicationError, WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

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
    ForkAvailability,
    ForkAvailabilityInput,
    ForkChildStarted,
    ForkOriginVerified,
    ForkPreparation,
    ForkPreparationInput,
    GeneratePayloadBundleInput,
    GradeAttemptInput,
    GradeAttemptResult,
    PayloadBundle,
    PayloadCandidateResult,
    SealAttemptInput,
    SealAttemptResult,
    SelectedSourceReference,
    StartForkChildInput,
    StreamStart,
    VerifyBlobsInput,
    VerifyForkOriginInput,
)

SEAL_ATTEMPT = "shogym.protocol_v2.SealAttemptActivity"
GRADE_ATTEMPT = "shogym.protocol_v2.GradeAttemptActivity"
GENERATE_PAYLOAD_BUNDLE = "shogym.protocol_v2.GeneratePayloadBundleActivity"
VERIFY_BLOBS = "shogym.protocol_v2.VerifyBlobsActivity"
FORK_AVAILABILITY = "shogym.protocol_v2.ForkAvailabilityActivity"
START_FORK_CHILD = "shogym.protocol_v2.StartForkChildActivity"
VERIFY_FORK_ORIGIN = "shogym.protocol_v2.VerifyForkOriginActivity"
PREPARE_FORK_CHILD = "shogym.protocol_v2.PrepareForkChildActivity"

#: The workflow type a fork creates its children as, named rather than imported: the workflow
#: module reads this one, so a name imported the other way would close the cycle.
STREAM_WORKFLOW_TYPE = "ShogymStreamV2"
#: The Query a gated child asks its parent, by the name the workflow registers it under, for the
#: same reason. A controller reads the fork's own status through the typed runtime call instead.
FORK_CHILD_QUERY = "fork_child"

#: What an origin reading asked after its parent's answers stopped standing comes back as. It is a
#: name rather than a literal because the child that receives it decides on it: the window does not
#: reopen, so this is the one failure of that reading a child records instead of asking again.
EXPIRED_AUTHORITY_FAILURE = "ExpiredAuthority"

#: What a reading that authenticated a disagreement comes back as. It is a name rather than a
#: literal because the parent that receives it decides on it: the reading is permanent, so the fork
#: ends on it rather than asking again or replacing what is already there.
ORIGIN_DISAGREEMENT_FAILURE = "OriginDisagreement"
#: What a read of a name the store cannot produce the bytes for comes back as. It is a fact about
#: the store rather than about the obligation, which is why a preparation turns it into the name it
#: could not produce rather than letting it out as a failure of the call.
UNAVAILABLE_EVIDENCE_FAILURE = "UnavailableEvidence"

#: How long an origin question waits for an answer before it is retried.
_QUERY_TIMEOUT = timedelta(seconds=10)

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
            str(error), type=UNAVAILABLE_EVIDENCE_FAILURE, non_retryable=True
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


@activity.defn(name=FORK_AVAILABILITY)
async def fork_availability_activity(request: ForkAvailabilityInput) -> ForkAvailability:
    """Read the three objects one fork requires, and measure what the store produced.

    The manifest is read under its commitment and both eligible bodies under that manifest's own
    entries, because none of the three is in an ordinary claim's verified set and the committed
    identities prove what was published rather than that the bytes are still there. Nothing here
    decides anything: what comes back is which names produced bytes and how many, and the parent
    that asked is what refuses a barrier over an object nobody could read.
    """
    store = FilesystemBlobStore(Path(request.blob_root))
    asked = [request.source_commitment, *request.body_references]
    present: List[str] = []
    missing: List[str] = []
    measured: List[int] = []
    for reference in asked:
        try:
            measured.append(len(store.read(reference)))
        except WireFormatError:
            missing.append(reference)
            continue
        present.append(reference)
    return ForkAvailability(present=present, missing=missing, measured_bytes=measured)


@activity.defn(name=START_FORK_CHILD)
async def start_fork_child_activity(request: StartForkChildInput) -> ForkChildStarted:
    """Create one child under its derived identity, or resolve the one that is already there.

    The reuse policy is set rather than left to the default, which allows a duplicate: a child that
    failed its first activation is an existing child and not permission to create a replacement
    under its id, and a closed failed child stays failed. A duplicate is therefore resolved rather
    than started, and what is resolved is verified to be this child before it is adopted.
    """
    client = activity.client()
    try:
        handle = await client.start_workflow(
            STREAM_WORKFLOW_TYPE,
            request.start,
            id=request.child_workflow_id,
            task_queue=request.task_queue,
            id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
        )
    except WorkflowAlreadyStartedError:
        return ForkChildStarted(
            child_ordinal=request.child_ordinal,
            child_workflow_id=request.child_workflow_id,
            child_run_id=await _the_child_already_there(request),
            created=False,
        )
    return ForkChildStarted(
        child_ordinal=request.child_ordinal,
        child_workflow_id=request.child_workflow_id,
        child_run_id=handle.result_run_id or handle.first_execution_run_id or "",
        created=True,
    )


async def _the_child_already_there(request: StartForkChildInput) -> str:
    """Return the run an existing child was first created under, having checked it is this child.

    The comparison is against that original execution rather than against whatever runs under the
    identity now: a child that has since continued as new carries a different carrier under the
    same id, and comparing the latest run would refuse a child that is lawfully further on. What is
    compared is how that first execution was created, which is the workflow type, the task queue
    and the start argument.

    The start is compared as the converter's own complete payload for the value the history holds
    against its complete payload for the value this request carries, which is the one comparison
    that is total: the recorded argument is read back through the converter first, so a codec that
    writes different bytes for one value each time is decoded away rather than read as a
    disagreement, and neither side has to be a shape any particular digest is defined over.

    The whole payload is compared and not the body inside it, because a converter is free to keep
    part of what it encodes beside the body: two starts that differ in a member a converter puts in
    the metadata have equal bodies, and adopting an execution on that comparison would confirm an
    execution created from a start this fork never built.

    A disagreement here is authenticated and permanent. Something else is running under this
    child's derived identity, and the answer to that is never to replace it: the fork keeps the
    identity, records the child unconfirmed, and stops.
    """
    client = activity.client()
    described = await client.get_workflow_handle(request.child_workflow_id).describe()
    run_id = described.raw_info.first_run_id or described.run_id
    handle = client.get_workflow_handle(request.child_workflow_id, run_id=run_id)
    started = None
    async for event in handle.fetch_history_events(page_size=1):
        started = event.workflow_execution_started_event_attributes
        break
    if started is None or not started.workflow_type.name:
        raise ApplicationError(
            f"the first execution of {request.child_workflow_id!r} could not be read, so what "
            "runs under this child's identity is not established either way",
            type="UnreadableChild",
        )
    if (
        started.workflow_type.name != STREAM_WORKFLOW_TYPE
        or started.task_queue.name != request.task_queue
    ):
        raise ApplicationError(
            f"{request.child_workflow_id!r} was created as {started.workflow_type.name!r} on "
            f"{started.task_queue.name!r}, and this child is a {STREAM_WORKFLOW_TYPE!r} on "
            f"{request.task_queue!r}",
            type=ORIGIN_DISAGREEMENT_FAILURE,
            non_retryable=True,
        )
    converter = client.data_converter.payload_converter
    try:
        [original] = await client.data_converter.decode_wrapper(
            started.input, [StreamStart]
        )
        held = _the_whole_payload(converter.to_payloads([original])[0])
    except Exception as error:  # noqa: BLE001 - anything unreadable here is another execution
        raise ApplicationError(
            f"{request.child_workflow_id!r} was created from something other than a start this "
            "fork built",
            type=ORIGIN_DISAGREEMENT_FAILURE,
            non_retryable=True,
        ) from error
    asked = _the_whole_payload(converter.to_payloads([request.start])[0])
    if held != asked:
        raise ApplicationError(
            f"{request.child_workflow_id!r} was created from the start "
            f"{sha256(held).hexdigest()[:16]}, and this child's start is "
            f"{sha256(asked).hexdigest()[:16]}",
            type=ORIGIN_DISAGREEMENT_FAILURE,
            non_retryable=True,
        )
    return run_id


def _the_whole_payload(payload: Any) -> bytes:
    """Return everything one encoded value is, the body and what the converter kept beside it.

    A payload is its metadata and its body, and both are what a converter made of the value: what
    an encoding names and anything a converter chose to keep out there belong to the value exactly
    as the bytes in the body do. The serialization is asked for in a fixed order so that two
    encodings of one value are one string of bytes.
    """
    return payload.SerializeToString(deterministic=True)


@activity.defn(name=VERIFY_FORK_ORIGIN)
async def verify_fork_origin_activity(request: VerifyForkOriginInput) -> ForkOriginVerified:
    """Ask the parent a gated child's lineage names for the row it committed about that child.

    The read is an Activity rather than a workflow call because the SDK's external workflow handle
    exposes signal and cancel and no Query at all, and it is recorded in the child's history so a
    replay performs no fresh client I/O and reaches the same answer. It is pinned to the exact
    preparing execution, so a parent replaced under the same identity answers for the execution
    that prepared this child.

    The three ways it can fail are kept apart. A parent, a Worker or a history that cannot be read
    is infrastructure and is retried while the child stays gated. A history the service has already
    deleted is expired authority, which is its own result and neither of the other two: the window
    a parent's answers stand in does not reopen, so the answer comes back once instead of being
    asked for again for ever, and what the child does with it is record it and stay gated. A parent
    that answers and names no such child is authenticated, so that is permanent and never becomes a
    retry.
    """
    handle = activity.client().get_workflow_handle(
        request.parent_workflow_id, run_id=request.parent_run_id
    )
    try:
        answer: ForkOriginVerified = await handle.query(
            FORK_CHILD_QUERY,
            args=[request.fork_id, request.child_ordinal],
            result_type=ForkOriginVerified,
            rpc_timeout=_QUERY_TIMEOUT,
        )
    except RPCError as error:
        if error.status is RPCStatusCode.NOT_FOUND:
            raise ApplicationError(
                f"the execution {request.parent_run_id} that prepared this child can no longer be "
                "read, so its answer window has closed",
                type=EXPIRED_AUTHORITY_FAILURE,
                non_retryable=True,
            ) from error
        raise
    if answer.record.child_workflow_id != request.child_workflow_id:
        raise ApplicationError(
            f"the parent recorded its child {request.child_ordinal} as "
            f"{answer.record.child_workflow_id!r} and this execution runs as "
            f"{request.child_workflow_id!r}",
            type=ORIGIN_DISAGREEMENT_FAILURE,
            non_retryable=True,
        )
    return answer


@activity.defn(name=PREPARE_FORK_CHILD)
async def fork_preparation_activity(request: ForkPreparationInput) -> ForkPreparation:
    """Read one child's two objects, and resolve the cell its parent selected for it.

    The manifest is read again here under its own commitment, which is what a bare call of the
    resolver would not do: the resolver opens the store for the selected body alone, so a manifest
    lost after the child's successful claim would let a child publish readiness over a source
    object nobody could produce. The earlier claim is evidence of an earlier read and not of this
    one.

    What resolves the body is the payload route's own resolver, wrapped rather than reimplemented,
    so a child's candidate comes back through the same code a fresh capture's does and the checks
    on the way back are the ones that route already names.

    Nothing here decides anything. A name the store cannot produce comes back as a name, and the
    child that asked is what says which refusal that is and which row to write about it. That
    holds whichever of the two reads meets the absence: the objects are checked once before the
    resolver runs and the selected body is opened again inside it, and an object lost in between
    is the same fact about the store as one lost before either.
    """
    selected = request.payload.selected
    if selected is None:
        raise ApplicationError(
            "a child prepares one committed cell and this request selected none",
            type="SelectedSourceRequired",
            non_retryable=True,
        )
    policy = _renderer_for(request.payload.policy_digest)
    if policy.exposure != ARTIFACT:
        raise ApplicationError(
            f"a child's body is an environment's own committed bytes and {policy.policy_name} "
            "renders one instead",
            type="PolicyViolation",
            non_retryable=True,
        )
    store = FilesystemBlobStore(Path(request.blob_root))
    missing = store.unverified([request.source_commitment, selected.body_sha256])
    if missing:
        return ForkPreparation(missing=missing)
    try:
        bundle = _resolved_bundle(request.payload, policy)
    except ApplicationError as error:
        if error.type != UNAVAILABLE_EVIDENCE_FAILURE:
            raise
        # The object was there when this read the store and gone when the resolver read it again.
        # That window is the one a repair covers, and what a repair answers is a name: an object
        # lost between the two reads is the same absence as one lost before the first, so it comes
        # back as the name it could not produce rather than as a failure of the call.
        return ForkPreparation(missing=[selected.body_sha256])
    return ForkPreparation(missing=[], bundle=bundle)


def kernel_activities() -> list:
    """Return the Activities a stream Worker registers."""
    return [
        seal_attempt_activity,
        grade_attempt_activity,
        generate_payload_bundle_activity,
        verify_blobs_activity,
    ]


def fork_activities() -> list:
    """Return the infrastructure Activities one fork adds, which are registered explicitly.

    A Worker takes whatever Activity list it is handed wholesale, and an environment that brings
    its own terminal hands over only its seal, its grade, the payload bundle Activity and the blob
    verification Activity. These four are none of those, so they are added beside whatever an
    environment supplied rather than being left to a caller to remember.
    """
    return [
        fork_availability_activity,
        start_fork_child_activity,
        verify_fork_origin_activity,
        fork_preparation_activity,
    ]
