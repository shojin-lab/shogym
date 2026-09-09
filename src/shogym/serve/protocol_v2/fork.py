"""The controller's half of one fork: what it reads, what it asks the harness for, what it keeps.

A fork is made of two halves that cannot do each other's work. The platform freezes nothing and
restores nothing, because settling a provider call, snapshotting a container and resuming one are
things it has no way to do; the harness answers for no generation, because the cursor, the digest
and the attestation a fork is cut over are the stream's own record. This module is where the two
meet: the operations a run controller performs in order, each with a typed result at a version, a
durable key it is answered under, and an outcome a controller that came back after a crash reads
rather than repeats.

The order is checkpoint, attachment, preparation, equality, binding, release, and each step is
evidence for the next. What is kept is what a later step or a later process needs: the checkpoint
one fork was cut over, the ownership a claim installed, the readiness a child published, the
comparison a restored container passed, the container each child was bound to, and which children
were resumed. What is not kept is anything the harness holds, because a controller reading its own
memory for a witness would be comparing a value against itself.

The harness half is a protocol here and an implementation elsewhere. An adapter that cannot supply
an operation says so with its reason, and this refuses rather than approximating it: a fork over a
freeze nobody performed measures something nobody asked for.

Nothing here is an agent-visible protocol event. Every refusal is a controller refusal, no message
kind or code reaches a model, and the children this drives are ordinary generations pulling
ordinary payloads.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple, Type, TypeVar, Union

from temporalio.api.common.v1 import Payload
from temporalio.client import Client
from temporalio.converter import default as default_converter

from shogym.serve.episode import ServedEpisode
from shogym.serve.protocol_v2.artifact import read_source_artifact
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore, create_directory, flush_directory
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.gateway import (
    EnvironmentTerminal,
    EpisodeOpener,
    RefusalSink,
    StreamGateway,
    attach_gateway,
)
from shogym.serve.protocol_v2.identity import length_prefixed
from shogym.serve.protocol_v2.kernel import (
    FORK_CONFIGURATION_VIOLATION,
    FORK_EXPIRED_AUTHORITY,
    FORK_INTERRUPTED_COPY,
    FORK_INVALID_CHECKPOINT,
    FORK_NOT_QUIET,
    FORK_ORIGIN_DISAGREEMENT,
    FORK_REPAIRABLE_ABSENCE,
    FORK_UNREADABLE_PARENT,
    FORK_WITNESS_MISMATCH,
    CheckpointComponent,
    CheckpointEvidence,
    CheckpointManifest,
    CheckpointEvidenceAnswer,
    ChildReady,
    ForkChildPlan,
    ForkChildReceipt,
    ForkReceipt,
    ForkRefused,
    ForkRequest,
    StreamStart,
    check_checkpoint_evidence,
    check_checkpoint_manifest,
    check_child_ready,
    check_fork_receipt,
    checkpoint_manifest_reference,
    complete_start_digest,
    fork_can_be_retried,
    fork_preparation_epoch,
    fork_preparation_operation_identity,
    fork_refusal,
    origin_unverified,
    unpack_carrier,
)
from shogym.serve.protocol_v2.rundir import OWN_DATABASE_ROOT, run_directory_of

#: The shape of every result these operations record. The six are one family: they are written by
#: one build, read by one build, and a controller that could read five of them and not the sixth
#: would hold a lineage it cannot finish, so they move together or not at all.
FORK_OPERATION_SCHEMA_VERSION = 1

#: What the file one outcome is kept in is called, under the directory the lineage shares.
OPERATIONS_DIRECTORY = "fork"

# How long a controller waits for a child to authorize the lineage it carries before it gives up.
# A gated child owns nothing until its own history has read the parent's record of it, which is one
# Activity and its retries, so the wait is short and the bound is generous.
_GATE_WAIT_SECONDS = 0.05
_GATE_TRIES = 600

_CONVERTER = default_converter().payload_converter


class UnsupportedCapability(RuntimeError):
    """The harness cannot supply one of the things a fork needs, and says which and why.

    An adapter that cannot settle, snapshot, isolate or resume part of its own state reports it
    here rather than doing something close to it, because a container that was nearly frozen is a
    measurement of something nobody registered. The controller refuses on it and records the run
    incomplete under the reason it was given.
    """

    def __init__(self, capability: str, reason: str) -> None:
        self.capability = capability
        self.reason = reason
        super().__init__(f"this harness cannot {capability}: {reason}")


def checkpoint_key(*, parent_workflow_id: str, attestation_id: str) -> str:
    """The name one checkpoint retrieval is answered under.

    It is the generation and the attestation that committed the acknowledgement, which is what
    makes a retrieval the same one twice: a controller asking again after a crash is asking about
    one acknowledgement of one generation, and every other value in the answer follows from those.
    """
    return _operation_key("checkpoint", parent_workflow_id, attestation_id)


def attachment_key(*, fork_id: str, child_workflow_id: str) -> str:
    """The name one child's attachment is answered under."""
    return _operation_key("attachment", fork_id, child_workflow_id)


def preparation_key(*, fork_id: str, child_workflow_id: str, creation_epoch: int) -> str:
    """The name one child's preparation is answered under, which is the operation's own identity.

    The epoch in it is the one the child's first preparation was created under and never the one
    the child currently holds, so a repair that steps ownership repeats one logical preparation
    rather than opening a second operation nothing joins to the first. That epoch is the child's
    to say and this side never derives it: a controller that attached again holds a later one, so
    the name is read off the readiness the child publishes and checked here rather than minted.
    """
    return fork_preparation_operation_identity(fork_id, child_workflow_id, creation_epoch)


def refused_preparation_key(*, fork_id: str, child_workflow_id: str) -> str:
    """The name a preparation refused before it froze an episode keeps its decision under.

    A refusal recorded inside an episode is kept under that episode's own identity, like the
    readiness it stands in place of, because the two are outcomes of one operation and either has
    to be retrievable under the identity the child froze. This is the other case and it is a real
    one: a generation cut from no fork, or one owing no payload for the attempt its lineage names,
    is refused before any preparation exists to be named. The decision is still about one child of
    one fork, and that is what it is kept under.
    """
    return _operation_key("preparation", fork_id, child_workflow_id)


def equality_key(*, fork_id: str, child_workflow_id: str) -> str:
    """The name one child's restored container and its comparison are answered under."""
    return _operation_key("equality", fork_id, child_workflow_id)


def binding_key(*, fork_id: str, child_workflow_id: str) -> str:
    """The name one container-to-child binding is answered under."""
    return _operation_key("binding", fork_id, child_workflow_id)


def release_key(*, fork_id: str, child_workflow_id: str) -> str:
    """The name one child's release is answered under."""
    return _operation_key("release", fork_id, child_workflow_id)


def _operation_key(operation: str, *parts: str) -> str:
    """The durable name one outcome is written under, safe to be a file name.

    A workflow id is a path as often as not, so the parts are hashed rather than joined: the
    operation stays readable at the front of the name and what it was about is the digest behind
    it, which is exactly as unique as the parts are.
    """
    preimage = b"".join(
        length_prefixed(part.encode("utf-8")) for part in (operation, *parts)
    )
    return f"{operation}.{sha256(preimage).hexdigest()[:32]}"


# What the harness answers with. These are the adapter's own values rather than the controller's
# records: a restored container says which transcript it reproduced, a comparison says what it was
# allowed to differ in and what it actually differs in, and a resumed container says what its first
# act returned. Each of them becomes part of a record below, under a key.


@dataclass(frozen=True)
class RestoredContainer:
    """One container copy, and what it attests it came back holding.

    The manifest's own assertions are preflight evidence and this is the evidence afterwards: a
    restore that reproduced another transcript, or the right transcript without the acknowledgement
    entry in it, is the failure the two witnesses exist to catch. Neither side proves the other's
    half, so the adapter attests and the controller compares.
    """

    container_id: str
    transcript_reference: str
    acknowledgement_entry_sha256: str


@dataclass(frozen=True)
class ContainerEquality:
    """How one restored container compares against the parent it was copied from.

    ``allowlist_version`` is the version of the rebinding allowlist the comparison was made under,
    and it is the adapter's: what a container is allowed to differ from its parent in is a fact
    about containers. ``rebound`` is what the allowlist admitted and ``differences`` is what it did
    not, so a comparison that admits nothing still says what it looked at.
    """

    allowlist_version: str
    rebound: List[str]
    differences: List[str]


@dataclass(frozen=True)
class ResumedContainer:
    """One container let go, and the message the first act of the agent inside it returned.

    A child's first act is an ordinary pull and nothing is injected to bring it about, so what
    comes back is the payload that child's own disposition names.
    """

    container_id: str
    first_message_id: str


class ForkAdapter(Protocol):
    """The harness half of a fork, which this package calls and does not implement.

    An adapter freezes one container, restores one independently owned copy per child, compares
    each against the parent, binds it to the child generation it serves, and resumes it. Every one
    of those reaches inside a container, so the implementation is the run controller's and what is
    here is the shape it is called through.

    Any of them may raise :class:`UnsupportedCapability`, which is how an adapter that cannot do
    one of these says so rather than doing something close to it.
    """

    async def checkpoint(
        self, *, parent_workflow_id: str, attestation_id: str
    ) -> CheckpointManifest:
        """Return the immutable checkpoint this harness committed for that acknowledgement."""
        ...

    async def restore(
        self, *, fork_id: str, child_workflow_id: str, checkpoint_manifest_reference: str
    ) -> RestoredContainer:
        """Restore one container copy for that child, and attest what it came back holding."""
        ...

    async def compare(
        self, *, fork_id: str, child_workflow_id: str, container_id: str
    ) -> ContainerEquality:
        """Compare that container against the parent under the versioned rebinding allowlist."""
        ...

    async def bind(
        self, *, fork_id: str, child_workflow_id: str, container_id: str, consumer_id: str
    ) -> None:
        """Bind that container to that child generation, and to no other."""
        ...

    async def release(
        self, *, fork_id: str, child_workflow_id: str, container_id: str
    ) -> ResumedContainer:
        """Resume that container, so the agent inside it pulls."""
        ...


# What the controller keeps. One record per operation per key, each carrying the version of the
# shape it was written in, and each holding what a later step or a later process reads rather than
# what the step happened to have in hand.


@dataclass(frozen=True)
class CheckpointRetrieved:
    """The freeze one fork is cut over, with both witnesses named together.

    The stream side is what the generation recorded and the rest is what the harness committed, and
    the two agreeing is what this operation establishes: a checkpoint naming another cursor,
    another digest or another acknowledgement is refused here rather than at the barrier, before
    anything is fenced.
    """

    parent_workflow_id: str
    source_attempt_id: str
    attestation_id: str
    checkpoint_manifest_reference: str
    components: List[CheckpointComponent]
    transcript_reference: str
    acknowledgement_entry_sha256: str
    acknowledgement_message_id: str
    acknowledged_visible_sha256: str
    acknowledged_cursor: str
    projection_digest: str
    schema_version: int = FORK_OPERATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ChildAttached:
    """One child's directory, the manifest that says which generation lives in it, and its owner.

    ``retained_absent`` is the objects copied for retention that the parent's store could not
    produce. Nothing the child does depends on them, the oracle body being the case that matters,
    so they are recorded as an observation rather than refused: a row saying an attempt's evidence
    is unavailable would move that attempt's availability while its evidence is intact.
    """

    fork_id: str
    child_workflow_id: str
    run_directory: str
    configuration_hash: str
    complete_start_digest: str
    installed: List[str]
    retained_absent: List[str]
    ownership_epoch: int
    consumer_id: str
    schema_version: int = FORK_OPERATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ChildPrepared:
    """What one child's preparation came to: its readiness, or the reason it will not have one.

    A repairable absence is neither, and it is not recorded: an object the store lost and a
    controller can put back leaves one logical preparation to be repeated under the owner that
    follows, and a record of it would answer that repetition with the failure it repaired.

    ``preparation_operation`` is the identity the child froze at its first attempt: the readiness
    carries it, and so does a refusal the child recorded inside an episode, which is the epoch that
    episode was created under rather than the one the child holds now. It is empty only where the
    refusal came before any episode existed, and that decision is kept under the fork and the child
    instead, because there was no operation for it to be about.
    """

    fork_id: str
    child_workflow_id: str
    preparation_operation: str
    ready: Optional[ChildReady] = None
    reason: str = ""
    schema_version: int = FORK_OPERATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ChildCompared:
    """One restored container, and how it compares against the parent it was copied from.

    The comparison is the adapter's and the transcript check is this side's: a container that came
    back reproducing another transcript, or the right transcript without the acknowledgement entry
    in it, differs in the one way the two witnesses exist to catch.
    """

    fork_id: str
    child_workflow_id: str
    container_id: str
    allowlist_version: str
    rebound: List[str]
    differences: List[str]
    equal: bool
    schema_version: int = FORK_OPERATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ChildBound:
    """One container bound to one child generation, and to no other."""

    fork_id: str
    child_workflow_id: str
    container_id: str
    consumer_id: str
    schema_version: int = FORK_OPERATION_SCHEMA_VERSION


@dataclass(frozen=True)
class ChildReleased:
    """What became of one child's container: resumed, or held and why.

    Only a resumption is kept. A hold is what the recorded comparisons say when they are read
    together, so a controller that comes back derives it again rather than reading a record that
    would have to be rewritten the moment the hold was lifted.
    """

    fork_id: str
    child_workflow_id: str
    container_id: str
    resumed: bool
    first_message_id: str = ""
    held: str = ""
    schema_version: int = FORK_OPERATION_SCHEMA_VERSION


_RESULT = TypeVar("_RESULT")


@dataclass(frozen=True)
class ForkOperations:
    """Where one lineage keeps the outcomes of its control operations, one file per key.

    A crash can land between any two of these operations, and what makes that survivable is that
    each of them is answered under a name derived from what it was about rather than from when it
    ran. So a controller that comes back reads what it already did: which container it restored,
    which epoch its claim installed, which children it resumed.

    A record is written once. One that says the same thing is adopted, because a repeat of an
    idempotent operation is how a lost reply is recovered, and one that says anything else is
    refused: two answers under one name is not a record of anything.
    """

    root: Path

    @classmethod
    def under(cls, run_root: Union[str, Path]) -> "ForkOperations":
        """The operations of the lineage whose generations live under ``run_root``."""
        return cls(Path(run_root) / OPERATIONS_DIRECTORY)

    def recorded(self, key: str, kind: Type[_RESULT]) -> Optional[_RESULT]:
        """The outcome kept under ``key``, or nothing where the operation has not run."""
        path = self.root / f"{key}.json"
        if not path.is_file():
            return None
        return self._read(path, kind)

    def record(self, key: str, result: _RESULT) -> _RESULT:
        """Keep ``result`` under ``key``, adopting one that says the same and refusing another."""
        path = self.root / f"{key}.json"
        if path.is_file():
            standing = self._read(path, type(result))
            if standing != result:
                raise WireFormatError(
                    f"the operation {key} was answered once already, and this answer is not that "
                    "one"
                )
            return standing
        create_directory(self.root)
        _publish(path, _written(result))
        return result

    def all_of(self, kind: Type[_RESULT]) -> List[_RESULT]:
        """Every outcome of one kind this lineage has kept, in the order their names sort."""
        if not self.root.is_dir():
            return []
        kept: List[_RESULT] = []
        for path in sorted(self.root.glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("kind") == kind.__name__:
                kept.append(self._read(path, kind))
        return kept

    def _read(self, path: Path, kind: Type[_RESULT]) -> _RESULT:
        """Read one kept outcome back, refusing a file that holds another kind or another shape.

        The outcome's own version is one of the shapes to admit and not the only one. A
        preparation carries a child's readiness inside it, that readiness is a declared shape with
        a version of its own, and what comes back out of a file is what a container is then
        restored and released against. So the value inside is admitted here, where it arrives,
        rather than by the live route a recovered outcome never takes.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("kind") != kind.__name__:
            raise WireFormatError(
                f"{path} holds {payload.get('kind')!r} and this read is for {kind.__name__!r}"
            )
        held = Payload()
        held.metadata["encoding"] = b"json/plain"
        held.data = json.dumps(payload["result"], sort_keys=True).encode("utf-8")
        result = _CONVERTER.from_payload(held, kind)
        version = getattr(result, "schema_version", FORK_OPERATION_SCHEMA_VERSION)
        if version != FORK_OPERATION_SCHEMA_VERSION:
            raise WireFormatError(
                f"this build reads a fork operation at version {FORK_OPERATION_SCHEMA_VERSION}, "
                f"and {path} is written at {version!r}"
            )
        _admit_what_it_carries(result)
        return result


def _admit_what_it_carries(result: Any) -> None:
    """Refuse a declared shape this build cannot read that arrived inside another one.

    One outcome carries another: a preparation holds the readiness a child published, which is a
    versioned shape and is refused at its own version wherever it is received. Reading it back
    from a file is one of those places, and the one where the check has most to do, because
    nothing later asks: what a recovered readiness becomes is the evidence a container is restored
    against and a child is released under.
    """
    if isinstance(result, ChildPrepared) and result.ready is not None:
        check_child_ready(result.ready)


def _written(result: Any) -> str:
    """One outcome as the file that holds it: what kind it is, and the value itself."""
    encoded = json.loads(_CONVERTER.to_payloads([result])[0].data.decode("utf-8"))
    return json.dumps({"kind": type(result).__name__, "result": encoded}, sort_keys=True) + "\n"


def _publish(path: Path, payload: str) -> None:
    """Put a file in place whole: write it beside itself, get it to the disk, then rename."""
    descriptor, staged = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged, path)
    except BaseException:
        Path(staged).unlink(missing_ok=True)
        raise
    flush_directory(path.parent)


# The operations, in the order one fork performs them. Each reads its own record first or writes
# one afterwards, and which of the two it is depends on what repeating the operation would cost: a
# restore and a resume happen once, and reading a checkpoint or claiming a directory again is how a
# lost reply is recovered.


class ParentOfAFork(Protocol):
    """What a controller reads the stream's half of a freeze through.

    A gateway is what a controller serving the parent already holds and a handle is what one
    driving it directly holds, and both answer out of the generation's own record rather than out
    of anything they remember. Nothing else about either of them is needed here.
    """

    async def checkpoint_evidence(self) -> CheckpointEvidenceAnswer:
        """The stream's half of a freeze, or the reason this generation has none."""
        ...


class ChildOfAFork(Protocol):
    """What a controller has one child build its own body through.

    It is the transport whose claim installed the ownership the preparation runs under, which is
    the gateway the attachment returned.
    """

    async def prepare_child(self, *, fork_id: str) -> ChildReady:
        """Build this child's body for the obligation it inherited, and read its readiness."""
        ...


async def retrieve_checkpoint(
    parent: ParentOfAFork, adapter: ForkAdapter, operations: ForkOperations
) -> CheckpointRetrieved:
    """Read both witnesses of one freeze and keep them, or refuse the pair that disagree.

    The stream is asked first, because what the harness committed is a claim about a moment the
    generation is the record of. The two are then compared field by field, before anything is
    fenced and before a child exists: a snapshot taken before the acknowledgement and a transcript
    written after it both hash correctly and name the right identifiers, and it is the manifest
    binding them to one cursor and one projection digest that says which moment they are of.
    """
    evidence = await _evidence(parent)
    key = checkpoint_key(
        parent_workflow_id=evidence.parent_workflow_id, attestation_id=evidence.attestation_id
    )
    kept = operations.recorded(key, CheckpointRetrieved)
    if kept is not None:
        return kept
    manifest = await adapter.checkpoint(
        parent_workflow_id=evidence.parent_workflow_id, attestation_id=evidence.attestation_id
    )
    try:
        check_checkpoint_manifest(manifest)
    except WireFormatError as error:
        raise ForkRefused(FORK_INVALID_CHECKPOINT, str(error)) from error
    _refuse_a_checkpoint_of_another_moment(manifest, evidence)
    return operations.record(
        key,
        CheckpointRetrieved(
            parent_workflow_id=evidence.parent_workflow_id,
            source_attempt_id=evidence.source_attempt_id,
            attestation_id=evidence.attestation_id,
            checkpoint_manifest_reference=checkpoint_manifest_reference(manifest),
            components=list(manifest.components),
            transcript_reference=manifest.transcript_reference,
            acknowledgement_entry_sha256=manifest.acknowledgement_entry_sha256,
            acknowledgement_message_id=evidence.acknowledgement_message_id,
            acknowledged_visible_sha256=evidence.acknowledged_visible_sha256,
            acknowledged_cursor=evidence.acknowledged_cursor,
            projection_digest=evidence.projection_digest,
        ),
    )


async def fork_request_for(
    parent: ParentOfAFork,
    *,
    retrieval: CheckpointRetrieved,
    fork_id: str,
    plans: Sequence[ForkChildPlan],
) -> ForkRequest:
    """Compose the typed request for one fork over one retrieved checkpoint.

    The execution scope is read again here rather than kept from the retrieval, because a turnover
    between the two is an ordinary event: the same logical fork is resubmitted against the
    execution that now exists, over the same checkpoint and the same plans, and the digest that
    keys it excludes the run id for exactly that reason.

    What is compared is the stream side, which a continuation preserves byte for byte. A generation
    that has moved past the acknowledgement the checkpoint was taken at is refused here, because a
    request composed from a fresh cursor and an old snapshot would name a freeze nobody took.
    """
    evidence = await _evidence(parent)
    _refuse_evidence_of_another_moment(evidence, retrieval)
    return ForkRequest(
        parent_workflow_id=evidence.parent_workflow_id,
        parent_run_id=evidence.parent_run_id,
        parent_execution_ordinal=evidence.execution_ordinal,
        parent_configuration_hash=evidence.configuration_hash,
        source_attempt_id=evidence.source_attempt_id,
        attestation_id=evidence.attestation_id,
        acknowledgement_message_id=evidence.acknowledgement_message_id,
        acknowledged_visible_sha256=evidence.acknowledged_visible_sha256,
        acknowledged_cursor=evidence.acknowledged_cursor,
        projection_digest=evidence.projection_digest,
        checkpoint_manifest_reference=retrieval.checkpoint_manifest_reference,
        fork_id=fork_id,
        child_plans=list(plans),
    )


async def _evidence(parent: ParentOfAFork) -> CheckpointEvidence:
    """The stream's half of a freeze, or the refusal that says the generation has none."""
    answer = await parent.checkpoint_evidence()
    if not answer.found or answer.evidence is None:
        raise ForkRefused(FORK_NOT_QUIET, answer.reason)
    check_checkpoint_evidence(answer.evidence)
    return answer.evidence


def _refuse_a_checkpoint_of_another_moment(
    manifest: CheckpointManifest, evidence: CheckpointEvidence
) -> None:
    """Refuse a harness checkpoint whose stream side is not the one the generation recorded."""
    for name, held, recorded in (
        ("acknowledgement", manifest.acknowledgement_message_id,
         evidence.acknowledgement_message_id),
        ("visible digest", manifest.acknowledged_visible_sha256,
         evidence.acknowledged_visible_sha256),
        ("cursor", manifest.acknowledged_cursor, evidence.acknowledged_cursor),
        ("projection digest", manifest.projection_digest, evidence.projection_digest),
    ):
        if held != recorded:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                f"this checkpoint claims the {name} {held!r} and the generation recorded "
                f"{recorded!r}",
            )


def _refuse_evidence_of_another_moment(
    evidence: CheckpointEvidence, retrieval: CheckpointRetrieved
) -> None:
    """Refuse a generation that has moved past the acknowledgement its checkpoint was taken at."""
    for name, standing, retrieved in (
        ("attempt", evidence.source_attempt_id, retrieval.source_attempt_id),
        ("attestation", evidence.attestation_id, retrieval.attestation_id),
        ("acknowledgement", evidence.acknowledgement_message_id,
         retrieval.acknowledgement_message_id),
        ("visible digest", evidence.acknowledged_visible_sha256,
         retrieval.acknowledged_visible_sha256),
        ("cursor", evidence.acknowledged_cursor, retrieval.acknowledged_cursor),
        ("projection digest", evidence.projection_digest, retrieval.projection_digest),
    ):
        if standing != retrieved:
            raise ForkRefused(
                FORK_WITNESS_MISMATCH,
                f"this generation now stands at the {name} {standing!r} and the checkpoint it "
                f"would be forked over was taken at {retrieved!r}",
            )


async def child_start(client: Client, *, child: ForkChildReceipt) -> StreamStart:
    """The start one child was created with, read off its own first execution and authenticated.

    It is the original execution that is read rather than whichever run is current, because a child
    that has continued as new carries a different projection under the same identifier and would
    answer for a start it was never created with.

    What makes reading it safe is the comparison against the receipt: the complete start digest
    covers the carrier, the store and the lineage, so a start that is not the one the parent
    committed to for this child fails here rather than being attached to.
    """
    handle = client.get_workflow_handle(child.child_workflow_id, run_id=child.child_run_id)
    history = await handle.fetch_history()
    started = history.events[0].workflow_execution_started_event_attributes
    # Through this client's own converter, because that is the converter the child was created
    # through: the Activity that started it encoded the start with the configured one, so a
    # deployment carrying a codec wrote bytes the default converter cannot read at all, and a
    # decoding that ignored the configuration would refuse a child the service created correctly.
    [start] = await client.data_converter.decode_wrapper(started.input, [StreamStart])
    if complete_start_digest(start) != child.complete_start_digest:
        raise ForkRefused(
            FORK_ORIGIN_DISAGREEMENT,
            f"the execution {child.child_workflow_id} was created with a start the fork did not "
            "record for it",
        )
    return start


def copied_closure(start: StreamStart, child: ForkChildReceipt) -> Tuple[List[str], List[str]]:
    """The objects one child's store is given: what it requires, and what it keeps for the record.

    The required half is the inventory the carrier holds plus the body this child was selected for.
    It is stated as a union rather than read off what the start declares, because the constructor
    seeds a committed list from the descriptors and the carrier then overwrites it, so a set built
    from the declaration alone would miss everything the prefix committed.

    The retained half is every other cell of every source the prefix sealed, the oracle among them.
    No operation the fork performs needs any of it, and it is copied because a lineage that kept
    the record of a comparison without the bodies it was made over would be a record of nothing.
    """
    if start.carry is None:
        raise ForkRefused(
            FORK_CONFIGURATION_VIOLATION,
            f"the child {child.child_workflow_id} carries no projection, and its closure is what "
            "that projection cites",
        )
    carried = unpack_carrier(start.carry, _CONVERTER)
    required = list(
        dict.fromkeys([*carried.committed_blobs, child.selected_body_reference])
    )
    retained: List[str] = []
    for attempt in carried.attempts:
        if attempt.source_artifact is None:
            continue
        descriptor = read_source_artifact(attempt.source_artifact)
        for cell in descriptor.cells.values():
            if cell.sha256 not in required and cell.sha256 not in retained:
                retained.append(cell.sha256)
    return required, retained


async def attach_child(
    client: Client,
    episode: ServedEpisode,
    operations: ForkOperations,
    *,
    fork_id: str,
    child: ForkChildReceipt,
    source: FilesystemBlobStore,
    consumer_id: Optional[str] = None,
    database_root: str = OWN_DATABASE_ROOT,
    environment: Optional[EnvironmentTerminal] = None,
    open_episode: Optional[EpisodeOpener] = None,
    on_refusal: Optional[RefusalSink] = None,
) -> Tuple[StreamGateway, ChildAttached]:
    """Give one child its objects and its transport, and keep what the claim installed.

    The order is the one a fork depends on: the child exists already and owns nothing, so its
    objects go in while its container is still paused, its directory is adopted, and its own
    gateway takes it with a first claim. A child that has not authorized its lineage yet rejects
    that claim and says so, which is a state rather than a refusal, so this waits it out.

    Every step of it is idempotent, which is what a lost reply is recovered by: installing an
    object that is installed writes nothing, a manifest that says the same thing is adopted, and a
    claim after a claim is an ownership transition rather than a second generation. So the work is
    done again on a repeat and the record is the first one: the epoch it names is the one the first
    claim installed, and every claim after it installs another. Which epoch this child's
    preparation was frozen at is the child's own to say, and it says it in its readiness.

    The record is read before the attachment rather than after it, because it holds the consumer
    this child was attached as and a repeat has to arrive with that consumer rather than mint one.
    A caller that named none and left the record unread would ask for a fresh logical consumer,
    which is the one thing the generation will not adopt. Where nothing was written down the
    generation is asked instead, so a reply lost before the outcome reached the disk recovers too.
    """
    key = attachment_key(fork_id=fork_id, child_workflow_id=child.child_workflow_id)
    kept = operations.recorded(key, ChildAttached)
    start = await child_start(client, child=child)
    required, retained = copied_closure(start, child)
    absent = _installed(source, start, child, required, retained)
    gateway = await _attached(
        client,
        episode,
        start=start,
        child=child,
        consumer_id=consumer_id or (None if kept is None else kept.consumer_id or None),
        database_root=database_root,
        environment=environment,
        open_episode=open_episode,
        on_refusal=on_refusal,
    )
    if kept is not None:
        return gateway, kept
    state = await gateway.stream_state()
    return gateway, operations.record(
        key,
        ChildAttached(
            fork_id=fork_id,
            child_workflow_id=child.child_workflow_id,
            run_directory=str(run_directory_of(start.blob_root or "")),
            configuration_hash=state.configuration_hash,
            complete_start_digest=child.complete_start_digest,
            installed=required,
            retained_absent=absent,
            ownership_epoch=state.ownership_epoch,
            consumer_id=state.consumer_id or "",
        ),
    )


def _installed(
    source: FilesystemBlobStore,
    start: StreamStart,
    child: ForkChildReceipt,
    required: Sequence[str],
    retained: Sequence[str],
) -> List[str]:
    """Copy one child's objects into its own store, and return what was kept and is not there.

    Each child gets a store of its own rather than a share of one, because a shared store is a path
    two children can both reach and what is in it is the evidence one of them is being measured on.

    A required object the parent cannot produce is a repairable absence and an object that does not
    read back after it was installed is an interrupted copy, and both of them are the controller's
    to repair. A retained object that never arrived blocks nothing and is reported.

    The absence is the same absence wherever the read finds it. Verifying the source and copying it
    are two reads of every required object, the objects are files and nothing holds them open in
    between, so one produced a moment ago can be gone by the time it is installed. That is reported
    as the repairable absence it is, naming the object, rather than reaching a controller as
    whatever the store raised: a refusal with no reason on it is one nothing can decide to retry.
    The objects installed before it stay installed, because copying is idempotent under retry and
    an installed object is the object its name promises.
    """
    if not start.blob_root:
        raise ForkRefused(
            FORK_CONFIGURATION_VIOLATION,
            f"the child {child.child_workflow_id} names no store, and a generation that verifies "
            "its objects against nothing verifies nothing",
        )
    into = FilesystemBlobStore(Path(start.blob_root))
    unavailable = source.unverified(required)
    if unavailable:
        raise ForkRefused(
            FORK_REPAIRABLE_ABSENCE,
            f"the store this fork copies from cannot produce {sorted(unavailable)}",
        )
    for reference in required:
        try:
            copied = source.read(reference)
        except WireFormatError as gone:
            raise ForkRefused(
                FORK_REPAIRABLE_ABSENCE,
                f"the store this fork copies from produced {reference} and could not produce it "
                "again while it was being copied",
            ) from gone
        into.put(copied)
    interrupted = into.unverified(required)
    if interrupted:
        raise ForkRefused(
            FORK_INTERRUPTED_COPY,
            f"the store of {child.child_workflow_id} does not hold {sorted(interrupted)} after "
            "they were installed in it",
        )
    absent: List[str] = []
    for reference in retained:
        try:
            into.put(source.read(reference))
        except WireFormatError:
            absent.append(reference)
    return absent


async def _attached(
    client: Client,
    episode: ServedEpisode,
    *,
    start: StreamStart,
    child: ForkChildReceipt,
    consumer_id: Optional[str],
    database_root: str,
    environment: Optional[EnvironmentTerminal],
    open_episode: Optional[EpisodeOpener],
    on_refusal: Optional[RefusalSink],
) -> StreamGateway:
    """Attach to one child, waiting out the gate it owns nothing behind.

    The directory is derived from the store the authenticated start names rather than passed
    beside it: the start is the value the generation was created from and its store sits inside
    the directory the generation lives in, so a directory handed in would be a second opinion
    about where that is.

    A child that says its parent's answer window has closed is the one thing here that is not
    waited out. The gate is a wait because a reading that has not happened yet happens; a window
    that has closed does not reopen, so waiting is spent on a question with no answer left. The
    child keeps its identity and everything it holds either way, and what comes back says which
    of the two it is.
    """
    for _try in range(_GATE_TRIES):
        try:
            return await attach_gateway(
                client,
                episode,
                workflow_id=child.child_workflow_id,
                start=start,
                run_directory=run_directory_of(start.blob_root or ""),
                database_root=database_root,
                consumer_id=consumer_id,
                open_episode=open_episode,
                environment=environment,
                on_refusal=on_refusal,
            )
        except Exception as error:  # noqa: BLE001 - the gate is waited out and nothing else is
            if fork_refusal(error) == FORK_EXPIRED_AUTHORITY:
                raise ForkRefused(
                    FORK_EXPIRED_AUTHORITY,
                    f"the child {child.child_workflow_id} asked for the record it verifies "
                    "against and the window its parent's answers stood in had closed",
                ) from error
            if not origin_unverified(error):
                raise
            await asyncio.sleep(_GATE_WAIT_SECONDS)
    raise ForkRefused(
        FORK_UNREADABLE_PARENT,
        f"the child {child.child_workflow_id} never authorized the lineage it carries, so the "
        "record it verifies against could not be read",
    )


async def prepare_child(
    gateway: ChildOfAFork, operations: ForkOperations, *, attachment: ChildAttached
) -> ChildPrepared:
    """Have one child build the body it owes, and keep the readiness or the reason it has none.

    The operation is the child's own and so is its name: the epoch inside the identity is the one
    the child's first attempt was made at, and an attachment that recovered a lost reply installed
    a later one. So the readiness is what names the operation, the outcome is kept under the name
    it carries, and what a controller that came back reads is the outcome kept for this child of
    this fork whatever epoch the child froze. A readiness published and then lost on the way back
    is read again from the child, which installs no second body and republishes what it holds.

    A permanent reason is kept and a repairable one is not. An object the store lost is put back by
    a controller and the same logical preparation runs again under the owner that follows, and a
    record of that absence would answer the repair with the failure it repaired.

    A refusal recorded inside an episode is kept under that episode's own identity, like the
    readiness it stands in place of. The two are the outcomes of one operation and the contract is
    that either is retrievable under the identity the child froze, so a decision kept under the
    fallback name alone would lose the very thing the surface is keyed by. The child says which
    epoch it froze and this side derives the name from it, so what a file is called is never a
    string a transport handed over, and a refusal raised before any episode existed says so with a
    zero and keeps the pair's own name.
    """
    kept = _kept_preparation(operations, attachment)
    if kept is not None:
        return kept
    try:
        ready = await gateway.prepare_child(fork_id=attachment.fork_id)
    except Exception as error:  # noqa: BLE001 - a decision is kept and everything else is raised
        reason = fork_refusal(error)
        if reason is None or fork_can_be_retried(error):
            raise
        frozen = fork_preparation_epoch(error)
        operation = (
            ""
            if frozen < 1
            else preparation_key(
                fork_id=attachment.fork_id,
                child_workflow_id=attachment.child_workflow_id,
                creation_epoch=frozen,
            )
        )
        return operations.record(
            operation
            or refused_preparation_key(
                fork_id=attachment.fork_id, child_workflow_id=attachment.child_workflow_id
            ),
            ChildPrepared(
                fork_id=attachment.fork_id,
                child_workflow_id=attachment.child_workflow_id,
                preparation_operation=operation,
                reason=reason,
            ),
        )
    _refuse_a_readiness_of_another_operation(ready, attachment)
    return operations.record(
        ready.preparation_operation,
        ChildPrepared(
            fork_id=attachment.fork_id,
            child_workflow_id=attachment.child_workflow_id,
            preparation_operation=ready.preparation_operation,
            ready=ready,
        ),
    )


def _kept_preparation(
    operations: ForkOperations, attachment: ChildAttached
) -> Optional[ChildPrepared]:
    """What one child's preparation already came to, under whichever name it was kept under.

    A preparation is answered once, and which name that answer is under depends on what the child
    published: the identity it froze where it published a readiness, and the pair where it
    published nothing. So the outcome is found by the fork and the child it is about rather than
    by an epoch this side would have to guess at.
    """
    for kept in operations.all_of(ChildPrepared):
        if (
            kept.fork_id == attachment.fork_id
            and kept.child_workflow_id == attachment.child_workflow_id
        ):
            return kept
    return None


def _refuse_a_readiness_of_another_operation(
    ready: ChildReady, attachment: ChildAttached
) -> None:
    """Refuse a readiness whose operation is not one this child of this fork could have frozen.

    The identity is minted from the fork, the child and the epoch the child's first preparation was
    created under. The first two are values this side holds, and what is checked about the third is
    that the child has held it: an operation frozen at an epoch this child never reached is not an
    operation of it, and one frozen before the epoch it holds now is the ordinary case of a
    preparation that followed an attachment recovering a lost reply. The name the outcome is then
    kept under is one this side recomputed, so what a file is called is never a string a transport
    handed over.
    """
    if (
        ready.fork_id != attachment.fork_id
        or ready.child_workflow_id != attachment.child_workflow_id
    ):
        raise ForkRefused(
            FORK_ORIGIN_DISAGREEMENT,
            f"this readiness is {ready.child_workflow_id} of the fork {ready.fork_id}, and the "
            f"attachment it would answer is {attachment.child_workflow_id} of "
            f"{attachment.fork_id}",
        )
    for epoch in range(1, ready.ownership_epoch + 1):
        if ready.preparation_operation == preparation_key(
            fork_id=ready.fork_id,
            child_workflow_id=ready.child_workflow_id,
            creation_epoch=epoch,
        ):
            return
    raise ForkRefused(
        FORK_ORIGIN_DISAGREEMENT,
        f"the child {attachment.child_workflow_id} published a readiness under the operation "
        f"{ready.preparation_operation}, which no preparation of it was created under",
    )


async def compare_child(
    adapter: ForkAdapter,
    operations: ForkOperations,
    *,
    retrieval: CheckpointRetrieved,
    preparation: ChildPrepared,
) -> ChildCompared:
    """Restore one container copy for a ready child and compare it against the parent.

    The restore happens once. It is the operation a repeat would cost the most, because a second
    copy over a child that has already been resumed would put the parent's state back over work
    that happened, so the record is read before the adapter is asked and the container it names is
    the one this child keeps.

    Two things are compared. The adapter's own comparison is under its versioned rebinding
    allowlist, which is a fact about containers and its to state. This side checks the one thing
    the two witnesses exist for: that what came back reproduces the transcript the checkpoint named
    and holds the acknowledgement entry inside it, since a snapshot taken before the acknowledgement
    hashes correctly and restores a container that never saw it.
    """
    ready = preparation.ready
    if ready is None:
        raise ValueError(
            f"the child {preparation.child_workflow_id} published no readiness, and a container is "
            f"restored for a child that is ready to serve one: {preparation.reason}"
        )
    if ready.checkpoint_manifest_reference != retrieval.checkpoint_manifest_reference:
        raise ForkRefused(
            FORK_WITNESS_MISMATCH,
            f"the child {preparation.child_workflow_id} was cut at "
            f"{ready.checkpoint_manifest_reference} and this checkpoint is "
            f"{retrieval.checkpoint_manifest_reference}",
        )
    key = equality_key(
        fork_id=preparation.fork_id, child_workflow_id=preparation.child_workflow_id
    )
    kept = operations.recorded(key, ChildCompared)
    if kept is not None:
        return kept
    restored = await adapter.restore(
        fork_id=preparation.fork_id,
        child_workflow_id=preparation.child_workflow_id,
        checkpoint_manifest_reference=retrieval.checkpoint_manifest_reference,
    )
    differences = _restoration_differences(restored, retrieval)
    equality = await adapter.compare(
        fork_id=preparation.fork_id,
        child_workflow_id=preparation.child_workflow_id,
        container_id=restored.container_id,
    )
    differences.extend(equality.differences)
    return operations.record(
        key,
        ChildCompared(
            fork_id=preparation.fork_id,
            child_workflow_id=preparation.child_workflow_id,
            container_id=restored.container_id,
            allowlist_version=equality.allowlist_version,
            rebound=list(equality.rebound),
            differences=differences,
            equal=not differences,
        ),
    )


def _restoration_differences(
    restored: RestoredContainer, retrieval: CheckpointRetrieved
) -> List[str]:
    """What a restored container came back holding that the checkpoint did not name."""
    differences: List[str] = []
    if restored.transcript_reference != retrieval.transcript_reference:
        differences.append(
            f"the restored transcript is {restored.transcript_reference} and the checkpoint names "
            f"{retrieval.transcript_reference}"
        )
    if restored.acknowledgement_entry_sha256 != retrieval.acknowledgement_entry_sha256:
        differences.append(
            "the restored transcript does not hold the acknowledgement entry the checkpoint names"
        )
    return differences


async def bind_child(
    adapter: ForkAdapter,
    operations: ForkOperations,
    *,
    attachment: ChildAttached,
    comparison: ChildCompared,
) -> ChildBound:
    """Bind one compared container to the one child generation it serves.

    A container is bound to one child and a child to one container, and the lineage's own record is
    what says so: a binding that named a container another child is already bound to would let two
    generations write one mutable state, which is the whole thing the copies exist to prevent.

    The evidence has to be this child's own before any of that means anything. The identity a
    binding is keyed by comes from the attachment and the container comes from the comparison, so
    two comparisons swapped between siblings would bind one child to the container restored for
    the other, and the already-bound scan cannot see it because neither container is bound yet.
    Both identity fields are therefore compared first, and the comparison this side kept for this
    exact pair is what the container is read out of, so a value handed in is checked against the
    record rather than trusted beside it.
    """
    if (
        comparison.fork_id != attachment.fork_id
        or comparison.child_workflow_id != attachment.child_workflow_id
    ):
        raise ForkRefused(
            FORK_ORIGIN_DISAGREEMENT,
            f"this comparison is {comparison.child_workflow_id} of the fork {comparison.fork_id}, "
            f"and the attachment it would bind is {attachment.child_workflow_id} of "
            f"{attachment.fork_id}",
        )
    if not comparison.equal:
        raise ValueError(
            f"the container of {comparison.child_workflow_id} differs from its parent in "
            f"{comparison.differences}, and a container is bound to a child after it compares equal"
        )
    recorded = operations.recorded(
        equality_key(
            fork_id=attachment.fork_id, child_workflow_id=attachment.child_workflow_id
        ),
        ChildCompared,
    )
    if recorded is None or recorded != comparison:
        raise ForkRefused(
            FORK_ORIGIN_DISAGREEMENT,
            f"the comparison this lineage kept for {attachment.child_workflow_id} is not the one "
            "this binding was handed, and a container is bound on the evidence of record",
        )
    key = binding_key(
        fork_id=attachment.fork_id, child_workflow_id=attachment.child_workflow_id
    )
    kept = operations.recorded(key, ChildBound)
    if kept is not None:
        return kept
    for standing in operations.all_of(ChildBound):
        if (
            standing.container_id == comparison.container_id
            and standing.child_workflow_id != attachment.child_workflow_id
        ):
            raise ValueError(
                f"the container {comparison.container_id} is bound to "
                f"{standing.child_workflow_id} already, and a container serves one child"
            )
    await adapter.bind(
        fork_id=attachment.fork_id,
        child_workflow_id=attachment.child_workflow_id,
        container_id=comparison.container_id,
        consumer_id=attachment.consumer_id,
    )
    return operations.record(
        key,
        ChildBound(
            fork_id=attachment.fork_id,
            child_workflow_id=attachment.child_workflow_id,
            container_id=comparison.container_id,
            consumer_id=attachment.consumer_id,
        ),
    )


async def release_children(
    adapter: ForkAdapter, operations: ForkOperations, *, receipt: ForkReceipt
) -> List[ChildReleased]:
    """Resume one container per child, and only where every child of the fork is ready and equal.

    Release is the asymmetric step, so the policy is stated once and read out of the record rather
    than out of whatever this process remembers. A child that published no readiness, or whose
    restored container did not compare equal, holds the whole fork: nothing is resumed, the failure
    stands against the exact child it names, and the child that was fine is held rather than
    promoted, because a link with one arm running is not the comparison anybody registered.

    Which children that is over is the fork's own receipt and never a list a caller chose. A
    subset would make the gate a statement about whichever children were named: passing one child
    would leave the other's readiness, comparison and binding out of the decision entirely, and
    the one arm would run against a sibling nobody had established was ready. So the roster is the
    parent's typed evidence, and every child in it is what the hold is derived from.

    A resumption is kept and read back. A controller that crashed between the first release and the
    second recovers the child that is already running from its own record and resumes only the one
    still paused, so nothing is restored over work that happened and no child is created twice.

    The receipt's own version is checked before any of it is read, because this is where a receipt
    becomes a decision about two containers: a shape this build does not admit is refused rather
    than consumed for the one field the refusal would have been read out of.
    """
    check_fork_receipt(receipt)
    fork_id = receipt.fork_id
    children = [child.child_workflow_id for child in receipt.child_receipts]
    if len(set(children)) != receipt.children or len(children) != receipt.children:
        raise ForkRefused(
            FORK_ORIGIN_DISAGREEMENT,
            f"the fork {fork_id} created {receipt.children} children and its receipt names "
            f"{sorted(set(children))}",
        )
    bound: Dict[str, ChildBound] = {}
    withheld: Dict[str, str] = {}
    for child in children:
        comparison = operations.recorded(
            equality_key(fork_id=fork_id, child_workflow_id=child), ChildCompared
        )
        binding = operations.recorded(
            binding_key(fork_id=fork_id, child_workflow_id=child), ChildBound
        )
        if comparison is None:
            withheld[child] = "no container of its own was restored and compared"
        elif not comparison.equal:
            withheld[child] = (
                f"its restored container differs from the parent in {comparison.differences}"
            )
        elif binding is None:
            withheld[child] = "its container is bound to no generation"
        else:
            bound[child] = binding
    if withheld:
        return [
            ChildReleased(
                fork_id=fork_id,
                child_workflow_id=child,
                container_id=bound[child].container_id if child in bound else "",
                resumed=False,
                held=withheld.get(child)
                or f"the child {sorted(withheld)[0]} of this fork "
                f"{withheld[sorted(withheld)[0]]}",
            )
            for child in children
        ]
    released: List[ChildReleased] = []
    for child in children:
        key = release_key(fork_id=fork_id, child_workflow_id=child)
        kept = operations.recorded(key, ChildReleased)
        if kept is not None and kept.resumed:
            released.append(kept)
            continue
        resumed = await adapter.release(
            fork_id=fork_id, child_workflow_id=child, container_id=bound[child].container_id
        )
        released.append(
            operations.record(
                key,
                ChildReleased(
                    fork_id=fork_id,
                    child_workflow_id=child,
                    container_id=resumed.container_id,
                    resumed=True,
                    first_message_id=resumed.first_message_id,
                ),
            )
        )
    return released
