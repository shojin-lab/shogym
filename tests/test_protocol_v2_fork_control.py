"""The controller's operations over one fork, driven against a harness that keeps real copies.

A fork is finished by a controller working through six operations in order, and what this file is
about is those operations rather than the generations they are about: the checkpoint both witnesses
have to agree on, the objects and the claim a child is attached with, the readiness it publishes,
the container copy that is compared against the parent, the binding that says which child a
container serves, and the release that lets exactly the right ones go.

Every one of them is answered under a durable key, so the case that matters most here is the crash:
a controller that came back reads what it already did rather than doing it again. That is driven by
throwing the controller's memory away and building a second one over the same directory, which is
what a second process would find.

The harness below is a stand-in and it is a real one: the parent's container is a directory of
files, memory, buffers and weights, a restore is a copy of it, and a comparison is a walk of both.
The adapter a measured run uses is the experiment controller's, and what this file proves about it
is the shape it is called through and the order it is called in.
"""

from __future__ import annotations

import json
import shutil
import zlib
from dataclasses import dataclass, field, fields, replace
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, get_args

import pytest

pytest.importorskip("temporalio")

from temporalio.api.common.v1 import Payload, Payloads  # noqa: E402
from temporalio.converter import DataConverter, PayloadCodec  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.serve.protocol_v2.blobs import BlobRef, FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.errors import WireFormatError  # noqa: E402
from shogym.serve.protocol_v2.gateway import _Idle, _Recovery  # noqa: E402
from shogym.serve.protocol_v2.fork import (  # noqa: E402
    FORK_OPERATION_SCHEMA_VERSION,
    ChildAttached,
    ChildBound,
    ChildCompared,
    ChildPrepared,
    ChildReleased,
    CheckpointRetrieved,
    ContainerEquality,
    ForkOperations,
    RestoredContainer,
    ResumedContainer,
    UnsupportedCapability,
    _installed,
    attachment_key,
    binding_key,
    bind_child,
    checkpoint_key,
    child_start,
    compare_child,
    equality_key,
    fork_request_for,
    prepare_child,
    preparation_key,
    refused_preparation_key,
    release_children,
    release_key,
    retrieve_checkpoint,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    FORK_INTERRUPTED_COPY,
    FORK_ORIGIN_DISAGREEMENT,
    FORK_REPAIRABLE_ABSENCE,
    FORK_WRONG_SOURCE,
    CheckpointComponent,
    CheckpointEvidence,
    CheckpointEvidenceAnswer,
    CheckpointManifest,
    ChildReady,
    ForkChildPlan,
    ForkChildReceipt,
    ForkOrigin,
    ForkReceipt,
    ForkRefused,
    SettledHarness,
    StreamStart,
    TerminalTool,
    checkpoint_manifest_reference,
    complete_start_digest,
    fork_preparation_operation_identity,
    fork_request_digest,
)

PARENT = "stream/the-parent/1"
PARENT_RUN = "the-run-the-parent-is-in"
FORK = "the-fork"
ATTEMPT = "attempt-1"
FIRST = "stream/the-parent/1.fork.1.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
SECOND = "stream/the-parent/1.fork.2.bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
CHILDREN = (FIRST, SECOND)


def digest_of(text: str) -> str:
    """The digest of one string, which is how every identifier in this file is minted."""
    return sha256(text.encode("utf-8")).hexdigest()


#: What one container's transcript holds, and the acknowledgement entry inside it. A restore that
#: reproduces both is the one the checkpoint named, and a restore of an older snapshot is not.
ACK_LINE = "the acknowledgement entry inside it"
TRANSCRIPT_TEXT = f"the transcript the harness persisted\n{ACK_LINE}"
TRANSCRIPT = digest_of(TRANSCRIPT_TEXT)
ACK_ENTRY = digest_of(ACK_LINE)
ACK_MESSAGE = "message-of-the-acknowledgement"
CURSOR = ACK_MESSAGE
VISIBLE = digest_of("the acknowledgement as it was presented")
PROJECTION = digest_of("the projection at that commitment")


def a_manifest(**changes: Any) -> CheckpointManifest:
    """The checkpoint one harness committed for the acknowledgement it persisted."""
    declared = CheckpointManifest(
        transcript_reference=TRANSCRIPT,
        acknowledgement_locator="entry 41",
        acknowledgement_entry_sha256=ACK_ENTRY,
        components=[
            CheckpointComponent(
                component_id="the container",
                sha256=digest_of("the snapshot of the container"),
                size=4096,
                media_type="application/octet-stream",
                restores_transcript=TRANSCRIPT,
            )
        ],
        adapter_version="the adapter of this run",
        container_image_digest=digest_of("the image the container runs"),
        harness_configuration=digest_of("the configuration the harness holds"),
        settled=SettledHarness(
            transport=True, recovery=True, provider=True, model=True, compaction=True
        ),
        frozen_plan_digest=digest_of("the plan both children are parked under"),
        acknowledgement_message_id=ACK_MESSAGE,
        acknowledged_visible_sha256=VISIBLE,
        acknowledged_cursor=CURSOR,
        projection_digest=PROJECTION,
    )
    return replace(declared, **changes) if changes else declared


def an_evidence(**changes: Any) -> CheckpointEvidence:
    """The stream's half of the same freeze, as the generation recorded it."""
    declared = CheckpointEvidence(
        parent_workflow_id=PARENT,
        parent_run_id=PARENT_RUN,
        execution_ordinal=0,
        configuration_hash="c" * 64,
        capacity_in_use=0,
        source_attempt_id=ATTEMPT,
        attestation_id="the-attestation",
        acknowledgement_message_id=ACK_MESSAGE,
        acknowledged_visible_sha256=VISIBLE,
        acknowledged_cursor=CURSOR,
        projection_digest=PROJECTION,
    )
    return replace(declared, **changes) if changes else declared


class AParent:
    """A generation answering with the evidence it recorded, and counting who asked."""

    def __init__(self, answer: Optional[CheckpointEvidenceAnswer] = None) -> None:
        self.answer = answer or CheckpointEvidenceAnswer(found=True, evidence=an_evidence())
        self.reads = 0

    async def checkpoint_evidence(self) -> CheckpointEvidenceAnswer:
        self.reads += 1
        return self.answer


class AChild:
    """A child answering one preparation, with whatever its own history came to."""

    def __init__(self, ready: Optional[ChildReady] = None, refusal: Optional[Exception] = None):
        self.ready = ready
        self.refusal = refusal
        self.preparations = 0

    async def prepare_child(self, *, fork_id: str) -> ChildReady:
        self.preparations += 1
        if self.refusal is not None:
            raise self.refusal
        assert self.ready is not None
        return self.ready


def a_readiness(child: str, *, epoch: int = 1, **changes: Any) -> ChildReady:
    """What one child published when it had built the body it owed."""
    declared = ChildReady(
        fork_id=FORK,
        child_workflow_id=child,
        child_run_id=f"the-run-of-{child}",
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_manifest()),
        origin_digest=digest_of(f"the lineage of {child}"),
        verified_set_digest=digest_of(f"the objects {child} read back"),
        source_attempt_id=ATTEMPT,
        selected_cell="graded",
        selected_body_reference=digest_of(f"the body {child} delivers"),
        selected_policy_digest=digest_of("the policy that body is served under"),
        receipt_contract_id="the-contract",
        ownership_epoch=epoch,
        consumer_id=f"the-transport-of-{child}",
        preparation_operation=fork_preparation_operation_identity(FORK, child, epoch),
    )
    return replace(declared, **changes) if changes else declared


def an_attachment(child: str, *, epoch: int = 1, **changes: Any) -> ChildAttached:
    """What one child's attachment installed, the epoch being the one its first claim did."""
    declared = ChildAttached(
        fork_id=FORK,
        child_workflow_id=child,
        run_directory=f"/runs/{child}",
        configuration_hash=digest_of(f"what {child} is"),
        complete_start_digest=digest_of(f"the start {child} was created with"),
        installed=[digest_of("the inventory the prefix committed")],
        retained_absent=[],
        ownership_epoch=epoch,
        consumer_id=f"the-transport-of-{child}",
    )
    return replace(declared, **changes) if changes else declared


#: What a container is made of here: the transcript the acknowledgement was persisted into, and the
#: mutable things each child has to own a copy of rather than a share of.
CONTAINER_FILES = ("transcript.jsonl", "memory.md", "buffer.bin", "weights.bin", "binding.json")

#: What this harness's rebinding allowlist admits. The file naming which generation a container
#: serves is a container's own and is expected to differ; everything else is compared.
ALLOWLIST_VERSION = "containers.rebinding.1"
REBOUND = ("binding.json",)

def digest_of_file(path: Path) -> str:
    """The digest of the bytes one file holds."""
    return sha256(path.read_bytes()).hexdigest()


class AHarness:
    """A harness whose containers are directories: one for the parent, one copy per child.

    It is a stand-in for the adapter a measured run uses and it does the same things: it settles,
    it persists the acknowledgement into its own transcript, it commits a checkpoint over that, it
    restores one independently owned copy per child, it compares each copy against the parent, it
    binds each to the child generation it serves, and it resumes them so the agent inside pulls.
    """

    def __init__(
        self,
        root: Path,
        *,
        manifest: Optional[CheckpointManifest] = None,
        unsupported: str = "",
        restored_as: Optional[Callable[[Path], None]] = None,
        pull: Optional[Callable[[str], str]] = None,
        crash_after: Optional[int] = None,
    ) -> None:
        self.root = root
        self.parent = root / "parent"
        self.parent.mkdir(parents=True)
        (self.parent / "transcript.jsonl").write_text(TRANSCRIPT_TEXT, encoding="utf-8")
        (self.parent / "memory.md").write_text("what the agent learned", encoding="utf-8")
        (self.parent / "buffer.bin").write_bytes(b"the recovery buffer")
        (self.parent / "weights.bin").write_bytes(b"the weights this harness owns")
        (self.parent / "binding.json").write_text(f'{{"serves": "{PARENT}"}}', encoding="utf-8")
        self.manifest = manifest or a_manifest()
        self.unsupported = unsupported
        self.restored_as = restored_as
        self.pull = pull or (lambda child: f"the-first-pull-of-{child}")
        # How many containers this harness resumes before the process running it goes away. It is
        # a crash rather than a refusal, so nothing is recorded about it and the recovery is a
        # controller reading what it already did.
        self.crash_after = crash_after
        self.restores: List[str] = []
        self.compared: List[str] = []
        self.bound: Dict[str, str] = {}
        self.resumed: List[str] = []

    def container(self, child: str) -> Path:
        """Where one child's own copy lives."""
        return self.root / f"container-of-{child.rsplit('.', 1)[-1]}"

    async def checkpoint(
        self, *, parent_workflow_id: str, attestation_id: str
    ) -> CheckpointManifest:
        self._supported("commit a checkpoint")
        return self.manifest

    async def restore(
        self, *, fork_id: str, child_workflow_id: str, checkpoint_manifest_reference: str
    ) -> RestoredContainer:
        self._supported("restore a container")
        path = self.container(child_workflow_id)
        # Keyed by the child rather than by the call, so a controller that crashed between this
        # and its own record is answered with the copy that exists rather than given a second one.
        if not path.exists():
            shutil.copytree(self.parent, path)
            if self.restored_as is not None:
                self.restored_as(path)
            self.restores.append(child_workflow_id)
        transcript = path / "transcript.jsonl"
        return RestoredContainer(
            container_id=path.name,
            transcript_reference=digest_of_file(transcript),
            acknowledgement_entry_sha256=digest_of(
                transcript.read_text(encoding="utf-8").splitlines()[-1]
            ),
        )

    async def compare(
        self, *, fork_id: str, child_workflow_id: str, container_id: str
    ) -> ContainerEquality:
        self._supported("compare a container")
        self.compared.append(child_workflow_id)
        path = self.root / container_id
        return ContainerEquality(
            allowlist_version=ALLOWLIST_VERSION,
            rebound=list(REBOUND),
            differences=[
                name
                for name in CONTAINER_FILES
                if name not in REBOUND
                and digest_of_file(self.parent / name) != digest_of_file(path / name)
            ],
        )

    async def bind(
        self, *, fork_id: str, child_workflow_id: str, container_id: str, consumer_id: str
    ) -> None:
        self._supported("bind a container")
        (self.root / container_id / "binding.json").write_text(
            f'{{"serves": "{child_workflow_id}", "through": "{consumer_id}"}}', encoding="utf-8"
        )
        self.bound[child_workflow_id] = container_id

    async def release(
        self, *, fork_id: str, child_workflow_id: str, container_id: str
    ) -> ResumedContainer:
        self._supported("resume a container")
        if self.crash_after is not None and len(self.resumed) >= self.crash_after:
            raise RuntimeError(f"the process resuming {child_workflow_id} went away")
        # A container that is already running is answered with the state it is running in, which
        # is what makes the interval between the resumption and its record survivable: the effect
        # is outside the controller, so what recovers it is an outcome that can be read back.
        if child_workflow_id not in self.resumed:
            self.resumed.append(child_workflow_id)
        return ResumedContainer(
            container_id=container_id, first_message_id=self.pull(child_workflow_id)
        )

    def _supported(self, capability: str) -> None:
        """Report the one thing this harness was built without, where it is asked for it."""
        if self.unsupported == capability:
            raise UnsupportedCapability(capability, "this harness was built without it")


async def a_compared_child(
    harness: AHarness, operations: ForkOperations, child: str, retrieval: CheckpointRetrieved
) -> ChildCompared:
    """Take one child from its readiness to its comparison, which is what a release reads."""
    prepared = await prepare_child(
        AChild(a_readiness(child)), operations, attachment=an_attachment(child)
    )
    return await compare_child(harness, operations, retrieval=retrieval, preparation=prepared)


def test_an_outcome_kept_under_its_key_is_what_a_controller_that_came_back_reads(
    tmp_path: Path,
) -> None:
    """A crash between two operations is survived by reading, not by doing the work again.

    The key is derived from what the operation was about rather than from when it ran, so the
    process that comes back asks the same question and finds the answer. Nothing here is in memory:
    the second reader is a second object over the same directory, which is what a second process
    would find.
    """
    operations = ForkOperations.under(tmp_path)
    binding = ChildBound(
        fork_id=FORK,
        child_workflow_id=FIRST,
        container_id="container-of-1",
        consumer_id="the-transport-of-1",
    )
    operations.record(binding_key(fork_id=FORK, child_workflow_id=FIRST), binding)

    came_back = ForkOperations.under(tmp_path)
    assert came_back.recorded(binding_key(fork_id=FORK, child_workflow_id=FIRST), ChildBound) == (
        binding
    )
    assert came_back.recorded(binding_key(fork_id=FORK, child_workflow_id=SECOND), ChildBound) is (
        None
    )
    assert came_back.all_of(ChildBound) == [binding]
    assert came_back.all_of(ChildReleased) == []


def test_an_answer_that_says_the_same_is_adopted_and_one_that_says_anything_else_is_refused(
    tmp_path: Path,
) -> None:
    """One name, one answer. Repeating an idempotent operation is how a lost reply is recovered."""
    operations = ForkOperations.under(tmp_path)
    key = attachment_key(fork_id=FORK, child_workflow_id=FIRST)
    attached = an_attachment(FIRST)
    assert operations.record(key, attached) == attached
    assert operations.record(key, an_attachment(FIRST)) == attached

    with pytest.raises(WireFormatError):
        operations.record(key, an_attachment(FIRST, ownership_epoch=2))


def test_an_outcome_this_build_cannot_read_is_refused_rather_than_taken_for_another(
    tmp_path: Path,
) -> None:
    """A file holding another kind, or a shape at another version, is not this outcome."""
    operations = ForkOperations.under(tmp_path)
    key = binding_key(fork_id=FORK, child_workflow_id=FIRST)
    operations.record(
        key,
        ChildBound(
            fork_id=FORK,
            child_workflow_id=FIRST,
            container_id="container-of-1",
            consumer_id="the-transport-of-1",
        ),
    )
    with pytest.raises(WireFormatError):
        operations.recorded(key, ChildReleased)

    written = operations.root / f"{key}.json"
    written.write_text(
        written.read_text(encoding="utf-8").replace(
            f'"schema_version": {FORK_OPERATION_SCHEMA_VERSION}',
            f'"schema_version": {FORK_OPERATION_SCHEMA_VERSION + 1}',
        ),
        encoding="utf-8",
    )
    with pytest.raises(WireFormatError):
        operations.recorded(key, ChildBound)


async def test_a_readiness_at_a_version_this_build_cannot_read_is_refused_where_it_is_read_back(
    tmp_path: Path,
) -> None:
    """The shape inside a kept outcome is admitted too, and it is admitted before it is evidence.

    A recovered preparation is what a controller that came back reads instead of preparing again,
    so nothing later asks the child anything: the readiness inside it goes straight on to be the
    evidence a container is restored and released against. A version this build cannot read is a
    shape it cannot act on, and an outcome carrying one is refused where it is read rather than
    acted on because its own outer version was one this build writes.
    """
    operations = ForkOperations.under(tmp_path)
    attachment = an_attachment(FIRST)
    prepared = await prepare_child(
        AChild(a_readiness(FIRST)), operations, attachment=attachment
    )
    assert prepared.ready is not None

    written = operations.root / f"{prepared.preparation_operation}.json"
    held = json.loads(written.read_text(encoding="utf-8"))
    held["result"]["ready"]["schema_version"] = 999
    written.write_text(json.dumps(held, sort_keys=True) + "\n", encoding="utf-8")

    came_back = ForkOperations.under(tmp_path)
    with pytest.raises(WireFormatError) as raised:
        came_back.recorded(prepared.preparation_operation, ChildPrepared)
    assert "999" in str(raised.value)
    with pytest.raises(WireFormatError):
        came_back.all_of(ChildPrepared)
    with pytest.raises(WireFormatError):
        await prepare_child(AChild(a_readiness(FIRST)), came_back, attachment=attachment)

    # And nothing was restored on the strength of it: the refusal lands before a container is.
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, ForkOperations.under(tmp_path))
    with pytest.raises(WireFormatError):
        await compare_child(
            harness,
            came_back,
            retrieval=retrieval,
            preparation=came_back.recorded(prepared.preparation_operation, ChildPrepared),
        )
    assert harness.restores == []
    assert harness.compared == []


def test_every_operation_of_one_fork_is_answered_under_a_name_of_its_own() -> None:
    """Six operations, six names, and one of them is the operation's own frozen identity."""
    keys = [
        checkpoint_key(parent_workflow_id=PARENT, attestation_id="the-attestation"),
        attachment_key(fork_id=FORK, child_workflow_id=FIRST),
        preparation_key(fork_id=FORK, child_workflow_id=FIRST, creation_epoch=1),
        equality_key(fork_id=FORK, child_workflow_id=FIRST),
        binding_key(fork_id=FORK, child_workflow_id=FIRST),
        release_key(fork_id=FORK, child_workflow_id=FIRST),
    ]
    assert len(set(keys)) == len(keys)
    assert preparation_key(
        fork_id=FORK, child_workflow_id=FIRST, creation_epoch=1
    ) == fork_preparation_operation_identity(FORK, FIRST, 1)
    # The epoch a preparation was created under is inside its name, and the child and the fork are
    # too, so no two of them are one operation.
    assert preparation_key(
        fork_id=FORK, child_workflow_id=FIRST, creation_epoch=2
    ) != preparation_key(fork_id=FORK, child_workflow_id=FIRST, creation_epoch=1)
    assert binding_key(fork_id=FORK, child_workflow_id=FIRST) != binding_key(
        fork_id=FORK, child_workflow_id=SECOND
    )
    # A preparation that published no readiness froze no identity to be named by, and the name it
    # keeps its refusal under is one of these and not any of them.
    assert refused_preparation_key(fork_id=FORK, child_workflow_id=FIRST) not in keys
    # A name is a file name, whatever a workflow id happens to be made of.
    assert "/" not in attachment_key(fork_id=FORK, child_workflow_id=FIRST)


async def test_both_witnesses_of_one_freeze_are_read_and_kept_together(tmp_path: Path) -> None:
    """The stream's half is read off the generation and the harness's half is asked for.

    Neither of them is the boundary on its own. What the retrieval establishes is that the two are
    of one moment: the same acknowledgement, the same bytes, the same cursor and the same
    projection digest, compared before anything is fenced and before a child exists.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    parent = AParent()

    retrieval = await retrieve_checkpoint(parent, harness, operations)

    assert retrieval.parent_workflow_id == PARENT
    assert retrieval.source_attempt_id == ATTEMPT
    assert retrieval.attestation_id == "the-attestation"
    assert retrieval.checkpoint_manifest_reference == checkpoint_manifest_reference(a_manifest())
    assert retrieval.components == a_manifest().components
    assert retrieval.transcript_reference == TRANSCRIPT
    assert retrieval.acknowledgement_entry_sha256 == ACK_ENTRY
    assert (retrieval.acknowledged_cursor, retrieval.projection_digest) == (CURSOR, PROJECTION)
    assert retrieval.schema_version == FORK_OPERATION_SCHEMA_VERSION
    kept = checkpoint_key(parent_workflow_id=PARENT, attestation_id="the-attestation")
    assert operations.recorded(kept, CheckpointRetrieved) == retrieval


async def test_a_retrieval_asked_for_twice_is_answered_from_the_record_it_kept(
    tmp_path: Path,
) -> None:
    """The harness is asked once. What a second ask reads is what the first one established."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    asked: List[str] = []
    original = harness.checkpoint

    async def counted(**arguments: Any) -> CheckpointManifest:
        asked.append(arguments["attestation_id"])
        return await original(**arguments)

    harness.checkpoint = counted  # type: ignore[method-assign]
    first = await retrieve_checkpoint(AParent(), harness, operations)
    again = await retrieve_checkpoint(AParent(), ForkOperations.under(tmp_path), operations)

    assert again == first
    assert asked == ["the-attestation"]


@pytest.mark.parametrize(
    "changed",
    [
        {"acknowledgement_message_id": "another-message"},
        {"acknowledged_visible_sha256": digest_of("other bytes entirely")},
        {"acknowledged_cursor": "another-cursor"},
        {"projection_digest": digest_of("another projection")},
    ],
)
async def test_a_checkpoint_taken_at_another_moment_than_the_generation_recorded_is_refused(
    tmp_path: Path, changed: Dict[str, str]
) -> None:
    """A snapshot before the acknowledgement and a transcript after it both hash correctly.

    Each of these is a manifest that is internally sound and is about a moment this generation did
    not commit at, which is the substitution the pair of witnesses exists to catch. Nothing is kept
    and nothing is fenced.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers", manifest=a_manifest(**changed))

    with pytest.raises(ApplicationError) as raised:
        await retrieve_checkpoint(AParent(), harness, operations)

    assert raised.value.type == "witness_mismatch"
    assert operations.all_of(CheckpointRetrieved) == []


async def test_a_checkpoint_over_a_harness_that_still_owes_something_is_never_retrieved(
    tmp_path: Path,
) -> None:
    """Settlement comes before the fence, so a manifest that says otherwise is invalid here."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(
        tmp_path / "containers",
        manifest=a_manifest(
            settled=SettledHarness(
                transport=False, recovery=True, provider=True, model=True, compaction=True
            )
        ),
    )

    with pytest.raises(ApplicationError) as raised:
        await retrieve_checkpoint(AParent(), harness, operations)

    assert raised.value.type == "invalid_checkpoint"
    assert "transport" in str(raised.value)
    assert operations.all_of(CheckpointRetrieved) == []


async def test_a_generation_with_no_acknowledgement_standing_has_no_checkpoint_to_retrieve(
    tmp_path: Path,
) -> None:
    """A controller that asked while the agent was working is told to ask again, not refused."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    working = AParent(
        CheckpointEvidenceAnswer(found=False, reason="0 of this generation's attempts hold one")
    )

    with pytest.raises(ApplicationError) as raised:
        await retrieve_checkpoint(working, harness, operations)

    assert raised.value.type == "not_quiet"
    assert raised.value.non_retryable is False


async def test_a_harness_that_cannot_commit_a_checkpoint_says_so_rather_than_taking_one(
    tmp_path: Path,
) -> None:
    """An adapter that cannot supply an operation reports it, and this refuses on the reason."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers", unsupported="commit a checkpoint")

    with pytest.raises(UnsupportedCapability) as raised:
        await retrieve_checkpoint(AParent(), harness, operations)

    assert raised.value.capability == "commit a checkpoint"
    assert operations.all_of(CheckpointRetrieved) == []


def a_plan(slot: str, cell: str) -> ForkChildPlan:
    """One child a controller asks for, with the identities that child alone gets."""
    return ForkChildPlan(
        branch_slot=slot,
        dispositions=[],
        target_cell=cell,
        run_directory=f"/runs/{slot}",
        consumer_claim_hash=digest_of(f"the consumer of {slot}"),
        hidden_execution_id=f"execution-{slot}",
        frozen_plan_digest=digest_of("the plan both children are parked under"),
    )


PLANS = [a_plan("first", "graded"), a_plan("second", "placebo")]


async def test_the_typed_request_is_composed_from_the_witnesses_rather_than_from_a_caller(
    tmp_path: Path,
) -> None:
    """Every value a request carries about the freeze comes from a record rather than a memory."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)

    request = await fork_request_for(
        AParent(), retrieval=retrieval, fork_id=FORK, plans=PLANS
    )

    assert (request.parent_workflow_id, request.parent_run_id) == (PARENT, PARENT_RUN)
    assert request.parent_execution_ordinal == 0
    assert request.source_attempt_id == ATTEMPT
    assert request.attestation_id == "the-attestation"
    assert request.acknowledgement_message_id == ACK_MESSAGE
    assert request.acknowledged_visible_sha256 == VISIBLE
    assert (request.acknowledged_cursor, request.projection_digest) == (CURSOR, PROJECTION)
    assert request.checkpoint_manifest_reference == retrieval.checkpoint_manifest_reference
    assert request.child_plans == PLANS


async def test_a_fork_resubmitted_against_an_execution_that_moved_is_the_same_logical_fork(
    tmp_path: Path,
) -> None:
    """A turnover between the read and the request is ordinary: read again and submit again.

    The execution scope is what changed and the freeze is not, because a continuation preserves the
    cursor and the projection digest byte for byte. The digest that keys the fork excludes the run
    id for exactly this reason, so what comes back is the same fork rather than a second one.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    before = await fork_request_for(AParent(), retrieval=retrieval, fork_id=FORK, plans=PLANS)

    moved = AParent(
        CheckpointEvidenceAnswer(
            found=True,
            evidence=an_evidence(parent_run_id="the-execution-after-the-boundary"),
        )
    )
    after = await fork_request_for(moved, retrieval=retrieval, fork_id=FORK, plans=PLANS)

    assert after.parent_run_id == "the-execution-after-the-boundary"
    assert fork_request_digest(after) == fork_request_digest(before)


@pytest.mark.parametrize(
    "changed",
    [
        {"source_attempt_id": "another-attempt"},
        {"attestation_id": "another-attestation"},
        {"acknowledged_cursor": "a-cursor-further-on"},
        {"projection_digest": digest_of("the projection somewhere else")},
    ],
)
async def test_a_generation_that_moved_past_the_freeze_is_refused_the_fork_it_was_read_for(
    tmp_path: Path, changed: Dict[str, str]
) -> None:
    """A fresh cursor over an old snapshot would name a freeze nobody took."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    moved = AParent(CheckpointEvidenceAnswer(found=True, evidence=an_evidence(**changed)))

    with pytest.raises(ApplicationError) as raised:
        await fork_request_for(moved, retrieval=retrieval, fork_id=FORK, plans=PLANS)

    assert raised.value.type == "witness_mismatch"


def a_child_start(store: Path) -> StreamStart:
    """As much of one child's start as the copy into its own store reads: where that store is."""
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash="d" * 64,
        initial_cursor=ACK_MESSAGE,
        done_message_id="the-message-that-ends-it",
        id_key_hex="ab" * 32,
        hidden_execution_id="the-execution-of-the-first-child",
        canonicalization_version="kernel.1",
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=[]
        ),
        tasks=[],
        blob_root=str(store),
    )


def a_child_receipt(child: str, *, selected: str) -> ForkChildReceipt:
    """What one fork recorded about one child it created."""
    return ForkChildReceipt(
        child_ordinal=1,
        child_workflow_id=child,
        child_run_id=f"the-run-of-{child}",
        configuration_hash=digest_of(f"what {child} is"),
        complete_start_digest=digest_of(f"the start {child} was created with"),
        origin_digest=digest_of(f"the lineage of {child}"),
        branch_slot="first",
        target_cell="graded",
        selected_body_reference=selected,
        acknowledged_cursor=CURSOR,
        projection_digest=PROJECTION,
        start_differences=[],
        next_task_body_sha256=digest_of("the task both children work next"),
        next_assignment_id="the-assignment-of-that-task",
    )


def a_fork_receipt(*children: str) -> ForkReceipt:
    """The typed evidence one fork answered with, which is the roster a release is over.

    It is the parent's own record of which children it created, so a release derived from it is a
    decision about the whole fork rather than about whichever children a caller listed.
    """
    return ForkReceipt(
        fork_id=FORK,
        parent_workflow_id="stream/the-parent/1",
        parent_run_id="the-run-of-the-parent",
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_manifest()),
        children=len(children),
        child_receipts=[
            replace(
                a_child_receipt(child, selected=digest_of(f"the body of {child}")),
                child_ordinal=ordinal,
            )
            for ordinal, child in enumerate(children, start=1)
        ],
        boundary_evidence=[],
    )


class ALosingStore(FilesystemBlobStore):
    """A store that takes an object and does not keep it, which is what an interrupted copy is."""

    def put(self, data: bytes, *, media_type: str = "application/octet-stream") -> BlobRef:
        return BlobRef(sha256=sha256(data).hexdigest(), size=len(data), media_type=media_type)


@dataclass(frozen=True)
class AVanishingSource(FilesystemBlobStore):
    """A parent store that produces one object once and no longer holds it at the second read.

    The copy reads every required object twice, once to establish that the source can produce it
    and once to install it, and the interval between those two reads is a real one: the objects
    are files in a directory and nothing holds them open across it.
    """

    lose: str = ""
    reads: List[str] = field(default_factory=list)

    def read(self, digest: str) -> bytes:
        self.reads.append(digest)
        if digest == self.lose and self.reads.count(digest) == 2:
            self.path_for(digest).unlink()
        return super().read(digest)


def a_lineage() -> ForkOrigin:
    """The lineage a child carries, which is what makes its start a complete child start."""
    return ForkOrigin(
        parent_workflow_id=PARENT,
        parent_run_id="the-run-of-the-parent",
        parent_configuration_hash=digest_of("what the parent is"),
        child_configuration_hash="c" * 64,
        parent_execution_ordinal=0,
        parent_hidden_execution_id="the-execution-of-the-parent",
        acknowledged_cursor=CURSOR,
        projection_digest=PROJECTION,
        attestation_id="the-attestation",
        acknowledged_visible_sha256=VISIBLE,
        checkpoint_manifest_reference=checkpoint_manifest_reference(a_manifest()),
        source_attempt_id=ATTEMPT,
        source_seal_id="the-seal-of-the-parent",
        source_submission_digest=digest_of("what the parent submitted"),
        source_canonicalization_version="kernel.1",
        source_score=1.0,
        source_seal_ordinal=1,
        source_graded_evidence=digest_of("the verdict behind that score"),
        source_commitment=digest_of("the source the parent committed"),
        source_artifact_references=[digest_of("the body of the first child")],
        branch_slot="first",
        dispositions_digest=digest_of("the rows of first"),
        start_differences=[],
        parent_turnovers=0,
        fork_id=FORK,
        child_ordinal=1,
        children=2,
    )


class AZlibCodec(PayloadCodec):
    """A codec of the kind a deployment configures, which the default converter cannot read."""

    async def encode(self, payloads: Sequence[Payload]) -> List[Payload]:
        return [
            Payload(
                metadata={"encoding": b"binary/test-zlib"},
                data=zlib.compress(one.SerializeToString()),
            )
            for one in payloads
        ]

    async def decode(self, payloads: Sequence[Payload]) -> List[Payload]:
        return [
            Payload.FromString(zlib.decompress(one.data))
            if one.metadata.get("encoding") == b"binary/test-zlib"
            else one
            for one in payloads
        ]


class AStartedChild:
    """A child handle whose history is the one start event that created it."""

    def __init__(self, recorded: Payloads) -> None:
        self.history = SimpleNamespace(
            events=[
                SimpleNamespace(
                    workflow_execution_started_event_attributes=SimpleNamespace(
                        input=recorded
                    )
                )
            ]
        )

    async def fetch_history(self) -> Any:
        return self.history


class ACodecClient:
    """A client configured the way the one that created the child was, codec and all."""

    def __init__(self, data_converter: DataConverter, recorded: Payloads) -> None:
        self.data_converter = data_converter
        self.recorded = recorded

    def get_workflow_handle(self, *_arguments: Any, **_named: Any) -> AStartedChild:
        return AStartedChild(self.recorded)


async def test_a_childs_recorded_start_is_read_through_the_converter_it_was_created_with(
    tmp_path: Path,
) -> None:
    """The producer and the consumer of a recorded start are one configuration.

    A child is created through the client's own DataConverter, so a deployment that configured a
    codec wrote bytes the default converter cannot decode at all. Reading the history back through
    anything else would refuse a child the service created correctly, which is the one thing a
    missing history may never do: the recorded identity stands, and a replacement under it is never
    licensed.
    """
    configured = DataConverter(payload_codec=AZlibCodec())
    start = replace(a_child_start(tmp_path / "the-child"), fork_origin=a_lineage())
    recorded = Payloads(payloads=await configured.encode([start]))
    assert recorded.payloads[0].metadata["encoding"] == b"binary/test-zlib"
    child = replace(
        a_child_receipt(FIRST, selected=digest_of("the body of the first child")),
        complete_start_digest=complete_start_digest(start),
    )

    assert await child_start(ACodecClient(configured, recorded), child=child) == start

    # And the comparison it is read for still holds: a start that is not the recorded one is
    # refused after the decoding rather than instead of it.
    with pytest.raises(ForkRefused) as raised:
        await child_start(
            ACodecClient(configured, recorded),
            child=replace(child, complete_start_digest=digest_of("another start")),
        )
    assert raised.value.reason == FORK_ORIGIN_DISAGREEMENT


def test_a_required_object_the_parent_cannot_produce_is_a_repairable_absence(
    tmp_path: Path,
) -> None:
    """A child's store is filled from the parent's, so an object nobody has is a repair.

    It is retryable rather than a decision: the bytes are content addressed, an independently
    verified copy of them is the same object, and putting them back is what a controller does
    before it asks again.
    """
    source = FilesystemBlobStore(tmp_path / "the-parent")
    held = source.put(b"the inventory the prefix committed")
    lost = digest_of("the body this child was selected for")

    with pytest.raises(ApplicationError) as raised:
        _installed(
            source,
            a_child_start(tmp_path / "the-child"),
            a_child_receipt(FIRST, selected=lost),
            [held.sha256, lost],
            [],
        )

    assert raised.value.type == FORK_REPAIRABLE_ABSENCE
    assert lost in str(raised.value)


def test_a_required_object_lost_while_it_is_being_copied_is_a_repairable_absence_too(
    tmp_path: Path,
) -> None:
    """The verification and the copy are two reads, and an object can go missing between them.

    What the source could produce a moment ago is not what it can produce now, and the loss has
    the same repair as one found before either read: the bytes are content addressed, an
    independently verified copy of them is the same object, and putting them back and asking
    again is what a controller does. So it is reported as the repairable absence it is, naming
    the object, rather than as whatever the store raised on the way past.

    The objects installed before it stay installed. Copying is idempotent under retry and there
    is nothing to roll back: each of them is the object its name promises, and the retry writes
    nothing where the bytes are already right.
    """
    source = FilesystemBlobStore(tmp_path / "the-parent")
    held = source.put(b"the inventory the prefix committed")
    body = b"the body this child was selected for"
    lost = source.put(body)
    vanishing = AVanishingSource(source.root, lose=lost.sha256)
    child = a_child_receipt(FIRST, selected=lost.sha256)
    start = a_child_start(tmp_path / "the-child")

    with pytest.raises(ApplicationError) as raised:
        _installed(vanishing, start, child, [held.sha256, lost.sha256], [])

    assert raised.value.type == FORK_REPAIRABLE_ABSENCE
    assert lost.sha256 in str(raised.value)
    # The source produced it once and could not produce it again, which is the interval this is
    # about rather than a store that never held it.
    assert vanishing.reads.count(lost.sha256) == 2
    assert vanishing.unverified([lost.sha256]) == [lost.sha256]
    # And what was already copied is still copied.
    into = FilesystemBlobStore(tmp_path / "the-child")
    assert into.unverified([held.sha256]) == []
    assert into.unverified([lost.sha256]) == [lost.sha256]

    # The repair is those exact bytes back in the source, and the same child asked again.
    assert source.put(body).sha256 == lost.sha256
    assert _installed(source, start, child, [held.sha256, lost.sha256], []) == []
    assert into.unverified([held.sha256, lost.sha256]) == []


def test_an_object_that_does_not_read_back_after_it_was_installed_is_an_interrupted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy's own acceptance test is reading the objects back out of the store it filled.

    A name is a promise about bytes, so a store that took an object and cannot produce it holds
    nothing under that name. It is the copy that was interrupted rather than the source that was
    wrong, and reinstalling the same authoritative bytes is what repairs it.
    """
    source = FilesystemBlobStore(tmp_path / "the-parent")
    held = source.put(b"the inventory the prefix committed")
    monkeypatch.setattr("shogym.serve.protocol_v2.fork.FilesystemBlobStore", ALosingStore)

    with pytest.raises(ApplicationError) as raised:
        _installed(
            source,
            a_child_start(tmp_path / "the-child"),
            a_child_receipt(FIRST, selected=held.sha256),
            [held.sha256],
            [],
        )

    assert raised.value.type == FORK_INTERRUPTED_COPY
    assert held.sha256 in str(raised.value)
    assert not (tmp_path / "the-child").exists()


def test_a_retained_object_the_parent_cannot_produce_is_reported_and_blocks_nothing(
    tmp_path: Path,
) -> None:
    """The oracle body is the case: no operation the fork performs reads it.

    A row saying an attempt's evidence is unavailable would move that attempt's availability while
    its evidence is intact, so what a lineage lost is an observation the attachment carries.
    """
    source = FilesystemBlobStore(tmp_path / "the-parent")
    held = source.put(b"the inventory the prefix committed")
    oracle = digest_of("the cell nothing here reads")

    absent = _installed(
        source,
        a_child_start(tmp_path / "the-child"),
        a_child_receipt(FIRST, selected=held.sha256),
        [held.sha256],
        [oracle],
    )

    assert absent == [oracle]
    assert FilesystemBlobStore(tmp_path / "the-child").unverified([held.sha256]) == []


async def test_a_readiness_published_and_lost_is_read_back_rather_than_prepared_again(
    tmp_path: Path,
) -> None:
    """The child publishes once, and the record of it is what a lost reply is recovered by.

    The name the answer is kept under is the identity the child froze at its first attempt, which
    is what the readiness carries, so the answer the controller kept and the operation the child
    ran are the same one.
    """
    operations = ForkOperations.under(tmp_path)
    attachment = an_attachment(FIRST)
    child = AChild(a_readiness(FIRST))

    prepared = await prepare_child(child, operations, attachment=attachment)

    assert prepared.ready == a_readiness(FIRST)
    assert prepared.reason == ""
    assert prepared.preparation_operation == a_readiness(FIRST).preparation_operation
    again = await prepare_child(child, ForkOperations.under(tmp_path), attachment=attachment)
    assert again == prepared
    assert child.preparations == 1


async def test_a_preparation_refused_for_good_is_kept_under_the_episode_it_was_refused_in(
    tmp_path: Path,
) -> None:
    """A decision is recorded and never retried, which is what a controller reads it for.

    It is kept under the episode's own identity, like the readiness it stands in place of, because
    the two are the outcomes of one operation and the surface is keyed by that identity. The child
    says which epoch it froze and this side derives the name from it, so what a file is called is
    never a string a transport handed over.
    """
    operations = ForkOperations.under(tmp_path)
    attachment = an_attachment(FIRST)
    child = AChild(
        refusal=ForkRefused(
            FORK_WRONG_SOURCE, "this result names another source", creation_epoch=3
        )
    )

    prepared = await prepare_child(child, operations, attachment=attachment)

    assert (prepared.ready, prepared.reason) == (None, FORK_WRONG_SOURCE)
    frozen = preparation_key(fork_id=FORK, child_workflow_id=FIRST, creation_epoch=3)
    assert prepared.preparation_operation == frozen
    assert operations.recorded(frozen, ChildPrepared) == prepared
    assert (
        operations.recorded(
            refused_preparation_key(fork_id=FORK, child_workflow_id=FIRST), ChildPrepared
        )
        is None
    )
    assert await prepare_child(child, ForkOperations.under(tmp_path), attachment=attachment) == (
        prepared
    )
    assert child.preparations == 1


async def test_a_preparation_refused_before_any_episode_existed_is_kept_under_the_pair(
    tmp_path: Path,
) -> None:
    """The other case, and it is a real one rather than a fallback nobody reaches.

    A generation cut from no fork, or one owing no payload for the attempt its lineage names, is
    refused before a preparation exists to be named. There is no episode to key the decision by,
    so it is kept under the child of the fork it is a decision about, and it says so by naming
    no operation at all.
    """
    operations = ForkOperations.under(tmp_path)
    child = AChild(
        refusal=ForkRefused(FORK_WRONG_SOURCE, "this generation was cut from no fork")
    )

    prepared = await prepare_child(child, operations, attachment=an_attachment(FIRST))

    assert prepared.preparation_operation == ""
    assert operations.recorded(
        refused_preparation_key(fork_id=FORK, child_workflow_id=FIRST), ChildPrepared
    ) == prepared


async def test_a_preparation_refused_over_an_object_that_can_be_repaired_is_not_kept_at_all(
    tmp_path: Path,
) -> None:
    """An absence a controller repairs leaves one logical preparation to be run again.

    A record of it would answer the repair with the failure it repaired, so nothing is kept and the
    refusal reaches the controller that has another verified copy of the object.
    """
    operations = ForkOperations.under(tmp_path)
    child = AChild(refusal=ForkRefused(FORK_REPAIRABLE_ABSENCE, "the store lost the body"))

    with pytest.raises(ApplicationError) as raised:
        await prepare_child(child, operations, attachment=an_attachment(FIRST))

    assert raised.value.type == FORK_REPAIRABLE_ABSENCE
    assert operations.all_of(ChildPrepared) == []


async def test_a_readiness_under_an_operation_this_child_could_not_have_frozen_is_refused(
    tmp_path: Path,
) -> None:
    """An identity minted for another child, or at an epoch this one never held, is not its own."""
    operations = ForkOperations.under(tmp_path)
    borrowed = AChild(
        a_readiness(
            FIRST, preparation_operation=fork_preparation_operation_identity(FORK, SECOND, 1)
        )
    )

    with pytest.raises(ApplicationError) as raised:
        await prepare_child(borrowed, operations, attachment=an_attachment(FIRST))

    assert raised.value.type == "origin_disagreement"

    ahead = AChild(
        a_readiness(
            FIRST, preparation_operation=fork_preparation_operation_identity(FORK, FIRST, 9)
        )
    )
    with pytest.raises(ApplicationError) as raised:
        await prepare_child(ahead, operations, attachment=an_attachment(FIRST))

    assert raised.value.type == "origin_disagreement"

    for_another = AChild(a_readiness(FIRST, fork_id="another-fork"))
    with pytest.raises(ApplicationError) as raised:
        await prepare_child(for_another, operations, attachment=an_attachment(FIRST))

    assert raised.value.type == "origin_disagreement"
    assert operations.all_of(ChildPrepared) == []


async def test_a_child_attached_again_before_it_prepared_is_kept_under_the_epoch_it_froze(
    tmp_path: Path,
) -> None:
    """A crash between the attachment and the preparation is an ordinary thing to come back from.

    The attachment is performed again, which is what gives a controller that came back a live
    transport, and the claim that does it installs a later epoch than the one the record holds. The
    child's first preparation is therefore created under an epoch no record names, so the identity
    it freezes is read off the readiness rather than derived from the attachment, and the answer is
    kept under it.
    """
    operations = ForkOperations.under(tmp_path)
    attachment = an_attachment(FIRST, epoch=1)
    child = AChild(a_readiness(FIRST, epoch=2))

    prepared = await prepare_child(child, operations, attachment=attachment)

    assert prepared.preparation_operation == fork_preparation_operation_identity(FORK, FIRST, 2)
    assert operations.recorded(
        preparation_key(fork_id=FORK, child_workflow_id=FIRST, creation_epoch=2), ChildPrepared
    ) == prepared
    came_back = ForkOperations.under(tmp_path)
    assert await prepare_child(child, came_back, attachment=attachment) == prepared
    assert child.preparations == 1


async def a_bound_fork(
    harness: AHarness, operations: ForkOperations, retrieval: CheckpointRetrieved
) -> List[ChildBound]:
    """Take both children from their readiness to their bindings, which is what a release reads."""
    bound = []
    for child in CHILDREN:
        comparison = await a_compared_child(harness, operations, child, retrieval)
        bound.append(
            await bind_child(
                harness, operations, attachment=an_attachment(child), comparison=comparison
            )
        )
    return bound


async def test_one_container_is_restored_for_each_child_and_compared_against_the_parent(
    tmp_path: Path,
) -> None:
    """A copy per child, compared under the allowlist, and restored exactly once.

    The restore is the operation a repeat would cost the most, because a second copy over a child
    that is already working would put the parent's state back over work that happened. So the
    record is read before the harness is asked, and the container it names is the one that child
    keeps.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)

    comparisons = [
        await a_compared_child(harness, operations, child, retrieval) for child in CHILDREN
    ]

    assert harness.restores == list(CHILDREN)
    assert len({comparison.container_id for comparison in comparisons}) == 2
    for comparison in comparisons:
        assert comparison.equal
        assert comparison.differences == []
        assert comparison.rebound == list(REBOUND)
        assert comparison.allowlist_version == ALLOWLIST_VERSION
    # Asked again, the record answers and the harness is not asked to restore a second copy.
    again = await a_compared_child(harness, ForkOperations.under(tmp_path), FIRST, retrieval)
    assert again == comparisons[0]
    assert harness.restores == list(CHILDREN)


async def test_a_container_restored_from_a_snapshot_taken_before_the_acknowledgement_is_not_equal(
    tmp_path: Path,
) -> None:
    """A digest validates bytes and never proves one object restores another.

    This copy hashes correctly, holds every other file the parent holds, and came back without the
    acknowledgement in its transcript. The manifest's assertion was preflight evidence and the
    attestation after the restore is the evidence afterwards, and it is this side that compares
    them.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(
        tmp_path / "containers",
        restored_as=lambda path: (path / "transcript.jsonl").write_text(
            "the transcript the harness persisted", encoding="utf-8"
        ),
    )
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)

    comparison = await a_compared_child(harness, operations, FIRST, retrieval)

    assert not comparison.equal
    assert any("transcript" in difference for difference in comparison.differences)
    assert any("acknowledgement entry" in difference for difference in comparison.differences)


async def test_a_child_cut_at_another_checkpoint_is_never_compared_against_this_one(
    tmp_path: Path,
) -> None:
    """One fork, one freeze: a readiness naming another checkpoint is not this fork's child."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    elsewhere = await prepare_child(
        AChild(a_readiness(FIRST, checkpoint_manifest_reference=digest_of("another freeze"))),
        operations,
        attachment=an_attachment(FIRST),
    )

    with pytest.raises(ApplicationError) as raised:
        await compare_child(harness, operations, retrieval=retrieval, preparation=elsewhere)

    assert raised.value.type == "witness_mismatch"
    assert harness.restores == []


async def test_a_child_that_published_no_readiness_has_no_container_restored_for_it(
    tmp_path: Path,
) -> None:
    """A container is restored for a child that is ready to serve one, and for no other."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    refused = await prepare_child(
        AChild(refusal=ForkRefused(FORK_WRONG_SOURCE, "this result names another source")),
        operations,
        attachment=an_attachment(FIRST),
    )

    with pytest.raises(ValueError):
        await compare_child(harness, operations, retrieval=retrieval, preparation=refused)

    assert harness.restores == []
    assert operations.all_of(ChildCompared) == []


async def test_a_harness_that_cannot_restore_a_container_leaves_nothing_recorded(
    tmp_path: Path,
) -> None:
    """An unsupported capability is refused rather than approximated, and nothing is kept."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    harness.unsupported = "restore a container"

    with pytest.raises(UnsupportedCapability) as raised:
        await a_compared_child(harness, operations, FIRST, retrieval)

    assert raised.value.capability == "restore a container"
    assert operations.all_of(ChildCompared) == []


async def test_one_container_serves_one_child_and_a_second_child_is_refused_it(
    tmp_path: Path,
) -> None:
    """The binding is unique both ways, and the lineage's own record is what says so."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    comparison = await a_compared_child(harness, operations, FIRST, retrieval)

    binding = await bind_child(
        harness, operations, attachment=an_attachment(FIRST), comparison=comparison
    )

    assert binding.container_id == comparison.container_id
    assert binding.consumer_id == an_attachment(FIRST).consumer_id
    assert harness.bound == {FIRST: comparison.container_id}
    # A second child whose own recorded comparison names a container the first is bound to is
    # refused by the scan over what is bound already.
    operations.record(
        equality_key(fork_id=FORK, child_workflow_id=SECOND),
        replace(comparison, child_workflow_id=SECOND),
    )
    with pytest.raises(ValueError):
        await bind_child(
            harness,
            operations,
            attachment=an_attachment(SECOND),
            comparison=replace(comparison, child_workflow_id=SECOND),
        )
    assert harness.bound == {FIRST: comparison.container_id}


async def test_a_container_is_bound_only_on_the_comparison_kept_for_that_exact_child(
    tmp_path: Path,
) -> None:
    """Two evidence fields meet here, and neither of them is checked by the other's presence.

    The identity a binding is keyed by comes from the attachment and the container comes from the
    comparison, so two siblings' comparisons swapped between them would bind one child to the
    container restored for the other. The scan over what is bound already cannot see it, because
    at that moment neither container is bound to anything. So the identity fields are compared
    first, and then the comparison this lineage kept for this exact pair is what the container is
    read out of: a value handed in is checked against the record rather than trusted beside it.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    first = await a_compared_child(harness, operations, FIRST, retrieval)
    second = await a_compared_child(harness, operations, SECOND, retrieval)
    assert first.container_id != second.container_id

    with pytest.raises(ForkRefused) as raised:
        await bind_child(
            harness, operations, attachment=an_attachment(FIRST), comparison=second
        )
    assert raised.value.reason == FORK_ORIGIN_DISAGREEMENT
    assert SECOND in raised.value.clause and FIRST in raised.value.clause

    # And a comparison of the right child that is not the one this lineage kept for it.
    with pytest.raises(ForkRefused) as again:
        await bind_child(
            harness,
            operations,
            attachment=an_attachment(FIRST),
            comparison=replace(first, container_id=second.container_id),
        )
    assert again.value.reason == FORK_ORIGIN_DISAGREEMENT
    assert "is not the one" in again.value.clause

    assert harness.bound == {}
    assert operations.all_of(ChildBound) == []


async def test_a_container_that_did_not_compare_equal_is_bound_to_nothing(
    tmp_path: Path,
) -> None:
    """Binding comes after a comparison that passed, and never instead of one."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(
        tmp_path / "containers",
        restored_as=lambda path: (path / "weights.bin").write_bytes(b"weights of its own"),
    )
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    comparison = await a_compared_child(harness, operations, FIRST, retrieval)

    with pytest.raises(ValueError):
        await bind_child(
            harness, operations, attachment=an_attachment(FIRST), comparison=comparison
        )

    assert harness.bound == {}
    assert operations.all_of(ChildBound) == []


async def test_both_children_are_released_and_each_ones_first_act_is_a_pull(
    tmp_path: Path,
) -> None:
    """The last step of a fork, and what a released container does with no message injected."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    await a_bound_fork(harness, operations, retrieval)

    # Everything below would resume both containers, so the version is what stops this one: a
    # receipt this build does not read is refused where it is received, before it becomes a
    # decision about anything, rather than consumed for the fields the decision is made out of.
    with pytest.raises(WireFormatError) as unreadable:
        await release_children(
            harness,
            operations,
            receipt=replace(a_fork_receipt(*CHILDREN), schema_version=999),
        )
    assert "999" in str(unreadable.value)
    assert harness.resumed == []
    assert operations.all_of(ChildReleased) == []

    released = await release_children(harness, operations, receipt=a_fork_receipt(*CHILDREN))

    assert [one.child_workflow_id for one in released] == list(CHILDREN)
    assert all(one.resumed for one in released)
    assert [one.first_message_id for one in released] == [
        f"the-first-pull-of-{child}" for child in CHILDREN
    ]
    assert harness.resumed == list(CHILDREN)
    for child, one in zip(CHILDREN, released):
        assert operations.recorded(
            release_key(fork_id=FORK, child_workflow_id=child), ChildReleased
        ) == one


async def test_a_controller_that_crashed_after_the_first_release_resumes_only_the_one_still_paused(
    tmp_path: Path,
) -> None:
    """The child that is already working is recovered from its record rather than restored over.

    A release is the operation whose repeat would be worst: the first child may have worked its
    next task by now, and putting the parent's state back over that would spend the measurement.
    So the record answers for it and only the still-paused container is resumed.

    The asymmetry is made by a process that went away between the two rather than by a caller
    naming one child, because a caller that could name one would be choosing which children the
    whole-fork gate is about. The roster is the fork's own receipt either way.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers", crash_after=1)
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    bound = await a_bound_fork(harness, operations, retrieval)
    with pytest.raises(RuntimeError):
        await release_children(harness, operations, receipt=a_fork_receipt(*CHILDREN))
    assert harness.resumed == [FIRST]
    first = operations.recorded(
        release_key(fork_id=FORK, child_workflow_id=FIRST), ChildReleased
    )
    assert first is not None and first.resumed
    assert (
        operations.recorded(
            release_key(fork_id=FORK, child_workflow_id=SECOND), ChildReleased
        )
        is None
    )

    harness.crash_after = None
    came_back = ForkOperations.under(tmp_path)
    released = await release_children(harness, came_back, receipt=a_fork_receipt(*CHILDREN))

    assert harness.resumed == [FIRST, SECOND]
    assert released[0] == first
    assert released[1].container_id == bound[1].container_id
    assert all(one.resumed for one in released)


class ALosingJournal(ForkOperations):
    """A journal that performs one write and then loses the process that was making it.

    It is the interval nothing else reaches: the adapter has already changed something outside
    this controller and the record of it has not reached the disk. What recovers it is not a
    smaller window, because there is no window small enough; it is the operation being one an
    outcome can be read back from.
    """

    def record(self, key: str, result: Any) -> Any:
        if key.startswith("release."):
            raise RuntimeError(f"the process recording {key} went away")
        return super().record(key, result)


async def test_a_release_that_happened_and_was_never_recorded_is_read_back_rather_than_repeated(
    tmp_path: Path,
) -> None:
    """The interval between an effect outside this process and the record of it.

    A container resumed and not written down is the worst crash a release has: the agent inside it
    is already working, and a controller that came back and simply resumed again would either
    restore over that work or record a second resumption of it. So the operation is one an outcome
    can be read back from, and the recovery is asking it again and being answered with the
    resumption that already happened.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    bound = await a_bound_fork(harness, operations, retrieval)

    with pytest.raises(RuntimeError):
        await release_children(
            harness,
            ALosingJournal(operations.root),
            receipt=a_fork_receipt(*CHILDREN),
        )
    assert harness.resumed == [FIRST]
    assert operations.all_of(ChildReleased) == []

    came_back = ForkOperations.under(tmp_path)
    released = await release_children(harness, came_back, receipt=a_fork_receipt(*CHILDREN))

    # One resumption per child, and the container each of them names is the one it was bound to.
    assert harness.resumed == [FIRST, SECOND]
    assert [one.container_id for one in released] == [one.container_id for one in bound]
    assert all(one.resumed for one in released)
    assert len(came_back.all_of(ChildReleased)) == 2


async def test_a_restore_that_happened_and_was_never_recorded_is_read_back_rather_than_repeated(
    tmp_path: Path,
) -> None:
    """The same interval one step earlier, where the copy exists and the record does not.

    A second copy over a child that already has one would be a second mutable state for one
    generation, which is the whole thing the copies exist to prevent, so the adapter answers with
    the copy that exists and this side records it once.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    prepared = await prepare_child(
        AChild(a_readiness(FIRST)), operations, attachment=an_attachment(FIRST)
    )

    class ALosingEquality(ForkOperations):
        def record(self, key: str, result: Any) -> Any:
            if key.startswith("equality."):
                raise RuntimeError(f"the process recording {key} went away")
            return super().record(key, result)

    with pytest.raises(RuntimeError):
        await compare_child(
            harness,
            ALosingEquality(operations.root),
            retrieval=retrieval,
            preparation=prepared,
        )
    assert harness.restores == [FIRST]
    assert operations.all_of(ChildCompared) == []

    came_back = ForkOperations.under(tmp_path)
    comparison = await compare_child(
        harness, came_back, retrieval=retrieval, preparation=prepared
    )
    assert harness.restores == [FIRST]
    assert comparison.container_id == harness.container(FIRST).name
    assert comparison.equal


@pytest.mark.parametrize("failing", [FIRST, SECOND])
async def test_a_child_whose_container_did_not_compare_equal_holds_the_whole_fork(
    tmp_path: Path, failing: str
) -> None:
    """Neither container is resumed, and the child that was fine is held rather than promoted.

    A link with one arm running is not the comparison anybody registered, so the failure stands
    against the exact child it names and the other one stays paused with the reason it is paused
    for.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(
        tmp_path / "containers",
        restored_as=lambda path: (
            (path / "memory.md").write_text("what another agent learned", encoding="utf-8")
            if path.name.endswith(failing.rsplit(".", 1)[-1])
            else None
        ),
    )
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    for child in CHILDREN:
        comparison = await a_compared_child(harness, operations, child, retrieval)
        if comparison.equal:
            await bind_child(
                harness, operations, attachment=an_attachment(child), comparison=comparison
            )

    released = await release_children(harness, operations, receipt=a_fork_receipt(*CHILDREN))

    assert harness.resumed == []
    assert not any(one.resumed for one in released)
    held = {one.child_workflow_id: one.held for one in released}
    assert "memory.md" in held[failing]
    other = next(child for child in CHILDREN if child != failing)
    assert failing in held[other]
    assert operations.all_of(ChildReleased) == []


async def test_a_release_is_over_the_roster_the_fork_recorded_and_never_a_chosen_part_of_it(
    tmp_path: Path,
) -> None:
    """The whole-pair gate is about the fork's own children rather than about a caller's list.

    A subset would make the hold a statement about whichever children were named: one child ready,
    compared and bound, its sibling entirely unprepared, and a release over the first alone would
    resume one arm against a sibling nobody had established was ready. So the roster comes from
    the parent's typed evidence, and the sibling's absence is what holds both.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    comparison = await a_compared_child(harness, operations, FIRST, retrieval)
    await bind_child(harness, operations, attachment=an_attachment(FIRST), comparison=comparison)

    released = await release_children(harness, operations, receipt=a_fork_receipt(*CHILDREN))

    assert harness.resumed == []
    assert [one.child_workflow_id for one in released] == list(CHILDREN)
    assert not any(one.resumed for one in released)
    assert SECOND in {one.child_workflow_id: one.held for one in released}[FIRST]
    assert operations.all_of(ChildReleased) == []

    # And a receipt that does not name the children it says it created is not a roster.
    with pytest.raises(ForkRefused) as raised:
        await release_children(
            harness, operations, receipt=replace(a_fork_receipt(FIRST), children=2)
        )
    assert raised.value.reason == FORK_ORIGIN_DISAGREEMENT
    assert harness.resumed == []


async def test_a_child_nothing_bound_a_container_to_holds_the_fork_the_same_way(
    tmp_path: Path,
) -> None:
    """Readiness and equality are read out of the record, so a step nobody took holds the rest."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    comparison = await a_compared_child(harness, operations, FIRST, retrieval)
    await bind_child(harness, operations, attachment=an_attachment(FIRST), comparison=comparison)

    released = await release_children(harness, operations, receipt=a_fork_receipt(*CHILDREN))

    assert harness.resumed == []
    assert [one.resumed for one in released] == [False, False]
    assert "restored and compared" in {
        one.child_workflow_id: one.held for one in released
    }[SECOND]


async def test_a_harness_that_cannot_resume_a_container_leaves_both_of_them_paused(
    tmp_path: Path,
) -> None:
    """An unsupported release is refused rather than approximated, and nothing is recorded."""
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    await a_bound_fork(harness, operations, retrieval)
    harness.unsupported = "resume a container"

    with pytest.raises(UnsupportedCapability):
        await release_children(harness, operations, receipt=a_fork_receipt(*CHILDREN))

    assert harness.resumed == []
    assert operations.all_of(ChildReleased) == []


async def test_each_child_owns_the_files_memory_buffers_and_weights_it_was_given(
    tmp_path: Path,
) -> None:
    """One mutable state per child, so no two of them and no child and its parent share one.

    The copies are the point rather than a convenience: two children writing one memory store, one
    buffer or one set of weights would be one learner reported as two, and the treatment nobody
    could tell apart afterwards.
    """
    operations = ForkOperations.under(tmp_path)
    harness = AHarness(tmp_path / "containers")
    retrieval = await retrieve_checkpoint(AParent(), harness, operations)
    bound = await a_bound_fork(harness, operations, retrieval)
    await release_children(harness, operations, receipt=a_fork_receipt(*CHILDREN))

    working = harness.root / bound[0].container_id
    (working / "memory.md").write_text("what this child learned", encoding="utf-8")
    (working / "buffer.bin").write_bytes(b"what this child is holding")
    (working / "weights.bin").write_bytes(b"the weights this child updated")
    (working / "notes.txt").write_text("a file only this child has", encoding="utf-8")

    sibling = harness.root / bound[1].container_id
    for name in ("memory.md", "buffer.bin", "weights.bin"):
        assert (sibling / name).read_bytes() == (harness.parent / name).read_bytes()
        assert (working / name).read_bytes() != (harness.parent / name).read_bytes()
    assert not (sibling / "notes.txt").exists()
    assert not (harness.parent / "notes.txt").exists()


#: Every record a continuation's own inventory keeps, and the member of a checkpoint's settled half
#: that says that record is finished. The freeze is over the harness's whole state and not over its
#: transcript alone, so one entry here is one thing a crash can leave owed.
THE_RECORD_UNION = ("transport", "recovery", "provider", "model", "compaction")

#: And every state the gateway's own recovery record can be in, which is what the recovery member
#: above is an attestation about. Eight of the nine are a call or an effect this transport still
#: owes and the ninth is the settled one, so the inventory is held against the union itself rather
#: than against a part of it somebody chose: a record added later fails this rather than quietly
#: becoming a state no freeze accounts for.
THE_RECOVERY_UNION = (
    "_Idle",
    "_RequestUncertain",
    "_PullRecovered",
    "_LeaseHeld",
    "_HorizonOwed",
    "_Offered",
    "_PresentationUncertain",
    "_PresentationRefused",
    "_ResultOwed",
)


async def test_no_unfinished_state_of_a_harness_yields_accepted_freeze_evidence(
    tmp_path: Path,
) -> None:
    """The inventory a freeze is attested over, and the manifest check that reads it.

    The harness settles and then the platform fences, so what a checkpoint is over is the whole of
    the harness's state. The settled half is exhaustive against that inventory rather than a chosen
    part of it: five records, five members, and a manifest owing any one of them is refused where
    it is read. The recovery member is an attestation about the transport's own record, so the
    inventory it answers for is held against that union rather than against a chosen part of it.

    What is proved here is the check and its inventory. The four moments a freeze can actually be
    interrupted at are driven through a real transport where that transport lives, because a
    boolean set by a test says nothing about what a gateway holding an unfinished call would
    attest.
    """
    assert [row.name for row in fields(SettledHarness)] == list(THE_RECORD_UNION)
    # The recovery member is an attestation about the transport's own record, so the inventory it
    # answers for is held against that union rather than against a chosen part of it. A state this
    # list does not name is a state no checkpoint accounts for, and it fails here.
    assert tuple(one.__name__ for one in get_args(_Recovery)) == THE_RECOVERY_UNION
    assert _Idle.__name__ == THE_RECOVERY_UNION[0]
    for owed in THE_RECORD_UNION:
        settled = {name: name != owed for name in THE_RECORD_UNION}
        with pytest.raises(ApplicationError) as raised:
            await retrieve_checkpoint(
                AParent(),
                AHarness(
                    tmp_path / f"owing-{owed}",
                    manifest=a_manifest(settled=SettledHarness(**settled)),
                ),
                ForkOperations.under(tmp_path / f"journal-{owed}"),
            )
        assert raised.value.type == "invalid_checkpoint"
        assert owed in str(raised.value)

    # The acknowledgement is decoded and not yet in a transcript, so the harness has no freeze to
    # commit and says which capability it cannot supply rather than committing something near it.
    with pytest.raises(UnsupportedCapability) as unsupported:
        await retrieve_checkpoint(
            AParent(),
            AHarness(tmp_path / "unwritten", unsupported="commit a checkpoint"),
            ForkOperations.under(tmp_path / "journal-unwritten"),
        )
    assert unsupported.value.capability == "commit a checkpoint"

    # The transcript is persisted and the snapshot is not, so the manifest names no component and
    # nothing in it asserts that anything restores that transcript.
    with pytest.raises(ApplicationError) as raised:
        await retrieve_checkpoint(
            AParent(),
            AHarness(tmp_path / "unpublished", manifest=a_manifest(components=[])),
            ForkOperations.under(tmp_path / "journal-unpublished"),
        )
    assert raised.value.type == "invalid_checkpoint"

    # And the freeze itself, which is the only one of the four that is kept.
    kept = ForkOperations.under(tmp_path / "journal-settled")
    retrieval = await retrieve_checkpoint(AParent(), AHarness(tmp_path / "settled"), kept)
    assert retrieval.transcript_reference == TRANSCRIPT
    assert retrieval.components == a_manifest().components
    assert kept.all_of(CheckpointRetrieved) == [retrieval]
