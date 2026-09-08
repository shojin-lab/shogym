"""The values that cross the durable stream's boundaries.

Three boundaries, one module. A stream is started with :class:`StreamStart`, whose queue,
capacity, public identifiers, and terminal tool are fixed for the life of the generation and
are the only things the workflow will ever treat as configuration. A caller reaches the
running stream through the Update arguments and results here. And the workflow reaches the
outside world through the Activity inputs and results here.

Every one of these is a plain frozen dataclass, because Temporal writes each of them into a
history that has to be replayed years later and reads it back by field name. The wire records
in :mod:`shogym.serve.protocol_v2.records` are not restated: an Update that carries a protocol
record carries that record. What this module adds around them is the operator-side material
the model never sees, which is why identifiers like the hidden execution ID and the seal ID
live here and not there.

A result the model can see travels as :class:`OfferedMessage`, which carries the canonical
bytes as text rather than a decoded record. The bytes are the authority: they are what the
digest covers, what the harness must insert verbatim, and what the presentation attests to.
Handing back a record instead would let a gateway re-serialize and present bytes nobody
committed to.
"""

from __future__ import annotations

import base64
import json
import lzma
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from shogym.serve.protocol_v2 import (
    AGENT_FILED,
    IMMEDIATE,
    PROTOCOL_VERSION,
    SCHEDULE_VERSION,
    Assignment,
    BlobRef,
    PresentationAck,
    ReleasePlan,
    TerminalMetadata,
    assignment_id_for,
    canonical_json,
    length_prefixed,
)
from temporalio.api.common.v1 import Payload

from shogym.serve.protocol_v2.artifact import (
    ELIGIBLE_CELLS,
    ReceiptContract,
    SourceArtifactManifest,
    check_source_artifact,
    contract_fields,
    mask_spans,
    source_commitment,
)
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.policy import (
    ARTIFACT,
    LEGACY,
    POLICIES,
    SINGLETON_SLOT,
    GradeIdentity,
    MatchedFamily,
    PayloadDisposition,
    PolicyProvenance,
    PublicGrade,
    disposition_key,
    number_text,
)


# Why an attempt ended without a filing. The set is closed and the reasons are declared before a
# generation runs, so what ends an attempt is a property of the configuration rather than
# something a caller composes at the moment it gives up on one.
STEP_CAP = "step_cap"
DEADLINE = "deadline"
ABANDONED = "abandoned"
# The two reasons no caller asks for: they are written where the batch behind an accepted
# terminal cannot go on, and a controller that named either would be reporting something only
# the seal can know. They are told apart because the two say different things about the run. The
# first is the seal's own work failing for good, which is a step that did not happen. The second
# is a step that happened and came back with a result the seal cannot vouch for, which is what a
# reader looks at when the Activities all report success and the attempts still end.
SEAL_FAILED = "seal_failed"
SEAL_UNUSABLE = "seal_unusable"
# The third of those, and a subtype of the second rather than a new ending: the batch came back
# with a candidate built under something other than the policy this obligation was resolved to.
# It ends the attempt the way any unusable result does and is named apart, because a reader
# looking at a run that served the wrong bodies needs to see that rather than a generic result
# the seal could not vouch for.
SEAL_RENDERER = "seal_renderer_mismatch"
FINAL_FAILURE_REASONS = (STEP_CAP, DEADLINE, ABANDONED)
FINAL_FAILURES = FINAL_FAILURE_REASONS + (SEAL_FAILED, SEAL_UNUSABLE, SEAL_RENDERER)


@dataclass(frozen=True)
class TaskItem:
    """One entry of the closed queue, with every public identifier it will ever use.

    The identifiers are preallocated rather than minted on the way out, so two generations
    built from the same manifest put the same IDs in the same positions and a transcript
    comparison between them is a comparison of content.
    """

    task_position: int
    attempt_id: str
    task_message_id: str
    ack_message_id: str
    payload_position: int
    payload_message_id: str
    body: str


@dataclass(frozen=True)
class TerminalTool:
    """The one tool that can end an attempt, and the argument names it declares.

    ``argument_names`` is the whole of the schema this kernel checks. A real environment's
    schema arrives with the environment.
    """

    public_tool_name: str
    native_terminal_name: str
    argument_names: List[str]


# The versions of the carried projection below. A generation continued under a carrier this code
# does not know is refused rather than served from a half-understood record, so a number moves
# whenever a field changes meaning.
#
# The three numbers name three dispositions rather than three stages of one. A generation that
# declares a receipt contract carries receipt evidence from its first boundary, whether or not it
# has captured anything yet, and writes the middle number for the whole of its life. A generation
# that declares a fork writes the highest one from its first boundary, before any fork exists and
# whether or not it has captured anything, and so does a child that holds a fork origin. A
# generation that declares none of either route's configuration and holds none of either's
# evidence writes the earliest one and stays there.
#
# Choosing the number from the state is not on its own the compatibility promise, because the
# number is a label and the bytes are the record: the codec serializes the dataclass through the
# converter and then shares and compresses the result, so a field defaulting to absent is an
# explicit member of that JSON and one attempt with it packs to a different string from the same
# attempt without it while both carriers report the earlier number. So each lower number is an
# exact serialization adapter as well: before packing it drops every member added since, on the
# carried attempt, on the carried obligation, on any nested candidate and on the start the
# continuation carries, emitting the member set that version already had. Each is applied to
# genuinely earlier state alone.
LEGACY_CARRIER_SCHEMA_VERSION = 6
RECEIPT_CARRIER_SCHEMA_VERSION = 7
FORK_CARRIER_SCHEMA_VERSION = 8
#: Every carrier this code reads back. A generation writes one of these three and never a fourth.
CARRIER_SCHEMA_VERSIONS = (
    LEGACY_CARRIER_SCHEMA_VERSION,
    RECEIPT_CARRIER_SCHEMA_VERSION,
    FORK_CARRIER_SCHEMA_VERSION,
)

# The members each route added, named once so that the adapters below are arithmetic on the names
# rather than a second copy of the records. The journal is deliberately absent from the lists: its
# rows carry offered messages and acknowledgement identifiers, and none of those grew a member
# here, so there is nothing in one to drop.
_RECEIPT_START_MEMBERS = ("receipt_contracts", "receipt_source")
_RECEIPT_ATTEMPT_MEMBERS = (
    "source_artifact",
    "source_commitment",
    "source_origin",
    "selected_cell",
    "selected_body_reference",
    "selected_policy_digest",
    "receipt_contract_id",
    "presentation_references",
)
_RECEIPT_CANDIDATE_MEMBERS = (
    "source_commitment",
    "body_reference",
    "resolver_id",
    "resolver_version",
)
_RECEIPT_PROJECTION_MEMBERS = ("operation_failures",)
# And the fork's own. The two slots and the origin are the start's, and the preparation gate is
# the one member a child's carried obligation holds that no other generation writes.
_FORK_START_MEMBERS = ("served_slot", "forkable_slots", "fork_origin")
_FORK_OBLIGATION_MEMBERS = ("pending_preparation",)


#: The shape of a fork origin this build writes and admits.
FORK_ORIGIN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StartDifference:
    """One classified start field whose value differs between a parent's start and a child's.

    The digest is of the child's value, so the projection says what the child holds rather than
    what the parent held, and a reader comparing a child against the record the parent kept for
    it has one value per named field to compare.
    """

    field_name: str
    value_digest: str


@dataclass(frozen=True)
class ForkOrigin:
    """Where a child generation was cut from, as immutable lineage rather than as authority.

    It rides on the child's start beside the carry and outside the configuration hash, and it is
    what makes a carried projection legal in a child at all. What it authorizes is narrow and
    stated where it is read: the parent identity an inherited source was sealed under, and the
    parent hash the carrier a fresh child was handed was composed against. It authorizes nothing
    about a carry the child itself hands on afterwards, which the service's own continuation fact
    authorizes exactly as it does for any other generation.

    The parent's and the child's configuration hashes are both here because they answer different
    questions: the first is what the inherited carrier was written under, the second is what this
    child is. The source seal's own identity is here whole, so a reader of a child can say what
    work it inherited without reading the parent's history.
    """

    parent_workflow_id: str
    parent_run_id: str
    parent_configuration_hash: str
    child_configuration_hash: str
    parent_execution_ordinal: int
    parent_hidden_execution_id: str
    acknowledged_cursor: str
    projection_digest: str
    attestation_id: str
    acknowledged_visible_sha256: str
    checkpoint_manifest_reference: str
    source_attempt_id: str
    source_seal_id: str
    source_submission_digest: str
    source_canonicalization_version: str
    source_score: Optional[float]
    source_seal_ordinal: int
    source_graded_evidence: str
    source_commitment: str
    source_artifact_references: List[str]
    branch_slot: str
    dispositions_digest: str
    start_differences: List[StartDifference]
    parent_turnovers: int
    fork_id: str
    child_ordinal: int
    children: int
    schema_version: int = FORK_ORIGIN_SCHEMA_VERSION


@dataclass(frozen=True)
class SourceOriginContext:
    """The identity one source's seal was computed under, carried beside the source.

    It is the source's validation context rather than the destination's. A seal id is minted from
    a hidden execution id, an execution ordinal and an attempt, and the descriptor carries the
    ordinal alone, so a reader checking an inherited source has to be told which hidden execution
    produced it rather than assuming the start the source arrived at. For an ordinary continuation
    the two are the same value, because continue-as-new hands the same start on; keeping the field
    is what stops that coincidence from becoming the rule.
    """

    hidden_execution_id: str
    execution_ordinal: int


def origin_fields(origin: SourceOriginContext) -> Dict[str, Any]:
    """Return one origin as the two values a carrier writes it as."""
    return {
        "hidden_execution_id": origin.hidden_execution_id,
        "execution_ordinal": origin.execution_ordinal,
    }


def read_source_origin(value: Any) -> SourceOriginContext:
    """Read one carried origin back, and refuse a shape that is not one.

    It is read rather than decoded for the reason the descriptor beside it is. The origin is what
    a carried seal id is recomputed from, so a member this build does not declare being dropped
    on the way in, or an ordinal written as a float being rounded to one, would leave restore
    comparing a value the decoder repaired against a value the start holds and finding them
    equal.
    """
    if not isinstance(value, Mapping):
        raise WireFormatError(
            f"a source origin is written as an object, and this one is {value!r}"
        )
    declared = ("hidden_execution_id", "execution_ordinal")
    if tuple(sorted(str(key) for key in value)) != tuple(sorted(declared)):
        raise WireFormatError(
            f"a source origin carries exactly {sorted(declared)}, and this one carries "
            f"{sorted(str(key) for key in value)}"
        )
    hidden = value["hidden_execution_id"]
    ordinal = value["execution_ordinal"]
    if not isinstance(hidden, str) or not hidden:
        raise WireFormatError(
            f"a source origin names a hidden execution, and this one names {hidden!r}"
        )
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
        raise WireFormatError(
            f"a source origin's execution ordinal is a whole number, and this one is {ordinal!r}"
        )
    return SourceOriginContext(hidden_execution_id=hidden, execution_ordinal=ordinal)


# Where an operation was when it failed over evidence. Five phases, closed, because a reader
# telling a publication that never committed from a delivery that lost its bytes afterwards is
# reading two different runs. The last of them belongs to the fork operation, which is named here
# so its rows have a home rather than because anything in this build writes one.
SOURCE_PUBLICATION = "source_publication"
OWNERSHIP_CLAIM = "ownership_claim"
CONTINUED_FIRST_DELIVERY = "continued_first_delivery"
PAYLOAD_OFFER = "payload_offer"
FORK_PREPARATION = "fork_preparation"
OPERATION_PHASES = (
    SOURCE_PUBLICATION,
    OWNERSHIP_CLAIM,
    CONTINUED_FIRST_DELIVERY,
    PAYLOAD_OFFER,
    FORK_PREPARATION,
)

# What was wrong with the evidence. The store answers the first: a name it cannot produce the
# exact bytes for. The other three are answered where the shape is known, by the capture that
# holds the committed record and by the resolver that holds the contract.
UNAVAILABLE_EVIDENCE = "unavailable_evidence"
CORRUPT_EVIDENCE = "corrupt_evidence"
CONTRACT_DRIFT = "contract_drift"
WRONG_SOURCE = "wrong_source"
OPERATION_REASONS = (UNAVAILABLE_EVIDENCE, CORRUPT_EVIDENCE, CONTRACT_DRIFT, WRONG_SOURCE)

# And where it stands now. A refusal is a row of its own and stays one: a later recovery appends
# rather than overwriting, so the sequence keeps both. Unrecoverable is the ending, written where
# the operation that failed can never be asked again.
REFUSED_OPERATION = "refused"
RECOVERED_OPERATION = "recovered"
UNRECOVERABLE_OPERATION = "unrecoverable"
OPERATION_OUTCOMES = (REFUSED_OPERATION, RECOVERED_OPERATION, UNRECOVERABLE_OPERATION)


@dataclass(frozen=True)
class OperationFailure:
    """One operation that could not produce the evidence it depended on, and what became of it.

    These are rows and not fields. They are kept in commit order for the life of the generation,
    successive refusals and recoveries of one logical operation each append, and nothing is
    dropped at a boundary: the latest word about an attempt is the last row naming it. The
    private Update journal does not do this job, being keyed by an Update identifier, exported
    nowhere and read by nobody.

    ``operation`` is the identity of the operation rather than of the transport call that carried
    it: a pull's canonical request identity, an ownership claim's claim operation identity, a fork
    preparation's preparation operation identity. That is what joins a refusal to the recovery of
    the same logical operation across a fresh fencing token and a different Update.

    ``generation`` is the generation whose operation failed, so a row a forked child inherits
    reads as the parent's operation rather than as the child's.

    The fields a pull has and a claim has not are written absent rather than invented: a claim
    carries no logical request id and names no payload position, and the epoch it was refused
    under is the epoch it witnessed.
    """

    operation: str
    phase: str
    reason: str
    outcome: str
    generation: str
    refused_epoch: int
    attempt_id: Optional[str] = None
    payload_position: Optional[int] = None
    references: List[str] = field(default_factory=list)
    request_id: Optional[str] = None
    recovered_epoch: Optional[int] = None
    protocol_version: int = PROTOCOL_VERSION


def ownership_claim_operation_identity(claimant_id: str, witnessed_epoch: int) -> str:
    """Return the identity of one ownership claim, as an operation rather than as a call.

    The fencing token, its hash and the transport's Update identifier are all outside it, so a
    claim retried under a fresh token is the same operation as the claim it repeats and appends to
    that episode rather than opening another. The witnessed epoch is inside it and the won one is
    not, so every attempt at one swap shares an identity and the next swap's attempts do not.
    """
    return sha256(
        length_prefixed(b"ownership-claim-operation-v2")
        + length_prefixed(claimant_id.encode("utf-8"))
        + length_prefixed(str(witnessed_epoch).encode("ascii"))
    ).hexdigest()


def fork_preparation_operation_identity(
    fork_id: str, child_workflow_id: str, creation_epoch: int
) -> str:
    """Return the identity of one fork preparation, on the claim identity's own terms.

    The epoch inside it is the one the child's first preparation was created under, captured once
    in the child's own record and never recomputed from the epoch the child currently holds: a
    repair that steps the epoch repeats one logical preparation rather than opening a second
    operation nothing joins to the first, and the current epoch changes only the derived Update
    identifier the retry is submitted under.
    """
    return sha256(
        length_prefixed(b"fork-preparation-operation-v1")
        + length_prefixed(fork_id.encode("utf-8"))
        + length_prefixed(child_workflow_id.encode("utf-8"))
        + length_prefixed(str(creation_epoch).encode("ascii"))
    ).hexdigest()


@dataclass(frozen=True)
class CarriedAttempt:
    """One attempt's mutable state, as the next execution has to find it.

    The task itself is not here. It comes from the roster the start already carries, so what
    crosses is what the generation did to the attempt rather than what the attempt was.

    Three fields the seal writes are missing on purpose: the canonical submission text, the
    environment's recovery token, and the finalizer key. They are written when a seal is
    prepared and never read again, and a prepared seal is one of the things a boundary refuses
    to cross, so nothing on the far side could ask for them.

    ``deadline_expired`` is missing for the other reason: an expiry that can be applied is
    applied before the boundary is considered, so at a legal boundary there is none to carry.

    The receipt fields are absent by default and cross together where a source was committed. The
    descriptor is carried whole rather than by digest, because a later derivation has to read its
    cell map and its contract with no store and no bank; ``source_commitment`` names its canonical
    bytes; ``source_origin`` is the identity that seal was computed under; and the four beside
    them are the selection, which lives here rather than on the obligation because an obligation
    drops its candidate once one has been presented and a reference hung on the candidate alone
    would be lost with it. No body crosses: what crosses is the reference.

    The descriptor and the origin cross as whatever they were written as, for the reason the seal
    result carries its own the same way. A field with a type is a field the decoder makes that
    type of before any code of this generation runs: a carried descriptor with a member this build
    does not declare is silently dropped there and one whose size was written as a float is
    coerced back to a whole number, and either way what restore then checks is a record the
    decoder repaired rather than the record that crossed. So the shape is permissive here and the
    authority is strict: :func:`shogym.serve.protocol_v2.artifact.read_source_artifact` and
    :func:`read_source_origin` decide what these are, at the restore that refuses the carrier.

    ``presentation_references`` is the objects this attempt's own committed presentations cited,
    in commit order. They are among what an ownership claim reads the store for, so a claim
    refused over one of them is refused over an object this attempt requires, and a row that named
    no attempt would leave the receipt whose evidence went missing reading as though nothing had.
    The association is the attempt's rather than the inventory's for that reason: the flat
    inventory says the object is required by somebody and cannot say by whom. A generation that
    declares no receipt contract records none of them.
    """

    attempt_id: str
    state: str
    task_start_checkpoint: Optional[str] = None
    environment_calls: int = 0
    terminal_request_id: Optional[str] = None
    terminal_identity: Optional[str] = None
    terminal_tool: Optional[str] = None
    terminal_source: Optional[str] = None
    seal_id: Optional[str] = None
    submission_digest: Optional[str] = None
    score: Optional[float] = None
    decode_state: Optional[str] = None
    graded_evidence: Optional[str] = None
    seal_ordinal: Optional[int] = None
    final_failure: Optional[str] = None
    deadline_at: Optional[int] = None
    failure_activity: Optional[str] = None
    failure_activity_id: Optional[str] = None
    failure_kind: Optional[str] = None
    failure_message: Optional[str] = None
    failure_retry_state: Optional[str] = None
    source_artifact: Any = None
    source_commitment: Optional[str] = None
    source_origin: Any = None
    selected_cell: Optional[str] = None
    selected_body_reference: Optional[str] = None
    selected_policy_digest: Optional[str] = None
    receipt_contract_id: Optional[str] = None
    presentation_references: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class CarriedObligation:
    """One obligation's state, and its candidate while one could still be offered.

    Whether the candidate was ever built is kept apart from the candidate, because the count of
    materializations is inside the projection hash and the body is not. An obligation that has
    been presented, or that ended without being rendered, keeps the fact and drops the bytes.

    ``pending_preparation`` is the one state in which an offerable obligation lawfully carries no
    candidate: a fork child's inherited obligation, selected against the parent's committed source
    and unbuilt until the child builds it. The parent writes it into the child's start and the
    child's own preparation clears it in the same transition that installs the candidate, so an
    obligation claiming it is told from a damaged carry by a value the parent committed to rather
    than by the absence of bytes. No other generation writes one.
    """

    attempt_id: str
    state: str
    materialized: bool = False
    candidate: Optional[PayloadCandidate] = None
    pending_preparation: bool = False


@dataclass(frozen=True)
class CarriedBinding:
    """One logical request, its canonical identity, and the message bound to it."""

    request_id: str
    identity: str
    message: OfferedMessage


@dataclass(frozen=True)
class CarriedFinalization:
    """One logical finalization, its identity, and the receipt it was answered with."""

    request_id: str
    identity: str
    receipt: AttemptFinalized


@dataclass(frozen=True)
class CarriedAttestation:
    """One attestation, its identity, and the acknowledgement it was answered with."""

    attestation_id: str
    identity: str
    ack: PresentationAck


# How a generation's carried projection is written into one string, and the version that says
# which way. Written out as it stands it runs to several megabytes on a long roster, and a
# continuation is one payload. So three things happen to it, in order, and none of them loses
# anything.
#
# First the structure is shared. The projection repeats itself enormously: a confirmation records
# a whole state reading and two hundred of them differ in a handful of entries, a world call
# records a lease whose attempt and cursor are two of a few hundred distinct strings, and one
# offered message reached under several identifiers is one value. Sharing writes each distinct
# value once and refers to it by number.
#
# Then it is written as canonical JSON, so two runs of one generation write the same bytes.
# Then it is packed and text-encoded, because a payload is JSON.
#
# The packer is LZMA over a window that covers the whole value, and the window is the reason. What
# is left after sharing is a few megabytes of rows that rhyme with rows written hundreds of
# thousands of bytes earlier: the same handler names, the same shapes, the same hexadecimal
# alphabet. Deflate looks back thirty two kilobytes and cannot see any of it; this looks back over
# all of it, and on the supported roster's largest carrier it writes half the bytes deflate does.
# The settings are named here rather than taken from a default, because they are part of what the
# encoding version means: raw LZMA2, the third preset, and a dictionary of eight mebibytes, which
# is twice the largest value the supported profile produces and keeps the window covering the
# whole of it as a roster grows. The third preset is chosen over the sixth for what it costs: the
# same margin to within half a percent, a tenth of the time, and a bounded encoder that a Worker
# crossing several boundaries at once can afford.
#
# What is shared is written once and read back as its own object. Two occurrences of one value do
# not become one object on the way back: a historical state reading and the live table it was a
# reading of must not be able to change each other.
CARRIER_ENCODING = "shared.canonical-json.lzma.base64.v1"
_CARRIER_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 3, "dict_size": 8 << 20}]


def _shared(value: Any) -> List[Any]:
    """Write one JSON value as a list of distinct nodes, each referring to the ones inside it.

    A node is a dictionary, a list, or a scalar. Scalars are keyed by their type together with
    their exact written form rather than by equality, because equality merges values a reader
    would have to be able to tell apart: a boolean equals an integer, and a negative zero equals a
    positive one, and neither pair may become one node.
    """
    nodes: List[Any] = []
    seen: Dict[Any, int] = {}

    def walk(node: Any) -> int:
        if isinstance(node, dict):
            shape: Any = ("d", tuple((walk(key), walk(item)) for key, item in sorted(node.items())))
        elif isinstance(node, list):
            shape = ("l", tuple(walk(item) for item in node))
        else:
            shape = ("s", type(node).__name__, repr(node))
        if shape not in seen:
            seen[shape] = len(nodes)
            nodes.append(
                [shape[0], list(shape[1])] if shape[0] != "s" else ["s", type(node).__name__, node]
            )
        return seen[shape]

    root = walk(value)
    return [root, nodes]


def _unshared(written: List[Any]) -> Any:
    """Read back what :func:`_shared` wrote, giving every occurrence its own containers.

    Sharing is how the bytes are written, not how the value is held. A dictionary that appears in
    two places is built twice here, so that a historical reading and a live table restored from
    one carrier cannot change one another by being the same object. Scalars are immutable and are
    handed back as they are.
    """
    root, nodes = written

    def build(index: int) -> Any:
        tag = nodes[index][0]
        if tag == "s":
            return nodes[index][2]
        if tag == "d":
            return {build(key): build(item) for key, item in nodes[index][1]}
        return [build(item) for item in nodes[index][1]]

    return build(root)


def resolved_echo(candidate: "PayloadCandidate") -> bool:
    """Say whether a candidate carries any of the values a resolved one echoes about its source.

    It is the same list the carrier version is chosen from, read once rather than written out
    twice. A route that resolves nothing asks these to be empty and the version rule reads a
    filled one as receipt evidence, so the two questions are one question: a value that would
    promote a legacy generation's carrier is a value the route that produced it has to refuse.
    """
    return any(getattr(candidate, name) for name in _RECEIPT_CANDIDATE_MEMBERS)


def fork_configuration(start: "StreamStart") -> bool:
    """Say whether one start declares a fork, which is the first thing the version rule reads.

    A served slot other than the one every generation serves, or a nonempty set of slots this
    generation's fork may create. The default is excluded on purpose: the served slot is a new
    field whose default is the slot every row carries today, and reading a defaulted singleton as
    a declaration of fork capability would promote every legacy and receipt-only start to the
    highest carrier and strand it on a codec that had never read it.
    """
    return start.served_slot != SINGLETON_SLOT or bool(start.forkable_slots)


def carried_fork_evidence(projection: "CarriedProjection") -> bool:
    """Say whether one projection holds a member only a fork puts there.

    The preparation gate is that member: no ordinary generation and no receipt one writes an
    obligation marked pending preparation, and a projection holding one may never be written as a
    record that had no room for it.
    """
    return any(owed.pending_preparation for owed in projection.obligations)


def carrier_version(start: "StreamStart", projection: "CarriedProjection") -> int:
    """Return the version one generation writes its carrier under.

    Configuration decides it first, and the fork's configuration before the receipt route's: a
    generation that declares a fork, or a child that holds a fork origin, writes the highest
    number from its first boundary, before any fork exists; a generation that declares a receipt
    contract writes the middle one from its first boundary, before it has captured anything.
    Evidence decides it otherwise, so a projection holding a preparation gate, a descriptor, a
    selection or a resolved candidate can never be written as a record that had no room for one.

    The number is selected from what a start declares rather than from the evidence it happens to
    hold, because an adapter run over a fork-capable start would strip the slots from the one copy
    the next execution rebuilds its configuration identity from, and the parent would refuse its
    own lawful continuation.
    """
    if fork_configuration(start) or start.fork_origin is not None:
        return FORK_CARRIER_SCHEMA_VERSION
    if carried_fork_evidence(projection):
        return FORK_CARRIER_SCHEMA_VERSION
    if start.receipt_contracts or start.receipt_source:
        return RECEIPT_CARRIER_SCHEMA_VERSION
    for row in projection.attempts:
        if any(getattr(row, name) for name in _RECEIPT_ATTEMPT_MEMBERS):
            return RECEIPT_CARRIER_SCHEMA_VERSION
    for owed in projection.obligations:
        candidate = owed.candidate
        if candidate is not None and resolved_echo(candidate):
            return RECEIPT_CARRIER_SCHEMA_VERSION
    if projection.operation_failures:
        return RECEIPT_CARRIER_SCHEMA_VERSION
    return LEGACY_CARRIER_SCHEMA_VERSION


def _without(written: Dict[str, Any], names: Sequence[str]) -> Dict[str, Any]:
    """Return one written document without the members named, and otherwise exactly as it was."""
    return {name: value for name, value in written.items() if name not in names}


def legacy_start_members(written: Dict[str, Any]) -> Dict[str, Any]:
    """Return one encoded start as the legacy carrier version had it, members and all."""
    return _without(written, (*_RECEIPT_START_MEMBERS, *_FORK_START_MEMBERS))


def receipt_start_members(written: Dict[str, Any]) -> Dict[str, Any]:
    """Return one encoded start as the receipt carrier version had it.

    It omits the fork's members and nothing else, so a receipt generation's continuation carries
    the contract and the bank source it is admitted over and none of the names the fork added.
    """
    return _without(written, _FORK_START_MEMBERS)


def legacy_projection_members(written: Dict[str, Any]) -> Dict[str, Any]:
    """Return one encoded projection as the legacy carrier version had it.

    Every other member is passed through exactly as the converter wrote it, so what comes out is
    the document that version produced rather than a document this one rebuilt to look like it.
    """
    adapted = _without(written, _RECEIPT_PROJECTION_MEMBERS)
    adapted["attempts"] = [
        _without(row, _RECEIPT_ATTEMPT_MEMBERS) for row in written.get("attempts", [])
    ]
    adapted["obligations"] = [
        _without(row, _FORK_OBLIGATION_MEMBERS)
        if row.get("candidate") is None
        else {
            **_without(row, _FORK_OBLIGATION_MEMBERS),
            "candidate": _without(row["candidate"], _RECEIPT_CANDIDATE_MEMBERS),
        }
        for row in written.get("obligations", [])
    ]
    return adapted


def receipt_projection_members(written: Dict[str, Any]) -> Dict[str, Any]:
    """Return one encoded projection as the receipt carrier version had it.

    One member goes, the preparation gate on the carried obligation, and every other name and
    value crosses as the converter wrote it. A receipt generation writes no gate, so what this
    removes is the field's default rather than a state anything held.
    """
    adapted = dict(written)
    adapted["obligations"] = [
        _without(row, _FORK_OBLIGATION_MEMBERS) for row in written.get("obligations", [])
    ]
    return adapted


def continuation_argument(start: "StreamStart", converter: Any, version: int) -> Any:
    """Return what a continuation is handed, written as the version it declares.

    A generation on the current version hands its own start over and the converter encodes it. An
    earlier one hands over what that same encoding says minus the members added since, so a reader
    of that version finds the document it has always found rather than one carrying names it never
    had. The value is a plain mapping there, and the execution that receives it decodes it back
    into a start with those members at their defaults, which is what absent means.
    """
    if version == FORK_CARRIER_SCHEMA_VERSION:
        return start
    written = json.loads(converter.to_payloads([start])[0].data.decode("utf-8"))
    if version == RECEIPT_CARRIER_SCHEMA_VERSION:
        return receipt_start_members(written)
    return legacy_start_members(written)


def pack_carrier(projection: Any, converter: Any, *, version: int) -> "StreamCarry":
    """Write one projection as the string a continuation carries, under one declared version.

    The version is the caller's rather than a constant read here, because what decides it is the
    generation's configuration and evidence and neither is visible from a projection alone.
    """
    value = json.loads(converter.to_payloads([projection])[0].data.decode("utf-8"))
    if version == LEGACY_CARRIER_SCHEMA_VERSION:
        value = legacy_projection_members(value)
    elif version == RECEIPT_CARRIER_SCHEMA_VERSION:
        value = receipt_projection_members(value)
    raw = json.dumps(
        _shared(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return StreamCarry(
        carrier_schema_version=version,
        encoding=CARRIER_ENCODING,
        data=base64.b64encode(
            lzma.compress(raw, format=lzma.FORMAT_RAW, filters=_CARRIER_FILTERS)
        ).decode("ascii"),
    )


def unpack_carrier(carry: "StreamCarry", converter: Any) -> "CarriedProjection":
    """Read back what :func:`pack_carrier` wrote, or refuse a writing this code cannot read."""
    if carry.encoding != CARRIER_ENCODING:
        raise WireFormatError(
            f"this generation's projection is written as {carry.encoding!r} and this code reads "
            f"{CARRIER_ENCODING!r}"
        )
    written = json.loads(
        lzma.decompress(
            base64.b64decode(carry.data.encode("ascii")),
            format=lzma.FORMAT_RAW,
            filters=_CARRIER_FILTERS,
        )
    )
    payload = Payload()
    payload.metadata["encoding"] = b"json/plain"
    payload.data = json.dumps(
        _unshared(written), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return converter.from_payload(payload, CarriedProjection)


@dataclass(frozen=True)
class AnsweredUpdate:
    """What one exact Update was answered with, read back through a Query.

    A transport that cannot reach a generation's handler still has to be able to find out what an
    Update it already sent was answered with. That happens where the generation has decided to
    continue as new and is holding traffic back, and where the generation has finished and its
    last execution accepts nothing further. Neither is a place to send the request again, and
    both are places where the answer already exists.

    So this is the answer as a value a caller can decode, with the row it came from already
    resolved. ``found`` is false for an identifier this generation never answered, which is the
    ordinary case for a request that never arrived and is not an error.
    """

    found: bool
    kind: str = ""
    handler: str = ""
    code: str = ""
    type_name: str = ""
    message: str = ""
    value: Optional[Dict[str, Any]] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class StreamCarry:
    """The projection a continuation carries, written as one string.

    What is inside it is :class:`CarriedProjection`. What is here is how it is written, so that
    the writing can change under a version without every reader of the projection changing with
    it, and so that a writing this code cannot read is refused rather than half understood.
    """

    carrier_schema_version: int
    encoding: str
    data: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class CarriedProjection:
    """The whole logical projection of a generation, as one execution hands it to the next.

    The durable service bounds one execution, and a roster longer than that bound needs more
    than one. This is what makes the second execution the same generation as the first: every
    fact a caller, a harness or a reader can observe is either in here or derived from the start
    that rides beside it. Nothing here is configuration, and nothing here is hashed into the
    generation's identity.

    Three invariants hold it together. It is versioned, and the version is the holder's rather
    than a second one here: one version says both how the string is written and what shape is
    inside it, so there is no way to read a projection whose version was never checked. It
    repeats the static configuration identity, so a carrier composed against another generation
    is refused before it is believed. And every unordered set and map inside it is written in
    canonical sorted form, because a set serialized in hash order would make the same boundary
    produce different bytes on a replay. The lists that mean an order, the presentations above
    all, keep the order they mean.

    What is not here is what a legal boundary forbids: there is no pending message, no held
    grant, no operation ticket, no prepared seal and no applicable expiry to carry, and the
    generation is open, not draining, and has not presented Done.
    """

    configuration_hash: str
    ownership_epoch: int
    fencing_token_hash: Optional[str]
    ownership_claims: int
    consumer_id: Optional[str]
    claim_epoch: int
    cursor: str
    queue_closed: bool
    hidden_ordinal: int
    seal_ordinal: int
    wait_count: int
    wait_reasons: Dict[str, int]
    offer_count: int
    eligibility_count: int
    handed_out_attempt_ids: List[str]
    activity_ordinal: int
    verification_batches: int
    attempts: List[CarriedAttempt]
    obligations: List[CarriedObligation]
    presented: List[PresentedMessage]
    committed_blobs: List[str]
    pull_requests: List[CarriedBinding]
    info_requests: List[CarriedBinding]
    terminal_requests: List[CarriedBinding]
    finalize_requests: List[CarriedFinalization]
    attestations: List[CarriedAttestation]
    # Every accepted Update this generation has completed, keyed by the exact identifier it was
    # answered under. It is packed rather than written out, because the same information as
    # literal rows runs past the size one continuation may be.
    # Every accepted Update this generation has completed, by the exact identifier it was
    # answered under, as rows rather than as a packed string: the whole projection is packed
    # together, so packing this again inside it would only hide it from the sharing.
    journal: List[List[Any]] = field(default_factory=list)
    turnovers: int = 0
    # Every operation of this generation that could not produce the evidence it depended on, in
    # commit order. They cross because the record of a refusal is worth nothing if a boundary can
    # drop it: a delivery refused in one execution is recovered in another, and both rows have to
    # be readable afterwards from one list.
    operation_failures: List[OperationFailure] = field(default_factory=list)
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class StreamStart:
    """Everything a generation is, fixed before it serves anything.

    Nothing here can be changed by an Update. ``id_key_hex`` and ``hidden_execution_id`` are
    operator material: the first keys the message IDs that are not preallocated, the second
    separates two executions that share a public attempt ID, and neither is ever copied into a
    model-visible message.

    ``assignments`` is the generation's roster and ``release`` is its schedule. They arrive
    together and before anything is served, which is what makes assignment a fact about the
    generation rather than a consequence of how it ran. An empty roster is filled in from the
    closed manifest at start, still before an offer. ``evaluation_only`` says this generation
    exists to score and not to deliver, and it is refused unless its plan is Never.

    ``configuration_hash`` is the environment's half of what this generation is: the contract,
    the instructions, the tools. The whole of it is :func:`configuration_hash`, which folds that
    value together with the manifest, the roster, the plan, the capacity, and the versions, and
    it is what a resume has to present. ``blob_root`` is the directory the blobs an event may
    reference are installed in. A generation without one verifies no reference, and says so.

    ``attempt_deadline_ms`` is how long an attempt may stay active before the generation ends it
    itself. Zero is off, which is the default: a deadline is a property of the run rather than of
    this kernel, so a generation that declares none has none and waits as long as it is asked to.

    ``profile`` and ``dispositions`` are what this generation delivers and under what. The
    profile is the run's own class rather than the name of whatever function composed it, and
    the dispositions are one resolved row per obligation per branch. Neither has a serving
    default: a generation created now says which profile it is and carries a row for every
    position, and one that says nothing is a history recorded before a policy was a fact about a
    generation, read as the legacy placeholder and never as honest.

    ``provenance`` is what entitles the generation to the profile it claims: the experiment that
    registered its rows, or the platform default it was stamped from, with the digest of the
    exact rows that authority answered for. A profile with no authority behind it is a word a
    caller wrote, so a generation created now carries both or does not start.

    ``families`` are the matched arms this generation's rows are cells of, each declaring the
    group its candidates are built in and the byte count they come to, so a concealed cell and
    an informative one cannot be told apart by their shape.

    ``receipt_contracts`` are the shapes this generation admits an environment's own bodies
    under, declared beside those families and named by a row through the same column. A
    generation may declare several: which contract a capture is validated against is the row's to
    say, and inferring the only one stops working at the second declaration.
    ``receipt_source`` is the bank source digest they are admitted over, which is what a
    descriptor's bundle digest is compared against; the environment's configuration digest hashes
    that source together with four other values, so the comparison cannot be recovered from it.
    Both are absent on every generation that declares no contract, and both are inside the
    configuration hash where one is declared.

    ``grade`` is what the environment said its grader is. A generation may resolve an obligation
    to a policy that publishes the score only where that grader is the environment's own, so the
    claim is carried here and checked at start rather than being a property of whichever builder
    composed the generation.

    ``budget`` is how many environment actions this generation gives an attempt, where it has
    declared a number to hand over. It is one number for the generation, so it is here rather
    than on a task, and every task offered under it carries it. A generation that declares none
    serves a task record without the key, which is what a generation composed before there was a
    budget to declare serves.

    ``info`` is whether this generation answers the tool that says how much of its queue is left.
    It is off by default, because telling an agent how much work there is is a decision about the
    run, and a generation that makes no such decision has no such tool: nothing about it is
    served, nothing about it is hashed, and the counts stay where they have always been, which is
    with the harness.

    ``served_slot`` is the branch this generation serves its dispositions on, and
    ``forkable_slots`` are the slots its fork may create children for. Both are inside the
    configuration hash where they are declared, so a child plan naming a slot the parent never
    declared is refused against a value the parent committed to before it forked. The default is
    the one slot every generation serves today, and it is hashed nowhere, so a generation that
    declares no fork hashes exactly what it hashed before there was a slot to declare.

    ``carry`` is the one field a running generation puts here rather than a caller. It is how one
    execution hands the whole logical projection to the next, and it is legal only there: a fresh
    start carrying one is refused, and a continued execution given none is refused too. It is
    outside :func:`configuration_hash` by construction, because what the generation is has not
    changed and every resume is held to that value.

    ``fork_origin`` is beside it and outside that hash for the same reason: it is the lineage a
    child was cut on rather than a thing the generation is, and a child's own configuration
    identity is inside the origin rather than derived from it.
    """

    configuration_hash: str
    consumer_claim_hash: str
    initial_cursor: str
    done_message_id: str
    id_key_hex: str
    hidden_execution_id: str
    canonicalization_version: str
    terminal_tool: TerminalTool
    tasks: List[TaskItem]
    capacity: int = 1
    wait_retry_after_ms: int = 1000
    attempt_deadline_ms: int = 0
    execution_ordinal: int = 0
    release: ReleasePlan = field(default_factory=lambda: IMMEDIATE)
    assignments: List[Assignment] = field(default_factory=list)
    evaluation_only: bool = False
    blob_root: Optional[str] = None
    profile: str = LEGACY
    grade: Optional[GradeIdentity] = None
    dispositions: List[PayloadDisposition] = field(default_factory=list)
    provenance: Optional[PolicyProvenance] = None
    families: List[MatchedFamily] = field(default_factory=list)
    receipt_contracts: List[ReceiptContract] = field(default_factory=list)
    receipt_source: str = ""
    served_slot: str = SINGLETON_SLOT
    forkable_slots: List[str] = field(default_factory=list)
    budget: Optional[int] = None
    info: bool = False
    schedule_version: str = SCHEDULE_VERSION
    protocol_version: int = PROTOCOL_VERSION
    carry: Optional[StreamCarry] = None
    fork_origin: Optional[ForkOrigin] = None


def configuration_hash(start: StreamStart) -> str:
    """Return the immutable hash of everything this generation is.

    A resume presents this value and is refused when it does not match, so what goes in is
    everything a changed value would make the running generation a different one: the two
    versions, the environment's own configuration digest, the capacity, the closed manifest,
    the roster the generation will serve, the release plan, and the terminal tool. The roster is
    the derived one, because a generation started without rows gets the ones its manifest
    implies and a hash over the empty list would not cover them.

    Secrets are covered by their hashes rather than by value. What has to be true is that the
    key changed, not that a reader of this hash can tell what it changed to.

    The blob store's location is not here. Where a run keeps its bytes is deployment, and the
    same generation moved to another directory is the same generation.

    What a generation delivers is here, and it is folded in only where a generation declares it.
    A history recorded before a policy was a fact about a generation hashed exactly these keys,
    and adding one to what it presents would refuse every resume of it, so the legacy profile
    hashes what it always hashed and a generation that names a profile hashes its dispositions
    along with everything else. What a generation tells an agent it may spend is here on the same
    terms: a declared budget is a number on every task the generation serves, so it is part of
    what a resume has to serve identically, and a generation that declared none hashes exactly
    what it hashed before there was one to declare. A declared info tool is here on those terms
    too: it is a tool the agent may call and an answer the generation mints, so a replacement
    serving it where the original did not is serving something else. Each of those names its
    policy by the digest of the policy's
    preimage, so what a body was allowed to say is inside the identity a resume is held to,
    along with the authority that decided it and the matched families its rows are cells of.

    The grader is here whole, every field of it. Which grader, which version, whether its number
    is the environment's own, which measure the headline is, how fine that measure is, and what
    it may publish beside it are each a fact the record depends on, and a claimant composed over
    another of them is composing a different generation. The resolution is one of them for the
    reason the components' is: a run recorded under one headline precision and resumed by a
    process composed for another would take the generation over and disagree with it afterwards,
    at the first seal, rather than being refused before it owned anything.
    """
    roster = list(start.assignments) or assignments_for(start.tasks, start.release)
    plan = start.release
    declared: Dict[str, Any] = {
        "protocol_version": start.protocol_version,
        "schedule_version": start.schedule_version,
        "environment_configuration": start.configuration_hash,
        "consumer_claim_hash": start.consumer_claim_hash,
        "capacity": start.capacity,
        "wait_retry_after_ms": start.wait_retry_after_ms,
        "attempt_deadline_ms": start.attempt_deadline_ms,
        "evaluation_only": start.evaluation_only,
        "canonicalization_version": start.canonicalization_version,
        "execution_ordinal": start.execution_ordinal,
        "initial_cursor": start.initial_cursor,
        "done_message_id": start.done_message_id,
        "id_key_sha256": sha256(start.id_key_hex.encode("utf-8")).hexdigest(),
        "hidden_execution_sha256": sha256(
            start.hidden_execution_id.encode("utf-8")
        ).hexdigest(),
        "terminal_tool": {
            "public_tool_name": start.terminal_tool.public_tool_name,
            "native_terminal_name": start.terminal_tool.native_terminal_name,
            "argument_names": list(start.terminal_tool.argument_names),
        },
        "manifest": [
            {
                "task_position": item.task_position,
                "attempt_id": item.attempt_id,
                "task_message_id": item.task_message_id,
                "ack_message_id": item.ack_message_id,
                "payload_position": item.payload_position,
                "payload_message_id": item.payload_message_id,
                "body_sha256": sha256(item.body.encode("utf-8")).hexdigest(),
            }
            for item in start.tasks
        ],
        "roster": [
            {
                "assignment_id": row.assignment_id,
                "attempt_id": row.attempt_id,
                "task_position": row.task_position,
                "payload_position": row.payload_position,
                "task_message_id": row.task_message_id,
                "ack_message_id": row.ack_message_id,
                "payload_message_id": row.payload_message_id,
                "release_plan_id": row.release_plan_id,
                "creates_payload_obligation": row.creates_payload_obligation,
            }
            for row in roster
        ],
        "release": {
            "release_plan_id": plan.release_plan_id,
            "predicate": plan.predicate,
            "predicate_version": plan.predicate_version,
            "priority": plan.priority,
            "tie_key": plan.tie_key,
            "gates": [
                {
                    "attempt_id": gate.attempt_id,
                    "after_payload_position": gate.after_payload_position,
                    "after_sealed_attempt_id": gate.after_sealed_attempt_id,
                }
                for gate in sorted(plan.gates, key=lambda gate: gate.attempt_id)
            ],
        },
    }
    if start.budget is not None:
        declared["budget"] = start.budget
    if start.info:
        declared["info"] = True
    if start.profile != LEGACY:
        declared["profile"] = start.profile
        declared["grade"] = (
            None
            if start.grade is None
            else {
                "grader_id": start.grade.grader_id,
                "grader_version": start.grade.grader_version,
                "stand_in": start.grade.stand_in,
                "score_component": start.grade.score_component,
                "score_places": start.grade.score_places,
                "public_components": [
                    {
                        "name": number.name,
                        "minimum": number_text(number.minimum),
                        "maximum": number_text(number.maximum),
                        "places": number.places,
                    }
                    for number in start.grade.public_components
                ],
            }
        )
        declared["dispositions"] = [
            {
                "attempt_id": row.attempt_id,
                "payload_position": row.payload_position,
                "branch_slot": row.branch_slot,
                "kind": row.kind,
                "policy_digest": row.policy_digest,
                "cell": row.cell,
                "reason": row.reason,
                "resolution_source": row.resolution_source,
                "family_id": row.family_id,
            }
            for row in sorted(start.dispositions, key=disposition_key)
        ]
        declared["provenance"] = (
            None
            if start.provenance is None
            else {
                "authority": start.provenance.authority,
                "roster_digest": start.provenance.roster_digest,
                "experiment_id": start.provenance.experiment_id,
                "descriptor_digest": start.provenance.descriptor_digest,
            }
        )
        declared["families"] = [
            {
                "family_id": family.family_id,
                "match_group": family.match_group,
                "cells": [list(cell) for cell in family.cells],
                "visible_byte_count": family.visible_byte_count,
            }
            for family in sorted(start.families, key=lambda family: family.family_id)
        ]
    # And the shapes it admits an environment's own bodies under, where it admits any. They are
    # folded in only where they are declared, on the terms the profile is: a generation composed
    # before there was a contract to declare hashed exactly the keys above, and a formula that
    # grew one would refuse every resume of every generation recorded under it.
    if start.receipt_contracts:
        declared["receipt_contracts"] = [
            contract_fields(contract)
            for contract in sorted(
                start.receipt_contracts, key=lambda contract: contract.contract_id
            )
        ]
    if start.receipt_source:
        declared["receipt_source"] = start.receipt_source
    # And the branch it serves and the branches its fork may create, on the same terms. A
    # generation serving the one slot every generation serves declares nothing here, so a history
    # recorded before a fork was a thing hashes what it always hashed; one that declares a slot of
    # its own is held to it, and a child plan naming an undeclared slot is refused against a value
    # the parent committed to before it forked.
    if start.served_slot != SINGLETON_SLOT:
        declared["served_slot"] = start.served_slot
    if start.forkable_slots:
        declared["forkable_slots"] = sorted(start.forkable_slots)
    return sha256(canonical_json(declared)).hexdigest()


def assignments_for(
    tasks: List[TaskItem], release: ReleasePlan, *, without_payload: Sequence[str] = ()
) -> List[Assignment]:
    """Return the roster a closed manifest implies under ``release``.

    The manifest and the roster are different objects and stay that way: the manifest is the
    queue a stream serves from, the roster is the set of rows an analysis counts. They are
    built together so a row cannot name a position the queue does not have.

    ``without_payload`` names the attempts this generation delivers nothing against. Their
    tasks are served, worked, and scored like any other, and no payload obligation is created
    for them, which is what a leg's filler needs and what the release plan cannot express.

    A plan that releases nothing at all leaves every row saying so. The column is what this
    generation does rather than what its manifest could have asked for, so a Never roster reads
    afterwards the way it behaved: no row created a payload, because none did.
    """
    silent = set(without_payload)
    return [
        Assignment(
            assignment_id=assignment_id_for(item.attempt_id),
            attempt_id=item.attempt_id,
            task_position=item.task_position,
            payload_position=item.payload_position,
            task_message_id=item.task_message_id,
            ack_message_id=item.ack_message_id,
            payload_message_id=item.payload_message_id,
            release_plan_id=release.release_plan_id,
            creates_payload_obligation=(
                release.creates_obligations and item.attempt_id not in silent
            ),
        )
        for item in tasks
    ]


@dataclass(frozen=True)
class Writer:
    """Who is calling, in the only terms the generation checks: an epoch and a token.

    Every call that can change the stream carries one. The epoch says which owner is speaking
    and the token proves it is that owner, so a writer whose epoch has been superseded is
    refused before it reads anything, including a writer whose call was already in flight when
    the new owner claimed.
    """

    ownership_epoch: int
    fencing_token: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class OwnershipClaim:
    """A claim to be the generation's writer, presented as a compare and swap.

    ``previous_epoch`` is the epoch the claimant read before claiming. It is the whole of the
    compare: a claimant that read a stale epoch loses, which is what makes two would-be owners
    resolve to one rather than to both. ``fencing_token`` is the new owner's unguessable secret,
    and only its hash is kept. ``configuration_hash`` is what the claimant believes it is
    resuming, and a claim that believes something else changes nothing.

    ``restored_checkpoints`` is what this claimant put back before it claimed: an attempt, and
    the exact task-start checkpoint it materialized for that attempt. An active attempt whose
    world the generation has since authorized a change to is continued only by a claimant that
    says this, because the alternative is a new owner carrying on in a world nobody restored.
    A claimant that restores nothing sends nothing, and is refused for exactly those attempts.
    """

    claimant_id: str
    previous_epoch: int
    fencing_token: str
    configuration_hash: str
    reason: str = "fresh"
    restored_checkpoints: Dict[str, str] = field(default_factory=dict)
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class OwnershipReceipt:
    """What a successful claim returns: the epoch it won, and what it replaced."""

    ownership_epoch: int
    previous_epoch: int
    fencing_token_hash: str
    configuration_hash: str
    claimant_id: str
    reason: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class VerifyBlobsInput:
    """Ask the store whether it holds the exact bytes these references name."""

    blob_root: str
    references: List[str]
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class BlobsVerified:
    """Which references the store could produce, and which it could not.

    A reference is unverified whether the object is absent or the bytes under its name hash to
    something else. The authority treats both the same way, because neither is the object the
    event would have been citing.
    """

    verified: List[str]
    unverified: List[str]
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class ConsumerClaim:
    """A caller's claim to be the generation's one logical consumer."""

    consumer_id: str
    claim_hash: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class ConsumerReceipt:
    """The bound consumer, its claim epoch, and where its cursor starts."""

    consumer_id: str
    claim_epoch: int
    initial_cursor: str
    configuration_hash: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class OfferedMessage:
    """One offered protocol result: the exact bytes, and enough to route them.

    ``visible_text`` is the canonical encoding of a Task, Payload, Wait, Done, SealAck,
    SealReject, or Info. A harness presents those bytes and attests to their hash; it does not
    rebuild them from ``kind`` and ``message_id``.
    """

    message_id: str
    kind: str
    visible_text: str
    attempt_id: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SealRequest:
    """A terminal call, as the harness saw it, with the metadata only the harness can set.

    ``terminal_source`` is who made it. The stream does not care which of the two filed: the
    seal, the grade and the acknowledgement are the same either way. What it does is write the
    answer down, because a reader counting how a generation's attempts ended is asking a
    question the score cannot answer, and an ending the agent chose and one its own step budget
    forced on it are two different things to count.
    """

    metadata: TerminalMetadata
    public_tool_name: str
    native_terminal_name: str
    native_arguments: Dict[str, Any] = field(default_factory=dict)
    terminal_source: str = AGENT_FILED


@dataclass(frozen=True)
class EnvironmentCall:
    """One ordinary environment call, named so the stream can hold the generation for it.

    The call never reaches the stream and the world it changes is one the stream cannot see.
    What the stream can do is decide whether that call may happen at all, and stay held while
    it does, which is what makes the decision and the change one thing rather than two.
    """

    call_id: str
    attempt_id: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class EnvironmentLease:
    """The stream's permission for one environment call, and what it was granted against.

    ``held`` is the answer to giving one back: true when this call was the one holding the
    generation, false when it had already been taken from it, which is what a caller reading a
    lost answer needs to tell apart.
    """

    call_id: str
    attempt_id: str
    cursor: str
    held: bool
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class FinalizeRequest:
    """A controller's request to end one attempt that nothing is going to finish.

    ``reason`` comes from the closed set above. With the attempt it is the whole of what the
    caller supplies: there is no score here and no message, because a finalization writes the
    floor and produces nothing the model can read.

    ``request_id`` is this call's logical identity, and it is here for the same reason a pull
    carries one. A controller that loses the answer retries the request it made, and the retry
    has to reach the answer it already has rather than a second ending or a stale refusal.
    """

    request_id: str
    attempt_id: str
    reason: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class AttemptFinalized:
    """The controller's receipt for an ended attempt: why, at what score, and what came back.

    ``capacity_in_use`` is read after the release, so the receipt says the capacity is free
    rather than promising that it will be.

    ``also_finalized`` names every other attempt the same transition floored: an attempt whose
    gate waits on a fact this ending made impossible has no way left to run, so it is ended
    here rather than left waiting for it. They are in the receipt because the floor is one
    atomic fact about several attempts, and a receipt naming only the one that was asked about
    would say less than happened.
    """

    attempt_id: str
    reason: str
    score: float
    capacity_in_use: int
    obligation_state: Optional[str] = None
    also_finalized: List[str] = field(default_factory=list)
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class QueueClosed:
    """The controller's receipt for a closed queue."""

    task_count: int
    closed: bool = True
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class StreamState:
    """What the harness may ask a live generation about itself.

    This is the answer to a Query, so it writes nothing and can be asked at any time. It is
    harness-only, which is what lets it carry the hidden Wait reasons: a model that could read
    them would learn from a Wait exactly what the Wait record is shaped to withhold. A generation
    that declares the info tool publishes three of these counts and no more, and it publishes
    them through a record the stream mints rather than through this read: the reasons, the
    identifiers and everything else here stay where they are.

    Assignment, release, materialization, eligibility, offer, and presentation are six counts
    and not one. Collapsing them would make an offer look like a delivery, and only the last of
    them, and only for a Payload, is one: ``payload_delivery_count`` counts the payloads whose
    bytes were handed to the transport and nothing else. What the model consumed of them is
    attested by the harness transcript rather than by any count here.

    ``ownership_epoch`` is what a resume reads before it claims, so this Query is also the
    compare half of the compare and swap. ``blob_verification`` says whether this generation
    checks the references a presentation carries, which depends on whether it was given a store.

    ``pending_origin`` and ``pending_request_id`` say which call the reserved result is owed to,
    and ``environment_call`` names a hold the generation is still under. An owner that opened
    neither of them knows both from here: the cursor is the one the reserved result was offered
    against, because a reservation and an advanced cursor cannot both be true at once.

    ``prepared_seals`` is the same fact for a filing that has no result yet. A seal accepted by
    an owner that was fenced before it committed leaves its attempt prepared and reserves
    nothing, so the request it is waiting on is named per attempt instead. The filing itself is
    not here: the arguments a model wrote belong in the transcript that holds them, and what a
    replacement cannot rebuild from there is the identity the call was made under.

    ``environment_calls`` is how many calls to a world this generation has authorized against
    each attempt since that attempt's task was presented or its checkpoint was restored. The
    calls themselves never reach the stream, so a grant is the only trace of one, and this is
    what a transport enforcing an environment's step budget is counting. A transport that kept
    none of its predecessor's memory reads the spent budget here rather than starting the
    attempt again at nothing. ``restoration_required`` is the same fact asked as a question
    about a claim.

    ``task_checkpoints`` names the checkpoint each attempt would be restored from, per attempt,
    and ``restoration_required`` names the active attempts a claim may not simply continue:
    ones the generation has authorized a change to a world for since that checkpoint committed.
    A replacement reads both before it claims, because the claim it then makes has to say which
    of them it put back.

    ``graded_evidence`` names the object each committed score was taken out of. The score is a
    number, and a reader that has the run's store can go from it to the verdict behind it: the
    reference is the generation's own, and it is committed with the score rather than left in
    the result of a call nobody kept.

    ``final_failures`` says why each ended attempt ended. It is here and nowhere else: a model
    that could read it would learn from an ending what the protocol gives it no message about.

    ``deadline_expired`` names the attempts whose deadline has passed and which the generation
    has not ended yet. There is one reason to be in it: the attempt is holding a call to a world
    this stream cannot see, so the ending is waiting for that call rather than cancelling an
    effect nothing here can observe. It is the operator's signal that a call is not coming back,
    and the takeover that ends the grant by name is what releases it.

    ``dispositions`` says what this generation delivers against each obligation and under what,
    one entry per obligation per branch. It is here rather than derivable, because what a run
    told its agent is a fact about the run and an operator reading a live generation should not
    have to infer it from which bodies happened to come out.

    ``profile`` and ``experiment_id`` are which kind of run this is and, where it is an
    experiment, which experiment registered it. They travel with the dispositions because the
    two questions an analysis asks of a body are what it said and who decided it would: a
    stamped honest body and a registered honest cell are the same delivery and different facts.
    """

    generation_state: str
    cursor: str
    configuration_hash: str
    stream_state_sha256: str
    ownership_epoch: int
    fencing_token_hash: Optional[str]
    ownership_claims: int
    blob_verification: str
    consumer_id: Optional[str]
    queue_closed: bool
    tasks_remaining: int
    capacity: int
    capacity_in_use: int
    pending_message_id: Optional[str]
    pending_kind: Optional[str]
    pending_origin: Optional[str]
    pending_request_id: Optional[str]
    environment_call: Optional[str]
    prepared_seals: Dict[str, str]
    task_checkpoints: Dict[str, str]
    graded_evidence: Dict[str, str]
    environment_calls: Dict[str, int]
    restoration_required: List[str]
    attempts: Dict[str, str]
    obligations: Dict[str, str]
    release_plan_id: str
    release_predicate: str
    assignment_count: int
    materialization_count: int
    eligibility_count: int
    offer_count: int
    presentation_count: int
    payload_delivery_count: int
    wait_count: int
    wait_reasons: Dict[str, int]
    final_failures: Dict[str, str]
    deadline_expired: List[str] = field(default_factory=list)
    dispositions: Dict[str, str] = field(default_factory=dict)
    profile: str = LEGACY
    experiment_id: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION
    # Where the generation is in the chain of executions it has run in, whether it is on its way
    # to another one, and the size that stopped it starting another. All three are operational:
    # none is inside the projection a presentation attests against, and none says anything about
    # what the generation serves.
    #
    # ``turnover_requested`` is what a transport waits on. A generation that has decided to
    # continue as new rejects every arriving Update that cannot bring the boundary about, and a
    # caller so rejected has to be able to tell a generation working towards a boundary from one
    # that has stopped. Reading this is a Query, which costs nothing against the cap, so a call
    # held up by a boundary can ask as often as it likes.
    #
    # ``turnover_refused_bytes`` is the honest end of an unsupported profile: the carrier would
    # not fit under the ceiling, so the generation kept serving in the execution it was already
    # in, and this is the number a launcher reports rather than guessing why the run stopped.
    turnovers: int = 0
    turnover_requested: bool = False
    # Why this generation gave up on a boundary, where it did. A carrier that would not fit is
    # one way; a boundary whose clearing work spent the reserve before it succeeded is the other.
    # Both leave the generation serving out the execution it is in, so both can end at the
    # service cap, and this is what a launcher reports that by rather than guessing.
    turnover_refused: Optional[str] = None
    turnover_refused_bytes: Optional[int] = None
    # The store reads this generation has in flight, and how many it has finished. An ownership
    # claim reads the store before it may swap the epoch, and that read is an Activity with its
    # own timeout and its own retries, so a claim can hold a boundary open for minutes while
    # moving nothing else a caller could see. These say that is what is happening.
    #
    # Neither is inside the projection a presentation attests to. They differ from each other in
    # what a boundary does to them. The count of finished reads is a count of what the generation
    # has done, like the offers and the eligibilities beside it, so it crosses: a generation that
    # has read the store twice has read it twice whichever execution is answering. The reads in
    # flight cannot cross, because a legal boundary has no handler running and therefore no read
    # in flight, so it is zero on both sides of one.
    verifying: int = 0
    verification_batches: int = 0
    # How many accepted handlers are running. A boundary waits for every one of them, and this is
    # how a caller held off for that boundary can see that it is waiting for something. The
    # projection cannot show it: an owner replaced while its filing was still grading leaves work
    # that no attempt state, pending message or grant names any more. Operational, like the four
    # above, and in no hash.
    unfinished_handlers: int = 0


@dataclass(frozen=True)
class SourceProvenance:
    """What one attempt's committed source was, and which cell of it this row was served.

    It is controller side and never anything an agent is shown. The whole of it is read out of
    the descriptor the attempt already holds, so a run answers this after its world is gone, its
    bank is unreachable and its cell store is empty: the descriptor is recorded state and not a
    blob somebody has to resolve.

    The bindings are what a later reader checks a source against. The hidden execution id is
    deliberately not among them, as it is not in the descriptor: it is the preimage the seal ids
    are minted from, and a row is not the place to publish it. What stands in its place is the
    execution ordinal, which is what the seal id is recomputed with.

    ``canonical_submission_sha256`` is the plain digest of the versioned canonical submission
    text, and ``kernel_submission_digest`` is the domain separated one over the attempt, the
    terminal name and those bytes. Two filings with identical canonical text share the first and
    the bank's own filing digest tells them apart, so neither name stands in for the other here.

    The three selection fields are absent where no selection was made, which is a state rather
    than a gap: a position the roster gave no payload obligation captures its source, records its
    grade and delivers nothing, and capture is never conditional on exposure.
    """

    source_commitment: str
    bundle_digest: str
    environment_task_id: str
    source_attempt_id: str
    source_seal_id: str
    execution_ordinal: int
    canonicalization_version: str
    renderer_configuration: str
    canonical_submission_sha256: str
    kernel_submission_digest: str
    selected_cell: Optional[str] = None
    selected_body_reference: Optional[str] = None
    selected_policy_digest: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class AttemptRecord:
    """One attempt as a record: what it was assigned, what it filed, what it scored.

    This is the row an analysis counts. It is harness-only: the acknowledgement commits to what
    was filed and says nothing about how good it was, and this row carries the assignment, the
    ending, the disposition and the score together, which is more than any policy publishes.
    What the agent is told is the payload, and the policy this row names is what decided it.

    The positions are the assignment's, so a row says where the attempt sat in its manifest
    rather than what order a stream happened to serve. The three message identifiers are the
    ones the manifest preallocated, and each is reported as delivered rather than as offered,
    because an offer is a reservation and a delivery is the exact bytes handed to the transport
    that carries them. What the model consumed of those bytes is attested by the harness
    transcript and not by this row.

    ``score`` is ``None`` until the seal that made it authoritative, and it stays ``None``
    rather than becoming a zero: an attempt nobody has sealed has no score, which is not the
    same fact as an attempt that scored nothing.

    An attempt that ended without a filing does have one, and it is the floor. That is a third
    fact again, and the number alone cannot say so: a floored attempt and one the environment
    graded at nothing carry the same zero. ``final_failure`` is what tells them apart, and it
    says which ending it was, because a run out of steps, a run out of time, a gate that can no
    longer open and a seal whose batch could not go on are four different things to count.
    ``deadline_expired`` is the clock on its own: an expiry the generation has recorded and not
    yet acted on, which is what a row carries while the attempt is still holding a call to a
    world nothing here can see.

    ``terminal_source`` is who filed, for the attempts that were filed at all. A generation over
    an environment whose horizon is a graded ending seals an attempt that spent its budget
    without waiting for the agent to file, so a sealed row is no longer proof the agent chose to
    end there. The two are separated here rather than left to be guessed from a step count, and
    an attempt nobody filed carries neither.

    ``creates_payload_obligation`` is the roster's own column, and ``payload_state`` is what
    became of the obligation it promised. Without them ``payload_delivered`` being false says
    two different things at once: a row this generation owed a payload and never delivered, and
    a row that was never going to have one, which is what a filler is. Those are a missed
    treatment and a structural absence, and an analysis that cannot tell them apart is counting
    fillers as failures. A row that creates no obligation has no obligation state.

    ``payload_policy`` and ``payload_disposition`` are what this row was told and under what
    rules. A generation recorded before a policy was a fact about one reads as the legacy
    placeholder rather than as honest, because absence is a history that predates the question
    and never an answer to it.

    ``profile`` and ``payload_resolution_source`` are who decided that. An analysis counting
    these rows is asking which of two mistakes a run made, if either: an experiment cell served
    as an ordinary default, or an ordinary run blinded by something nobody registered. Neither
    question can be answered from the policy name alone, because the same name is a correct
    answer to both.

    The five ``failure_`` fields are what stopped the seal that ended this attempt. An accepted
    terminal runs the seal and the grade, and the renderer as well when the attempt carries a
    payload obligation, and any of them can fail for good, so ``failure_activity`` and
    ``failure_activity_id`` name which one did, ``failure_kind`` is the semantic type it failed
    as, ``failure_message`` is a bounded line of what it said, and ``failure_retry_state`` is why
    the retries stopped. A result the seal could not vouch for is the other ending, and it fills
    two of the five: the Activity behind it succeeded, so there is a kind and a message and no
    retry state, and no Activity is named as having failed. They are read out of what the durable
    history recorded, so a row rebuilt from that history says the same thing every time, and they
    are absent on every row whose seal did not end it. None of it is ever shown to a model: a
    message an environment raised with can name what it was grading.

    The last three are the receipt half, and they are three separate facts rather than one. A row
    names the contract its capture is validated against from the moment the roster resolved it,
    before anything is sealed and whether or not it will ever deliver a body, so an attempt that
    has captured nothing yet is still one somebody expects a source from. The provenance arrives
    with that source and holds the bindings and the selection made from it. And the visible digest
    is the join to what was committed: the presentation row for this attempt's payload says which
    bytes the generation stands behind, and a harness reconciling its own transcript compares
    against that rather than against anything this row could say about the body itself.
    """

    attempt_id: str
    task_position: int
    payload_position: int
    state: str
    terminal_tool: Optional[str]
    terminal_source: Optional[str]
    canonicalization_version: str
    submission_digest: Optional[str]
    score: Optional[float]
    decode_state: Optional[str]
    seal_ordinal: Optional[int]
    final_failure: Optional[str]
    deadline_expired: bool
    task_message_id: str
    task_delivered: bool
    ack_message_id: str
    ack_delivered: bool
    payload_message_id: str
    payload_delivered: bool
    creates_payload_obligation: bool
    payload_state: Optional[str]
    payload_policy: Optional[str]
    payload_disposition: Optional[str]
    profile: str = LEGACY
    payload_resolution_source: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION
    failure_activity: Optional[str] = None
    failure_activity_id: Optional[str] = None
    failure_kind: Optional[str] = None
    failure_message: Optional[str] = None
    failure_retry_state: Optional[str] = None
    receipt_contract_id: Optional[str] = None
    source_provenance: Optional[SourceProvenance] = None
    payload_visible_sha256: Optional[str] = None


@dataclass(frozen=True)
class PresentedMessage:
    """One message this generation committed to deliver, in the order it committed them.

    A row here is a commitment and not a handoff. What the generation verified is that the
    digest a transport claimed matched the bytes it was holding out, and what it then did is
    accept that claim and advance its cursor past the message. All of that happens before the
    transport has a result to hand anybody: an acknowledgement that never gets back to the call
    that asked for it leaves a message this generation has counted and no bytes anywhere, and a
    generation whose owner is replaced while such a result is owed never hands those bytes over
    at all.

    So the reconciliation is what establishes both of the things this cannot. Whether the bytes
    reached the transport, and whether a model then read them, are facts about the harness, and
    the harness's own transcript is where a claim about either has to be checked against these
    rows. A reader holding these rows alone holds an account of what was committed.

    An attempt's record says which of its three messages were committed, and that is the fact an
    analysis counts. This says which bytes each of them was, and it says it for every kind rather
    than for those three: a Wait, a SealReject, an Info answer and the Done that ends the
    generation are committed under the same attestation, and a harness reconciling its own
    transcript against what was committed would report a run as whole while its own record was
    missing one.

    ``visible_bytes_sha256`` is the digest the presentation was verified against. It is the whole
    of what a reconciliation needs and less than the bytes themselves: it does not say what the
    bytes were, and whoever compares already holds what the harness wrote down, so what is being
    asked is only whether the two are the same.
    """

    order: int
    kind: str
    message_id: str
    attempt_id: Optional[str]
    visible_bytes_sha256: str
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class GenerationRecords:
    """One generation's attempts and its commitments, read out of the same moment.

    The two are answered together because a generation that is still serving moves between two
    questions. Asked separately, a payload can be owed in the first answer and committed in the
    second, and the pair describes a generation that never existed in either state. Asked here,
    both halves are read off the one projection, so a row and the commitments beside it are the
    same run at the same point.

    The operation failures are the third list, read off that same moment for the same reason: an
    attempt whose evidence went missing reads as delivered in one answer and as refused in
    another, and which of the two is true is a question about one point in the run.
    """

    attempts: List[AttemptRecord]
    presentations: List[PresentedMessage]
    protocol_version: int = PROTOCOL_VERSION
    operation_failures: List[OperationFailure] = field(default_factory=list)


@dataclass(frozen=True)
class StreamOutcome:
    """What a finished generation returns: counts, not content.

    ``payloads_delivered`` counts the payloads handed to the transport, which is the fact this
    generation holds. Whether a model read one of them is not something a server can know, so
    the count is named for the delivery and not for the reading.
    """

    generation_state: str
    cursor: str
    sealed: int
    payloads_delivered: int
    finalized: int = 0
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SealAttemptInput:
    """Ask the environment for its canonical pre-verdict submission and its local seal.

    ``blob_root`` is where the environment installs the submission bytes it captured, so the
    reference it returns names an object a later event may cite. A generation without a store
    gets the reference anyway and nothing can be read back under it.

    ``execution_ordinal`` is which execution of this generation is asking. It is what a
    descriptor carries in place of the hidden execution id, so a workflow can recompute the seal
    id its own start implies rather than being handed the identity it is checking against. Minus
    one is the value a caller composed before there was one to send, and no source is published
    under it.

    ``receipt_contract`` is the registered shape this generation admits an environment's own
    bodies under. It is absent by default, and an environment given none publishes no descriptor
    at all: a generation that declared no contract is served exactly as it was before there was
    one to declare.
    """

    attempt_id: str
    seal_id: str
    native_terminal_name: str
    canonicalization_version: str
    native_arguments: Dict[str, Any] = field(default_factory=dict)
    blob_root: Optional[str] = None
    execution_ordinal: int = -1
    receipt_contract: Optional[ReceiptContract] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class SealAttemptResult:
    """The immutable submission, captured before any verifier ran.

    ``canonical_submission_text`` is the byte string the digest covers. It is carried inline
    because a kernel submission is small; ``canonical_submission`` names the same bytes by
    hash, which is the form a larger one takes.

    ``source_artifact`` is the descriptor an environment that publishes its own bodies validated
    and installed before it returned anything, and ``source_commitment`` is the digest of that
    descriptor's canonical bytes, which is also the address of the blob holding them. No cell
    body comes back here: the bodies are in the store under the references the descriptor names,
    and a result that carried one would be handing a body to a transition that has not yet
    decided which cell this obligation is served. Both are absent on a generation that declared
    no contract, which is every generation recorded before there was one to declare.

    Those two arrive as whatever was encoded, for the reason every field of a grade result does.
    A field with a type is a field the decoder has to make that type of, and a descriptor missing
    a name would fail that decoding rather than the check: the failure happens while the
    generation is being handed the result, before any code of its own runs, so the generation
    would fail that step again on every retry and record nothing about why. So the wire shape is
    permissive and the authority is strict: :func:`shogym.serve.protocol_v2.artifact
    .read_source_artifact` decides what a descriptor is, at the recorded boundary that reads it,
    and a mapping carrying an unknown name or missing a declared one ends the attempt with a
    reason.
    """

    attempt_id: str
    seal_id: str
    canonicalization_version: str
    canonical_submission_text: str
    canonical_submission: BlobRef
    environment_recovery_token: str
    source_artifact: Any = None
    source_commitment: Any = ""
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class GradeAttemptInput:
    """Grade sealed evidence, out of process, against a world nobody can still change.

    ``blob_root`` is where the environment installs the verdict it took, for the reason the
    seal is given one: the reference the result carries is the run's own authoritative record
    of what the score was taken from, and a reference to bytes no store holds is a name nothing
    can resolve. A generation without a store gets the reference anyway and nothing can be read
    back under it.
    """

    attempt_id: str
    seal_id: str
    submission_digest: str
    canonical_submission_text: str
    environment_recovery_token: str
    blob_root: Optional[str] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class GradeAttemptResult:
    """A score and how the submission decoded.

    ``seal_id`` says which seal was graded, and the attempt ID does not: two branches of a fork
    share the public ID, so a score carrying only that could belong to either.

    ``decode_state`` distinguishes a submission that said nothing from one the grader could
    not read, and both are successful results: neither is an infrastructure failure and
    neither is retried.

    ``grade`` is the grader saying which grader it is. A generation is built over a declared
    grade identity and its honest bodies publish that grader's number, so the number arriving
    here carries the same identity or the generation is publishing one grader's verdict under
    another's name. The kernel's grade computes from the shape of a filing, reaches no world, and
    says so, which is what stops a transport fixture from being printed as a verdict. It is
    absent by default because a result recorded before graders said this is one nobody can now
    ask, and an absent one is read as a stand-in rather than as an environment's own.

    ``public_components`` is what the environment published for the agent beside the score, as
    numbers under token names. It is not the verdict and not the evidence: those stay in the
    reference this result carries, which a harness can resolve and a renderer cannot. What names
    may appear is the roster in the grade identity, declared before the run rather than taken
    from whatever the grader returned.

    Every field arrives as whatever the grader put in it. They are declared that way on purpose,
    and the reason is the same for the identity and the reference as it is for the numbers: a
    field with a type is a field the decoder has to make that type of, and a grader that returned
    a string, a list or an object under one would fail the decoding rather than the check. That
    failure is not an ending. It happens while the generation is being handed the result, before
    any code of its own runs, so the generation fails that step again on every retry, records
    nothing about why, and answers no question while it does. So the wire shape is permissive and
    the authority is strict: what a score, a name, an identity and a reference are gets decided
    where the result is read, and a value that is none of them ends the attempt with a reason.
    """

    attempt_id: Any
    seal_id: Any
    score: Any
    decode_state: Any
    evidence: Any
    grade: Any = None
    public_components: Any = field(default_factory=dict)
    protocol_version: Any = PROTOCOL_VERSION


@dataclass(frozen=True)
class SelectedSourceReference:
    """The one committed cell an artifact obligation is served, and how to fetch it.

    It is one reference and never the map. All three cells of a source travel in one envelope
    and a store verifies any digest without knowing which cell it is, so a resolver handed the
    map could satisfy every hash and size check while returning the cell nobody selected. What
    crosses is therefore the entry the selection names, with the shape it has to come back in.

    ``masked_body_sha256`` and ``slot_spans`` are the publication's own evidence and the
    registered geometry it was taken under. They cross so the check that the delivered body is
    the pair's is made against what was registered rather than against a layout the resolver
    proposed.

    ``blob_root`` is where the controller keeps the run's objects. It is deployment rather than
    identity, which is why it is here beside the reference and nowhere inside a digest.
    """

    source_commitment: str
    cell: str
    contract_id: str
    body_sha256: str
    body_size: int
    body_encoding: str
    media_type: str
    masked_body_sha256: str
    slot_spans: Tuple[Tuple[int, int], ...]
    blob_root: str


def source_seal_id(origin: SourceOriginContext, attempt_id: str) -> str:
    """Return the seal id a source published for ``attempt_id`` under one origin."""
    return hidden_seal_id(origin.hidden_execution_id, origin.execution_ordinal, attempt_id)


def derived_selection(
    *,
    source: SourceArtifactManifest,
    origin: SourceOriginContext,
    commitment: str,
    cell: str,
    policy_digest: str,
    blob_root: str,
) -> SelectedSourceReference:
    """Return the one reference a derivation of this committed source resolves for ``cell``.

    This is the whole of what a derivation takes: committed source evidence, the identity that
    source was sealed under, an allowed target policy and cell, and where the run keeps its
    objects. No live world, no canonical answer text, no environment recovery token, no second
    seal, no grade and no render. That is what makes a candidate for either eligible cell
    derivable long after the generation that captured the source has continued as new and the
    world behind it is gone.

    The origin is a parameter rather than a value read off whichever start is calling, and that is
    the point of the field. A seal id is minted from a hidden execution id, an ordinal and an
    attempt, and the descriptor carries the ordinal alone; a derivation that recomputed it from
    its own start would be checking an inherited source against the generation it arrived at
    rather than against the one that produced it, and would quietly accept a source no origin
    vouches for.

    Everything here refuses rather than repairs. The commitment is recomputed from the
    descriptor's own canonical bytes, the seal id from the origin, and the cell has to be one an
    arm may be served, declared by an artifact policy and admitted by the source's own contract.
    The oracle is refused at the first of those: it is named by the descriptor, retained with the
    rest, and is not a cell any derivation resolves.
    """
    check_source_artifact(source)
    if commitment != source_commitment(source):
        raise WireFormatError(
            "a derivation names a source commitment other than the one the descriptor's own "
            "canonical bytes hash to"
        )
    if source.source_seal_id != source_seal_id(origin, source.source_attempt_id):
        raise WireFormatError(
            f"the source for attempt {source.source_attempt_id} was sealed under another hidden "
            "execution or another ordinal than the origin this derivation was given"
        )
    if source.execution_ordinal != origin.execution_ordinal:
        raise WireFormatError(
            f"this source names the execution ordinal {source.execution_ordinal} and its origin "
            f"names {origin.execution_ordinal}"
        )
    if cell not in ELIGIBLE_CELLS:
        raise WireFormatError(
            f"an arm is served {sorted(ELIGIBLE_CELLS)} and this derivation asked for {cell!r}"
        )
    policy = POLICIES.get(policy_digest)
    if policy is None or policy.exposure != ARTIFACT or cell not in policy.cells:
        raise WireFormatError(
            f"a committed cell is delivered by a policy this build implements that declares "
            f"{cell!r}, and this derivation named {policy_digest[:16]!r}"
        )
    contract = source.receipt_contract
    if (policy_digest, cell) not in contract.cells:
        raise WireFormatError(
            f"the contract {contract.contract_id} admits {[cell for _d, cell in contract.cells]} "
            f"and this derivation asked for {cell!r} under a policy it does not admit"
        )
    if not blob_root:
        raise WireFormatError(
            "a committed cell is an object of the run's own store, and this derivation was given "
            "no store to resolve it in"
        )
    reference = source.cells[cell]
    return SelectedSourceReference(
        source_commitment=commitment,
        cell=cell,
        contract_id=contract.contract_id,
        body_sha256=reference.sha256,
        body_size=reference.size,
        body_encoding=contract.body_encoding,
        media_type=reference.media_type,
        masked_body_sha256=source.pair_parity.masked_body_sha256,
        slot_spans=mask_spans(contract),
        blob_root=blob_root,
    )


@dataclass(frozen=True)
class GeneratePayloadBundleInput:
    """Build every candidate this obligation might deliver, before the acknowledgement.

    ``policy_digest`` is what this obligation was resolved to, and it decides which renderer
    runs. A digest this build does not implement is a failure rather than a body: there is no
    renderer to fall back to, because falling back is how a run comes to serve something other
    than what its record says it served. An empty digest is a request from a history recorded
    before policies existed and renders the placeholder it recorded.

    ``public_grade`` is the whole of what a renderer may know about the verdict, and it is
    present only where the resolved policy publishes it. A blinded renderer is not given a grade
    to withhold: it is handed a request with no grade in it, so leaking one is not a discipline
    it keeps but a value it does not have.

    ``selected`` is the committed cell an artifact policy resolves, and it is the whole of what
    such a request carries about the source. The canonical submission text is empty under one and
    the grade is absent: an artifact body is copied rather than rendered, so a route that needed
    the filing's text would be underivable once the generation had continued and the text had
    stopped crossing.
    """

    attempt_id: str
    payload_position: int
    payload_message_id: str
    submission_digest: str
    canonical_submission_text: str = ""
    policy_digest: str = ""
    cell: str = ""
    public_grade: Optional[PublicGrade] = None
    selected: Optional[SelectedSourceReference] = None
    protocol_version: int = PROTOCOL_VERSION


@dataclass(frozen=True)
class PayloadCandidate:
    """One rendered candidate, with the proof metadata a family gate will need.

    ``policy_digest`` and ``renderer_version`` are echoed from the request rather than being the
    renderer's own opinion of itself. What they catch is a Worker running code the generation did
    not ask for: a build that never learned about policies returns neither, and the seal that
    asked for one refuses the candidate instead of serving whatever came back under an honest
    label. Both are empty where the request carried no policy, which is what a replayed legacy
    result looks like.

    ``source_commitment``, ``body_reference`` and the two resolver fields are the same echo for a
    body that was resolved rather than rendered: which source it came out of, which committed
    entry was read, and which implementation read it. They are empty on every candidate a
    renderer built from a projection, which is what a historical scalar result carries, and under
    an artifact policy the inner hash is required to be that body reference, so the bytes that
    came back are the bytes the source committed.

    This is the candidate a generation keeps and carries, so its values are the validated ones.
    What an Activity returns is :class:`PayloadCandidateResult`, which carries those four as they
    were encoded; the strict reader at the seal boundary is what makes this record of one.
    """

    cell: str
    renderer_id: str
    match_group: str
    body: str
    inner_sha256: str
    visible_sha256: str
    visible_byte_count: int
    renderer_version: str = ""
    policy_digest: str = ""
    source_commitment: str = ""
    body_reference: str = ""
    resolver_id: str = ""
    resolver_version: str = ""


@dataclass(frozen=True)
class PayloadCandidateResult:
    """One candidate as an Activity result carries it, before anything has read it.

    It is the record above written for the wire, and it exists because the two boundaries want
    opposite things. What a generation keeps and carries is typed: the values have been compared
    against the policy this obligation was resolved to and against the source it was selected
    from, so a carried candidate holding anything else is a record this build never wrote. What
    comes back from a Worker has been compared against nothing yet.

    So the four resolved values arrive as whatever was encoded, the way the seal result carries
    its own descriptor. A field with a type is a field the decoder has to make that type of, and
    a mapping written where a digest belongs would fail that decoding while the generation was
    being handed the result, before any code of its own ran: the Workflow Task would fail on
    every retry, the seal would never be answered, and the record would say nothing about why.
    The nine values beside them are the ones this route inherited and they keep the types they
    had. :func:`shogym.serve.protocol_v2.kernel.workflow._read_candidate` is what makes a
    candidate of this, at the recorded boundary the rest of the result is checked at.
    """

    cell: str
    renderer_id: str
    match_group: str
    body: str
    inner_sha256: str
    visible_sha256: str
    visible_byte_count: int
    renderer_version: str = ""
    policy_digest: str = ""
    source_commitment: Any = ""
    body_reference: Any = ""
    resolver_id: Any = ""
    resolver_version: Any = ""


@dataclass(frozen=True)
class PayloadBundle:
    """The complete candidate set for one obligation. The kernel family has one member.

    ``submission_digest`` names the filing these candidates were rendered from. The attempt ID
    and the position are public and survive a fork, so they say which obligation asked for a
    bundle and not which filing it answers.
    """

    attempt_id: str
    payload_position: int
    submission_digest: str
    candidates: List[PayloadCandidateResult]
    protocol_version: int = PROTOCOL_VERSION


def finalize_request_identity(request: FinalizeRequest) -> str:
    """Return the canonical identity of a finalization request.

    It covers the whole request and not the fields the ending happens to read, the declared
    protocol version among them. A retry carrying the same request ID and the same identity
    replays its receipt; one carrying the same ID and anything else is a conflict, which is the
    point of hashing the request rather than trusting the ID beside a few of its fields.
    """
    return sha256(
        length_prefixed(b"finalize-request-v2")
        + length_prefixed(
            canonical_json(
                {
                    "request_id": request.request_id,
                    "attempt_id": request.attempt_id,
                    "reason": request.reason,
                    "protocol_version": request.protocol_version,
                }
            )
        )
    ).hexdigest()


def hidden_seal_id(hidden_execution_id: str, execution_ordinal: int, attempt_id: str) -> str:
    """Return the environment's idempotency key for sealing one attempt.

    The public attempt ID alone would not do: two branches of a fork share it, so an
    environment keyed on it would treat the second branch's filing as a replay of the first.
    Mixing in the hidden execution ID and the execution ordinal makes the key branch-local
    while leaving the public ID branch-neutral.
    """
    return sha256(
        length_prefixed(b"seal-id-v2")
        + length_prefixed(hidden_execution_id.encode("utf-8"))
        + execution_ordinal.to_bytes(8, "big")
        + length_prefixed(attempt_id.encode("utf-8"))
    ).hexdigest()
