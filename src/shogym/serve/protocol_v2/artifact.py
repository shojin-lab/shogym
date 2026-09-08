"""The source a receipt body is cut from, as a record rather than as a convention.

An environment that renders its own feedback bytes has something no scalar policy has: a body
that exists before any obligation is resolved, that no renderer can rebuild from a grade, and
that two derivations of one source have to agree on byte for byte. This module is where that
source is written down. :class:`SourceArtifactManifest` names the world the bodies were made in,
the filing they answer, the three cells that were made in one act, and the evidence that the two
eligible cells differ only where they are allowed to. Its canonical bytes are one blob and the
digest of those bytes is the source commitment, so there is no second digest to reconcile and
the preimage of the commitment is an object a reader can fetch.

:class:`ReceiptContract` is the other half: the shape an environment's bodies are admitted under,
declared before the run. It fixes the size a body comes to, the encoding and media type it
travels as, and the geometry the mask is expanded from, so the comparison that establishes a pair
is made against a registered layout rather than against a layout whichever candidate arrived
proposed.

Everything here refuses rather than repairs. A version this build does not admit, a cell map that
is not exactly the three kinds, a digest that is not a digest, a size that disagrees with the
contract, an encoding that is not the declared one and a mapping carrying a name this record does
not declare are each a refusal at the boundary that read it, because a descriptor half understood
is a provenance claim nobody can check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from shogym.serve.protocol_v2.blobs import BlobRef
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.jcs import encode as canonical_json
from shogym.serve.protocol_v2.policy import (
    ARTIFACT,
    ARTIFACT_POLICY_DIGESTS,
    DELIVER,
    EXPERIMENT,
    GRADED_RECEIPT_ARTIFACT_V1,
    KERNEL_MATCH_GROUP,
    PLACEBO_RECEIPT_ARTIFACT_V1,
    POLICIES,
    SELECTABLE,
    WITHHOLD,
    GradeIdentity,
    MatchedFamily,
    PayloadDisposition,
    PolicyViolation,
    PublishedNumber,
    check_grade,
    number_text,
)
from shogym.serve.protocol_v2.records import Payload, require_opaque_id, visible_bytes

#: What a descriptor written by this build says it is. A generation admits the versions its
#: contract names, and this is the one this code writes and reads.
ARTIFACT_SCHEMA_VERSION = "shogym.receipt-artifact.1"

#: The three cells one source holds. The map is closed: a source names all three or it is not a
#: source, and a fourth kind is a descriptor this build does not understand.
GRADED_CELL = "graded"
PLACEBO_CELL = "placebo"
ORACLE_CELL = "oracle"
CELL_KINDS = (GRADED_CELL, PLACEBO_CELL, ORACLE_CELL)
#: The two an arm may be served. The oracle is retained and never delivered, so it is named by
#: the manifest and is not a cell a live arm selects.
ELIGIBLE_CELLS = (GRADED_CELL, PLACEBO_CELL)

#: How a body is written and what it travels as. Both are declared by the contract and checked
#: against it, and both have exactly one legal value here: the mask arithmetic is byte offsets
#: into the serialized body, which only means anything where a character is a byte.
BODY_ENCODING = "ascii"
BODY_MEDIA_TYPE = "text/plain"

#: What the manifest's own bytes are installed as. The commitment is that blob's plain SHA-256.
MANIFEST_MEDIA_TYPE = "application/json"

#: What the mask writes over a registered slot. It is frozen rather than chosen: blanking the
#: slots with NULs would satisfy every equality sentence the pair check makes and would commit a
#: different source, so the fill is part of the evidence rather than an implementation detail.
MASK_FILL = b"?"

# The largest integer the canonical encoding writes. A count above it has no canonical form, so a
# descriptor carrying one is refused here rather than serializing well and failing when its bytes
# are taken.
_SAFE_INTEGER_MAX = 9007199254740991
_SHA256_HEX = 64
# A name a generation declares, under the grammar the matched families are named by, so a
# contract id and a family id are the same kind of word and can be held to being distinct.
_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,31}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ReceiptContract:
    """The registered shape one generation's receipt bodies are published under.

    ``cells`` is the closed pair this contract admits, one graded and one placebo, each naming
    the policy that may deliver it. ``body_size`` is the fixed count every cell of the family
    serializes to, and it is a property of the registered envelope rather than of a draw, which
    is what lets a body that is one byte short be refused rather than measured.

    ``rows_start``, ``row_line_width``, ``row_count`` and ``slot_spans`` are the geometry the
    mask is expanded from. They are declared here, ahead of any publication, because the mask is
    what establishes that two bodies differ only inside registered slots: a mask taken from
    whichever body arrived would be a comparison the thing being compared got to choose.

    ``manifest_schema_versions`` is what this generation admits a descriptor to be written as,
    and ``renderer_configuration`` is the renderer identity a descriptor has to carry, so a
    source rendered by other code is refused rather than served.
    """

    contract_id: str
    match_group: str
    cells: Tuple[Tuple[str, str], ...]
    body_size: int
    body_encoding: str
    media_type: str
    rows_start: int
    row_line_width: int
    row_count: int
    slot_spans: Tuple[Tuple[int, int], ...]
    manifest_schema_versions: Tuple[str, ...]
    renderer_configuration: str


@dataclass(frozen=True)
class PairParity:
    """What publication proved about the two eligible cells, carried inside the commitment.

    ``masked_body_sha256`` is the hash of the masked body both cells produce. It is a
    consistency check on the selected body against publication's evidence and not a proof about
    the cell that was not selected: one body and one committed hash cannot say what the other
    cell's mask was, so what establishes the pair is publication verifying both while it held
    both.

    ``encoded_body_bytes`` is the body's contribution to a canonical JSON string with its two
    enclosing quotes left out, so a wire count is this number plus the wrapper an obligation
    comes to with an empty body, exactly. Storing the quoted length instead would overshoot
    every candidate by two bytes and refuse filings every other check passed.
    """

    body_size: int
    encoded_body_bytes: int
    masked_body_sha256: str


@dataclass(frozen=True)
class SourceArtifactManifest:
    """One sealed source, named by everything a later derivation has to be able to check.

    Every field is required. The hidden execution id is deliberately absent: the manifest
    carries the ordinal, and a workflow recomputes the seal id from its own start's hidden
    execution id, that ordinal and the attempt, so a descriptor cannot hand a generation the
    identity it is supposed to be checked against.

    ``bank_filing_digest`` and ``bank_cell_digests`` are the bank's own length-prefixed digests
    and are never blob addresses. They are named apart from ``cells`` for that reason: the same
    bytes have two names in this system, and a record that let one stand in for the other would
    make a store lookup out of a value no store ever addressed.
    """

    schema_version: str
    bundle_digest: str
    environment_task_id: str
    source_attempt_id: str
    source_seal_id: str
    execution_ordinal: int
    canonicalization_version: str
    canonical_submission: BlobRef
    kernel_submission_digest: str
    renderer_configuration: str
    receipt_contract: ReceiptContract
    grade_identity: GradeIdentity
    cells: Dict[str, BlobRef]
    bank_filing_digest: str
    bank_cell_digests: Dict[str, str]
    pair_parity: PairParity


def mask_spans(contract: ReceiptContract) -> Tuple[Tuple[int, int], ...]:
    """Return the byte ranges a masked body blanks, expanded from one contract's geometry.

    Zero based offsets into the serialized body, half open, one span per registered slot per
    printed row, in row then slot order. The expansion is arithmetic on the declaration and
    nothing else: it reads no body, so the same spans are produced by the environment that
    publishes a pair and by a workflow that checks one long after the world is gone.
    """
    spans: List[Tuple[int, int]] = []
    for row in range(contract.row_count):
        line = contract.rows_start + row * contract.row_line_width
        for low, high in contract.slot_spans:
            spans.append((line + low, line + high))
    return tuple(spans)


def masked_body(body: bytes, spans: Iterable[Tuple[int, int]]) -> bytes:
    """Return ``body`` with every span written over by the registered fill byte."""
    out = bytearray(body)
    for low, high in spans:
        out[low:high] = MASK_FILL * (high - low)
    return bytes(out)


def encoded_body_bytes(body: str) -> int:
    """Return what one body contributes to a canonical JSON string, its two quotes left out.

    The empty body's encoding is subtracted rather than the number two written down, so the
    count follows the encoder: it is the length the encoder gives this body less the length it
    gives no body at all, which is the quotes and nothing else.
    """
    return len(canonical_json(body)) - len(canonical_json(""))


def payload_wire_count(
    *, payload_message_id: str, attempt_id: str, encoded_body_bytes: int
) -> int:
    """Return the model-visible byte count one obligation comes to carrying this body.

    It is the wrapper this obligation's message comes to with no body at all, plus the body's
    own stored contribution, exactly. The wrapper is measured rather than written down, so a
    change to what a payload record carries moves this with it; the body's half is the count the
    parity evidence stored, which leaves the two enclosing quotes out and is therefore added to a
    wrapper that already has them. Adding the quoted length instead overshoots every candidate by
    two bytes, which refuses filings every other check passed.
    """
    empty = visible_bytes(
        Payload(message_id=payload_message_id, attempt_id=attempt_id, body="")
    )
    return len(empty) + encoded_body_bytes


def blob_fields(reference: BlobRef) -> Dict[str, Any]:
    """Return one blob reference as the three values the canonical encoding writes."""
    return {
        "sha256": reference.sha256,
        "size": reference.size,
        "media_type": reference.media_type,
    }


def contract_fields(contract: ReceiptContract) -> Dict[str, Any]:
    """Return one contract as the value its canonical encoding is taken over."""
    return {
        "contract_id": contract.contract_id,
        "match_group": contract.match_group,
        "cells": [[digest, cell] for digest, cell in contract.cells],
        "body_size": contract.body_size,
        "body_encoding": contract.body_encoding,
        "media_type": contract.media_type,
        "rows_start": contract.rows_start,
        "row_line_width": contract.row_line_width,
        "row_count": contract.row_count,
        "slot_spans": [[low, high] for low, high in contract.slot_spans],
        "manifest_schema_versions": list(contract.manifest_schema_versions),
        "renderer_configuration": contract.renderer_configuration,
    }


def grade_fields(grade: GradeIdentity) -> Dict[str, Any]:
    """Return one grade identity as the value a descriptor carries it as.

    The bounds are text. The canonical encoding writes whole numbers only, and a range written
    as a rounding of itself would name a different range, so each bound is written the way this
    kernel writes every number it has to put inside a digest.
    """
    return {
        "grader_id": grade.grader_id,
        "grader_version": grade.grader_version,
        "stand_in": grade.stand_in,
        "score_component": grade.score_component,
        "score_places": grade.score_places,
        "public_components": [
            {
                "name": number.name,
                "minimum": number_text(number.minimum),
                "maximum": number_text(number.maximum),
                "places": number.places,
            }
            for number in grade.public_components
        ],
    }


def manifest_fields(manifest: SourceArtifactManifest) -> Dict[str, Any]:
    """Return one descriptor as the value its canonical bytes are written from."""
    return {
        "schema_version": manifest.schema_version,
        "bundle_digest": manifest.bundle_digest,
        "environment_task_id": manifest.environment_task_id,
        "source_attempt_id": manifest.source_attempt_id,
        "source_seal_id": manifest.source_seal_id,
        "execution_ordinal": manifest.execution_ordinal,
        "canonicalization_version": manifest.canonicalization_version,
        "canonical_submission": blob_fields(manifest.canonical_submission),
        "kernel_submission_digest": manifest.kernel_submission_digest,
        "renderer_configuration": manifest.renderer_configuration,
        "receipt_contract": contract_fields(manifest.receipt_contract),
        "grade_identity": grade_fields(manifest.grade_identity),
        "cells": {kind: blob_fields(reference) for kind, reference in manifest.cells.items()},
        "bank_filing_digest": manifest.bank_filing_digest,
        "bank_cell_digests": dict(manifest.bank_cell_digests),
        "pair_parity": {
            "body_size": manifest.pair_parity.body_size,
            "encoded_body_bytes": manifest.pair_parity.encoded_body_bytes,
            "masked_body_sha256": manifest.pair_parity.masked_body_sha256,
        },
    }


def manifest_preimage(manifest: SourceArtifactManifest) -> bytes:
    """Return the canonical bytes one descriptor commits to.

    They are the object as well as the preimage. The bytes are installed as one blob and the
    commitment is that blob's plain SHA-256, so a reader holding the commitment can fetch what
    was committed rather than being told that something was hashed.
    """
    return canonical_json(manifest_fields(manifest))


def source_commitment(manifest: SourceArtifactManifest) -> str:
    """Return the digest that names one descriptor's canonical bytes."""
    return sha256(manifest_preimage(manifest)).hexdigest()


def check_receipt_contract(contract: ReceiptContract) -> None:
    """Refuse a contract that could not hold, before anything is published under it.

    The geometry is checked as arithmetic rather than against a body: every span sits inside one
    row line, the spans of a row are disjoint and in order, and the last row ends inside the
    declared body. A contract that fails any of those describes a mask that would read bytes the
    body does not have, and a mask nobody can expand is not a comparison.
    """
    _token("contract_id", contract.contract_id)
    if contract.match_group != KERNEL_MATCH_GROUP:
        raise WireFormatError(
            f"this build publishes every body in {KERNEL_MATCH_GROUP!r} and the contract "
            f"{contract.contract_id} declares {contract.match_group!r}"
        )
    _contract_cells(contract)
    _count("body_size", contract.body_size, minimum=1)
    if contract.body_encoding != BODY_ENCODING:
        raise WireFormatError(
            f"a receipt body is written as {BODY_ENCODING!r}, and the contract "
            f"{contract.contract_id} declares {contract.body_encoding!r}: byte offsets into a "
            "body only mean anything where a character is a byte"
        )
    if contract.media_type != BODY_MEDIA_TYPE:
        raise WireFormatError(
            f"a receipt body travels as {BODY_MEDIA_TYPE!r}, and the contract "
            f"{contract.contract_id} declares {contract.media_type!r}"
        )
    _geometry(contract)
    if tuple(contract.manifest_schema_versions) != (ARTIFACT_SCHEMA_VERSION,):
        raise WireFormatError(
            f"this build writes and reads {ARTIFACT_SCHEMA_VERSION!r}, and the contract "
            f"{contract.contract_id} admits {list(contract.manifest_schema_versions)}"
        )
    _text("renderer_configuration", contract.renderer_configuration)


def _contract_cells(contract: ReceiptContract) -> None:
    """Refuse a contract whose pair is not the registered artifact pair, one cell each.

    The pair is named rather than described. Two policies deliver an environment's own committed
    bytes, one declares the graded cell and the other the placebo, and a contract admitting
    anything else would be admitting a source into a route that renders its body from a
    projection instead of resolving it.
    """
    cells = tuple(contract.cells)
    if len(cells) != 2 or len(set(cells)) != 2:
        raise WireFormatError(
            f"a receipt contract admits one graded cell and one placebo cell, and "
            f"{contract.contract_id} declares {len(cells)}"
        )
    for digest, cell in cells:
        policy = POLICIES.get(digest)
        if policy is None or policy.policy_name not in SELECTABLE:
            raise WireFormatError(
                f"the contract {contract.contract_id} admits a cell under a policy this build "
                "does not implement or does not let a generation select"
            )
        if digest not in ARTIFACT_POLICY_DIGESTS:
            raise WireFormatError(
                f"a receipt contract admits the cells of "
                f"{GRADED_RECEIPT_ARTIFACT_V1.policy_name} and "
                f"{PLACEBO_RECEIPT_ARTIFACT_V1.policy_name}, and the contract "
                f"{contract.contract_id} admits one under {policy.policy_name}"
            )
        if cell not in policy.cells:
            raise WireFormatError(
                f"{policy.policy_name} declares the cells {list(policy.cells)}, and the contract "
                f"{contract.contract_id} admits {cell!r} under it"
            )
    if tuple(sorted(cell for _digest, cell in cells)) != tuple(sorted(ELIGIBLE_CELLS)):
        raise WireFormatError(
            f"a receipt contract admits the cells {sorted(ELIGIBLE_CELLS)}, and "
            f"{contract.contract_id} declares {sorted(cell for _d, cell in cells)}"
        )


def _geometry(contract: ReceiptContract) -> None:
    """Refuse a slot layout that does not expand to spans inside the declared body."""
    _count("rows_start", contract.rows_start, minimum=0)
    _count("row_line_width", contract.row_line_width, minimum=1)
    _count("row_count", contract.row_count, minimum=1)
    spans = tuple(contract.slot_spans)
    if not spans:
        raise WireFormatError(
            f"the contract {contract.contract_id} registers no slot, and a pair whose cells may "
            "differ nowhere is not a pair this build publishes"
        )
    reached = 0
    for low, high in spans:
        _count("slot span start", low, minimum=0)
        _count("slot span end", high, minimum=0)
        if low < reached or high <= low or high >= contract.row_line_width:
            raise WireFormatError(
                f"the contract {contract.contract_id} registers the slot span ({low}, {high}), "
                f"and a row line is {contract.row_line_width} bytes counting its own newline: "
                "spans are disjoint, in order, and inside the line"
            )
        reached = high
    end = contract.rows_start + contract.row_count * contract.row_line_width
    if end > contract.body_size:
        raise WireFormatError(
            f"the contract {contract.contract_id} registers {contract.row_count} rows ending at "
            f"{end} in a body of {contract.body_size} bytes"
        )


def check_receipt_contracts(
    contracts: Sequence[ReceiptContract],
    *,
    profile: str,
    dispositions: Sequence[PayloadDisposition],
    families: Sequence[MatchedFamily] = (),
    source: str = "",
) -> None:
    """Admit one generation's receipt contracts, and the rows that name them.

    Admission renders nothing. What the shipped path did for a matched family was render every
    cell with no grade over two invented submission digests and require one length, which a body
    cut from a filing cannot answer: the same registration produces a different count for every
    legal filing. So a receipt row declares a contract instead, and what is checked here is the
    declaration: the pair it admits, the size and encoding its bodies come to, the geometry its
    mask is expanded from, the descriptor versions it admits, and the renderer it publishes.

    ``source`` is the bank source digest those contracts are admitted over. A generation that
    declares a contract declares it, because it is what a descriptor's bundle digest is compared
    against and the environment's configuration digest is a hash the comparison cannot be
    recovered from. A generation that declares no contract declares no source either: the field
    is inside the configuration hash only where a contract is, so a value carried outside both
    would be part of what a generation is and not part of what it hashes.

    A withholding's association is validated against the declared contract ids and nothing else.
    A withholding names why and nothing else, its policy and its cell refused outright by the
    shape check, so the membership test a family makes, which asks the row's policy and cell to
    be one of the declared cells, would reject every withheld binding there is. What a
    withholding says by naming a contract is which capture it was made under, and an attempt that
    captures and delivers nothing still has to say that.
    """
    declared: Dict[str, ReceiptContract] = {}
    family_ids = {family.family_id for family in families}
    for contract in contracts:
        if profile != EXPERIMENT:
            raise PolicyViolation(
                f"a receipt contract is admitted by an {EXPERIMENT} generation, and this "
                f"{profile} generation declares {contract.contract_id!r}"
            )
        try:
            check_receipt_contract(contract)
        except WireFormatError as error:
            raise PolicyViolation(str(error)) from error
        if contract.contract_id in declared:
            raise PolicyViolation(
                f"the receipt contract {contract.contract_id} is declared twice, and a contract "
                "is one admitted shape"
            )
        if contract.contract_id in family_ids:
            raise PolicyViolation(
                f"{contract.contract_id!r} is declared as a receipt contract and as a matched "
                "family, and a row names one of the two in one column"
            )
        declared[contract.contract_id] = contract
    _admitted_source(source, declared)
    for row in dispositions:
        policy = POLICIES.get(row.policy_digest or "")
        contract = declared.get(row.family_id) if row.family_id else None
        if policy is not None and policy.exposure == ARTIFACT and row.kind == DELIVER:
            if contract is None:
                raise PolicyViolation(
                    f"the row for attempt {row.attempt_id} delivers {policy.policy_name}, and a "
                    "row that delivers a committed source cell names the contract that source "
                    "was captured under"
                )
        if contract is None or row.kind == WITHHOLD:
            continue
        if (row.policy_digest, row.cell) not in contract.cells:
            raise PolicyViolation(
                f"the row for attempt {row.attempt_id} names the receipt contract "
                f"{contract.contract_id}, and what it delivers is not one of that contract's "
                "cells"
            )


def _admitted_source(source: str, declared: Mapping[str, ReceiptContract]) -> None:
    """Refuse a bank source that no contract is admitted over, and a contract with none."""
    if declared and not source:
        raise PolicyViolation(
            f"this generation declares the receipt contracts {sorted(declared)} and names no "
            "bank source they are admitted over, so a descriptor's bundle digest would be "
            "compared against nothing"
        )
    if source and not declared:
        raise PolicyViolation(
            "this generation names a bank source for receipt contracts and declares none, and "
            "the source is inside what a generation is only where a contract is"
        )
    if source and _DIGEST.fullmatch(source) is None:
        raise PolicyViolation(
            f"a bank source is named by its own digest, and this generation names {source!r}"
        )


def check_source_artifact(manifest: SourceArtifactManifest) -> None:
    """Refuse a descriptor that is not one this build admits, field by field.

    What is checked here is the descriptor against itself and against the contract it carries: a
    version, a kind, a digest's shape, a size, an encoding and a media type. What is not checked
    here is whether a digest names the bytes it claims to, because that is a question for
    whoever holds the bytes. The bank's digests and the blob addresses are the same shape and
    are told apart by recomputing both from the body, which publication does while it holds all
    three cells.
    """
    contract = manifest.receipt_contract
    check_receipt_contract(contract)
    if manifest.schema_version not in contract.manifest_schema_versions:
        raise WireFormatError(
            f"the contract {contract.contract_id} admits "
            f"{list(contract.manifest_schema_versions)} and this descriptor is written as "
            f"{manifest.schema_version!r}"
        )
    _digest("bundle_digest", manifest.bundle_digest)
    _text("environment_task_id", manifest.environment_task_id)
    require_opaque_id("source_attempt_id", manifest.source_attempt_id)
    _digest("source_seal_id", manifest.source_seal_id)
    _count("execution_ordinal", manifest.execution_ordinal, minimum=0)
    _text("canonicalization_version", manifest.canonicalization_version)
    _submission(manifest.canonical_submission)
    _digest("kernel_submission_digest", manifest.kernel_submission_digest)
    if manifest.renderer_configuration != contract.renderer_configuration:
        raise WireFormatError(
            f"the contract {contract.contract_id} publishes what "
            f"{contract.renderer_configuration!r} renders, and this descriptor was rendered by "
            f"{manifest.renderer_configuration!r}"
        )
    try:
        check_grade(manifest.grade_identity)
    except PolicyViolation as violation:
        raise WireFormatError(str(violation)) from violation
    _cells(manifest, contract)
    _digest("bank_filing_digest", manifest.bank_filing_digest)
    _kinds("bank_cell_digests", manifest.bank_cell_digests)
    for kind in CELL_KINDS:
        _digest(f"the bank digest for the {kind} cell", manifest.bank_cell_digests[kind])
    parity = manifest.pair_parity
    if parity.body_size != contract.body_size:
        raise WireFormatError(
            f"the contract {contract.contract_id} fixes a body at {contract.body_size} bytes and "
            f"this descriptor's parity evidence is for {parity.body_size}"
        )
    _count("encoded_body_bytes", parity.encoded_body_bytes, minimum=1)
    _digest("masked_body_sha256", parity.masked_body_sha256)
    # And the whole of it has to have canonical bytes, because the commitment is those bytes.
    # Every check above is about one field; this is the one check about the record. A value the
    # canonical encoding cannot write leaves a descriptor that reads back and cannot be committed
    # to, and finding that out where the commitment is taken is an activation that fails rather
    # than an attempt that ends, so it is found here, at the boundary that refuses with a reason.
    manifest_preimage(manifest)


def _submission(reference: BlobRef) -> None:
    """Refuse a canonical submission reference that is not one an object could answer."""
    _digest("the canonical submission digest", reference.sha256)
    _count("the canonical submission size", reference.size, minimum=1)
    if reference.media_type != BODY_MEDIA_TYPE:
        raise WireFormatError(
            f"the canonical submission is {BODY_MEDIA_TYPE!r} and this descriptor names it as "
            f"{reference.media_type!r}"
        )


def _cells(manifest: SourceArtifactManifest, contract: ReceiptContract) -> None:
    """Refuse a cell map that is not exactly three bodies of the registered shape."""
    _kinds("cells", manifest.cells)
    for kind in CELL_KINDS:
        reference = manifest.cells[kind]
        _digest(f"the {kind} cell's digest", reference.sha256)
        if reference.size != contract.body_size:
            raise WireFormatError(
                f"the contract {contract.contract_id} fixes a body at {contract.body_size} bytes "
                f"and the {kind} cell is {reference.size}"
            )
        if reference.media_type != contract.media_type:
            raise WireFormatError(
                f"the contract {contract.contract_id} publishes {contract.media_type!r} and the "
                f"{kind} cell is named as {reference.media_type!r}"
            )


def _kinds(name: str, value: Mapping[str, Any]) -> None:
    """Refuse a map that is not keyed by exactly the three cells one source holds."""
    if not isinstance(value, Mapping) or tuple(sorted(value)) != tuple(sorted(CELL_KINDS)):
        raise WireFormatError(
            f"{name} holds exactly the cells {sorted(CELL_KINDS)}, and this one holds "
            f"{sorted(value) if isinstance(value, Mapping) else type(value).__name__}"
        )


def _token(name: str, value: Any) -> str:
    """Return one identifier, refusing anything that is not a token this build writes."""
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise WireFormatError(f"{name} is named by a token, and this one is {value!r}")
    return value


def _text(name: str, value: Any) -> str:
    """Return one non-empty string the canonical encoding can write.

    A lone surrogate is refused here rather than where the bytes are taken. It is a string
    Python holds and JSON has no encoding for, so a descriptor carrying one reads back, passes
    every field check and then has no canonical bytes at all: the commitment over it would raise
    at the encoder, which is not a boundary anything is recorded at, and the attempt would fail
    an activation nobody can explain rather than end with a reason written against it.
    """
    if not isinstance(value, str) or not value:
        raise WireFormatError(f"{name} is a non-empty string, and this one is {value!r}")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise WireFormatError(
            f"{name} is written in Unicode scalar values, and this one holds a surrogate, which "
            "the canonical encoding has no way to write"
        )
    return value


def _count(name: str, value: Any, *, minimum: int = 0) -> int:
    """Return one whole number the canonical encoding can write.

    A boolean is refused rather than read as one or nothing. It is an integer in Python and a
    different value on the wire, and a size that arrived as ``true`` is a decoder being helpful
    about a field nobody meant to fill.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise WireFormatError(f"{name} is a whole number, and this one is {value!r}")
    if not minimum <= value <= _SAFE_INTEGER_MAX:
        raise WireFormatError(
            f"{name} lies between {minimum} and {_SAFE_INTEGER_MAX}, and this one is {value}"
        )
    return value


def _digest(name: str, value: Any) -> str:
    """Return one digest, refusing anything that does not name bytes."""
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise WireFormatError(
            f"{name} is {_SHA256_HEX} lower-case hexadecimal characters, and this one is "
            f"{value!r}"
        )
    return value


def _exactly(name: str, value: Any, declared: Sequence[str]) -> Mapping[str, Any]:
    """Return one mapping whose keys are exactly the ones a record declares.

    A missing name and an unknown one are refused together. Reading a record with a name this
    build does not know is reading a record written by something else, and filling in a name it
    did not carry is inventing the field that was left out.
    """
    if not isinstance(value, Mapping):
        raise WireFormatError(f"{name} is written as an object, and this one is {value!r}")
    if tuple(sorted(str(key) for key in value)) != tuple(sorted(declared)):
        raise WireFormatError(
            f"{name} carries exactly {sorted(declared)}, and this one carries "
            f"{sorted(str(key) for key in value)}"
        )
    return value


def read_blob_ref(name: str, value: Any) -> BlobRef:
    """Read one blob reference out of a mapping, or refuse the shape that is not one."""
    fields = _exactly(name, value, ("sha256", "size", "media_type"))
    return BlobRef(
        sha256=_digest(f"{name} sha256", fields["sha256"]),
        size=_count(f"{name} size", fields["size"], minimum=1),
        media_type=_text(f"{name} media_type", fields["media_type"]),
    )


def read_grade_identity(value: Any) -> GradeIdentity:
    """Read one grade identity back, bounds and all, out of the text they were written as."""
    fields = _exactly(
        "grade_identity",
        value,
        (
            "grader_id",
            "grader_version",
            "stand_in",
            "score_component",
            "score_places",
            "public_components",
        ),
    )
    published = fields["public_components"]
    if not isinstance(published, (list, tuple)):
        raise WireFormatError("a grade identity's published numbers are written as an array")
    numbers: List[PublishedNumber] = []
    for entry in published:
        declared = _exactly(
            "a published number", entry, ("name", "minimum", "maximum", "places")
        )
        numbers.append(
            PublishedNumber(
                name=_text("a published number's name", declared["name"]),
                minimum=_bound("minimum", declared["minimum"]),
                maximum=_bound("maximum", declared["maximum"]),
                places=_count("a published number's places", declared["places"]),
            )
        )
    if not isinstance(fields["stand_in"], bool):
        raise WireFormatError("a grade identity says whether its grader is a stand-in")
    return GradeIdentity(
        grader_id=_text("grader_id", fields["grader_id"]),
        grader_version=_text("grader_version", fields["grader_version"]),
        stand_in=fields["stand_in"],
        score_component=_text("score_component", fields["score_component"]),
        score_places=_count("score_places", fields["score_places"]),
        public_components=tuple(numbers),
    )


def _bound(name: str, value: Any) -> float:
    """Return one declared bound, read back from the exact text it was written as."""
    text = _text(f"a published number's {name}", value)
    try:
        bound = float(text)
    except ValueError as error:
        raise WireFormatError(
            f"a published number's {name} is written as a number, and this one is {text!r}"
        ) from error
    if number_text(bound) != text:
        raise WireFormatError(
            f"a published number's {name} is written as the shortest text that reads back as "
            f"itself, and this one is {text!r}"
        )
    return bound


def read_receipt_contract(value: Any) -> ReceiptContract:
    """Read one contract back out of a mapping, and refuse a shape that is not one.

    The contract is validated after it is read rather than as it is read, so a mapping that is
    the wrong shape is refused as a decoding and a contract that is the wrong contract is
    refused as a declaration. The two failures say different things and the message says which.
    """
    fields = _exactly(
        "a receipt contract",
        value,
        (
            "contract_id",
            "match_group",
            "cells",
            "body_size",
            "body_encoding",
            "media_type",
            "rows_start",
            "row_line_width",
            "row_count",
            "slot_spans",
            "manifest_schema_versions",
            "renderer_configuration",
        ),
    )
    contract = ReceiptContract(
        contract_id=_text("contract_id", fields["contract_id"]),
        match_group=_text("match_group", fields["match_group"]),
        cells=tuple(
            (_digest("a contract cell's policy", digest), _text("a contract cell", cell))
            for digest, cell in _pairs("cells", fields["cells"])
        ),
        body_size=_count("body_size", fields["body_size"], minimum=1),
        body_encoding=_text("body_encoding", fields["body_encoding"]),
        media_type=_text("media_type", fields["media_type"]),
        rows_start=_count("rows_start", fields["rows_start"]),
        row_line_width=_count("row_line_width", fields["row_line_width"], minimum=1),
        row_count=_count("row_count", fields["row_count"], minimum=1),
        slot_spans=tuple(
            (_count("a slot span start", low), _count("a slot span end", high))
            for low, high in _pairs("slot_spans", fields["slot_spans"])
        ),
        manifest_schema_versions=tuple(
            _text("an admitted schema version", version)
            for version in _array("manifest_schema_versions", fields["manifest_schema_versions"])
        ),
        renderer_configuration=_text(
            "renderer_configuration", fields["renderer_configuration"]
        ),
    )
    check_receipt_contract(contract)
    return contract


def _array(name: str, value: Any) -> Sequence[Any]:
    """Return one array, refusing a scalar or a mapping written where a list belongs."""
    if not isinstance(value, (list, tuple)):
        raise WireFormatError(f"{name} is written as an array, and this one is {value!r}")
    return value


def _pairs(name: str, value: Any) -> List[Tuple[Any, Any]]:
    """Return one array of two-element arrays, refusing anything else."""
    out: List[Tuple[Any, Any]] = []
    for entry in _array(name, value):
        pair = _array(f"an entry of {name}", entry)
        if len(pair) != 2:
            raise WireFormatError(
                f"an entry of {name} is two values, and this one is {len(pair)}"
            )
        out.append((pair[0], pair[1]))
    return out


def read_source_artifact(value: Any) -> SourceArtifactManifest:
    """Read one descriptor back out of a mapping, and refuse a shape that is not one.

    This is the boundary a stored record crosses on its way back into a typed value. Every name
    the descriptor declares has to be there, no name it does not declare may be, and every value
    has to already be what its field is: nothing is coerced, so a size written as text and a
    digest written as an object are refusals rather than values that survive one more layer.
    """
    fields = _exactly(
        "a source artifact manifest",
        value,
        (
            "schema_version",
            "bundle_digest",
            "environment_task_id",
            "source_attempt_id",
            "source_seal_id",
            "execution_ordinal",
            "canonicalization_version",
            "canonical_submission",
            "kernel_submission_digest",
            "renderer_configuration",
            "receipt_contract",
            "grade_identity",
            "cells",
            "bank_filing_digest",
            "bank_cell_digests",
            "pair_parity",
        ),
    )
    _kinds("cells", fields["cells"])
    parity = _exactly(
        "pair_parity",
        fields["pair_parity"],
        ("body_size", "encoded_body_bytes", "masked_body_sha256"),
    )
    manifest = SourceArtifactManifest(
        schema_version=_text("schema_version", fields["schema_version"]),
        bundle_digest=_text("bundle_digest", fields["bundle_digest"]),
        environment_task_id=_text("environment_task_id", fields["environment_task_id"]),
        source_attempt_id=_text("source_attempt_id", fields["source_attempt_id"]),
        source_seal_id=_text("source_seal_id", fields["source_seal_id"]),
        execution_ordinal=_count("execution_ordinal", fields["execution_ordinal"]),
        canonicalization_version=_text(
            "canonicalization_version", fields["canonicalization_version"]
        ),
        canonical_submission=read_blob_ref(
            "canonical_submission", fields["canonical_submission"]
        ),
        kernel_submission_digest=_text(
            "kernel_submission_digest", fields["kernel_submission_digest"]
        ),
        renderer_configuration=_text(
            "renderer_configuration", fields["renderer_configuration"]
        ),
        receipt_contract=read_receipt_contract(fields["receipt_contract"]),
        grade_identity=read_grade_identity(fields["grade_identity"]),
        cells={
            kind: read_blob_ref(f"the {kind} cell", fields["cells"][kind])
            for kind in CELL_KINDS
        },
        bank_filing_digest=_text("bank_filing_digest", fields["bank_filing_digest"]),
        bank_cell_digests={
            kind: _text(f"the bank digest for the {kind} cell", digest)
            for kind, digest in _exactly(
                "bank_cell_digests", fields["bank_cell_digests"], CELL_KINDS
            ).items()
        },
        pair_parity=PairParity(
            body_size=_count("body_size", parity["body_size"], minimum=1),
            encoded_body_bytes=_count(
                "encoded_body_bytes", parity["encoded_body_bytes"], minimum=1
            ),
            masked_body_sha256=_text("masked_body_sha256", parity["masked_body_sha256"]),
        ),
    )
    check_source_artifact(manifest)
    return manifest


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BODY_ENCODING",
    "BODY_MEDIA_TYPE",
    "CELL_KINDS",
    "ELIGIBLE_CELLS",
    "GRADED_CELL",
    "MANIFEST_MEDIA_TYPE",
    "MASK_FILL",
    "ORACLE_CELL",
    "PLACEBO_CELL",
    "PairParity",
    "ReceiptContract",
    "SourceArtifactManifest",
    "blob_fields",
    "check_receipt_contract",
    "check_receipt_contracts",
    "check_source_artifact",
    "contract_fields",
    "encoded_body_bytes",
    "grade_fields",
    "manifest_fields",
    "manifest_preimage",
    "mask_spans",
    "masked_body",
    "payload_wire_count",
    "read_blob_ref",
    "read_grade_identity",
    "read_receipt_contract",
    "read_source_artifact",
    "source_commitment",
]
