"""The typed source descriptor: what it commits to, and what it refuses.

A descriptor is a provenance claim, so the whole of it has to be checkable by somebody who holds
nothing but the bytes. That is what these tests are about. The commitment is the plain hash of
the descriptor's own canonical bytes, those bytes read back into the same value, and every field
is held to being what the record says it is: a version this build admits, exactly the three
cells, digests that are digests, sizes and encodings the contract declared, and no name the
record does not declare.

The refusals are the point rather than the happy path. A descriptor half understood is worse than
none: it would let a run's provenance be read off fields nobody checked, and a decoder that fills
in a missing name or drops an unknown one is exactly how a record written by something else comes
to be read as one of ours.

Nothing here touches a store, a world or a workflow. What is under test is the record and the
arithmetic over it, so the bodies are invented and the digests are stand-ins: whether a digest
names the bytes it claims to is a question for whoever holds the bytes, and it is asked where
publication holds all three cells.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from typing import Any, Dict

import pytest

from shogym.serve.protocol_v2.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    BODY_MEDIA_TYPE,
    CELL_KINDS,
    MASK_FILL,
    PairParity,
    ReceiptContract,
    SourceArtifactManifest,
    check_receipt_contract,
    check_source_artifact,
    encoded_body_bytes,
    manifest_fields,
    manifest_preimage,
    mask_spans,
    masked_body,
    read_source_artifact,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import BlobRef
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.policy import (
    BLINDED_RECEIPT_V1_DIGEST,
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    HONEST_V1_DIGEST,
    LEGACY_PLACEHOLDER_V1_DIGEST,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_RECEIPT_V1_DIGEST,
    GradeIdentity,
    PublishedNumber,
)

RENDERER = "receipts-render-v1"


def contract(**changes: Any) -> ReceiptContract:
    """One registered contract, with whatever a test wants moved."""
    declared = ReceiptContract(
        contract_id="ledger_receipt",
        match_group="kernel",
        cells=(
            (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, "graded"),
            (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, "placebo"),
        ),
        body_size=2657,
        body_encoding="ascii",
        media_type=BODY_MEDIA_TYPE,
        rows_start=121,
        row_line_width=59,
        row_count=24,
        slot_spans=((40, 44), (46, 58)),
        manifest_schema_versions=(ARTIFACT_SCHEMA_VERSION,),
        renderer_configuration=RENDERER,
    )
    return replace(declared, **changes) if changes else declared


GRADE = GradeIdentity(
    grader_id="receipts-grade-v2",
    grader_version="1",
    stand_in=False,
    score_component="component_score",
    score_places=6,
    public_components=(PublishedNumber(name="solved", minimum=0.0, maximum=1.0),),
)


def manifest(**changes: Any) -> SourceArtifactManifest:
    """One descriptor over that contract, with whatever a test wants moved."""
    declared = SourceArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        bundle_digest="a" * 64,
        environment_task_id="03005e274867c52a",
        source_attempt_id="b" * 32,
        source_seal_id="c" * 64,
        execution_ordinal=0,
        canonicalization_version="shogym.receipts.1",
        canonical_submission=BlobRef(sha256="d" * 64, size=118, media_type=BODY_MEDIA_TYPE),
        kernel_submission_digest="e" * 64,
        renderer_configuration=RENDERER,
        receipt_contract=contract(),
        grade_identity=GRADE,
        cells={
            kind: BlobRef(sha256=str(index) * 64, size=2657, media_type=BODY_MEDIA_TYPE)
            for index, kind in enumerate(CELL_KINDS)
        },
        bank_filing_digest="f" * 64,
        bank_cell_digests={kind: f"{index + 4}" * 64 for index, kind in enumerate(CELL_KINDS)},
        pair_parity=PairParity(
            body_size=2657, encoded_body_bytes=2688, masked_body_sha256="8" * 64
        ),
    )
    return replace(declared, **changes) if changes else declared


def written(**changes: Any) -> Dict[str, Any]:
    """One descriptor as the mapping a record holds it as."""
    return json.loads(manifest_preimage(manifest(**changes)).decode("utf-8"))


def test_a_descriptor_commits_to_its_own_canonical_bytes_and_reads_back_from_them() -> None:
    """The commitment is the hash of the object, and the object is the preimage.

    There is no second digest here and nothing is hashed that a reader cannot fetch: the bytes
    the commitment names are the bytes a store holds, and reading them back gives the same value
    field for field rather than something equal in the fields somebody thought to compare.
    """
    declared = manifest()
    preimage = manifest_preimage(declared)
    assert source_commitment(declared) == sha256(preimage).hexdigest()
    assert read_source_artifact(json.loads(preimage.decode("utf-8"))) == declared

    # Canonical bytes: no whitespace, and keys in order, so two writers of one value write one
    # byte string and the commitment is a fact about the value rather than about the writer.
    assert b" " not in preimage.split(b'"canonicalization_version"')[0]
    encoded_order = list(json.loads(preimage.decode("utf-8")))
    assert encoded_order == sorted(manifest_fields(declared))


def test_a_descriptor_missing_a_name_or_carrying_an_unknown_one_is_refused() -> None:
    """Both directions of the same rule, because both are a record written by something else."""
    short = written()
    short.pop("bank_filing_digest")
    with pytest.raises(WireFormatError, match="carries exactly"):
        read_source_artifact(short)

    long = written()
    long["hidden_execution_id"] = "z" * 64
    with pytest.raises(WireFormatError, match="carries exactly"):
        read_source_artifact(long)


def test_a_descriptor_written_as_a_version_the_contract_does_not_admit_is_refused() -> None:
    """The admitted versions are the contract's, and the descriptor is held to them."""
    with pytest.raises(WireFormatError, match="written as"):
        check_source_artifact(manifest(schema_version="shogym.receipt-artifact.2"))


def test_a_cell_map_that_is_not_the_three_kinds_is_refused() -> None:
    """A source names all three cells: the two an arm may serve, and the one it never does."""
    cells = dict(manifest().cells)
    cells.pop("oracle")
    with pytest.raises(WireFormatError, match="holds exactly"):
        check_source_artifact(manifest(cells=cells))

    cells = dict(manifest().cells)
    cells["counterfactual"] = BlobRef(sha256="7" * 64, size=2657, media_type=BODY_MEDIA_TYPE)
    with pytest.raises(WireFormatError, match="holds exactly"):
        check_source_artifact(manifest(cells=cells))


def test_a_cell_whose_size_or_media_type_is_not_the_contract_s_is_refused() -> None:
    """The registered body size is what a cell is, so a shorter one is refused before it is read."""
    cells = dict(manifest().cells)
    cells["graded"] = BlobRef(sha256="0" * 64, size=2656, media_type=BODY_MEDIA_TYPE)
    with pytest.raises(WireFormatError, match="fixes a body at 2657 bytes"):
        check_source_artifact(manifest(cells=cells))

    cells = dict(manifest().cells)
    cells["placebo"] = BlobRef(sha256="1" * 64, size=2657, media_type="application/json")
    with pytest.raises(WireFormatError, match="publishes 'text/plain'"):
        check_source_artifact(manifest(cells=cells))


def test_a_digest_that_does_not_name_bytes_is_refused_wherever_one_is_declared() -> None:
    """A digest is 64 lower-case hexadecimal characters, in every field that holds one."""
    for field, value in (
        ("bundle_digest", "a" * 63),
        ("source_seal_id", "C" * 64),
        ("kernel_submission_digest", ""),
    ):
        with pytest.raises(WireFormatError, match="hexadecimal"):
            check_source_artifact(manifest(**{field: value}))

    with pytest.raises(WireFormatError, match="hexadecimal"):
        check_source_artifact(
            manifest(bank_cell_digests={kind: "not a digest" for kind in CELL_KINDS})
        )


def test_a_count_written_as_text_or_as_a_flag_is_refused() -> None:
    """Nothing is coerced. A size that arrived as text is a decoder being helpful about a field.

    A boolean is refused for the same reason and separately, because it is an integer in Python
    and a different value on the wire, so a count filled in as ``true`` would otherwise be read
    as one.
    """
    written_size = written()
    written_size["canonical_submission"]["size"] = "118"
    with pytest.raises(WireFormatError, match="whole number"):
        read_source_artifact(written_size)

    written_ordinal = written()
    written_ordinal["execution_ordinal"] = True
    with pytest.raises(WireFormatError, match="whole number"):
        read_source_artifact(written_ordinal)

    with pytest.raises(WireFormatError, match="lies between"):
        check_source_artifact(manifest(execution_ordinal=-1))


def test_a_textual_value_the_canonical_encoding_cannot_write_is_refused_by_the_reader() -> None:
    """A descriptor this build cannot take the bytes of is refused where it is read.

    A lone surrogate is a string Python holds and the canonical encoding has no way to write. A
    reader that asked only for a non-empty string would accept it, every other field would agree
    with every other field, and the commitment over it would then raise at the encoder, which is
    not a boundary anything is recorded at. So the reader refuses the value, and the record as a
    whole is required to have canonical bytes, which is what a commitment over it is.
    """
    for field in ("environment_task_id", "canonicalization_version", "bank_filing_digest"):
        written_text = written()
        written_text[field] = "\ud800"
        with pytest.raises(WireFormatError):
            read_source_artifact(written_text)

    # And the same holds of a value that reaches the check as a typed record rather than as a
    # mapping, which is the shape a restore and a derivation hand it in.
    with pytest.raises(WireFormatError, match="surrogate"):
        check_source_artifact(manifest(environment_task_id="\ud800"))
    with pytest.raises(WireFormatError, match="surrogate"):
        check_source_artifact(manifest(canonicalization_version="one\ud800"))


def test_a_contract_that_does_not_declare_the_shape_this_build_publishes_is_refused() -> None:
    """The encoding, the media type, the group and the admitted versions each have one value."""
    with pytest.raises(WireFormatError, match="written as 'ascii'"):
        check_receipt_contract(contract(body_encoding="utf-8"))
    with pytest.raises(WireFormatError, match="travels as 'text/plain'"):
        check_receipt_contract(contract(media_type="application/json"))
    with pytest.raises(WireFormatError, match="publishes every body in 'kernel'"):
        check_receipt_contract(contract(match_group="receipts"))
    with pytest.raises(WireFormatError, match="admits"):
        check_receipt_contract(contract(manifest_schema_versions=("shogym.receipt-artifact.1", "2")))
    with pytest.raises(WireFormatError, match="named by a token"):
        check_receipt_contract(contract(contract_id="Ledger Receipt"))


def test_a_contract_whose_slots_leave_the_line_or_overlap_is_refused() -> None:
    """A mask nobody can expand is not a comparison, so the geometry is checked as arithmetic."""
    with pytest.raises(WireFormatError, match="inside the line"):
        check_receipt_contract(contract(slot_spans=((40, 44), (43, 58))))
    with pytest.raises(WireFormatError, match="inside the line"):
        check_receipt_contract(contract(slot_spans=((40, 44), (46, 59))))
    with pytest.raises(WireFormatError, match="inside the line"):
        check_receipt_contract(contract(slot_spans=((44, 44),)))
    with pytest.raises(WireFormatError, match="registers no slot"):
        check_receipt_contract(contract(slot_spans=()))
    with pytest.raises(WireFormatError, match="rows ending at"):
        check_receipt_contract(contract(row_count=100))


def test_a_contract_admits_one_graded_cell_and_one_placebo_cell_and_nothing_else() -> None:
    """The pair is closed, both halves are the registered pair, and each declares its cell.

    The shipped concealing pair is refused here as well, and that is the point of registering a
    second one: both of those render one string for both of their cells, so a contract admitting
    them would admit a source into a route that never reads it.
    """
    with pytest.raises(WireFormatError, match="admits the cells of"):
        check_receipt_contract(
            contract(
                cells=(
                    (BLINDED_RECEIPT_V1_DIGEST, "graded"),
                    (PLACEBO_RECEIPT_V1_DIGEST, "placebo"),
                )
            )
        )
    with pytest.raises(WireFormatError, match="one graded cell and one placebo cell"):
        check_receipt_contract(
            contract(cells=((GRADED_RECEIPT_ARTIFACT_V1_DIGEST, "graded"),))
        )
    with pytest.raises(WireFormatError, match="admits"):
        check_receipt_contract(
            contract(
                cells=(
                    (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, "graded"),
                    (HONEST_V1_DIGEST, "honest"),
                )
            )
        )
    with pytest.raises(WireFormatError, match="does not implement or does not let"):
        check_receipt_contract(
            contract(
                cells=(
                    (LEGACY_PLACEHOLDER_V1_DIGEST, "graded"),
                    (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, "placebo"),
                )
            )
        )
    with pytest.raises(WireFormatError, match="does not implement or does not let"):
        check_receipt_contract(
            contract(
                cells=(("9" * 64, "graded"), (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, "placebo"))
            )
        )


def test_the_mask_is_one_span_per_slot_per_row_in_row_then_slot_order() -> None:
    """The expansion is arithmetic on the declaration, so it reads no body to produce spans."""
    spans = mask_spans(contract())
    assert len(spans) == 24 * 2
    assert spans[0] == (121 + 40, 121 + 44)
    assert spans[1] == (121 + 46, 121 + 58)
    assert spans[2] == (121 + 59 + 40, 121 + 59 + 44)
    assert spans[-1] == (121 + 23 * 59 + 46, 121 + 23 * 59 + 58)
    assert list(spans) == sorted(spans)

    # And the fill is the registered byte rather than a hole, because a body blanked with NULs
    # would satisfy the same equalities and commit a different source.
    body = bytes(range(121)) + b"X" * (2657 - 121)
    assert masked_body(body, spans)[121 + 40: 121 + 44] == MASK_FILL * 4
    assert masked_body(body, spans)[:121] == body[:121]


def test_a_body_counts_towards_a_wire_length_without_its_two_enclosing_quotes() -> None:
    """The stored count is the encoded length less the empty body's, which is the quotes.

    A body of one letter and a newline shows the two bytes the rule is about at a size anybody
    can check by hand: five characters quoted, three stored. An escaped character costs one more,
    which is why a fixed wrapper allowance rejects a legal filing that holds a quote.
    """
    assert encoded_body_bytes("X\n") == 3
    assert encoded_body_bytes("") == 0
    assert encoded_body_bytes('X"') == encoded_body_bytes("XX") + 1
    assert encoded_body_bytes("X\\") == encoded_body_bytes("XX") + 1


def test_a_published_number_s_bounds_are_written_as_text_and_read_back_exactly() -> None:
    """The canonical encoding holds whole numbers only, so a bound travels as its own shortest
    text and a bound written some other way is refused rather than rounded into place."""
    declared = manifest(
        grade_identity=replace(
            GRADE,
            public_components=(PublishedNumber(name="solved", minimum=0.0, maximum=0.5),),
        )
    )
    assert b'"maximum":"0.5"' in manifest_preimage(declared)
    assert read_source_artifact(json.loads(manifest_preimage(declared).decode("utf-8"))) == declared

    rounded = written()
    rounded["grade_identity"]["public_components"][0]["maximum"] = "1.0"
    with pytest.raises(WireFormatError, match="shortest text"):
        read_source_artifact(rounded)


def test_a_grade_identity_no_body_could_publish_is_refused() -> None:
    """The roster is the environment's declaration, and a descriptor carries the whole of it."""
    with pytest.raises(WireFormatError, match="decimal places"):
        check_source_artifact(manifest(grade_identity=replace(GRADE, score_places=9)))
