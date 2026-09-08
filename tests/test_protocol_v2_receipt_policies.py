"""The pair a committed receipt is delivered through, and the record that admits one.

Two policies ship for this and neither is a rename of anything. The shipped concealing pair
renders one string for both of its cells, so a run recorded under its graded label proves that no
receipt was delivered; a body cut from an environment's own committed bytes is a different claim
and gets its own registration, its own exposure class and its own resolver. The four policies
that were here before keep the exact bytes their digests were taken over, which is what these
tests pin.

The admission record is the other half. A matched family fixes a byte count, which a body cut
from a filing cannot answer: the same registration comes to a different count for every legal
filing. So a receipt row declares a contract instead, naming it in the column a row names its
family in, and admission checks the declaration rather than rendering anything. A withholding
names one too, because a capture is validated against a contract whether or not anything is
delivered from it.

Nothing here reaches an environment. The bodies are three short rows written by hand, which is
enough for the arithmetic and the refusals; what a real ledger's cells come to is measured where
the ledger is.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, List

import pytest

pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.serve.protocol_v2.artifact import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    BODY_ENCODING,
    BODY_MEDIA_TYPE,
    GRADED_CELL,
    PLACEBO_CELL,
    ReceiptContract,
    check_receipt_contracts,
    encoded_body_bytes,
    mask_spans,
    masked_body,
    payload_wire_count,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    GeneratePayloadBundleInput,
    SelectedSourceReference,
    StreamStart,
    TaskItem,
    TerminalTool,
    configuration_hash,
    generate_payload_bundle_activity,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    ARTIFACT,
    BLINDED_RECEIPT_V1,
    DELIVER,
    EXPERIMENT,
    GRADED_RECEIPT_ARTIFACT_V1,
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    HONEST_V1,
    KERNEL_MATCH_GROUP,
    LEGACY_PLACEHOLDER_V1,
    NO_GRADE_NUMBERS,
    ORDINARY,
    PLACEBO_RECEIPT_ARTIFACT_V1,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_RECEIPT_V1,
    POLICIES,
    REGISTERED,
    SELECTABLE,
    WITHHOLD,
    GradeIdentity,
    MatchedFamily,
    PayloadDisposition,
    PolicyProvenance,
    PolicyViolation,
    PublishedNumber,
    descriptor_digests,
    policy_digest,
    policy_preimage,
    render_body,
    roster_digest,
)

ATTEMPT = "00000000000000000000000000000100"
OTHER_ATTEMPT = "00000000000000000000000000000104"
SOURCE = "a" * 64
COMMITMENT = "b" * 64

# The bytes a run of this file stands in for a rendered cell with: three printed rows, one slot
# per row, and a newline the row width counts. It is not a ledger receipt and does not pretend to
# be one; what it is is a body the registered geometry expands over exactly.
GRADED_BODY = "abGRDwxyz\n" * 3
PLACEBO_BODY = "abPLCwxyz\n" * 3

# The digests the four shipped policies were recorded under. They are written out rather than
# recomputed, because what these pin is that the bytes behind them did not move when a fifth and
# a sixth policy were registered beside them.
SHIPPED_DIGESTS = {
    "honest-v1": "a6e008f006181cfeaa1bfc6f551f23a5edb1b3e61b9da9ac6736e880aed815b7",
    "blinded-receipt-v1": "1be372db7584e6c439d6b0ede693b6ae41623228bbade90258d1dde7ed6a876c",
    "placebo-receipt-v1": "cca134385708f37a2b558d6d3afeb18eb3036de5b84d12040cc2fedbb0688754",
    "legacy-placeholder-v1": "db17860afe262630d1e19cd267a310a2bef22467a5d1922b592becae8edfed2f",
}

# What an experiment generation over a real grader hashes when it declares none of this. The
# value is a regression pin: a generation that admits no receipt contract and names no bank
# source writes exactly the document it wrote before either could be declared.
WITHOUT_RECEIPTS = "e70cde635c01ad43248729c3407cd093f66ab6d9538d2b7099b6a4a46a33cb35"

REAL_GRADE = GradeIdentity(
    grader_id="ledger-grade",
    grader_version="1",
    stand_in=False,
    score_component="component_score",
    score_places=6,
    public_components=(PublishedNumber(name="solved", minimum=0.0, maximum=1.0),),
)


def oid(value: int) -> str:
    return f"{value:032x}"


def contract(**changes: Any) -> ReceiptContract:
    """The registered shape this file's bodies are admitted under."""
    declared = ReceiptContract(
        contract_id="ledger_receipt",
        match_group=KERNEL_MATCH_GROUP,
        cells=(
            (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
            (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
        ),
        body_size=len(GRADED_BODY),
        body_encoding=BODY_ENCODING,
        media_type=BODY_MEDIA_TYPE,
        rows_start=0,
        row_line_width=10,
        row_count=3,
        slot_spans=((2, 5),),
        manifest_schema_versions=(ARTIFACT_SCHEMA_VERSION,),
        renderer_configuration="ledger-render-1",
    )
    return replace(declared, **changes) if changes else declared


def delivering(
    policy: Any, *, attempt_id: str = ATTEMPT, position: int = 0, family: str = "ledger_receipt"
) -> PayloadDisposition:
    """One obligation an experiment registered a committed cell against."""
    return PayloadDisposition(
        attempt_id=attempt_id,
        payload_position=position,
        kind=DELIVER,
        policy_digest=policy_digest(policy),
        cell=policy.cells[0],
        resolution_source=REGISTERED,
        family_id=family,
    )


def withholding(
    *, attempt_id: str = OTHER_ATTEMPT, position: int = 1, family: str = "ledger_receipt"
) -> PayloadDisposition:
    """One position that captures a source and delivers nothing against it."""
    return PayloadDisposition(
        attempt_id=attempt_id,
        payload_position=position,
        kind=WITHHOLD,
        reason="the arm delivers nothing at this position",
        resolution_source=REGISTERED,
        family_id=family,
    )


def admitting(
    rows: List[PayloadDisposition],
    *,
    contracts: Any = None,
    families: Any = (),
    profile: str = EXPERIMENT,
    source: str = SOURCE,
) -> None:
    """Admit one declaration the way a composition and the stream both admit it."""
    check_receipt_contracts(
        [contract()] if contracts is None else list(contracts),
        profile=profile,
        dispositions=rows,
        families=list(families),
        source=source,
    )


def start_of(
    rows: List[PayloadDisposition],
    *,
    contracts: Any = (),
    source: str = "",
    bodies: int = 1,
) -> StreamStart:
    """One generation, composed here so a test can say exactly what it declared."""
    tasks = [
        TaskItem(
            task_position=index,
            attempt_id=oid(0x100 + index * 4),
            task_message_id=oid(0x101 + index * 4),
            ack_message_id=oid(0x102 + index * 4),
            payload_position=index,
            payload_message_id=oid(0x103 + index * 4),
            body=f"file the report {index}",
        )
        for index in range(bodies)
    ]
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash="d" * 64,
        initial_cursor=oid(1),
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id="execution-1",
        canonicalization_version="kernel.1",
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=["answer"]
        ),
        tasks=tasks,
        profile=EXPERIMENT,
        grade=REAL_GRADE,
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_subject_of_this_run",
        ),
        receipt_contracts=list(contracts),
        receipt_source=source,
    )


# The pair, and the four policies it was registered beside.


def test_a_committed_receipt_is_delivered_under_its_own_pair_and_its_own_exposure() -> None:
    """Two registered records, one cell each, a resolver rather than a renderer's projection.

    The shipped concealing pair renders the same string for both of its cells, so a run recorded
    under its graded label is a run that delivered no receipt. These two are what a run that did
    deliver one records instead: the exposure is a third class, so the graded half does not
    inherit the blinded label's promise that its body holds no verdict, and the projection is two
    typed fields, so no grade, no manifest, no cell map and no canonical text reaches the
    resolver.
    """
    for policy in (GRADED_RECEIPT_ARTIFACT_V1, PLACEBO_RECEIPT_ARTIFACT_V1):
        assert POLICIES[policy_digest(policy)] is policy
        assert policy.policy_name in SELECTABLE
        assert policy.policy_version == "1"
        assert policy.exposure == ARTIFACT
        assert policy.renderer_id == "kernel-receipt-artifact-1"
        assert policy.renderer_version == "1"
        assert policy.resolver_id == "blob-store-artifact-1"
        assert policy.resolver_version == "1"
        assert policy.number_format == NO_GRADE_NUMBERS
        assert [field for field, _type in policy.projection] == ["source_cell", "contract"]
    assert GRADED_RECEIPT_ARTIFACT_V1.cells == (GRADED_CELL,)
    assert PLACEBO_RECEIPT_ARTIFACT_V1.cells == (PLACEBO_CELL,)
    assert GRADED_RECEIPT_ARTIFACT_V1_DIGEST != PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST
    assert len({policy_digest(policy) for policy in POLICIES.values()}) == len(POLICIES)


def test_the_policies_that_shipped_keep_the_bytes_their_digests_were_taken_over() -> None:
    """A resolver is written into a preimage only where there is one, and four digests are pinned.

    A member defaulting to the empty string would be a new key in the document every shipped
    policy's digest was taken over, and those digests are recorded in histories and in generation
    identities that no build may refuse. So the two new fields enter the preimage when they are
    not empty and not otherwise.
    """
    for policy in (HONEST_V1, BLINDED_RECEIPT_V1, PLACEBO_RECEIPT_V1, LEGACY_PLACEHOLDER_V1):
        assert policy_digest(policy) == SHIPPED_DIGESTS[policy.policy_name]
        written = json.loads(policy_preimage(policy).decode("utf-8"))
        assert "resolver_id" not in written and "resolver_version" not in written
        assert policy.resolver_id == "" and policy.resolver_version == ""
    resolved = json.loads(policy_preimage(GRADED_RECEIPT_ARTIFACT_V1).decode("utf-8"))
    assert resolved["resolver_id"] == "blob-store-artifact-1"
    assert resolved["resolver_version"] == "1"
    assert resolved["exposure"] == ARTIFACT


def test_a_generation_declaring_none_of_this_hashes_the_document_it_hashed_before() -> None:
    """The two new keys are written where they are declared and nowhere else.

    A generation that admits no receipt contract and names no bank source is the generation it
    was, so its identity is the value it was: a resume presents that hash, and a formula that
    grew a key would refuse every resume of every run recorded under it. Declaring either moves
    it, which is the other half of the same rule.
    """
    plain = start_of([delivering(GRADED_RECEIPT_ARTIFACT_V1, family="")])
    assert configuration_hash(plain) == WITHOUT_RECEIPTS
    declared = replace(
        plain,
        dispositions=[delivering(GRADED_RECEIPT_ARTIFACT_V1)],
        receipt_contracts=[contract()],
        receipt_source=SOURCE,
    )
    assert configuration_hash(declared) != WITHOUT_RECEIPTS
    assert configuration_hash(replace(declared, receipt_source="c" * 64)) != configuration_hash(
        declared
    )
    assert configuration_hash(
        replace(declared, receipt_contracts=[contract(body_size=31)])
    ) != configuration_hash(declared)


# The admission record.


def test_a_receipt_row_names_the_contract_its_capture_was_validated_against() -> None:
    """A delivering row names one, a withholding names one, and admission checks both.

    The two are validated differently and on purpose. A delivery is one of the contract's own
    cells, so the pair it names is checked against them. A withholding names why and nothing
    else, its policy and its cell refused outright by the shape check, so what is checked is that
    the id it names is a declared contract: the membership test a family makes would reject every
    withheld binding there is.
    """
    admitting([delivering(GRADED_RECEIPT_ARTIFACT_V1), withholding()])
    admitting([delivering(PLACEBO_RECEIPT_ARTIFACT_V1)])

    # A row delivering a committed cell and naming no contract is refused: the capture it was
    # cut from was validated against something, and the row is where that is written down.
    with pytest.raises(PolicyViolation, match="names the contract"):
        admitting([delivering(GRADED_RECEIPT_ARTIFACT_V1, family="")])
    with pytest.raises(PolicyViolation, match="names the contract"):
        admitting([delivering(GRADED_RECEIPT_ARTIFACT_V1, family="other_receipt")])

    # And a row naming a contract whose cells it does not deliver is refused as a cell of it.
    other = replace(delivering(GRADED_RECEIPT_ARTIFACT_V1), cell=PLACEBO_CELL)
    with pytest.raises(PolicyViolation, match="not one of that contract's cells"):
        admitting([other])
    scalar = replace(
        delivering(BLINDED_RECEIPT_V1), policy_digest=policy_digest(BLINDED_RECEIPT_V1)
    )
    with pytest.raises(PolicyViolation, match="not one of that contract's cells"):
        admitting([scalar])


def test_a_start_may_declare_two_contracts_and_a_withheld_attempt_binds_to_the_second() -> None:
    """Inferring the only contract stops working at the second declaration, so nothing infers.

    The case is the one a point measurement makes: one position delivers a cell of the first
    contract and another captures under the second and delivers nothing at all. Both rows say
    which contract they were captured under, in the same column, and the withheld one is admitted
    against the declared ids alone.
    """
    second = contract(contract_id="filler_receipt")
    rows = [
        delivering(GRADED_RECEIPT_ARTIFACT_V1),
        withholding(family="filler_receipt"),
    ]
    admitting(rows, contracts=[contract(), second])

    start = start_of(rows, contracts=[contract(), second], source=SOURCE, bodies=2)
    assert [row.family_id for row in start.dispositions] == ["ledger_receipt", "filler_receipt"]
    assert configuration_hash(start) != configuration_hash(
        replace(start, receipt_contracts=[contract()])
    )

    # A withholding bound to a name nothing declared is refused where a row's column is
    # resolved, which is the whole of what a withholding's association is checked against.
    from shogym.serve.protocol_v2.policy import check_families

    with pytest.raises(PolicyViolation, match="declares no such family"):
        check_families(
            [],
            profile=EXPERIMENT,
            dispositions=[withholding(family="nobody_receipt")],
            contract_ids=["ledger_receipt", "filler_receipt"],
        )


def test_a_contract_is_registered_material_and_carries_the_source_it_is_admitted_over() -> None:
    """An experiment declares one, and it declares the bank source the contract is admitted over.

    The source is on the start because nothing else there holds one: the environment's
    configuration digest hashes it together with four other values, so a descriptor's bundle
    digest compared against that hash would reject every legitimate source. It is inside what the
    generation is exactly where a contract is, so neither may be declared without the other.
    """
    rows = [delivering(GRADED_RECEIPT_ARTIFACT_V1)]
    with pytest.raises(PolicyViolation, match="names no bank source"):
        admitting(rows, source="")
    with pytest.raises(PolicyViolation, match="named by its own digest"):
        admitting(rows, source="the ledger bundle")
    with pytest.raises(PolicyViolation, match="declares none"):
        check_receipt_contracts([], profile=EXPERIMENT, dispositions=[], source=SOURCE)
    with pytest.raises(PolicyViolation, match="is admitted by an experiment generation"):
        admitting(rows, profile=ORDINARY)

    # And the two lists hold distinct names, because a row names one of them in one column.
    family = MatchedFamily(
        family_id="ledger_receipt",
        match_group=KERNEL_MATCH_GROUP,
        cells=(
            (policy_digest(BLINDED_RECEIPT_V1), GRADED_CELL),
            (policy_digest(PLACEBO_RECEIPT_V1), PLACEBO_CELL),
        ),
        visible_byte_count=100,
    )
    with pytest.raises(PolicyViolation, match="in one column"):
        admitting(rows, families=[family])
    with pytest.raises(PolicyViolation, match="declared twice"):
        admitting(rows, contracts=[contract(), contract()])


def test_a_matched_family_cannot_hold_a_cell_that_is_an_environments_own_body() -> None:
    """The family keeps its meaning and its byte count, which a receipt body cannot promise.

    A family's cells are held to one model-visible count declared before the run. A receipt body
    is a function of the filing, and the same registration comes to different counts for filings
    that differ only in an escaped character, so a family holding one would be promising
    something no fixed count can keep. The contract is what a receipt row declares instead.
    """
    from shogym.serve.protocol_v2.policy import check_families

    family = MatchedFamily(
        family_id="ledger_arm",
        match_group=KERNEL_MATCH_GROUP,
        cells=(
            (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
            (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
        ),
        visible_byte_count=2833,
    )
    with pytest.raises(PolicyViolation, match="declares a contract instead"):
        check_families(
            [family],
            profile=EXPERIMENT,
            dispositions=[delivering(GRADED_RECEIPT_ARTIFACT_V1, family="ledger_arm")],
        )


def test_an_artifact_policy_renders_no_body_and_has_no_scalar_one_to_fall_back_to() -> None:
    """There is nothing deterministic to rebuild, so the renderer refuses rather than substitutes.

    Falling back is the failure this refusal exists for: a scalar receipt served under a record
    that says a committed source was delivered is exactly the run whose transcript and whose
    record describe two different things.
    """
    with pytest.raises(PolicyViolation, match="no scalar one to fall back to"):
        render_body(
            GRADED_RECEIPT_ARTIFACT_V1,
            grade=None,
            payload_position=0,
            submission_digest="f" * 64,
        )


def test_both_halves_of_a_contract_are_installed_and_read_back_like_a_familys_cells(
    tmp_path: Path,
) -> None:
    """A leg serves one cell of a contract, and what the counterpart was allowed to say is kept.

    The preimage of the cell nobody served is a required object for the reason the served one is:
    a record that says which of two cells this leg was assigned, over a store that cannot produce
    what the other one was, is a comparison nobody can read afterwards.
    """
    from shogym.serve.protocol_v2.gateway import install_policies

    rows = [delivering(GRADED_RECEIPT_ARTIFACT_V1)]
    start = start_of(rows, contracts=[contract()], source=SOURCE)
    named = descriptor_digests(rows, [], [contract()])
    assert GRADED_RECEIPT_ARTIFACT_V1_DIGEST in named
    assert PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST in named

    blobs = FilesystemBlobStore(tmp_path / "blobs")
    install_policies(blobs, start)
    assert blobs.read(GRADED_RECEIPT_ARTIFACT_V1_DIGEST) == policy_preimage(
        GRADED_RECEIPT_ARTIFACT_V1
    )
    assert blobs.read(PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST) == policy_preimage(
        PLACEBO_RECEIPT_ARTIFACT_V1
    )


# The selected candidate route, at the boundary the store is read from.


def selection(store: FilesystemBlobStore, body: str, **changes: Any) -> SelectedSourceReference:
    """The one reference an artifact obligation is served, over a store holding ``body``."""
    installed = store.put(body.encode(BODY_ENCODING), media_type=BODY_MEDIA_TYPE)
    declared = SelectedSourceReference(
        source_commitment=COMMITMENT,
        cell=GRADED_CELL,
        contract_id=contract().contract_id,
        body_sha256=installed.sha256,
        body_size=installed.size,
        body_encoding=BODY_ENCODING,
        media_type=BODY_MEDIA_TYPE,
        masked_body_sha256=sha256(
            masked_body(body.encode(BODY_ENCODING), mask_spans(contract()))
        ).hexdigest(),
        slot_spans=mask_spans(contract()),
        blob_root=str(store.root),
    )
    return replace(declared, **changes) if changes else declared


def asking(selected: Any, **changes: Any) -> GeneratePayloadBundleInput:
    """One obligation asking for the cell it was selected for."""
    declared = GeneratePayloadBundleInput(
        attempt_id=ATTEMPT,
        payload_position=0,
        payload_message_id=oid(0x103),
        submission_digest="f" * 64,
        policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        cell=GRADED_CELL,
        selected=selected,
    )
    return replace(declared, **changes) if changes else declared


async def test_the_resolver_returns_the_committed_cell_exactly_as_it_was_installed(
    tmp_path: Path,
) -> None:
    """The body is the object under the reference, and the candidate says where it came from.

    Nothing here renders. What comes back is the stored bytes, its own hashes over them, and the
    echo a workflow holds to the selection: the source it names, the entry it resolved, and the
    resolver that read it.
    """
    store = FilesystemBlobStore(tmp_path / "blobs")
    selected = selection(store, GRADED_BODY)
    bundle = await generate_payload_bundle_activity(asking(selected))
    [candidate] = bundle.candidates
    assert candidate.body == GRADED_BODY
    assert candidate.cell == GRADED_CELL
    assert candidate.renderer_id == "kernel-receipt-artifact-1"
    assert candidate.renderer_version == "1"
    assert candidate.policy_digest == GRADED_RECEIPT_ARTIFACT_V1_DIGEST
    assert candidate.source_commitment == COMMITMENT
    assert candidate.body_reference == selected.body_sha256
    assert candidate.resolver_id == "blob-store-artifact-1"
    assert candidate.resolver_version == "1"
    # The inner hash is the committed entry, which is what makes the returned bytes the ones the
    # source committed rather than bytes that merely describe themselves correctly.
    assert candidate.inner_sha256 == selected.body_sha256
    assert candidate.visible_byte_count == payload_wire_count(
        payload_message_id=oid(0x103),
        attempt_id=ATTEMPT,
        encoded_body_bytes=encoded_body_bytes(GRADED_BODY),
    )


async def test_a_resolver_asked_for_a_body_it_cannot_produce_returns_none_at_all(
    tmp_path: Path,
) -> None:
    """A missing object, a store that is not there and bytes of the wrong shape are refusals.

    A missing cell never licenses a search for another one and never licenses a rendered body in
    its place, so each of these ends the attempt rather than producing something to serve.
    """
    store = FilesystemBlobStore(tmp_path / "blobs")
    selected = selection(store, GRADED_BODY)
    with pytest.raises(ApplicationError, match="no blob is installed") as missing:
        await generate_payload_bundle_activity(
            asking(replace(selected, body_sha256="9" * 64))
        )
    assert missing.value.type == "UnavailableEvidence"
    assert missing.value.non_retryable is True

    elsewhere = replace(selected, blob_root=str(tmp_path / "nowhere"))
    with pytest.raises(ApplicationError, match="no blob is installed") as absent:
        await generate_payload_bundle_activity(asking(elsewhere))
    assert absent.value.type == "UnavailableEvidence"

    with pytest.raises(ApplicationError, match="fixes a body at") as short:
        await generate_payload_bundle_activity(
            asking(replace(selected, body_size=len(GRADED_BODY) - 1))
        )
    assert short.value.type == "CorruptEvidence"


async def test_a_request_that_selected_nothing_is_refused_rather_than_rendered(
    tmp_path: Path,
) -> None:
    """There is no body to build in its place, and no scalar receipt to serve instead.

    The filing's text and the grade are refused as well. An artifact request carries neither, so
    a call arriving with one was composed by something that thinks this route renders, and a
    route that read past it would be answering a request this build does not make.
    """
    store = FilesystemBlobStore(tmp_path / "blobs")
    selected = selection(store, GRADED_BODY)
    with pytest.raises(ApplicationError, match="selected none") as none:
        await generate_payload_bundle_activity(asking(None))
    assert none.value.type == "SelectedSourceRequired"

    with pytest.raises(ApplicationError, match="the filing or the grade") as filed:
        await generate_payload_bundle_activity(
            asking(selected, canonical_submission_text="answer='42'")
        )
    assert filed.value.type == "PolicyViolation"

    # And a selection for another cell than the one this obligation was assigned is refused
    # before the store is read at all, whichever direction the two disagree in.
    with pytest.raises(ApplicationError, match="declares the cells") as crossed:
        await generate_payload_bundle_activity(
            asking(replace(selected, cell=PLACEBO_CELL))
        )
    assert crossed.value.type == "SelectedSourceMismatch"
    with pytest.raises(ApplicationError, match="declares the cells"):
        await generate_payload_bundle_activity(
            asking(replace(selected, cell=PLACEBO_CELL), cell=PLACEBO_CELL)
        )
    with pytest.raises(ApplicationError, match="implements no payload policy"):
        await generate_payload_bundle_activity(asking(selected, policy_digest="0" * 64))


def test_a_wire_count_is_the_empty_body_wrapper_plus_the_stored_count_and_never_the_quoted_one(
    tmp_path: Path,
) -> None:
    """The stored number leaves the two enclosing quotes out, so the wrapper is what carries them.

    Adding the quoted length instead overshoots by exactly two, which refuses a candidate every
    other check passed. The body here is three short rows; the same two bytes separate the two
    sums at any size.
    """
    quoted = len(json.dumps(GRADED_BODY).encode("utf-8"))
    stored = encoded_body_bytes(GRADED_BODY)
    assert stored == quoted - 2
    wrapper = payload_wire_count(
        payload_message_id=oid(0x103), attempt_id=ATTEMPT, encoded_body_bytes=0
    )
    assert (
        payload_wire_count(
            payload_message_id=oid(0x103), attempt_id=ATTEMPT, encoded_body_bytes=stored
        )
        == wrapper + stored
    )
    assert wrapper + quoted == wrapper + stored + 2
