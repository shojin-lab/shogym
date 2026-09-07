"""How a receipts attempt ends under the durable stream, and what it is worth.

The stream's stand-in grade computes from the shape of a filing: a terminal call carrying
something scores one and an empty one scores nothing. A filing is a table of answers, so under
the stand-in every attempt that files at all is worth one however many records it got right, and
the fraction it did get right, which is the quantity this environment exists to measure, reaches
the stream nowhere. That number cannot be published to an agent as a grade, because it is not one.

So this port replaces the environment's half of the terminal. The seal does what a v1 episode's
finalize does, once and under the seal id: it canonicalizes the filing, renders the fork through
the bank's own path, and scores what the parser made, and every later call under that key reads
what the first one wrote. The grade is a projection of that record, which is what makes a retry
return the first seal's numbers rather than a second reading of a world that may no longer be
there.

The seal reads the filing out of the terminal call rather than out of the world, and that is why
these two Activities are written here rather than assembled from :func:`terminal_for`. A receipts
world holds a table and an answer key, and it never holds the agent's filing at all: the terminal
call does not reach the environment under this protocol, so what an attempt filed exists in the
call and in the record this seal writes from it. The world is still resolved, because it is what
says which instance the filing is against, and a seal that reaches no world refuses rather than
scoring an empty table.

The headline is ``component_score``: the fraction of records the filing got right, equal weight
per row, over the printed row count, which is the number a v1 run seals. It goes out at six
places because that is where the scorer rounds it. The resolution is the scorer's own rather than
this port's second opinion, so the number a generation commits is the number a v1 run reports for
the same filing and not a coarser one that would have to be explained.

Beside it a body may say whether the whole table was right, and nothing else. The per-row
verdicts and the corrections are the receipt's, and the receipt is not this environment's to
deliver: an experiment decides which cell of the fork a branch is served, and an environment that
handed the grade back at the terminal would be putting a receipt in every arm, including the one
that is meant to be empty. So the acknowledgement carries the submission digest and the version it
was captured under and no verdict, and the three cells stay under the seal.

The seal id is what those cells are keyed by, and it is the only name here that identifies one of
them. A public attempt id does not: two executions of one attempt share it, and each has a filing
and a fork of its own, so cells looked up under it would be whichever execution wrote first. A
seal id is minted from the hidden execution, the ordinal of that execution and the attempt
together, which is exactly the identity a rendered cell belongs to. Where the records go is the
bank's own fork store, beside the committed fork whose bytes they are, so a Worker that replaced
the one which sealed reads what was sealed and a process that stopped did not take it with it.
A generation that declares a receipt contract gets one thing more from the same seal: the source
is published. The three cells are installed in the run's blob store, a descriptor over them says
which world, which filing and which contract they belong to, and the digest of that descriptor's
canonical bytes is the commitment two derivations of one source share. Publication is a
verification and a copy rather than a second render. The capture is the first winning write, the
bodies come out of it, every digest is recomputed from those bytes, the pair is checked against
the slots the contract registers, and a retry that finds the record already there installs the
same bytes again under the same names. A retry that finds a record belonging to another attempt,
another renderer, another contract or another grade identity refuses by name instead, because a
source that can be overwritten is a provenance claim nobody can check.

The horizon is the floor, and that is a decision rather than an omission. A graded horizon files
the terminal for the attempt as its last step commits, and the filing it makes is the world the
attempt left: the gateway writes no arguments into it, and it refuses a generation whose terminal
declares any. ``submit_filing`` declares the filing itself, so there is nothing at this horizon
for a gateway to file, and an attempt that reaches it filed nothing. That is what the floor
records, and it is the number a v1 episode reports for the same ending.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shogym.envs._grading import (
    GRADED,
    SEALED,
    CaptureStore,
    WorldRoute,
    check_recovery_token,
    configuration_digest as digest_of,
    encoded,
    graded_result,
    installed,
    recovery_token,
    refusal,
    terminal_activities,
)
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import streams
from shogym.envs.receipts.protocol import Instance
from shogym.envs.receipts.receipt_ast import ReceiptAST, wrapper_lines
from shogym.serve.protocol_v2.artifact import (
    ARTIFACT_SCHEMA_VERSION,
    BODY_ENCODING,
    BODY_MEDIA_TYPE,
    CELL_KINDS,
    ELIGIBLE_CELLS,
    GRADED_CELL,
    MANIFEST_MEDIA_TYPE,
    PLACEBO_CELL,
    PairParity,
    ReceiptContract,
    SourceArtifactManifest,
    check_receipt_contract,
    check_source_artifact,
    contract_fields,
    encoded_body_bytes,
    grade_fields,
    manifest_fields,
    manifest_preimage,
    mask_spans,
    masked_body,
    read_grade_identity,
    read_receipt_contract,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import BlobRef, FilesystemBlobStore
from shogym.serve.protocol_v2.errors import WireFormatError
from shogym.serve.protocol_v2.identity import submission_digest
from shogym.serve.protocol_v2.kernel.messages import (
    GradeAttemptInput,
    GradeAttemptResult,
    SealAttemptInput,
    SealAttemptResult,
)
from shogym.serve.protocol_v2.policy import (
    KERNEL_MATCH_GROUP,
    GradeIdentity,
    PublishedNumber,
)

#: The version this environment declares for the submission its terminal captures. It names what
#: goes into the canonical text and what does not, so a run recorded under it is comparable with
#: another run recorded under it and with nothing else. The name is this environment's own and
#: names no platform, because the version is in the acknowledgement that answers a filing and a
#: model reads it there at the end of every task it finishes.
CANONICALIZATION_VERSION = "receipts.1"

#: The argument the score terminal is filed with. The stream holds a terminal call to the names
#: the tool declares, so this is where the filing arrives and the only place it exists.
FILING_ARGUMENT = "filing"

#: What the three cells this seal rendered are held under, inside the record the seal wrote.
CELLS = "cells"

#: How fine the headline is. The scorer rounds the fraction of records the filing got right to six
#: places, so six is what the number means: declaring fewer would publish a rounding of the
#: environment's own number rather than the number, and a run under this grade and a v1 run over
#: the same filing would then report two different scores for one filing.
_SCORE_PLACES = 6

#: What this environment's grader is. The score is the scorer's own verdict over the filing rather
#: than a fact about its shape, which is what lets a generation over this environment publish it.
#: The roster is the one number a body may print beside that score: the whole table was right or it
#: was not, which is a whole number between nothing and one, and a roster entry says so by
#: declaring no decimal places at all. What the receipt says row by row is not on the roster and
#: never becomes a number a body prints: an arm that carries the verdicts carries them as the
#: environment's own rendered cell, at the byte count the envelope fixes.
RECEIPTS_GRADE = GradeIdentity(
    grader_id="receipts-grade-v2",
    grader_version="1",
    stand_in=False,
    score_component="component_score",
    score_places=_SCORE_PLACES,
    public_components=(
        PublishedNumber(name="solved", minimum=0.0, maximum=1.0),
    ),
)


def receipts_terminal(
    route: WorldRoute, *, store: CaptureStore
) -> Tuple[str, List[Any]]:
    """The version this environment declares, and the Activities that end an attempt in it.

    ``route`` says which world an attempt was worked in, and it is asked when a seal has to read
    one rather than now: these Activities are registered once and a generation may serve several
    tasks, each in a world of its own.

    ``store`` is where the records go, and there is no default. What a seal writes here is the
    cells a later arm delivers as well as the numbers, so a caller has to say where they are kept
    rather than have somewhere chosen for it: a store this process holds and nothing else would
    make the evidence of an attempt a property of the Worker that happened to seal it. The
    environment answers with a directory in the bank's own fork store.
    """

    def seal(request: SealAttemptInput) -> SealAttemptResult:
        """Capture what this attempt filed, publish its source, or return what is held.

        The capture is the first winning write and everything after it is a read, a hash check
        or a byte-for-byte reinstall. So a retry that arrives with another filing, or after the
        world it was worked in has gone, publishes the source that was committed rather than
        recomputing one, and a retry that arrives against a record belonging to another attempt
        or another seal is refused by name instead of publishing it under this one's identity.

        A contract this environment cannot publish under is refused before the world is read at
        all, because the capture is bound to the contract at its first write: a geometry that
        expands to a mask outside the body, or a call with nowhere to install three cells, is a
        source that could never be published and is not one to commit and then refuse.

        A record belonging to another attempt or another seal is refused by name before anything
        is installed. The held capture is what the submission text is written from, so a call
        that read one somebody else committed would put that capture's bytes in this run's store
        on its way to the refusal: a retry compares the identity it was called with against the
        identity that was held, and refuses before returning or reinstalling anything.
        """
        if request.canonicalization_version != CANONICALIZATION_VERSION:
            raise refusal(
                f"this world is captured as {CANONICALIZATION_VERSION!r} and the generation was "
                f"started as {request.canonicalization_version!r}",
                "CanonicalizationMismatch",
            )
        contract = request.receipt_contract
        if contract is not None:
            _admissible(request, contract)
        state = store.once(request.seal_id, SEALED, lambda: _filed(route, request))
        if contract is not None:
            _committed(request, state)
        text = encoded(
            {
                "canonicalization_version": CANONICALIZATION_VERSION,
                "submission": _submission(state),
            }
        )
        submission = installed(request.blob_root, text)
        manifest = None if contract is None else _published(request, contract, state, text)
        return SealAttemptResult(
            attempt_id=request.attempt_id,
            seal_id=request.seal_id,
            canonicalization_version=CANONICALIZATION_VERSION,
            canonical_submission_text=text,
            canonical_submission=submission,
            environment_recovery_token=recovery_token(request.seal_id, state),
            # The descriptor travels as the value its canonical bytes are written from rather than
            # as a typed field. What decides whether a mapping is a descriptor is the workflow's
            # own strict reader, at the boundary that records a refusal, and a typed field here
            # would put that decision in a converter that fails an activation instead.
            source_artifact=None if manifest is None else manifest_fields(manifest),
            source_commitment="" if manifest is None else source_commitment(manifest),
        )

    def grade(request: GradeAttemptInput) -> GradeAttemptResult:
        """Score the capture, or return the score this key was already given."""
        state = store.held(request.seal_id, SEALED)
        if state is None:
            raise refusal(
                f"this machine holds nothing sealed under {request.seal_id}, so there is nothing "
                "here to grade",
                "NoSealedCapture",
            )
        check_recovery_token(request.seal_id, state, request.environment_recovery_token)
        verdict = store.once(request.seal_id, GRADED, lambda: _verdict(state))
        return graded_result(request, verdict=verdict, grade=RECEIPTS_GRADE)

    return CANONICALIZATION_VERSION, terminal_activities(seal=seal, grade=grade)


def cells_for(store: CaptureStore, seal_id: str) -> Optional[Dict[str, str]]:
    """The cells one seal rendered, or ``None`` where nothing was sealed under that key.

    The key is the seal rather than the attempt, because the seal is what one render belongs to:
    two executions of one attempt carry one public id between them and a filing and a fork each,
    so an answer looked up by attempt would be whichever of them wrote first. The bytes are the
    fork's own, as the seal committed them, and nothing here renders: a cell built a second time
    is how two branches of one fork come to differ.
    """
    held = store.held(seal_id, SEALED)
    if held is None:
        return None
    cells = held.get(CELLS) or {}
    return {str(kind): str(text) for kind, text in cells.items()}


def _filed(route: WorldRoute, request: SealAttemptInput) -> Dict[str, Any]:
    """Read the world this attempt was worked in, and what the filing in this call comes to.

    The two refusals are separate on purpose. A seal that resolves to no world at all reached a
    process that never served this attempt, which is what a resumed generation sees for every
    attempt; a seal that resolves to a world whose session is gone reached the right process after
    it let the world go. Both would otherwise be a filing scored against an instance nobody looked
    up, and a zero there is a grade: it says the agent got no record right, and what happened is
    that nobody read the table it was answering.
    """
    world = route(request.attempt_id)
    if world is None:
        raise refusal(
            f"attempt {request.attempt_id} was worked in no world this process opened, so there "
            "is nothing here to seal",
            "NoWorldHere",
        )
    env, session_id = world
    filed = env.sealed_filing(session_id, request.native_arguments.get(FILING_ARGUMENT))
    if filed is None:
        raise refusal(
            f"the world attempt {request.attempt_id} was worked in has been let go, so the "
            "instance this filing answers cannot be read: a score for a table nobody read is not "
            "a score",
            "NoSessionHere",
        )
    contract = request.receipt_contract
    if contract is None:
        return filed
    return {**filed, **_bound(request, contract, filed)}


def _bound(
    request: SealAttemptInput, contract: ReceiptContract, filed: Dict[str, Any]
) -> Dict[str, Any]:
    """What the source record gains at its first write, beside the cells and the numbers.

    These are the bindings a later derivation is checked against, and they are written once,
    with the bytes, by the call that read the world. Origin comes from this record afterwards
    and never from a replacement world, from the renderer constants a later build happens to
    hold, or from the arguments a retry echoed back: a retry that disagrees with what is here is
    refused rather than allowed to overwrite it.
    """
    return {
        "source_attempt_id": request.attempt_id,
        "source_seal_id": request.seal_id,
        "execution_ordinal": int(request.execution_ordinal),
        "canonicalization_version": CANONICALIZATION_VERSION,
        "renderer_configuration": bank_mod.RENDERER_CONFIGURATION,
        "receipt_contract": contract_fields(contract),
        "grade_identity": grade_fields(RECEIPTS_GRADE),
        "pair_parity": _parity_fields(_pair_parity(contract, _bodies(filed))),
    }


def _submission(state: Dict[str, Any]) -> Dict[str, Any]:
    """What the agent filed: the parser's canonical reading of it, and nothing the scorer made.

    The canonical text is what the digest covers and what a payload renderer is handed, so the
    numbers a grade is made of are not in it, and neither is the key: a run composed to withhold
    its score would otherwise be handing the renderer the score in the field beside the one it
    withheld, and one composed to serve a placebo would be handing it the answers.
    """
    return {"filing": str(state["filing"])}


def _verdict(state: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict this port commits, out of the record the seal wrote.

    ``decode_state`` is how the filing read rather than whether anything went wrong. A filing
    nothing scorable could be made of and a filing that was read and got no record right are both
    worth nothing and are different facts about the attempt, and both are answers the grade stands
    on: neither is retried and the score behind them is the score.

    What the cells say stays out of it. The verdict is installed as the run's own evidence, where
    a harness can resolve it and a renderer cannot, and the numbers on it are counts of what the
    reading had to decide: nothing here names an option of the convention or a record's correct
    value in any form, because a fork's own cells are where those live.
    """
    return {
        "component_score": float(state["component_score"]),
        "solved": 1.0 if bool(state["solved"]) else 0.0,
        "rows_filed": float(state["rows_filed"]),
        "rows_omitted": float(state["rows_omitted"]),
        "no_filing": state["no_filing"],
        "task_id": str(state["task_id"]),
        "filing_digest": str(state["filing_digest"]),
        "cell_digests": dict(state["cell_digests"]),
        "decode_state": "ambiguous_zero" if state["no_filing"] else "decoded",
    }


def receipt_contract(
    instance: Instance,
    side: str,
    *,
    contract_id: str,
    cells: Sequence[Tuple[str, str]],
) -> ReceiptContract:
    """Return the contract this family's cells are admitted and published under.

    The geometry is read off the registered envelope and the family's own row count, never off a
    body somebody rendered: the envelope is fixed before launch and does not move with the drawn
    convention, so the mask a pair is compared under is a property of the registration rather
    than of the draw a seal happened to see. The wrapper is measured through the serializer's own
    lines, so a layout change moves this with it instead of leaving a constant behind.

    ``cells`` is the pair of policy digests and cells this contract admits, which the caller
    holds because it is the caller that registered them.
    """
    task = instance.side(side)
    envelope = instance.envelope
    wrapper = ReceiptAST(kind=GRADED_CELL, task_id=task.task_id, row_count=task.n_rows)
    rows_start = len("\n".join(wrapper_lines(wrapper))) + 1 + len(envelope.header_line()) + 1
    contract = ReceiptContract(
        contract_id=contract_id,
        match_group=KERNEL_MATCH_GROUP,
        cells=tuple((digest, cell) for digest, cell in cells),
        body_size=envelope.size,
        body_encoding=BODY_ENCODING,
        media_type=BODY_MEDIA_TYPE,
        rows_start=rows_start,
        row_line_width=envelope.row_line_width,
        row_count=task.n_rows,
        slot_spans=tuple(envelope.slot_span(spec.name) for spec in envelope.slots),
        manifest_schema_versions=(ARTIFACT_SCHEMA_VERSION,),
        renderer_configuration=bank_mod.RENDERER_CONFIGURATION,
    )
    check_receipt_contract(contract)
    return contract


def _admissible(request: SealAttemptInput, contract: ReceiptContract) -> None:
    """Refuse a call that could never publish a source, before one is captured under it.

    Each is a property of the call rather than of the bytes, so each is answered before the world
    is read: a contract this build cannot publish under, a renderer other than the one this
    environment's cells come out of, and a call with no store to install them in. The ordinal is
    here for the same reason. It is what a descriptor carries in place of the hidden execution
    id, and a call that did not say which execution is asking is one whose descriptor could not
    be filled in.
    """
    try:
        check_receipt_contract(contract)
    except WireFormatError as error:
        raise refusal(str(error), "ReceiptContractRefused") from error
    if contract.renderer_configuration != bank_mod.RENDERER_CONFIGURATION:
        raise refusal(
            f"this environment renders {bank_mod.RENDERER_CONFIGURATION!r} and the contract "
            f"{contract.contract_id} publishes what {contract.renderer_configuration!r} renders",
            "ContractDrift",
        )
    if request.blob_root is None:
        raise refusal(
            "a generation that publishes its own bodies needs a store to install them in, and "
            "this one was given none",
            "UnavailableEvidence",
        )
    if request.execution_ordinal < 0:
        raise refusal(
            "a published source names the execution its seal id was computed under, and this "
            "call names none",
            "SourceArtifactRefused",
        )


def _published(
    request: SealAttemptInput,
    asked: ReceiptContract,
    state: Dict[str, Any],
    text: str,
) -> SourceArtifactManifest:
    """Verify the committed source, install its bodies, and return the descriptor over them.

    Everything the descriptor says comes out of the record the capture wrote. The contract and
    the grade identity are read back out of it rather than copied from the arguments this call
    arrived with, which is what makes a retry a reading of what was committed instead of a
    second commitment agreeing with whatever asked for it last.

    The order is the guarantee. The record is checked against the call, the bytes are checked
    against the digests committed with them, the pair is checked against the registered slots,
    and only then are the three bodies and the bound manifest installed and read back. A crash
    anywhere in it leaves a record a later call verifies and reinstalls byte for byte, because
    nothing here renders and nothing here writes to the record.

    ``asked`` is the contract this call carries, already admitted by :func:`_admissible`. What is
    published is the one the capture holds, which the record is checked against first. The
    capture's own identity is compared before this runs at all, by the caller, because the
    submission bytes are installed between the two and a wrong source must be refused before
    anything of it is written back.
    """
    contract = _held_contract(state)
    if contract != asked:
        raise refusal(
            f"the source held here was published under the contract {contract.contract_id} as it "
            f"stood then, and this call carries {asked.contract_id} as it stands now",
            "ContractDrift",
        )
    grade = _held_grade(state)
    if grade != RECEIPTS_GRADE:
        raise refusal(
            f"the source held here was captured under {grade.grader_id}/{grade.grader_version} "
            f"and this build grades by {RECEIPTS_GRADE.grader_id}/{RECEIPTS_GRADE.grader_version}",
            "ContractDrift",
        )
    bodies = _bodies(state)
    bank_digests = _bank_digests(state, bodies)
    parity = _pair_parity(contract, bodies)
    if _parity_fields(parity) != state.get("pair_parity"):
        raise refusal(
            "the cells held under this seal no longer come to the parity evidence committed "
            "with them",
            "CorruptEvidence",
        )
    blob_root = request.blob_root
    # A call with nowhere to install three cells is refused by :func:`_admissible` before the
    # world is read at all, and this runs only under a contract that call carried, so a store
    # is what an admitted call names.
    assert blob_root is not None, (
        "a generation that publishes its own bodies is admitted only with a store to install "
        "them in"
    )
    store = FilesystemBlobStore(Path(blob_root))
    cells = {
        kind: store.put(bodies[kind].encode(BODY_ENCODING), media_type=BODY_MEDIA_TYPE)
        for kind in CELL_KINDS
    }
    raw = text.encode("utf-8")
    # Every value the record holds is put in as it was found, and the descriptor's own check is
    # what says whether it is a digest, a count or a version. Coercing here would make a size
    # written as text into a size, which is the repair this whole record refuses to make.
    try:
        held = {
            "bundle_digest": state["source_digest"],
            "environment_task_id": state["task_id"],
            "source_attempt_id": state["source_attempt_id"],
            "source_seal_id": state["source_seal_id"],
            "execution_ordinal": state["execution_ordinal"],
            "canonicalization_version": state["canonicalization_version"],
            "renderer_configuration": state["renderer_configuration"],
            "bank_filing_digest": state["filing_digest"],
        }
    except KeyError as error:
        raise refusal(
            f"the source held here carries no {error.args[0]}, so there is nothing to publish "
            "under this seal",
            "CorruptEvidence",
        ) from error
    manifest = SourceArtifactManifest(
        schema_version=ARTIFACT_SCHEMA_VERSION,
        bundle_digest=held["bundle_digest"],
        environment_task_id=held["environment_task_id"],
        source_attempt_id=held["source_attempt_id"],
        source_seal_id=held["source_seal_id"],
        execution_ordinal=held["execution_ordinal"],
        canonicalization_version=held["canonicalization_version"],
        # The submission is installed by the seal itself, under the media type a record it wrote
        # travels as. The store addresses an object by the hash of its bytes and by nothing
        # else, so what the descriptor names here is the same object read as the text a
        # derivation is handed.
        canonical_submission=BlobRef(
            sha256=sha256(raw).hexdigest(), size=len(raw), media_type=BODY_MEDIA_TYPE
        ),
        kernel_submission_digest=submission_digest(
            request.attempt_id, request.native_terminal_name, raw
        ),
        renderer_configuration=held["renderer_configuration"],
        receipt_contract=contract,
        grade_identity=grade,
        cells=cells,
        bank_filing_digest=held["bank_filing_digest"],
        bank_cell_digests=bank_digests,
        pair_parity=parity,
    )
    try:
        check_source_artifact(manifest)
    except WireFormatError as error:
        raise refusal(str(error), "SourceArtifactRefused") from error
    bound = store.put(manifest_preimage(manifest), media_type=MANIFEST_MEDIA_TYPE)
    missing = store.unverified(
        [bound.sha256, manifest.canonical_submission.sha256]
        + [reference.sha256 for reference in cells.values()]
    )
    if missing:
        raise refusal(
            f"the store cannot produce {len(missing)} of the objects this source was published "
            "over, so there is no source here to return",
            "UnavailableEvidence",
        )
    return manifest


def _committed(request: SealAttemptInput, state: Dict[str, Any]) -> None:
    """Refuse a call that is not the one the held source belongs to.

    A record whose attempt and seal are not this call's is another attempt's source, and
    publishing it here would put one capture's cells under another's identity however well every
    hash inside it checks out. A record written by another renderer is this build having changed
    underneath a source it already committed, and the answer to that is to refuse rather than to
    reissue the descriptor with this build's own values in it.
    """
    held_attempt = state.get("source_attempt_id")
    held_seal = state.get("source_seal_id")
    if held_attempt != request.attempt_id or held_seal != request.seal_id:
        raise refusal(
            f"the source held here was captured for attempt {held_attempt!r} under seal "
            f"{held_seal!r}, and this call is attempt {request.attempt_id!r} under "
            f"{request.seal_id!r}",
            "WrongSource",
        )
    if state.get("renderer_configuration") != bank_mod.RENDERER_CONFIGURATION:
        raise refusal(
            f"the source held here was rendered by {state.get('renderer_configuration')!r} and "
            f"this build renders {bank_mod.RENDERER_CONFIGURATION!r}",
            "ContractDrift",
        )


def _held_contract(state: Dict[str, Any]) -> ReceiptContract:
    """The contract the capture was published under, read back out of the record."""
    try:
        return read_receipt_contract(state.get("receipt_contract"))
    except WireFormatError as error:
        raise refusal(str(error), "CorruptEvidence") from error


def _held_grade(state: Dict[str, Any]) -> GradeIdentity:
    """The grade identity the capture was taken under, read back out of the record."""
    try:
        return read_grade_identity(state.get("grade_identity"))
    except WireFormatError as error:
        raise refusal(str(error), "CorruptEvidence") from error


def _bodies(state: Dict[str, Any]) -> Dict[str, str]:
    """The three cells the capture holds, refusing a set that is not three printable cells."""
    cells = state.get(CELLS) or {}
    if sorted(cells) != sorted(CELL_KINDS):
        raise refusal(
            f"a source holds the cells {sorted(CELL_KINDS)} and this one holds {sorted(cells)}",
            "CorruptEvidence",
        )
    bodies: Dict[str, str] = {}
    for kind in CELL_KINDS:
        text = cells[kind]
        if not isinstance(text, str) or not text.isascii():
            raise refusal(
                f"the {kind} cell held here is not {BODY_ENCODING} text, so the offsets a mask "
                "is expanded to are not byte offsets into it",
                "CorruptEvidence",
            )
        bodies[kind] = text
    return bodies


def _bank_digests(state: Dict[str, Any], bodies: Dict[str, str]) -> Dict[str, str]:
    """The bank's digests for the held cells, recomputed from the held bytes.

    They are the bank's own length-prefixed hashes and never blob addresses, and they are
    recomputed rather than copied: a record whose committed digest no longer names the bytes
    beside it is a source this seal will not publish, whichever of the two was moved.
    """
    recorded = state.get("cell_digests") or {}
    if sorted(recorded) != sorted(CELL_KINDS):
        raise refusal(
            f"a source records a bank digest per cell and this one records {sorted(recorded)}",
            "CorruptEvidence",
        )
    for kind in CELL_KINDS:
        if streams.digest(bodies[kind].encode(BODY_ENCODING)) != recorded[kind]:
            raise refusal(
                f"the {kind} cell held here does not hash to the bank digest committed with it",
                "CorruptEvidence",
            )
    return {kind: str(recorded[kind]) for kind in CELL_KINDS}


def _pair_parity(contract: ReceiptContract, bodies: Dict[str, str]) -> PairParity:
    """Check the two eligible cells against each other, and return what that proved.

    Three equalities and one size, in the order a reader should read them. Every cell is exactly
    the registered body size, so a body that is short is refused rather than measured. The two
    eligible bodies are byte identical outside the registered slots, under the mask this
    contract's own geometry expands to and never one a candidate proposed. And the two come to
    one encoded count, which is what makes their wire counts equal under any common wrapper.

    The oracle is held to the size and to nothing else. It shares the envelope and has no rows
    to align, so a comparison against it would be a kind check rather than a parity check.
    """
    spans = mask_spans(contract)
    raw = {kind: bodies[kind].encode(BODY_ENCODING) for kind in CELL_KINDS}
    for kind in CELL_KINDS:
        if len(raw[kind]) != contract.body_size:
            raise refusal(
                f"the contract {contract.contract_id} fixes a body at {contract.body_size} bytes "
                f"and the {kind} cell is {len(raw[kind])}",
                "PairRefused",
            )
    masked = masked_body(raw[GRADED_CELL], spans)
    if masked_body(raw[PLACEBO_CELL], spans) != masked:
        raise refusal(
            "the graded and placebo cells differ outside the slots this contract registers, so "
            "they are two bodies rather than a pair",
            "PairRefused",
        )
    counts = {kind: encoded_body_bytes(bodies[kind]) for kind in ELIGIBLE_CELLS}
    if counts[GRADED_CELL] != counts[PLACEBO_CELL]:
        raise refusal(
            f"the graded cell encodes to {counts[GRADED_CELL]} bytes and the placebo cell to "
            f"{counts[PLACEBO_CELL]}, so the two would not come to one wire count",
            "PairRefused",
        )
    return PairParity(
        body_size=contract.body_size,
        encoded_body_bytes=counts[GRADED_CELL],
        masked_body_sha256=sha256(masked).hexdigest(),
    )


def _parity_fields(parity: PairParity) -> Dict[str, Any]:
    """One pair's evidence as the record holds it."""
    return {
        "body_size": parity.body_size,
        "encoded_body_bytes": parity.encoded_body_bytes,
        "masked_body_sha256": parity.masked_body_sha256,
    }


def configuration_digest(
    *, genre: str, side: Optional[str], source: str, dealable: bool
) -> str:
    """Return the digest of what this environment is configured as.

    The source decides which instances could be dealt and under which conventions, because a
    bundle is frozen at its digest and a bank is named by its own bytes, and the genre decides
    what a family means. A generation resumed against a different one of those is a different
    measurement, and it is refused rather than scored against a draw nobody made for it.

    Whether the source was admitted is here as well as which source it is. An unbundled bank and a
    verified bundle are two different claims about the same instances, and a run that resumed
    across that line would be reporting scores from a development draw under a generation composed
    over a dealt one.

    Which sibling a task is comes from the position it was served at rather than from here, so a
    generation over a whole family declares nothing about siblings at all: what would otherwise
    happen is that A and B could never sit in one generation, because the world opened for the
    second position would carry a configuration the generation was not started as and be refused.
    The narrowing an environment can still be given is here, because an environment narrowed to
    one sibling serves a different roster and a position in it means something else.
    """
    return digest_of(
        {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "genre": genre,
            "side": side,
            "source_digest": source,
            "dealable": bool(dealable),
        }
    )


__all__ = [
    "CANONICALIZATION_VERSION",
    "CELLS",
    "FILING_ARGUMENT",
    "RECEIPTS_GRADE",
    "cells_for",
    "configuration_digest",
    "receipt_contract",
    "receipts_terminal",
]
