"""What a receipts seal publishes when its generation declares a contract, and what it refuses.

The seal already captured a filing, rendered the fork once and kept the cells under the seal id
that owns them. What this adds is publication: the three cells become objects in the run's own
store and a descriptor over them says which bundle, which task, which attempt and which contract
they belong to, with the digest of the descriptor's canonical bytes as the commitment both
eligible derivations of that source share.

The properties under test are the ones a provenance claim rests on rather than the ones a happy
path shows. Publication is a verification and a copy: the capture is the first winning write, and
a retry with another filing, or with no world left to read, republishes what was committed rather
than rendering a second source. A record that belongs to another attempt or another seal is
refused by name instead of being published under this call's identity. A renderer, a contract or a
grade identity that moved under a committed capture is refused rather than overwriting it. The
bank's digests and the store's addresses are separate namespaces and neither may stand in for the
other. And an interruption after any install is recoverable, because every step after the capture
is a read, a hash check or a byte-for-byte reinstall.

The Activities are driven as functions here, which is the whole of what these properties need: the
store is the environment's own, the world is a real served episode, and nothing about publication
depends on a workflow having asked for it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest

pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.envs.receipts import bank as bank_mod  # noqa: E402
from shogym.envs.receipts import streams  # noqa: E402
from shogym.envs.receipts import protocol_v2 as port  # noqa: E402
from shogym.envs.receipts.env_v1 import ReceiptsV1Env, sibling  # noqa: E402
from shogym.envs.receipts.generators.ledger import GENERATOR  # noqa: E402
from shogym.envs.receipts.receipt_ast import KINDS, mask_slots, slot_ranges  # noqa: E402
from shogym.envs.receipts.protocol_v2 import (  # noqa: E402
    CANONICALIZATION_VERSION,
    CELLS,
    RECEIPTS_GRADE,
    receipt_contract,
)
from shogym.envs._grading import DirectoryCaptures, SEALED  # noqa: E402
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2.artifact import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    BODY_MEDIA_TYPE,
    CELL_KINDS,
    ELIGIBLE_CELLS,
    GRADED_CELL,
    ORACLE_CELL,
    PLACEBO_CELL,
    manifest_preimage,
    mask_spans,
    read_source_artifact,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.identity import submission_digest  # noqa: E402
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    GradeAttemptInput,
    SealAttemptInput,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
)
from tests._fixtures.receipts_bundle import private_bundle  # noqa: E402

ATTEMPT = "b" * 32
OTHER_ATTEMPT = "c" * 32
TERMINAL = "submit_filing"


def seal_of(name: str) -> str:
    """One test's own seal id, so no two tests read each other's committed source."""
    return sha256(name.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def frozen_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One admission bundle that actually verifies, shared by the module.

    This module's own copy of the process's verified template, under a room no other module
    writes into: what an environment opened on it forks and seals goes beside it.
    """
    return private_bundle(tmp_path_factory.mktemp("bundles"))


async def _episode(frozen_bundle: Path, tmp_path: Path, task: int = 0) -> ServedEpisode:
    """One world of this environment for a seal to read."""
    return await ServedEpisode.start(
        "receipts_v1",
        task=task,
        env_config={"bundle": str(frozen_bundle)},
        ends_on_horizon=False,
        trace_path=tmp_path / f"run-{task}.jsonl",
    )


def _filing(env: ReceiptsV1Env, position: int = 0) -> str:
    """The filing of an agent that applied the drawn convention exactly."""
    ordinal, index = env._position(position)
    task = env.instance(ordinal).side(sibling(index))
    return "\n".join(
        f"{identifier},{value}"
        for identifier, value in zip(GENERATOR.row_identifiers(task.table), task.key)
    )


def _contract(env: ReceiptsV1Env, position: int = 0) -> Any:
    """The contract this family's cells are published under."""
    ordinal, index = env._position(position)
    return receipt_contract(
        env.instance(ordinal),
        sibling(index),
        contract_id="ledger_receipt",
        cells=(
            (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
            (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
        ),
    )


def _activities(episode: ServedEpisode) -> List[Any]:
    """This environment's own Activities, asked for the way a generation asks for them."""
    _version, activities, _digest = episode.env.protocol_v2_terminal(
        lambda _attempt: (episode.env, episode.session_id)
    )
    return activities


def _nowhere(episode: ServedEpisode) -> List[Any]:
    """The same Activities over a route that reaches no world, which is what a resume sees."""
    _version, activities, _digest = episode.env.protocol_v2_terminal(lambda _attempt: None)
    return activities


async def _seal(
    activities: List[Any],
    *,
    seal_id: str,
    filing: str,
    blobs: Optional[Path],
    contract: Any,
    attempt_id: str = ATTEMPT,
    ordinal: int = 0,
) -> Any:
    """Ask the seal Activity for one attempt, the way a generation under a contract asks."""
    return await activities[0](
        SealAttemptInput(
            attempt_id=attempt_id,
            seal_id=seal_id,
            native_terminal_name=TERMINAL,
            canonicalization_version=CANONICALIZATION_VERSION,
            native_arguments={"filing": filing},
            blob_root=None if blobs is None else str(blobs),
            execution_ordinal=ordinal,
            receipt_contract=contract,
        )
    )


def _source(sealed: Any) -> Any:
    """The descriptor one seal published, read back the way a workflow reads it.

    The result carries the value the canonical bytes are written from rather than a typed field,
    so the strict reader is what makes a descriptor of it. Reading it here the same way is what
    keeps these assertions about the thing a generation would be handed.
    """
    assert sealed.source_artifact is not None
    return read_source_artifact(sealed.source_artifact)


def _record(episode: ServedEpisode, seal_id: str) -> Dict[str, Any]:
    """The capture this seal committed, read off the disk it was committed to."""
    held = DirectoryCaptures(episode.env.seals).held(seal_id, SEALED)
    assert held is not None
    return held


def _rewrite(episode: ServedEpisode, seal_id: str, record: Dict[str, Any]) -> None:
    """Put a doctored record back where the committed one was, as a damaged store would hold it."""
    path = DirectoryCaptures(episode.env.seals).directory(seal_id) / f"{SEALED}.json"
    path.unlink()
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")


async def test_a_seal_under_a_contract_publishes_three_cells_and_a_descriptor_over_them(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The source is installed, described, and committed to by the hash of its own bytes.

    The descriptor binds the world it was made in as well as the bytes: the bundle the fork is
    keyed by, the task the filing answers, the attempt and seal that captured it, the ordinal the
    seal id was computed under, and the contract and grade identity it was published against. The
    commitment is the manifest blob's own address, so a reader holding the commitment can fetch
    what was committed.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        sealed = await _seal(
            _activities(episode),
            seal_id=seal_of("publishes"),
            filing=_filing(env),
            blobs=blobs,
            contract=contract,
        )
        manifest = _source(sealed)
        assert manifest.schema_version == ARTIFACT_SCHEMA_VERSION
        assert manifest.receipt_contract == contract
        assert manifest.grade_identity == RECEIPTS_GRADE
        assert manifest.source_attempt_id == ATTEMPT
        assert manifest.source_seal_id == seal_of("publishes")
        assert manifest.execution_ordinal == 0
        assert manifest.canonicalization_version == CANONICALIZATION_VERSION
        assert manifest.renderer_configuration == bank_mod.RENDERER_CONFIGURATION
        assert manifest.bundle_digest == env._served.digest
        assert manifest.environment_task_id == env.describe("0").task_id

        # The commitment is the manifest blob's own address, and the blob is the preimage.
        store = FilesystemBlobStore(blobs)
        assert sealed.source_commitment == source_commitment(manifest)
        assert store.read(sealed.source_commitment) == manifest_preimage(manifest)

        # The three cells are installed under their own plain digests, at the registered size,
        # and the submission the kernel digests is the one the seal returned.
        assert sorted(manifest.cells) == sorted(CELL_KINDS)
        for kind in CELL_KINDS:
            reference = manifest.cells[kind]
            assert reference.size == contract.body_size == 2657
            assert reference.media_type == BODY_MEDIA_TYPE
            assert sha256(store.read(reference.sha256)).hexdigest() == reference.sha256
        raw = sealed.canonical_submission_text.encode("utf-8")
        assert manifest.canonical_submission.sha256 == sha256(raw).hexdigest()
        assert manifest.canonical_submission.size == len(raw)
        assert manifest.kernel_submission_digest == submission_digest(ATTEMPT, TERMINAL, raw)
        assert store.read(manifest.canonical_submission.sha256) == raw

        # And the parity evidence is the pair the environment's own mask makes of those cells.
        ordinal, index = env._position(0)
        instance = env.instance(ordinal)
        judged = _judged(instance, sibling(index), _filing(env))
        ranges = slot_ranges(judged.asts[GRADED_CELL], instance.envelope)
        assert list(mask_spans(contract)) == ranges
        graded = store.read(manifest.cells[GRADED_CELL].sha256)
        expected = sha256(mask_slots(graded, ranges)).hexdigest()
        assert manifest.pair_parity.masked_body_sha256 == expected
        assert manifest.pair_parity.body_size == 2657
        assert manifest.pair_parity.encoded_body_bytes == 2688
    finally:
        await episode.close()


def _judged(instance: Any, side: str, filing: str) -> Any:
    """The three cells of one filing, rendered again here to compare a mask against."""
    from shogym.envs.receipts.checks import judge_cells

    task = instance.side(side)
    return judge_cells(
        GENERATOR,
        task,
        GENERATOR.parse_and_canonicalize(task, filing),
        instance.convention,
        instance.envelope,
    )


def test_the_cells_a_source_holds_are_the_cells_this_environment_renders() -> None:
    """One set of kinds, named in two places, and a test that keeps them one set.

    The protocol names the cells a descriptor may hold and the environment names the cells it
    renders. They are the same three, and a build where they drifted apart would publish a
    descriptor whose map could never be filled.
    """
    assert tuple(CELL_KINDS) == tuple(KINDS)
    assert tuple(ELIGIBLE_CELLS) == (GRADED_CELL, PLACEBO_CELL)
    assert ORACLE_CELL not in ELIGIBLE_CELLS


async def test_a_seal_with_no_contract_publishes_no_descriptor(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """A generation that declared none is served exactly as it was before there was one.

    Its capture holds the cells and the numbers and none of the bindings publication is checked
    against, so nothing about the record it writes changed either.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    try:
        sealed = await _seal(
            _activities(episode),
            seal_id=seal_of("no contract"),
            filing=_filing(episode.env),
            blobs=tmp_path / "blobs",
            contract=None,
        )
        assert sealed.source_artifact is None
        assert sealed.source_commitment == ""
        held = _record(episode, seal_of("no contract"))
        assert "source_attempt_id" not in held
        assert "receipt_contract" not in held
        assert sorted(held[CELLS]) == sorted(CELL_KINDS)
    finally:
        await episode.close()


async def test_a_generation_that_publishes_bodies_and_was_given_no_store_is_refused(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """There is nowhere to install three cells, so there is no source to return."""
    episode = await _episode(frozen_bundle, tmp_path)
    try:
        with pytest.raises(ApplicationError, match="needs a store") as refused:
            await _seal(
                _activities(episode),
                seal_id=seal_of("no store"),
                filing=_filing(episode.env),
                blobs=None,
                contract=_contract(episode.env),
            )
        assert refused.value.type == "UnavailableEvidence"
        assert refused.value.non_retryable is True
    finally:
        await episode.close()


async def test_a_contract_this_environment_cannot_publish_under_is_refused_before_a_capture(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The capture is bound to the contract at its first write, so the contract is checked first.

    A geometry whose rows end outside the body, a renderer other than the one this environment's
    cells come out of, and a call that does not say which execution is asking are each a source
    that could never be published. None of them leaves a capture behind: a record committed and
    then refused would be a source nothing could ever republish.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        captures = DirectoryCaptures(env.seals)
        for name, changes, kind, ordinal in (
            ("wide geometry", {"row_count": 100}, "ReceiptContractRefused", 0),
            ("other renderer", {"renderer_configuration": "elsewhere-v1"}, "ContractDrift", 0),
            ("no ordinal", {}, "SourceArtifactRefused", -1),
        ):
            seal_id = seal_of(f"refused before capture {name}")
            with pytest.raises(ApplicationError) as refused:
                await _seal(
                    _activities(episode),
                    seal_id=seal_id,
                    filing=_filing(env),
                    blobs=blobs,
                    contract=replace(contract, **changes),
                    ordinal=ordinal,
                )
            assert refused.value.type == kind
            assert captures.held(seal_id, SEALED) is None
    finally:
        await episode.close()


async def test_a_retry_with_another_filing_and_no_world_returns_the_first_committed_source(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The capture is the first winning write, and everything after it reads that record.

    The retry is given a filing the first call never saw and a route that reaches no world at
    all, which is what a resumed generation sees for every attempt. What comes back is the source
    that was committed: the same commitment, the same cells, the same submission and the same
    parity evidence, with nothing rendered a second time.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        contract = _contract(episode.env)
        first = await _seal(
            _activities(episode),
            seal_id=seal_of("changed input"),
            filing=_filing(episode.env),
            blobs=blobs,
            contract=contract,
        )
        again = await _seal(
            _nowhere(episode),
            seal_id=seal_of("changed input"),
            filing="every record answered wrongly",
            blobs=blobs,
            contract=contract,
        )
        assert again.source_commitment == first.source_commitment
        assert again.source_artifact == first.source_artifact
        assert again.canonical_submission_text == first.canonical_submission_text
        assert again.environment_recovery_token == first.environment_recovery_token
    finally:
        await episode.close()


async def test_two_captures_exchanged_under_each_other_s_seal_are_refused_as_the_wrong_source(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """A record whose cells all hash correctly can still be the wrong record.

    Both captures are complete and internally consistent after the exchange: every cell hashes to
    the bank digest committed beside it, so nothing about the bytes says anything is wrong. What
    is wrong is whose they are, and the attempt and seal the capture was bound to at its first
    write are what say so.

    The refusal comes before anything is written back. The submission text a call installs is
    written out of the held record, so a call that read somebody else's capture would put that
    capture's bytes into this run's store on its way to refusing: the two refused calls below are
    given a store of their own and it is empty afterwards.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        mine, theirs = seal_of("exchanged one"), seal_of("exchanged two")
        await _seal(
            _activities(episode),
            seal_id=mine,
            filing=_filing(env),
            blobs=blobs,
            contract=contract,
        )
        await _seal(
            _activities(episode),
            seal_id=theirs,
            filing=_filing(env) + "\n",
            blobs=blobs,
            contract=contract,
            attempt_id=OTHER_ATTEMPT,
        )
        first, second = _record(episode, mine), _record(episode, theirs)
        _rewrite(episode, mine, second)
        _rewrite(episode, theirs, first)

        # The exchanged records are complete: their cells still hash to their own bank digests.
        for kind in CELL_KINDS:
            body = second[CELLS][kind].encode("ascii")
            assert streams.digest(body) == second["cell_digests"][kind]

        untouched = tmp_path / "untouched"
        with pytest.raises(ApplicationError, match="captured for attempt") as refused:
            await _seal(
                _activities(episode),
                seal_id=mine,
                filing=_filing(env),
                blobs=untouched,
                contract=contract,
            )
        assert refused.value.type == "WrongSource"
        assert refused.value.non_retryable is True
        with pytest.raises(ApplicationError, match="captured for attempt"):
            await _seal(
                _activities(episode),
                seal_id=theirs,
                filing=_filing(env),
                blobs=untouched,
                contract=contract,
                attempt_id=OTHER_ATTEMPT,
            )
        # Nothing of either exchanged capture reached the store this call was given.
        assert _names(untouched) == set()
    finally:
        await episode.close()


async def test_a_bank_digest_and_a_blob_address_never_stand_in_for_one_another(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Two names for the same bytes, in two namespaces, and neither answers for the other.

    The bank's digest is length prefixed and keys a fork; the blob address is the plain hash and
    is what a store resolves. Both are sixty four hexadecimal characters, so nothing about their
    shape tells them apart, and the descriptor names them in separate fields for that reason. A
    record that had one written where the other belongs is refused by recomputing both from the
    held bytes, and a store asked for a bank digest holds nothing under it.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        sealed = await _seal(
            _activities(episode),
            seal_id=seal_of("two namespaces"),
            filing=_filing(env),
            blobs=blobs,
            contract=contract,
        )
        manifest = _source(sealed)
        store = FilesystemBlobStore(blobs)
        for kind in CELL_KINDS:
            assert manifest.cells[kind].sha256 != manifest.bank_cell_digests[kind]
            assert store.unverified([manifest.cells[kind].sha256]) == []
            assert store.unverified([manifest.bank_cell_digests[kind]]) == [
                manifest.bank_cell_digests[kind]
            ]

        # And the other direction: a record holding the store's address where the bank's digest
        # belongs is refused rather than published under a digest nothing computed that way.
        held = _record(episode, seal_of("two namespaces"))
        held["cell_digests"][GRADED_CELL] = manifest.cells[GRADED_CELL].sha256
        _rewrite(episode, seal_of("two namespaces"), held)
        with pytest.raises(ApplicationError, match="does not hash to the bank digest") as refused:
            await _seal(
                _activities(episode),
                seal_id=seal_of("two namespaces"),
                filing=_filing(env),
                blobs=blobs,
                contract=contract,
            )
        assert refused.value.type == "CorruptEvidence"
    finally:
        await episode.close()


async def test_two_raw_filings_with_one_canonical_text_share_a_submission_and_not_a_source(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Three filing identities, kept apart, shown on the one case that separates them.

    The two raw filings read to the same canonical text, so the canonical submission and the
    kernel's digest over it are identical. The bank keys its fork by the raw filing, so its
    digest differs. The descriptor carries both and lets neither stand in for the other, which is
    what stops one filing's feedback from being replayed for another.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        plain = _filing(env)
        spaced = plain + "\n"
        first = await _seal(
            _activities(episode),
            seal_id=seal_of("one canonical text"),
            filing=plain,
            blobs=blobs,
            contract=contract,
        )
        second = await _seal(
            _activities(episode),
            seal_id=seal_of("one canonical text again"),
            filing=spaced,
            blobs=blobs,
            contract=contract,
        )
        assert bank_mod.filing_digest(plain) != bank_mod.filing_digest(spaced)
        assert first.canonical_submission_text == second.canonical_submission_text
        one, two = _source(first), _source(second)
        assert one.canonical_submission == two.canonical_submission
        assert one.kernel_submission_digest == two.kernel_submission_digest
        assert one.bank_filing_digest != two.bank_filing_digest
        assert one.bank_filing_digest == bank_mod.filing_digest(plain)
        assert first.source_commitment != second.source_commitment
    finally:
        await episode.close()


async def test_publication_interrupted_after_any_install_publishes_the_same_source(
    frozen_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between installs leaves a source a later call verifies and reinstalls exactly.

    The capture commits before the first object is written, so every step after it is a read, a
    hash check or a byte-for-byte reinstall. The run is cut after each of the objects publication
    installs, in turn, and each time the source that comes back afterwards is the one the first
    uninterrupted publication would have produced.

    After each install and not before it. A cut taken before the write would leave one object
    fewer than the number it was asked for, so the five cuts would cover none through four and
    the state a crash after the last install leaves, the bound manifest already written, would
    never be reached at all. Which objects the store holds is asserted before the resume, so the
    cut is where the loop says it is rather than wherever the counter happened to land.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    try:
        env = episode.env
        contract = _contract(env)
        filing = _filing(env)
        written = _counted(monkeypatch)
        whole = await _seal(
            _activities(episode),
            seal_id=seal_of("uninterrupted"),
            filing=filing,
            blobs=tmp_path / "whole",
            contract=contract,
        )
        monkeypatch.undo()
        # The submission, the three cells and the bound manifest. The loop below cuts after each
        # of them in turn, and this is what says the loop covers every one.
        installs = written["count"]
        assert installs == 5
        for cut in range(1, installs + 1):
            blobs = tmp_path / f"cut-{cut}"
            seal_id = seal_of(f"interrupted after {cut}")
            monkeypatch.setattr(FilesystemBlobStore, "put", _fails_after(cut))
            with pytest.raises(RuntimeError, match="the machine went away"):
                await _seal(
                    _activities(episode),
                    seal_id=seal_id,
                    filing=filing,
                    blobs=blobs,
                    contract=contract,
                )
            monkeypatch.undo()
            # What the interrupted run left behind, before anything resumes over it.
            interrupted = _names(blobs)
            resumed = await _seal(
                _nowhere(episode),
                seal_id=seal_id,
                filing=filing,
                blobs=blobs,
                contract=contract,
            )
            # The same bytes under the same names, and the descriptor differs from the
            # uninterrupted one only in the seal that captured it.
            source, entire = _source(resumed), _source(whole)
            assert source.cells == entire.cells
            assert source.pair_parity == entire.pair_parity
            assert replace(source, source_seal_id=entire.source_seal_id) == entire
            # And the cut fell where the loop says it did: the submission, the three cells and
            # the bound manifest, in that order, as far as this cut got.
            order = [
                source.canonical_submission.sha256,
                *[source.cells[kind].sha256 for kind in CELL_KINDS],
                resumed.source_commitment,
            ]
            assert interrupted == set(order[:cut])
            store = FilesystemBlobStore(blobs)
            assert store.unverified(order) == []
    finally:
        await episode.close()


def _names(root: Path) -> Set[str]:
    """Every object one store holds, by the name it is installed under."""
    return {
        path.name
        for path in root.rglob("*")
        if path.is_file() and not path.name.endswith(".partial")
    }


def _fails_after(installs: int) -> Any:
    """A store that goes away once the ``installs``-th write has landed.

    The write happens and then the machine does not come back, which is the state a crash after
    an install leaves. Raising in front of the write instead would leave one object fewer than
    the cut names and would never reach the last install at all.
    """
    written = {"count": 0}
    real = FilesystemBlobStore.put

    def put(
        self: FilesystemBlobStore, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> Any:
        reference = real(self, data, media_type=media_type)
        written["count"] += 1
        if written["count"] == installs:
            raise RuntimeError("the machine went away")
        return reference

    return put


def _counted(monkeypatch: pytest.MonkeyPatch) -> Dict[str, int]:
    """Count what one publication installs, so a cut can be made after each of them."""
    written = {"count": 0}
    real = FilesystemBlobStore.put

    def put(
        self: FilesystemBlobStore, data: bytes, *, media_type: str = "application/octet-stream"
    ) -> Any:
        written["count"] += 1
        return real(self, data, media_type=media_type)

    monkeypatch.setattr(FilesystemBlobStore, "put", put)
    return written


async def test_a_renderer_a_contract_or_a_grade_that_drifted_under_a_capture_is_refused(
    frozen_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed source is not republished under whatever the build now happens to be.

    Each of the three is a different claim about the bytes that were committed: which code
    rendered them, which shape they were admitted under, and which grader's verdict the capture
    was taken with. A build that moved any of them is refused rather than allowed to reissue the
    descriptor with its own values in it.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        filing = _filing(env)
        seal_id = seal_of("drift")
        await _seal(
            _activities(episode), seal_id=seal_id, filing=filing, blobs=blobs, contract=contract
        )

        # Another contract over the same bytes, well formed and not the one they were published
        # under.
        with pytest.raises(ApplicationError, match="as it stood then") as contract_drift:
            await _seal(
                _activities(episode),
                seal_id=seal_id,
                filing=filing,
                blobs=blobs,
                contract=replace(contract, contract_id="ledger_receipt_two"),
            )
        assert contract_drift.value.type == "ContractDrift"

        # Another renderer, declared by the contract as well, so what refuses is the record.
        monkeypatch.setattr(bank_mod, "RENDERER_CONFIGURATION", "receipts-render-v2")
        with pytest.raises(ApplicationError, match="was rendered by") as renderer_drift:
            await _seal(
                _activities(episode),
                seal_id=seal_id,
                filing=filing,
                blobs=blobs,
                contract=replace(contract, renderer_configuration="receipts-render-v2"),
            )
        assert renderer_drift.value.type == "ContractDrift"
        monkeypatch.undo()

        # And another grader, which the capture-time identity is what catches.
        monkeypatch.setattr(port, "RECEIPTS_GRADE", replace(RECEIPTS_GRADE, grader_version="2"))
        with pytest.raises(ApplicationError, match="this build grades by") as grade_drift:
            await _seal(
                _activities(episode),
                seal_id=seal_id,
                filing=filing,
                blobs=blobs,
                contract=contract,
            )
        assert grade_drift.value.type == "ContractDrift"
    finally:
        await episode.close()


async def test_a_held_record_this_build_cannot_read_is_refused_rather_than_repaired(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Nothing is filled in and nothing is coerced on the way out of the record.

    A name the record no longer carries, a contract written in a shape this build does not read
    and a count that arrived as text are each a source that cannot be published. Repairing any of
    them would publish a descriptor over values nobody committed, so each is a refusal at the
    boundary that read it.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        filing = _filing(env)
        seal_id = seal_of("unreadable record")
        await _seal(
            _activities(episode), seal_id=seal_id, filing=filing, blobs=blobs, contract=contract
        )
        committed = _record(episode, seal_id)

        for name, doctor, kind, message in (
            ("a missing name", _without("task_id"), "CorruptEvidence", "carries no task_id"),
            (
                "an unreadable contract",
                _without("row_count", inside="receipt_contract"),
                "CorruptEvidence",
                "carries exactly",
            ),
            (
                "a count written as text",
                _as_text("execution_ordinal"),
                "SourceArtifactRefused",
                "whole number",
            ),
        ):
            _rewrite(episode, seal_id, doctor(json.loads(json.dumps(committed))))
            with pytest.raises(ApplicationError, match=message) as refused:
                await _seal(
                    _activities(episode),
                    seal_id=seal_id,
                    filing=filing,
                    blobs=blobs,
                    contract=contract,
                )
            assert refused.value.type == kind, name
    finally:
        await episode.close()


def _without(name: str, *, inside: Optional[str] = None) -> Any:
    """A record with one name taken out of it, at the top or inside one of its values."""

    def doctor(record: Dict[str, Any]) -> Dict[str, Any]:
        (record if inside is None else record[inside]).pop(name)
        return record

    return doctor


def _as_text(name: str) -> Any:
    """A record with one count written as the text of itself."""

    def doctor(record: Dict[str, Any]) -> Dict[str, Any]:
        record[name] = str(record[name])
        return record

    return doctor


async def test_cells_that_stopped_being_a_pair_are_refused_at_publication(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Publication is the one place holding both cells, so it is where the pair is established.

    A held placebo changed outside the registered slots is refused even with its bank digest
    moved to match, because what the pair check compares is the masked bodies rather than each
    body against its own hash. A body that is not the registered size is refused before the mask
    is expanded at all.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        filing = _filing(env)
        seal_id = seal_of("no longer a pair")
        await _seal(
            _activities(episode), seal_id=seal_id, filing=filing, blobs=blobs, contract=contract
        )
        held = _record(episode, seal_id)
        body = held[CELLS][PLACEBO_CELL]
        outside = contract.rows_start - 1
        moved = body[:outside] + ("Z" if body[outside] != "Z" else "Y") + body[outside + 1:]
        held[CELLS][PLACEBO_CELL] = moved
        held["cell_digests"][PLACEBO_CELL] = streams.digest(moved.encode("ascii"))
        _rewrite(episode, seal_id, held)
        with pytest.raises(ApplicationError, match="outside the slots") as refused:
            await _seal(
                _activities(episode),
                seal_id=seal_id,
                filing=filing,
                blobs=blobs,
                contract=contract,
            )
        assert refused.value.type == "PairRefused"

        held[CELLS][PLACEBO_CELL] = body[:-1]
        held["cell_digests"][PLACEBO_CELL] = streams.digest(body[:-1].encode("ascii"))
        _rewrite(episode, seal_id, held)
        with pytest.raises(ApplicationError, match="fixes a body at 2657 bytes") as short:
            await _seal(
                _activities(episode),
                seal_id=seal_id,
                filing=filing,
                blobs=blobs,
                contract=contract,
            )
        assert short.value.type == "PairRefused"
    finally:
        await episode.close()


async def test_a_retry_after_the_capture_scores_nothing_and_renders_nothing_again(
    frozen_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Everything after the first winning write is a read, a hash check or a copy.

    The scorer and the renderer are one call here: reading the world, scoring the filing and
    rendering the fork's cells all happen in ``sealed_filing``, and the capture is what commits
    what they produced. Made fatal afterwards, it is never reached again: the seal republishes
    the committed source out of the held record, and the grade that follows reads the verdict
    that capture already computed and installs it as the run's evidence.
    """
    episode = await _episode(frozen_bundle, tmp_path)
    blobs = tmp_path / "blobs"
    try:
        env = episode.env
        contract = _contract(env)
        activities = _activities(episode)
        seal_id = seal_of("fatal-after-capture")
        first = await _seal(
            activities, seal_id=seal_id, filing=_filing(env), blobs=blobs, contract=contract
        )

        def fatal(*arguments: Any, **named: Any) -> Any:
            raise AssertionError("nothing scores or renders after the capture has committed")

        monkeypatch.setattr(ReceiptsV1Env, "sealed_filing", fatal)

        again = await _seal(
            activities, seal_id=seal_id, filing=_filing(env), blobs=blobs, contract=contract
        )
        assert again == first

        graded = await activities[1](
            GradeAttemptInput(
                attempt_id=ATTEMPT,
                seal_id=seal_id,
                submission_digest=submission_digest(
                    ATTEMPT, TERMINAL, first.canonical_submission_text.encode("utf-8")
                ),
                canonical_submission_text=first.canonical_submission_text,
                environment_recovery_token=first.environment_recovery_token,
                blob_root=str(blobs),
            )
        )
        assert graded.attempt_id == ATTEMPT and graded.seal_id == seal_id
        assert graded.grade == RECEIPTS_GRADE
        assert graded.decode_state == "decoded"
        assert graded.score > 0
        # The verdict is installed as the run's own evidence, out of the same held record.
        assert FilesystemBlobStore(blobs).read(graded.evidence.sha256)

        # And the fatality is load-bearing: a capture that has not committed still needs it.
        with pytest.raises(AssertionError):
            await _seal(
                activities,
                seal_id=seal_of("fatal-after-capture-second"),
                filing=_filing(env),
                blobs=blobs,
                contract=contract,
            )
    finally:
        await episode.close()
