"""One committed source, routed to an agent through each of the two receipt policies.

Everything before this step could publish a source and nothing could deliver one. What is under
test here is the whole of the route: a generation declares a contract, its seal publishes the
three cells, the controller selects one of the two eligible ones, the payload Activity resolves
that one reference through the run's own store, and the workflow checks what came back without
rebuilding any of it.

The two runs use one set of public identifiers and one seal, so the second is a second derivation
of the first's committed source rather than a second render of the same filing. What that lets
these tests say is the thing the pair exists for: the graded body and the placebo body differ
inside the slots the contract registers, are identical everywhere else, and come to the same
model-visible byte count, while each carries its own hashes.

The filings are the other half. A body is cut from what the agent filed, so its wire count is a
fact about one filing rather than about the registration: an empty filing, a malformed one, a
short one and one holding a character the canonical encoding escapes each make a legal pair, and
the count follows the bytes.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.client import WorkflowFailureError, WorkflowUpdateFailedError  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.envs._grading import SEALED, DirectoryCaptures  # noqa: E402
from shogym.envs.receipts import streams  # noqa: E402
from shogym.envs.receipts.env_v1 import ReceiptsV1Env, sibling  # noqa: E402
from shogym.envs.receipts.generators.ledger import GENERATOR  # noqa: E402
from shogym.envs.receipts.protocol_v2 import (  # noqa: E402
    CANONICALIZATION_VERSION,
    CELLS,
    RECEIPTS_GRADE,
    receipt_contract,
)
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import (  # noqa: E402
    IMMEDIATE,
    Payload,
    PullRequest,
    TerminalMetadata,
    visible_bytes,
)
from shogym.serve.protocol_v2.artifact import (  # noqa: E402
    CELL_KINDS,
    ELIGIBLE_CELLS,
    GRADED_CELL,
    MASK_FILL,
    ORACLE_CELL,
    PLACEBO_CELL,
    encoded_body_bytes,
    mask_spans,
    masked_body,
    payload_wire_count,
    read_source_artifact,
    source_commitment,
)
from shogym.serve.protocol_v2.blobs import FilesystemBlobStore  # noqa: E402
from shogym.serve.protocol_v2.gateway import install_policies  # noqa: E402
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    STREAM_TASK_QUEUE,
    ConsumerClaim,
    GeneratePayloadBundleInput,
    OfferedMessage,
    PayloadCandidateResult,
    SealAttemptInput,
    SealRequest,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    generate_payload_bundle_activity,
    protocol_error_code,
    start_stream,
    stream_worker,
    verify_blobs_activity,
)
from shogym.serve.protocol_v2.kernel import workflow as kernel_workflow  # noqa: E402
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GENERATE_PAYLOAD_BUNDLE,
    SEAL_ATTEMPT,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    DELIVER,
    EXPERIMENT,
    GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
    PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
    REGISTERED,
    WITHHOLD,
    PayloadDisposition,
    PolicyProvenance,
    roster_digest,
)
from tests._fixtures.receipts_bundle import verified_bundle  # noqa: E402
from tests._fixtures.temporal_server import time_skipping_environment  # noqa: E402
from tests._fixtures.upstream_gate import environmental_skip  # noqa: E402

ATTEMPT = "b" * 32
WITHHELD_ATTEMPT = "c" * 32
TERMINAL = "submit_filing"
CONTRACT = "ledger_receipt"
EXECUTION = "execution-1"

# What the pinned ledger comes to: a body of this size, that many bytes on the wire once the
# obligation's own wrapper is around it, and two more where the filing holds a character the
# canonical encoding escapes.
BODY_SIZE = 2657
STORED_BYTES = 2688
WIRE_BYTES = 2833


def oid(value: int) -> str:
    return f"{value:032x}"


@pytest.fixture(scope="module")
def frozen_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One admission bundle that actually verifies, shared by the module."""
    return verified_bundle(tmp_path_factory.mktemp("bundles"))


@pytest_asyncio.fixture
async def env() -> Any:
    try:
        environment = await time_skipping_environment()
    except Exception as error:  # noqa: BLE001 - an unusable server is the machine's, not the test's
        environmental_skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


@pytest_asyncio.fixture
async def world(frozen_bundle: Path, tmp_path: Path) -> Any:
    """One served ledger world, with the store this run keeps its objects in."""
    episode = await ServedEpisode.start(
        "receipts_v1",
        task=0,
        env_config={"bundle": str(frozen_bundle)},
        ends_on_horizon=False,
        trace_path=tmp_path / "run.jsonl",
    )
    try:
        yield episode
    finally:
        await episode.close()


def contract_of(env_v1: ReceiptsV1Env, position: int = 0) -> Any:
    """The contract this family's cells are admitted and published under."""
    ordinal, index = env_v1._position(position)
    return receipt_contract(
        env_v1.instance(ordinal),
        sibling(index),
        contract_id=CONTRACT,
        cells=(
            (GRADED_RECEIPT_ARTIFACT_V1_DIGEST, GRADED_CELL),
            (PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST, PLACEBO_CELL),
        ),
    )


def filing_of(env_v1: ReceiptsV1Env, position: int = 0) -> str:
    """The filing of an agent that applied the drawn convention exactly."""
    ordinal, index = env_v1._position(position)
    task = env_v1.instance(ordinal).side(sibling(index))
    return "\n".join(
        f"{identifier},{value}"
        for identifier, value in zip(GENERATOR.row_identifiers(task.table), task.key)
    )


def activities_of(episode: ServedEpisode) -> List[Any]:
    """This environment's own Activities, over a route that reaches its one world."""
    _version, activities, _digest = episode.env.protocol_v2_terminal(
        lambda _attempt: (episode.env, episode.session_id)
    )
    return list(activities)


def start_for(
    episode: ServedEpisode,
    contract: Any,
    blobs: Path,
    *,
    policy_digest: str,
    cell: str,
    store: bool = True,
) -> StreamStart:
    """One generation delivering one committed cell, with every identifier fixed here.

    The identifiers are the same whichever cell is being served, which is what makes two runs of
    this two derivations of one source: the seal id is minted from the hidden execution, the
    ordinal and the attempt, so the second run finds the capture the first committed and
    publishes it again rather than rendering a second one.
    """
    item = TaskItem(
        task_position=0,
        attempt_id=ATTEMPT,
        task_message_id=oid(0x101),
        ack_message_id=oid(0x102),
        payload_position=0,
        payload_message_id=oid(0x103),
        body=episode.env.describe("0").instructions,
    )
    row = PayloadDisposition(
        attempt_id=ATTEMPT,
        payload_position=0,
        kind=DELIVER,
        policy_digest=policy_digest,
        cell=cell,
        resolution_source=REGISTERED,
        family_id=CONTRACT,
    )
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash="d" * 64,
        initial_cursor=oid(1),
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id=EXECUTION,
        canonicalization_version=CANONICALIZATION_VERSION,
        terminal_tool=TerminalTool(
            public_tool_name=TERMINAL,
            native_terminal_name=TERMINAL,
            argument_names=["filing"],
        ),
        tasks=[item],
        blob_root=str(blobs) if store else None,
        profile=EXPERIMENT,
        grade=RECEIPTS_GRADE,
        dispositions=[row],
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest([row]),
            experiment_id="the_subject_of_this_run",
        ),
        receipt_contracts=[contract],
        receipt_source=episode.env._served.digest,
    )


class Caller:
    """One consumer of a generation, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: Any, cursor: str, blobs: FilesystemBlobStore) -> None:
        self.stream = stream
        self.cursor = cursor
        self._counter = 0
        self.transcript = blobs.put(b"the transcript", media_type="text/plain").sha256
        self.turn = blobs.put(b"the provider turn", media_type="text/plain").sha256
        self.checkpoint = blobs.put(b"the checkpoint", media_type="text/plain").sha256

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x1000 + self._counter)

    async def pull(self) -> OfferedMessage:
        return await self.stream.pull(
            PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )

    async def present(self, message: OfferedMessage) -> None:
        ack = await self.stream.present(
            message,
            attestation_id=self.next_id(),
            transcript_blob=self.transcript,
            provider_turn_blob=self.turn if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=self.checkpoint if message.kind == "task" else None,
        )
        self.cursor = ack.cursor

    async def seal(self, filing: str) -> OfferedMessage:
        return await self.seal_of(ATTEMPT, filing)

    async def seal_of(self, attempt_id: str, filing: str) -> OfferedMessage:
        return await self.stream.seal(
            SealRequest(
                metadata=TerminalMetadata(
                    request_id=self.next_id(),
                    last_presented_cursor=self.cursor,
                    attempt_id=attempt_id,
                ),
                public_tool_name=TERMINAL,
                native_terminal_name=TERMINAL,
                native_arguments={"filing": filing},
            )
        )


async def opened(environment: Any, start: StreamStart, blobs: Path, workflow_id: str) -> Caller:
    """Install what this generation's history will cite, then start it and claim it."""
    store = FilesystemBlobStore(blobs)
    if start.blob_root is not None:
        install_policies(store, start)
    stream = await start_stream(environment.client, start, workflow_id=workflow_id)
    receipt = await stream.claim_consumer(
        ConsumerClaim(consumer_id="harness-1", claim_hash="d" * 64)
    )
    return Caller(stream, receipt.initial_cursor, store)


async def delivered(
    environment: Any, start: StreamStart, blobs: Path, workflow_id: str, filing: str
) -> OfferedMessage:
    """Work one attempt to the end and return the payload the generation offers for it."""
    caller = await opened(environment, start, blobs, workflow_id)
    await caller.present(await caller.pull())
    ack = await caller.seal(filing)
    await caller.present(ack)
    return await caller.pull()


async def refused(
    environment: Any, start: StreamStart, blobs: Path, workflow_id: str, filing: str
) -> str:
    """Work one attempt to a seal that cannot be vouched for, and return what refused it."""
    caller = await opened(environment, start, blobs, workflow_id)
    await caller.present(await caller.pull())
    try:
        await caller.seal(filing)
    except WorkflowUpdateFailedError as error:
        cause: Any = error.cause
        while cause is not None and not isinstance(cause, ApplicationError):
            cause = getattr(cause, "cause", None)
        assert isinstance(cause, ApplicationError), error.cause
        return str(cause.type)
    raise AssertionError("the seal was accepted")


def body_of(payload: OfferedMessage) -> str:
    """The body an offered payload carries, out of the bytes it would be presented as."""
    return json.loads(payload.visible_text)["body"]


def committed_cells(episode: ServedEpisode, seal_id: str) -> Dict[str, str]:
    """The three cells the capture under one seal holds, read off the disk it committed to."""
    held = DirectoryCaptures(episode.env.seals).held(seal_id, SEALED)
    assert held is not None
    return {str(kind): str(text) for kind, text in held[CELLS].items()}


def seal_id_for() -> str:
    """The seal id this file's generations mint, which is one seal for all of them."""
    from shogym.serve.protocol_v2.kernel import hidden_seal_id

    return hidden_seal_id(EXECUTION, 0, ATTEMPT)


async def published(episode: ServedEpisode, contract: Any, blobs: Path, *, seal: str, filing: str) -> Any:
    """The descriptor one seal committed, read back by asking that seal again.

    A seal under a key that already holds a capture publishes what was committed, so this is a
    read of the source rather than a second render of the filing. The result carries the value the
    descriptor's canonical bytes are written from, so the strict reader is what makes a descriptor
    of it here as it is in the workflow.
    """
    returned = (
        await activities_of(episode)[0](
            SealAttemptInput(
                attempt_id=ATTEMPT,
                seal_id=seal,
                native_terminal_name=TERMINAL,
                canonicalization_version=CANONICALIZATION_VERSION,
                native_arguments={"filing": filing},
                blob_root=str(blobs),
                execution_ordinal=0,
                receipt_contract=contract,
            )
        )
    ).source_artifact
    return read_source_artifact(returned)


async def test_one_committed_source_is_routed_through_both_policies_under_one_identity(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """Two generations, one source, and the pair that source promised.

    The identifiers are the same in both runs, so the second seal finds the capture the first
    committed: what differs between the two deliveries is which cell the controller selected and
    nothing else. The bodies are the environment's own committed bytes, they differ only inside
    the slots the contract registers, they come to the same measurement on the wire, and each
    hashes to its own committed entry rather than to a shared one.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    filing = filing_of(world.env)
    async with stream_worker(env.client, activities=activities_of(world)):
        graded = await delivered(
            env,
            start_for(
                world,
                contract,
                blobs,
                policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                cell=GRADED_CELL,
            ),
            blobs,
            "stream/receipt-artifact-graded/1",
            filing,
        )
        placebo = await delivered(
            env,
            start_for(
                world,
                contract,
                blobs,
                policy_digest=PLACEBO_RECEIPT_ARTIFACT_V1_DIGEST,
                cell=PLACEBO_CELL,
            ),
            blobs,
            "stream/receipt-artifact-placebo/1",
            filing,
        )
    assert graded.kind == "payload" and placebo.kind == "payload"

    # Each body is the cell the capture holds, byte for byte, and neither is the oracle's.
    held = committed_cells(world, seal_id_for())
    served = {GRADED_CELL: body_of(graded), PLACEBO_CELL: body_of(placebo)}
    assert served[GRADED_CELL] == held[GRADED_CELL]
    assert served[PLACEBO_CELL] == held[PLACEBO_CELL]
    assert held[ORACLE_CELL] not in served.values()

    # One source, one commitment, and each body hashing to its own committed entry.
    manifest = await published(world, contract, blobs, seal=seal_id_for(), filing=filing)
    for cell in ELIGIBLE_CELLS:
        raw = served[cell].encode("ascii")
        assert len(raw) == BODY_SIZE == manifest.cells[cell].size
        assert sha256(raw).hexdigest() == manifest.cells[cell].sha256
    assert manifest.cells[GRADED_CELL].sha256 != manifest.cells[PLACEBO_CELL].sha256

    # They differ inside the registered slots and nowhere else, under the mask the contract's own
    # geometry expands to, and that masked body is the one publication committed a hash for.
    spans = mask_spans(contract)
    one = served[GRADED_CELL].encode("ascii")
    two = served[PLACEBO_CELL].encode("ascii")
    assert one != two
    assert masked_body(one, spans) == masked_body(two, spans)
    assert sha256(masked_body(one, spans)).hexdigest() == manifest.pair_parity.masked_body_sha256

    # And they come to one measurement on the wire, which is the wrapper plus the stored count.
    counts = {
        cell: len(message.visible_text.encode("utf-8"))
        for cell, message in ((GRADED_CELL, graded), (PLACEBO_CELL, placebo))
    }
    assert counts[GRADED_CELL] == counts[PLACEBO_CELL] == WIRE_BYTES
    assert manifest.pair_parity.encoded_body_bytes == STORED_BYTES
    assert WIRE_BYTES == payload_wire_count(
        payload_message_id=oid(0x103),
        attempt_id=ATTEMPT,
        encoded_body_bytes=manifest.pair_parity.encoded_body_bytes,
    )
    # The quoted length would have made that sum 2835, which is a candidate every other check
    # passed being refused, so the two sums are kept apart here as well.
    assert WIRE_BYTES + 2 == payload_wire_count(
        payload_message_id=oid(0x103),
        attempt_id=ATTEMPT,
        encoded_body_bytes=len(json.dumps(served[GRADED_CELL]).encode("utf-8")),
    )
    # Equal counts and separately correct hashes: the two are one measurement and two bodies.
    assert sha256(graded.visible_text.encode("utf-8")).hexdigest() != sha256(
        placebo.visible_text.encode("utf-8")
    ).hexdigest()


def filings_of(exact: str) -> Dict[str, str]:
    """One filing of each shape an agent can legally make, including the two that escape."""
    rows = exact.splitlines()
    first = rows[0].split(",")[0]
    return {
        "exact": exact,
        "empty": "",
        "malformed": "this is not a table at all",
        "omitted": "\n".join(rows[:5]),
        "truncated": exact[: len(exact) // 2],
        "quote": "\n".join([f'{first},"'] + rows[1:]),
        "backslash": "\n".join([f"{first},\\"] + rows[1:]),
    }


@pytest.mark.parametrize(
    "shape",
    ["exact", "empty", "malformed", "omitted", "truncated", "quote", "backslash"],
)
async def test_every_legal_filing_publishes_one_pair_and_one_wire_count(
    world: ServedEpisode, tmp_path: Path, shape: str
) -> None:
    """The registration fixes the body and the filing fixes the count, which is why both exist.

    Each of these is a filing an agent can make and each makes a pair: three cells at the
    registered size, two of them identical outside the registered slots, and one stored count
    between them. What moves is the wire count, and it moves exactly with what the canonical
    encoding has to escape: a filing carrying one double quote or one backslash comes to one byte
    more than the pinned ledger's, which is why a fixed wrapper allowance would refuse it and the
    measured contribution does not.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    filing = filings_of(filing_of(world.env))[shape]
    manifest = await published(
        world,
        contract,
        blobs,
        seal=sha256(shape.encode("utf-8")).hexdigest(),
        filing=filing,
    )
    store = FilesystemBlobStore(blobs)
    bodies = {
        kind: store.read(manifest.cells[kind].sha256).decode("ascii") for kind in CELL_KINDS
    }
    for kind in CELL_KINDS:
        assert len(bodies[kind].encode("ascii")) == BODY_SIZE

    spans = mask_spans(contract)
    one = bodies[GRADED_CELL].encode("ascii")
    two = bodies[PLACEBO_CELL].encode("ascii")
    assert masked_body(one, spans) == masked_body(two, spans)
    assert encoded_body_bytes(bodies[GRADED_CELL]) == encoded_body_bytes(bodies[PLACEBO_CELL])
    assert manifest.pair_parity.encoded_body_bytes == encoded_body_bytes(bodies[GRADED_CELL])

    escaped = bodies[GRADED_CELL].count('"') + bodies[GRADED_CELL].count("\\")
    wire = payload_wire_count(
        payload_message_id=oid(0x103),
        attempt_id=ATTEMPT,
        encoded_body_bytes=manifest.pair_parity.encoded_body_bytes,
    )
    assert wire == WIRE_BYTES + escaped
    assert escaped == (1 if shape in ("quote", "backslash") else 0)


# What publication refuses, with every hash inside the record still checking out.


def held_record(episode: ServedEpisode, seal_id: str) -> Dict[str, Any]:
    """The capture one seal committed, read off the disk it was committed to."""
    held = DirectoryCaptures(episode.env.seals).held(seal_id, SEALED)
    assert held is not None
    return dict(held)


def rewrite(episode: ServedEpisode, seal_id: str, record: Dict[str, Any]) -> None:
    """Put a doctored record where the committed one was, as a damaged store would hold it."""
    path = DirectoryCaptures(episode.env.seals).directory(seal_id) / f"{SEALED}.json"
    path.unlink()
    path.write_text(json.dumps(record, sort_keys=True, separators=(",", ":")), encoding="utf-8")


async def test_a_pair_that_differs_by_one_byte_outside_a_slot_is_refused_at_publication(
    world: ServedEpisode, tmp_path: Path
) -> None:
    """The cells are compared under the registered mask, with their own hashes checking out.

    The doctored cell's bank digest is recomputed with it, so nothing inside the record
    disagrees with anything else: what is wrong is that the two bodies are no longer one body
    outside the slots the contract registers, which is the whole of what a pair is.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    seal = sha256(b"one byte outside").hexdigest()
    await published(world, contract, blobs, seal=seal, filing=filing_of(world.env))

    record = held_record(world, seal)
    outside = min(low for low, _high in mask_spans(contract)) - 1
    body = record[CELLS][PLACEBO_CELL]
    moved = body[:outside] + ("Z" if body[outside] != "Z" else "Y") + body[outside + 1:]
    record[CELLS] = {**record[CELLS], PLACEBO_CELL: moved}
    record["cell_digests"] = {
        **record["cell_digests"],
        PLACEBO_CELL: streams.digest(moved.encode("ascii")),
    }
    rewrite(world, seal, record)

    with pytest.raises(ApplicationError, match="differ outside the slots") as refusal:
        await published(world, contract, blobs, seal=seal, filing=filing_of(world.env))
    assert refusal.value.type == "PairRefused"
    assert refusal.value.non_retryable is True


async def test_a_mask_of_nuls_is_not_the_mask_this_pair_was_committed_under(
    world: ServedEpisode, tmp_path: Path
) -> None:
    """Blanking the slots with NULs satisfies every equality sentence and commits another source.

    So the fill is frozen rather than chosen, and evidence taken under a different one is refused
    against the bytes rather than accepted because the equalities still hold.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    seal = sha256(b"a mask of nuls").hexdigest()
    manifest = await published(world, contract, blobs, seal=seal, filing=filing_of(world.env))

    spans = mask_spans(contract)
    graded = FilesystemBlobStore(blobs).read(manifest.cells[GRADED_CELL].sha256)
    blanked = bytearray(graded)
    for low, high in spans:
        blanked[low:high] = b"\x00" * (high - low)
    assert MASK_FILL == b"?"
    assert sha256(masked_body(graded, spans)).hexdigest() == manifest.pair_parity.masked_body_sha256
    assert sha256(bytes(blanked)).hexdigest() != manifest.pair_parity.masked_body_sha256

    record = held_record(world, seal)
    record["pair_parity"] = {
        **record["pair_parity"],
        "masked_body_sha256": sha256(bytes(blanked)).hexdigest(),
    }
    rewrite(world, seal, record)
    with pytest.raises(ApplicationError, match="parity evidence committed") as refusal:
        await published(world, contract, blobs, seal=seal, filing=filing_of(world.env))
    assert refusal.value.type == "CorruptEvidence"


# What delivery refuses, with every measurement the candidate reports about itself correct.


def substituting(replacement: Any) -> Any:
    """A Worker that resolves the cell it was asked for and returns other bytes for it.

    Every measurement is recomputed over what it returns and the echo is left exactly as the
    request asked for it, which is what a faulty, substituted or half upgraded resolver can
    always produce: describing a body correctly costs nothing, and the body is the claim.
    """

    @activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
    async def rogue(request: GeneratePayloadBundleInput) -> Any:
        bundle = await generate_payload_bundle_activity(request)
        [candidate] = bundle.candidates
        body = replacement(candidate.body)
        serialized = visible_bytes(
            Payload(
                message_id=request.payload_message_id,
                attempt_id=request.attempt_id,
                body=body,
            )
        )
        return replace(
            bundle,
            candidates=[
                replace(
                    candidate,
                    body=body,
                    inner_sha256=sha256(body.encode("utf-8")).hexdigest(),
                    visible_sha256=sha256(serialized).hexdigest(),
                    visible_byte_count=len(serialized),
                )
            ],
        )

    return rogue


def echoing(**changes: Any) -> Any:
    """A Worker that resolves the right bytes and describes them as something else."""

    @activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
    async def rogue(request: GeneratePayloadBundleInput) -> Any:
        bundle = await generate_payload_bundle_activity(request)
        [candidate] = bundle.candidates
        return replace(bundle, candidates=[replace(candidate, **changes)])

    return rogue


def wider(body: str, spans: Any) -> str:
    """The same body under a mask two bytes wider than the one the contract registers."""
    raw = bytearray(body.encode("ascii"))
    for low, high in spans:
        raw[low - 2: high + 2] = MASK_FILL * (high - low + 4)
    return raw.decode("ascii")


async def refusing(
    environment: Any,
    world_: ServedEpisode,
    blobs: Path,
    rogue: Any,
    workflow_id: str,
    *,
    cell: str = GRADED_CELL,
    policy_digest: str = GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
) -> str:
    """Run one generation whose resolver is ``rogue``, and return what refused its candidate."""
    activities = activities_of(world_)
    async with stream_worker(
        environment.client,
        activities=[activities[0], activities[1], rogue, verify_blobs_activity],
    ):
        return await refused(
            environment,
            start_for(
                world_,
                contract_of(world_.env),
                blobs,
                policy_digest=policy_digest,
                cell=cell,
            ),
            blobs,
            workflow_id,
            filing_of(world_.env),
        )


async def test_a_body_that_is_not_the_committed_entry_never_reaches_the_agent(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The authority is the committed reference, so bytes that describe themselves are not enough.

    Three substitutions, each of which a resolver could make and each of which every hash the
    candidate reports about itself would agree with: the oracle body under the graded label, a
    body masked more widely than the contract registers, and one byte moved outside every
    registered slot. The reference the source committed for the selected cell is what refuses
    them, which is why that comparison is not optional.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    seal = seal_id_for()
    async with stream_worker(env.client, activities=activities_of(world)):
        await delivered(
            env,
            start_for(
                world,
                contract,
                blobs,
                policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                cell=GRADED_CELL,
            ),
            blobs,
            "stream/receipt-artifact-committed/1",
            filing_of(world.env),
        )
    held = committed_cells(world, seal)
    spans = mask_spans(contract)

    oracle = await refusing(
        env,
        world,
        blobs,
        substituting(lambda _body: held[ORACLE_CELL]),
        "stream/receipt-artifact-oracle/1",
    )
    assert oracle == "UnusableActivityResult"

    masked = await refusing(
        env,
        world,
        blobs,
        substituting(lambda body: wider(body, spans)),
        "stream/receipt-artifact-wider-mask/1",
    )
    assert masked == "UnusableActivityResult"

    outside = min(low for low, _high in mask_spans(contract)) - 1
    moved = await refusing(
        env,
        world,
        blobs,
        substituting(lambda body: body[:outside] + "Z" + body[outside + 1:]),
        "stream/receipt-artifact-outside/1",
    )
    assert moved == "UnusableActivityResult"


async def test_a_candidate_that_describes_another_cell_or_another_source_is_refused(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The echo is held to the selection, so a correct body under a wrong label is refused too.

    A label is the half a substituted resolver gets right for free, and each of these is a claim
    about which cell of which source the agent was served: the cell, the entry that was resolved,
    the source it came out of, the resolver that read it, and what it came to on the wire. The
    last of those is the pinned ledger's own two sums: this generation serves 2833 bytes and a
    candidate reporting the quoted length's 2835 is refused rather than measured against.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    async with stream_worker(env.client, activities=activities_of(world)):
        await delivered(
            env,
            start_for(
                world,
                contract,
                blobs,
                policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                cell=GRADED_CELL,
            ),
            blobs,
            "stream/receipt-artifact-echo/1",
            filing_of(world.env),
        )
    manifest = await published(
        world, contract, blobs, seal=seal_id_for(), filing=filing_of(world.env)
    )
    assert (
        await refusing(
            env, world, blobs, echoing(cell=PLACEBO_CELL), "stream/receipt-artifact-kind/1"
        )
        == "RendererDescriptorMismatch"
    )
    assert (
        await refusing(
            env,
            world,
            blobs,
            echoing(body_reference=manifest.cells[PLACEBO_CELL].sha256),
            "stream/receipt-artifact-reference/1",
        )
        == "UnusableActivityResult"
    )
    assert (
        await refusing(
            env,
            world,
            blobs,
            echoing(source_commitment="0" * 64),
            "stream/receipt-artifact-source/1",
        )
        == "UnusableActivityResult"
    )
    assert (
        await refusing(
            env,
            world,
            blobs,
            echoing(resolver_id="", resolver_version=""),
            "stream/receipt-artifact-resolver/1",
        )
        == "RendererDescriptorMismatch"
    )
    assert (
        await refusing(
            env,
            world,
            blobs,
            echoing(visible_byte_count=WIRE_BYTES + 2),
            "stream/receipt-artifact-quoted-sum/1",
        )
        == "UnusableActivityResult"
    )


@pytest.mark.parametrize(
    "name,changes",
    [
        ("a source commitment", {"source_commitment": {"sha256": "0" * 64}}),
        ("a body reference", {"body_reference": ["0" * 64]}),
        ("a resolver id", {"resolver_id": {"id": "blob-store-artifact-1"}}),
        ("a resolver version", {"resolver_version": [1]}),
    ],
)
async def test_a_candidate_receipt_value_in_a_shape_this_build_cannot_hold_ends_the_attempt(
    env: Any, world: ServedEpisode, tmp_path: Path, name: str, changes: Dict[str, Any]
) -> None:
    """The four values a resolved candidate echoes are read after the converter, not by it.

    Each of these is an object or a list where a digest or an implementation name belongs, and
    each crosses the real Activity boundary. A field with a type on the result would make every
    one of them a decoding failure raised while the generation was being handed the result: the
    Workflow Task would fail on every retry, the seal would never be answered, and the record
    would say nothing about why. So the result carries them as they were encoded and the strict
    reader decides what they are, and each of these ends the attempt with a reason recorded
    against it, exactly as a moved descriptor does.
    """
    blobs = tmp_path / "blobs"
    workflow_id = f"stream/receipt-artifact-echo-shape/{abs(hash(name)) % 10_000}/1"
    ending = await asyncio.wait_for(
        refusing(env, world, blobs, echoing(**changes), workflow_id), timeout=90
    )
    assert ending == "UnusableActivityResult"


def test_a_candidate_receipt_value_is_made_a_record_of_where_the_result_is_read() -> None:
    """What the reader accepts is text the record can be written in, and nothing else.

    The shapes are refused before any comparison runs, which is what the comparisons cannot do
    for themselves: a route that renders from a projection asks these four to be empty, and an
    empty object is not a digest and is also not something that check would notice. A string
    holding a lone surrogate is refused for the other half of the same reason. It reads back
    cleanly and raises where the carrier is packed, which is outside every recorded refusal
    there is, so the reading and the writing are asked as one question here.
    """
    declared = PayloadCandidateResult(
        cell=GRADED_CELL,
        renderer_id="kernel-receipt-artifact-1",
        match_group="kernel",
        body="a body",
        inner_sha256="1" * 64,
        visible_sha256="2" * 64,
        visible_byte_count=7,
        renderer_version="1",
        policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        source_commitment="3" * 64,
        body_reference="4" * 64,
        resolver_id="blob-store-artifact-1",
        resolver_version="1",
    )
    read = kernel_workflow._read_candidate(declared)
    assert (read.source_commitment, read.body_reference) == ("3" * 64, "4" * 64)
    assert (read.resolver_id, read.resolver_version) == ("blob-store-artifact-1", "1")
    for field in ("source_commitment", "body_reference", "resolver_id", "resolver_version"):
        for value, complaint in (
            ({}, "dict"),
            ([], "list"),
            (7, "int"),
            (None, "NoneType"),
            ("\ud800", "cannot"),
        ):
            with pytest.raises(ApplicationError, match=complaint) as raised:
                kernel_workflow._read_candidate(replace(declared, **{field: value}))
            assert raised.value.type == "UnusableActivityResult"


async def test_a_generation_that_selects_the_oracle_or_has_no_store_never_serves_a_body(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The oracle is retained and undelivered, and a body of the store needs a store.

    An arm selects graded or placebo. The oracle is named by every descriptor, is required by
    publication and by nothing else, and is not a cell a generation may be created to serve: the
    refusal is where the generation is created rather than where a body would have gone out.

    A generation that delivers a committed cell and was given nowhere to keep one is refused at
    its first seal, before a source is captured under it, because there would be nothing to
    install the three cells in and nothing to read one back out of.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    oracle = start_for(
        world,
        contract,
        blobs,
        policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
        cell=ORACLE_CELL,
    )
    async with stream_worker(env.client, activities=activities_of(world)):
        handle = await env.client.start_workflow(
            StreamWorkflow.run,
            oracle,
            id="stream/receipt-artifact-oracle-cell/1",
            task_queue=STREAM_TASK_QUEUE,
        )
        with pytest.raises(WorkflowFailureError) as refusal:
            await handle.result()
        assert protocol_error_code(refusal.value.cause) == "configuration_mismatch"

        assert (
            await refused(
                env,
                start_for(
                    world,
                    contract,
                    blobs,
                    policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                    cell=GRADED_CELL,
                    store=False,
                ),
                blobs,
                "stream/receipt-artifact-no-store/1",
                filing_of(world.env),
            )
            == "UnavailableEvidence"
        )


async def test_a_withheld_position_captures_under_the_contract_its_own_row_names(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """Two contracts, two positions, and a capture that is not conditional on exposure.

    One position delivers a cell of the first contract and the other delivers nothing at all and
    is bound to the second. Both seal, both publish a source, and each is published under the
    contract its own row names: a generation that inferred the only contract would stop working
    here, and a position that captures nothing because it delivers nothing would leave the
    measurement it was created for with no evidence behind it.
    """
    blobs = tmp_path / "blobs"
    first = contract_of(world.env)
    second = replace(first, contract_id="filler_receipt")
    filing = filing_of(world.env)
    start = two_positions(world, first, second, blobs)
    async with stream_worker(env.client, activities=activities_of(world)):
        caller = await opened(env, start, blobs, "stream/receipt-artifact-withheld/1")
        await caller.present(await caller.pull())
        await caller.present(await caller.seal(filing))
        payload = await caller.pull()
        assert payload.kind == "payload"
        await caller.present(payload)
        task = await caller.pull()
        assert task.kind == "task"
        await caller.present(task)
        await caller.present(await caller.seal_of(WITHHELD_ATTEMPT, filing))

    from shogym.serve.protocol_v2.kernel import hidden_seal_id

    delivering_seal = hidden_seal_id(EXECUTION, 0, ATTEMPT)
    withholding_seal = hidden_seal_id(EXECUTION, 0, WITHHELD_ATTEMPT)
    assert (
        held_record(world, delivering_seal)["receipt_contract"]["contract_id"]
        == first.contract_id
    )
    assert (
        held_record(world, withholding_seal)["receipt_contract"]["contract_id"]
        == second.contract_id
    )
    # The withheld position kept its cells and delivered none of them.
    assert sorted(committed_cells(world, withholding_seal)) == sorted(CELL_KINDS)


def two_positions(
    episode: ServedEpisode, first: Any, second: Any, blobs: Path
) -> StreamStart:
    """One generation whose second position is worked and scored and delivered nothing against."""
    from shogym.serve.protocol_v2.kernel import assignments_for

    items = [
        TaskItem(
            task_position=0,
            attempt_id=ATTEMPT,
            task_message_id=oid(0x101),
            ack_message_id=oid(0x102),
            payload_position=0,
            payload_message_id=oid(0x103),
            body=episode.env.describe("0").instructions,
        ),
        TaskItem(
            task_position=1,
            attempt_id=WITHHELD_ATTEMPT,
            task_message_id=oid(0x105),
            ack_message_id=oid(0x106),
            payload_position=1,
            payload_message_id=oid(0x107),
            body=episode.env.describe("0").instructions,
        ),
    ]
    rows = [
        PayloadDisposition(
            attempt_id=ATTEMPT,
            payload_position=0,
            kind=DELIVER,
            policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
            cell=GRADED_CELL,
            resolution_source=REGISTERED,
            family_id=first.contract_id,
        ),
        PayloadDisposition(
            attempt_id=WITHHELD_ATTEMPT,
            payload_position=1,
            kind=WITHHOLD,
            reason="this position is the filler of the arm",
            resolution_source=REGISTERED,
            family_id=second.contract_id,
        ),
    ]
    return replace(
        start_for(
            episode,
            first,
            blobs,
            policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
            cell=GRADED_CELL,
        ),
        tasks=items,
        assignments=assignments_for(items, IMMEDIATE, without_payload=[WITHHELD_ATTEMPT]),
        dispositions=rows,
        provenance=PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(rows),
            experiment_id="the_subject_of_this_run",
        ),
        receipt_contracts=[first, second],
    )


def sealing(seal: Any, doctor: Any) -> Any:
    """A Worker that seals for real and hands the workflow a descriptor it has moved."""

    @activity.defn(name=SEAL_ATTEMPT)
    async def rogue(request: SealAttemptInput) -> Any:
        return doctor(await seal(request))

    return rogue


async def test_a_source_bound_to_another_attempt_or_another_grade_is_refused_by_the_workflow(
    env: Any, world: ServedEpisode, tmp_path: Path
) -> None:
    """The bindings are checked against what this generation is, not against the descriptor.

    A descriptor carries the ordinal rather than the hidden execution id, so the seal id it names
    is recomputed here from this start's own identity and the ordinal it asked under. The rest is
    the same rule applied to the attempt, the grader and the commitment: each is a value this
    generation already holds, and a source that names something else is refused before anything
    of it commits.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    activities = activities_of(world)

    async def refusing_seal(doctor: Any, workflow_id: str) -> str:
        async with stream_worker(
            env.client,
            activities=[
                sealing(activities[0], doctor),
                activities[1],
                generate_payload_bundle_activity,
                verify_blobs_activity,
            ],
        ):
            return await refused(
                env,
                start_for(
                    world,
                    contract,
                    blobs,
                    policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                    cell=GRADED_CELL,
                ),
                blobs,
                workflow_id,
                filing_of(world.env),
            )

    def rebound(sealed: Any) -> Any:
        moved = {**sealed.source_artifact, "source_attempt_id": "f" * 32}
        return replace(
            sealed,
            source_artifact=moved,
            source_commitment=source_commitment(read_source_artifact(moved)),
        )

    def uncommitted(sealed: Any) -> Any:
        return replace(sealed, source_commitment="0" * 64)

    assert await refusing_seal(rebound, "stream/receipt-artifact-rebound/1") == (
        "UnusableActivityResult"
    )
    assert await refusing_seal(uncommitted, "stream/receipt-artifact-uncommitted/1") == (
        "UnusableActivityResult"
    )


def _without(manifest: Dict[str, Any], name: str) -> Dict[str, Any]:
    """The same descriptor with one of its declared names left out."""
    return {key: value for key, value in manifest.items() if key != name}


def _coerced(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The same descriptor with one whole number written as a number that is not whole.

    The value is the size it already was, so nothing else inside the record disagrees with it and
    the commitment is still the commitment. A decoder that made an integer of it would accept the
    descriptor and hash it to the same value, which is the point: what refuses this is the reader
    that will not make a size out of something that was not written as one.
    """
    cells = {
        kind: {**entry, "size": float(entry["size"])} if kind == GRADED_CELL else entry
        for kind, entry in manifest["cells"].items()
    }
    return {**manifest, "cells": cells}


def _unwritable(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """The same descriptor with one textual value the canonical encoding cannot write.

    A lone surrogate is a string Python holds and JSON has no encoding for, and the durable
    service carries it through the raw result exactly as it arrived. A reader that asks only for
    a non-empty string accepts it, and taking the descriptor's canonical bytes then raises at the
    encoder, which is not a boundary anything is recorded at: the attempt would fail the same
    activation for ever with nothing written about why. So the reading and the canonicalization
    are one boundary, and this ends the attempt with a reason like every shape above it.
    """
    return {**manifest, "environment_task_id": "\ud800"}


@pytest.mark.parametrize(
    "name,doctor",
    [
        ("an unknown name", lambda held: {**held, "hidden_execution_id": EXECUTION}),
        ("a declared name left out", lambda held: _without(held, "cells")),
        ("a count written as a fraction of itself", _coerced),
        ("a value the canonical encoding cannot write", _unwritable),
    ],
)
async def test_a_descriptor_written_in_a_shape_this_build_does_not_read_is_refused_and_recorded(
    env: Any, world: ServedEpisode, tmp_path: Path, name: str, doctor: Any
) -> None:
    """The seal result crosses a real converter, and the shape is decided after it, not by it.

    Each of these is a mapping the converter has an answer for. It discards a name this build does
    not declare, it makes a whole number out of one that is not, and a declared name left out is a
    decoding failure raised while the generation is being handed the result, before any code of
    its own runs. The first two would be accepted silently, with the commitment still recomputing
    to itself, and the third would fail the same activation on every retry with the record saying
    nothing about why.

    So the result carries the value and the workflow's own strict reader decides what a descriptor
    is. Every one of these ends the attempt with a reason a reader can find, which is what an
    unknown or missing member being refused at a recorded boundary means.
    """
    blobs = tmp_path / "blobs"
    contract = contract_of(world.env)
    activities = activities_of(world)
    workflow_id = f"stream/receipt-artifact-shape/{abs(hash(name)) % 10_000}/1"

    def moved(sealed: Any) -> Any:
        return replace(sealed, source_artifact=doctor(sealed.source_artifact))

    async with stream_worker(
        env.client,
        activities=[
            sealing(activities[0], moved),
            activities[1],
            generate_payload_bundle_activity,
            verify_blobs_activity,
        ],
    ):
        ending = await asyncio.wait_for(
            refused(
                env,
                start_for(
                    world,
                    contract,
                    blobs,
                    policy_digest=GRADED_RECEIPT_ARTIFACT_V1_DIGEST,
                    cell=GRADED_CELL,
                ),
                blobs,
                workflow_id,
                filing_of(world.env),
            ),
            timeout=90,
        )
    assert ending == "UnusableActivityResult"
