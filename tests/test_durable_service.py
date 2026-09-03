"""Where the embedded service keeps a generation's history, and what that placement decides.

The service keeps it in a SQLite file. One path shared by every run is one write lock, and the
second serving process to start dies on a busy database. The file belongs to the run, so the
first test starts two generations at the same time, each with its own run directory, and reads a
task record out of both.

The same placement decides where a resume looks. A run's history is in that run's directory, so the
client that takes the generation over is opened against that directory and a client opened
without it holds an empty database and reaches nothing. The second test is that pair, across a
service that really has gone away: it leaves a Task offered and unpresented, drops the Worker and
the service, and takes the generation over out of the directory that holds it.

They are marked ``network`` because the first embedded service downloads its binary, and they
skip when no service starts at all, which is what an offline machine looks like. A machine that
started one and failed on the other is the failure these tests exist for, so that is not a skip.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import List

import pytest
from temporalio.service import RPCError

from shogym.serve.episode import ServedEpisode
from shogym.serve.protocol_v2 import FilesystemBlobStore, PullRequest
from shogym.serve.protocol_v2.gateway import (
    durable_client,
    open_gateway,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel import (
    STREAM_TASK_QUEUE,
    ConsumerClaim,
    StreamStart,
    TaskItem,
    TerminalTool,
    configuration_hash,
    resume_run_directory,
    start_stream,
)
from shogym.serve.protocol_v2.kernel.runtime import TEMPORAL_ADDRESS_ENV
from shogym.serve.protocol_v2.rundir import (
    MANIFEST_FILE,
    create_run_directory,
    open_run_directory,
)

TEST_ENV = "wordle_v1"

CLAIM = ConsumerClaim(consumer_id="harness-1", claim_hash="d" * 64)
RESUMED = "stream/durable-resume/1"


def _oid(value: int) -> str:
    return f"{value:032x}"


async def _serve_one_task(run_directory: Path, started: List[str]) -> str:
    """Bring up one generation in its own directory and return the attempt it offers."""
    async with durable_client(run_directory=run_directory) as client:
        started.append(str(run_directory))
        async with stream_worker(client):
            episode = await ServedEpisode.start(TEST_ENV, task=0, ends_on_horizon=False)
            try:
                gateway = await open_gateway(client, episode, run_directory=run_directory)
                await gateway.close_queue()
                record = json.loads(await gateway.pull({}))
                assert record["kind"] == "task"
                return record["attempt_id"]
            finally:
                await episode.close()


@pytest.mark.network
async def test_two_generations_serve_at_the_same_time(tmp_path: Path) -> None:
    """Two run directories, two services, two attempts, and neither one waits for the other."""
    if os.environ.get(TEMPORAL_ADDRESS_ENV):
        pytest.skip("a named service is one service, and the database per run is what is tested")
    first = tmp_path / "one"
    second = tmp_path / "two"
    started: List[str] = []
    try:
        one, other = await asyncio.gather(
            _serve_one_task(first, started), _serve_one_task(second, started)
        )
    except Exception as error:  # noqa: BLE001 - re-raised below unless no service came up
        if started:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")
    assert one != other
    for directory in (first, second):
        # Each run holds its own history beside its own manifest, and the manifest still reads
        # back.
        assert [path for path in directory.glob("*.sqlite") if path.is_file()]
        assert (directory / MANIFEST_FILE).is_file()
        assert open_run_directory(directory).manifest.workflow_id.startswith("stream/")


def _one_task_start(root: Path) -> StreamStart:
    """One task, every public identifier fixed before anything is served."""
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash=CLAIM.claim_hash,
        initial_cursor=_oid(1),
        done_message_id=_oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id="execution-1",
        canonicalization_version="kernel.1",
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=["answer"]
        ),
        tasks=[
            TaskItem(
                task_position=0,
                attempt_id=_oid(0x100),
                task_message_id=_oid(0x101),
                ack_message_id=_oid(0x102),
                payload_position=0,
                payload_message_id=_oid(0x103),
                body="file the report",
            )
        ],
        blob_root=str(FilesystemBlobStore.under(root).root),
    )


@pytest.mark.network
async def test_a_generation_comes_back_out_of_the_directory_that_holds_it(tmp_path: Path) -> None:
    """The run holds the history, so the run's directory is what a new owner opens.

    The Task is left offered and unpresented, and then the Worker and the service that offered it
    both go away, which is the case the directory exists for. A client opened without that
    directory starts a database of its own and reaches nothing, and one opened against it takes
    the generation over and replays the same offer.
    """
    if os.environ.get(TEMPORAL_ADDRESS_ENV):
        pytest.skip("a named service holds every run, and where a run holds its own is the point")
    root = tmp_path / "run"
    start = _one_task_start(root)
    request = PullRequest(request_id=_oid(0x1001), last_presented_cursor=_oid(1))
    started = False
    try:
        async with durable_client(run_directory=root) as client:
            started = True
            async with stream_worker(client, cached_workflows=0):
                create_run_directory(
                    root,
                    workflow_id=RESUMED,
                    task_queue=STREAM_TASK_QUEUE,
                    configuration_hash=configuration_hash(start),
                )
                stream = await start_stream(client, start, workflow_id=RESUMED)
                receipt = await stream.claim_consumer(CLAIM)
                assert receipt.initial_cursor == request.last_presented_cursor
                offered = await stream.pull(request)
                assert offered.kind == "task"
    except Exception as error:  # noqa: BLE001 - re-raised below unless no service came up
        if started:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")

    # The form that reads the run directory and not the run's history. It finds the manifest,
    # asks its own empty database for the generation the manifest names, and is told there is
    # no such workflow.
    async with durable_client() as elsewhere:
        with pytest.raises(RPCError):
            await resume_run_directory(elsewhere, root, start=start)

    async with durable_client(run_directory=root) as client:
        async with stream_worker(client, cached_workflows=0):
            taken = await resume_run_directory(
                client, root, start=start, claimant_id="the-next-owner"
            )
            state = await taken.stream_state()
            # The epoch this owner claimed, the cursor nothing had moved, and the one offer the
            # last owner made, replayed to its exact request as the same bytes rather than
            # minted again.
            assert state.ownership_epoch == 2
            assert state.cursor == receipt.initial_cursor
            assert (state.offer_count, state.presentation_count) == (1, 0)
            assert await taken.pull(request) == offered
