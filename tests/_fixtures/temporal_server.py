"""The Temporal test server the durable tests run against, as something a stage prepares.

``WorkflowEnvironment.start_time_skipping()`` is lazy about its binary: called with no arguments
it looks in the system temporary directory for ``temporal-test-server-sdk-python-<version>`` and,
finding nothing, fetches it from Temporal's download service (64 MB) before the first test runs.
On a cold runner that is a download inside the run, which is exactly what the provisioning mode
exists to move out of it, and on a runner where the service has a bad minute it used to be four
test modules skipping for a reason that had nothing to do with them.

So the binary is prepared like every other fetched thing. :func:`prepare_test_server` downloads
it into ``~/.cache/shogym/temporal``, beside the dev-server binary the serve layer already caches
there, and :func:`time_skipping_environment` starts *that* file rather than looking for one. Under
``offline`` a binary nobody prepared fails the test naming the path it was looked for at; under
``local`` it is downloaded on demand, as it always was, into the cache instead of a temp dir that
the machine may clear between runs.
"""

from __future__ import annotations

import os
from pathlib import Path

import temporalio.service
from temporalio.testing import WorkflowEnvironment

from shogym.envs._upstream import provisioning_mode
from shogym.serve.protocol_v2.kernel.runtime import temporal_home
from tests._fixtures.upstream_gate import require_prepared_path


def test_server_path() -> Path:
    """Where the prepared test server binary lives.

    The name is the SDK's own: its core builds ``<prefix>-<sdk name>-<sdk version>`` and looks for
    that file before downloading, so a binary written here under that name is the one a later run
    finds. It is keyed by the SDK version for the same reason the sources are keyed by their SHA:
    a dependency bump must not be answered with the previous version's binary."""
    suffix = ".exe" if os.name == "nt" else ""
    version = temporalio.service.__version__
    return temporal_home() / f"temporal-test-server-sdk-python-{version}{suffix}"


async def prepare_test_server() -> Path:
    """Download the test server binary into the shogym cache and return its path. Idempotent.

    Starting the environment is the download: the SDK fetches the binary the first time one is
    asked for and reuses it afterwards. Starting it also proves the file runs, which a downloaded
    archive on its own does not.

    The mode is read before the binary is looked for, so a machine that already holds one still
    refuses a value nobody recognizes, exactly as the source and data paths do."""
    provisioning_mode()
    path = test_server_path()
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    environment = await WorkflowEnvironment.start_time_skipping(
        download_dest_dir=str(path.parent)
    )
    await environment.shutdown()
    if not path.is_file():
        raise RuntimeError(
            f"the Temporal test server was downloaded but is not at {path}; the SDK's naming "
            f"convention for it has changed and this fixture has to change with it."
        )
    return path


async def time_skipping_environment() -> WorkflowEnvironment:
    """Start a time-skipping workflow environment on the prepared binary.

    Under ``offline`` an absent binary fails here, naming the path and the stage that fills it,
    rather than being downloaded from inside a run that promised to fetch nothing."""
    path = test_server_path()
    require_prepared_path(path, what="the Temporal test server")
    if path.is_file():
        return await WorkflowEnvironment.start_time_skipping(test_server_existing_path=str(path))
    # Only the modes that may fetch reach this line, the gate above having failed the one that
    # may not: fetch on demand, as this always did, into the cache rather than a temp dir.
    path.parent.mkdir(parents=True, exist_ok=True)
    return await WorkflowEnvironment.start_time_skipping(download_dest_dir=str(path.parent))


__all__ = ["prepare_test_server", "test_server_path", "time_skipping_environment"]
