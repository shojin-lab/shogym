"""Fetch everything the offline suite needs, in the one stage that is allowed to fetch it.

The suite reaches the network in five places: three envs provision a SHA-pinned upstream source on
first import, the tau2 envs read domain data that no source carries, the frontier_bench Docker
tests build task images whose Dockerfiles install from apt, PyPI and astral.sh, one vendored
oracle installs its own packages while it runs, and the durable tests start a Temporal test server
the SDK downloads on first use. All five used to happen inside the run that calls itself offline,
where a slow mirror or a bad minute at a third party reads as a failing test, or as a skip that
quietly deletes a module's coverage.

So they happen here instead. Run this with ``SHOGYM_PROVISIONING=prepare`` before the suite, and
the suite itself with ``SHOGYM_PROVISIONING=offline``: what is prepared is used, and what is not
prepared fails by name rather than being fetched. Preparing is idempotent, so a warm machine pays
almost nothing.

    uv run python tests/prepare_offline_suite.py

Docker is optional. A machine with no daemon skips the images and the oracle packages, which is
the same machine on which every test that would need one is skipped, so the two agree.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

# This script is run by path, so the repository root is not on `sys.path` the way pytest puts it
# there. The preparation of the Temporal test server is shared with the fixture that requires it,
# and shared code beats a second copy of the SDK's naming convention.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shogym.envs import _upstream  # noqa: E402
from shogym.envs.frontier_bench import docker_backend as dk  # noqa: E402
from shogym.envs.frontier_bench import manifest, mcp_server  # noqa: E402
from shogym.envs.tau2 import adapter as tau2_adapter  # noqa: E402
from tests._fixtures.temporal_server import prepare_test_server  # noqa: E402

#: The adapter module of each env whose upstream source is provisioned at runtime. Importing it
#: and calling its ``ensure_source`` is what the gated test modules do on their first import, so
#: preparing means doing it once, here, where a download is expected.
UPSTREAM_ADAPTERS = (
    "shogym.envs.tau2.adapter",
    "shogym.envs.yc_bench.adapter",
    "shogym.envs.automationbench.adapter",
)


def prepare_sources() -> None:
    """Provision every pinned upstream source, so the suite can bind them without a network."""
    for module_name in UPSTREAM_ADAPTERS:
        module = importlib.import_module(module_name)
        source = module.ensure_source()
        print(f"prepared source: {module_name} -> {source}", flush=True)


def prepare_data() -> None:
    """Provision the upstream data a source does not carry, which today is tau2's domains.

    tau2's envs are made of it: the policy, the task set and the databases of each domain live in
    upstream's ``data/``, which the source extraction drops because it is ~700 MB of which the
    envs read 139. Without it every tau2 test that constructs an env used to skip, in every mode,
    which is the failure this stage exists to turn into either a prepared machine or a named
    one."""
    print(f"prepared data: tau2 -> {tau2_adapter.ensure_data()}", flush=True)


def prepare_durable_service() -> None:
    """Download the Temporal test server binary the durable tests run against."""
    print(f"prepared binary: {asyncio.run(prepare_test_server())}", flush=True)


def prepare_images() -> None:
    """Build every frontier_bench task image the Docker-gated tests use, environment and verifier,
    and fetch the packages each task's own oracle installs while it runs.

    Both images, because a scored episode builds both: the environment image once per task, and
    the verifier image at each finalization. Preparing the verifier image too is what lets a run
    that may not fetch use the image it was given instead of rebuilding one.

    The oracle packages are the one download that happens *inside* a task container rather than
    while building one, so no image can hold them without changing the world the agent is given.
    They are fetched here as files the oracle installs from instead.
    """
    if not dk.docker_available():
        print("no Docker daemon; skipping task images (their tests skip too)", flush=True)
        return
    for name in manifest.task_names():
        print(f"prepared image: {mcp_server.build_task_image(name)}", flush=True)
        print(f"prepared image: {mcp_server.build_verifier_image(name)}", flush=True)
        packages = mcp_server.prepare_oracle_packages(name)
        if packages is not None:
            print(f"prepared oracle packages: {name} -> {packages}", flush=True)


def main() -> int:
    mode = _upstream.provisioning_mode()
    if mode == _upstream.OFFLINE:
        print(
            f"{_upstream.PROVISIONING_ENV_VAR}={mode} forbids the downloads this stage exists to "
            f"perform; run it with {_upstream.PROVISIONING_ENV_VAR}={_upstream.PREPARE}.",
            file=sys.stderr,
        )
        return 2
    prepare_sources()
    prepare_data()
    prepare_durable_service()
    prepare_images()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
