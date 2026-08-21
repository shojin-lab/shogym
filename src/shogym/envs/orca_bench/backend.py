"""The seam where the ORCA-bench compose backend plugs in: interface and pins only.

Running an ORCA-bench task means bringing up the benchmark's recorded observability stack (28
services replaying a frozen telemetry snapshot behind Grafana), letting an agent query it, and
then running the task's own verifier over the report it wrote. That backend is **phase 2** of
this port (issue #77). The calls it needed are settled and recorded in the env README; it is not
implemented here, and this module contains no Docker code.

What is here is the shape it must fill, and the measurements phase 1 established so phase 2 does
not have to rediscover them:

  - :class:`OrcaBackend`, the protocol the env drives: bring a task's stack up, act on the
    agent's container, run the verifier, tear down.
  - :data:`SNAPSHOT_IMAGE` / :data:`SNAPSHOT_IMAGE_DIGEST`, the published environment image,
    pinned **by digest**. The upstream repo cannot rebuild it (six files its setup references are
    absent, and the published entrypoint diverges from the repo template by one load-bearing
    line), so the image is treated as the artifact and the repo as documentation.
  - the staging contract: the image carries a ~46 GB snapshot cache under ``/app`` that each
    trial hard-links from. Upstream bind-mounts a host directory at an identical inside/outside
    path; the phase-2 design stages it into a **named Docker volume** once instead, which removes
    both the host-side copy and the checksum race that upstream's runner patches around.

Constructing a backend raises until phase 2 lands, so an env can be built and described today
and only *serving* an episode fails, with a message that says why.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable

from shogym.envs.orca_bench.judge import CapturedReport

# The environment image every task's Dockerfile builds from, pinned by digest. The tag is
# mutable; the digest is what makes two runs comparable. Why the repo cannot re-derive this image,
# and the measurements below, are in issue #77.
SNAPSHOT_IMAGE = "orcabench/sre-otel-snapshot:data-0418-harbor-template"
SNAPSHOT_IMAGE_DIGEST = "sha256:19c8c097ec10be561d6fd49c9b0fff0c6188b583bcb41ec1c5945d7f5fdbd671"
SNAPSHOT_IMAGE_PINNED = f"{SNAPSHOT_IMAGE.split(':')[0]}@{SNAPSHOT_IMAGE_DIGEST}"

# The env var the task entrypoint requires: where the staged snapshot cache is readable from.
# Upstream sets it to a host directory bind-mounted at the same path inside and out (so the
# entrypoint's `cp -al` needs no path translation); phase 2 supplies a named volume instead.
SNAPSHOT_CACHE_ENV = "SNAPSHOT_CACHE_HOST_DIR"
# The named volume phase 2 stages the snapshot cache into, once per host.
SNAPSHOT_VOLUME = "shogym-orca-snapshot-cache"

# Where the agent works inside the task container, and the file the verifier grades.
WORKDIR = "/app"
REPORT_PATH = f"{WORKDIR}/report.md"
SOURCE_PATH = f"{WORKDIR}/opentelemetry-demo"

# Measured on one host during the feasibility spike, for sizing rather than for assertion:
# ~86.8 GB image + ~46 GB snapshot cache before the first trial, ~3.4 GB per trial, 4.10 GiB RSS
# for the running stack, ~135 s warm start, ~8 s teardown. A hosted CI runner has ~14 GB of disk,
# which is why this env's live path can never run there (see the env README).
DISK_REQUIRED_GB = 133
STACK_RSS_GIB = 4.10
WARM_START_SECONDS = 135


class BackendUnavailableError(RuntimeError):
    """No backend can run an ORCA-bench episode in this build (phase 2 is not implemented)."""


@runtime_checkable
class OrcaBackend(Protocol):
    """What the env needs from whatever runs a task's stack.

    Deliberately the same four verbs the frontier_bench Docker backend exposes plus a verifier
    run, so phase 2 is a new implementation rather than a new env shape: ``exec`` /
    ``read_file`` / ``write_file`` are the agent's world, and ``run_verifier`` produces the
    files :func:`shogym.envs.orca_bench.judge.read_verdict` parses.
    """

    def exec(self, command: str, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Run one shell command in the agent's container; return ``{ok, exit_code, stdout, stderr}``."""
        ...

    def read_file(self, path: str) -> Dict[str, Any]:
        """Read a file from the agent's container; return ``{ok, content}``."""
        ...

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        """Write a file into the agent's container; return ``{ok}``."""
        ...

    def capture_report(self) -> CapturedReport:
        """Read the agent's report **once**, at the seal, and hold those exact bytes.

        Required, and required in this shape. Sealing an episode stops the agent's MCP calls; it
        does not stop processes the agent already started inside its container. An implementation
        that validates :data:`REPORT_PATH` and then lets the verifier reopen it grades whatever is
        there the second time, so a watcher started before the seal can delete the report in
        between and recreate the judge-error exclusion this port filters results on, or swap the
        bytes so the score belongs to something that did not exist when the episode was sealed.
        Statting and letting the verifier reopen is therefore not sufficient.

        So: read the report through a single descriptor (see
        :func:`shogym.envs.orca_bench.judge.capture_report`, which does exactly this for a local
        path), classify it against those bytes, and put them somewhere no agent process can
        reach: outside the agent-writable mount, on the host or in a path only the verifier
        container sees. The captured bytes are the submission, and nothing else is."""
        ...

    def run_verifier(self, captured: CapturedReport) -> Dict[str, Any]:
        """Run the task's own verifier over **exactly** the captured bytes.

        Not over :data:`REPORT_PATH`: the verifier must be pointed at the copy taken at the seal
        (``--predictions`` at the captured file), or the reopen that
        :meth:`capture_report` exists to prevent comes back one layer down.

        Returns the two parsed payloads the judge writes, as ``{"reward": …, "details": …}``
        (either may be ``None``), which :func:`shogym.envs.orca_bench.judge.parse_verdict` turns
        into a verdict, including the explicit judge-error grade. Called only when the capture
        found nothing wrong with the submission; a capture that did is a graded zero and never
        reaches the judge."""
        ...

    def teardown(self) -> None:
        """Stop and remove everything this backend started. Idempotent."""
        ...


def create_backend(task_dir: Any, **_kwargs: Any) -> OrcaBackend:
    """Bring up a backend for one task. Raises until the compose backend lands (phase 2).

    Phase 1 ships the dataset loader, the redacted task contract, the judge preflight, and the
    verdict parsing: everything that is correct to build and test without the stack. Serving an
    episode needs the stack itself, including the pinned clock that restores the benchmark's
    expired snapshot lookback (see the env README).
    """
    raise BackendUnavailableError(
        "the orca_bench compose backend is not implemented yet: this port's phase 1 covers "
        "describe / dataset loading / judge preflight / verdict parsing, all offline. Running an "
        f"episode needs the {SNAPSHOT_IMAGE} stack (~{DISK_REQUIRED_GB} GB of disk, "
        f"~{WARM_START_SECONDS} s warm start), which is phase 2 of this port (issue #77)."
    )


__all__ = [
    "BackendUnavailableError",
    "CapturedReport",
    "DISK_REQUIRED_GB",
    "OrcaBackend",
    "REPORT_PATH",
    "SNAPSHOT_CACHE_ENV",
    "SNAPSHOT_IMAGE",
    "SNAPSHOT_IMAGE_DIGEST",
    "SNAPSHOT_IMAGE_PINNED",
    "SNAPSHOT_VOLUME",
    "SOURCE_PATH",
    "STACK_RSS_GIB",
    "WARM_START_SECONDS",
    "WORKDIR",
    "create_backend",
]
