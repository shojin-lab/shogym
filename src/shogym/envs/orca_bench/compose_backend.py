"""The live Docker backend for ``orca_bench``: one task's recorded stack, brought up for real.

This is phase 2 of the port (issue #77), the implementation of the contract
:mod:`shogym.envs.orca_bench.backend` declares. What it drives is unusual enough to be worth
stating plainly, because every design choice here follows from it:

The benchmark ships **one image** that is itself a runner. Its entrypoint materializes a
per-trial copy of the recorded telemetry, starts a 28-service compose project on the **host**
daemon through a mounted socket (docker-out-of-docker, so the services are siblings of the agent
container rather than children), joins that project's network, restores the OpenSearch snapshot,
and only then hands off to the agent command. So this module does not orchestrate the stack. It
prepares what the entrypoint reads, starts that container, waits for the readiness marker the
entrypoint writes, and tears the whole tree down afterwards.

Three things follow from that, and they are the substance of this module:

  - **Staging.** The image carries ~46 GB of snapshot cache under ``/app``. Upstream copies it to
    a host directory and bind-mounts it at an identical path inside and out. Copying it into a
    **named volume** instead keeps every byte inside the daemon's own filesystem: the volume's
    mountpoint is a path the daemon can bind-mount for the sibling services, which is what makes
    the identical-path requirement hold without a host round-trip. Staged once per image digest
    and shared by every trial.
  - **The clock.** The recorded data is from 2026-04-19..23 and Jaeger's published lookback is 90
    days, so on any machine after ~2026-07-22 the service list comes back empty and no run is
    comparable to the paper's. Two runtime knobs restore it, and they are one decision: Jaeger
    sends one index name per day of lookback in the request line, so the widened lookback only
    works alongside a raised OpenSearch request-line limit. Both land in the staged cache the
    entrypoint copies out, so the image is untouched. See :func:`install_clock_override` for the
    two mechanisms this replaced and why they are dead.
  - **Teardown.** The inner services are siblings, so they outlive the outer container unless
    something kills them. The entrypoint traps for that; this module verifies it and sweeps
    whatever is left, because a leaked project holds gigabytes and a port.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from shogym.envs.orca_bench.backend import (
    REPORT_PATH,
    SNAPSHOT_CACHE_ENV,
    SNAPSHOT_IMAGE_DIGEST,
    SNAPSHOT_IMAGE_PINNED,
    SNAPSHOT_VOLUME,
    BackendUnavailableError,
)
from shogym.envs.orca_bench.judge import CapturedReport, ResolvedJudge, capture_report

_DOCKER = "docker"

# The image is published for linux/amd64 only. On an arm64 host the daemon refuses the pull
# without an explicit platform ("no matching manifest for linux/arm64/v8"), and every container
# then runs under emulation. Stated rather than hidden: it works, and it is slower.
IMAGE_PLATFORM = "linux/amd64"

# Where the staged cache and the per-trial context live inside the daemon. Both are named volumes
# mounted at their own mountpoint path, so a path means the same thing to the outer container and
# to the daemon starting the sibling services (see the module docstring).
CONTEXT_VOLUME_PREFIX = "shogym-orca-context"

# The marker the image's entrypoint writes once the stack is up and the snapshot restored.
READY_MARKER = "/tmp/env-ready"
PORTS_FILE = "/tmp/env-ports"

# Measured on this port's own runs; see the env README. The stack is 28 services on an emulated
# amd64 daemon, so these are generous rather than tight.
DEFAULT_START_TIMEOUT_SECONDS = 900.0
DEFAULT_EXEC_TIMEOUT_SECONDS = 600.0
DEFAULT_TEARDOWN_TIMEOUT_SECONDS = 300.0
DEFAULT_VERIFIER_TIMEOUT_SECONDS = 1800.0

# The staging cost, before the first trial. The image is ~87 GB and the `/app` cache it carries is
# ~46 GB, and the two hazards are separate: a *pull* needs room for both, while a *copy* onto a
# host that already holds the image needs room for the second alone. Guarding only the pull would
# leave the copy reachable on a full daemon, which is the same accident with a smaller number.
STAGING_DISK_BYTES = 140 * 1024**3
SNAPSHOT_CACHE_BYTES = 50 * 1024**3

# ----- the clock (issue #77) -----
#
# The recorded telemetry is from 2026-04-19..23 and the published Jaeger config sets
# `max_span_age: 2160h` (90 days), so from about 2026-07-22 onward a live stack answers
# `GET /api/services` with an empty list and an agent's first move sees a system with no services.
# The fix is two runtime knobs, both scoped to a compose override and a config file the entrypoint
# already reads out of the staged cache, so the image and its digest are untouched.
#
# The two numbers are **one decision, not two**. Jaeger's OpenSearch reader turns the lookback into
# one daily index name per day in the window and puts all of them in the request line, so a longer
# lookback buys a longer URL: at ~40 bytes per index name, 10 years is ~3650 names and ~146 KB of
# request line, against OpenSearch's 4 KB default. Raising only the lookback is what fails with
# `too_long_http_line_exception`; raising only the limit changes nothing. Both, together, work.
PUBLISHED_LOOKBACK = "2160h"
SNAPSHOT_LOOKBACK = "87600h"  # 10 years, so the snapshot stays reachable from any plausible run date
OPENSEARCH_MAX_LINE = "512kb"  # ~3.5x the request line the lookback above actually spends
PUBLISHED_JAEGER_CONFIG = "jaeger-config-snapshot.yml"
SHADOW_JAEGER_CONFIG = "jaeger-config-shogym.yml"
# Mounted beside the published config rather than over it: shadowing a path the base compose file
# already binds would make two mounts fight for one target, and `--config` is a clean override.
CONTAINER_JAEGER_CONFIG = "/etc/jaeger/config-shogym.yml"

CLOCK_OVERRIDE_COMMENT = """\
# Appended by shogym's orca_bench port (issue #77). The recorded telemetry is from 2026-04-19..23
# and the published max_span_age is 2160h, so from ~2026-07-22 the service list comes back empty
# and no run is comparable to the paper's numbers. Two runtime knobs restore the pre-expiry
# behaviour without editing the image: OpenSearch accepts a longer request line, and Jaeger reads a
# shadow copy of its own config whose lookback reaches the snapshot. Neither knob alone is enough.
"""


class DockerError(RuntimeError):
    """A ``docker`` invocation failed."""


def _run(
    args: Sequence[str],
    *,
    timeout: Optional[float] = None,
    check: bool = True,
    capture: bool = True,
    env: Optional[Mapping[str, str]] = None,
) -> subprocess.CompletedProcess:
    """Run one docker command.

    ``env`` is added to the CLI's own environment, which is how a secret reaches a container
    without ever being written down: ``docker run -e NAME`` with no value forwards the variable
    from here, whereas ``-e NAME=value`` would put it in this process's argv, where any local
    ``ps`` can read it for as long as the call runs."""
    try:
        proc = subprocess.run(
            [_DOCKER, *args],
            text=True,
            capture_output=capture,
            timeout=timeout,
            env=None if env is None else {**os.environ, **env},
        )
    except FileNotFoundError as exc:
        raise DockerError("docker CLI not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        # A timeout is an outcome of the call, not a different kind of event: the daemon is busy,
        # which under emulation it routinely is while 28 services start. Callers that pass
        # `check=False` are asking to handle failure themselves, and a raised TimeoutExpired would
        # bypass them and take down a run that only needed to wait longer. 124 is the shell's own
        # convention for it.
        if check:
            raise DockerError(f"`docker {' '.join(args[:3])} …` timed out after {timeout}s") from exc
        return subprocess.CompletedProcess(list(args), 124, "", "timed out")
    if check and proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip() if capture else ""
        raise DockerError(f"`docker {' '.join(args[:3])} …` failed ({proc.returncode}): {detail}")
    return proc


def docker_available() -> bool:
    """True iff a working daemon answers."""
    try:
        return _run(["info", "--format", "{{.ServerVersion}}"], timeout=60, check=False).returncode == 0
    except DockerError:
        return False


def daemon_free_bytes() -> Optional[int]:
    """Free bytes on the daemon's own filesystem, or ``None`` if it cannot be measured.

    The number that matters is the daemon's, not the client's: on Docker Desktop the images live
    in a VM with its own disk, and a host with 200 GB free can still be a daemon with none."""
    try:
        proc = _run(
            ["run", "--rm", "--platform", IMAGE_PLATFORM, "alpine:latest", "df", "-kP", "/"],
            timeout=300,
            check=False,
        )
    except DockerError:
        return None
    if proc.returncode != 0:
        return None
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    fields = lines[-1].split()
    if len(fields) < 4:
        return None
    try:
        return int(fields[3]) * 1024
    except ValueError:
        return None


def image_present(reference: str = SNAPSHOT_IMAGE_PINNED) -> bool:
    return _run(["image", "inspect", reference], timeout=120, check=False).returncode == 0


def ensure_image(*, reference: str = SNAPSHOT_IMAGE_PINNED, pull_timeout: float = 7200.0) -> None:
    """Ensure the pinned image is present, pulling it by digest if it is not.

    Refuses to start a pull that the daemon plainly cannot finish: the image plus the cache it is
    staged into is ~140 GB, and a pull that fills the disk takes the daemon down with it rather
    than failing politely."""
    if image_present(reference):
        return
    free = daemon_free_bytes()
    if free is not None and free < STAGING_DISK_BYTES:
        raise BackendUnavailableError(
            f"the docker daemon has {free / 1024**3:.0f} GB free and staging "
            f"{SNAPSHOT_IMAGE_PINNED} needs about {STAGING_DISK_BYTES / 1024**3:.0f} GB (the "
            "image plus the snapshot cache it is copied into). Free space and retry."
        )
    _run(
        ["pull", "--platform", IMAGE_PLATFORM, reference],
        timeout=pull_timeout,
        capture=False,
    )


# ----- staging -----


def volume_mountpoint(name: str) -> str:
    """Where a named volume's data lives on the daemon's filesystem.

    This is the address that makes the whole staging design work: the outer container mounts the
    volume *at this same path*, so a path written into a compose file inside the container names
    the same bytes when the daemon bind-mounts it for a sibling service."""
    proc = _run(["volume", "inspect", "--format", "{{.Mountpoint}}", name], timeout=60)
    return proc.stdout.strip()


def _volume_exists(name: str) -> bool:
    return _run(["volume", "inspect", name], timeout=60, check=False).returncode == 0


def snapshot_volume_name(digest: str = SNAPSHOT_IMAGE_DIGEST) -> str:
    """One staged cache per image digest, so a re-pin stages beside the old one."""
    return f"{SNAPSHOT_VOLUME}-{digest.split(':')[-1][:12]}"


def _staged_marker(mountpoint: str) -> str:
    return f"{mountpoint}/.shogym-staged"


def ensure_snapshot_cache(
    *, digest: str = SNAPSHOT_IMAGE_DIGEST, timeout: float = 3600.0
) -> str:
    """Stage the image's ``/app`` into a named volume once; return the volume's mountpoint.

    Upstream copies this ~46 GB tree to a host directory. Inside a named volume it never leaves
    the daemon's filesystem, which is both far faster and what lets the sibling services
    bind-mount it at a path that means the same thing on both sides.

    Idempotent: a marker file inside the volume records that the copy finished, so an interrupted
    staging is redone rather than half-trusted."""
    name = snapshot_volume_name(digest)
    if not _volume_exists(name):
        _run(["volume", "create", name], timeout=120)
    mountpoint = volume_mountpoint(name)
    marker = _staged_marker(mountpoint)
    probe = _run(
        [
            "run", "--rm", "--platform", IMAGE_PLATFORM,
            "--mount", f"type=volume,source={name},target={mountpoint}",
            "alpine:latest", "test", "-f", marker,
        ],
        timeout=300,
        check=False,
    )
    if probe.returncode == 0:
        return mountpoint

    ensure_image()
    free = daemon_free_bytes()
    if free is not None and free < SNAPSHOT_CACHE_BYTES:
        raise BackendUnavailableError(
            f"the docker daemon has {free / 1024**3:.0f} GB free and copying "
            f"{SNAPSHOT_IMAGE_PINNED}'s snapshot cache into a volume needs about "
            f"{SNAPSHOT_CACHE_BYTES / 1024**3:.0f} GB. Free space and retry."
        )
    _run(
        [
            "run", "--rm", "--platform", IMAGE_PLATFORM,
            "--mount", f"type=volume,source={name},target={mountpoint}",
            "--entrypoint", "sh",
            SNAPSHOT_IMAGE_PINNED,
            "-c",
            # `cp -a` inside the daemon's filesystem: no host round-trip, ownership preserved.
            f"set -e; rm -rf {mountpoint}/* {mountpoint}/.[!.]* 2>/dev/null || true; "
            f"cp -a /app/. {mountpoint}/; touch {marker}",
        ],
        timeout=timeout,
        capture=False,
    )
    return mountpoint


def clock_override_yaml() -> str:
    """The compose override that restores the pre-expiry telemetry window.

    Pure so it can be read and tested without a daemon, because what it says is the decision:

    - ``opensearch`` gets a longer HTTP request line. Jaeger names one index per day of lookback
      and sends them all in the URL, so the 4 KB default is what makes a long lookback fail.
    - ``jaeger`` is pointed at a **shadow** copy of its own config with the lookback widened.
      ``command`` replaces rather than appends when compose merges, so this is the only
      ``--config`` the service sees, and the published file is left mounted and unread.

    The staged cache is named by ``${SNAPSHOT_CACHE_HOST_DIR}`` rather than by the path this
    process happens to see, because compose expands it on the daemon at up time. That is what lets
    a sibling service bind-mount it, and it keeps a host-specific path out of the staged file."""
    return (
        f"\n{CLOCK_OVERRIDE_COMMENT}"
        "  opensearch:\n"
        "    environment:\n"
        f"      - http.max_initial_line_length={OPENSEARCH_MAX_LINE}\n"
        "  jaeger:\n"
        "    command:\n"
        f"      - --config=file:{CONTAINER_JAEGER_CONFIG}\n"
        "    volumes:\n"
        f"      - ${{{SNAPSHOT_CACHE_ENV}}}/{SHADOW_JAEGER_CONFIG}:{CONTAINER_JAEGER_CONFIG}:ro\n"
    )


def install_clock_override(mountpoint: str, *, timeout: float = 300.0) -> None:
    """Write the shadow Jaeger config and append the compose override, once.

    Both land inside the staged cache, which the entrypoint copies out and hands to
    ``docker compose`` as the second ``-f``, so this is the one place a service can be
    reconfigured without editing the image or the entrypoint.

    The shadow config is derived from the published one by substituting the lookback, so it tracks
    every other setting in that file rather than restating it: a re-pin that changes the storage
    backend does not silently keep an old copy. Marker-guarded, because staging is shared.

    Two mechanisms were tried before this one and are recorded because the decision record was
    wrong about both. ``libfaketime`` (the mechanism issue #77 originally chose) is **inert**: the
    Jaeger query service is ``jaegertracing/jaeger:2.12.0``, a statically linked Go binary the
    loader refuses to preload into at all (``/lib/ld-musl-x86_64.so.1: /cmd/jaeger/jaeger-linux:
    Not a valid dynamic program``), so a live run with the pin installed still saw no services.
    Widening the lookback **alone**, the option that decision rejected as re-expiring later, fails
    immediately instead: at 2760h and 4000h the query dies with ``An HTTP line is larger than 4096
    bytes [type=too_long_http_line_exception]``. The pair is what works."""
    published = f"{mountpoint}/{PUBLISHED_JAEGER_CONFIG}"
    shadow = f"{mountpoint}/{SHADOW_JAEGER_CONFIG}"
    override = clock_override_yaml()
    marker = "shogym's orca_bench port"
    compose = f"{mountpoint}/docker-compose.snapshot.yml"
    script = (
        "set -e; "
        f"sed 's/max_span_age: {PUBLISHED_LOOKBACK}/max_span_age: {SNAPSHOT_LOOKBACK}/' "
        f"\"{published}\" > \"{shadow}\"; "
        f"grep -q 'max_span_age: {SNAPSHOT_LOOKBACK}' \"{shadow}\"; "
        f"grep -q \"{marker}\" \"{compose}\" || cat >> \"{compose}\" <<'SHOGYM_EOF'\n"
        f"{override}SHOGYM_EOF\n"
    )
    _run(
        [
            "run", "--rm", "--platform", IMAGE_PLATFORM,
            "--mount", f"type=volume,source={snapshot_volume_name()},target={mountpoint}",
            "alpine:latest", "sh", "-c", script,
        ],
        timeout=timeout,
        capture=False,
    )


def prepare_stack(*, digest: str = SNAPSHOT_IMAGE_DIGEST) -> str:
    """Everything that has to exist before any trial: the image, the staged cache, the clock.

    Idempotent and safe to call per episode; on a warm host it is a few cheap probes.

    The clock override is installed here rather than per episode because it belongs to the staged
    cache, which every trial on this host shares. :meth:`ComposeBackend.telemetry_reach` is kept as
    the per-run witness: the override is verified by what the stack actually answers, not by having
    been written."""
    mountpoint = ensure_snapshot_cache(digest=digest)
    install_clock_override(mountpoint)
    return mountpoint


# ----- one episode -----


@dataclass(frozen=True)
class StackHandle:
    """The containers and volumes one episode owns."""

    container: str
    context_volume: str
    project_hint: str


class ComposeBackend:
    """One task's stack, live: the agent's container plus the compose project it starts.

    Implements :class:`shogym.envs.orca_bench.backend.OrcaBackend`. The tool surface is the
    container's shell, because that is upstream's own agent interface: the instruction tells the
    agent to query Grafana over HTTP and to write ``/app/report.md``, and both are shell work.
    """

    def __init__(
        self,
        task_dir: Path,
        *,
        judge: ResolvedJudge,
        snapshot: str,
        start_timeout: float = DEFAULT_START_TIMEOUT_SECONDS,
        exec_timeout: float = DEFAULT_EXEC_TIMEOUT_SECONDS,
        keep_stack: bool = False,
    ) -> None:
        self._task_dir = task_dir
        self._judge = judge
        self._snapshot = snapshot
        self._start_timeout = start_timeout
        self._exec_timeout = exec_timeout
        self._keep_stack = keep_stack
        self._handle: Optional[StackHandle] = None
        self._captured: Optional[CapturedReport] = None

    # ----- lifecycle -----

    def start(self) -> StackHandle:
        """Bring the stack up and wait for the entrypoint's readiness marker."""
        mountpoint = prepare_stack()
        trial = uuid.uuid4().hex[:12]
        context_volume = f"{CONTEXT_VOLUME_PREFIX}-{trial}"
        _run(["volume", "create", context_volume], timeout=120)
        context_path = volume_mountpoint(context_volume)
        container = f"shogym-orca-{trial}"

        args = [
            "run", "--detach", "--name", container,
            "--platform", IMAGE_PLATFORM,
            # Privileged with the daemon socket: the entrypoint starts the 28 services as
            # siblings on this same daemon. This is upstream's design, not a choice here.
            "--privileged",
            "-v", "/var/run/docker.sock:/var/run/docker.sock",
            "--mount", f"type=volume,source={snapshot_volume_name()},target={mountpoint}",
            "--mount", f"type=volume,source={context_volume},target={context_path}",
            "-e", f"{SNAPSHOT_CACHE_ENV}={mountpoint}",
            "-e", f"CONTEXT_DIR={context_path}",
            "-e", f"SNAPSHOT_NAME={self._snapshot}",
            "-e", "HOST_AGENT_LOGS_PATH=/logs/agent",
        ]
        # By name only. The values ride in the docker CLI's environment (see `_run`) so the
        # judge's API key is never in this process's argv.
        for key in self._judge.environment:
            args += ["-e", key]
        args += [SNAPSHOT_IMAGE_PINNED, "sleep", "infinity"]

        _run(args, timeout=300, env=self._judge.environment)
        self._handle = StackHandle(
            container=container, context_volume=context_volume, project_hint=trial
        )
        try:
            self._await_ready()
        except BaseException:
            self.teardown()
            raise
        return self._handle

    def telemetry_reach(self) -> Dict[str, Any]:
        """Whether the stack's own service list answers, and what it said.

        This is the check the clock decision exists for, so the run records it rather than
        assuming it. The recorded telemetry is from 2026-04-19..23 against a 2160h lookback, and
        an agent's natural first move is exactly this query: when it comes back empty the agent
        sees a system with no services, and the run is not comparable to the paper's numbers even
        though nothing failed."""
        probe = self.exec(
            "curl -s --max-time 30 $JAEGER_URL/api/services", timeout=180
        )
        body = (probe["stdout"] or "").strip()
        services: List[str] = []
        error = ""
        try:
            parsed = json.loads(body) if body else {}
        except ValueError:
            parsed = {}
            error = body[:200]
        if isinstance(parsed, dict):
            data = parsed.get("data")
            if isinstance(data, list):
                services = [str(entry) for entry in data]
            errors = parsed.get("errors")
            if isinstance(errors, list) and errors:
                error = str(errors[0])[:300]
        return {"services": services, "count": len(services), "error": error}

    def _await_ready(self) -> None:
        """Wait for ``/tmp/env-ready``, the entrypoint's own statement that the stack is up.

        The probe runs without a shell and tolerates its own timeout. While the stack is starting
        the container is busy enough that even ``docker exec`` can take a minute to be scheduled,
        and treating that as a failed start throws away a run that was going fine."""
        handle = self._require_handle()
        deadline = time.monotonic() + self._start_timeout
        while time.monotonic() < deadline:
            probe = _run(
                ["exec", handle.container, "test", "-f", READY_MARKER],
                timeout=180,
                check=False,
            )
            if probe.returncode == 0:
                return
            if not self._container_running():
                logs = self._container_logs(tail=40)
                raise BackendUnavailableError(
                    f"the task container exited before the stack came up:\n{logs}"
                )
            time.sleep(5)
        logs = self._container_logs(tail=40)
        raise BackendUnavailableError(
            f"the stack did not report ready within {self._start_timeout:.0f}s:\n{logs}"
        )

    def _container_running(self) -> bool:
        handle = self._require_handle()
        proc = _run(
            ["inspect", "--format", "{{.State.Running}}", handle.container],
            timeout=60,
            check=False,
        )
        return proc.returncode == 0 and proc.stdout.strip() == "true"

    def _container_logs(self, *, tail: int = 40) -> str:
        handle = self._handle
        if handle is None:
            return ""
        proc = _run(["logs", "--tail", str(tail), handle.container], timeout=120, check=False)
        return ((proc.stdout or "") + (proc.stderr or "")).strip()

    def _require_handle(self) -> StackHandle:
        if self._handle is None:
            raise BackendUnavailableError("the stack was never started")
        return self._handle

    # ----- the agent's surface -----

    def _raw_exec(
        self, command: str, *, timeout: Optional[float] = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        handle = self._require_handle()
        return _run(
            ["exec", handle.container, "bash", "-lc", command],
            timeout=timeout or self._exec_timeout,
            check=check,
        )

    def exec(self, command: str, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Run one shell command in the agent's container, with the stack's URLs in scope."""
        # The entrypoint writes the service URLs to /tmp/env-ports and deliberately unsets the
        # Harbor paths that would leak the task's name; sourcing it is how upstream's own agent
        # discovers Grafana.
        wrapped = f"[ -f {PORTS_FILE} ] && . {PORTS_FILE}; {command}"
        proc = self._raw_exec(wrapped, timeout=timeout, check=False)
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
        }

    def read_file(self, path: str) -> Dict[str, Any]:
        proc = self._raw_exec(f"cat {json.dumps(path)}", check=False)
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "content": proc.stdout if proc.returncode == 0 else "",
            "stderr": proc.stderr or "",
        }

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        # The content goes in as UTF-8 explicitly rather than as text, whose encoding would be
        # whatever this process's locale says. An agent writes prose, and a report with an accent
        # in it must not depend on how the harness happened to be launched.
        handle = self._require_handle()
        quoted = json.dumps(path)
        script = f"mkdir -p \"$(dirname {quoted})\" && cat > {quoted}"
        proc = subprocess.run(
            [_DOCKER, "exec", "-i", handle.container, "bash", "-lc", script],
            input=content.encode("utf-8"),
            capture_output=True,
            timeout=self._exec_timeout,
        )
        stderr = (proc.stderr or b"").decode("utf-8", "replace")
        return {"ok": proc.returncode == 0, "exit_code": proc.returncode, "stderr": stderr}

    # ----- grading -----

    def capture_report(self) -> CapturedReport:
        """Copy the agent's report out of the container **once**, and validate those bytes.

        The contract's reason, restated because it is what this method is for: the seal stops the
        agent's tool calls, not processes it already started in the container, so validating the
        live path and letting the verifier reopen it grades whatever is there the second time.
        ``docker cp`` takes one copy into core-owned storage the agent cannot reach, and
        everything after this point reads that copy."""
        handle = self._require_handle()
        staging = Path(tempfile.mkdtemp(prefix="shogym-orca-report-"))
        local = staging / "report.md"
        proc = _run(
            ["cp", f"{handle.container}:{REPORT_PATH}", str(local)],
            timeout=300,
            check=False,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or "").strip()
            if "No such container:path" in detail or "no such file" in detail.lower():
                self._captured = CapturedReport(
                    source=REPORT_PATH, problem=f"no report was written at {REPORT_PATH}"
                )
                return self._captured
            self._captured = CapturedReport(
                source=REPORT_PATH,
                problem=f"the report at {REPORT_PATH} could not be read: {detail}",
            )
            return self._captured
        captured = capture_report(local)
        # Re-source the container path so the message names what the agent sees, not a temp file.
        self._captured = CapturedReport(
            source=REPORT_PATH,
            data=captured.data,
            problem=captured.problem.replace(str(local), REPORT_PATH),
        )
        return self._captured

    def run_verifier(self, captured: CapturedReport) -> Dict[str, Any]:
        """Run the task's own verifier over exactly the captured bytes.

        The captured report is written back into the container at a path outside the agent's
        working tree and handed to ``check_prediction.py`` with ``--predictions``, so the bytes
        graded are the bytes captured at the seal and not whatever ``/app/report.md`` holds by
        then. It travels as **bytes**, by ``docker cp``: a report is agent-authored text, and
        pushing it through a text pipe would re-encode it in the subprocess's locale encoding,
        which is not necessarily UTF-8."""
        handle = self._require_handle()
        graded_path = "/tmp/shogym-sealed-report.md"
        sealed = Path(tempfile.mkdtemp(prefix="shogym-orca-sealed-")) / "report.md"
        sealed.write_bytes(captured.data or b"")
        _run(["cp", str(sealed), f"{handle.container}:{graded_path}"], timeout=300)
        self._raw_exec("mkdir -p /logs/verifier /tests", check=False)
        tests_dir = self._task_dir / "tests"
        for name in ("check_prediction.py", "expected.json", "test.sh"):
            source = tests_dir / name
            if source.is_file():
                _run(["cp", str(source), f"{handle.container}:/tests/{name}"], timeout=300)
        rubrics = tests_dir / "rubrics"
        if rubrics.is_dir():
            _run(["cp", str(rubrics), f"{handle.container}:/tests/rubrics"], timeout=300)

        command = (
            "python3 /tests/check_prediction.py "
            f"--predictions {graded_path} "
            "--expected /tests/expected.json "
            "--rubrics-dir /tests/rubrics "
            f"--model {self._judge.model} --effort {self._judge.effort} "
            "--reward /logs/verifier/reward.json "
            "--details /logs/verifier/reward-details.json"
        )
        self._raw_exec(command, timeout=DEFAULT_VERIFIER_TIMEOUT_SECONDS, check=False)
        return {
            "reward": self._read_json("/logs/verifier/reward.json"),
            "details": self._read_json("/logs/verifier/reward-details.json"),
        }

    def _read_json(self, path: str) -> Optional[Dict[str, Any]]:
        result = self.read_file(path)
        if not result["ok"]:
            return None
        try:
            value = json.loads(result["content"])
        except ValueError:
            return None
        return value if isinstance(value, dict) else None

    # ----- teardown -----

    def teardown(self) -> None:
        """Stop the agent container and sweep the sibling project it started.

        The entrypoint traps to tear the inner services down, so stopping the container politely
        is the first move. It is not the last: the services are siblings on the host daemon, and a
        trap that was killed leaves a 28-service project and gigabytes of per-trial data behind."""
        handle = self._handle
        if handle is None or self._keep_stack:
            return
        _run(
            ["stop", "--timeout", "150", handle.container],
            timeout=DEFAULT_TEARDOWN_TIMEOUT_SECONDS,
            check=False,
        )
        _run(["rm", "--force", "--volumes", handle.container], timeout=300, check=False)
        self._sweep_orphans()
        _run(["volume", "rm", "--force", handle.context_volume], timeout=300, check=False)
        self._handle = None

    def _sweep_orphans(self) -> None:
        """Remove any inner compose project whose agent container is gone.

        Identified by compose's own project label rather than by name matching, and only for
        projects this port started (their agent container no longer exists)."""
        proc = _run(
            [
                "ps", "--all", "--quiet",
                "--filter", "label=com.docker.compose.project",
                "--filter", "status=running",
            ],
            timeout=120,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return
        containers = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        for container in containers:
            labels = _run(
                ["inspect", "--format", "{{index .Config.Labels \"com.docker.compose.project\"}}", container],
                timeout=60,
                check=False,
            )
            project = labels.stdout.strip()
            if project.startswith("otel-") and not self._agent_container_alive(project):
                _run(["rm", "--force", "--volumes", container], timeout=120, check=False)

    def _agent_container_alive(self, project: str) -> bool:
        """Whether some shogym agent container is still running for this compose project."""
        proc = _run(
            ["ps", "--quiet", "--filter", "name=shogym-orca-", "--filter", "status=running"],
            timeout=120,
            check=False,
        )
        return bool(proc.stdout.strip())


def create_backend(
    task_dir: Path,
    *,
    judge: ResolvedJudge,
    snapshot: str,
    **kwargs: Any,
) -> ComposeBackend:
    """Bring up a live backend for one task."""
    if not docker_available():
        raise BackendUnavailableError(
            "orca_bench needs a running Docker daemon to serve an episode: the benchmark's "
            "environment is a 28-service compose project the task image starts on the host "
            "daemon. Start Docker and retry."
        )
    backend = ComposeBackend(task_dir, judge=judge, snapshot=snapshot, **kwargs)
    backend.start()
    return backend


__all__ = [
    "ComposeBackend",
    "DockerError",
    "IMAGE_PLATFORM",
    "StackHandle",
    "clock_override_yaml",
    "create_backend",
    "daemon_free_bytes",
    "docker_available",
    "ensure_image",
    "ensure_snapshot_cache",
    "install_clock_override",
    "prepare_stack",
    "snapshot_volume_name",
    "volume_mountpoint",
]
