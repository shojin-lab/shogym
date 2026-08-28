"""The single seam between the ``frontier_bench`` env and Docker.

Frontier-Bench (via the Harbor framework) draws the boundary shogym wants: a task is a Docker
*environment* an agent operates through a shell, and a *verifier* that runs the task's
``tests/`` against the container's declared ``artifacts`` and reads a 0/1 off
``/logs/verifier/reward.txt`` — never off the transcript. This module reimplements that
protocol over the local Docker CLI, mirroring Harbor's ``BaseEnvironment.exec`` /
``upload_file`` / ``download_file`` and ``Verifier.verify`` for the CPU-only,
single-container, ``environment_mode="separate"`` case (issue #44):

  - :class:`Container` — a running task container: ``exec`` a shell command, read/write files,
    ``docker cp`` files in/out. This is the served agent's world.
  - :func:`build_image` — build an image from a task Dockerfile + build context.
  - :func:`run_separate_verifier` — the SEPARATE-mode dance: build the ``tests/`` image, start
    a fresh verifier container, copy the agent's declared ``artifacts`` out of the agent
    container and into the verifier at the *same* paths (Harbor's anti-reward-hacking design),
    run ``tests/test.sh`` (writes ``/logs/verifier/reward.txt``), and parse the recorded 0/1.

Every ``docker`` subprocess call funnels through here, so the rest of the env is Docker-free
and the offline test suite never shells out. ``import``-ing this module does **not** require
Docker; only calling into it does.
"""

from __future__ import annotations

import functools
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence

# Harbor's environment path conventions (linux). The verifier writes its 0/1 under
# /logs/verifier/reward.txt; the tests image owns /tests/ (separate mode skips the upload).
VERIFIER_DIR = "/logs/verifier"
REWARD_TXT = f"{VERIFIER_DIR}/reward.txt"
TESTS_DIR = "/tests"

DEFAULT_EXEC_TIMEOUT = 600.0
_DOCKER = "docker"


class DockerError(RuntimeError):
    """A ``docker`` CLI invocation failed (non-zero exit or spawn error)."""


@dataclass
class ExecResult:
    """Outcome of one command run inside a container."""

    exit_code: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


def _run_docker(
    args: Sequence[str], *, timeout: Optional[float] = None, check: bool = True
) -> subprocess.CompletedProcess:
    try:
        proc = subprocess.run(
            [_DOCKER, *args],
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:  # docker not installed
        raise DockerError("docker CLI not found on PATH") from exc
    if check and proc.returncode != 0:
        raise DockerError(
            f"`docker {' '.join(args[:3])} …` failed (exit {proc.returncode}): "
            f"{proc.stderr.strip() or proc.stdout.strip()}"
        )
    return proc


@functools.lru_cache(maxsize=1)
def _daemon_cpus() -> Optional[int]:
    """The Docker *daemon's* CPU count (``docker info --format '{{.NCPU}}'``), cached for the run.

    We clamp ``--cpus`` against this, not the client's ``os.cpu_count()``: ``os.cpu_count()``
    reports the topology of the Python client process, but ``docker run`` rejects a ``--cpus``
    larger than what the *daemon* enforces. On Docker Desktop / rootless / a remote daemon the
    client can see 8+ CPUs while the daemon is capped at 2, so a client-side clamp still emits
    a request the daemon refuses. The daemon's NCPU doesn't change within a run, so cache it.

    Returns ``None`` (query failed or unparseable) so the caller can fall back rather than crash;
    a broken ``docker info`` must never block a container start on an otherwise healthy daemon.
    """
    try:
        proc = _run_docker(["info", "--format", "{{.NCPU}}"], timeout=30, check=False)
    except (DockerError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        ncpu = int(proc.stdout.strip())
    except (TypeError, ValueError):
        return None
    return ncpu if ncpu > 0 else None


def docker_available() -> bool:
    """True iff a working Docker daemon is reachable (``docker info`` succeeds)."""
    try:
        proc = subprocess.run(
            [_DOCKER, "info"], text=True, capture_output=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def build_image(
    *,
    context_dir: Path,
    dockerfile: Path,
    tag: str,
    timeout: float,
    platform: str = "linux/amd64",
) -> None:
    """Build ``tag`` from ``dockerfile`` with ``context_dir`` as the build context.

    Pins ``--platform linux/amd64`` so every host builds the architecture the tasks were
    authored and validated on. The vendored bases are multi-arch image indexes, so an arm64 host
    emulates rather than silently building a different architecture.
    """
    _run_docker(
        [
            "build",
            "--platform",
            platform,
            "-f",
            str(dockerfile),
            "-t",
            tag,
            str(context_dir),
        ],
        timeout=timeout,
    )


def image_exists(tag: str) -> bool:
    return _run_docker(["image", "inspect", tag], check=False).returncode == 0


def remove_image(tag: str) -> None:
    _run_docker(["image", "rm", "-f", tag], check=False)


class Container:
    """A running task container — the served agent's world (mirrors Harbor ``BaseEnvironment``).

    Started detached with an idle entrypoint so the harness can ``exec`` into it repeatedly,
    exactly like Harbor keeps the environment alive while the agent operates it. ``cpus`` /
    ``memory_mb`` / ``gpus`` map to the task.toml resource caps. Network is left at Docker's
    default (Frontier-Bench intends open egress); callers may pass ``network="none"`` for the
    verifier, which needs none.
    """

    def __init__(
        self,
        *,
        image: str,
        name: Optional[str] = None,
        workdir: str = "/app",
        cpus: Optional[float] = None,
        memory_mb: Optional[int] = None,
        network: Optional[str] = None,
        platform: str = "linux/amd64",
    ) -> None:
        self.image = image
        self.name = name or f"shogym-frontier-{uuid.uuid4().hex[:12]}"
        self.workdir = workdir
        self._cpus = cpus
        self._memory_mb = memory_mb
        self._network = network
        self._platform = platform
        self._started = False

    def start(self) -> None:
        args: List[str] = [
            "run",
            "-d",
            "--platform",
            self._platform,
            "--name",
            self.name,
            "-w",
            self.workdir,
        ]
        if self._cpus is not None:
            # Clamp to the Docker *daemon's* CPU count: `docker run` rejects a `--cpus` request
            # larger than what the daemon enforces (e.g. a 2-CPU CI runner or a 2-CPU Docker
            # Desktop VM can't grant the task's 4). We query the daemon's NCPU rather than the
            # client's `os.cpu_count()` because the client can report more CPUs than the daemon
            # allows (Docker Desktop / rootless / a remote daemon). If that query fails we fall
            # back to the client count — a broken `docker info` must not block starts on a
            # healthy daemon. A roomier daemon still honors the task's full request, so this
            # preserves benchmark behavior wherever the resources exist and only bounds it to
            # the daemon's physical limit.
            cpu_limit = _daemon_cpus() or os.cpu_count() or 1
            args += ["--cpus", str(min(self._cpus, cpu_limit))]
        if self._memory_mb is not None:
            args += ["--memory", f"{self._memory_mb}m"]
        if self._network is not None:
            args += ["--network", self._network]
        # Idle entrypoint: keep the container alive for repeated exec, regardless of the
        # image's own CMD. `tail -f /dev/null` is present in every slim/bookworm base here.
        args += ["--entrypoint", "tail", self.image, "-f", "/dev/null"]
        _run_docker(args)
        self._started = True

    def exec(
        self,
        command: str,
        *,
        workdir: Optional[str] = None,
        user: Optional[str] = None,
        timeout: float = DEFAULT_EXEC_TIMEOUT,
    ) -> ExecResult:
        """Run ``command`` in a shell inside the container; capture exit/stdout/stderr."""
        args = ["exec"]
        if workdir is not None:
            args += ["-w", workdir]
        if user is not None:
            args += ["-u", user]
        args += [self.name, "bash", "-lc", command]
        try:
            proc = _run_docker(args, timeout=timeout, check=False)
        except DockerError as exc:
            return ExecResult(exit_code=1, stdout="", stderr=str(exc))
        return ExecResult(
            exit_code=proc.returncode, stdout=proc.stdout, stderr=proc.stderr
        )

    def read_file(self, path: str, *, max_bytes: int = 1_000_000) -> ExecResult:
        """Return a file's contents on stdout (``cat``), capped defensively via ``head -c``."""
        return self.exec(
            f"head -c {int(max_bytes)} -- {_q(path)}", timeout=DEFAULT_EXEC_TIMEOUT
        )

    def write_file(self, path: str, content: str) -> ExecResult:
        """Write ``content`` to ``path`` from the host (mkdir -p parent, then ``docker cp``)."""
        parent = PurePosixPath(path).parent.as_posix()
        if parent and parent != path:
            mk = self.exec(f"mkdir -p -- {_q(parent)}", user="root")
            if not mk.ok:
                return mk
        with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as tf:
            tf.write(content)
            host_tmp = tf.name
        try:
            self.copy_in(Path(host_tmp), path)
        finally:
            Path(host_tmp).unlink(missing_ok=True)
        return ExecResult(exit_code=0, stdout="", stderr="")

    def copy_in(self, host_path: Path, container_path: str) -> None:
        """``docker cp`` a host file into the container (Harbor ``upload_file``)."""
        _run_docker(["cp", str(host_path), f"{self.name}:{container_path}"])

    def copy_out(self, container_path: str, host_path: Path) -> bool:
        """``docker cp`` a container file to the host (Harbor ``download_file``). Best-effort:
        returns False if the source path does not exist in the container."""
        host_path.parent.mkdir(parents=True, exist_ok=True)
        proc = _run_docker(
            ["cp", f"{self.name}:{container_path}", str(host_path)], check=False
        )
        return proc.returncode == 0

    def path_is_dir(self, path: str) -> bool:
        return self.exec(f"test -d {_q(path)}", user="root").ok

    def stop(self) -> None:
        if self._started:
            _run_docker(["rm", "-f", self.name], check=False)
            self._started = False


def _q(arg: str) -> str:
    """POSIX single-quote a shell argument."""
    return "'" + arg.replace("'", "'\"'\"'") + "'"


@dataclass
class VerifierOutcome:
    """The recorded verifier verdict + provenance for the ``done`` tool / trace."""

    reward: float  # the 0/1 read off /logs/verifier/reward.txt (Harbor protocol)
    reward_found: bool  # whether reward.txt existed and parsed
    artifacts_collected: Dict[str, bool]  # declared artifact -> copied out of agent env
    test_exit_code: int
    test_stdout_tail: str


def run_separate_verifier(
    *,
    agent: Container,
    tests_context_dir: Path,
    tests_dockerfile: Path,
    verifier_image_tag: str,
    artifacts: Sequence[str],
    build_timeout: float,
    verifier_timeout: float,
    keep_image: bool = False,
) -> VerifierOutcome:
    """Run the task's verifier in SEPARATE mode and return the recorded 0/1 verdict.

    Matches Harbor's ``environment_mode="separate"`` flow:

      1. Build the ``tests/`` image (it ``COPY . /tests/``, so it owns ``/tests/test.sh`` and
         golden files — the tests upload is skipped, exactly like Harbor's
         ``skip_tests_upload``).
      2. Start a fresh verifier container (its own filesystem — the agent can't have tampered
         with the verifier).
      3. Copy each declared ``artifact`` out of the *agent* container and into the verifier at
         the same path (mkdir -p parent first). This is Harbor's artifact-download-then-upload
         dance; only the declared artifacts cross the boundary.
      4. Run ``bash /tests/test.sh`` (chmod +x first), which runs pytest and writes
         ``/logs/verifier/reward.txt``.
      5. Read ``reward.txt`` back and parse the float (Harbor's ``_parse_reward_text``).

    The reward is computed off the container end-state, never the trajectory.
    """
    build_image(
        context_dir=tests_context_dir,
        dockerfile=tests_dockerfile,
        tag=verifier_image_tag,
        timeout=build_timeout,
    )
    verifier = Container(
        image=verifier_image_tag,
        workdir="/",
        network="none",  # the verifier needs no network for a pytest check
    )
    verifier.start()
    collected: Dict[str, bool] = {}
    try:
        # Ensure the reward directory exists and is root-owned BEFORE running test.sh. Upstream
        # scripts rely on the pytest-ctrf plugin to create /logs/verifier when it writes
        # ctrf.json (which precedes the `echo … > reward.txt` redirect), so a passing run already
        # lands the reward file — but pre-creating it makes the reward write robust regardless of
        # that ordering, and (created as root, mode 755) keeps the reward channel writable only by
        # root, so unprivileged agent code a verifier may run cannot forge it.
        verifier.exec(f"mkdir -p {_q(VERIFIER_DIR)}", user="root")
        # --- artifact transfer: agent end-state -> verifier, at the same paths ---
        staging = Path(tempfile.mkdtemp(prefix="shogym-frontier-artifacts-"))
        try:
            for src in artifacts:
                ok = _transfer_artifact(agent, verifier, src, staging)
                collected[src] = ok
        finally:
            shutil.rmtree(staging, ignore_errors=True)

        # --- run the verifier ---
        verifier.exec(f"chmod +x {_q(TESTS_DIR + '/test.sh')}", user="root")
        run = verifier.exec(
            f"bash {_q(TESTS_DIR + '/test.sh')}",
            workdir="/",
            user="root",
            timeout=verifier_timeout,
        )

        # --- read the recorded 0/1 verdict ---
        reward_read = verifier.read_file(REWARD_TXT)
        reward, found = _parse_reward(reward_read.stdout if reward_read.ok else "")
        tail = (run.stdout or run.stderr or "")[-2000:]
        return VerifierOutcome(
            reward=reward,
            reward_found=found,
            artifacts_collected=collected,
            test_exit_code=run.exit_code,
            test_stdout_tail=tail,
        )
    finally:
        verifier.stop()
        if not keep_image:
            remove_image(verifier_image_tag)


def _transfer_artifact(
    agent: Container, verifier: Container, src: str, staging: Path
) -> bool:
    """Copy one declared artifact out of the agent env and into the verifier at the same path.

    Best-effort (Harbor collects artifacts best-effort): a missing artifact in the agent
    container simply doesn't land in the verifier, so the pytest check fails naturally →
    reward 0. Handles both files and directories.
    """
    is_dir = agent.path_is_dir(src)
    host_target = staging / src.lstrip("/")
    if not agent.copy_out(src, host_target):
        return False
    # Ensure the parent dir exists in the verifier, then copy in at the same absolute path.
    parent = PurePosixPath(src).parent.as_posix()
    if parent and parent != src:
        verifier.exec(f"mkdir -p -- {_q(parent)}", user="root")
    if is_dir:
        # docker cp of a dir places its *contents* under the target when the target exists;
        # copy the staged dir's contents into the parent to reproduce the source layout.
        verifier.copy_in(host_target, parent or "/")
    else:
        verifier.copy_in(host_target, src)
    return True


def _parse_reward(text: str) -> tuple[float, bool]:
    """Parse the float in ``reward.txt`` (Harbor ``_parse_reward_text``). Empty/garbage → 0."""
    stripped = text.strip()
    if not stripped:
        return 0.0, False
    try:
        return float(stripped), True
    except (TypeError, ValueError):
        return 0.0, False


__all__ = [
    "DockerError",
    "ExecResult",
    "Container",
    "VerifierOutcome",
    "docker_available",
    "build_image",
    "image_exists",
    "remove_image",
    "run_separate_verifier",
    "VERIFIER_DIR",
    "REWARD_TXT",
    "TESTS_DIR",
]
