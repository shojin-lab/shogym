"""The single seam between the ``appworld`` port and Docker: the image, and how a worker is run.

The port's whole isolation claim used to be a list of things not put where agent-authored code
could reach them, and the port's own README said in those words that on one uid no filesystem
arrangement is a boundary. This module is where that stops being true. Every process that runs a
line an agent wrote is started here, in a container with `--network none`, no inherited
environment, and a mount set that is one task's served tree and this episode's own output
directory. What the previous layout hid, this one does not mount: the run tree, the grader's tree,
the repository, the corpus, the user's home, and every other episode's world are not unreadable,
they are absent.

Three roles run through here, and only the first ever runs agent-authored code:

``serve`` is one episode's world, held open for the length of the episode, talking to the parent
over its own stdin and stdout (see :mod:`shogym.envs.appworld.worker`). ``seed`` is one
short-lived container that writes one task's seeded database log into a staging directory; it
exists because that log has to be written through upstream's own model layer and upstream is not
installed on the host at all. ``grade`` and ``unpack`` are short-lived containers for the base
task's evaluator and for opening the pinned data bundle. Only the first ever runs agent-authored
code.

**The transport is stdio, not a port.** A container-loopback listener is not forwardable, a
published port is not loopback-only, and ``--network none`` and ``-p`` are mutually exclusive. So
the worker speaks length-prefixed JSON frames over the pipe pair the parent created for it, which
no other process on the machine can open, and the container is run with no network stack at all.

**Nothing about the image is inherited from the machine.** The base is digest-pinned, the release
is version-pinned, and the tag is a digest over the Dockerfile and the worker together, so an
edit to either builds a new image rather than reusing one built under the old text.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from shogym.envs._upstream import _locked

_DOCKER = "docker"

#: Where the served corpus is mounted, and where an episode's world therefore lives. A fixed name
#: rather than the host's own path: the host path names a directory in the run's cache, and a name
#: an agent can read is a name an agent can reason from.
CORPUS_MOUNT = "/corpus"

#: Where the grader's view of a task is mounted. Only the grading container ever sees it.
GRADED_MOUNT = "/graded"

#: Where one episode's own output tree is mounted, and the only writable mount a world is given.
#: AppWorld joins its experiment name onto its own output root, so an absolute name replaces that
#: root outright: the world is told its experiment *is* this directory, and writes its end state,
#: its logs and anything the evaluator leaves behind straight into it rather than into the corpus.
OUTPUTS_MOUNT = "/outputs"

#: The container's writable scratch, a tmpfs. ``HOME`` and the working directory both point here,
#: so an episode's own temporary files exist for the length of the container and nowhere else.
SCRATCH_MOUNT = "/scratch"

#: How long the image build may take. Generous: it compiles ``psutil`` from source on arm64.
_BUILD_TIMEOUT_SECONDS = 1800.0

#: How long a ``docker`` control command (``inspect``, ``rm``) may take.
_CONTROL_TIMEOUT_SECONDS = 60.0

DOCKERFILE = Path(__file__).with_name("worker.Dockerfile")
WORKER = Path(__file__).with_name("worker.py")


class DockerError(RuntimeError):
    """Docker is not usable, or a ``docker`` invocation failed.

    Its own type because the env raises it at construction rather than at the first ``execute``:
    a machine with no Docker daemon cannot serve this env at all, and finding that out an hour
    into a run is finding it out too late."""


@dataclass(frozen=True)
class Mount:
    """One bind mount: a host path, where it appears in the container, and whether it is writable.

    Writable is the exception and is spelled out at every call site. The served corpus is
    read-only; the one directory an episode may write is its own output tree."""

    source: Path
    target: str
    writable: bool = False

    def as_argument(self) -> str:
        return f"{self.source}:{self.target}:{'rw' if self.writable else 'ro'}"


def _run(args: Sequence[str], *, timeout: float, check: bool = True) -> subprocess.CompletedProcess:
    try:
        finished = subprocess.run(
            [_DOCKER, *args], text=True, capture_output=True, timeout=timeout
        )
    except FileNotFoundError as exc:
        raise DockerError("the docker CLI is not on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise DockerError(f"`docker {args[0]}` did not finish within {timeout:.0f}s") from exc
    if check and finished.returncode != 0:
        detail = (finished.stderr.strip() or finished.stdout.strip())[-2000:]
        raise DockerError(f"`docker {' '.join(args[:2])}` failed ({finished.returncode}): {detail}")
    return finished


def docker_available() -> bool:
    """Whether there is a Docker daemon this process can reach."""
    try:
        finished = subprocess.run(
            [_DOCKER, "info"], text=True, capture_output=True, timeout=_CONTROL_TIMEOUT_SECONDS
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return finished.returncode == 0


def require_docker() -> None:
    """Refuse, with the reason, on a machine that cannot run the worker at all.

    Called when an env is *constructed*, not when an episode first runs code. The env has no
    non-container path: a host worker would run agent-authored code as the user running the run,
    which is the arrangement this module exists to end."""
    if not docker_available():
        raise DockerError(
            "the appworld env runs each episode's world in a container and no Docker daemon is "
            "reachable (`docker info` failed). There is no host fallback: the code an agent "
            "writes runs as the worker, so a worker on the host runs it as the user running the "
            "run. Start Docker, or do not serve this env on this machine."
        )


@lru_cache(maxsize=1)
def _image_tag() -> str:
    """A digest over the Dockerfile and the worker, so an edit to either builds a new image."""
    material = DOCKERFILE.read_bytes() + b"\x00" + WORKER.read_bytes()
    return hashlib.sha256(material).hexdigest()[:12]


def image_name() -> str:
    """The image an episode's world runs in, named for what it was built from."""
    return f"shogym-appworld-worker:{_image_tag()}"


def image_exists(name: Optional[str] = None) -> bool:
    return (
        _run(["image", "inspect", name or image_name()], timeout=_CONTROL_TIMEOUT_SECONDS, check=False)
        .returncode
        == 0
    )


def ensure_image(*, cache: Path, timeout: float = _BUILD_TIMEOUT_SECONDS) -> str:
    """Build the worker image if this machine does not have it already, and return its name.

    Under the cache lock, because a paired run starts two processes at once and both would
    otherwise build the same image; Docker would cope, and the second build would be minutes of
    work for a layer cache hit. The build context is the port's own directory, which is why the
    Dockerfile can ``COPY worker.py`` and nothing else."""
    name = image_name()
    if image_exists(name):
        return name
    cache.mkdir(parents=True, exist_ok=True)
    with _locked(cache):
        if image_exists(name):
            return name
        args = ["build", "-f", str(DOCKERFILE), "-t", name]
        platform = os.environ.get("SHOGYM_APPWORLD_PLATFORM")
        if platform:
            # Only when asked for. The image builds natively on both architectures this port has
            # been run on, and emulating one on the other would be paid on every execute.
            args += ["--platform", platform]
        args += [str(DOCKERFILE.parent)]
        _run(args, timeout=timeout)
    return name


def _identity() -> str:
    """The uid and gid the container runs as: the host user's own.

    Not root, and not a uid baked into the image. A bind mount carries the host's ownership, so a
    container writing an episode's end state has to be a uid that may write there, and the only
    such uid the port can know is the one it is running as. On a host running as root this is root
    and the port says so rather than pretending; every other case is an unprivileged user."""
    return f"{os.getuid()}:{os.getgid()}"


def run(
    *,
    role: str,
    arguments: Sequence[str] = (),
    mounts: Sequence[Mount] = (),
    environment: Optional[Dict[str, str]] = None,
    name: Optional[str] = None,
    scratch_megabytes: int = 512,
) -> Tuple[subprocess.Popen, str]:
    """Start one worker container, wired for frames on its stdin and stdout.

    Returns the running ``docker run`` client and the container's name. The name is the handle
    teardown needs: killing the client does not stop the container, so :func:`remove` is what ends
    one and it is called on every path, including the failing ones.

    The flags, and what each is for:

    ``--network none``
        No interfaces at all. The transport is the pipe pair this call creates, so nothing is
        given up by having no network, and an episode cannot reach the machine's own services,
        the run's own provenance server, or the internet.
    ``--user``
        The host user's uid, so the output tree it writes is owned by the run. See :func:`_identity`.
    ``--read-only`` with a ``tmpfs`` scratch
        Every writable byte is either the tmpfs, which dies with the container, or the one mount
        marked writable. An episode cannot leave anything behind in the image.
    ``--cap-drop ALL``, ``--security-opt no-new-privileges``
        A world driving nine simulated apps needs no capability, and a setuid binary in the base
        image is not a route this has to leave open.
    ``--pids-limit``
        Agent-authored code runs in this container. A fork bomb is a plausible accident and would
        otherwise be the host's problem.
    """
    container = name or f"shogym-appworld-{role}-{uuid.uuid4().hex[:12]}"
    args: List[str] = [
        "run",
        "--rm",
        "-i",
        "--name",
        container,
        "--network",
        "none",
        "--user",
        _identity(),
        "--read-only",
        "--tmpfs",
        f"{SCRATCH_MOUNT}:rw,size={int(scratch_megabytes)}m,mode=0777",
        "--tmpfs",
        "/tmp:rw,size=64m,mode=1777",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--pids-limit",
        "512",
        "-w",
        SCRATCH_MOUNT,
    ]
    platform = os.environ.get("SHOGYM_APPWORLD_PLATFORM")
    if platform:
        args += ["--platform", platform]
    # Nothing is inherited. `docker run` passes only what the image declares and what is named
    # here, so the serving process's provider keys and run paths are not in this container's
    # environment because they were never offered to it.
    for key, value in sorted((environment or {}).items()):
        args += ["-e", f"{key}={value}"]
    for mount in mounts:
        args += ["-v", mount.as_argument()]
    args += [image_name(), role, *arguments]
    process = subprocess.Popen(
        [_DOCKER, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    return process, container


def remove(container: str) -> None:
    """Remove a container, whatever state it is in, and never raise.

    Teardown's job, and it runs on the crash paths too. ``--rm`` covers the ordinary exit and the
    case where the parent dies (the worker reads end-of-file on its stdin and stops), but a
    container whose world is wedged in a native call ignores that, and one nobody removed is a
    world holding a mount open for the rest of the machine's day."""
    try:
        _run(["rm", "-f", container], timeout=_CONTROL_TIMEOUT_SECONDS, check=False)
    except DockerError:
        pass


def running(container: str) -> bool:
    """Whether a container by that name is still up. For teardown's own tests."""
    finished = _run(
        ["ps", "--quiet", "--filter", f"name=^{container}$"],
        timeout=_CONTROL_TIMEOUT_SECONDS,
        check=False,
    )
    return bool(finished.stdout.strip())


__all__ = [
    "CORPUS_MOUNT",
    "DOCKERFILE",
    "GRADED_MOUNT",
    "OUTPUTS_MOUNT",
    "SCRATCH_MOUNT",
    "DockerError",
    "Mount",
    "docker_available",
    "ensure_image",
    "image_exists",
    "image_name",
    "remove",
    "require_docker",
    "run",
    "running",
]
