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

import calendar
import hashlib
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from shogym.envs._upstream import _locked

_DOCKER = "docker"

#: Where the served corpus is mounted, and where an episode's world therefore lives. A fixed name
#: rather than the host's own path: the host path names a directory in the run's cache, and a name
#: an agent can read is a name an agent can reason from.
CORPUS_MOUNT = "/corpus"

#: Where the grader's view of a task is mounted. Only the grading container ever sees it.
GRADED_MOUNT = "/graded"

#: Every proxy variable Docker's client injects into a container it creates, in both cases the
#: daemon accepts. They are not passed by this port and never appear in its argv, which is how they
#: were missed: the client adds them from whatever proxy profile is configured, and Docker's own
#: documentation notes that a proxy URL can carry credentials or an internal host name. A world
#: with no network has no use for one, and an agent reading its environment would read them.
_PROXY_VARIABLES: Tuple[str, ...] = tuple(
    name
    for base in ("http_proxy", "https_proxy", "ftp_proxy", "no_proxy", "all_proxy")
    for name in (base, base.upper())
)

#: What every worker gets for ``/etc/resolv.conf``. Docker writes one from the host's or the
#: daemon's resolver configuration even under ``--network none``, and the file it wrote named a
#: nameserver and said it was based on the host's. There is no network to resolve anything on, so
#: what the file says is host metadata and nothing else; this says nothing.
_NEUTRAL_RESOLV = "# no resolver: this container has no network\n"

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

#: How long one ``docker`` control command (``inspect``, ``stop``, ``rm``) may take.
#:
#: **Bounded well below the serve layer's, because these compose.** The core abandons a session's
#: release after 60 seconds and marks the episode closed; a single control call that could itself
#: consume 60 seconds would put cleanup after that point, still holding mounts the next episode
#: may want. Teardown makes at most a stop, a removal and two inspects, so the whole of it fits
#: inside a fraction of the outer bound.
_CONTROL_TIMEOUT_SECONDS = 10.0

#: How long a container gets to stop politely before it is killed. Zero: there is nothing inside
#: worth a graceful shutdown (the state upstream persists is written at the end of every block,
#: not at exit), and every second here is a second of the teardown budget.
_STOP_GRACE_SECONDS = 0

#: Every container this port starts carries these. The first says whose they are, the second says
#: which process started them, and together they are what lets a later run tell an abandoned
#: container from a live one. A random name cannot: it says nothing to anybody who did not start
#: it, and a parent that died holding one leaves nothing behind that names it.
LABEL_OWNER = "shogym.appworld"
LABEL_PARENT = "shogym.appworld.parent"
LABEL_BOOT = "shogym.appworld.boot"

#: When the starting process itself began, so a pid reused within one boot is not read as the
#: parent still being alive. A pid alone answers "is something running under that number", which
#: is a different question from "is the process that started this container still running".
LABEL_BIRTH = "shogym.appworld.birth"

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


@lru_cache(maxsize=4)
def image_identity(name: str) -> str:
    """What was actually built, as the daemon has it: the image id and the platform.

    The tag is a digest over the Dockerfile and the worker, so it moves when this repository's
    inputs move. It does not move when the base image's own contents move under a digest pin that
    was re-pushed, when a build resolves a different transitive version, or when the same tag was
    built on another platform. Two runs whose rows are one measurement have to agree on the
    interpreter their worlds ran under, and this is the value that says which one that was."""
    finished = _run(
        ["image", "inspect", "--format", "{{.Id}} {{.Os}}/{{.Architecture}}", name],
        timeout=_CONTROL_TIMEOUT_SECONDS,
        check=False,
    )
    if finished.returncode != 0:
        raise DockerError(f"the image {name} is not built, so it has no identity to record")
    return finished.stdout.strip()


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


@lru_cache(maxsize=1)
def limits() -> Tuple[str, str]:
    """The cpu and memory a serving container gets, read once for this process.

    **Read once and then fixed, because it is part of what a run measured.** These decide latency,
    what a call timeout means, and whether a world is killed for allocating; two arms that ran
    under different ones are two arms that were given different machines, which is exactly the
    kind of opportunity the deadline and the capacity are already identity-bearing for. So the
    value is captured here, every launch in this process uses the captured one, and
    :func:`~shogym.envs.appworld.env_v1.run_fingerprint` records it: an environment changed under
    a running process cannot move it, and a run relaunched under a changed one does not pass for
    the earlier measurement."""
    return (
        os.environ.get("SHOGYM_APPWORLD_CPUS", "2"),
        os.environ.get("SHOGYM_APPWORLD_MEMORY", "2g"),
    )


@lru_cache(maxsize=1)
def neutral_resolver() -> Path:
    """A fixed ``resolv.conf`` to mount over the one Docker generates.

    Written once per process into this port's cache, because a bind needs a file on the host and
    a temporary one would have to outlive the container that mounts it."""
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base).expanduser().resolve() if base else Path.home() / ".cache" / "shogym"
    home = root / "appworld"
    home.mkdir(parents=True, exist_ok=True)
    path = home / "neutral-resolv.conf"
    if not path.exists() or path.read_text() != _NEUTRAL_RESOLV:
        path.write_text(_NEUTRAL_RESOLV)
    return path


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
        # Whose it is, who started it, and on which boot. See :func:`reap`: a random name tells a
        # later run nothing, and a parent that died holding one leaves nothing that names it.
        "--label",
        f"{LABEL_OWNER}=1",
        "--label",
        f"{LABEL_PARENT}={os.getpid()}",
        "--label",
        f"{LABEL_BOOT}={_boot_id()}",
        "--label",
        f"{LABEL_BIRTH}={process_birth(os.getpid())}",
        # A constant, because the default is the container's own short id and Docker puts it in
        # the environment. It is not a secret and it is not this episode's to hand out.
        "--hostname",
        "worker",
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
        # A quota, not just a count. Agent-authored code runs here, and a block that spins or
        # allocates without bound is one episode taking the machine away from its own control
        # group: the other arm of a pair is a sibling container on the same host, and an arm that
        # ran slower because its twin was busy is a difference the treatment did not make.
        "--cpus",
        limits()[0],
        "--memory",
        limits()[1],
        "-w",
        SCRATCH_MOUNT,
    ]
    platform = os.environ.get("SHOGYM_APPWORLD_PLATFORM")
    if platform:
        args += ["--platform", platform]
    # Emptied rather than omitted. The client puts these into every container it creates from the
    # active proxy profile, so leaving them out of this list leaves them in the container: an
    # explicit assignment is what overrides an injected one, and an empty value is what a world
    # with no network should see.
    for name in _PROXY_VARIABLES:
        args += ["-e", f"{name}="]
    # Nothing else is inherited. `docker run` passes only what the image declares and what is named
    # here, so the serving process's provider keys and run paths are not in this container's
    # environment because they were never offered to it.
    for key, value in sorted((environment or {}).items()):
        args += ["-e", f"{key}={value}"]
    # Over the one the daemon generates from the host's resolver configuration. A world with no
    # network has nothing to resolve, so what that file holds is host metadata and nothing else.
    args += ["-v", f"{neutral_resolver()}:/etc/resolv.conf:ro"]
    for mount in mounts:
        args += ["-v", mount.as_argument()]
    # By resolved id, not by tag. The tag is what the fingerprint was resolved from, and a tag is
    # mutable: a rebuild between resolving the identity and starting the world would run bytes the
    # run recorded nothing about. The id names one image and cannot move.
    args += [image_identity(image_name()).split()[0], role, *arguments]
    process = subprocess.Popen(
        [_DOCKER, *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    return process, container


#: What the daemon says when the object is genuinely not there. Anything else on a nonzero exit
#: is the daemon, the context or the CLI failing, which is a different fact entirely.
_NOT_FOUND = ("no such object", "no such container")


def absent(container: str) -> bool:
    """Whether the daemon has no container by that name, running or stopped.

    ``docker inspect`` rather than ``docker ps``: a container that exited but was not removed is
    still a container, still holds its mounts, and is still something a name can collide with.

    **A nonzero exit is not the same fact as "not found", and this used to treat it as one.**
    Every daemon failure, every unreachable context, every permission error and every timeout
    exits nonzero, and reading those as absence is reading "I could not look" as "it is gone".
    That is the one direction this must never fail in, because absence is what allows grading. So
    the daemon's own not-found wording is what returns ``True``, presence returns ``False``, and
    anything else raises: unknown is not a boolean."""
    finished = _run(["inspect", container], timeout=_CONTROL_TIMEOUT_SECONDS, check=False)
    if finished.returncode == 0:
        return False
    said = (finished.stderr + finished.stdout).strip().lower()
    if any(phrase in said for phrase in _NOT_FOUND):
        return True
    raise DockerError(
        f"cannot tell whether the container {container} is still there: `docker inspect` exited "
        f"{finished.returncode} saying {said[:200]!r}"
    )


def remove(container: str, *, confirm: bool = False) -> bool:
    """Remove a container and, when asked, refuse to pretend it worked.

    ``--rm`` covers the ordinary exit and the case where the parent dies (the worker reads
    end-of-file on its stdin and stops), but a container whose world is wedged in a native call
    ignores that, and one nobody removed is a world holding its mounts open.

    Returns whether the container is known to be gone. Teardown reads that; finalization does
    not have to, because for it the alternative to certainty is an exception.

    **Two callers want two different failure modes, so they get two.** Teardown runs on the crash
    paths and must not raise: for it, a removal that failed is a container the next attempt or the
    reaper will get. Finalization is the other: it removes the world's container precisely so that
    nothing can write to the tree it is about to grade, and a removal it did not confirm is an
    invariant it cannot claim. ``confirm=True`` asks the daemon whether the container is really
    gone and raises when it is not."""
    # Stopped first and then removed. `rm -f` alone is a signal followed by the daemon's own
    # timeout, and an explicit stop with no grace period is the shortest path to "no process in
    # there is running", which is the fact grading depends on.
    stopped = _run(
        ["stop", "--time", str(_STOP_GRACE_SECONDS), container],
        timeout=_CONTROL_TIMEOUT_SECONDS,
        check=False,
    )
    removed = _run(["rm", "-f", container], timeout=_CONTROL_TIMEOUT_SECONDS, check=False)
    if not confirm:
        # **A nonzero status is not a removal, and teardown used to read it as one.** It may not
        # raise, so what it does instead is hand the name to the sweep: a container the daemon
        # refused to stop or remove outlives this process otherwise, because the ordinary sweep
        # skips containers whose parent is still alive.
        if stopped.returncode != 0 or removed.returncode != 0:
            disowned(container)
            return False
        return True
    if not absent(container):
        # One retry, because a container in an uninterruptible call can outlive the first.
        _run(["rm", "-f", container], timeout=_CONTROL_TIMEOUT_SECONDS, check=False)
    # `absent` raises when it cannot tell, and that raise is the point: an unconfirmed removal and
    # an unanswerable daemon are the same fact to anything downstream, which is that nothing may
    # treat what that container could write as final.
    if not absent(container):
        raise DockerError(
            f"the container {container} is still there after two removals; the daemon has not "
            "confirmed that it stopped, so nothing may treat what it could write as final"
        )
    return True


def reap(*, alive: Optional[Callable[..., bool]] = None) -> List[str]:
    """Remove this port's containers whose parent process is gone, and say which.

    The case it exists for is the one teardown cannot reach: a parent that dies while a world is
    wedged inside a command. The worker only notices its parent through end-of-file on the next
    read, and a container whose process never returns to that read never exits, so ``--rm`` never
    fires. What is left is a container nobody is going to remove and nothing names.

    So every container carries the pid that started it and this machine's boot time, and this runs
    at construction: a labelled container whose parent is not running is one nobody is coming back
    for. The boot time is in the label because pids are reused across a reboot, and removing a
    live episode's world because an unrelated process now holds its parent's number would be a
    worse failure than the one this fixes."""
    listed = _run(
        ["ps", "--all", "--quiet", "--filter", f"label={LABEL_OWNER}=1",
         "--filter", f"label={LABEL_BOOT}={_boot_id()}"],
        timeout=_CONTROL_TIMEOUT_SECONDS,
        check=False,
    )
    running = alive if alive is not None else _process_is_alive
    # The ones nobody could remove, whoever is alive. See `disowned`.
    removed: List[str] = _sweep_disowned()
    for identifier in listed.stdout.split():
        # Read as structure rather than as words. Two labels printed side by side and split on
        # whitespace is a value rebuilt rather than read, and a value rebuilt is a value whose
        # spacing can change: that is how a live parent came to look like a different process.
        answered = _run(
            ["inspect", "--format", "{{json .Config.Labels}}", identifier],
            timeout=_CONTROL_TIMEOUT_SECONDS,
            check=False,
        )
        try:
            labels = json.loads(answered.stdout or "{}") or {}
        except ValueError:
            labels = {}
        parent = str(labels.get(LABEL_PARENT, ""))
        birth = str(labels.get(LABEL_BIRTH, ""))
        # An unreadable or unparseable label is left alone. This removes containers, so the
        # ambiguous case has to be the one where nothing happens.
        if not parent.isdigit() or running(int(parent), birth):
            continue
        _run(["rm", "-f", identifier], timeout=_CONTROL_TIMEOUT_SECONDS, check=False)
        removed.append(identifier)
    return removed


def process_birth(pid: int) -> str:
    """When ``pid`` started, as the process table reports it, or the empty string if unknown.

    A pid is reused, and within one boot it is reused quickly. ``kill(pid, 0)`` answers "is
    something running under that number", and the question the reaper is asking is "is the
    process that started this container still running". The start time is what separates them: a
    number that came back with a different birth is a different process wearing the same badge.

    **A number, not a rendering.** ``ps`` blank-pads a single-digit day, so a label written on the
    third of the month came back with its spacing changed by anything that rebuilt it from words,
    and a live parent then read as a different process: the sweep would remove that parent's
    running episode, which is the one failure this must never have. What is returned is epoch
    seconds, which has no spacing to lose and compares exactly.

    **Read in a fixed locale and a fixed zone.** `ps` renders this start time for a human, so the
    same live process under `TZ=UTC` and under `America/Los_Angeles` prints two different strings,
    and a stamp written by one harness process and compared by another under a different zone
    would say the parent had been replaced. The environment is pinned rather than the output
    parsed, because the format is the platform's and the pinning is what makes it comparable.

    One second of precision is what `ps` offers, so a pid reused inside the same second is still
    indistinguishable. That is the residual, and it is narrower than the pid alone was.

    Unknown is the empty string rather than an error. It is compared for equality, and two empty
    strings comparing equal keeps the reaper on the safe side of its own rule: it removes only
    what it can positively tell is abandoned."""
    try:
        finished = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=_CONTROL_TIMEOUT_SECONDS,
            env={**os.environ, "TZ": "UTC", "LC_ALL": "C", "LANG": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if finished.returncode != 0:
        return ""
    rendered = " ".join(finished.stdout.split())
    if not rendered:
        return ""
    try:
        # Normalised first, so the padded day the platform writes and the single space a reader
        # might rebuild parse to the same instant. The zone is pinned above, so this is UTC.
        return str(calendar.timegm(time.strptime(rendered, "%a %b %d %H:%M:%S %Y")))
    except ValueError:
        # A format this does not know is not a birth this can compare. Unknown is the empty
        # string, which keeps the sweep on the safe side of its own rule.
        return ""


def _ledger() -> Path:
    """Where containers nobody could remove are written down, one name per line.

    Session-scoped would be no use: the case is a removal that failed while this process is still
    running and still holding the session it belongs to. So it is a file, under this port's own
    cache, that any later sweep reads."""
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base).expanduser().resolve() if base else Path.home() / ".cache" / "shogym"
    home = root / "appworld"
    home.mkdir(parents=True, exist_ok=True)
    return home / "disowned.txt"


def disowned(container: str) -> None:
    """Record a container this process could not remove, so a sweep will try again.

    **Appended and never rewritten.** The sweep used to read the whole file and publish a new one,
    so a name appended between its read and its write was dropped, and the case where that matters
    is exactly this one: a container whose parent is alive, which the ordinary sweep skips. Every
    line is an event now, ``+`` for a name to try and ``-`` for one confirmed gone, and what is
    live is what the events say rather than what any writer last decided.

    Failures here are swallowed: this is called from teardown, which may not raise, and a name
    that could not be written is a container the ordinary reaper still finds once its parent
    exits."""
    _append(f"+{container}")


def _append(line: str) -> None:
    try:
        with open(_ledger(), "a") as handle:
            handle.write(line + "\n")
    except OSError:
        pass


def outstanding() -> List[str]:
    """The names appended and not yet tombstoned, in the order they were first written."""
    try:
        lines = _ledger().read_text().splitlines()
    except OSError:
        return []
    live: List[str] = []
    gone = set()
    for line in lines:
        line = line.strip()
        if line.startswith("-"):
            gone.add(line[1:])
        elif line.startswith("+") and line[1:] not in live:
            live.append(line[1:])
    return [name for name in live if name not in gone]


def _sweep_disowned() -> List[str]:
    """Remove every container the ledger still names, and tombstone the ones that went.

    Nothing here consults the parent, because the ledger's whole point is a container whose parent
    is alive and out of ways to remove it. A name that cannot be removed is left with no tombstone,
    so the next sweep tries it again."""
    removed = []
    for name in outstanding():
        try:
            remove(name, confirm=True)
        except DockerError:
            continue
        _append(f"-{name}")
        removed.append(name)
    return removed


def _process_is_alive(pid: int, birth: str = "") -> bool:
    """Whether ``pid`` is the same live process that was born at ``birth``."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process, which is somebody else's business and certainly alive.
        return True
    if not birth:
        return True
    now = process_birth(pid)
    # A recorded birth that no longer matches is a recycled number, and the parent that owned this
    # container is gone. An unreadable birth now says nothing, so it says nothing.
    return not now or now == birth


@lru_cache(maxsize=1)
def _boot_id() -> str:
    """Something that changes when the machine restarts, so a reused pid is not a live parent.

    **The number, not the rendering.** `sysctl -n kern.boottime` prints a struct and then a human
    date, and the date is rendered in the caller's zone: hashing the whole line gave two different
    boot identities for one boot under two `TZ` values, which would hide an orphan from a sweep run
    in another zone. The seconds field inside the struct is the kernel's own value and does not
    move, so that is what is taken. Linux's boot id is already a value rather than a rendering."""
    try:
        stamp = subprocess.run(
            ["sysctl", "-n", "kern.boottime"],
            capture_output=True,
            text=True,
            timeout=_CONTROL_TIMEOUT_SECONDS,
            env={**os.environ, "TZ": "UTC", "LC_ALL": "C", "LANG": "C"},
        )
        if stamp.returncode == 0 and stamp.stdout.strip():
            seconds = re.search(r"sec\s*=\s*(\d+)", stamp.stdout)
            if seconds:
                return seconds.group(1)
            return hashlib.sha256(stamp.stdout.strip().encode()).hexdigest()[:12]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    try:
        return hashlib.sha256(Path("/proc/sys/kernel/random/boot_id").read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


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
    "LABEL_BIRTH",
    "LABEL_BOOT",
    "LABEL_OWNER",
    "LABEL_PARENT",
    "OUTPUTS_MOUNT",
    "_PROXY_VARIABLES",
    "SCRATCH_MOUNT",
    "DockerError",
    "Mount",
    "absent",
    "docker_available",
    "neutral_resolver",
    "ensure_image",
    "image_identity",
    "limits",
    "image_exists",
    "image_name",
    "process_birth",
    "reap",
    "disowned",
    "outstanding",
    "remove",
    "require_docker",
    "run",
    "running",
]
