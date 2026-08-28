"""The seam between shogym and AppWorld: the pins, the corpus, and one worker per episode.

**AppWorld cannot be installed beside shogym**, and that is a fact about the two projects rather
than a packaging preference: ``appworld`` pins ``pydantic<2`` and shogym's MCP layer needs
``pydantic>=2.7``. No environment satisfies both. So this port provisions an interpreter of its
own, installs the pinned release into it, and runs every world under it, talking to it over a
loopback socket (:mod:`shogym.envs.appworld.worker`). The separation the design wanted for
isolation is the same one packaging forces, so it costs nothing extra.

Three things are provisioned, and all three are pinned here:

- **the interpreter**, a virtual environment holding ``appworld`` at :data:`UPSTREAM_VERSION`;
- **the app sources**, which the wheel ships packed and unpacks on first use;
- **the data bundle**, which is a 33 MB download the package fetches without checking anything.
  This module fetches it itself and refuses a bundle whose digest is not
  :data:`DATA_BUNDLE_SHA256`, because every task, every database and every ground truth in the
  measurement comes out of that file and "we downloaded something from S3" is not a pin.

The corpus is then *derived* rather than used in place. Each served task gets a directory whose
files are links to the task's own, except the one database log this port rewrites to carry the
seeded backlog. Deriving costs one small file per task, leaves the downloaded corpus untouched,
and makes the seeded rows part of the task's input state, which is where they have to be for the
scenario's own score to survive them.

Importing this module imports nothing from upstream. Provisioning happens when an env is built.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import select
import signal
import shutil
import subprocess
import time
import tempfile
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shogym.envs._upstream import _locked

# ----- the pins -----

#: The release this port reproduces, and the commit it was cut from.
UPSTREAM_VERSION = "0.1.3.post1"
UPSTREAM_SHA = "66ad8099e12188ece0d3fe45e661dbc01880813b"

#: The data bundle, its size and its digest. The corpus's own version file says ``0.1.0``, which
#: is also what the task specs declare and what upstream refuses to run against a mismatch of.
DATA_VERSION = "0.1.0"
DATA_BUNDLE_URL = "https://s3.us-west-2.amazonaws.com/appworld.dev/data-0.1.0.bundle"
DATA_BUNDLE_SHA256 = "fd9f9608c2ec71ed0ac25c3633a738b9129a318a129e31230425b9188e508250"
DATA_BUNDLE_BYTES = 34280074

#: The split this port serves. AppWorld's authors ask that the test splits not be used to teach or
#: tune, and this port does not report an AppWorld score: the base task's checks ride along as
#: context for the appended chore and the port publishes them beside its own number rather than
#: as a benchmark result.
SPLIT = "test_challenge"

#: Where the corpus is, if it is already somewhere. Upstream's own variable, so a machine that
#: already ran ``appworld download data`` needs nothing further.
ROOT_ENV_VAR = "APPWORLD_ROOT"

_DOWNLOAD_TIMEOUT_SECONDS = 300.0

#: How long a worker gets to stop after it is signalled, before it is killed. Short: teardown
#: runs on the shared loop and a wedged world may not hold the others.
_CLOSE_SECONDS = 10.0

#: How long the grader gets. Generous, because the base task's evaluator replays a whole task's
#: database changes, and bounded, because a grader that never finishes would hold a sealed
#: episode's terminal open for the life of the run.
_GRADE_TIMEOUT_SECONDS = 600.0

#: How long a worker gets to bind its port and say so. Generous, because a cold interpreter
#: importing upstream and its clock-patching library is not fast, and bounded, because a worker
#: that never speaks would otherwise hang the episode that started it with nothing to read.
_SPAWN_TIMEOUT_SECONDS = 180.0


class ProvisioningError(RuntimeError):
    """A step that builds the interpreter or the corpus failed.

    Its own type because the two things it can mean are far apart: a machine with no network, and
    a pin that no longer resolves. A caller that wants to tolerate the first without hiding the
    second needs to be able to tell this apart from every other failure."""



def cache_root() -> Path:
    """Where this port keeps what it provisions, as an absolute path.

    Resolved rather than taken as given. A derived corpus is a tree of symlinks whose targets are
    written verbatim, and a relative target is read relative to the link's own directory rather
    than to the directory the run was launched from, so a relative cache root produces a tree of
    links that resolve to nothing."""
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base).expanduser().resolve() if base else Path.home() / ".cache" / "shogym"
    return root / "appworld"


# ----- provisioning -----

WORKER = Path(__file__).with_name("worker.py")


def runtime() -> Path:
    """The interpreter every world runs under, building it if it is not there yet.

    A virtual environment rather than the running one, because ``appworld`` pins ``pydantic<2``
    and shogym needs ``pydantic>=2.7``: installing it beside shogym is not a thing pip will do,
    and a port that pretended otherwise would fail at resolve time on every machine. Built with
    ``uv`` where it is on the path and with the standard library's own tools otherwise, so the
    port needs no tool the user did not already have."""
    home = cache_root() / f"runtime-{UPSTREAM_VERSION}"
    python = home / ("Scripts" if os.name == "nt" else "bin") / "python"
    if python.exists():
        return python
    home.parent.mkdir(parents=True, exist_ok=True)
    with _locked(home.parent):
        if python.exists():
            return python
        _build_runtime(home)
    return python


def _build_runtime(home: Path) -> None:
    """Create the environment and install the pinned release into it."""
    import shutil
    import venv

    staging = home.with_name(home.name + ".building")
    shutil.rmtree(staging, ignore_errors=True)
    requirement = f"appworld=={UPSTREAM_VERSION}"
    uv = shutil.which("uv")
    if uv:
        _run([uv, "venv", "--python", _python_series(), str(staging)])
        _run([uv, "pip", "install", "--python", str(_interpreter(staging)), requirement])
    else:
        venv.EnvBuilder(with_pip=True).create(str(staging))
        _run([str(_interpreter(staging)), "-m", "pip", "install", "--quiet", requirement])
    os.replace(staging, home)


def _interpreter(home: Path) -> Path:
    return home / ("Scripts" if os.name == "nt" else "bin") / "python"


def _python_series() -> str:
    """The Python this port asks for, matching the one shogym is running under."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _run(command: list) -> None:
    """Run a provisioning command, and say what it was if it fails."""
    finished = subprocess.run(command, capture_output=True, text=True)
    if finished.returncode != 0:
        raise ProvisioningError(
            f"provisioning the appworld runtime failed: {' '.join(command)}\n"
            f"{finished.stderr.strip()[-2000:]}"
        )


def ensure_apps() -> None:
    """Unpack the app sources the wheel ships packed, if they are not unpacked already.

    A fresh install has ``appworld.apps`` with the shared library in it and none of the nine
    apps, so the first import of a model module fails with a plain ``ModuleNotFoundError`` that
    says nothing about the missing step. Idempotent, and silent when there is nothing to do."""
    python = runtime()
    installed = _installed_package(python)
    if (installed / "apps" / "todoist" / "models.py").exists():
        return
    with _locked(installed):
        if (installed / "apps" / "todoist" / "models.py").exists():
            return
        _run([str(python), str(WORKER), "install"])


def _installed_package(python: Path) -> Path:
    """Where ``appworld`` sits inside the provisioned interpreter."""
    finished = subprocess.run(
        [str(python), "-c", "import appworld, os; print(os.path.dirname(appworld.__file__))"],
        capture_output=True,
        text=True,
    )
    if finished.returncode != 0:
        raise ProvisioningError(
            f"the provisioned appworld runtime cannot import appworld:\n"
            f"{finished.stderr.strip()[-2000:]}"
        )
    return Path(finished.stdout.strip())


def ensure_corpus() -> Path:
    """The directory whose ``data/`` is the corpus, provisioning it if it is not already there.

    An existing corpus named by :data:`ROOT_ENV_VAR` is used as it stands, which is the offline
    path and the one a machine that already downloaded the data takes. Otherwise the pinned bundle
    is fetched into this port's cache and checked against its digest before anything is unpacked."""
    named = os.environ.get(ROOT_ENV_VAR)
    if named:
        root = Path(named).expanduser().resolve()
        if (root / "data" / "tasks").is_dir():
            return root
    root = cache_root() / f"corpus-{DATA_VERSION}"
    if (root / "data" / "tasks").is_dir():
        return root
    root.parent.mkdir(parents=True, exist_ok=True)
    # Both of these take the same lock, on this port's own cache directory, and an ``flock`` taken
    # twice through two opens in one process blocks on itself. So the interpreter and the app
    # sources are provisioned *before* the corpus lock is taken, never inside it. A genuinely cold
    # machine takes exactly this path.
    runtime()
    ensure_apps()
    with _locked(root.parent):
        if (root / "data" / "tasks").is_dir():
            return root
        _fetch_corpus(root)
    return root


def _fetch_corpus(root: Path) -> None:
    """Download, verify and unpack the pinned bundle into ``root``.

    Unpacked by the provisioned interpreter, because the bundle is an encrypted archive whose
    format is upstream's business. What this function owns is the check in front of it. The
    interpreter is provisioned by the caller, outside this function's lock (see
    :func:`ensure_corpus`)."""
    import shutil

    python = runtime()
    staging = root.with_name(root.name + ".building")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    bundle = staging / Path(DATA_BUNDLE_URL).name
    with urllib.request.urlopen(DATA_BUNDLE_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        bundle.write_bytes(response.read())
    _verify(bundle)
    _run([str(python), str(WORKER), "unpack", "--bundle", str(bundle), "--into", str(staging)])
    bundle.unlink()
    if not (staging / "data" / "tasks").is_dir():
        raise ProvisioningError(
            f"the bundle at {DATA_BUNDLE_URL} unpacked without a data/tasks tree"
        )
    os.replace(staging, root)


def _verify(bundle: Path) -> None:
    """Refuse a bundle that is not the pinned one, and say which half of the pin it failed."""
    size = bundle.stat().st_size
    digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
    if size != DATA_BUNDLE_BYTES or digest != DATA_BUNDLE_SHA256:
        bundle.unlink(missing_ok=True)
        raise ProvisioningError(
            f"the data bundle at {DATA_BUNDLE_URL} is not the pinned one: got {size} bytes "
            f"with sha256 {digest}, expected {DATA_BUNDLE_BYTES} bytes with sha256 "
            f"{DATA_BUNDLE_SHA256}"
        )


# ----- the served roster -----

MANIFEST = Path(__file__).with_name("task_manifest.txt")


@lru_cache(maxsize=1)
def task_ids() -> Tuple[str, ...]:
    """The tasks this port serves, in split order.

    Not every task of the split. A task is served only if a backlog could be drawn for it that
    separates all 64 conventions, and only if its own evaluation names none of the models the
    appended chore adds, since ignoring a model a scenario asserts on would turn a passing check
    into a failing one. Both are decided once, before any episode, and committed beside the code:
    a roster settled per run is a roster that can move with whatever else changed."""
    return tuple(line.strip() for line in MANIFEST.read_text().splitlines() if line.strip())


@lru_cache(maxsize=None)
def task_specs(root: Path, task_id: str) -> Dict[str, Any]:
    """One task's shipped specification: its instruction, its supervisor and its datetime."""
    return json.loads((root / "data" / "tasks" / task_id / "specs.json").read_text())


# ----- the derived corpus -----


def derived_root() -> Path:
    """Where the seeded copy of the corpus lives, named for what generated it.

    The name carries a digest of the backlog generator's own constants, so changing a cut value,
    an option set or the number of requests derives a new corpus instead of serving a stale one
    that was built under the old ones."""
    return cache_root() / f"seeded-{DATA_VERSION}-{_generator_digest()}"


def private_home() -> Path:
    """The directory holding everything an agent's world must not be handed.

    Not a sibling of the served root and not under this port's ordinary cache, because the served
    root's own path is in the worker's environment and a neighbour of it is a guess away."""
    base = cache_root().parent
    return base.parent / f"{base.name}-private" / "appworld"


def graded_root() -> Path:
    """Where the grader's view of the corpus lives: a private directory with an unguessable name.

    **This raises the cost of finding it and does not close the route.** The worker runs as the
    same user as the process that built this, so no directory mode keeps it out: 0700 stops other
    users and stops nothing else. What closes it is a namespace in which the directory is not
    mounted at all, which is a container and is not built here (see the port's README). What this
    does is stop the tree being derivable from what the worker is given, which the previous
    layout, a fixed name beside the served root, was."""
    home = private_home()
    return home / f"graded-{DATA_VERSION}-{_generator_digest()}-{_private_tag()}"


def _read_tag(keyfile: Path) -> Optional[str]:
    """The published tag, or ``None`` if there is not a complete one there yet."""
    try:
        tag = keyfile.read_text().strip()
    except FileNotFoundError:
        return None
    return tag if len(tag) == 16 else None


@lru_cache(maxsize=4)
def corpus_digest(root: Path) -> str:
    """What the corpus at ``root`` actually holds, as sixteen hex characters.

    The pinned bundle's digest says what *should* be there and cannot say what is: ``APPWORLD_ROOT``
    takes any directory with a ``data/tasks`` in it, so a repointed or edited corpus would
    otherwise be served under a name that claims to be the pinned one, and would reuse a derived
    tree built from something else.

    **Every scoring-relevant file is read, not sized.** This once hashed `specs.json` in full and
    took path and size for everything else, which left the ground truth and the evaluation
    material identifiable only by length: a same-length edit to an answer key passed unnoticed
    under a digest whose whole job is to say what the corpus holds. A size is not a summary of
    contents, and a fingerprint that says it covers the scoring inputs has to have read them.

    The task tree is the scoring input, so all of it is read. Reading it costs a second or so on
    every fresh process, which is the price of the digest meaning what it says."""
    digest = hashlib.sha256()
    data = root / "data"
    version = data / "version.txt"
    digest.update(version.read_bytes() if version.exists() else b"")
    for path in sorted((data / "tasks").rglob("*")):
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(data)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def episode_outputs(session_id: str) -> Path:
    """One episode's own output tree, named for the episode and not under the private home.

    AppWorld joins its experiment name onto its own output root, so an absolute name replaces that
    root outright. That is what keeps an episode's end state, its logs and anything the evaluator
    leaves behind out of the corpus tree every other episode is served from. A shared output tree
    is a place the other arm of a pair can read an earlier grade.

    **Deliberately not a child of** :func:`private_home`. The runtime keeps this value on the live
    AppWorld object, where agent-authored code can read it, so whatever directory it names is
    named *to the agent*. Under the private home it named the grader's own parent, and the
    unguessable tag that protects the grader stops protecting it once something hands the name
    over. This tree is separate: reading it tells you where episode outputs go and nothing about
    where the answers are.

    The name is still absolute, so on the host this remains a path the worker's user could reach
    if it went looking. Making it unreachable is the container's job (see the PR that runs the
    worker in its own mount namespace); what belongs here is not handing over the address."""
    home = episodes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / f"episode-{session_id}"


def episodes_home() -> Path:
    """Where per-episode output trees live: this port's ordinary cache, not the private one."""
    return cache_root() / f"episodes-{DATA_VERSION}"


def episode_view(session_id: str) -> Path:
    """The root of one episode's own served corpus (see :func:`~world.derive_view`)."""
    return cache_root() / f"views-{DATA_VERSION}" / f"view-{session_id}"


@lru_cache(maxsize=1)
def _private_tag() -> str:
    """Sixteen hex characters, drawn once per installation and kept beside the private tree.

    Persisted rather than redrawn, because a name that changed per process would derive the whole
    corpus again on every run."""
    home = private_home()
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    keyfile = home / ".tag"
    existing = _read_tag(keyfile)
    if existing is not None:
        return existing
    # Written whole, then published by rename. An exclusive create publishes the *name* before the
    # bytes, so a concurrent reader could see a real file holding nothing, and a crash in the gap
    # left that empty file behind for good. `os.replace` is atomic within a directory, so the name
    # appears only once it already holds a complete tag.
    tag = secrets.token_hex(8)
    staged = home / f".tag.{os.getpid()}.{tag}"
    handle = os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(handle, tag.encode())
        os.fsync(handle)
    finally:
        os.close(handle)
    try:
        # A loser takes the winner's tag rather than keeping its own: two processes that each kept
        # theirs would name two private trees, and a private tree is a copy of the corpus, so the
        # cost of getting this wrong is 134 MB and a rebuild for every loser.
        os.link(staged, keyfile)
    except FileExistsError:
        pass
    finally:
        os.unlink(staged)
    published = _read_tag(keyfile)
    if published is None:
        raise RuntimeError(f"the private tag at {keyfile} is not a tag; remove it to rebuild")
    return published


@lru_cache(maxsize=1)
def _generator_digest() -> str:
    """Eight hex characters over everything that decides what a backlog looks like."""
    from shogym.envs.appworld import ledger

    material = repr(
        (
            ledger.ROLES,
            ledger.BASIS_OPTIONS,
            ledger.BOUNDARY_OPTIONS,
            ledger.MISSING_OPTIONS,
            ledger.CUTS,
            ledger.BANDS,
            ledger.SECTIONS,
            ledger.SPAN,
            ledger.DATED,
            ledger.UNDATED,
            ledger.ATTEMPTS,
        )
    )
    return hashlib.sha256(material.encode()).hexdigest()[:8]


# ----- one episode's worker -----


class WorkerError(RuntimeError):
    """The worker refused a command, or the world raised inside one."""


#: What a worker's environment is allowed to carry. Agent-authored code runs as that process, so
#: everything the serving process holds is otherwise one ``os.environ`` away from it: provider
#: keys, the run's own paths, whatever the operator exported. The list is what a Python process
#: needs to start and no more, and ``HOME`` and the caches are pointed at a scratch directory of
#: the episode's own.
_ENV_ALLOW_LIST: Tuple[str, ...] = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "TMPDIR")


def _worker_environment(scratch: Path) -> Dict[str, str]:
    """A scrubbed environment for one worker."""
    scrubbed = {
        name: os.environ[name] for name in _ENV_ALLOW_LIST if os.environ.get(name) is not None
    }
    scrubbed["HOME"] = str(scratch)
    scrubbed["APPWORLD_CACHE"] = str(scratch / "appworld-cache")
    return scrubbed


@dataclass
class Worker:
    """A handle on one episode's world, running in a process of its own.

    The port is the process's, the token is this object's, and neither is ever put anywhere an
    agent can read: not on the worker's command line, not in the instructions the env publishes,
    not in a tool's schema, and not in a tool's result.

    What this is and is not: it keeps the world's own grading routes and the serving process's
    environment out of the agent's reach, and it is not a sandbox. The code an agent writes runs
    as the worker, with the worker's filesystem. See the port's README."""

    root: Path
    process: subprocess.Popen
    port: int
    token: str
    scratch: Path

    @classmethod
    def spawn(cls, root: Path) -> "Worker":
        """Start a worker on ``root`` and wait for it to say which port it bound.

        The token and the root go over stdin, which is read once and closed, rather than on the
        command line, which any code running in that process can read back off ``sys.argv`` for
        the life of it."""
        token = secrets.token_urlsafe(32)
        scratch = Path(tempfile.mkdtemp(prefix="shogym-appworld-"))
        process = subprocess.Popen(
            [str(runtime()), str(WORKER), "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            cwd=str(scratch),
            env=_worker_environment(scratch),
            # Its own process group, so stopping the episode stops everything it started. Agent
            # code runs in this process and is free to spawn; signalling the direct child alone
            # would leave those descendants running against the world after it was scored.
            start_new_session=True,
        )
        assert process.stdin is not None
        process.stdin.write(json.dumps({"root": str(root), "token": token}) + "\n")
        process.stdin.flush()
        process.stdin.close()
        assert process.stdout is not None
        line = _first_line(process, _SPAWN_TIMEOUT_SECONDS)
        if not line:
            process.kill()
            process.wait(timeout=10)
            shutil.rmtree(scratch, ignore_errors=True)
            raise WorkerError(
                "the appworld worker never bound a port "
                f"(status {process.returncode}, waited {_SPAWN_TIMEOUT_SECONDS:.0f}s)"
            )
        return cls(
            root=root,
            process=process,
            port=int(json.loads(line)["port"]),
            token=token,
            scratch=scratch,
        )

    def call(self, command: str, **body: Any) -> Any:
        """Send one command and return what the world answered."""
        payload = json.dumps(body).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/{command}",
            data=payload,
            headers={"Content-Type": "application/json", "X-Shogym-Worker-Token": self.token},
        )
        try:
            with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
                return json.loads(response.read())["output"]
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")
            raise WorkerError(f"appworld worker refused {command!r}: {detail}") from exc

    def close(self) -> None:
        """Stop the worker, promptly and with a bound.

        Signalled and reaped rather than asked over the socket. Teardown runs on the serving
        process's shared loop, so a close that waited on an HTTP round trip into a wedged world
        would hold every other episode with it; and there is nothing to ask for, because the end
        state was flushed when the episode was read. A process that ignores the signal is killed.

        **The group, and not only its leader.** This used to signal only while the leader was
        still alive and then wait only for that leader. A leader that had already exited left its
        children unsignalled, and a child that outlived the leader was never noticed: the world
        had been scored and something was still running in it. The group is signalled whatever the
        leader is doing, and what is waited for is the group emptying."""
        pgid = _group_of(self.process)
        if self.process.poll() is None:
            _stop(self.process, signal.SIGTERM)
            try:
                self.process.wait(timeout=_CLOSE_SECONDS)
            except subprocess.TimeoutExpired:
                _stop(self.process, signal.SIGKILL)
                try:
                    self.process.wait(timeout=_CLOSE_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
        elif pgid is not None:
            # The leader is already gone, so nothing above signalled anything. Its children are
            # still in its group.
            _signal_group(pgid, signal.SIGTERM)
        if pgid is not None:
            _reap_group(pgid)
        for stream in (self.process.stdout, self.process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
        shutil.rmtree(self.scratch, ignore_errors=True)


def grade(
    *,
    root: Path,
    task_id: str,
    experiment: str,
    ignore: Sequence[str],
    timeout: float = _GRADE_TIMEOUT_SECONDS,
) -> Any:
    """The base task's own checks, from a process that has never run a line the agent wrote.

    A second, short-lived worker rather than the one that served the episode. It is the only place
    ground truth is loaded, it starts after the world is sealed, and it reads the end state off
    disk, so the answers are never objects in the process the agent's code ran as."""
    opening = json.dumps(
        {"root": str(root), "task_id": task_id, "experiment": experiment, "ignore": list(ignore)}
    )
    scratch = Path(tempfile.mkdtemp(prefix="shogym-appworld-grade-"))
    process = subprocess.Popen(
        [str(runtime()), str(WORKER), "grade"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(scratch),
        env=_worker_environment(scratch),
    )
    try:
        # Bounded, killed and reaped. An evaluator that hangs would otherwise hold a sealed
        # episode's terminal open forever: `to_thread` does not make a child process cancellable,
        # so a deadline on the coroutine stops the waiting and leaves the child running.
        out, err = process.communicate(input=opening + "\n", timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise WorkerError(
            f"grading {task_id} did not finish within {timeout:.0f}s; the grader was killed"
        ) from None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    if process.returncode != 0:
        raise WorkerError(
            f"grading {task_id} failed (status {process.returncode}): {err.strip()[-2000:]}"
        )
    return json.loads(out.strip().splitlines()[-1])["output"]


def _group_of(process: subprocess.Popen) -> Optional[int]:
    """The worker's process group, while it is still possible to ask."""
    try:
        return os.getpgid(process.pid)
    except (OSError, AttributeError):
        return None


def _signal_group(pgid: int, how: int) -> None:
    try:
        os.killpg(pgid, how)
    except (OSError, AttributeError):
        pass


def _group_members(pgid: int) -> Sequence[int]:
    """Every live process still in ``pgid``, this process excluded.

    Asked of `ps` rather than of `/proc`, which macOS does not have. An answer this cannot get is
    reported as an empty group: the caller uses it to decide whether to escalate, and a reaper
    that raised on an unreadable process table would turn a diagnostic into a failed teardown."""
    try:
        listing = subprocess.run(
            ["ps", "-o", "pid=,pgid=", "-A"],
            capture_output=True,
            text=True,
            timeout=_CLOSE_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    live: List[int] = []
    for line in listing.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pid, group = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if group == pgid and pid != os.getpid():
            live.append(pid)
    return live


def _reap_group(pgid: int) -> None:
    """Wait for the worker's group to empty, escalating once if it does not.

    Waiting for the leader says nothing about what the leader started. Agent code runs in that
    process and may have left something behind, and something still running after the world has
    been scored is either changing what was scored or holding a port the next episode wants."""
    deadline = time.monotonic() + _CLOSE_SECONDS
    escalated = False
    while _group_members(pgid):
        if time.monotonic() >= deadline:
            if escalated:
                return
            _signal_group(pgid, signal.SIGKILL)
            escalated = True
            deadline = time.monotonic() + _CLOSE_SECONDS
        time.sleep(0.02)


def _stop(process: subprocess.Popen, how: int) -> None:
    """Signal the worker's whole process group, or the worker alone if it has no group of its own.

    The group is the point: agent code runs in that process and may have started others, and a
    signal to the direct child alone leaves them running against a world that has been scored."""
    try:
        os.killpg(os.getpgid(process.pid), how)
    except (OSError, AttributeError):
        try:
            process.send_signal(how)
        except OSError:
            pass


def _first_line(process: subprocess.Popen, timeout: float) -> str:
    """The worker's first line of output, or the empty string if it does not arrive in time.

    ``readline`` on a pipe cannot be given a deadline, so the descriptor is waited on instead: a
    worker that dies without printing closes the pipe and is readable immediately, and one that
    hangs on an import is caught by the deadline rather than hanging its caller with it."""
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        return ""
    return process.stdout.readline()


__all__ = [
    "DATA_BUNDLE_SHA256",
    "DATA_BUNDLE_URL",
    "DATA_VERSION",
    "MANIFEST",
    "ProvisioningError",
    "ROOT_ENV_VAR",
    "SPLIT",
    "UPSTREAM_SHA",
    "UPSTREAM_VERSION",
    "Worker",
    "WorkerError",
    "WORKER",
    "cache_root",
    "corpus_digest",
    "derived_root",
    "episode_outputs",
    "episode_view",
    "episodes_home",
    "grade",
    "graded_root",
    "private_home",
    "ensure_apps",
    "ensure_corpus",
    "runtime",
    "task_ids",
    "task_specs",
]
