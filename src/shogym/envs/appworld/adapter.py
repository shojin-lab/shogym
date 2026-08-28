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
import platform
import secrets
import select
import signal
import shutil
import subprocess
import time
import tempfile
import threading
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

#: How long SIGTERM is given before SIGKILL follows it, whatever happened.
#:
#: Short on purpose. SIGTERM is catchable, so every second of it is a second in which the process
#: that ran agent-authored code can still write into the tree that is about to be graded. What the
#: grace buys is an ordinary exit for a process that wants one; what it must not buy is time.
_TERM_GRACE_SECONDS = 0.5

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


def task_specs(root: Path, task_id: str) -> Dict[str, Any]:
    """One task's shipped specification: its instruction, its supervisor and its datetime.

    **Read every time, not memoized.** This was cached on ``(root, task_id)``, which is a key that
    says where a spec was rather than what it said: a corpus edited in place produced a new
    ``corpus_digest`` and a new cache name, and the same process went on serving the instruction,
    the supervisor and the datetime it had read the first time. The identity moved and the task
    did not. It is a few kilobytes of JSON, and the cache was also unbounded across a roster of
    318."""
    return json.loads((root / "data" / "tasks" / task_id / "specs.json").read_text())


def runtime_digest() -> str:
    """What the worker's interpreter actually is, as sixteen hex characters.

    The runtime cache is named for the *direct* AppWorld version alone, while it is built by
    resolving that release's ranged dependencies against whatever the host's Python and the index
    offer on the day. Two machines, or one machine a month apart, therefore run a world under
    different transitive versions under a name that says they are the same. This reads what was
    realized rather than what was asked for: the interpreter the environment records for itself,
    and the distribution set on its path, by name and version.

    Read off the filesystem rather than by running the interpreter, so it costs a directory
    listing. Not memoized, for the reason :func:`corpus_digest` is not."""
    home = runtime().parent.parent
    material = hashlib.sha256()
    material.update(f"{platform.system()}|{platform.machine()}".encode())
    config = home / "pyvenv.cfg"
    material.update(config.read_bytes() if config.exists() else b"")
    for packages in sorted(home.glob("lib/python*/site-packages")):
        for entry in sorted(packages.iterdir()):
            # `name-version.dist-info` is the realized distribution set, which is the thing the
            # cache name does not carry.
            if entry.name.endswith(".dist-info"):
                material.update(entry.name.encode())
    return material.hexdigest()[:16]


# ----- the derived corpus -----


#: Bumped when the shape of a derived tree changes: what is copied, what is linked, what is
#: sealed. It is part of a cache's name and of its stamp, so a tree built under an older layout is
#: a different cache rather than one this code will read as its own.
DERIVATION_VERSION = 1

#: What a derived cache was built from, written inside it once it is complete.
_SOURCE_FILE = ".shogym-source"


def derived_root(source: Optional[str] = None) -> Path:
    """Where the seeded copy of the corpus lives, named for what it was built from.

    Three things are in the name, and each of them changes what the tree holds. The generator
    digest covers the backlog generator's own constants, so changing a cut value, an option set or
    the number of requests derives a new corpus instead of serving a stale one. The derivation
    version covers the *layout*: what is copied, what is linked and what is sealed. And the source
    digest covers the corpus this was derived from, which used to be missing entirely:
    ``APPWORLD_ROOT`` takes any directory with a ``data/tasks`` in it, so a process pointed at a
    second corpus computed a fingerprint for that one and then reused and served task material
    derived from the first."""
    return cache_root() / f"seeded-{DATA_VERSION}-{DERIVATION_VERSION}-{_source(source)}"


def private_home() -> Path:
    """The directory holding everything an agent's world must not be handed.

    Not a sibling of the served root and not under this port's ordinary cache, because the served
    root's own path is in the worker's environment and a neighbour of it is a guess away."""
    base = cache_root().parent
    return base.parent / f"{base.name}-private" / "appworld"


def graded_root(source: Optional[str] = None) -> Path:
    """Where the grader's view of the corpus lives: a private directory with an unguessable name.

    **This raises the cost of finding it and does not close the route.** The worker runs as the
    same user as the process that built this, so no directory mode keeps it out: 0700 stops other
    users and stops nothing else. What closes it is a namespace in which the directory is not
    mounted at all, which is a container and is not built here (see the port's README). What this
    does is stop the tree being derivable from what the worker is given, which the previous
    layout, a fixed name beside the served root, was."""
    home = private_home()
    return home / f"graded-{DATA_VERSION}-{DERIVATION_VERSION}-{_source(source)}-{_private_tag()}"


def _read_tag(keyfile: Path) -> Optional[str]:
    """The published tag, or ``None`` if there is not a complete one there yet."""
    try:
        tag = keyfile.read_text().strip()
    except FileNotFoundError:
        return None
    return tag if len(tag) == 16 else None


def _source(source: Optional[str]) -> str:
    """The source digest a cache name is keyed by, computed from the served corpus if not given.

    The argument exists so that the env computes this once and hands the same value to both roots;
    the default is for callers that only want to name a path (tests, tooling) and would otherwise
    have to reach for the corpus themselves."""
    return f"{_generator_digest()}-{source or corpus_digest(ensure_corpus())}"


def stamp_cache(root: Path, *, source: str) -> None:
    """Record what a derived cache was built from, and refuse one built from something else.

    The name already carries the same values, so this cannot normally disagree. It is here for the
    case the name cannot cover: a tree edited, moved or restored in place under a name that still
    claims the old identity. A cache is the material a run is scored against, and the failure mode
    without this is silent, so the check is worth one file read per construction.

    Published by link rather than written in place, for the reason :func:`world._publish` gives:
    two cold processes reach this together and the loser must find the winner's whole stamp rather
    than half of its own."""
    root.mkdir(parents=True, exist_ok=True)
    stamp = root / _SOURCE_FILE
    material = json.dumps(
        {"source": source, "derivation": DERIVATION_VERSION, "data": DATA_VERSION}, sort_keys=True
    )
    held = ""
    try:
        held = stamp.read_text().strip()
    except OSError:
        held = ""
    if held:
        if held != material:
            raise ProvisioningError(
                f"the derived cache at {root} says it was built from {held}, but this run serves "
                f"{material}; a cache is the material a run is scored against, so this refuses "
                "rather than reusing it. Remove that directory, or point SHOGYM_CACHE elsewhere"
            )
        return
    staged = root / f"{_SOURCE_FILE}.{os.getpid()}.{secrets.token_hex(8)}"
    try:
        staged.write_text(material)
        os.link(staged, stamp)
    except FileExistsError:
        # Another process published first. Its stamp is the one that counts, and it is checked
        # rather than trusted: this is the same comparison as above.
        if stamp.read_text().strip() != material:
            raise ProvisioningError(
                f"the derived cache at {root} was stamped by another process with a different "
                "identity while this one was building it"
            ) from None
    finally:
        try:
            staged.unlink()
        except OSError:
            pass


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

    **Everything under ``data`` is read, not only the tasks.** This once covered ``version.txt``
    and the task tree, which left out ``base_dbs``, ``datasets`` and ``api_docs`` entirely. Those
    are 134 MB of starting state and documentation that every episode reads as input: a world
    built on different base databases is a different world, and a digest whose job is to say what
    the corpus holds cannot leave out the largest thing in it.

    **Not memoized, deliberately.** It was cached on the root path, so a corpus that changed under
    one path during a process kept the digest it had when the process first looked, which is the
    one case the cache would have had to answer. The env computes this once in its constructor and
    keeps the value; the cost is about two seconds on a fresh corpus, dominated by the fourteen
    thousand small files in the task tree, and it is the price of the digest meaning what it
    says."""
    digest = hashlib.sha256()
    data = root / "data"
    for path in sorted(data.rglob("*")):
        if path.is_symlink():
            # **Refused rather than skipped.** A skipped link is a file the digest does not cover
            # and derivation copies anyway: `_materialise` follows links, so the served world held
            # bytes the identity had never read, and changing what a link pointed at changed the
            # world without changing the digest that claims to say what the world is. The pinned
            # bundle contains none, so this refuses a corpus rather than growing a rule about how
            # to hash one.
            raise ProvisioningError(
                f"the corpus at {root} contains a symbolic link ({path.relative_to(data)}), and "
                "this port cannot state the identity of a tree whose contents are somewhere else"
            )
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(data)).encode())
        with path.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                digest.update(block)
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
    #: The group this worker leads, read once while the worker was certainly alive.
    #:
    #: A group used to be resolved from the stored pid at close time, and a pid is only that
    #: process's for as long as that process exists. An ordinary episode closes its worker twice:
    #: once when the sealed world is read, and once from teardown. Between them runs the grader,
    #: which is allowed ten minutes, and a pid freed at the start of that window can belong to
    #: something else by the end of it. Asked then, the kernel answers about the stranger, and the
    #: second close signals a group this port never started. Read at spawn, the answer is about
    #: this worker or it is nothing.
    pgid: Optional[int] = None
    #: Whether :meth:`close` has begun. Set before the first signal, so a close interrupted part
    #: way through is still a worker no later call will signal for.
    closed: bool = False
    #: Whether the execution domain is *provably* gone: the leader was signalled while it was
    #: alive, was reaped, and the process table was read and showed nothing left in the group.
    #: What finalize grades on, and false wherever the answer is "I could not tell".
    stopped: bool = False

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
        # Read here and kept, rather than resolved from the pid later: see `pgid`.
        pgid = _group_of(process)
        assert process.stdin is not None
        process.stdin.write(json.dumps({"root": str(root), "token": token}) + "\n")
        process.stdin.flush()
        process.stdin.close()
        assert process.stdout is not None
        line = _first_line(process, _SPAWN_TIMEOUT_SECONDS)
        if not line:
            _stop(process, signal.SIGKILL, pgid)
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
            pgid=pgid,
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

    def close(self, *, confirm: bool = False) -> None:
        """Stop the worker, promptly and with a bound, and say whether it is provably gone.

        Signalled and reaped rather than asked over the socket. There is no close command to ask
        for: this is the process that runs agent-authored code, so a reply from it saying that it
        had stopped is a reply the episode could have written. The fact the host needs is that the
        process stopped, and that fact is the kernel's.

        **The group, and not only its leader.** Agent code runs in that process and is free to
        spawn, so a signal to the direct child alone leaves descendants running against a world
        that has been scored. The group is signalled and what is waited for is the group emptying.

        **Only while the group is still this worker's.** A pgid is a number, and the kernel
        recycles numbers once nothing holds them. The group is therefore signalled before the
        leader is reaped, which is the window in which the number is certainly this worker's even
        if the leader is already a zombie; a leader something else reaped first leaves a number
        nothing here may use, and the honest report is then that the execution domain cannot be
        addressed rather than that it was cleaned up (see :meth:`_stop_the_group`).

        **Stateful, and that is what makes it idempotent.** An ordinary episode calls this twice:
        once from finalize, once from teardown. Between them runs the grader, which is allowed ten
        minutes, and that is long enough for a freed pid to be handed out again. Nothing here asks
        about a pid after the first call.

        ``confirm`` turns a best effort into an assertion. It returns only when the leader was
        signalled while it was alive, was reaped, and the process table was read and showed nothing
        left in the group. Anything else raises, including a process table that could not be read,
        because "I could not look" is not "there is nothing there" and an episode graded on the
        strength of it is graded on a tree something may still be writing to."""
        if not self.closed:
            self.closed = True
            self.stopped = self._stop_the_group()
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            shutil.rmtree(self.scratch, ignore_errors=True)
        if confirm and not self.stopped:
            raise WorkerError(
                "the appworld worker's execution domain could not be confirmed stopped, so there "
                "is no state it is safe to grade: the process group was not observed empty (see "
                "`Worker.close`)"
            )

    def _stop_the_group(self) -> bool:
        """Signal and reap this worker's group, and report whether it is provably empty.

        **Signalled before anything reaps the leader, which is what makes the number safe to
        use.** A pid is reserved until its parent reaps it, and a process group exists while it
        has any member, a zombie included. So a leader that exited on its own, even one an agent
        killed from inside its own block, is still holding this group when this runs: the number
        is unambiguously this worker's and the signal reaches whatever it started. What is never
        safe is signalling *after* the reap, because the kernel is free to hand the number on from
        that instant.

        The one case this cannot serve is a leader something else already reaped, which
        ``returncode`` is the record of. Then the number may be anybody's, so it is neither
        signalled nor enumerated and the stop is reported unconfirmed.

        The ordering matters for a second reason. Reaping first and refusing an already-exited
        leader would have handed an episode a way out of a bad score: kill the worker from inside
        a block and the seal cannot be confirmed, so the row is unscored rather than low. Under
        this order that episode is stopped, confirmed and graded on what upstream persisted, which
        is the same answer any other episode gets."""
        pgid = self.pgid
        if self.process.returncode is not None:
            # Reaped before this call, so the pid and the group number may already have been
            # handed on. Nothing is signalled and nothing is enumerated.
            return False
        if pgid is None:
            # A platform without `setsid`: there is no group, so the leader is all there is to
            # stop and there is nothing that could say what it started.
            _stop(self.process, signal.SIGTERM, pgid)
            try:
                self.process.wait(timeout=_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                _stop(self.process, signal.SIGKILL, pgid)
            return False
        # SIGTERM, a short grace, then SIGKILL whatever happened. The grace used to be ten
        # seconds of a signal any process may catch, ignore or take its time over, and every one
        # of those seconds is time in which the process that ran agent-authored code is still
        # able to write into the tree about to be graded. It is short because it buys only an
        # ordinary exit, and it ends in a signal nothing can decline.
        _signal_group(pgid, signal.SIGTERM)
        # The grace, watched through the process table rather than through `wait`: `wait` reaps,
        # and a group is enumerable only while its number is still held. A leader that exits
        # tidily becomes a zombie, which the enumeration steps over, so this returns as soon as
        # the group is really empty and otherwise costs the grace.
        _group_emptied(pgid, within=_TERM_GRACE_SECONDS)
        _signal_group(pgid, signal.SIGKILL)
        # Enumerated while the leader is still unreaped, and that ordering is the point. Reaping
        # releases the pid, and a process group exists only while it has a member: reap first and
        # the number is free, so an enumeration or an escalation after it is about whatever holds
        # it now. An unreaped leader is a zombie, which holds the number and is excluded from the
        # count (see `_group_members`), so this asks about descendants and about nothing else.
        empty = _group_emptied(pgid, within=_CLOSE_SECONDS) is True
        # Reaped last.
        try:
            self.process.wait(timeout=_CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
            return False
        return empty


class SnapshotError(RuntimeError):
    """The stopped output tree is not something that may be graded.

    Its own type because it is an episode-level failure with a cause worth naming: the tree the
    world left behind holds something that is not a plain file, and a grader that opened it would
    be resolving a path the agent chose in a process that also holds the answers."""


#: What a snapshot may hold before it is refused. An episode's real output tree is tens of
#: kilobytes across a few dozen files at a depth of four, so these bound a pathological case and
#: not an ordinary one: the tree was writable by the process that ran agent-authored code, and an
#: unbounded walk of it holds finalization open and fills the host's disk.
_SNAPSHOT_MAX_NODES = 20_000
_SNAPSHOT_MAX_BYTES = 1 << 30
_SNAPSHOT_MAX_DEPTH = 24
_SNAPSHOT_SECONDS = 60.0


def snapshot_outputs(
    outputs: Path, *, into: Path, stop: "Optional[threading.Event]" = None
) -> Path:
    """Copy a stopped episode's output tree into one the grader can be given, or refuse.

    **Why a copy and not the tree itself.** The grading process is pointed at the root that holds
    the answers, and it also has to read the state to grade, which was writable by the process
    that ran the agent's code. A symlink left in there resolves in the grader, so a link planted
    under the output tree could make the digest, the filing and the evaluator read bytes from the
    graded tree instead of from what the episode submitted. Nothing here returns those bytes to
    the agent, so this is score integrity rather than a leak, and it is still not a thing to leave
    open.

    **The root is checked before it is resolved.** ``resolve()`` erases the question: a root that
    was itself a symlink came back as whatever it pointed at, and only its descendants were then
    inspected, so substituting the episode's own output directory substituted the whole tree. It
    is ``lstat``-ed first, and a link there refuses the episode like a link anywhere else.

    **Bounded in four ways, because this walks a tree an episode wrote.** Nodes, bytes, depth and
    elapsed time. Without them a large or deep tree holds finalization open for as long as it
    likes and fills the host's disk on the way, and neither the finalize deadline nor the grader's
    own timeout covers it: a deadline cancels the *await*, and the thread doing the copying does
    not stop for that. Which is what ``stop`` is for: the caller sets it when its await is
    cancelled and this checks it once per file, so a cancelled finalization stops the copy at the
    next file rather than at the end of the tree.

    Every refusal is an episode refused outright rather than an entry skipped, because a grade
    computed over a tree with something quietly dropped is a grade over a tree nobody submitted.

    Safe to walk because the worker is already gone: this runs after a confirmed stop, so nothing
    can add a link between the check and the copy."""
    if outputs.is_symlink():
        raise SnapshotError(
            f"the episode's output root {outputs} is a symbolic link, so what would be graded is "
            "whatever it names rather than what the episode wrote"
        )
    root = outputs.resolve()
    if not root.is_dir():
        raise SnapshotError(f"the episode left no output tree at {outputs}")
    began = time.monotonic()
    nodes = 0
    total = 0
    shutil.rmtree(into, ignore_errors=True)
    into.mkdir(parents=True)
    # One pass: what is checked is what is copied. A validate-then-`copytree` would walk the tree
    # twice and bound neither walk.
    pending: List[Tuple[Path, Path, int]] = [(root, into, 0)]
    while pending:
        source, target, depth = pending.pop()
        if depth > _SNAPSHOT_MAX_DEPTH:
            raise SnapshotError(
                f"the episode's output tree is deeper than {_SNAPSHOT_MAX_DEPTH} directories"
            )
        for entry in sorted(source.iterdir()):
            if stop is not None and stop.is_set():
                raise SnapshotError("the snapshot was abandoned before it finished")
            if time.monotonic() - began > _SNAPSHOT_SECONDS:
                raise SnapshotError(
                    f"the episode's output tree took longer than {_SNAPSHOT_SECONDS:.0f}s to copy"
                )
            nodes += 1
            if nodes > _SNAPSHOT_MAX_NODES:
                raise SnapshotError(
                    f"the episode's output tree holds more than {_SNAPSHOT_MAX_NODES} entries"
                )
            if entry.is_symlink():
                raise SnapshotError(
                    f"the episode left a symbolic link in its output tree ({entry.name} -> "
                    f"{os.readlink(entry)}), which a grader must not resolve"
                )
            if entry.is_dir():
                (target / entry.name).mkdir()
                pending.append((entry, target / entry.name, depth + 1))
                continue
            if not entry.is_file():
                raise SnapshotError(
                    f"the episode left {entry.name}, which is not a file or directory"
                )
            total += entry.stat().st_size
            if total > _SNAPSHOT_MAX_BYTES:
                raise SnapshotError(
                    f"the episode's output tree is larger than {_SNAPSHOT_MAX_BYTES} bytes"
                )
            shutil.copyfile(entry, target / entry.name, follow_symlinks=False)
    return into


def grade(
    *,
    root: Path,
    task_id: str,
    outputs: Path,
    ignore: Sequence[str],
    filing: Dict[str, str],
    timeout: float = _GRADE_TIMEOUT_SECONDS,
) -> Any:
    """The base task's own checks and the episode's own filing, from a process that has never run
    a line the agent wrote.

    A second, short-lived worker rather than the one that served the episode. It is the only place
    ground truth is loaded, it starts after the serving worker has been confirmed stopped, and it
    reads the end state off disk, so the answers are never objects in the process the agent's code
    ran as.

    **The filing and the digests come back from here too.** They used to be asked of the serving
    world over the protocol, which made the process that runs agent-authored code the process
    reporting what the episode had done. One process now reads one stopped tree, so the filing,
    the databases' digest and the evaluator's verdicts are one state by construction rather than
    two observations that happened to agree."""
    opening = json.dumps(
        {
            "root": str(root),
            "task_id": task_id,
            "experiment": str(outputs),
            "ignore": list(ignore),
            "filing": dict(filing),
        }
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
    """The worker's process group, asked once while the answer is still about this worker.

    Called at spawn and nowhere else. A pid names a process only while that process exists, so
    asking later is asking about whoever holds the pid then (see :attr:`Worker.pgid`)."""
    try:
        return os.getpgid(process.pid)
    except (OSError, AttributeError):
        return None


def _signal_group(pgid: int, how: int) -> None:
    try:
        os.killpg(pgid, how)
    except (OSError, AttributeError):
        pass


def _group_members(pgid: int) -> Optional[List[int]]:
    """Every live process still in ``pgid`` but this one, or ``None`` if the table was unreadable.

    Asked of `ps` rather than of `/proc`, which macOS does not have.

    **A table this could not read is not an empty table.** It used to answer both with an empty
    sequence, so a `ps` that would not run reported the same fact as a group that had emptied, and
    a caller confirming a stop confirmed it on no evidence. The two answers are now different
    values and the caller that needs proof treats the missing one as a refusal.

    Exited-but-unreaped entries are excluded. A process that was killed sits in the table until
    somebody waits on it, and a zombie holds no memory, no descriptors and no ability to write, so
    counting one as live would report a completed stop as an incomplete one."""
    try:
        listing = subprocess.run(
            ["ps", "-o", "pid=,pgid=,stat=", "-A"],
            capture_output=True,
            text=True,
            timeout=_CLOSE_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if listing.returncode != 0:
        return None
    live: List[int] = []
    mine = os.getpid()
    for line in listing.stdout.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        try:
            pid, group = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if group == pgid and pid != mine and not fields[2].startswith("Z"):
            live.append(pid)
    return live


def _group_emptied(pgid: int, *, within: float) -> Optional[bool]:
    """Whether ``pgid`` emptied inside ``within`` seconds, or ``None`` if that cannot be read.

    Waiting for the leader says nothing about what the leader started. Agent code runs in that
    process and may have left something behind, and something still running after the world has
    been scored is either changing what was scored or holding a port the next episode wants.

    Three answers rather than two, because the caller grades on the strength of this: ``True`` is
    a process table that was read and held nothing of this group, ``False`` is one that still did
    when the time ran out, and ``None`` is a table that could not be read at all.

    Escalation belongs to the caller. This used to send its own SIGKILL when its first deadline
    passed, which made "how long a tidy exit gets" and "how long a killed group gets to disappear"
    one number and put the signal in the helper that reports rather than in the one that stops
    (see :meth:`Worker._stop_the_group`)."""
    deadline = time.monotonic() + within
    while True:
        members = _group_members(pgid)
        if members is None:
            return None
        if not members:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.02)


def _stop(process: subprocess.Popen, how: int, pgid: Optional[int]) -> None:
    """Signal the worker's whole process group, or the worker alone if it has no group of its own.

    The group is the point: agent code runs in that process and may have started others, and a
    signal to the direct child alone leaves them running against a world that has been scored.

    ``pgid`` is passed in rather than looked up. The group is read once, while the process is
    certainly alive, and a signal aimed at a group resolved from a pid afterwards is a signal
    aimed at whoever holds that pid now (see :attr:`Worker.pgid`).

    **A worker that has a group is signalled through it and through nothing else.** A ``killpg``
    that fails on a group this worker leads means the group is empty, which is the answer "there
    is nothing left to signal", and not an invitation to try the stored pid, which is the very
    value that may since have been handed to somebody else. The pid is the fallback only where there
    was never a group to begin with, which is a platform without ``setsid`` rather than a worker
    that has finished."""
    if pgid is not None:
        try:
            os.killpg(pgid, how)
        except (OSError, AttributeError):
            pass
        return
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
    "DERIVATION_VERSION",
    "MANIFEST",
    "ProvisioningError",
    "ROOT_ENV_VAR",
    "SPLIT",
    "SnapshotError",
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
    "runtime_digest",
    "snapshot_outputs",
    "stamp_cache",
    "task_ids",
    "task_specs",
]
