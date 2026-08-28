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
import stat
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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from shogym.envs._upstream import _locked

# ----- the pins -----

#: The release this port reproduces, and the commit it was cut from.
#:
#: **Both are load-bearing, and they are load-bearing in different ways.** The version is what is
#: installed and what the installed distribution is checked to be (see :func:`_check_pin`), so a
#: runtime holding some other release is refused rather than served. The commit is not checkable
#: against the artifact at all: the published wheel carries no marker of the tree it was built
#: from, so nothing on this machine can say whether ``0.1.3.post1`` was cut from this commit. What
#: it does instead is *name* the runtime and go into its stamp, so changing the pin builds a
#: second interpreter under a second name rather than reusing the first, and every run's
#: fingerprint moves with it. What the realized bytes turned out to be is :func:`runtime_digest`'s
#: job, which is the question a pin can never answer.
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

#: How long one provisioning subprocess gets: creating the venv, resolving and installing the
#: pinned release, unpacking the app sources, unpacking the bundle. See :func:`_run`.
_PROVISION_TIMEOUT_SECONDS = 1800.0

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


#: What a built runtime says it is, written inside it once it is complete and checked before it is
#: reused. The name carries the same values; this is for the case a name cannot cover, which is a
#: tree replaced, restored or half-deleted under a name that still claims the old identity.
_RUNTIME_FILE = ".shogym-runtime"

#: What a runtime whose app sources have been unpacked says about itself, written inside it once
#: the unpack has finished. Beside the runtime stamp rather than inside the installed package,
#: because a runtime that is rebuilt is published by a rename over this directory and takes its
#: marker with it, and because it can then be read without starting the interpreter to find out
#: where the package lives (see :func:`ensure_apps`).
_APPS_FILE = ".shogym-apps"


def runtime() -> Path:
    """The interpreter every world runs under, building it if it is not there yet.

    A virtual environment rather than the running one, because ``appworld`` pins ``pydantic<2``
    and shogym needs ``pydantic>=2.7``: installing it beside shogym is not a thing pip will do,
    and a port that pretended otherwise would fail at resolve time on every machine. Built with
    ``uv`` where it is on the path and with the standard library's own tools otherwise, so the
    port needs no tool the user did not already have.

    **Named for both pins, and reused only against its own stamp.** The old test was that
    ``bin/python`` existed, which is true of a venv whose install failed after the interpreter was
    created, of one a later pin change should have rebuilt, and of one somebody edited. The name
    now carries the version and the commit, and the stamp inside says the same thing, so what is
    reused is a tree this code finished building under these pins rather than a directory that
    happens to hold a Python."""
    home = cache_root() / f"runtime-{UPSTREAM_VERSION}-{UPSTREAM_SHA[:12]}"
    python = _interpreter(home)
    if _stamped(home) and python.exists():
        return python
    home.parent.mkdir(parents=True, exist_ok=True)
    # Required, not advisory: the staging tree below has a fixed name and is deleted before it is
    # written, so two cold builders without exclusion remove and publish each other's half-built
    # interpreter. See `_upstream._locked`.
    with _locked(home.parent, required=True):
        if _stamped(home) and python.exists():
            return python
        _build_runtime(home)
    return python


def _runtime_stamp() -> str:
    """What a runtime built by this code under these pins says about itself."""
    return json.dumps(
        {"version": UPSTREAM_VERSION, "sha": UPSTREAM_SHA, "python": _python_series()},
        sort_keys=True,
    )


def _stamped(home: Path) -> bool:
    """Whether ``home`` holds this code's own stamp, refusing one that holds a different stamp.

    A missing stamp is a cold cache, or a runtime built by a head that had none, and both are
    answered by building. A stamp that says something else is a tree under a name that claims an
    identity it does not have, and the only honest answer to that is to stop: the interpreter is
    what every world runs under, so serving out of one whose provenance disagrees with its own
    name would put the disagreement in the scores rather than in an error."""
    try:
        held = (home / _RUNTIME_FILE).read_text().strip()
    except OSError:
        return False
    if held != _runtime_stamp():
        raise ProvisioningError(
            f"the appworld runtime at {home} says it was built as {held}, but this run pins "
            f"{_runtime_stamp()}; the interpreter is what every world runs under, so this refuses "
            "rather than serving out of it. Remove that directory, or point SHOGYM_CACHE elsewhere"
        )
    return True


def _build_runtime(home: Path) -> None:
    """Create the environment, install the pinned release into it, and check what arrived.

    Checked inside the staging tree and stamped there, before the rename that publishes it. So the
    final name never exists holding an interpreter that failed its own pin, and a builder that
    dies part way through leaves a staging directory rather than a runtime other processes would
    reuse."""
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
    _check_pin(staging)
    (staging / _RUNTIME_FILE).write_text(_runtime_stamp())
    _publish_runtime(staging, home)


def _publish_runtime(staging: Path, home: Path) -> None:
    """Give the staged tree its final name, moving any incumbent aside first.

    ``os.replace`` will not rename a directory over a non-empty one, and there is now a way to
    reach this with something already under the name: a tree that is not one this code stamped is
    rebuilt rather than reused, and the tree it is rebuilding over is still there. So the
    incumbent is renamed away and the publish stays the single atomic step it has to be. A worker
    already running out of the old tree keeps every file it has open, because a rename moves a
    name and not an inode; what it loses is the ability to open new ones by that path, which is
    the same thing it would lose to any rebuild.

    Removing the incumbent afterwards is housekeeping and is allowed to fail: the publish has
    already happened by then, and a directory left behind costs disk rather than correctness."""
    displaced: Optional[Path] = None
    if home.exists():
        displaced = home.with_name(f"{home.name}.replaced.{os.getpid()}.{secrets.token_hex(8)}")
        os.replace(home, displaced)
    os.replace(staging, home)
    if displaced is not None:
        shutil.rmtree(displaced, ignore_errors=True)


def _check_pin(home: Path) -> None:
    """Refuse a runtime whose installed ``appworld`` is not the pinned release.

    The requirement string asks for one version and the resolver is what answers; a build that
    resolved something else, or an index that moved under the name, would otherwise be served
    under a cache name that says the pin was honored. Read off the distribution metadata rather
    than by running the interpreter, so a broken install fails here rather than at the first
    episode.

    **This is the whole of what the pins can be checked against, and it is half of one of them.**
    The wheel carries no record of the commit it was cut from, so :data:`UPSTREAM_SHA` is not
    verifiable here or anywhere else on this machine (see the pins). What covers the realized code
    is :func:`runtime_digest`, which reads the bytes rather than the label."""
    found = sorted(
        entry.name[: -len(".dist-info")]
        for packages in _site_packages(home)
        for entry in packages.iterdir()
        if entry.name.startswith("appworld-") and entry.name.endswith(".dist-info")
    )
    if found != [f"appworld-{UPSTREAM_VERSION}"]:
        raise ProvisioningError(
            f"the appworld runtime at {home} installed {found or 'no appworld distribution'}, "
            f"but this port pins appworld=={UPSTREAM_VERSION}; every task, every database and "
            "every ground truth in the measurement is read by this interpreter, so a release "
            "nobody asked for is refused rather than served"
        )


def _site_packages(home: Path) -> List[Path]:
    """The provisioned interpreter's ``site-packages`` directories, in a stable order.

    Both layouts, because the port names an interpreter and not a platform: POSIX venvs put it
    under ``lib/pythonX.Y/`` and Windows ones under ``Lib/``. One definition, because the pin
    check and the runtime digest have to be looking at the same tree."""
    found = list(home.glob("lib/python*/site-packages")) + list(home.glob("Lib/site-packages"))
    return sorted(path for path in found if path.is_dir())


def _interpreter(home: Path) -> Path:
    return home / ("Scripts" if os.name == "nt" else "bin") / "python"


def _python_series() -> str:
    """The Python this port asks for, matching the one shogym is running under."""
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def _run(command: list, *, timeout: float = _PROVISION_TIMEOUT_SECONDS) -> None:
    """Run a provisioning command, and say what it was if it fails or if it never finishes.

    **Bounded, because every one of these is a network call wearing a subprocess.** A ``pip
    install`` against an index that accepts the connection and then stops sending has no timeout of
    its own, and neither does an unpack whose child wedged. Construction is what waits on this, and
    a construction that never returns is a queue that never starts and a run that reports nothing
    at all, which is strictly worse than a run that says which command hung. The bound is generous
    on purpose: it is a liveness bound and not a performance one, so a cold resolve on a slow link
    stays inside it and a wedge does not."""
    try:
        finished = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as expired:
        # `subprocess.run` has already killed and reaped the child by the time this is raised.
        raise ProvisioningError(
            f"provisioning the appworld runtime timed out after {timeout:.0f}s: "
            f"{' '.join(command)}"
        ) from expired
    if finished.returncode != 0:
        raise ProvisioningError(
            f"provisioning the appworld runtime failed: {' '.join(command)}\n"
            f"{finished.stderr.strip()[-2000:]}"
        )


def ensure_apps() -> None:
    """Unpack the app sources the wheel ships packed, if they are not unpacked already.

    A fresh install has ``appworld.apps`` with the shared library in it and none of the nine
    apps, so the first import of a model module fails with a plain ``ModuleNotFoundError`` that
    says nothing about the missing step. Idempotent, and silent when there is nothing to do.

    **What says it is done is a stamp written after the unpack returned, not a file the unpack
    happens to create early.** The test used to be that ``apps/todoist/models.py`` existed.
    Upstream's ``install_package`` goes on doing in-place work after the individual app files are
    there, and the runtime is already published and stamped by the time this runs, so an unpack
    interrupted anywhere after that one file appeared left a runtime every later construction
    skipped: a half-unpacked interpreter under a name and a stamp that both say it is finished.
    The stamp below is written only once the unpacking subprocess has exited zero, so what is
    trusted is a completed unpack rather than a step of one.

    **And it is read before the interpreter is asked anything**, which is what makes the warm path
    free. Locating the installed package means importing ``appworld`` in a subprocess, and that is
    most of a second on every env construction; the stamp lives in the runtime's own directory, so
    a runtime that has been unpacked is recognised without starting a process at all."""
    python = runtime()
    marker = python.parent.parent / _APPS_FILE
    if _apps_unpacked(marker):
        return
    installed = _installed_package(python)
    # Required: this one unpacks *in place*, into the interpreter every world runs under, so two
    # unpackers without exclusion are two processes writing one tree. There is no staging name to
    # publish from and no rename to make it atomic.
    with _locked(installed, required=True):
        if _apps_unpacked(marker):
            return
        _run([str(python), str(WORKER), "install"])
        _compile_runtime(python)
        _mark_apps(marker)


def _compile_runtime(python: Path) -> None:
    """Rewrite every bytecode cache in the interpreter as a hash-based one.

    **This is what lets :func:`runtime_digest` leave ``__pycache__`` out and still be true.** A
    default ``.pyc`` records the source's modification time and size and is honoured whenever those
    two numbers still match, whatever bytes the cache itself holds, so a cache under the installed
    tree is executable code that no digest over the sources has read. A hash-based cache records a
    hash of the source and the import system checks it on every import, so a cache that disagrees
    with the source beside it is discarded and the source is what runs. Every ``.pyc`` a worker can
    consult then stands for a source :func:`runtime_digest` did read.

    Forced, because ``compileall`` decides whether a cache is current by comparing a *timestamp*
    header, so an existing timestamp cache that is up to date is skipped and left in the form this
    is here to replace.

    Run after the app sources are unpacked, because those are the modules with the most in them,
    and run once: three seconds at provisioning, against three on every worker start, which is
    what sending each worker to a cache directory of its own would have cost instead."""
    _run(
        [
            str(python),
            "-m",
            "compileall",
            "-q",
            "-f",
            "-j",
            "0",
            "--invalidation-mode",
            "checked-hash",
            *(str(packages) for packages in _site_packages(python.parent.parent)),
        ]
    )


def _apps_stamp() -> str:
    """What a runtime whose app sources are unpacked says about itself."""
    return json.dumps({"version": UPSTREAM_VERSION, "sha": UPSTREAM_SHA, "apps": "unpacked"})


def _apps_unpacked(marker: Path) -> bool:
    """Whether this runtime's apps were unpacked by a run that got to the end of the unpack.

    Anything other than this code's own stamp reads as not done and is answered by unpacking
    again, which is idempotent. There is no refusal here of the kind :func:`_stamped` makes: the
    runtime directory the marker sits in already carries the pins in its name and in its own
    stamp, so a marker that says something else is a leftover rather than a claim to be an
    interpreter this run did not build."""
    try:
        return marker.read_text().strip() == _apps_stamp()
    except OSError:
        return False


def _mark_apps(marker: Path) -> None:
    """Record that the unpack finished, whole or not at all.

    Written to a name of this process's own and renamed into place, because a marker written
    directly is a name that exists before it holds anything: a crash between the create and the
    write leaves a file that is not this stamp, which reads as unfinished and unpacks again, and a
    concurrent reader would see the same. ``os.replace`` is atomic within a directory."""
    staged = marker.with_name(f"{marker.name}.{os.getpid()}.{secrets.token_hex(8)}")
    staged.write_text(_apps_stamp())
    os.replace(staged, marker)


#: How long the provisioned interpreter gets to import ``appworld`` and say where it lives.
#:
#: Bounded for the reason :func:`_run` is bounded, and it is the same reason stated about a
#: different subprocess: env construction waits on this one, and a construction that never returns
#: is a queue that never starts and a run that reports nothing at all. Generous, because it is a
#: cold import of a large package on a machine that may just have installed it.
_IMPORT_TIMEOUT_SECONDS = 300.0


def _installed_package(python: Path) -> Path:
    """Where ``appworld`` sits inside the provisioned interpreter.

    **Under a deadline**, which it was not. This starts a process and waits for it with no bound,
    on an interpreter that was provisioned rather than shipped: an import that wedges on a broken
    shared library or a filesystem that stopped answering held construction open for the life of
    the run, before any task existed to file a timeout row against."""
    try:
        finished = subprocess.run(
            [str(python), "-c", "import appworld, os; print(os.path.dirname(appworld.__file__))"],
            capture_output=True,
            text=True,
            timeout=_IMPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as expired:
        # `subprocess.run` has already killed and reaped the child by the time this is raised.
        raise ProvisioningError(
            f"the provisioned appworld runtime at {python} did not finish importing appworld "
            f"within {_IMPORT_TIMEOUT_SECONDS:.0f}s"
        ) from expired
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
    # Required, for the reason the runtime's lock is: `_fetch_corpus` stages under a fixed
    # `.building` name it deletes first, so two cold fetchers without exclusion delete and publish
    # each other's half-unpacked corpus, and the corpus is the material every score is computed
    # against.
    with _locked(root.parent, required=True):
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
    318.

    **A served env does not call this.** It reads its whole roster's specs once, out of the same
    walk that computes its corpus digest (:func:`corpus_snapshot`), and serves those for its life:
    an env whose fingerprint and cache names were fixed at construction must not go on reading
    authored text from a corpus that can change under it. This remains for the callers that want
    one spec off a corpus as it stands rather than a pinned view of one, which is tooling and the
    tests that check a committed table against the corpus it was generated from."""
    return json.loads((root / "data" / "tasks" / task_id / "specs.json").read_text())


def runtime_digest() -> str:
    """What the worker's interpreter actually holds, as sixteen hex characters.

    The runtime cache is named for the pins, while it is built by resolving the pinned release's
    ranged dependencies against whatever the host's Python and the index offer on the day. Two
    machines, or one machine a month apart, therefore run a world under different transitive
    versions under a name that says they are the same. This reads what was realized rather than
    what was asked for.

    **Every installed byte, not the distribution names.** This once hashed the platform, the venv
    config and the ``.dist-info`` directory *names*, which is a list of what was asked for a second
    time: two different artifacts published under one version were one identity, and so was a
    module edited in place, which is the one thing a digest over a tree an interpreter runs from
    exists to catch. A ``.dist-info`` name is a label, and a label is not a fingerprint of the code
    behind it.

    **What is hashed, exactly: the installed source and data bytes of ``site-packages``, plus the
    base interpreter's own binary.** Every regular file under every ``site-packages`` of the
    virtual environment, by relative path and by content; every symbolic link there by its target
    text, by where that text resolves to and by the resolved file's bytes; ``pyvenv.cfg``; the
    platform; the pins; and the executable ``bin/python`` resolves to, which is the real
    interpreter the venv borrows and which nothing else here would have covered.

    **What is not hashed, and why each is out of scope.** The base interpreter's *standard
    library* is not read: it is thousands of files belonging to the host's Python rather than to
    anything this port installs, and the base binary's own bytes already move with a reinstalled
    or repointed interpreter. Bytecode caches are not read: see below. A digest that named those
    would be a longer walk and a claim this code cannot keep, so the claim is the narrower one and
    it is true.

    **``__pycache__`` is skipped, and the runtime is built so that skipping it is not a hole.**
    Bytecode caches are written lazily by whatever process imported a module first, so hashing
    them would make the interpreter's identity depend on which worker ran before rather than on
    what is installed. What makes leaving them out honest is not that a stale cache is unlikely to
    be executed: an ordinary ``.pyc`` whose recorded source mtime and size match the source beside
    it is executed whatever its contents say. It is that every cache in this interpreter is
    rewritten at provisioning as a hash-based one, which the import system checks against the
    source's own hash, and that a worker writes none back (see :func:`_compile_runtime` and
    :func:`_worker_environment`). Every ``.pyc`` that can be consulted therefore stands for a
    source this function read.

    Read off the filesystem rather than by running the interpreter, so a broken install still gets
    an answer. It costs about a second warm over the roughly fifteen thousand installed files of
    this runtime, of which the base binary is under ten milliseconds, paid once per env
    construction beside a corpus digest that costs half as much again (:func:`corpus_digest` is
    the only other thing on the same path, and both are read once in a constructor). Not
    memoized, for the reason :func:`corpus_digest` is not."""
    python = runtime()
    home = python.parent.parent
    material = hashlib.sha256()
    material.update(f"{platform.system()}|{platform.machine()}".encode())
    material.update(f"{UPSTREAM_VERSION}|{UPSTREAM_SHA}".encode())
    config = home / "pyvenv.cfg"
    material.update(config.read_bytes() if config.exists() else b"")
    # The interpreter the venv borrows. `pyvenv.cfg` names its directory, which is a label; this
    # is the executable that actually runs, read through whatever chain of links `bin/python` is.
    _absorb_target(material, b"base", python)
    for packages in _site_packages(home):
        for relative, path in _installed_files(packages):
            material.update(relative.encode())
            material.update(b"\0")
            if path.is_symlink():
                material.update(b"link\0")
                material.update(os.readlink(path).encode())
                material.update(b"\0")
                # And what the link resolves to, which the target text alone does not say: a
                # relative target reaches a different file from a different directory, and the
                # bytes behind an unchanged target can change without the text moving.
                _absorb_target(material, b"target", path)
                continue
            material.update(b"file\0")
            with path.open("rb") as handle:
                while True:
                    block = handle.read(1 << 20)
                    if not block:
                        break
                    material.update(block)
    return material.hexdigest()[:16]


def _absorb_target(material: Any, label: bytes, path: Path) -> None:
    """Fold in where ``path`` resolves to and the bytes that are there, if it is a regular file.

    Resolved and then read, rather than opened through the link: what goes into the identity is
    the pair, because the same target text under two directories is two files and the same file
    under one name can be replaced. A resolution that is not a regular file (a directory, a
    dangling link, a device) contributes its path and the fact that it is not readable as bytes;
    following a link to a *directory* would make this walk unbounded, which is the same reason
    :func:`_installed_files` never descends into one."""
    material.update(label)
    material.update(b"\0")
    try:
        resolved = Path(os.path.realpath(path))
        material.update(str(resolved).encode())
        material.update(b"\0")
        if not resolved.is_file():
            material.update(b"not-a-file\0")
            return
        with resolved.open("rb") as handle:
            while True:
                block = handle.read(1 << 20)
                if not block:
                    break
                material.update(block)
    except OSError:
        material.update(b"unreadable\0")


def _installed_files(packages: Path) -> "Iterator[Tuple[str, Path]]":
    """Every file under ``packages`` bar the bytecode caches, in one order on every machine.

    Sorted at each level rather than by collecting and sorting the whole tree, so a digest over a
    hundred thousand files still holds one directory's names at a time.

    A symlinked *directory* is yielded like any other link and never descended into, so it
    contributes where it points and not a second copy of whatever is over there. Descending would
    also be how one link makes this walk unbounded."""
    for directory, children, names in os.walk(packages, followlinks=False):
        here = Path(directory)
        linked = [child for child in children if (here / child).is_symlink()]
        children[:] = sorted(
            child for child in children if child != "__pycache__" and child not in linked
        )
        for name in sorted([*names, *linked]):
            path = here / name
            yield str(path.relative_to(packages)), path


# ----- the derived corpus -----


#: Bumped when the shape of a derived tree changes: what is copied, what is linked, what is
#: sealed. It is part of a cache's name and of its stamp, so a tree built under an older layout is
#: a different cache rather than one this code will read as its own.
DERIVATION_VERSION = 1

#: What a derived cache was built from, written inside it once it is complete.
_SOURCE_FILE = ".shogym-source"


def derived_root(source: Optional[str] = None, *, runtime: Optional[str] = None) -> Path:
    """Where the seeded copy of the corpus lives, named for what it was built from.

    Four things are in the name, and each of them changes what the tree holds. The generator
    digest covers the backlog generator's constants *and its implementation*, so changing a cut
    value, an option set, the number of requests or the code that draws them derives a new corpus
    instead of serving a stale one. The derivation version covers the *layout*: what is copied,
    what is linked and what is sealed. The runtime digest covers the interpreter that wrote the
    seeded rows, because it is the interpreter and not this process that writes them: a task's
    database file is a replayable statement log written through upstream's own model layer, so a
    resolved dependency that changed how a row is serialized changed the world under a name that
    had not moved. And the source digest covers the corpus this was derived from, which used to be
    missing entirely: ``APPWORLD_ROOT`` takes any directory with a ``data/tasks`` in it, so a
    process pointed at a second corpus computed a fingerprint for that one and then reused and
    served task material derived from the first.

    ``runtime`` is passed in rather than read, for the reason ``source`` is: an env reads it once
    at construction and hands the same value to both roots and to its own fingerprint, so the
    three cannot disagree, and a caller that only wants to name a path does not have to."""
    return cache_root() / f"seeded-{DATA_VERSION}-{DERIVATION_VERSION}-{_source(source, runtime)}"


def private_home() -> Path:
    """The directory holding everything an agent's world must not be handed.

    Not a sibling of the served root and not under this port's ordinary cache, because the served
    root's own path is in the worker's environment and a neighbour of it is a guess away."""
    base = cache_root().parent
    return base.parent / f"{base.name}-private" / "appworld"


def graded_root(source: Optional[str] = None, *, runtime: Optional[str] = None) -> Path:
    """Where the grader's view of the corpus lives: a private directory with an unguessable name.

    **This raises the cost of finding it and does not close the route.** The worker runs as the
    same user as the process that built this, so no directory mode keeps it out: 0700 stops other
    users and stops nothing else. What closes it is a namespace in which the directory is not
    mounted at all, which is a container and is not built here (see the port's README). What this
    does is stop the tree being derivable from what the worker is given, which the previous
    layout, a fixed name beside the served root, was."""
    home = private_home()
    return (
        home
        / f"graded-{DATA_VERSION}-{DERIVATION_VERSION}-{_source(source, runtime)}-{_private_tag()}"
    )


def _read_tag(keyfile: Path) -> Optional[str]:
    """The published tag, or ``None`` if there is not a complete one there yet."""
    try:
        tag = keyfile.read_text().strip()
    except FileNotFoundError:
        return None
    return tag if len(tag) == 16 else None


def _source(source: Optional[str], runtime: Optional[str] = None) -> str:
    """Everything but the layout that a cache name is keyed by: the generator, the interpreter
    that ran it, and the corpus it read.

    The arguments exist so that the env reads each of these once and hands the same values to both
    roots and to its own fingerprint; the defaults are for callers that only want to name a path
    (tests, tooling) and would otherwise have to reach for the corpus and the runtime
    themselves."""
    return "-".join(
        (
            _generator_digest(),
            runtime if runtime is not None else runtime_digest(),
            source or corpus_digest(ensure_corpus()),
        )
    )


def stamp_cache(root: Path, *, source: str, runtime: str) -> None:
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
        {
            "source": source,
            "runtime": runtime,
            "generator": _generator_digest(),
            "derivation": DERIVATION_VERSION,
            "data": DATA_VERSION,
        },
        sort_keys=True,
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


@dataclass(frozen=True)
class CorpusSnapshot:
    """One reading of a corpus: what it held, the authored text it held, and how to prove it still
    holds it.

    The first two travel together because they have to be one observation. A digest and a later
    ``specs.json`` read are two, and the gap between them is a corpus that can change: the env
    fixed its fingerprint and its cache names from the first and then went on serving instructions
    and dates read from the second, so an in-place edit served authored text under an identity
    that had never seen it. Here the spec is parsed from the very bytes the digest was computed
    over, so ``digest`` states what ``specs`` came from and there is no window between them to
    edit.

    **``units`` is what closes the same gap for everything the digest is not made of.** Holding a
    task's *text* was never enough: derivation copies a task's databases and its ground truth out
    of the live corpus, and it copies the shared base out of it too, so a corpus edited after
    construction could still put changed bytes into a served world and into the tree it is graded
    against, under the unchanged ``config_digest`` computed before the edit. Rehashing all 183 MB
    before every task is a second and a half nobody can pay per episode. So the one walk that
    computes the whole digest also records a digest per *unit*, a unit being one task or one
    top-level entry of ``data``, and derivation checks the unit it is about to read (see
    :meth:`verify`)."""

    #: What the whole of ``data`` held, as sixteen hex characters.
    digest: str
    #: Task id to its shipped specification, for the tasks that were asked for and no others.
    specs: Dict[str, Dict[str, Any]]
    #: Unit name to what that unit held, as sixteen hex characters. A unit is ``tasks/<task_id>``
    #: or a top-level name under ``data``; the whole corpus is partitioned by them.
    units: Dict[str, str]

    def verify(self, root: Path, unit: str) -> None:
        """Refuse to derive from bytes that are not the ones this snapshot read.

        **What this verifies:** that every regular file under ``data/<unit>`` of the corpus at
        ``root`` is byte-for-byte what it was when this snapshot was taken, path for path. For a
        task that is its ``specs.json``, its databases and its ground truth, which is the whole of
        what deriving that task reads. For a shared entry it is every file of it.

        **What it does not:** anything outside the named unit, and the instant after it returns. A
        corpus edited *during* the derivation it guards is not covered, and cannot be by anything
        short of a copy taken under the same lock. What it removes is the window that mattered,
        which was measured in minutes and episodes rather than in the microseconds between this
        check and the read that follows it: an env states its identity once at construction and
        then derives tasks lazily, first-use, for as long as the run lasts.

        Cost is the unit's own size: about a millisecond for a task's 32 KB, paid once per task on
        the cold path and never on the warm one, because derivation asks nothing of a task that is
        already on disk."""
        held = self.units.get(unit)
        found = _unit_digest(root / "data", unit)
        if held is None or found != held:
            raise ProvisioningError(
                f"the corpus at {root} no longer holds the {unit} this run was built against "
                f"(read as {held or 'absent'}, now {found or 'absent'}); every task, every "
                "database and every ground truth in the measurement comes out of these bytes, so "
                "this refuses to derive from them rather than serving a world the run's identity "
                "has never seen"
            )


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
    return corpus_snapshot(root, task_ids=()).digest


def corpus_snapshot(root: Path, *, task_ids: Sequence[str]) -> CorpusSnapshot:
    """Read the corpus at ``root`` once: its digest, and the specs of the tasks named.

    One walk rather than two, and that is the point rather than an optimization. The specs are
    parsed out of the same bytes the digest is computed from, as they are read, so a caller that
    holds both holds one statement about one corpus at one instant. See :class:`CorpusSnapshot`
    for what the two-observation version let through.

    A named task whose spec the walk never reached is a manifest and a corpus that disagree, which
    is refused here rather than at the two-hundredth episode of a run.

    **The same walk also records what each unit held on its own**, so that a derivation months of
    episodes later can prove the bytes it is about to copy are still the ones this reading saw
    without rehashing the corpus (see :meth:`CorpusSnapshot.verify`). It costs a second hash
    update over bytes already in memory, about a tenth of a second on this corpus, and a
    dictionary of a few hundred entries."""
    digest = hashlib.sha256()
    data = root / "data"
    wanted = set(task_ids)
    specs: Dict[str, Dict[str, Any]] = {}
    units: Dict[str, Any] = {}
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
        relative = path.relative_to(data)
        # `tasks/<task_id>/specs.json` and nothing else: a `specs.json` anywhere else in the tree
        # is not a task's, and a task not asked for is not read into memory.
        keeping = (
            relative.parts[:1] == ("tasks",)
            and len(relative.parts) == 3
            and relative.parts[2] == "specs.json"
            and relative.parts[1] in wanted
        )
        unit = units.setdefault(_unit_of(relative), hashlib.sha256())
        kept = _absorb((digest, unit), str(relative), path, keep=keeping)
        if keeping:
            specs[relative.parts[1]] = json.loads(kept)
    missing = sorted(wanted - set(specs))
    if missing:
        raise ProvisioningError(
            f"the corpus at {root} has no specification for {len(missing)} of the tasks this port "
            f"serves (first: {missing[0]}); the manifest at {MANIFEST} and this corpus are not "
            "describing the same split"
        )
    return CorpusSnapshot(
        digest=digest.hexdigest()[:16],
        specs=specs,
        units={name: unit.hexdigest()[:16] for name, unit in units.items()},
    )


def _unit_of(relative: Path) -> str:
    """Which unit of a corpus a file belongs to: its task, or its top-level entry.

    One task per unit rather than one for the whole task tree, because that is the granularity
    derivation works at: a task is materialised on its first use and the rest of the tree is not
    touched, so a check at task granularity is a millisecond and a check at tree granularity is a
    second and a half."""
    head, *rest = relative.parts
    if head == "tasks" and rest:
        return f"tasks/{rest[0]}"
    return head


def _absorb(
    material: "Sequence[Any]", relative: str, path: Path, *, keep: bool = False
) -> bytes:
    """Fold one file's name and content into every hasher given, and hand back the bytes if asked.

    One definition, used by the walk that reads a whole corpus and by the check that re-reads one
    unit of it. They have to agree byte for byte on what a file contributes or the second would
    report a corpus that never changed as changed, and a check that cries wolf is a check somebody
    turns off."""
    for hasher in material:
        hasher.update(relative.encode())
    kept: List[bytes] = []
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            for hasher in material:
                hasher.update(block)
            if keep:
                kept.append(block)
    return b"".join(kept)


def _unit_digest(data: Path, unit: str) -> Optional[str]:
    """Re-read one unit of a corpus exactly as :func:`corpus_snapshot` read it.

    ``None`` where there is nothing readable to state a digest over: the unit is gone, or
    something in it is a symbolic link, which the whole-corpus walk refuses outright rather than
    hashes. Both are answered by the caller as a mismatch, because a unit that cannot be read is
    not a unit that was proved unchanged."""
    top = data / unit
    try:
        mode = top.lstat().st_mode
    except OSError:
        return None
    if stat.S_ISLNK(mode):
        return None
    material = hashlib.sha256()
    paths = sorted(top.rglob("*")) if stat.S_ISDIR(mode) else [top]
    for path in paths:
        if path.is_symlink():
            return None
        if not path.is_file():
            continue
        try:
            _absorb((material,), str(path.relative_to(data)), path)
        except OSError:
            return None
    return material.hexdigest()[:16]


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


def _generator_sources() -> Tuple[Path, ...]:
    """The files whose bytes decide what a derived corpus holds.

    What a backlog is drawn as (``ledger``), what a derived tree is made of and what the seeded
    rows say (``world``), and how those rows are actually written into a task's database log
    (``worker``)."""
    here = Path(__file__).parent
    return tuple(here / name for name in ("ledger.py", "world.py", "worker.py"))


@lru_cache(maxsize=1)
def _generator_digest() -> str:
    """Eight hex characters over everything that decides what a backlog looks like: the constants
    it is drawn from, and the code that draws it.

    **The implementation as well as its constants, which it was not.** This hashed eleven values
    out of :mod:`~shogym.envs.appworld.ledger` and nothing else, so an edit to *how* a backlog is
    drawn, how a task is materialised or how a seeded row is written left every one of them alone
    and left this digest alone with them. The cache the digest names is the world every episode of
    a task starts in and the baseline it is graded against, so an unchanged key meant a new
    generator comparing itself against rows an old one had seeded, silently and for the life of
    the cache.

    Read off the files rather than declared in a constant somebody has to remember to bump, which
    is the failure this is fixing rather than a variant of it. The price is that an edit anywhere
    in those three files, a comment included, derives the corpus again: a few minutes once, in the
    direction that cannot be wrong. The constants are still hashed on their own, so that moving
    one out of ``ledger.py`` cannot quietly take it out of the key.

    Memoized, because it reads three files and nothing under it can change inside a process."""
    from shogym.envs.appworld import ledger

    material = hashlib.sha256()
    material.update(
        repr(
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
        ).encode()
    )
    for source in _generator_sources():
        material.update(source.name.encode())
        material.update(b"\0")
        material.update(source.read_bytes())
    return material.hexdigest()[:8]


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
    """A scrubbed environment for one worker.

    **``PYTHONDONTWRITEBYTECODE`` is half of what makes the runtime digest's claim true**, and
    :func:`ensure_apps` is the other half. :func:`runtime_digest` skips ``__pycache__``, and the
    usual defence of that is that Python discards a cache whose source has changed. It discards
    one whose recorded source *mtime and size* have changed, which is weaker than it sounds: a
    ``.pyc`` carrying the right two numbers is executed whatever bytes are in it, so a cache
    written or edited under the installed tree would be executable code the run's identity has
    never read. Provisioning therefore compiles the whole interpreter with hash-based caches,
    which the import system validates against the source's own hash rather than against its
    timestamp; this is what stops a *new* cache being written in the default timestamp form
    afterwards, by a worker or by anything else. Between them, every ``.pyc`` that can be
    consulted is one whose source hash still matches a source the digest did read.

    Not the same thing as ``PYTHONPYCACHEPREFIX``, which was the first attempt and which sends
    each worker to a cache directory of its own. That is also correct and costs about three
    seconds of recompilation on every worker start, of which an episode pays two."""
    scrubbed = {
        name: os.environ[name] for name in _ENV_ALLOW_LIST if os.environ.get(name) is not None
    }
    scrubbed["HOME"] = str(scratch)
    scrubbed["APPWORLD_CACHE"] = str(scratch / "appworld-cache")
    scrubbed["PYTHONDONTWRITEBYTECODE"] = "1"
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
        the life of it.

        **Nothing escapes this without an owner.** Only the empty-line branch used to clean up, so
        a handshake that failed any other way left a live worker process, its whole group, and its
        scratch directory behind with nobody holding a reference: a stdin the child never read and
        closed, a first line that is not JSON, a JSON object with no ``port`` in it, a port that is
        not an integer. All of those happen at construction, before there is a task to file a
        failure row against, so the run's own record would not have said either. The process, its
        group and the scratch belong to this method until a ``Worker`` is returned, and are killed,
        reaped and removed on every other exit.

        The whole handshake is under :data:`_SPAWN_TIMEOUT_SECONDS`. The write is not, and does not
        need to be: it is a couple of hundred bytes into an empty pipe, which cannot block."""
        token = secrets.token_urlsafe(32)
        scratch = Path(tempfile.mkdtemp(prefix="shogym-appworld-"))
        process: Optional[subprocess.Popen] = None
        pgid: Optional[int] = None
        published = False
        try:
            process = subprocess.Popen(
                [str(runtime()), str(WORKER), "serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                cwd=str(scratch),
                env=_worker_environment(scratch),
                # Its own process group, so stopping the episode stops everything it started.
                # Agent code runs in this process and is free to spawn; signalling the direct
                # child alone would leave those descendants running against the world after it
                # was scored.
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
                raise WorkerError(
                    "the appworld worker never bound a port "
                    f"(status {process.poll()}, waited {_SPAWN_TIMEOUT_SECONDS:.0f}s)"
                )
            worker = cls(
                root=root,
                process=process,
                port=int(json.loads(line)["port"]),
                token=token,
                scratch=scratch,
                pgid=pgid,
            )
            published = True
            return worker
        finally:
            if not published:
                _abandon(process, pgid, scratch)

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

#: How much of one file is moved between two readings of the clock and the stop flag. A bound that
#: is only checked between files is not a bound on a tree that may hold one enormous file, and a
#: cancelled finalization would wait out the whole of it. A megabyte is small against the sixty
#: seconds and large against the cost of a check.
_SNAPSHOT_CHUNK_BYTES = 1 << 20


class _Bound:
    """The four bounds one snapshot runs under, and the only place any of them is read.

    A class rather than four locals threaded through the walk, because the walk is no longer one
    loop: enumerating a directory, removing the previous copy and moving one file's bytes are
    three operations that each have to be able to say "the budget is gone" from inside themselves.
    The limits are read off the module once, at construction, so a caller that shrinks them for a
    test shrinks them for the whole of the call and not for half of it."""

    def __init__(self, stop: "Optional[threading.Event]") -> None:
        self.stop = stop
        self.began = time.monotonic()
        self.nodes = 0
        self.bytes = 0
        self.max_nodes = _SNAPSHOT_MAX_NODES
        self.max_bytes = _SNAPSHOT_MAX_BYTES
        self.max_depth = _SNAPSHOT_MAX_DEPTH
        self.seconds = _SNAPSHOT_SECONDS

    def alive(self) -> None:
        """The two bounds that are about *when*, checked before anything that can take time.

        Once per directory entry as it arrives and once per chunk of a file, which is what makes
        the deadline and the cancellation bounds on the work rather than on the gaps between
        pieces of it."""
        if self.stop is not None and self.stop.is_set():
            raise SnapshotError("the snapshot was abandoned before it finished")
        if time.monotonic() - self.began > self.seconds:
            raise SnapshotError(
                f"the episode's output tree took longer than {self.seconds:.0f}s to copy"
            )

    def node(self) -> None:
        """Count one directory entry, having first checked that there is still a budget to spend."""
        self.alive()
        self.nodes += 1
        if self.nodes > self.max_nodes:
            raise SnapshotError(
                f"the episode's output tree holds more than {self.max_nodes} entries"
            )

    def descend(self, depth: int) -> None:
        if depth > self.max_depth:
            raise SnapshotError(
                f"the episode's output tree is deeper than {self.max_depth} directories"
            )

    def offer(self, size: int) -> None:
        """Refuse a file whose length alone breaks the budget, before any of it is read."""
        if self.bytes + size > self.max_bytes:
            raise SnapshotError(
                f"the episode's output tree is larger than {self.max_bytes} bytes"
            )

    def spend(self, count: int) -> None:
        """Account for bytes actually moved, so a file that grew past its own ``stat`` is caught."""
        self.bytes += count
        if self.bytes > self.max_bytes:
            raise SnapshotError(
                f"the episode's output tree is larger than {self.max_bytes} bytes"
            )


def _names(source: Path, bound: _Bound) -> List[str]:
    """One directory's entry names, counted against the bound as they arrive, then sorted.

    ``sorted(source.iterdir())`` read and sorted the whole directory before a single bound was
    consulted, so a directory holding a million names spent all of that time and memory *after*
    the deadline had passed and after the finalization had been cancelled. ``os.scandir`` hands
    them over one at a time, so the bound is spent per entry and the enumeration stops inside the
    directory rather than at the end of it.

    The order is still the sorted one, and the sort still happens: it runs on the names that got
    past the bound, which is the difference. A deterministic order is worth keeping, because it
    decides which of several refusals an episode gets told about."""
    names: List[str] = []
    with os.scandir(source) as entries:
        for entry in entries:
            bound.node()
            names.append(entry.name)
    return sorted(names)


def _clear(target: Path, bound: _Bound, depth: int = 0) -> None:
    """Remove ``target`` and everything under it, under the same bound as the copy that follows.

    ``shutil.rmtree`` was outside every bound, and what it removes is the previous snapshot, which
    lives at a name one character away from the output root the world is handed: the process that
    ran the agent's code could work out the name and fill the tree. So a cancelled finalization
    waited out a deletion the episode had sized, before it reached the copy the bounds cover.

    Depth is bounded here as well as in the copy, and for a second reason: this recurses, so a
    tree nested ten thousand deep would otherwise be an interpreter stack rather than a refusal."""
    bound.alive()
    try:
        mode = target.lstat().st_mode
    except FileNotFoundError:
        return
    if stat.S_ISDIR(mode) and not stat.S_ISLNK(mode):
        bound.descend(depth)
        for name in _names(target, bound):
            _clear(target / name, bound, depth + 1)
        target.rmdir()
        return
    # A symlink is unlinked, never followed: what it names is outside the tree being removed.
    target.unlink()


def _copy(source: Path, target: Path, bound: _Bound) -> None:
    """Move one file's bytes, reading the clock and the stop flag between chunks.

    ``shutil.copyfile`` was one call with no way in: a single large or slow file ran to completion
    however long the copy had already taken and however long ago the finalization it belongs to
    was abandoned. Checking once per file bounds a tree of small files and bounds nothing about a
    tree with one big one in it.

    A refused copy leaves a partial file behind in the destination, and that is harmless by
    construction: a snapshot that raises is an episode refused outright, nothing reads the
    destination afterwards, and the next call removes it before it writes anything."""
    with source.open("rb") as reader, target.open("wb") as writer:
        while True:
            bound.alive()
            block = reader.read(_SNAPSHOT_CHUNK_BYTES)
            if not block:
                return
            bound.spend(len(block))
            writer.write(block)


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
    cancelled, and it is read from inside the work rather than between pieces of it.

    **Every operation that can consume the bound is inside it, which three of them were not.** The
    previous snapshot was removed by an unbounded ``rmtree`` before the clock was ever consulted;
    each directory was read and sorted in full before the first check of the entries it produced;
    and one file was copied by a single call that could not be interrupted however large it was.
    So a tree an episode had sized could hold finalization open through any of the three while
    every stated bound stood unbroken. The removal, the enumeration and the copy now each spend
    the same budget as they go (see :class:`_Bound`).

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
    bound = _Bound(stop)
    try:
        _clear(into, bound)
    except OSError as exc:
        # Typed rather than let out raw: a destination that will not go away is this episode
        # ending unscored, and that is a snapshot's own kind of failure rather than a bug.
        raise SnapshotError(
            f"the previous snapshot at {into} could not be removed ({exc}), so this episode "
            "cannot be handed to the grader as the tree it submitted"
        ) from exc
    into.mkdir(parents=True)
    # One pass: what is checked is what is copied. A validate-then-`copytree` would walk the tree
    # twice and bound neither walk.
    pending: List[Tuple[Path, Path, int]] = [(root, into, 0)]
    while pending:
        source, target, depth = pending.pop()
        bound.descend(depth)
        for name in _names(source, bound):
            entry = source / name
            bound.alive()
            if entry.is_symlink():
                raise SnapshotError(
                    f"the episode left a symbolic link in its output tree ({name} -> "
                    f"{os.readlink(entry)}), which a grader must not resolve"
                )
            if entry.is_dir():
                (target / name).mkdir()
                pending.append((entry, target / name, depth + 1))
                continue
            if not entry.is_file():
                raise SnapshotError(f"the episode left {name}, which is not a file or directory")
            # Offered before it is opened, so a file whose length alone breaks the budget costs a
            # `stat` rather than a gigabyte, and accounted for again as it is read, so a file that
            # is longer than it said cannot spend more than the budget either.
            bound.offer(entry.stat().st_size)
            _copy(entry, target / name, bound)
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


def _abandon(process: Optional[subprocess.Popen], pgid: Optional[int], scratch: Path) -> None:
    """Take back everything a worker that never got published was given.

    Killed rather than asked to stop: a handshake that did not complete is a process that never
    said anything, so there is nothing to be polite to and nothing that could be lost. The whole
    group, because the leader may already have started something. Reaped, because an unreaped
    child holds its pid and this is the one moment at which nobody else will ever wait on it. And
    the scratch directory last, which is this worker's ``HOME``, its working directory and its
    bytecode cache."""
    if process is not None:
        _stop(process, signal.SIGKILL, pgid)
        try:
            process.wait(timeout=_CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
    shutil.rmtree(scratch, ignore_errors=True)


#: How much of a worker's first line will be read before it is refused. A handshake is one small
#: JSON object; anything past this is a process saying something else, and a deadline alone would
#: let it say it for the whole of the spawn timeout at whatever rate it liked.
_HANDSHAKE_MAX_BYTES = 1 << 16


def _first_line(process: subprocess.Popen, timeout: float) -> str:
    """The worker's first line of output, or the empty string if it does not arrive in time.

    ``readline`` on a pipe cannot be given a deadline, so the descriptor is waited on instead: a
    worker that dies without printing closes the pipe and is readable immediately, and one that
    hangs on an import is caught by the deadline rather than hanging its caller with it.

    **The whole line is under the deadline, not the first byte of it.** Waiting for readability
    once and then calling ``readline`` bounds only the wait: readability means *some* byte arrived,
    and the blocking read that followed it ran until a newline that a worker which wrote half a
    line and then wedged was never going to send. That is a construction that hangs for good, on
    the path that runs before any task exists to record a timeout against. So the descriptor is
    waited on before every read, what is waited for is the time that is left, and each read takes
    only what is already there.

    Read off the file descriptor rather than through ``process.stdout``. A text stream buffers
    ahead, and a ``select`` on the descriptor underneath a stream holding buffered bytes reports
    nothing to read while the line sits in the buffer, which is the deadlock this is meant to
    remove rather than a version of it. Nothing else reads this descriptor before the handshake."""
    assert process.stdout is not None
    handle = process.stdout.fileno()
    deadline = time.monotonic() + timeout
    chunks: List[bytes] = []
    seen = 0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ""
        ready, _, _ = select.select([handle], [], [], remaining)
        if not ready:
            return ""
        try:
            block = os.read(handle, 4096)
        except OSError:
            return ""
        if not block:
            # The worker closed its pipe without finishing a line, which is a worker that died.
            return ""
        chunks.append(block)
        seen += len(block)
        if b"\n" in block:
            break
        if seen > _HANDSHAKE_MAX_BYTES:
            return ""
    return b"".join(chunks).decode(errors="replace").split("\n", 1)[0] + "\n"


__all__ = [
    "CorpusSnapshot",
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
    "corpus_snapshot",
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
