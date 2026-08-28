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

import ast
import calendar
import fcntl
import hashlib
import json
import os
import platform
import re
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
    already happened by then, and a directory left behind costs disk rather than correctness.

    **A publish that fails puts the incumbent back.** Two renames are two operations, and the
    displaced tree used to be removed whatever had happened between them: a failure at the second
    left the canonical name absent and the only remaining interpreter under a ``.replaced`` name
    nothing ever looks for. Injected once, the probe found no runtime at all afterwards. That is
    worse than a failed build, because a worker already serving out of the old tree keeps its open
    files but resolves every new import through the name that has just disappeared. So the
    incumbent goes back, and if the restore itself fails the displaced copy is *retained* under
    its own name rather than removed, because it is then the only copy of the interpreter there
    is. Either way the caller is told the publish did not happen: :func:`runtime` returns a path
    to an interpreter, and returning one that was never published would hand every world a name
    with nothing under it. The same rule, for the same reason, as
    :func:`~shogym.envs.appworld.world._publish`."""
    displaced = home.with_name(f"{home.name}.replaced.{os.getpid()}.{secrets.token_hex(8)}")
    published = False
    # Whether the incumbent is sitting under `displaced` and is the only copy of it there is.
    aside = False
    try:
        if home.exists():
            os.replace(home, displaced)
            aside = True
        os.replace(staging, home)
        published = True
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
        if aside:
            try:
                os.replace(displaced, home)
            except OSError:
                # Left where it is. The `finally` below removes a displaced tree only after a
                # publish that worked, so this one survives this call and can be found by name.
                pass
        raise
    finally:
        if published and aside:
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

    **Two things, and neither of them is the identity.** A default ``.pyc`` records the source's
    modification time and size and is honoured whenever those two numbers still match, whatever
    bytes the cache itself holds; a hash-based one records a hash of the source and the import
    system checks it, so a cache whose source was edited beside it is discarded and the source is
    what runs. That closes the stale-cache case and not the changed-payload one, because the
    recorded hash covers the source and not the marshalled code, which is why
    :func:`runtime_digest` reads these bytes rather than reasoning about them.

    What this buys is the other half: the whole tree is compiled once, here, so no later import
    has a cache to write, and the identity computed after this call is the identity every worker
    afterwards runs under.

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

    **What is hashed, exactly: every byte of ``site-packages`` an import can reach, plus the base
    interpreter's own binary.** Every regular file under every ``site-packages`` of the virtual
    environment, by relative path and by content, ``__pycache__`` included; every symbolic link
    there by its target text, by where that text resolves to and by the resolved file's bytes;
    ``pyvenv.cfg``; the platform; the pins; and the executable ``bin/python`` resolves to, which is
    the real interpreter the venv borrows and which nothing else here would have covered.

    **The bytecode caches are in, and used not to be.** The argument for leaving them out was that
    provisioning rewrites every cache as a hash-based one, which the import system validates
    against the source's own hash, so a ``.pyc`` stood for a source this function had read. That
    validation compares the *source hash recorded in the cache header* with the source; it does
    not bind the marshalled payload to those bytes. So a cache whose header still matched and
    whose payload had been changed was executable code this digest had never read, and the
    identity said nothing. Reading the caches costs a quarter of a second on this runtime, against
    the second and a half per worker start that deleting them and re-parsing every import would
    cost instead, so they are read.

    That is stable rather than lucky, because nothing writes one after provisioning:
    :func:`_compile_runtime` compiles the whole tree once with ``-f``, every process this port
    starts under that interpreter afterwards carries ``PYTHONDONTWRITEBYTECODE`` (see
    :func:`_worker_environment`), and a valid checked-hash cache is not rewritten by the import
    that reads it. A cache appearing or changing under this tree is therefore a change to the code
    the worker runs, which is exactly what this digest is for.

    **What is not hashed, and why it is out of scope.** The base interpreter's *standard library*
    is not read: it is thousands of files belonging to the host's Python rather than to anything
    this port installs, and the base binary's own bytes already move with a reinstalled or
    repointed interpreter. A digest that named that would be a longer walk and a claim this code
    cannot keep, so the claim is the narrower one and it is true.

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
    """Every file under ``packages``, bytecode caches included, in one order on every machine.

    Sorted at each level rather than by collecting and sorting the whole tree, so a digest over a
    hundred thousand files still holds one directory's names at a time.

    A symlinked *directory* is yielded like any other link and never descended into, so it
    contributes where it points and not a second copy of whatever is over there. Descending would
    also be how one link makes this walk unbounded."""
    for directory, children, names in os.walk(packages, followlinks=False):
        here = Path(directory)
        linked = [child for child in children if (here / child).is_symlink()]
        children[:] = sorted(child for child in children if child not in linked)
        for name in sorted([*names, *linked]):
            path = here / name
            yield str(path.relative_to(packages)), path


# ----- the derived corpus -----


#: Bumped when the shape of a derived tree changes: what is copied, what is linked, what is
#: sealed, and what a task carries to say it is whole. It is part of a cache's name and of its
#: stamp, so a tree built under an older layout is a different cache rather than one this code
#: will read as its own. At 2 because a derived task now carries a manifest of everything in it
#: (see :func:`~world._write_manifest`), and a tree from before it has nothing to be checked
#: against.
DERIVATION_VERSION = 2

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


#: The modules that generate a world, from which the rest of the generator is reached. Between
#: them they hold every decision a derived corpus is made of: ``env_v1`` draws the backlog and
#: names the seeds it and the episode's world are started from, ``world`` decides what a derived
#: tree is made of and what the seeded rows say, and ``worker`` writes those rows into a task's
#: database log through upstream's own model layer.
_GENERATOR_ENTRY_POINTS: Tuple[str, ...] = (
    "shogym.envs.appworld.env_v1",
    "shogym.envs.appworld.world",
    "shogym.envs.appworld.worker",
)

#: How far the walk from those entry points goes. This port's own package, and the one module
#: outside it that derivation calls into. The boundary is deliberate rather than incidental: past
#: it is shogym's own machinery (the serve layer, the core env, the MCP transport), which decides
#: how an episode is dispensed and recorded and decides nothing about what bytes end up in a
#: derived tree. Naming the whole distribution here would put the entire package's source in the
#: name of a 134 MB cache, so an unrelated edit to the serve layer would re-derive the corpus.
_GENERATOR_ROOTS: Tuple[str, ...] = ("shogym.envs.appworld", "shogym.envs._upstream")

#: The installed root of the ``shogym`` package, which is what a dotted module name is resolved
#: against. Read off this module's own location rather than off ``sys.modules``, so the walk is a
#: file operation and needs nothing imported.
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=1)
def _generator_sources() -> Tuple[Tuple[str, Path], ...]:
    """Every module whose bytes decide what a derived corpus holds, as (name, file) pairs.

    **Walked from the entry points rather than listed by hand, because a hand-kept list is what
    failed.** This was three names in a tuple: ``ledger.py``, ``world.py`` and ``worker.py``. The
    two helpers that decide a task's seeded backlog and the seed its live world is started from
    live in ``env_v1``, which was not one of them, so an edit to either produced the same cache
    name and the same run fingerprint over a different world. A list somebody has to remember to
    extend is a list that is right until the next helper is written somewhere else.

    So the closure is computed: parse each entry point, take every module it imports that falls
    inside :data:`_GENERATOR_ROOTS`, and repeat. ``ast.walk`` rather than the module's top level,
    because the imports this port defers into function bodies (the ledger inside
    :func:`_generator_digest`, ``mcp_server`` inside a session's setup) are imports all the same.
    Static rather than through ``sys.modules``: what is hashed then is what is on disk under the
    names the code actually writes, whether or not this process has imported it, and there is no
    ordering in which a module can be missed for having been imported later.

    **What the boundary costs.** The closure is wider than the three files it replaces, so an edit
    anywhere in this port derives the corpus again, a comment edit included. That is a few minutes
    once, in the direction that cannot be wrong, and it is a real cost to
    shojin-lab/shogym#140, which rewrites four of these modules and will re-derive on the first
    construction after it lands. The alternative on offer was the hand-kept list, which is cheap
    right up to the edit it does not notice.

    Memoized, because it parses ten files and none of them can change inside a process."""
    found: Dict[str, Path] = {}
    pending = list(_GENERATOR_ENTRY_POINTS)
    while pending:
        name = pending.pop()
        if name in found:
            continue
        source = _module_file(name)
        if source is None:
            continue
        found[name] = source
        pending.extend(_imported_modules(name, source))
    return tuple(sorted(found.items()))


def _module_file(name: str) -> Optional[Path]:
    """Where a dotted module inside :data:`_GENERATOR_ROOTS` lives, or ``None`` for anything else.

    A package contributes its ``__init__`` and nothing else; its submodules arrive on their own
    names, through the imports that actually reach them, so importing a package does not drag in
    every file beside it."""
    if not any(name == root or name.startswith(root + ".") for root in _GENERATOR_ROOTS):
        return None
    within = _PACKAGE_ROOT.joinpath(*name.split(".")[1:])
    module = within.with_suffix(".py")
    if module.is_file():
        return module
    package = within / "__init__.py"
    return package if package.is_file() else None


def _imported_modules(name: str, source: Path) -> "Iterator[str]":
    """Every dotted name ``source`` imports, whether or not it resolves to anything.

    ``from x import y`` yields both ``x`` and ``x.y``: the name after ``import`` is a submodule or
    an attribute and the syntax does not say which, so both are offered and :func:`_module_file`
    keeps whichever is a file. Relative imports are resolved against the module's own package;
    this port writes none today, and a walk that silently skipped them would be the hand-kept
    list's failure wearing a different hat."""
    tree = ast.parse(source.read_bytes(), filename=str(source))
    package = name if source.name == "__init__.py" else name.rpartition(".")[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                anchor = ".".join(parts[: len(parts) - node.level + 1])
                base = f"{anchor}.{base}" if base else anchor
            if not base:
                continue
            yield base
            for alias in node.names:
                yield f"{base}.{alias.name}"


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
    is the failure this is fixing rather than a variant of it. The constants are still hashed on
    their own, so that moving one out of ``ledger.py`` cannot quietly take it out of the key.

    **And off every file the generator reaches, rather than three named ones.** The three were
    ``ledger``, ``world`` and ``worker``, and the helpers that decide a task's seeded backlog and
    the seed its world is started from are in ``env_v1``, which was not among them: an
    implementation change to either reused a cache and a run identity claiming the generator
    before it. What is hashed is now the import closure of the modules that generate a world (see
    :func:`_generator_sources`), by dotted name and by source bytes, in one order on every
    machine. The price is that an edit anywhere in this port, a comment included, derives the
    corpus again: a few minutes once, in the direction that cannot be wrong.

    Memoized, because it reads ten files and none of them can change inside a process."""
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
    for name, source in _generator_sources():
        material.update(name.encode())
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

#: What every worker's scratch directory is named for. It is also what a sweep checks before
#: removing one whose name it read out of a file: the ledger is a file, and a file can be edited.
_SCRATCH_PREFIX = "shogym-appworld-"

#: A grader's scratch directory, which is a worker's with a word added. It carries the prefix
#: above because a grader is written into the same ledger as a serving worker and is reclaimed by
#: the same sweep, which will only remove a directory whose name it recognises (see
#: :func:`_clear_scratch`).
_GRADE_SCRATCH_PREFIX = _SCRATCH_PREFIX + "grade-"


def _close_descriptor(descriptor: Optional[int]) -> None:
    """Close a raw descriptor, and treat one that is already closed as closed."""
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _worker_environment(scratch: Path) -> Dict[str, str]:
    """A scrubbed environment for one worker.

    **``PYTHONDONTWRITEBYTECODE`` is what keeps the runtime's identity a constant of the run.**
    :func:`runtime_digest` reads every ``.pyc`` under the installed tree, because a cache is
    executable input and a checked-hash header binds the source and not the payload. That digest
    names the derived cache, so a cache file appearing or changing mid-run would rename the world
    every later episode is served from. Nothing this port starts under that interpreter writes
    one: provisioning compiles the tree once (:func:`_compile_runtime`) and every process after it
    carries this variable, so the only way those bytes move is somebody changing what the worker
    executes, which is the fact the digest exists to carry.

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


# ----- workers whose parent is gone -----

#: Where the workers this port has started are written down, one event per line. A file rather
#: than anything held in memory, because the case it exists for is a serving process that is no
#: longer there to hold anything: what a later construction reads has to have outlived the process
#: that wrote it.
_WORKER_LEDGER = "workers.txt"

#: What one sweep holds while it decides. Its own file rather than the ledger's, because the two
#: locks bound different things: the ledger's is taken for one short append or one rewrite, and
#: this one is held across a ``ps`` per record and the signals that follow. Taking the ledger's
#: for the whole of a sweep would block every sibling's spawn on this machine behind it.
_REAP_LOCK = "workers.reaping"

#: When the ledger is rewritten rather than appended to. Two lines per worker that started and
#: stopped, so a file only ever appended to is a file every later construction reads in full.
_LEDGER_MAX_LINES = 4096

#: What one sweep spends before it leaves the rest for the next one. Deciding whether an owner is
#: still alive can cost a ``ps`` per record, and this runs inside a constructor a serve layer may
#: call while it is dispensing. The work is idempotent and the leftovers do not spoil, so stopping
#: early costs nothing but a second pass.
_REAP_SECONDS = 10.0
_REAP_MAX_WORKERS = 16


def _ledger() -> Path:
    """The ledger's path, made if this is the first worker on this machine."""
    home = cache_root()
    home.mkdir(parents=True, exist_ok=True)
    return home / _WORKER_LEDGER


def _reap_lock() -> Path:
    """The file one sweep locks, made if this is the first construction on this machine."""
    home = cache_root()
    home.mkdir(parents=True, exist_ok=True)
    return home / _REAP_LOCK


def _append(line: str) -> None:
    """Add one event, under the lock a rewrite takes, and never raise.

    One ``O_APPEND`` write of one short line, which the kernel does not interleave with another
    process's. The lock is not what makes that atomic; it is what keeps this out of the middle of
    a compaction, which is a read and a write with a gap in it (see :func:`_compact`)."""
    try:
        with open(_ledger(), "a") as handle:
            _exclusive(handle)
            handle.write(line + "\n")
    except OSError:
        pass


def _exclusive(handle: Any) -> bool:
    """Take the ledger's own lock, or say that this filesystem does not have one.

    Blocking, because everyone who takes it holds it for one small write and the honest reading of
    "somebody else has it" is that they are a line ahead of this one."""
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except OSError:
        # Including the errnos that mean "this mount has no locks". The appends stay correct
        # without it, because each is one `O_APPEND` write; what is not attempted without it is
        # the rewrite, which is the operation that needs the exclusion.
        return False
    return True


def record_worker(**fields: Any) -> str:
    """Write down one worker this process has started, and return the name it was written under.

    Appended before the handshake rather than after it, so a parent that dies during a spawn has
    still said what it started. The record is what a later construction has instead of the
    ``Worker`` object: who owns it (:func:`process_birth` beside the owner's pid, so a recycled
    number is not read as a live owner), which boot it belongs to, what to signal, and what to
    remove.

    Never raises. A worker that could not be written down is a worker whose parent-death pipe
    still stops it; the ledger is what lets a later run *find* it, not what stops it."""
    name = f"{os.getpid()}-{secrets.token_hex(8)}"
    _append(
        "+"
        + json.dumps(
            {
                "name": name,
                "parent": os.getpid(),
                "birth": _own_birth(),
                "boot": _boot_id(),
                **fields,
            },
            sort_keys=True,
        )
    )
    return name


def forget_worker(name: str) -> None:
    """Tombstone a worker this process has stopped."""
    _append(f"-{name}")


def outstanding() -> List[Dict[str, Any]]:
    """The workers written down and not yet tombstoned, in the order they were first written."""
    try:
        lines = _ledger().read_text().splitlines()
    except OSError:
        return []
    return _live(lines)


def _live(lines: Sequence[str]) -> List[Dict[str, Any]]:
    """The records a ledger's events leave outstanding, in the order they were first written.

    Every line is an event, ``+`` for a worker to account for and ``-`` for one confirmed stopped,
    and what is live is what the events say rather than what any writer last decided. A line that
    will not parse is a line from a version of this code or a torn write, and is skipped rather
    than treated as a worker."""
    live: Dict[str, Dict[str, Any]] = {}
    gone: set = set()
    for line in lines:
        line = line.strip()
        if line.startswith("-"):
            gone.add(line[1:])
        elif line.startswith("+"):
            try:
                record = json.loads(line[1:])
            except ValueError:
                continue
            name = str(record.get("name", ""))
            if name and name not in live:
                live[name] = record
    return [record for name, record in live.items() if name not in gone]


def _spent(began: float, reclaimed: Sequence[str]) -> bool:
    """Whether this sweep has used the budget one call gets."""
    return len(reclaimed) >= _REAP_MAX_WORKERS or time.monotonic() - began > _REAP_SECONDS


def reap(*, alive: Optional[Any] = None) -> List[str]:
    """Stop and clear away this port's workers whose owner is gone, and say which.

    **The case teardown cannot reach.** A worker is started in a session of its own so that
    stopping an episode stops everything the episode spawned, and only the parent's own ``Worker``
    object held its port, its token, its process handle and its group. A serving process that died
    abruptly took all four with it: the worker was reparented and went on running, and a resumed
    harness could neither adopt it nor name it for teardown and simply started another. What is
    left is a process nobody is going to stop and nothing names.

    So every worker is written down with the pid and the birth of the process that started it, and
    this runs at construction: a record whose owner is not running is a worker nobody is coming
    back for. The birth is beside the pid because pids are reused, and the boot id is beside both
    because they are reused across a restart; stopping a live episode's world because an unrelated
    process now holds its owner's number would be worse than the failure this fixes.

    **Bounded in aggregate**, because deciding an owner is dead can cost a ``ps`` per record and
    this is called from a constructor a serve layer may run while it is dispensing. Nothing is lost
    by stopping early: what is left is still in the ledger, and the next construction starts again
    from the front of it.

    This is the second defence and not the first. The worker holds the read end of a pipe from its
    parent and stops itself when that closes (see :func:`~worker.watch_parent`), so in the ordinary
    crash the process is already gone by the time anything reads this file and what is reclaimed
    here is the scratch directory and the record. This is what covers a worker that was stopped
    before it armed the pipe, or one whose group outlived it.

    **One sweep at a time on this machine, and that is what makes a reclamation a decision rather
    than a race.** Reading the record, checking the leader's birth and signalling its group are
    three steps, and the ledger's own lock covers none of that: it is taken for one append and
    released. Two constructions therefore used to snapshot the same outstanding record, both pass
    the birth check, and both signal; a synchronized probe caught exactly that, two kills aimed at
    one record. The second of them is the dangerous one, because by then the first has killed the
    leader and the number it is aiming at may already have been handed to somebody else. So a
    sweep takes an exclusive lock of its own first and reads the ledger under it, which makes the
    check, the signal and the tombstone one owner's from end to end.

    Taken without waiting: a second construction that finds the lock held has nothing to add,
    because the sweep already running will read every record it would have read. It returns having
    reclaimed nothing, which is what this call already does when it runs out of budget.

    A filesystem with no locks gets no sweep at all. Every other operation on this ledger is a
    single append that survives without exclusion, but this one sends signals, and a signal that
    may be duplicated onto a recycled group number is the failure this whole file exists to
    prevent. The parent-death pipe is still the first defence there, and the records keep."""
    try:
        handle = open(_reap_lock(), "a")
    except OSError:
        return []
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Somebody else is sweeping this machine right now.
            return []
        except OSError:
            # Including the errnos that mean "this mount has no locks". See above.
            return []
        return _sweep(alive=alive)
    finally:
        handle.close()


def _sweep(*, alive: Optional[Any] = None) -> List[str]:
    """One pass over the outstanding records, under the lock :func:`reap` holds for the whole of
    it. Split out so that every path back to the caller releases that lock."""
    began = time.monotonic()
    running = alive if alive is not None else _process_is_alive
    reclaimed: List[str] = []
    for record in outstanding():
        if _spent(began, reclaimed):
            break
        name = str(record.get("name", ""))
        if record.get("boot") != _boot_id():
            # From before this machine last restarted. Every number in it belongs to somebody
            # else now, so nothing is signalled; the record goes, and so does whatever scratch
            # directory survived the reboot.
            _clear_scratch(record)
            forget_worker(name)
            reclaimed.append(name)
            continue
        owner = str(record.get("parent", ""))
        # An unreadable record is left alone. This stops processes, so the ambiguous case has to
        # be the one where nothing happens.
        if not owner.isdigit() or running(int(owner), str(record.get("birth", ""))):
            continue
        if not _stop_orphan(record):
            # Nothing was stopped and nothing was seen to be gone, so this record is still the
            # only durable trace of a worker that may be running. It used to be tombstoned here
            # anyway, along with its scratch directory: an entry with a live worker pid and an
            # unreadable birth was reported reclaimed by a sweep that had signalled nothing, which
            # leaves a world serving with nothing left naming it: exactly the failure this whole
            # file exists to close. Left where it is, for the next construction to try again.
            continue
        _clear_scratch(record)
        forget_worker(name)
        reclaimed.append(name)
    _compact()
    return reclaimed


def _stop_orphan(record: Dict[str, Any]) -> bool:
    """Stop an abandoned worker's group, and say whether the worker was positively dealt with.

    The same rule :meth:`Worker.close` follows, for the same reason and with less to go on: a pgid
    is a number, and a number is that worker's only while the process holding it exists. The
    leader's own birth was read at spawn and written down beside its pid, so this can ask whether
    the number still names the process it was written for. A leader that is gone leaves a number
    nothing here may signal, and its descendants are then beyond reach: that is the honest
    boundary rather than a signal aimed at whoever holds the number now.

    A birth that was never readable is treated the same way. Elsewhere an unknown birth reads as
    "says nothing" and the pid is believed, which is the safe direction for a question about
    whether to *leave something alone*; here the answer decides whether to send a signal, so
    unknown has to mean no.

    **The answer is what the caller tombstones on, and that is the change.** This used to return
    nothing whatever it had done, so every one of the silent paths below reported the same thing
    to :func:`reap` as a delivered kill: an entry with a live worker pid and a birth the process
    table would not answer for had its scratch removed and its ledger line tombstoned without a
    signal being sent, which is a running world with no durable record of itself left. Only two
    outcomes are true here now, and both of them are positive:

    * the group was addressed. The signal went out, or there was no group left to take it, which
      is the same fact about this worker's execution domain.
    * the worker was observed gone. Its pid names nothing at all, or it names a process born at
      some other time, and in either case the process this record was written for has exited.

    Everything else is ambiguity: a record this cannot read, a birth nobody wrote down, a process
    table that will not answer, a pid this uid may not query. Ambiguity keeps the record and the
    scratch rather than spending them."""
    pid, pgid = record.get("pid"), record.get("pgid")
    birth = str(record.get("pid_birth") or "")
    if not isinstance(pid, int) or not isinstance(pgid, int) or pid != pgid:
        return False
    if not birth:
        return False
    now = process_birth(pid)
    if now == birth:
        return _signal_group(pgid, signal.SIGKILL)
    if now:
        # A different process is wearing this number, so the worker that was written down here has
        # exited. Nothing of its own is left to signal, and the number is not this port's to aim
        # at.
        return True
    # The table said nothing, which is not the same as saying the process is gone. One question
    # left that does not need it: whether the number names anything at all.
    return _pid_is_absent(pid)


def _pid_is_absent(pid: int) -> bool:
    """Whether ``pid`` names no process at all, as far as this process is allowed to tell.

    The narrow half of :func:`_process_is_alive`, and it answers the opposite way where it cannot
    tell. A number this uid may not signal belongs to somebody else and is certainly not absent,
    and an error that is neither is not an observation of anything."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False
    return False


def _clear_scratch(record: Dict[str, Any]) -> None:
    """Remove an abandoned worker's scratch directory, which was its ``HOME``.

    Only one this port made. The name is read out of a file, and a file is a thing that can be
    edited, so what is removed has to carry the prefix :func:`Worker.spawn` gives every scratch
    directory. That is not a boundary against somebody who writes this file on purpose, and it is
    not meant to be: it is what keeps a mangled or truncated line from naming a directory this
    would then delete."""
    scratch = record.get("scratch")
    if not isinstance(scratch, str) or not scratch:
        return
    path = Path(scratch)
    if not path.name.startswith(_SCRATCH_PREFIX):
        return
    shutil.rmtree(path, ignore_errors=True)


def _compact() -> None:
    """Rewrite the ledger as just the records still outstanding, once it has grown enough to matter.

    **In place and under the file's own lock, because a rewrite is the one operation here that can
    lose a record.** Everything else is a single ``O_APPEND`` write, which the kernel will not
    interleave; a rewrite is a read and a write with a gap in the middle, and a record appended by
    a live sibling in that gap is a worker nobody would ever come back for. A rename would swap the
    inode the lock is on, so the file is truncated and rewritten rather than replaced.

    Never raises, and does nothing at all where the filesystem cannot lock: a ledger that is merely
    long is a file this port reads a little more of."""
    try:
        with open(_ledger(), "r+") as handle:
            if not _exclusive(handle):
                return
            lines = handle.read().splitlines()
            if len(lines) <= _LEDGER_MAX_LINES:
                return
            keeping = _live(lines)
            handle.seek(0)
            handle.write("".join("+" + json.dumps(r, sort_keys=True) + "\n" for r in keeping))
            handle.truncate()
    except OSError:
        pass


def _process_is_alive(pid: int, birth: str = "") -> bool:
    """Whether ``pid`` is the same live process that was born at ``birth``.

    **This process's own number is not a free pass.** The shortcut used to return ``True`` for it
    without reading the birth beside it, which throws away the very fact that field carries: a
    record written by an earlier holder of this number names an owner that has exited, and reading
    it as "the owner is me, so it is alive" leaves that worker unreclaimable for as long as this
    process runs. So the birth is compared here too, and only an unreadable one on either side
    reads as "says nothing"."""
    if pid == os.getpid():
        mine = _own_birth()
        return not birth or not mine or birth == mine
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Somebody else's process, which is somebody else's business and certainly alive.
        return True
    except OSError:
        return True
    if not birth:
        return True
    now = process_birth(pid)
    # A recorded birth that no longer matches is a recycled number, and the process that owned
    # this worker is gone. An unreadable birth now says nothing, so it says nothing.
    return not now or now == birth


@lru_cache(maxsize=1)
def _own_birth() -> str:
    """When this process started. Memoized: it is one ``ps`` and it cannot change.

    **It cannot change, but the process can.** A fork does not start a new interpreter: the child
    inherits every cached value the parent had warmed, and one of them says when the *parent* was
    born. A child that warmed it in its parent then wrote that birth into the ledger beside its
    own pid, and a sibling construction reading that record saw a live worker pid under a birth
    that did not match it: an active child's worker, offered up as an orphan to reclaim. A fork
    probe on this code confirmed the child answering with the parent's value rather than with its
    own. So the cache is cleared in the child at every fork, below, which is the one moment the
    answer stops being true."""
    return process_birth(os.getpid())


# The child of a fork inherits the parent's cache and none of its identity. Registered at import
# because a fork can happen at any moment after it, including inside a library this port never
# calls, and there is no later point that is reliably before the first read (see `_own_birth`).
os.register_at_fork(after_in_child=_own_birth.cache_clear)


def process_birth(pid: int) -> str:
    """When ``pid`` started, as the process table reports it, or the empty string if unknown.

    A pid is reused, and within one boot it is reused quickly. ``kill(pid, 0)`` answers "is
    something running under that number", and the question a sweep is asking is "is the process
    that started this worker still running". The start time is what separates them: a number that
    came back with a different birth is a different process wearing the same badge.

    **A number, not a rendering.** ``ps`` blank-pads a single-digit day and renders the time in the
    caller's zone, so the same live process read under two locales or two ``TZ`` values prints two
    different strings, and an owner would then read as replaced. The environment is pinned and the
    result converted to epoch seconds, which has no spacing to lose and compares exactly.

    One second of precision is what ``ps`` offers, so a pid reused inside the same second is still
    indistinguishable. That is the residual, and it is narrower than the pid alone was.

    Unknown is the empty string rather than an error, and two empty strings compare equal, which
    keeps a sweep on the safe side of its own rule: it stops only what it can positively tell is
    abandoned."""
    try:
        finished = subprocess.run(
            ["ps", "-o", "lstart=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=_CLOSE_SECONDS,
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
        # string, which keeps a sweep on the safe side of its own rule.
        return ""


@lru_cache(maxsize=1)
def _boot_id() -> str:
    """Something that changes when the machine restarts, so a reused pid is not a live owner.

    **The number, not the rendering.** ``sysctl -n kern.boottime`` prints a struct and then a human
    date, and the date is rendered in the caller's zone: hashing the whole line gives two boot
    identities for one boot under two ``TZ`` values, which would hide an orphan from a sweep run in
    another zone. The seconds field inside the struct is the kernel's own value and does not move,
    so that is what is taken. Linux's boot id is already a value rather than a rendering."""
    try:
        stamp = subprocess.run(
            ["sysctl", "-n", "kern.boottime"],
            capture_output=True,
            text=True,
            timeout=_CLOSE_SECONDS,
            env={**os.environ, "TZ": "UTC", "LC_ALL": "C", "LANG": "C"},
        )
        if stamp.returncode == 0 and stamp.stdout.strip():
            seconds = re.search(r"sec\s*=\s*(\d+)", stamp.stdout)
            if seconds:
                return seconds.group(1)
            return hashlib.sha256(stamp.stdout.strip().encode()).hexdigest()[:12]
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return hashlib.sha256(Path("/proc/sys/kernel/random/boot_id").read_bytes()).hexdigest()[:12]
    except OSError:
        return "unknown"


@dataclass
class Worker:
    """A handle on one episode's world, running in a process of its own.

    The port is the process's and the token is this object's, and neither is handed over on any of
    the standard surfaces: not on the worker's command line, not in its environment, not in the
    instructions the env publishes, not in a tool's schema, and not in a tool's result. That is
    the whole of the claim, and it is what the tests exercise.

    **It is not a secret from the code the agent writes**, and nothing here should be built as
    though it were. That code runs as the worker process, and the token is that process's own
    handler state; no arrangement inside one interpreter puts a value beyond the code running in
    it. What the token holds is the boundary it can hold: another process on this machine cannot
    drive this world without it (see :data:`~worker.TOKEN_HEADER`).

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
    #: The writing end of the pipe whose end-of-file tells the worker this process is gone. Held
    #: open for the worker's whole life and closed by nothing but :meth:`close` and the death of
    #: this process, which is the event it exists to signal (see :func:`~worker.watch_parent`).
    keepalive: Optional[int] = None
    #: What this worker is written down as in the durable ledger, so a later construction can tell
    #: an abandoned worker from a live one (see :func:`reap`).
    record: Optional[str] = None
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
        need to be: it is a couple of hundred bytes into an empty pipe, which cannot block.

        **The worker is given a way to notice this process dying, and is written down so a later
        one can find it.** Stdin is closed after the handshake, so nothing on it says anything
        afterwards; the keep-alive pipe is a second descriptor, held open for the worker's whole
        life and closed by the kernel when this process exits however it exits. Its number is sent
        in the handshake rather than fixed by convention, because ``pass_fds`` does not renumber.
        The ledger record beside it is what a construction after a crash reads (see :func:`reap`),
        and it is written before the handshake so that a parent which dies inside a spawn has still
        said what it started."""
        token = secrets.token_urlsafe(32)
        scratch = Path(tempfile.mkdtemp(prefix=_SCRATCH_PREFIX))
        process: Optional[subprocess.Popen] = None
        pgid: Optional[int] = None
        record: Optional[str] = None
        # Non-inheritable by default, so no other child of this process holds the writing end open
        # and reports this one alive after it has gone.
        listening, holding = os.pipe()
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
                pass_fds=(listening,),
            )
            # Read here and kept, rather than resolved from the pid later: see `pgid`.
            pgid = _group_of(process)
            record = record_worker(
                pid=process.pid,
                pid_birth=process_birth(process.pid),
                pgid=pgid,
                scratch=str(scratch),
            )
            assert process.stdin is not None
            process.stdin.write(
                json.dumps({"root": str(root), "token": token, "keepalive": listening}) + "\n"
            )
            process.stdin.flush()
            process.stdin.close()
            assert process.stdout is not None
            line = _first_line(process, _SPAWN_TIMEOUT_SECONDS)
            if not line:
                # The group is stopped first and the status read after it, because ``poll`` reaps
                # an exited child and a reaped leader's group number is the kernel's to hand on.
                # Read for the diagnostic before anything had stopped the group, it left the
                # cleanup below signalling a number that might already have been somebody else's,
                # which is the stale-group ordering :meth:`_stop_the_group` refuses. A worker that
                # closed its pipe without printing is also a worker that may have started
                # something first, and this is what stops that too.
                _stop(process, signal.SIGKILL, pgid)
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
                keepalive=holding,
                record=record,
            )
            published = True
            return worker
        finally:
            # The child has its own copy; a reading end still open here would be a pipe that never
            # reaches end-of-file for the worker either.
            _close_descriptor(listening)
            if not published:
                _close_descriptor(holding)
                _abandon(process, pgid, scratch)
                if record is not None:
                    forget_worker(record)

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
        strength of it is graded on a tree something may still be writing to.

        **What is spent, and only once the group is positively gone.** The ledger record and the
        scratch directory are the only handles a later construction has on this worker, and they
        used to be spent on every close whatever the stop had reported: a group that could not be
        confirmed empty had its record tombstoned and its scratch removed anyway, so a descendant
        the signal did not reach became a process nothing named. Finalization refuses the score in
        that case, which is right and is not enough on its own, because teardown then removes the
        trees while the untracked descendant may still be writing to them. An ambiguous stop keeps
        both now, which is the rule :func:`_stop_orphan` and :func:`reap` already follow: a record
        kept costs one line and one directory, and a record spent costs the only way back to a
        running world."""
        if not self.closed:
            self.closed = True
            self.stopped = self._stop_the_group()
            # After the stop, not before it. The ordinary teardown is the signal and the process
            # table; the pipe is the crash path, and closing it first would put a second killer
            # in the middle of the one that reports.
            _close_descriptor(self.keepalive)
            self.keepalive = None
            for stream in (self.process.stdout, self.process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
            if self.stopped:
                shutil.rmtree(self.scratch, ignore_errors=True)
                if self.record is not None:
                    forget_worker(self.record)
                    self.record = None
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
    two observations that happened to agree.

    **A session of its own, and the same two lifelines the serving worker has.** This used to be
    an ordinary child with captured pipes, which made the advertised bound a bound on nothing: the
    timeout killed the leader alone and then read the pipes again with no deadline, so a
    descendant holding either of them kept a sealed episode's terminal open indefinitely. It ran
    for twenty seconds under a one-second bound in a probe, with the descendant still running
    afterwards. The group is what is signalled now, its emptying is what is waited for, and every
    wait on the way down has a bound of its own (see :func:`_end_grader`).

    The other half is the crash: this process is short-lived but it is not instant, and a serving
    parent that dies inside those ten minutes would have left it running under init with nothing
    naming it. So it holds the reading end of a pipe from here and stops its own group when that
    reaches end of file (see :func:`~worker.watch_parent`), and it is written into the same
    durable ledger the serving workers are, so a later construction reclaims it (see
    :func:`reap`).

    **The group is read on the way out of every exit, and not only the timeout's.** An ordinary
    exit used to be taken as the end of everything this grader had started: the scratch directory
    went and the ledger line was tombstoned without anything asking what was left in the group. A
    descendant that no longer holds the captured pipes lets ``communicate`` return at once, so a
    leader that exited 0 with such a child behind it left an untracked process and no record of
    it, which a probe confirmed. What the group says decides now (see :func:`_grader_stopped`),
    and an answer that is not "empty" keeps both handles for :func:`reap`."""
    scratch = Path(tempfile.mkdtemp(prefix=_GRADE_SCRATCH_PREFIX))
    process: Optional[subprocess.Popen] = None
    pgid: Optional[int] = None
    record: Optional[str] = None
    # Set by the timeout path, which stops the group itself and reads it while the number is still
    # certainly this grader's. `None` means no exit has answered for the group yet.
    stopped: Optional[bool] = None
    # Non-inheritable by default, so no other child of this process holds the writing end open and
    # keeps the grader reporting this one alive after it has gone.
    listening, holding = os.pipe()
    try:
        process = subprocess.Popen(
            [str(runtime()), str(WORKER), "grade"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(scratch),
            env=_worker_environment(scratch),
            # Its own process group, for the reason the serving worker has one: what is stopped on
            # a timeout has to be everything this process started, and the evaluator is upstream
            # code this port does not get to promise is childless.
            start_new_session=True,
            pass_fds=(listening,),
        )
        # Read once, while the answer is certainly about this process (see `Worker.pgid`).
        pgid = _group_of(process)
        record = record_worker(
            pid=process.pid,
            pid_birth=process_birth(process.pid),
            pgid=pgid,
            scratch=str(scratch),
        )
        opening = json.dumps(
            {
                "root": str(root),
                "task_id": task_id,
                "experiment": str(outputs),
                "ignore": list(ignore),
                "filing": dict(filing),
                "keepalive": listening,
            }
        )
        try:
            # Bounded, killed and reaped. An evaluator that hangs would otherwise hold a sealed
            # episode's terminal open forever: `to_thread` does not make a child process
            # cancellable, so a deadline on the coroutine stops the waiting and leaves the child
            # running.
            out, err = process.communicate(input=opening + "\n", timeout=timeout)
        except subprocess.TimeoutExpired:
            stopped = _end_grader(process, pgid)
            raise WorkerError(
                f"grading {task_id} did not finish within {timeout:.0f}s; the grader's process "
                "group was stopped"
            ) from None
    finally:
        # The child has its own copy; a reading end still open here would be a pipe that never
        # reaches end-of-file for the grader either.
        _close_descriptor(listening)
        _close_descriptor(holding)
        if stopped is None:
            stopped = _grader_stopped(process, pgid)
        # Spent only against a group that is positively gone. Anything else keeps the record and
        # the scratch directory, which are what a later construction has instead of this frame.
        if stopped:
            shutil.rmtree(scratch, ignore_errors=True)
            if record is not None:
                forget_worker(record)
    # Reached only by a spawn that happened and a `communicate` that returned; anything else left
    # through the `finally` above.
    assert process is not None
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


def _signal_group(pgid: int, how: int) -> bool:
    """Signal every process in ``pgid``, and say whether the group was addressed at all.

    True is "the signal was delivered, or there was no member left to deliver it to": both are
    answers about this group, and the second is the one a stop is aiming at. False is this process
    being unable to address it at all (a platform with no ``killpg``, or a group this uid may not
    signal), which is not an outcome anything may be concluded from.

    :meth:`Worker._stop_the_group` ignores the answer because it reads the process table
    afterwards, which is the stronger evidence. :func:`_stop_orphan` has no such reader and
    tombstones a durable record on the strength of this, so it needs the difference."""
    try:
        os.killpg(pgid, how)
    except ProcessLookupError:
        return True
    except (OSError, AttributeError):
        return False
    return True


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


def _end_grader(process: subprocess.Popen, pgid: Optional[int]) -> bool:
    """Stop a grader that outran its bound, wait for its group to go, and say whether it did.

    Every step here is one the plain ``kill``-then-``communicate`` pair got wrong. The group is
    signalled rather than the leader, because the evaluator is upstream code that may have started
    something and a descendant of it holds the captured pipes; the signal goes out *before*
    anything reaps the leader, which is the window in which the number is certainly this grader's
    (the ordering :meth:`Worker._stop_the_group` exists for); the group's emptying is read from the
    process table rather than inferred; and the final read of the pipes carries a deadline, because
    an unbounded ``communicate`` on a descendant's pipe is exactly how a 600-second bound became no
    bound at all.

    Every wait is bounded, including this one: a teardown that cannot finish is reported by the
    caller as a grading failure, and an episode is better left unscored than left waiting on a
    process nothing can account for.

    **The answer is what the caller spends the grader's record on.** The emptying used to be
    waited for and then dropped, so a group that never emptied and a table that could not be read
    reached :func:`grade` as the same silence as a clean stop, and it tombstoned the record on all
    three. Only a table that was read and held nothing of this group is ``True`` here."""
    _stop(process, signal.SIGKILL, pgid)
    emptied: Optional[bool] = None
    if pgid is not None:
        # Enumerated while the leader is still unreaped, so this asks about descendants and about
        # nothing else (see `_group_members`).
        emptied = _group_emptied(pgid, within=_CLOSE_SECONDS)
    try:
        process.communicate(timeout=_CLOSE_SECONDS)
    except (subprocess.TimeoutExpired, ValueError, OSError):
        # A pipe something still holds, or one already closed underneath this. The group has been
        # killed and confirmed above; what is left here is a read this call will not wait on.
        pass
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass
    return emptied is True


def _grader_stopped(process: Optional[subprocess.Popen], pgid: Optional[int]) -> bool:
    """Whether a grader that ended on its own left nothing behind in its group.

    The exits :func:`_end_grader` does not cover: a leader that exited 0, one that exited nonzero,
    and a spawn that never produced a process at all. None of them used to ask about the group,
    and all of them spent the grader's record; a leader that exited 0 having started a child which
    holds none of the captured pipes is the shape that costs, because ``communicate`` returns at
    once and the descendant is left with nothing naming it.

    **Read for evidence, never signalled.** ``communicate`` has reaped the leader by the time this
    runs, and a reaped leader releases the pid and, once the last member goes, the group number
    with it. Signalling after that is signalling whoever holds the number now, which is what this
    file refuses everywhere else. Enumerating is safe in a way signalling is not: a member found
    under a recycled number can only produce a false "not empty", which costs a record kept for
    the next construction, and a record kept is the cheap direction. So the timeout path stays the
    only one that stops anything.

    A process that has no group of its own is the platform without ``setsid``, where there is
    nothing that could say what the leader started; the leader itself has been waited on, so that
    is as gone as this can be told."""
    if process is None or pgid is None:
        return True
    return _group_emptied(pgid, within=_CLOSE_SECONDS) is True


def _abandon(process: Optional[subprocess.Popen], pgid: Optional[int], scratch: Path) -> None:
    """Take back everything a worker that never got published was given.

    Killed rather than asked to stop: a handshake that did not complete is a process that never
    said anything, so there is nothing to be polite to and nothing that could be lost. The whole
    group, because the leader may already have started something. Reaped, because an unreaped
    child holds its pid and this is the one moment at which nobody else will ever wait on it. And
    the scratch directory last, which is this worker's ``HOME``, its working directory and its
    bytecode cache.

    **A leader something already reaped is not signalled**, which is the rule
    :meth:`Worker._stop_the_group` follows and which this used to break. A pid is reserved until
    its parent reaps it and a group exists while any member holds it, so an unreaped leader makes
    the stored number unambiguously this worker's; once the leader has been waited on, the kernel
    is free to hand both numbers to somebody else, and a ``killpg`` after that is a signal into a
    stranger's group. The reap can happen before this call as easily as inside it: the failed
    handshake's own diagnostic used to call ``poll()``, which reaps an exited child, and then this
    function signalled the released number. There is then nothing of this worker's left to stop,
    and the scratch directory is still cleared."""
    if process is not None:
        if process.returncode is None:
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
