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
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Tuple

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


class ProvisioningError(RuntimeError):
    """A step that builds the interpreter or the corpus failed.

    Its own type because the two things it can mean are far apart: a machine with no network, and
    a pin that no longer resolves. A caller that wants to tolerate the first without hiding the
    second needs to be able to tell this apart from every other failure."""



def cache_root() -> Path:
    """Where this port keeps what it provisions."""
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "shogym"
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
    with _locked(root.parent):
        if (root / "data" / "tasks").is_dir():
            return root
        _fetch_corpus(root)
    return root


def _fetch_corpus(root: Path) -> None:
    """Download, verify and unpack the pinned bundle into ``root``.

    Unpacked by the provisioned interpreter, because the bundle is an encrypted archive whose
    format is upstream's business. What this function owns is the check in front of it."""
    import shutil

    python = runtime()
    ensure_apps()
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


@dataclass
class Worker:
    """A handle on one episode's world, running in a process of its own.

    The port is the process's, the token is this object's, and neither is ever put anywhere an
    agent can read: not in the instructions the env publishes, not in a tool's schema, and not in
    a tool's result. That is what keeps the unauthenticated grading routes AppWorld's own server
    publishes out of reach of the thing being graded."""

    root: Path
    process: subprocess.Popen
    port: int
    token: str

    @classmethod
    def spawn(cls, root: Path) -> "Worker":
        """Start a worker on ``root`` and wait for it to say which port it bound."""
        token = secrets.token_urlsafe(32)
        process = subprocess.Popen(
            [str(runtime()), str(WORKER), "serve", "--root", str(root), "--token", token],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        assert process.stdout is not None
        line = process.stdout.readline()
        if not line:
            process.wait(timeout=5)
            raise WorkerError(
                f"the appworld worker exited before it bound a port (status {process.returncode})"
            )
        return cls(root=root, process=process, port=int(json.loads(line)["port"]), token=token)

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
        """Tell the worker to shut down, and make sure it did."""
        if self.process.poll() is None:
            try:
                self.call("close")
            except Exception:
                pass
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        if self.process.stdout is not None:
            self.process.stdout.close()


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
    "derived_root",
    "ensure_apps",
    "ensure_corpus",
    "runtime",
    "task_ids",
    "task_specs",
]
