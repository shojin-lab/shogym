"""The seam between shogym and AppWorld: the pins, the corpus, and one container per episode.

**AppWorld cannot be installed beside shogym**, and that is a fact about the two projects rather
than a packaging preference: ``appworld`` pins ``pydantic<2`` and shogym's MCP layer needs
``pydantic>=2.7``. No environment satisfies both. It is not installed on the host at all: it lives
in an image (:mod:`shogym.envs.appworld.container`), and every process that touches it, the
episode's world included, is a container talking to this one over its own stdin and stdout
(:mod:`shogym.envs.appworld.worker`). The separation the design wanted for isolation is the same
one packaging forces, so it costs nothing extra.

**And the container is the isolation, not an arrangement of paths.** The code an agent writes runs
as the worker, so a worker on the host runs it as the user running the run, and everything that
user can read it can read. The worker has no host path any more. It gets one task's served tree,
its own output directory, and no network; the run's provenance, the grader's tree, the corpus, the
repository and every other episode's world are not hidden from it, they are absent.

Two things are provisioned, and both are pinned here:

- **the image**, digest-pinned to its base and version-pinned to ``appworld`` at
  :data:`UPSTREAM_VERSION`, with the app sources the wheel ships packed unpacked at build time;
- **the data bundle**, which is a 33 MB download the package fetches without checking anything.
  This module fetches it itself and refuses a bundle whose digest is not
  :data:`DATA_BUNDLE_SHA256`, because every task, every database and every ground truth in the
  measurement comes out of that file and "we downloaded something from S3" is not a pin.

The corpus is *not* baked into the image, and that is deliberate rather than an omission: it
carries every task's ``ground_truth`` beside every task's ``specs.json``, so an image holding it
would put the answers inside the container that runs the agent's code.

The corpus is then *derived* rather than used in place. Each served task gets a directory whose
files are hard links to the task's own, except the one database log this port rewrites to carry
the seeded backlog. Deriving costs one small file per task, leaves the downloaded corpus
untouched, and makes the seeded rows part of the task's input state, which is where they have to
be for the scenario's own score to survive them.

Importing this module imports nothing from upstream and starts no container. Provisioning happens
when an env is built.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import secrets
import select
import shutil
import stat
import subprocess
import threading
import time
<<<<<<< HEAD
import tempfile
import sys
=======
>>>>>>> f4000d8 (appworld: run the episode worker in a container, over stdio rather than a port)
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from shogym.envs._upstream import _locked
from shogym.envs.appworld import container
from shogym.envs.appworld.container import CORPUS_MOUNT, GRADED_MOUNT, OUTPUTS_MOUNT, Mount

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

#: How long one command into a live world may take before the parent stops waiting. Generous,
#: because a block of agent code may drive the world for a while, and bounded, because a world
#: wedged in a native call would otherwise hold the episode that sent the command forever.
_CALL_TIMEOUT_SECONDS = 300.0

#: How long a worker gets to say it is ready. Generous, because a cold container importing
#: upstream and its clock-patching library is not fast, and bounded, because a worker that never
#: speaks would otherwise hang the episode that started it with nothing to read.
_SPAWN_TIMEOUT_SECONDS = 180.0

#: How long the ``docker run`` client gets to exit after its container has been removed.
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

#: How long seeding one task's database log gets.
_SEED_TIMEOUT_SECONDS = 600.0


class ProvisioningError(RuntimeError):
    """A step that builds the image or the corpus failed.

    Its own type because the two things it can mean are far apart: a machine with no network, and
    a pin that no longer resolves. A caller that wants to tolerate the first without hiding the
    second needs to be able to tell this apart from every other failure."""


def cache_root() -> Path:
    """Where this port keeps what it provisions, as an absolute path.

    Resolved rather than taken as given, for two reasons that both end in the same place. A
    relative path is read relative to whatever directory a process happens to be in, and the
    processes here are containers whose working directory is a tmpfs; and a bind mount's source
    has to be absolute or Docker reads it as a named volume and silently mounts an empty one."""
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base).expanduser().resolve() if base else Path.home() / ".cache" / "shogym"
    return root / "appworld"


# ----- provisioning -----

WORKER = container.WORKER


def ensure_image() -> str:
    """The image every world runs in, building it if this machine does not have it.

    Not a virtual environment any more. ``appworld`` pins ``pydantic<2`` and shogym needs
    ``pydantic>=2.7``, which used to force an interpreter of its own; it now forces an image of its
    own, which is the same separation with a boundary around it.

    A machine with no Docker daemon fails here as a :class:`ProvisioningError`, which is the type
    the test gate reads as "not provisioned on this machine" and skips on, and which
    ``SHOGYM_REQUIRE_UPSTREAM=1`` turns back into a failure. An env constructed on such a machine
    says so in plainer words first: see :func:`shogym.envs.appworld.container.require_docker`."""
    try:
        container.require_docker()
        return container.ensure_image(cache=cache_root())
    except container.DockerError as exc:
        raise ProvisioningError(f"the appworld worker image is not available: {exc}") from exc


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
    # The image build and the corpus fetch take the same lock, on this port's own cache directory,
    # and an ``flock`` taken twice through two opens blocks on itself even inside one process. So
    # the image is built *before* the corpus lock is taken, never inside it. A genuinely cold
    # machine takes exactly this path.
    ensure_image()
    with _locked(root.parent):
        if (root / "data" / "tasks").is_dir():
            return root
        _fetch_corpus(root)
    return root


def _fetch_corpus(root: Path) -> None:
    """Download, verify and unpack the pinned bundle into ``root``.

    Unpacked by a container, because the bundle is an encrypted archive whose format is upstream's
    business and upstream is not installed on this machine. What this function owns is the check in
    front of it. The image is built by the caller, outside this function's lock (see
    :func:`ensure_corpus`)."""
    # Named for the process that owns it rather than fixed, and never cleared before use. A fixed
    # name is a directory two cold starts both believe is theirs, and `_locked` is documented to
    # yield without exclusion on a filesystem that cannot `flock`, so "the lock stops that" is not
    # a thing this can assume. See the deferred note in the pull request.
    staging = root.with_name(f"{root.name}.{os.getpid()}.{secrets.token_hex(4)}.building")
    staging.mkdir(parents=True)
    bundle = staging / Path(DATA_BUNDLE_URL).name
    with urllib.request.urlopen(DATA_BUNDLE_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        bundle.write_bytes(response.read())
    _verify(bundle)
    _unpack_in_container(staging, bundle.name)
    bundle.unlink()
    if not (staging / "data" / "tasks").is_dir():
        raise ProvisioningError(
            f"the bundle at {DATA_BUNDLE_URL} unpacked without a data/tasks tree"
        )
    try:
        os.replace(staging, root)
    except OSError:
        # Somebody else published first, which is the only way this fails after the checks above.
        shutil.rmtree(staging, ignore_errors=True)
        if not (root / "data" / "tasks").is_dir():
            raise


def _unpack_in_container(staging: Path, bundle_name: str) -> None:
    """Open the verified bundle inside the worker image, into the staging directory."""
    work = "/work"
    process, name = container.run(
        role="unpack",
        arguments=["--bundle", f"{work}/{bundle_name}", "--into", work],
        mounts=[Mount(staging, work, writable=True)],
    )
    try:
        finished = process.wait(timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        container.remove(name)
        process.kill()
        raise ProvisioningError(
            f"unpacking the data bundle did not finish within {_DOWNLOAD_TIMEOUT_SECONDS:.0f}s"
        ) from None
    finally:
        _close_pipes(process)
    if finished != 0:
        container.remove(name)
        raise ProvisioningError(f"unpacking the data bundle failed (status {finished})")


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
    """What the worker runs under, as sixteen hex characters: the pins, and the host's Python.

    A pin rather than a reading of the installed bytes. It names the release this port asks for,
    the commit that release is recorded against, the Python series the virtual environment is
    built on, and the platform it is built for, which together are what decides which runtime
    directory is built and reused (see :func:`runtime`).

    **What it does not cover, said plainly.** The pinned release's own dependencies are ranges,
    so two machines can resolve different transitive versions under one pin, and a module edited
    in place inside the runtime moves nothing here. Reading the installed bytes is what would
    close that, and it belongs with the container the worker runs in
    (shojin-lab/shogym#140), where the image digest names the whole tree at once rather than a
    walk of the host's cache standing in for one."""
    return hashlib.sha256(
        "|".join(
            (
                UPSTREAM_VERSION,
                UPSTREAM_SHA,
                _python_series(),
                platform.system(),
                platform.machine(),
            )
        ).encode()
    ).hexdigest()[:16]


# ----- the derived corpus -----


def derived_root() -> Path:
    """Where the seeded copy of the corpus lives, named for what generated it.

    The name carries a digest of the backlog generator's own constants, so changing a cut value,
    an option set or the number of requests derives a new corpus instead of serving a stale one
    that was built under the old ones."""
    return cache_root() / f"seeded-{DATA_VERSION}-{_generator_digest()}"


def private_home() -> Path:
    """The directory holding everything an agent's world must not be handed.

    Kept out of this port's ordinary cache, and now also out of every mount an episode's container
    is given, which is the half that makes it a boundary rather than a hiding place."""
    base = cache_root().parent
    return base.parent / f"{base.name}-private" / "appworld"


def graded_root() -> Path:
    """Where the grader's view of the corpus lives: a private directory with an unguessable name.

    The unguessable name is now belt to the container's braces. It used to be the whole of the
    defence, and the port said so: the worker ran as the user who built this, so no directory mode
    kept it out and only the cost of finding it was raised. The episode's container mounts one
    task's served tree and nothing else, so this directory is not on a filesystem the world can
    see, whatever its name."""
    home = private_home()
    return home / f"graded-{DATA_VERSION}-{_generator_digest()}-{_private_tag()}"


@lru_cache(maxsize=4)
def corpus_digest(root: Path) -> str:
    """What the corpus at ``root`` actually holds, as sixteen hex characters.

    The pinned bundle's digest says what *should* be there and cannot say what is: ``APPWORLD_ROOT``
    takes any directory with a ``data/tasks`` in it, so a repointed or edited corpus would
    otherwise be served under a name that claims to be the pinned one, and would reuse a derived
    tree built from something else.

    Read in full for the files that decide what a task is and what it is scored against, and by
    name and size for the databases, which are 179 MB and would make this a minute of reading on
    every construction. A database edited without changing its length is therefore not caught, and
    that is the honest limit of this digest rather than a claim it does not make."""
    digest = hashlib.sha256()
    data = root / "data"
<<<<<<< HEAD
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
=======
    version = data / "version.txt"
    digest.update(version.read_bytes() if version.exists() else b"")
    for path in sorted((data / "tasks").rglob("*")):
>>>>>>> f4000d8 (appworld: run the episode worker in a container, over stdio rather than a port)
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(data)).encode())
        if path.name == "specs.json":
            digest.update(path.read_bytes())
        else:
            digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:16]


def episode_outputs(session_id: str) -> Path:
    """One episode's own output tree, under the private home and named for the episode.

    Outside every served corpus, and mounted alone into the two containers that need it. Upstream's
    evaluator writes a report quoting the requirement prose and the values behind it, and a shared
    output tree is a place the other arm of a pair can read an earlier grade; a per-episode tree
    that no other episode's container mounts is a place nothing else can read at all."""
    home = private_home() / f"episodes-{DATA_VERSION}-{_private_tag()}"
    home.mkdir(parents=True, exist_ok=True)
    return home / f"episode-{session_id}"


@lru_cache(maxsize=1)
def _private_tag() -> str:
    """Sixteen hex characters, drawn once per installation and kept beside the private tree.

    Persisted rather than redrawn, because a name that changed per process would derive the whole
    corpus again on every run."""
    home = private_home()
    home.mkdir(parents=True, exist_ok=True)
    os.chmod(home, 0o700)
    keyfile = home / ".tag"
    # Created exclusively, and a loser reads the winner's value rather than keeping its own. Two
    # processes that each kept their own would name two private trees, and a private tree is a
    # copy of the corpus: the cost of getting this wrong is 134 MB and a rebuild, per loser.
    try:
        handle = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        tag = keyfile.read_text().strip()
        if len(tag) == 16:
            return tag
        raise RuntimeError(f"the private tag at {keyfile} is not a tag; remove it to rebuild")
    try:
        tag = secrets.token_hex(8)
        os.write(handle, tag.encode())
    finally:
        os.close(handle)
    return tag


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
    """The worker refused a command, the world raised inside one, or the container went away."""


class _Frames:
    """Length-prefixed frames off a pipe, with a deadline on every read.

    A buffer of its own rather than a :class:`io.BufferedReader`, because a deadline needs
    ``select`` on the descriptor and a buffered reader may already hold bytes ``select`` will
    never announce again. Everything read goes through here, so nothing is ever left in two
    places."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.buffer = bytearray()

    def _fill(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the appworld worker did not answer in time")
        ready, _, _ = select.select([self.fd], [], [], remaining)
        if not ready:
            raise TimeoutError("the appworld worker did not answer in time")
        chunk = os.read(self.fd, 65536)
        if not chunk:
            raise EOFError("the appworld worker closed its output")
        self.buffer += chunk

    def frame(self, timeout: float) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            self._fill(deadline)
        header, _, rest = bytes(self.buffer).partition(b"\n")
        self.buffer = bytearray(rest)
        length = int(header.strip())
        while len(self.buffer) < length:
            self._fill(deadline)
        body = bytes(self.buffer[:length])
        del self.buffer[:length]
        return json.loads(body)

#: What every worker's scratch directory is named for, so a directory this port left behind on a
#: host is one an operator can recognise.
_SCRATCH_PREFIX = "shogym-appworld-"

#: A grader's scratch directory, which is a worker's with a word added.
_GRADE_SCRATCH_PREFIX = _SCRATCH_PREFIX + "grade-"


def _close_descriptor(descriptor: Optional[int]) -> None:
    """Close a raw descriptor, and treat one that is already closed as closed."""
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    except OSError:
        pass


def _send(process: subprocess.Popen, payload: Dict[str, Any]) -> None:
    assert process.stdin is not None
    encoded = json.dumps(payload).encode()
    process.stdin.write(b"%d\n" % len(encoded))
    process.stdin.write(encoded)
    process.stdin.flush()


def _close_pipes(process: subprocess.Popen) -> None:
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


def _worker_environment(root: str) -> Dict[str, str]:
    """What one worker container is given, which is all it has.

    Nothing is inherited: ``docker run`` passes the image's own environment and what is named
    here, so the serving process's provider keys and run paths are not absent because they were
    removed, they are absent because they were never offered. This used to be an allow-list over
    ``os.environ``, which is the same list with a different failure mode: a name nobody thought of
    still got through."""
    return {
        "APPWORLD_ROOT": root,
        "HOME": container.SCRATCH_MOUNT,
        "LANG": "C.UTF-8",
        # The root filesystem is read-only, so a `.pyc` written on first import could not be kept
        # anyway; the image byte-compiled everything at build time.
        "PYTHONDONTWRITEBYTECODE": "1",
    }


def served_mounts(*, root: Path, task_id: str, outputs: Path) -> List[Mount]:
    """Everything one episode's world can see, and it is a short list.

    The corpus's shared parts read-only, this one task's derived tree read-only, and this one
    episode's output tree, which is the only writable mount there is. ``data/tasks`` therefore
    holds exactly one directory inside the container: another episode's world is not a path that
    resolves and is refused, it is a path that does not exist.

    The output tree is mounted at a fixed name outside the corpus rather than under it, and the
    world is told its experiment *is* that directory. AppWorld joins an experiment name onto its
    own output root, so an absolute one replaces the root, which is what keeps an episode's end
    state and its evaluator artifacts out of the tree the next episode is served.

    ``root`` is the derived root, the parent of ``data``."""
    mounts = [
        Mount(entry, f"{CORPUS_MOUNT}/data/{entry.name}")
        for entry in sorted((root / "data").iterdir())
        if entry.name != "tasks"
    ]
    mounts.append(Mount(root / "data" / "tasks" / task_id, f"{CORPUS_MOUNT}/data/tasks/{task_id}"))
    mounts.append(Mount(outputs, OUTPUTS_MOUNT, writable=True))
    return mounts


def graded_mounts(*, graded: Path, task_id: str, outputs: Path) -> List[Mount]:
    """What the grading container sees: the answers, and the end state to grade against them.

    The same shape as :func:`served_mounts` and for a different reason. Nothing here ever runs a
    line an agent wrote, so this is not a boundary; it is the mount set that makes the grader's
    view exist at all, because the answers and the episode's output tree live in two different
    places on the host and the evaluator wants a root and an experiment."""
    mounts = [
        Mount(entry, f"{GRADED_MOUNT}/data/{entry.name}")
        for entry in sorted((graded / "data").iterdir())
        if entry.name != "tasks"
    ]
    mounts.append(Mount(graded / "data" / "tasks" / task_id, f"{GRADED_MOUNT}/data/tasks/{task_id}"))
    mounts.append(Mount(outputs, OUTPUTS_MOUNT, writable=True))
    return mounts


@dataclass
class Worker:
    """A handle on one episode's world, running in a container of its own.

    Reached over the pipe pair this process created for it and by nothing else: the container has
    no network stack, so there is no port to find and no token to need. What it can see is the
    mount set :func:`served_mounts` builds, which is one task's served tree and this episode's own
    output directory.

    The container's name is the handle teardown needs. Killing the ``docker run`` client does not
    stop a container, so :meth:`close` removes it by name and does so on every path out, including
    the ones where spawning failed halfway."""

    root: Path
    process: subprocess.Popen
    container: str
    frames: "_Frames"
    lock: threading.Lock = field(default_factory=threading.Lock)
    closed: bool = False
    #: Set by the first call that stopped waiting for an answer. A worker whose protocol has an
    #: outstanding command is a worker whose next answer belongs to a caller that is gone, so it
    #: is never used again; the field holds the reason, which is what the refusal says.
    poisoned: str = ""
    #: The identifier of the next command. Monotonic within a worker, and echoed on the answer.
    counter: int = 0

    @classmethod
    def spawn(cls, root: Path, *, task_id: str, outputs: Path) -> "Worker":
        """Start a worker on one task and wait for it to say it is ready.

        The output directory is created here rather than by Docker, because a bind mount whose
        source does not exist is created by the daemon and owned by root, and the container runs
        as this user."""
        outputs.mkdir(parents=True, exist_ok=True)
        process, name = container.run(
            role="serve",
            mounts=served_mounts(root=root, task_id=task_id, outputs=outputs),
            environment=_worker_environment(CORPUS_MOUNT),
        )
        assert process.stdout is not None
        frames = _Frames(process.stdout.fileno())
        try:
            opening = frames.frame(_SPAWN_TIMEOUT_SECONDS)
        except (TimeoutError, EOFError, ValueError) as exc:
            container.remove(name)
            process.kill()
            _close_pipes(process)
            raise WorkerError(
                f"the appworld worker container never became ready ({type(exc).__name__}: {exc}); "
                f"waited {_SPAWN_TIMEOUT_SECONDS:.0f}s"
            ) from exc
        if not opening.get("ready"):
            container.remove(name)
            process.kill()
            _close_pipes(process)
            raise WorkerError(f"the appworld worker answered {opening!r} instead of ready")
        return cls(root=root, process=process, container=name, frames=frames)

    def call(self, command: str, **body: Any) -> Any:
        """Send one command and return the answer to *that* command.

        Under a lock, because the frames on one pipe are one sequence: two callers interleaving
        writes would produce a frame neither of them sent.

        **Every frame carries an identifier and the answer echoes it.** The transport before this
        was HTTP, where a response belonged to its own request by construction. An ordered pipe
        has no such property: a command that timed out is a command still running, and its answer
        arrives later, into a stream the next caller is reading. Without an identifier that answer
        is read as the next command's, which is an earlier block's output returned under a later
        step, or a finalizer handed the wrong shape entirely. So an answer whose identifier is not
        the one this call sent is discarded and the wait continues.

        **And a call that stopped waiting poisons the worker.** Discarding a stale answer keeps
        one call honest; it does not make the world safe to keep using, because the command that
        timed out is still running inside it and may still be changing the world. There is no
        state in which a timed-out worker is worth reusing, so it is refused."""
        with self.lock:
            if self.poisoned:
                raise WorkerError(
                    f"the appworld worker is not usable and {command!r} was refused: "
                    f"{self.poisoned}"
                )
            self.counter += 1
            identifier = self.counter
            deadline = time.monotonic() + _CALL_TIMEOUT_SECONDS
            try:
                _send(self.process, {"id": identifier, "command": command, "body": body})
                while True:
                    answer = self.frames.frame(max(0.0, deadline - time.monotonic()))
                    if answer.get("id") == identifier:
                        break
                    # An answer to a command nobody is waiting for. Only reachable when a worker
                    # was reused after a timeout, which it is not; kept because reading a frame
                    # and hoping is exactly the failure this is here to prevent.
            except (BrokenPipeError, EOFError) as exc:
                self.poisoned = f"the container stopped during {command!r}: {exc}"
                raise WorkerError(f"the appworld worker container {self.poisoned}") from exc
            except TimeoutError as exc:
                self.poisoned = (
                    f"{command!r} did not answer within {_CALL_TIMEOUT_SECONDS:.0f}s and is "
                    "still running, so no later answer on this pipe can be trusted"
                )
                raise WorkerError(f"the appworld worker: {self.poisoned}") from exc
        if "error" in answer:
            raise WorkerError(f"appworld worker refused {command!r}: {answer['error']}")
        return answer["output"]

    def close(self, *, confirm: bool = False) -> None:
        """Stop the worker, promptly and with a bound.

        The container is removed first and by name. Closing the pipe would be the polite way and
        is not the reliable one: a world wedged in a native call never reads its stdin again, and
        the ``docker run`` client does not stop a container by dying. What ``--rm`` and the pipe
        do cover is the case this cannot: a parent that crashes leaves a worker reading
        end-of-file, which stops it, and :func:`~shogym.envs.appworld.container.reap` covers the
        parent that crashes while the worker cannot get back to that read.

        Idempotent, because it is called twice on the ordinary path: finalization stops the world
        before the grader reads its end state, and teardown then closes the session it was in.

        ``confirm`` is finalization's, and it is the difference between a removal and a fact.
        Finalization removes this container precisely so that nothing can write to the tree it is
        about to grade; a removal the daemon did not confirm leaves that invariant unproven, so it
        raises rather than proceeding, and the worker is *not* marked closed, so teardown will
        try again."""
        if self.closed:
            return
        container.remove(self.container, confirm=confirm)
        self.closed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=_TERM_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
<<<<<<< HEAD
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
=======
                self.process.kill()
                try:
                    self.process.wait(timeout=_CLOSE_SECONDS)
                except subprocess.TimeoutExpired:
                    pass
        _close_pipes(self.process)


def _one_shot(
    *,
    role: str,
    body: Dict[str, Any],
    mounts: Sequence[Mount],
    environment: Dict[str, str],
    timeout: float,
    what: str,
) -> Any:
    """Run one short-lived container, send it one frame, and return its one answer.

    The shape ``seed`` and ``grade`` share. Both are bounded, killed and reaped: a container that
    hangs would otherwise hold a sealed episode's terminal open forever, and removing it by name
    is the only thing that actually stops it."""
    process, name = container.run(
        role=role, mounts=list(mounts), environment=environment
    )
    frames = _Frames(process.stdout.fileno()) if process.stdout is not None else None
    try:
        assert frames is not None
        _send(process, {"body": body})
        answer = frames.frame(timeout)
    except TimeoutError as exc:
        container.remove(name)
        process.kill()
        raise WorkerError(f"{what} did not finish within {timeout:.0f}s; it was killed") from exc
    except (BrokenPipeError, EOFError, ValueError) as exc:
        container.remove(name)
        process.kill()
        raise WorkerError(f"{what} failed: {type(exc).__name__}: {exc}") from exc
    finally:
>>>>>>> f4000d8 (appworld: run the episode worker in a container, over stdio rather than a port)
        try:
            process.wait(timeout=_CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
<<<<<<< HEAD
            return False
        return empty
=======
            container.remove(name)
            process.kill()
        _close_pipes(process)
    if "error" in answer:
        raise WorkerError(f"{what} failed: {answer['error']}")
    return answer["output"]
>>>>>>> f4000d8 (appworld: run the episode worker in a container, over stdio rather than a port)


def seed(*, root: Path, source_dbs: Path, into: Path, rows: Dict[str, Any]) -> Any:
    """Write one task's seeded database log, in a container of its own.

<<<<<<< HEAD
    Its own type because it is an episode-level failure with a cause worth naming: the tree the
    world left behind holds something that is not a plain file, and a grader that opened it would
    be resolving a path the agent chose in a process that also holds the answers."""


def snapshot_outputs(outputs: Path, *, into: Path) -> Path:
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

    Every refusal is an episode refused outright rather than an entry skipped, because a grade
    computed over a tree with something quietly dropped is a grade over a tree nobody submitted.

    Safe to walk because the worker is already gone: this runs after the worker has been stopped
    and waited on, so nothing ordinary can add a link between the check and the copy.

    **Unbounded, and that is the host worker's boundary rather than an oversight.** The tree was
    written by the process that ran agent-authored code, so its size, its depth and how long it
    takes to walk are the episode's to choose. What bounds them is the container the worker runs
    in (shojin-lab/shogym#140), which owns the disk the tree is written to."""
    if outputs.is_symlink():
        raise SnapshotError(
            f"the episode's output root {outputs} is a symbolic link, so what would be graded is "
            "whatever it names rather than what the episode wrote"
        )
    root = outputs.resolve()
    if not root.is_dir():
        raise SnapshotError(f"the episode left no output tree at {outputs}")
    # Whatever is under the destination name goes, whatever it is. `rmtree` alone leaves a plain
    # file or a symbolic link exactly where it is, and the destination is a sibling of the tree the
    # episode wrote and so a name the episode can create: one left as a file turned the next
    # `mkdir` into a bare `FileExistsError` rather than this module's own refusal.
    if into.is_symlink() or into.is_file():
        into.unlink()
    else:
        shutil.rmtree(into, ignore_errors=True)
    into.mkdir(parents=True)
    # One pass: what is checked is what is copied. A validate-then-`copytree` would walk the tree
    # twice, and `copytree` on its own resolves the links this refuses.
    pending: List[Tuple[Path, Path]] = [(root, into)]
    while pending:
        source, target = pending.pop()
        for name in sorted(os.listdir(source)):
            entry = source / name
            if entry.is_symlink():
                raise SnapshotError(
                    f"the episode left a symbolic link in its output tree ({name} -> "
                    f"{os.readlink(entry)}), which a grader must not resolve"
                )
            if entry.is_dir():
                (target / name).mkdir()
                pending.append((entry, target / name))
                continue
            if not entry.is_file():
                raise SnapshotError(f"the episode left {name}, which is not a file or directory")
            shutil.copyfile(entry, target / name)
    return into
=======
    ``source_dbs`` is the task's own input databases in the downloaded corpus and ``into`` is the
    file being written inside a staging directory. ``root`` is the derived root, mounted because
    upstream's model layer resolves an app's base database under ``APPWORLD_ROOT`` and a load with
    no root reaches for one under the working directory.

    Three mounts, one of them writable, and that one is the staging directory. This container runs
    no agent code, so its mount set is a matter of keeping the seam narrow rather than a boundary:
    what it can see is one task's inputs and the tree it is writing into."""
    shared = [
        Mount(entry, f"{CORPUS_MOUNT}/data/{entry.name}")
        for entry in sorted((root / "data").iterdir())
        if entry.name != "tasks"
    ]
    return _one_shot(
        role="seed",
        body={**rows, "from_dbs": "/from", "into": f"/into/{into.name}"},
        mounts=[
            *shared,
            Mount(source_dbs, "/from"),
            Mount(into.parent, "/into", writable=True),
        ],
        environment=_worker_environment(CORPUS_MOUNT),
        timeout=_SEED_TIMEOUT_SECONDS,
        what=f"seeding {into.parent.parent.name}",
    )
>>>>>>> f4000d8 (appworld: run the episode worker in a container, over stdio rather than a port)


def grade(
    *,
    graded: Path,
    task_id: str,
    outputs: Path,
    ignore: Sequence[str],
    filing: Dict[str, str],
    timeout: float = _GRADE_TIMEOUT_SECONDS,
) -> Any:
    """The base task's own checks, from a container that has never run a line the agent wrote.

    A second, short-lived container rather than the one that served the episode. It is the only
    place ground truth is loaded, it starts after the world is sealed, and it reads the end state
    off the episode's own output directory, so the answers are never objects in the process the
    agent's code ran as and never files on a filesystem that process could see.

    It also reads the filing and digests the databases, off that same tree and in that same
    process, which is what makes the scored state and the graded state one state rather than two
    observations of a live world that happened to agree. ``filing`` names what to look for: the
    supervisor whose account holds it, the project, the row's title and the label. None of it is
    an answer; it is the address of the row the appended paragraph asked for.

    ``graded`` is the grader's root, the parent of ``data``."""
    return _one_shot(
        role="grade",
        body={
            "root": GRADED_MOUNT,
            "task_id": task_id,
            "experiment": OUTPUTS_MOUNT,
            "ignore": list(ignore),
            **filing,
        },
        mounts=graded_mounts(graded=graded, task_id=task_id, outputs=outputs),
        environment=_worker_environment(GRADED_MOUNT),
        timeout=timeout,
        what=f"grading {task_id}",
    )
<<<<<<< HEAD
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
        process = subprocess.Popen(
            [str(runtime()), str(WORKER), "grade"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(scratch),
            env=_worker_environment(scratch),
        )
        opening = json.dumps(
            {
                "root": str(root),
                "task_id": task_id,
                "experiment": str(outputs),
                "ignore": list(ignore),
                "filing": dict(filing),
            }
        )
        try:
            out, err = process.communicate(input=opening + "\n", timeout=timeout)
        except subprocess.TimeoutExpired:
            _abandon(process, scratch)
            raise WorkerError(
                f"grading {task_id} did not finish within {timeout:.0f}s; the grader was stopped"
            ) from None
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    # Reached only by a spawn that happened and a `communicate` that returned; anything else left
    # through the `finally` above.
    assert process is not None
    if process.returncode != 0:
        raise WorkerError(
            f"grading {task_id} failed (status {process.returncode}): {err.strip()[-2000:]}"
        )
    return json.loads(out.strip().splitlines()[-1])["output"]


def _abandon(process: Optional[subprocess.Popen], scratch: Path) -> None:
    """Take back everything a worker that never got published, or outran its bound, was given.

    Killed rather than asked to stop: a handshake that did not complete is a process that never
    said anything, and a grader past its deadline has had its time. Reaped, because an unreaped
    child holds its pid and this is the one moment at which nobody else will ever wait on it. And
    the scratch directory last, which is this worker's ``HOME`` and its working directory."""
    if process is not None:
        if process.returncode is None:
            try:
                process.kill()
            except OSError:
                pass
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
    ready, _, _ = select.select([process.stdout], [], [], timeout)
    if not ready:
        return ""
    return process.stdout.readline()
=======
>>>>>>> f4000d8 (appworld: run the episode worker in a container, over stdio rather than a port)


__all__ = [
    "CorpusSnapshot",
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
    "corpus_snapshot",
    "derived_root",
    "episode_outputs",
    "ensure_corpus",
    "ensure_image",
    "grade",
    "graded_mounts",
    "graded_root",
    "private_home",
<<<<<<< HEAD
    "ensure_apps",
    "ensure_corpus",
    "runtime",
    "runtime_digest",
    "snapshot_outputs",
    "stamp_cache",
=======
    "seed",
    "served_mounts",
>>>>>>> f4000d8 (appworld: run the episode worker in a container, over stdio rather than a port)
    "task_ids",
    "task_specs",
]
