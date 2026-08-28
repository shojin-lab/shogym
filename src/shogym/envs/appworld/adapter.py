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
import secrets
import select
import shutil
import stat
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from shogym.envs._upstream import _locked
from shogym.envs.appworld import container, world
from shogym.envs.appworld.container import CORPUS_MOUNT, GRADED_MOUNT, OUTPUTS_MOUNT, Mount

# ----- the pins -----

#: The release this port reproduces, and the commit it was cut from.
#:
#: **Both are load-bearing, and they are load-bearing in different ways.** The version is what the
#: image installs: ``worker.Dockerfile`` asks pip for exactly it, so a build that resolved some
#: other release fails rather than publishing, and a test holds the Dockerfile to this constant so
#: the two cannot drift apart. The commit is not checkable against the artifact at all: the
#: published wheel carries no marker of the tree it was built from, so nothing on this machine can
#: say whether ``0.1.3.post1`` was cut from this commit. What it does instead is go into the run
#: fingerprint, so changing the pin moves every run's identity rather than passing for the earlier
#: measurement. What the realized bytes turned out to be is :func:`runtime_digest`'s job, which is
#: the question a pin can never answer.
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
    # Required, not advisory: the corpus is the material every score is computed against, and a
    # window in which two fetchers are inside one cache directory is not made safe by staging.
    # See `_upstream._locked`.
    with _locked(root.parent, required=True):
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
    # Named for the process that owns it rather than fixed, and never cleared before use. The
    # caller's lock is required now, so this is belt and braces rather than the only exclusion:
    # what it buys is that an abandoned staging tree is one process's residue under its own name
    # rather than a directory the next cold start believes is its own.
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
        _release(process, name)
        raise ProvisioningError(
            f"unpacking the data bundle did not finish within {_DOWNLOAD_TIMEOUT_SECONDS:.0f}s"
        ) from None
    else:
        _close_pipes(process)
    if finished != 0:
        _release(process, name)
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


#: Bumped when the shape of a derived tree changes: what is copied, what is linked, what is
#: sealed. It is in both cache names, so a tree built under an older layout is a different cache
#: rather than a stale one wearing the same name.
#:
#: 2: the shared half of a derived root is the entries this port names rather than everything the
#: corpus had beside ``tasks`` (see :data:`~world.SHARED_ENTRIES`), so a tree built under 1 holds
#: entries a tree built under 2 does not.
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


def control_home() -> Path:
    """Where this port keeps what it says about episodes, as opposed to what an episode wrote.

    **Nothing here is ever mounted.** An episode's output tree is bound into its container
    writable, so anything kept inside it is a fact the subject of that fact can rewrite: an owner
    marker in there is cleanup authority an episode can forge, and a completion record in there is
    a claim about a save made by the thing that was saving. So the control plane is a directory of
    its own, beside the trees rather than inside them, and every mount set in this module is built
    from an explicit list that does not include it."""
    return cache_root() / f"control-{DATA_VERSION}"


def control_file(root: Path, kind: str) -> Path:
    """The control-plane file of ``kind`` for the per-episode tree at ``root``.

    Named for the tree's own directory and the home it sits in, so two homes cannot collide and a
    reader with the tree in hand can find the file without a lookup table."""
    home = control_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / f"{root.parent.name}--{root.name}.{kind}"


def episodes_home() -> Path:
    """Where per-episode output trees live: this port's ordinary cache, not the private one."""
    return cache_root() / f"episodes-{DATA_VERSION}"


def episode_outputs(session_id: str) -> Path:
    """One episode's own output tree, named for the episode and not under the private home.

    Outside every served corpus, and mounted alone into the two containers that need it. Upstream's
    evaluator writes a report quoting the requirement prose and the values behind it, and a shared
    output tree is a place the other arm of a pair can read an earlier grade; a per-episode tree
    that no other episode's container mounts is a place nothing else can read at all.

    **Deliberately not a child of** :func:`private_home`. Under the private home it named the
    grader's own parent, and the unguessable name that protects the grader stops protecting it
    the moment something hands an address inside it over. Nothing hands this one to the world,
    which sees it at a fixed mount point, but the host path is in the container's own mount table
    and so it must not be a path that says where the answers are."""
    home = episodes_home()
    home.mkdir(parents=True, exist_ok=True)
    return home / f"episode-{session_id}"


def runtime_digest() -> str:
    """What the worker's interpreter actually is, as sixteen hex characters.

    The branch below reads a virtual environment's realized distribution set, because there the
    runtime cache is named for the direct AppWorld release while it is built by resolving that
    release's ranges against whatever the host offers on the day. Here there is no virtual
    environment: the interpreter is an image, and the daemon holds one answer for what that image
    is. The image id is that answer, and it moves for every reason the distribution set would
    have, including the ones a tag cannot see (a re-pushed base, a transitive version that
    resolved differently, the same tag built on another architecture).

    Not memoized, for the reason :func:`corpus_digest` is not: the value has to be able to move
    when the thing it names does."""
    return hashlib.sha256(
        container.image_identity(container.image_name()).encode()
    ).hexdigest()[:16]


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
    # Written whole, then published by link. An exclusive create publishes the *name* before the
    # bytes, so a concurrent reader could see a real file holding nothing, and a crash in the gap
    # left that empty file behind for good. The link is atomic within a directory, so the name
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


#: The most a length header may be before the frame is refused. A decimal byte count is twenty
#: digits at the outside; anything longer is not a header this protocol writes.
_MAX_HEADER_BYTES = 32

#: The most one frame may declare. The parent's buffer is a host allocation and the container's
#: memory limit does not bound it, so a writer inside the container asking for more memory than
#: the host has is a writer this must refuse rather than obey. Sixteen mebibytes is far past any
#: block's printed output and far below anything that hurts.
_MAX_FRAME_BYTES = 16 * 1024 * 1024


class FramingError(RuntimeError):
    """A frame this protocol would not have written arrived on the pipe.

    Fatal by construction rather than by policy: the stream's position is no longer known, so
    there is no next frame to read, and the writer is reachable from inside the interpreter that
    runs agent code. Its caller poisons the worker and removes the container."""


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

    def frame(self, timeout: float, *, expect: Sequence[str]) -> Dict[str, Any]:
        """The next frame, bounded at both ends and of the shape this command's answer has.

        **Both limits are about this process rather than about the worker.** The buffer is a host
        allocation, and the container's memory limit does not reach it: a writer inside the
        container declaring a body of a hundred gigabytes would have the parent allocate it. So a
        header longer than a length, a header that is not a length, and a body larger than any
        block's output could be are all refused before anything is read, and the refusal is fatal:
        once a frame is not what this protocol writes, the stream's position is unknown and there
        is no next frame to look for."""
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            if len(self.buffer) > _MAX_HEADER_BYTES:
                raise FramingError(
                    f"the appworld worker sent {len(self.buffer)} bytes with no length header; "
                    f"a header is at most {_MAX_HEADER_BYTES}"
                )
            self._fill(deadline)
        header, _, rest = bytes(self.buffer).partition(b"\n")
        self.buffer = bytearray(rest)
        try:
            length = int(header.strip())
        except ValueError:
            raise FramingError(
                f"the appworld worker sent {header[:_MAX_HEADER_BYTES]!r} where a byte count "
                "belongs"
            ) from None
        if length < 0 or length > _MAX_FRAME_BYTES:
            raise FramingError(
                f"the appworld worker declared a {length} byte frame; the most this reads is "
                f"{_MAX_FRAME_BYTES}"
            )
        while len(self.buffer) < length:
            self._fill(deadline)
        body = bytes(self.buffer[:length])
        del self.buffer[:length]
        try:
            answer = json.loads(body)
        except ValueError as exc:
            raise FramingError(f"the appworld worker sent a frame that is not JSON: {exc}") from exc
        # **Shape is part of framing, and the shape is this command's.** Valid JSON of the wrong
        # kind used to travel straight into the caller, which then did `.get` on a list or indexed
        # `output` on an object that had none: an `AttributeError` or a `KeyError` out of the
        # transport, with the worker neither poisoned nor stopped and the spawn path skipping its
        # own release.
        #
        # **Per command, because "one of three keys" is not a shape.** A startup frame carrying
        # only `ready` satisfied that test, so a `ready` arriving in answer to an `execute` left
        # the protected read loop and raised `KeyError('output')` outside every handler here,
        # after the worker's lock had been released and while the command it belonged to might
        # still have been running. What a spawn expects and what a call expects are two different
        # frames, so each one says which it is waiting for. The writer is reachable from inside
        # the interpreter that runs agent code, so an answer that is not an answer to what was
        # asked is exactly the thing this boundary is for.
        if not isinstance(answer, dict):
            raise FramingError(
                f"the appworld worker sent a {type(answer).__name__} where a frame belongs"
            )
        if not (set(expect) & set(answer)):
            raise FramingError(
                f"the appworld worker sent a frame carrying none of "
                f"{', '.join(expect)}: {sorted(answer)[:8]}"
            )
        return answer

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


#: The two frames this protocol has. A spawn is answered by a startup frame and a command by an
#: answer; nothing on this pipe is ever both, so neither is ever accepted where the other belongs.
_READY_FRAME: Tuple[str, ...] = ("ready",)
_ANSWER_FRAME: Tuple[str, ...] = ("output", "error")

#: The most one request may be. Everything this sends is a short command or one block of the
#: agent's code, and a block arrives over the wire under the serve layer's own limits, so this is
#: an upper bound rather than a budget anything spends. It exists because the check below is a
#: bound on *waiting*: a request larger than the pipe's capacity cannot be written in one go, and
#: a refusal is a better answer than a partial write nobody can take back.
_MAX_REQUEST_BYTES = 16 * 1024 * 1024


def _send(process: subprocess.Popen, payload: Dict[str, Any], *, deadline: float) -> None:
    """Write one frame, under the same deadline the answer is read under.

    **The write is inside the bound, and it was not.** ``stdin.write`` and ``flush`` are blocking
    calls into a pipe with a finite buffer, and the worker is the only reader. A worker that has
    stopped reading — wedged in a native call, stopped by a signal, running agent code that never
    returns — leaves a request bigger than the remaining pipe capacity blocking in the host's
    ``write`` for ever, and the deadline was not consulted until the first read afterwards. So a
    call that promised to time out did not, and neither the poison nor the removal that follows a
    timeout ever ran.

    The descriptor is put in non-blocking mode and written through ``select``, so the wait is on
    the deadline rather than on the reader. Every write to this pipe goes through here, so nothing
    else can be surprised by that mode."""
    assert process.stdin is not None
    encoded = json.dumps(payload).encode()
    if len(encoded) > _MAX_REQUEST_BYTES:
        # Refused before a byte is written, so the stream is still in position and the worker is
        # still usable: this is a request this process declined to make, not a protocol that
        # broke. A partial write is the other thing entirely, and it poisons.
        raise WorkerError(
            f"a {len(encoded)} byte request is more than the {_MAX_REQUEST_BYTES} this writes"
        )
    fd = process.stdin.fileno()
    os.set_blocking(fd, False)
    pending = memoryview(b"%d\n" % len(encoded) + encoded)
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("the appworld worker did not read its request in time")
        _, ready, _ = select.select([], [fd], [], remaining)
        if not ready:
            raise TimeoutError("the appworld worker did not read its request in time")
        try:
            written = os.write(fd, pending)
        except BlockingIOError:
            continue
        pending = pending[written:]


def _release(process: subprocess.Popen, name: str) -> None:
    """Give up a container and the local client together, and never raise doing it.

    The ordering is deliberate and so is the ``finally``: the container is what holds a mount, so
    it goes first, and the pipes and the child are this process's own, so they go whatever the
    daemon said. A removal nobody could confirm is handed to the sweep by name, because the
    ordinary sweep skips a container whose parent is still alive and here the parent is."""
    try:
        if not container.remove(name):
            container.disowned(name)
    except container.DockerError:
        container.disowned(name)
    finally:
        try:
            process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=_CLOSE_SECONDS)
        except (subprocess.TimeoutExpired, OSError):
            pass
        _close_pipes(process)


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


def _shared_mounts(data: Path, at: str) -> List[Mount]:
    """The shared half of a corpus, named rather than enumerated, and only what is there.

    **An allowlist, where this was a denylist.** Every top-level entry except ``tasks`` was
    mounted, which puts whatever a corpus happens to carry inside the container that runs
    agent-authored code: the pinned bundle ships a `LICENSE` and a `README_BEFORE_SHARING.md`
    beside the four this port serves, and ``APPWORLD_ROOT`` takes any directory with a
    ``data/tasks`` in it. None of that was ground truth and none of it was a grade; what it was is
    a list this port described as exhaustive and did not build that way.

    Only what exists, because a bind whose source is missing is a directory the daemon creates and
    root owns, and a corpus without one of these fails where the world opens it, with upstream's
    own words, which is where it failed before this list existed."""
    return [
        Mount(data / name, f"{at}/data/{name}")
        for name in world.SHARED_ENTRIES
        if (data / name).exists()
    ]


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

    **The shared half is an allowlist, and was a denylist.** It used to be every top-level entry
    of the derived root except ``tasks``, which puts anything the corpus happens to carry inside
    the boundary by default: the pinned bundle already ships a `LICENSE` and a
    `README_BEFORE_SHARING.md`, and ``APPWORLD_ROOT`` takes any directory with a ``data/tasks`` in
    it, so a custom corpus's own files were mounted because nothing had said they should not be.
    None of that was ground truth and none of it was a grade; what it was is a list this port
    described as exhaustive and did not build that way. The named entries are the ones a world
    reads (:data:`~world.SHARED_ENTRIES`), and an entry not on the list is not mounted, whoever
    put it in the corpus.

    ``root`` is the derived root, the parent of ``data``."""
    mounts = _shared_mounts(root / "data", CORPUS_MOUNT)
    mounts.append(Mount(root / "data" / "tasks" / task_id, f"{CORPUS_MOUNT}/data/tasks/{task_id}"))
    mounts.append(Mount(outputs, OUTPUTS_MOUNT, writable=True))
    return mounts


def graded_mounts(*, graded: Path, task_id: str, outputs: Path) -> List[Mount]:
    """What the grading container sees: the answers, and the end state to grade against them.

    The same shape as :func:`served_mounts` and for a different reason. Nothing here ever runs a
    line an agent wrote, so this is not a boundary; it is the mount set that makes the grader's
    view exist at all, because the answers and the episode's output tree live in two different
    places on the host and the evaluator wants a root and an experiment."""
    mounts = _shared_mounts(graded / "data", GRADED_MOUNT)
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
    #: Whether this worker's persistence is known to be complete. A timeout or a broken frame
    #: interrupts a command that upstream ends with its own save, and the pinned saver clears its
    #: destination and writes several pieces in sequence, so stopping it can leave a tree that is
    #: stable and partial. Confirmed absence proves that writing stopped, not that it finished.
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
            opening = frames.frame(_SPAWN_TIMEOUT_SECONDS, expect=_READY_FRAME)
        except (TimeoutError, EOFError, ValueError, FramingError) as exc:
            # The container first, the local process and its pipes whatever that did. Removal can
            # raise on a control timeout, and everything after it used to be skipped: no worker
            # was returned, so nothing else was going to release the pipes, and the ordinary sweep
            # skips a labelled container whose parent is alive.
            _release(process, name)
            raise WorkerError(
                f"the appworld worker container never became ready ({type(exc).__name__}: {exc}); "
                f"waited {_SPAWN_TIMEOUT_SECONDS:.0f}s"
            ) from exc
        if not opening.get("ready"):
            _release(process, name)
            raise WorkerError(f"the appworld worker answered {opening!r} instead of ready")
        return cls(root=root, process=process, container=name, frames=frames)

    def settle(self, timeout: float) -> bool:
        """Wait, bounded, for whatever call is running to finish, and say whether it did.

        **A terminal may overtake an ordinary call**, which the serve layer does on purpose: a
        deadline has to be able to end an episode whose block is not coming back. What must not
        follow is stopping the container while upstream is in the middle of the save it ends every
        block with, because that leaves a tree that is stable and partial and a grade taken over
        it is a grade of half a save.

        So finalization asks first. The lock is held for the length of a call, so acquiring it is
        the fact that no call is in flight; failing to acquire it inside the bound is the fact that
        one is, and the caller poisons rather than stopping."""
        if self.lock.acquire(timeout=max(0.0, timeout)):
            self.lock.release()
            return True
        return False

    def _stop_after_failure(self) -> None:
        """Remove the container, and say nothing untrue about whether it went.

        ``closed`` means one thing: the daemon has confirmed this container is gone. A failure
        path that set it on a best-effort removal marked the container absent without asking, and
        finalization's own gate then returned early and graded a tree the container might still
        have been writing to. So this asks, and hands the container to the sweep when the answer
        does not come.

        **And it drops this process's own half either way.** It did not, and `close` returns at
        once on a worker already marked closed, so the only wait and the only descriptor close in
        this class were skipped for the whole life of a worker that failed: every timeout and every
        broken frame left an attached `docker run` client and a pipe pair behind, for a run that
        may serve hundreds of episodes. The container is somebody else's to worry about when the
        daemon will not answer; the client and the pipes are nobody's but this one's."""
        try:
            container.remove(self.container, confirm=True)
            self.closed = True
        except container.DockerError:
            self.disown()
        finally:
            self._release_local()

    def _release_local(self) -> None:
        """End the local ``docker run`` client and drop the pipe pair, and never raise.

        This process's own half of a worker, which is separate from the container's removal
        because the two fail independently and only one of them is anybody else's business.
        Killing the client does not stop a container and never did; what it does is stop this
        process holding a child and two descriptors for a world that is over.

        Idempotent, because every path out of a worker ends here and some of them arrive twice."""
        if self.process.poll() is None:
            self.process.kill()
            try:
                self.process.wait(timeout=_CLOSE_SECONDS)
            except subprocess.TimeoutExpired:
                pass
        _close_pipes(self.process)

    def disown(self) -> None:
        """Hand this container to the sweep, by name, because nothing here can remove it.

        The sweep skips containers whose parent is alive, which is right for the ordinary case and
        wrong for this one: a long-lived serving process that could not remove a container has no
        later chance to try, and the container holds a writable mount until the process exits.
        Writing the name where the sweep also looks is what gives it a second owner."""
        container.disowned(self.container)

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
                _send(
                    self.process,
                    {"id": identifier, "command": command, "body": body},
                    deadline=deadline,
                )
                while True:
                    answer = self.frames.frame(
                        max(0.0, deadline - time.monotonic()), expect=_ANSWER_FRAME
                    )
                    if answer.get("id") == identifier:
                        break
                    # An answer to a command nobody is waiting for. Only reachable when a worker
                    # was reused after a timeout, which it is not; kept because reading a frame
                    # and hoping is exactly the failure this is here to prevent.
            except FramingError as exc:
                # Fatal, and the container goes with it: the stream is out of position, and what
                # put it there is reachable from inside the interpreter that runs agent code.
                self.poisoned = f"the protocol was broken during {command!r}: {exc}"
                self._stop_after_failure()
                raise WorkerError(f"the appworld worker: {self.poisoned}") from exc
            except (BrokenPipeError, EOFError) as exc:
                self.poisoned = f"the container stopped during {command!r}: {exc}"
                raise WorkerError(f"the appworld worker container {self.poisoned}") from exc
            except TimeoutError as exc:
                self.poisoned = (
                    f"{command!r} did not finish within {_CALL_TIMEOUT_SECONDS:.0f}s ({exc}) and "
                    "is still running, so no later frame on this pipe can be trusted"
                )
                # Stopped, not merely disowned. Poisoning the handle stops this process using the
                # worker; it does nothing about the work, which goes on holding a cpu and a
                # writable mount. The other arm of a pair is a sibling container on the same host,
                # so a runaway left running is a difference the treatment did not make.
                #
                # **And it does not claim the removal worked.** `closed` means one thing: the
                # daemon has confirmed this container is gone. A timeout that set it marked the
                # container absent on a best-effort removal that ignores an ordinary nonzero
                # status, and finalization's own gate then returned early and graded a tree the
                # timed-out command might still have been writing to. Unusable and absent are two
                # facts; this sets the first and attempts the second.
                self._stop_after_failure()
                raise WorkerError(f"the appworld worker: {self.poisoned}") from exc
            if "error" in answer:
                # **Poisoned, because what it did before it failed is unknown.** Upstream returns
                # an agent's own exceptions as output, so an error on this channel is the worker's
                # own handling going wrong, and every command ends with a save this cannot say
                # happened.
                #
                # **Under the lock that read the answer, which it was not.** The lock is the whole
                # of what `settle` asks: acquiring it is the fact that no call is in flight. This
                # branch ran after the `with` had released it, so between the release and the
                # assignment there was an interval in which the worker was neither busy nor
                # poisoned. A terminal is deliberately allowed to overtake an ordinary call, so
                # finalization could land exactly there: `settle` returned true, the poison read
                # empty, and the container was removed and its tree graded on the strength of a
                # command that had just failed inside the world. Both facts are published while
                # the lock still holds, so nothing observes one without the other.
                self.poisoned = f"{command!r} failed inside the world: {answer['error']}"
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

        **Two callers, two failure modes, and the bounds compose.** ``confirm`` is finalization's,
        and it is the difference between a removal and a fact: finalization removes this container
        precisely so that nothing can write to the tree it is about to grade, so a removal the
        daemon will not confirm raises, and the worker is *not* marked closed so teardown tries
        again. Teardown's own call must never raise, whatever Docker does, because a teardown that
        raises abandons the pipes and the directories it was there to release. The whole of this
        is bounded well under the serve layer's own sixty seconds: three control calls of ten
        seconds each and one process wait, never both waits and never a second grace period."""
        if self.closed:
            return
        try:
            gone = container.remove(self.container, confirm=confirm)
        except container.DockerError:
            if confirm:
                # The caller is about to grade what this container could still be writing to, and
                # it may not. Left unclosed on purpose: teardown will come back to it.
                raise
            # Teardown's path. What is left here is this process's own handles, and dropping them
            # is not optional; the container is handed to the sweep by name so that a removal
            # nobody could confirm is still somebody's, even while this parent is alive.
            self.disown()
        else:
            # **Only what the daemon confirmed.** A nonzero stop or removal used to reach this
            # branch and be recorded as a removal, after which nothing tried again and the ordinary
            # sweep skipped the container because its parent was alive. `remove` says whether the
            # container is known to be gone, and when it is not it has already written the name
            # where the sweep will find it.
            self.closed = bool(gone)
        self._release_local()


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
        # One deadline over the write and the read together, for the reason `_send` gives: a
        # container that never reads its request is the same wait as one that never answers.
        deadline = time.monotonic() + timeout
        _send(process, {"body": body}, deadline=deadline)
        answer = frames.frame(max(0.0, deadline - time.monotonic()), expect=_ANSWER_FRAME)
    except TimeoutError as exc:
        _release(process, name)
        raise WorkerError(f"{what} did not finish within {timeout:.0f}s; it was killed") from exc
    except (BrokenPipeError, EOFError, ValueError, FramingError, WorkerError) as exc:
        _release(process, name)
        raise WorkerError(f"{what} failed: {type(exc).__name__}: {exc}") from exc
    else:
        try:
            process.wait(timeout=_CLOSE_SECONDS)
        except subprocess.TimeoutExpired:
            _release(process, name)
        _close_pipes(process)
    if "error" in answer:
        raise WorkerError(f"{what} failed: {answer['error']}")
    return answer["output"]


def seed(*, root: Path, source_dbs: Path, into: Path, rows: Dict[str, Any]) -> Any:
    """Write one task's seeded database log, in a container of its own.

    ``source_dbs`` is the task's own input databases in the downloaded corpus and ``into`` is the
    file being written inside a staging directory. ``root`` is the derived root, mounted because
    upstream's model layer resolves an app's base database under ``APPWORLD_ROOT`` and a load with
    no root reaches for one under the working directory.

    Three mounts, one of them writable, and that one is the staging directory. This container runs
    no agent code, so its mount set is a matter of keeping the seam narrow rather than a boundary:
    what it can see is one task's inputs and the tree it is writing into. Built from the same
    named list as the served set, so the three mount builders in this module cannot come to
    disagree about what the shared half of a corpus is."""
    shared = _shared_mounts(root / "data", CORPUS_MOUNT)
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


class SnapshotError(RuntimeError):
    """The stopped output tree is not something that may be graded.

    Its own type because it is an episode-level failure with a cause worth naming: the tree the
    world left behind holds something that is not a plain file, and a grader that opened it would
    be resolving a path the agent chose inside a namespace that also holds the answers."""


#: The name the world writes its save record under, mirrored from the worker so this module can
#: read it without importing a file that only exists inside the image.
_SAVE_MANIFEST = "save.manifest"

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


#: How much of one database log verification will read before it refuses. Streaming, so this is a
#: buffer rather than a file size: the tree may approach a gibibyte, and reading one of its files
#: whole would put that allocation in the serving process, where the container's memory limit
#: does not reach.
_VERIFY_CHUNK = 1 << 20

#: The longest single record verification will accept. A log is one JSON object per line, so a
#: line longer than this is not a record this port wrote.
_VERIFY_MAX_RECORD = 8 << 20

#: The manifest's own bounds, which are not the tree's. It is written inside a tree an episode
#: could fill to a gibibyte, and it is read into the serving process, where the container's memory
#: limit does not reach: a name and a length for a few dozen logs is a few kilobytes, so sixty-four
#: is generous and a megabyte would not be. The entry cap bounds the loop that reads it as well as
#: the allocation.
_MANIFEST_MAX_BYTES = 64 * 1024
_MANIFEST_MAX_FILES = 256


def verify_snapshot(
    snapshot: Path,
    *,
    task_id: str,
    expected: Path,
    blocks: int,
    stop: "Optional[threading.Event]" = None,
) -> None:
    """Refuse a snapshot that is not the whole of the save the host asked for.

    **A newline is not the end of a save.** Upstream clears the database directory and writes the
    logs one after another, one JSON record per line, so an interruption leaves every expected
    filename in place and a syntactically perfect tail with a suffix of state simply missing. No
    property of the bytes distinguishes that from a finished save, which is why this compares them
    against something written *after* the save returned: the world records the block it was
    serving and the length of every log it produced, and the host compares the block against the
    one it sent. A save that never finished leaves the manifest of the block before it.

    That manifest comes from the process that runs agent-authored code, so what it establishes is
    completeness against an *interruption* rather than against an adversary. Against an adversary
    the guarantee is the one that was always there: the tree is what is graded, an episode already
    controls its own tree, and nothing it can write there improves its grade.

    **Bounded and cancellable, like the copy before it.** A permitted tree may approach a
    gibibyte, so nothing here reads a file whole: the manifest is read through a handle under a
    cap of its own, each log is streamed in chunks with a byte budget and a record cap, and the
    deadline and the abandon flag are checked between chunks.

    **The manifest names nothing; it only answers about names the host already had.** It is
    written inside the tree the episode could write, so every key in it is text an episode chose.
    What is walked is the trusted set — the basenames of the served task's own input logs, read
    off a tree no episode can reach — and the manifest is consulted by those names and never
    iterated. A key that is not a plain ``.jsonl`` basename refuses the episode rather than being
    skipped, because a manifest naming ``../`` or an absolute path is not a save record with an
    extra field in it; and a name the trusted set lacks is never joined to a path, so no path
    outside the stopped snapshot is ever built, let alone opened."""
    began = time.monotonic()
    dbs = snapshot / "tasks" / task_id / "dbs"
    if not dbs.is_dir():
        raise SnapshotError(
            f"the episode's snapshot has no databases at tasks/{task_id}/dbs, so what it "
            "persisted is not a world to grade"
        )
    manifest = _save_record(dbs.parent / _SAVE_MANIFEST, began=began, stop=stop)
    if int(manifest.get("block", -1)) != int(blocks):
        raise SnapshotError(
            f"the episode's snapshot records the save after block {manifest.get('block')!r} while "
            f"the host ran {blocks}; the save for the last block did not finish"
        )
    listed = _save_lengths(manifest["files"])
    wanted = sorted(path.name for path in expected.iterdir() if path.suffix == ".jsonl")
    missing = [name for name in wanted if name not in listed]
    if missing:
        raise SnapshotError(
            f"the episode's snapshot is missing {', '.join(missing)}; upstream's saver clears its "
            "destination and writes the logs in sequence, so this is half of a save"
        )
    for name in wanted:
        if stop is not None and stop.is_set():
            raise SnapshotError("the snapshot check was abandoned before it finished")
        if time.monotonic() - began > _SNAPSHOT_SECONDS:
            raise SnapshotError(
                f"checking the episode's snapshot took longer than {_SNAPSHOT_SECONDS:.0f}s"
            )
        # `name` came from the served tree rather than from the manifest, and it is one path
        # component: this join cannot leave `dbs`, whatever the manifest said.
        path = dbs / name
        try:
            status = path.lstat()
        except OSError as exc:
            raise SnapshotError(f"the episode's snapshot is missing {name}: {exc}") from exc
        if not stat.S_ISREG(status.st_mode):
            # Unreachable through `snapshot_outputs`, which refuses a tree holding anything else,
            # and asserted here anyway: what follows opens this path and reads it to the end, and
            # a device or a fifo is a read with no bound that a deadline between chunks cannot
            # reach, because the first one never returns.
            raise SnapshotError(
                f"the episode's snapshot has {name} as something other than a plain file, which "
                "is not a database log and is not a thing this reads"
            )
        if status.st_size != listed[name]:
            raise SnapshotError(
                f"the episode's snapshot has {name} at {status.st_size} bytes where the save "
                f"that produced it recorded {listed[name]}; the save did not finish"
            )
        _verify_records(path, began=began, stop=stop)


def _save_record(
    recorded: Path, *, began: float, stop: "Optional[threading.Event]"
) -> Dict[str, Any]:
    """Read the save record, under a cap of its own and with the same clock as everything else.

    ``read_text()[:cap]`` reads the file and *then* takes the slice, so an episode that left a
    manifest approaching the tree's own gibibyte had that gibibyte allocated and decoded in the
    serving process before a single bound was consulted. The container's memory limit does not
    reach a host allocation, and cancelling the ``to_thread`` await does not interrupt one. So the
    handle is opened and at most a cap and one byte are read, in chunks, with the deadline and the
    abandon flag read between them; the extra byte is what tells an oversized manifest from one
    that merely fills the cap."""
    blocks: List[bytes] = []
    seen = 0
    try:
        with recorded.open("rb") as handle:
            while seen <= _MANIFEST_MAX_BYTES:
                if stop is not None and stop.is_set():
                    raise SnapshotError("the snapshot check was abandoned before it finished")
                if time.monotonic() - began > _SNAPSHOT_SECONDS:
                    raise SnapshotError(
                        f"checking the episode's snapshot took longer than "
                        f"{_SNAPSHOT_SECONDS:.0f}s"
                    )
                block = handle.read(min(_VERIFY_CHUNK, _MANIFEST_MAX_BYTES + 1 - seen))
                if not block:
                    break
                seen += len(block)
                blocks.append(block)
    except OSError as exc:
        raise SnapshotError(
            f"the episode's snapshot carries no record of the save that made it ({exc}); a save "
            "that finished writes one, so this did not finish"
        ) from exc
    if seen > _MANIFEST_MAX_BYTES:
        raise SnapshotError(
            f"the episode's snapshot carries a save record of more than {_MANIFEST_MAX_BYTES} "
            "bytes, which is not a record this port wrote"
        )
    try:
        manifest = json.loads(b"".join(blocks).decode(errors="replace"))
    except ValueError as exc:
        raise SnapshotError(
            f"the episode's snapshot carries no record of the save that made it ({exc}); a save "
            "that finished writes one, so this did not finish"
        ) from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise SnapshotError("the episode's snapshot carries a save record of the wrong shape")
    return manifest


def _save_lengths(files: Mapping[str, Any]) -> Dict[str, int]:
    """The lengths a save record claims, keyed by name, or a refusal.

    Every key is proved to be one plain ``.jsonl`` basename here, and that is the whole of what
    this port will accept in it. Nothing downstream joins one of these names to a path — the walk
    uses the served tree's own names — so this is not what stops the escape; what it stops is the
    quieter thing, which is a save record that says something this port's saver would never write
    being read as a save record with an extra entry."""
    if len(files) > _MANIFEST_MAX_FILES:
        raise SnapshotError(
            f"the episode's snapshot records {len(files)} files, and no world this port serves "
            f"writes more than {_MANIFEST_MAX_FILES} database logs"
        )
    lengths: Dict[str, int] = {}
    for name, size in files.items():
        name = str(name)
        if (
            name != os.path.basename(name)
            or name in (".", "..", "")
            or not name.endswith(".jsonl")
            or os.path.isabs(name)
        ):
            raise SnapshotError(
                f"the episode's snapshot records a saved file as {name!r}, which is not the name "
                "of a database log; a save record names the logs it wrote and nothing else"
            )
        try:
            lengths[name] = int(size)
        except (TypeError, ValueError) as exc:
            raise SnapshotError(
                f"the episode's snapshot records {name} at {size!r}, which is not a length"
            ) from exc
    return lengths


def _verify_records(path: Path, *, began: float, stop: "Optional[threading.Event]") -> None:
    """Read one log in chunks and refuse one that does not end on a whole record.

    Streaming rather than whole: a permitted file may be most of a gibibyte and reading it into
    the serving process is an allocation the container's memory limit does not cover. What is kept
    is the tail of the current line and nothing else."""
    tail = b""
    with open(path, "rb") as handle:
        while True:
            if stop is not None and stop.is_set():
                raise SnapshotError("the snapshot check was abandoned before it finished")
            if time.monotonic() - began > _SNAPSHOT_SECONDS:
                raise SnapshotError(
                    f"checking the episode's snapshot took longer than {_SNAPSHOT_SECONDS:.0f}s"
                )
            chunk = handle.read(_VERIFY_CHUNK)
            if not chunk:
                break
            tail = (tail + chunk).rsplit(b"\n", 1)[-1]
            if len(tail) > _VERIFY_MAX_RECORD:
                raise SnapshotError(
                    f"the episode's snapshot has a record in {path.name} longer than "
                    f"{_VERIFY_MAX_RECORD} bytes, which is not a record this port wrote"
                )
    if tail.strip():
        raise SnapshotError(
            f"the episode's snapshot has {path.name} ending mid-line, so the save that was "
            "writing it did not finish"
        )


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

    It also reads the filing, digests the databases and picks up the generator digest the world
    wrote, off that same tree and in that same process. That is what makes the scored state and
    the graded state one state rather than two observations of a live world that happened to
    agree, and it is why nothing here is asked of the process that ran the agent's code.

    ``outputs`` is a snapshot (see :func:`snapshot_outputs`), not the tree the world wrote: the
    grader's namespace holds the answers, so what it is given has to be regular files and nothing
    else. ``filing`` names what to look for: the
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
    "DERIVATION_VERSION",
    "control_file",
    "control_home",
    "corpus_digest",
    "corpus_snapshot",
    "runtime_digest",
    "derived_root",
    "episode_outputs",
    "episode_view",
    "episodes_home",
    "ensure_corpus",
    "ensure_image",
    "grade",
    "graded_mounts",
    "graded_root",
    "private_home",
    "SnapshotError",
    "seed",
    "served_mounts",
    "snapshot_outputs",
    "verify_snapshot",
    "stamp_cache",
    "task_ids",
    "task_specs",
]
