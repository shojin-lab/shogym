"""On-demand provisioning of the pinned ORCA-bench dataset from the Harbor hub.

The dataset (``orca-bench/orca-bench``, CC BY 4.0, 755 tasks, ~192 MB) is **never vendored**:
it is downloaded once into ``~/.cache/shogym/orca_bench/<revision-hash>/`` and reused after
that, the same fetch-and-cache shape the other ports use for their upstream sources. Nothing
here needs the ``harbor`` CLI or any third-party client. The hub is a PostgREST + object-store
API and this module speaks it with ``urllib``:

1. resolve the **pinned dataset revision** (:data:`DATASET_CONTENT_HASH`) to its row;
2. list that revision's 755 ``(task name, content hash)`` pairs;
3. fetch each task's ``dist.tar.gz`` and extract it to ``<dataset dir>/<task name>/``.

A task directory exists only once its extraction has been published by a single atomic rename,
so a partially-downloaded task is never visible to a reader, and a killed provisioner leaves
only a reclaimable ``.dl-*`` staging directory. Concurrent provisioners are serialized by the
same ``flock`` policy the upstream-source provisioner uses (see :mod:`shogym.envs._upstream`):
a filesystem that cannot lock degrades to redundant-but-correct work with one warning, and every
other lock error still propagates.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import socket
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Dict, Iterator, List, Mapping, Optional

# The flock helpers are shared deliberately rather than copied: they encode a *policy* (which
# errnos mean "this filesystem cannot lock" and may degrade, which stay fatal, and never deleting
# staging residue that cannot be locked). A second copy of that policy would be free to drift from
# the first, and the two would disagree exactly on the mounts where it matters.
from shogym.envs._upstream import _locked, _sweep_download_residue

# ----- upstream pin (fidelity provenance) -----

DATASET = "orca-bench/orca-bench"
DATASET_ORG, DATASET_NAME = DATASET.split("/")
# The pinned dataset revision. The hub also serves a mutable `latest` tag; pinning the revision's
# content hash is what makes a cold cache reproducible (revision 1 is a different 755-task build).
# Re-pinning is a deliberate act: it changes what every published number means. The measurements
# and the port design behind this pin are in issue #77.
DATASET_REVISION = 2
DATASET_CONTENT_HASH = "1ef729757d4974ffe4e835d541c601f957975edf8c93ef02eec97e26d3069b93"
DATASET_LICENSE = "CC BY 4.0"
DATASET_TASK_COUNT = 755

# The Harbor hub's package registry. The publishable key is a **public** anonymous read key (it
# ships in the `harbor` client and authorizes nothing but public reads), so it is a constant here
# rather than a credential. No key of the user's is ever read or sent.
HUB_URL = "https://ofhuhcpkvzjlejydnvyd.supabase.co"
HUB_ANON_KEY = "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"
HUB_WEBSITE = "https://hub.harborframework.com"
# The object-store bucket task archives are published in.
_STORAGE_BUCKET = "packages"

# What a cached task must contain to count as one. The first three are what the index and
# `describe` read now; the rest are what a graded episode needs, and they are included on purpose:
# the cache is written once and read by both phases, so an archive that arrived without its
# verifier would otherwise sit there looking fine until phase 2 tried to grade with it, long after
# the ~192 MB it came in was paid for. Every one of the seven is present in all 755 tasks of the
# pinned revision, so requiring them cannot reject a task the benchmark considers valid.
# Deliberately NOT required: `solution/` (the oracle, which this port never runs) and
# `tests/rubrics/` (present only for incident tasks, absent for the 138 controls).
REQUIRED_TASK_FILES = (
    "task.toml",
    "instruction.md",
    "environment/Dockerfile",
    "environment/docker-compose.yaml",
    "tests/test.sh",
    "tests/check_prediction.py",
    "tests/expected.json",
)

_REQUEST_TIMEOUT_SECONDS = 120.0
# Task archives are small (30 KB - 1 MB) and the fetch is latency-bound, so a handful of workers
# turns a ~10 minute serial download into well under a minute. Deliberately modest: the hub is a
# shared public service and this is a cold-cache cost paid once per host.
_DOWNLOAD_WORKERS = 8
# PostgREST caps a response at 1000 rows; the task listing is paged through this window.
_PAGE_SIZE = 1000

# The per-task publish lock (see `_publish_lock`). The section it guards is a few stats and two
# renames, so a wait this long means the holder died inside it rather than that it is slow.
_PUBLISH_LOCK_TIMEOUT_SECONDS = 60.0
# Recorded in a lock's token so a waiter can tell a holder of its own machine (whose pid it can
# ask about) from one on another machine sharing the cache (whose pid means nothing here).
_HOSTNAME = socket.gethostname()
_PUBLISH_LOCK_POLL_SECONDS = 0.02


class DatasetUnavailableError(RuntimeError):
    """The pinned dataset could not be listed or fetched (network, hub, or a moved revision)."""


# ----- cache location -----


def cache_dir() -> Path:
    """The root the pinned dataset is cached under.

    Honors ``SHOGYM_ORCA_BENCH_DATA_DIR`` first, then ``SHOGYM_CACHE`` (the shared cache root
    every port respects), else ``~/.cache/shogym/orca_bench``."""
    explicit = os.environ.get("SHOGYM_ORCA_BENCH_DATA_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base).expanduser() if base else Path.home() / ".cache" / "shogym"
    return root / "orca_bench"


def dataset_dir(root: Optional[Path] = None) -> Path:
    """Where the pinned revision's task directories live: ``<cache root>/<revision hash>``.

    Keyed by the revision's content hash so a re-pin lands beside the old tree instead of
    silently mixing two dataset builds in one directory."""
    return (root or cache_dir()) / DATASET_CONTENT_HASH


# ----- hub client (stdlib only) -----


@dataclass(frozen=True)
class HubTask:
    """One task package in the pinned dataset revision: its name and its pinned content hash."""

    name: str
    content_hash: str

    @property
    def archive_path(self) -> str:
        """The object-store path of this task version's ``dist.tar.gz``, inside the bucket.

        Note the leading ``packages/`` is part of the path *within* the ``packages`` bucket, not
        the bucket itself. The registry records it that way, so a fetch names it twice."""
        return f"packages/{DATASET_ORG}/{self.name}/{self.content_hash}/dist.tar.gz"


def _hub_request(url: str, *, data: Optional[bytes] = None, accept_json: bool = True):
    headers = {"apikey": HUB_ANON_KEY, "Authorization": f"Bearer {HUB_ANON_KEY}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if accept_json:
        headers["Accept"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers)


def _hub_json(path: str, *, params: Optional[Dict[str, str]] = None, body: Optional[dict] = None):
    """GET (or POST, when ``body`` is given) a JSON endpoint on the hub."""
    url = f"{HUB_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    data = json.dumps(body).encode("utf-8") if body is not None else None
    try:
        with urllib.request.urlopen(
            _hub_request(url, data=data), timeout=_REQUEST_TIMEOUT_SECONDS
        ) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise DatasetUnavailableError(_unreachable_message(exc)) from exc


def _unreachable_message(exc: BaseException) -> str:
    """One actionable line, rather than a stack of urllib internals."""
    return (
        f"could not reach the Harbor hub at {HUB_URL} ({type(exc).__name__}: {exc}). The "
        f"{DATASET} dataset ({DATASET_TASK_COUNT} tasks, ~192 MB) is downloaded on demand into "
        f"{cache_dir()} the first time the env is constructed. Check network access, or point "
        "SHOGYM_ORCA_BENCH_DATA_DIR at a warm cache. Offline tests pass `dataset_dir=` and need "
        "neither."
    )


def resolve_dataset_version() -> str:
    """Resolve the pinned revision to its hub id, asserting it is the revision we pinned."""
    rows = _hub_json(
        "/rest/v1/dataset_version",
        params={
            "select": "id,revision,content_hash,package:package_id!inner(name,type,org:org_id!inner(name))",
            "content_hash": f"eq.{DATASET_CONTENT_HASH}",
            "package.name": f"eq.{DATASET_NAME}",
            "package.type": "eq.dataset",
            "package.org.name": f"eq.{DATASET_ORG}",
        },
    )
    if not rows:
        raise DatasetUnavailableError(
            f"{DATASET} revision {DATASET_REVISION} (content hash {DATASET_CONTENT_HASH}) is not "
            f"published on the hub any more. The pin moved: re-pin deliberately (and re-check the "
            f"task count) rather than silently following `latest`. See {HUB_WEBSITE}/datasets."
        )
    row = rows[0]
    if int(row.get("revision", -1)) != DATASET_REVISION:
        raise DatasetUnavailableError(
            f"{DATASET} content hash {DATASET_CONTENT_HASH} now resolves to revision "
            f"{row.get('revision')}, not the pinned {DATASET_REVISION}"
        )
    return str(row["id"])


def list_dataset_tasks() -> List[HubTask]:
    """List the pinned revision's task packages, in the hub's own order.

    Authenticated against the pinned manifest before it is returned, so every later step works
    from identities the pin vouches for rather than from whatever the endpoint said (see
    :func:`_authenticate_listing`). The order is not the port's task order
    (:func:`shogym.envs.orca_bench.tasks.load_index` sorts by name for that); it is only how the
    rows are paged."""
    version_id = resolve_dataset_version()
    tasks: List[HubTask] = []
    offset = 0
    while True:
        rows = _hub_json(
            "/rest/v1/dataset_version_task",
            params={
                "select": "task_version:task_version_id(content_hash,package:package_id(name,org:org_id(name)))",
                "dataset_version_id": f"eq.{version_id}",
                "order": "task_version_id",
                "limit": str(_PAGE_SIZE),
                "offset": str(offset),
            },
        )
        for row in rows:
            version = row.get("task_version") or {}
            package = version.get("package") or {}
            name = package.get("name")
            content_hash = version.get("content_hash")
            if name and content_hash:
                tasks.append(HubTask(name=str(name), content_hash=str(content_hash)))
        if len(rows) < _PAGE_SIZE:
            break
        offset += _PAGE_SIZE
    if len(tasks) != DATASET_TASK_COUNT:
        raise DatasetUnavailableError(
            f"{DATASET} revision {DATASET_REVISION} listed {len(tasks)} tasks, expected "
            f"{DATASET_TASK_COUNT}; the pinned revision changed shape"
        )
    return _authenticate_listing(tasks)


# The pinned task manifest: every task name in the revision and the content hash of its package,
# one per line, sorted by name. This is the port's own anchor for the *set*, and it exists because
# the hub's dataset-version content hash cannot be reproduced from the client: `harbor` never
# sends one (`RegistryDB.publish_dataset_version` passes tasks and files, and the server returns
# the hash), and the client-side `DatasetManifest.compute_content_hash` does not reproduce the
# recorded value for this revision. Without an anchor of our own, `DATASET_CONTENT_HASH` would
# only *look up* the version row and every per-archive check would then trust whatever list the
# live endpoint returned: a truncated or duplicated listing would provision a truncated benchmark
# and report success. The file is not dataset content (no instructions, no metadata, no answers),
# it is a pin, in the shape frontier_bench pins its vendored tasks.
#
# Every hash in it was RECOMPUTED from the real package bytes with `content_hash` below, not
# copied from the listing, and the result equals the registry's recorded value for all 755 tasks.
_MANIFEST_PATH = Path(__file__).resolve().parent / "task_manifest.txt"
DATASET_MANIFEST_SHA256 = "b9e8dce68788f20cf75c8cb1e7ada81d8c22dbd25c97545549fd1adf95168abc"


@functools.lru_cache(maxsize=1)
def pinned_manifest() -> Mapping[str, str]:
    """The pinned ``{task name: content hash}`` map, checked against its own digest.

    The digest guards the file itself, so a manifest edited by accident (or on purpose) fails
    closed here rather than silently redefining what this port considers to be the benchmark.

    Read-only, and cached, so every caller shares one object: this is the process's trust anchor,
    consulted by the listing check, the warm predicate, content authentication, the residue check
    and indexing. Handing out a mutable dict would make an ordinary inspection (popping entries to
    derive a subset) a way to redefine all of them at once, without the file being re-read. The
    proxy prevents that rather than trusting callers not to do it."""
    raw = _MANIFEST_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATASET_MANIFEST_SHA256:
        raise DatasetUnavailableError(
            f"the pinned task manifest {_MANIFEST_PATH} hashes to {digest}, not the pinned "
            f"{DATASET_MANIFEST_SHA256}; it has been modified"
        )
    manifest: Dict[str, str] = {}
    for line in raw.decode("utf-8").splitlines():
        name, _, content_hash = line.partition(" ")
        manifest[name] = content_hash
    if len(manifest) != DATASET_TASK_COUNT:
        raise DatasetUnavailableError(
            f"the pinned task manifest holds {len(manifest)} tasks, expected {DATASET_TASK_COUNT}"
        )
    # The only reference to the underlying dict leaves with this frame, so the proxy is the whole
    # of what anyone can reach.
    return MappingProxyType(manifest)


def _authenticate_listing(tasks: List[HubTask]) -> List[HubTask]:
    """Check a live listing against the pin: the same identities, each with the pinned hash.

    The count alone proves nothing. A listing of 755 copies of one row passes it, provisions one
    directory, and leaves a caller believing it holds the benchmark. So the names are compared as
    a set, duplicates are rejected outright, and every hash must be the pinned one, because that
    hash is what every downloaded archive is then authenticated against."""
    manifest = pinned_manifest()
    seen: Dict[str, str] = {}
    for task in tasks:
        if task.name in seen:
            raise DatasetUnavailableError(
                f"{DATASET} revision {DATASET_REVISION} listed task {task.name} more than once; "
                "the listing does not describe the pinned revision"
            )
        seen[task.name] = task.content_hash
    unexpected = sorted(set(seen) - set(manifest))
    absent = sorted(set(manifest) - set(seen))
    if unexpected or absent:
        raise DatasetUnavailableError(
            f"{DATASET} revision {DATASET_REVISION} listed a different task set than the pin: "
            f"{len(unexpected)} unexpected (e.g. {unexpected[:3]}), {len(absent)} missing "
            f"(e.g. {absent[:3]})"
        )
    for name, content_hash in sorted(seen.items()):
        if content_hash != manifest[name]:
            raise DatasetUnavailableError(
                f"{DATASET} revision {DATASET_REVISION} lists task {name} at content hash "
                f"{content_hash}, but the pin says {manifest[name]}. Refusing to fetch bytes the "
                "pin does not vouch for."
            )
    return tasks


def _archive_bytes(task: HubTask) -> bytes:
    """Fetch one task version's archive from the hub's object store.

    One request on the happy path. The archive path is derived from the pinned content hash,
    because the hub publishes every task version at that deterministic path; the registry is
    consulted **only** when that returns 404, i.e. only if a future layout change moved it.
    Asking first would make a transiently unavailable registry fail a fetch that would have
    worked, without issuing a single valid archive request, and would add one registry call per
    task to every cold 755-task provision."""
    data = _fetch_archive(task.archive_path)
    if data is not None:
        return data
    resolved = _resolved_archive_path(task)
    if resolved is not None and resolved != task.archive_path:
        data = _fetch_archive(resolved)
        if data is not None:
            return data
    raise DatasetUnavailableError(
        f"no archive published for task {task.name}@{task.content_hash} in {DATASET}"
    )


def _fetch_archive(path: str) -> Optional[bytes]:
    """GET one object from the archive bucket; ``None`` iff it is a 404 (nothing there)."""
    url = f"{HUB_URL}/storage/v1/object/{_STORAGE_BUCKET}/{path}"
    try:
        with urllib.request.urlopen(
            _hub_request(url, accept_json=False), timeout=_REQUEST_TIMEOUT_SECONDS
        ) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise DatasetUnavailableError(_unreachable_message(exc)) from exc
    except (urllib.error.URLError, OSError) as exc:
        raise DatasetUnavailableError(_unreachable_message(exc)) from exc


def _resolved_archive_path(task: HubTask) -> Optional[str]:
    """Ask the registry where this task version's archive really is (the authority).

    The 404 fallback only. This response also carries the task's full ``[metadata]`` and its
    instruction, none of which is read here, which is a second reason to keep the call off the
    normal provisioning path rather than on it."""
    row = _hub_json(
        "/rest/v1/rpc/resolve_task_version",
        body={
            "p_org": DATASET_ORG,
            "p_name": task.name,
            "p_ref": f"sha256:{task.content_hash}",
        },
    )
    if not isinstance(row, dict):
        return None
    path = row.get("archive_path")
    return str(path) if path else None


# ----- provisioning -----


# ----- content hash (the registry's own algorithm) -----

# What the hub hashes when it publishes a task package: three optional single files plus every
# file under these directories, gitignore-filtered, sorted by path. Derived from the `harbor`
# publisher (`harbor.publisher.packager.Packager`) and checked against the registry's recorded
# value for all 755 tasks of the pinned revision.
_HASHED_FILES = ("task.toml", "instruction.md", "README.md")
_HASHED_DIRS = ("environment", "tests", "solution", "steps")
# The publisher's default ignore set, applied when a task ships no `.gitignore`. None of the
# pinned revision's tasks ships one; if a later revision does and it actually excludes a collected
# file, this hash and the registry's will disagree and the fetch fails closed, which is the right
# direction to be wrong in.
_IGNORED_NAMES = frozenset({".DS_Store"})
_IGNORED_SUFFIXES = (".pyc", ".swp", ".swo", "~")
_IGNORED_DIRS = frozenset({"__pycache__"})


def _hashed_files(task_dir: Path) -> List[Path]:
    """The files the publisher hashes, in the order it hashes them."""
    found: List[Path] = [task_dir / name for name in _HASHED_FILES]
    for directory in _HASHED_DIRS:
        found.extend(sorted((task_dir / directory).rglob("*")))
    keep: List[Path] = []
    for path in found:
        if not path.is_file():
            continue
        relative = path.relative_to(task_dir).as_posix()
        parts = relative.split("/")
        if _IGNORED_DIRS.intersection(parts[:-1]):
            continue
        if path.name in _IGNORED_NAMES or relative.endswith(_IGNORED_SUFFIXES):
            continue
        keep.append(path)
    keep.sort(key=lambda p: p.relative_to(task_dir).as_posix())
    return keep


def content_hash(task_dir: Path) -> str:
    """The registry's content hash for an extracted task package.

    ``sha256`` over ``<relative path>\0<sha256 of the file>\n`` for each collected file, in path
    order, which is what the hub records as a task version's ``content_hash``. Reimplemented here
    rather than taken on trust: this is the only thing that makes the revision pin a pin. The
    listing gives a hash per task and this port used it to *address* the archive; without
    recomputing it over what arrived, the cache held whatever the endpoint served, and phase 2
    executes that (the compose file, ``tests/test.sh``, the judge).
    """
    outer = hashlib.sha256()
    for path in _hashed_files(task_dir):
        relative = path.relative_to(task_dir).as_posix()
        inner = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                inner.update(chunk)
        outer.update(f"{relative}\0{inner.hexdigest()}\n".encode())
    return outer.hexdigest()


def missing_task_files(task_dir: Path) -> List[str]:
    """Which of :data:`REQUIRED_TASK_FILES` a cached task directory does not have.

    Empty means complete. Useful on its own for auditing a cache that predates this check."""
    return [rel for rel in REQUIRED_TASK_FILES if not (task_dir / rel).is_file()]


def _is_task_complete(dataset: Path, name: str) -> bool:
    """Whether a cached task carries the whole contract, not merely a ``task.toml``.

    An atomic rename proves nobody saw the extraction in progress; it proves nothing about what
    the archive contained. Presence of one file was the weaker claim, and it was the wrong one:
    a tree with only ``task.toml`` would be published and then trusted forever, failing later at
    whatever first read one of the files that never arrived."""
    return not missing_task_files(dataset / name)


@dataclass(frozen=True)
class _LockHolder:
    """Who is inside a publish lock, as its token records them."""

    host: str
    pid: int
    start: Optional[float]  # the holder process's start time, when it could be read cheaply
    token: str  # the file name that names this holder, and only this holder

    def __str__(self) -> str:
        return f"host={self.host} pid={self.pid}"


def _this_holder() -> str:
    """The identity written into a lock this process is taking."""
    return json.dumps(
        {"host": _HOSTNAME, "pid": os.getpid(), "start": _process_start(os.getpid())}
    )


def _process_start(pid: int) -> Optional[float]:
    """The process's start time, if it can be read cheaply. ``None`` means "cannot tell"."""
    try:
        import psutil
    except ImportError:  # pragma: no cover - psutil is a core dependency, but never required here
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 - any failure means "cannot tell", never "dead"
        return None


def _read_lock_holder(lock: Path) -> Optional[_LockHolder]:
    """Who holds ``lock``, or ``None`` if that cannot be established.

    ``None`` is not "nobody": it is "unknown", and an unknown holder is never broken."""
    try:
        entries = [entry for entry in lock.iterdir() if entry.is_file()]
    except OSError:
        return None
    if len(entries) != 1:
        return None
    try:
        recorded = json.loads(entries[0].read_text(encoding="utf-8"))
        host = str(recorded["host"])
        pid = int(recorded["pid"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    start = recorded.get("start")
    return _LockHolder(
        host=host,
        pid=pid,
        start=float(start) if isinstance(start, (int, float)) else None,
        token=entries[0].name,
    )


def _holder_is_provably_dead(holder: _LockHolder) -> bool:
    """Whether this holder can be shown to be gone. Anything short of a proof answers ``False``.

    Only a *same-host* holder can be judged at all: a pid from another machine says nothing here.
    That is not much of a restriction in practice, because this cache is per-user and lives under
    ``~/.cache``; a shared network cache is exactly the case where the answer should be "cannot
    tell", and it is.

    Signal zero asks the kernel whether the pid exists without touching the process. ``ESRCH``
    (``ProcessLookupError``) is the proof. A permission error means the pid exists and belongs to
    someone else, which is alive. Where a start time was recorded and can be read back, a
    disagreement means the pid was reused and the original holder is gone.
    """
    if holder.host != _HOSTNAME:
        return False
    try:
        os.kill(holder.pid, 0)
    except ProcessLookupError:
        return True
    except OSError:
        return False  # alive, or not ours to ask about
    if holder.start is not None:
        current = _process_start(holder.pid)
        if current is not None and abs(current - holder.start) > 1.0:
            return True  # the pid was recycled; the holder itself is gone
    return False


@contextlib.contextmanager
def _publish_lock(dataset: Path, name: str) -> Iterator[None]:
    """Hold every other publisher out of one task's directory, on any filesystem.

    :func:`_locked` is an ``flock``, and it degrades to nothing where the filesystem cannot
    provide one. That is the right trade for the *download* it guards, where the cost of losing
    exclusion is redundant work and the result is still correct. It is the wrong trade for the
    *publish*: deciding whether a destination needs replacing and then replacing it is a
    read-modify-write over a directory a peer may be publishing at the same instant, and acting on
    a decision that went stale in between destroys their work.

    Directory rename is atomic and conditional on every filesystem, including the ones ``flock``
    has already given up on, so it is the exclusion that is always available. A holder builds a
    directory containing its own **token** and renames it into place: the rename succeeds only
    onto a name that is free (absent, or an empty directory left by a releaser mid-exit), because
    a live lock always holds its token and rename refuses a non-empty target. So a lock arrives
    already stamped with its owner, with no window between taking it and owning it. Release is
    conditional on still being that owner (see :func:`_release_publish_lock`).

    **A lock is cleared only when its holder is provably dead**, never merely because it is old.
    A wall-clock lease cannot fence a holder that is still running: a filesystem operation can
    stall past any deadline, and the holder then resumes inside its critical section and mutates
    the destination on an observation from before its successor published. Ownership rules make
    the stalled holder's *cleanup* harmless, but nothing outside the holder can stop its *body*.
    So the timeout below does not authorize anything. It bounds how long this call waits for a
    live holder before failing closed, with a message a person can act on. Breaking is authorized
    only by :func:`_holder_is_provably_dead`, which needs the holder gone, not slow. A wrong guess
    in the safe direction (a dead holder that cannot be proven dead) costs an error; a wrong guess
    in the other direction would cost a reader the tree it was handed, so it is not available.

    Recovery of a dead holder's lock is itself conditional on that holder's token, so a proof
    cannot be spent on a generation that replaced the one it was about (see
    :func:`_recover_dead_publish_lock`).

    Held across observe-decide-replace and **never** across the download: the critical section is
    a few stats and two renames, so a waiter is never queued behind a network fetch.
    """
    lock = dataset / f".publish-{name}"
    token = uuid.uuid4().hex
    staging = Path(tempfile.mkdtemp(dir=str(dataset), prefix=".pubtmp-"))
    (staging / token).write_text(_this_holder(), encoding="utf-8")
    try:
        deadline = time.monotonic() + _PUBLISH_LOCK_TIMEOUT_SECONDS
        while True:
            try:
                os.rename(staging, lock)
                break
            except OSError:
                # The target is a live lock (non-empty, so rename refuses it).
                holder = _read_lock_holder(lock)
                if holder is not None and _holder_is_provably_dead(holder):
                    # Bound to `holder`, not to the path: by the time this runs, the path may be
                    # a live successor that cleared the same corpse first.
                    _recover_dead_publish_lock(lock, holder)
                    continue
                if time.monotonic() >= deadline:
                    raise DatasetUnavailableError(_held_message(lock, holder)) from None
                time.sleep(_PUBLISH_LOCK_POLL_SECONDS)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        yield
    finally:
        _release_publish_lock(lock, token)


def _held_message(lock: Path, holder: Optional[_LockHolder]) -> str:
    """What to tell a person who is waiting on a lock this process may not take."""
    who = str(holder) if holder is not None else "host=unknown pid=unknown"
    reason = (
        "that process is still running"
        if holder is not None and holder.host == _HOSTNAME
        else "this process cannot tell whether that holder is still running"
    )
    return (
        f"another provisioner still holds the publish lock {lock} ({who}) after "
        f"{_PUBLISH_LOCK_TIMEOUT_SECONDS:.0f}s, and {reason}, so it is not this call's to take. "
        "Waiting is the normal outcome of a concurrent cold start and needs nothing. If that "
        f"process is gone, remove {lock} and retry."
    )


def _release_publish_lock(lock: Path, token: str) -> None:
    """Release ``lock``, but only while this call still owns it."""
    try:
        os.unlink(lock / token)
    except OSError:
        # The token is gone, so this lock was broken and whatever sits there now belongs to
        # someone else. Releasing it would put two publishers in the critical section.
        return
    try:
        os.rmdir(lock)
    except OSError:
        # Not empty any more: a successor claimed the directory between the unlink and here, and
        # its token is what is refusing this. Leave it to them.
        pass


def _recover_dead_publish_lock(lock: Path, holder: _LockHolder) -> None:
    """Clear the lock a provably dead holder left, and only that one.

    This is :func:`_release_publish_lock` performed on someone else's behalf, and it is
    conditional in the same way and for the same reason: it names the **generation** that was
    proved dead, never the path. A proof is about a holder, and between proving and acting the
    path can have become a live successor that cleared the same corpse first. Removing whatever
    occupies the path would then evict that successor mid-critical-section and put two publishers
    inside it, which is the concurrent repair this lock exists to prevent. Renaming the corpse
    aside instead of unlinking it does not help: rename also acts on the path.

    So: unlink exactly this holder's token, and treat its absence as "someone else already dealt
    with this, and what is there now is not mine to touch". Then remove the directory only while
    it is empty, since anything in it is a successor's token. Either way the caller goes back to
    the acquire loop and either takes the free path or waits for whoever now holds it.
    """
    try:
        os.unlink(lock / holder.token)
    except OSError:
        # Gone already: a peer recovered this same corpse, and the path now belongs to whoever
        # took it. Nothing here is this call's to remove.
        return
    try:
        os.rmdir(lock)
    except OSError:
        # Not empty: a successor claimed the emptied directory between the unlink and here, and
        # its own token is what refuses this. Leave it to them.
        pass


def _download_task(task: HubTask, dataset: Path) -> None:
    """Fetch + extract one task package, publishing it with a single atomic rename.

    The extracted tree is validated **before** it is published, so an incomplete archive is an
    error at fetch time rather than a cache entry that reads fine until it does not. A damaged
    tree already at the destination is replaced rather than refused, which is what makes the warm
    path repair an incomplete task instead of erroring on it.

    The staging directory is held under an ``flock`` for as long as it is in use, which is what
    lets :func:`_sweep_download_residue` tell an abandoned staging dir from a live one."""
    archive = _archive_bytes(task)
    staging = tempfile.TemporaryDirectory(dir=str(dataset), prefix=".dl-")
    with staging as tmp, _locked(Path(tmp)):
        tmp_path = Path(tmp)
        archive_file = tmp_path / "dist.tar.gz"
        archive_file.write_bytes(archive)
        extracted = tmp_path / "x"
        extracted.mkdir()
        with tarfile.open(archive_file, mode="r:gz") as tf:
            # `data` rejects absolute paths, `..` traversal, and special file types.
            tf.extractall(extracted, filter="data")
        archive_file.unlink()
        # Authenticate before anything else: the pinned hash is what makes these bytes the
        # benchmark's rather than whatever answered the request.
        arrived = content_hash(extracted)
        if arrived != task.content_hash:
            raise DatasetUnavailableError(
                f"the archive for task {task.name} hashes to {arrived}, but {DATASET}@"
                f"{DATASET_REVISION} pins {task.content_hash}. Refusing to cache bytes the pinned "
                "revision does not vouch for."
            )
        absent = missing_task_files(extracted)
        if absent:
            raise DatasetUnavailableError(
                f"the archive for task {task.name} is missing {', '.join(absent)}; refusing to "
                "publish an incomplete task into the cache"
            )
        # Deciding and acting have to be one step: see `_publish_lock`.
        with _publish_lock(dataset, task.name):
            _publish(extracted, dataset, task.name, displaced=tmp_path / "displaced")


def _publish(extracted: Path, dataset: Path, name: str, *, displaced: Path) -> None:
    """Install a validated tree as ``dataset/name``. Call only under :func:`_publish_lock`.

    Observing the destination and replacing it are one step, which is what the lock buys: a
    decision to replace cannot go stale before it is acted on, so this can never undo a peer's
    published tree on the strength of something it learned before that tree existed.

    **A complete destination is a winner, not an obstacle.** Under the lock, a destination that is
    already complete is a peer's finished publish, and it may already have been handed to a reader.
    This call keeps it and discards its own download, which costs nothing: both trees came from the
    same pinned archive. The destructive path is reserved for a destination that is genuinely
    incomplete, i.e. the warm-cache repair.
    """
    destination = dataset / name
    if destination.exists():
        # Authentic, not merely complete: a peer's finished publish is worth keeping, a corrupt
        # tree is what this call is here to replace.
        if _is_task_authentic(dataset, name):
            return
        # Displaced by rename rather than deleted in place: `os.replace` will not overwrite a
        # non-empty directory, and moving the damaged tree aside (into the caller's staging dir,
        # which is cleaned up on exit) keeps the window in which the destination does not exist
        # down to the gap between two renames instead of a whole recursive delete.
        os.replace(destination, displaced)
    try:
        os.replace(extracted, destination)
    except OSError:
        # Only reachable if a publisher that is not holding this lock got there first, i.e. one
        # running against a build without it. Anything already there is a complete tree, so the
        # check is after the failed rename on purpose: before it, it would be a TOCTOU pair with
        # the rename, and the window between them is where the loser's error comes from.
        if not _is_task_authentic(dataset, name):
            raise


def ensure_dataset(root: Optional[Path] = None) -> Path:
    """Ensure the pinned dataset is cached and return its directory. Idempotent.

    On a warm cache this is a directory scan and touches no network. On a cold one it lists the
    pinned revision and fetches every missing task, holding the dataset directory's lock so
    concurrent provisioners pay for the download once between them rather than once each.
    """
    dataset = dataset_dir(root)
    if _cached_task_names(dataset):
        return dataset
    dataset.mkdir(parents=True, exist_ok=True)
    with _locked(dataset):
        # Re-check inside the lock: waiting on it usually means waiting for exactly this
        # download, and the winner published before the wait returned.
        if _cached_task_names(dataset):
            return dataset
        _sweep_download_residue(dataset)
        # Before paying for anything: a cache that is not exactly this revision is not one this
        # call can complete into one, and the listing is not worth fetching to find that out.
        _refuse_residue(dataset)
        # Incomplete counts as missing, so a damaged or truncated task is re-fetched here
        # rather than raising somewhere downstream that reads the file it never got.
        listed = list_dataset_tasks()
        # Authentic, not merely complete: a task whose bytes moved after publication is re-fetched
        # exactly like an incomplete one. Unlike residue, the pin says precisely what these bytes
        # must be, so the repair is deterministic and cannot destroy anything a person authored.
        missing = [t for t in listed if not _is_task_authentic(dataset, t.name)]
        if missing:
            with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
                # Consume the results so any worker's exception is raised here rather than
                # swallowed into a half-provisioned cache that the next call would call warm.
                list(pool.map(lambda t: _download_task(t, dataset), missing))
        # Postcondition, because "no worker raised" is not the same claim as "the benchmark is
        # here": say so about the tree that now exists, in the terms the pin defines.
        unusable = [name for name in pinned_manifest() if not _is_task_authentic(dataset, name)]
        if unusable:
            modified = modified_task_names(dataset)
            absent = [name for name in unusable if name not in set(modified)]
            detail = []
            if absent:
                detail.append(f"{len(absent)} absent or incomplete (e.g. {absent[:3]})")
            if modified:
                detail.append(
                    f"{len(modified)} present but modified since publication, so their bytes are "
                    f"not the ones {DATASET}@{DATASET_REVISION} pins (e.g. {modified[:3]}); "
                    "re-fetching them did not restore them"
                )
            raise DatasetUnavailableError(
                f"provisioning {DATASET}@{DATASET_REVISION} into {dataset} finished with "
                f"{len(unusable)} of {DATASET_TASK_COUNT} pinned tasks unusable: "
                + "; ".join(detail)
            )
        # Exact set, both ways: what arrived, and nothing besides.
        _refuse_residue(dataset)
    return dataset


def ensure_task(name: str, root: Optional[Path] = None) -> Path:
    """Ensure a single task of the pinned revision is cached; return its directory.

    The whole-dataset fetch is the normal path; this is the cheap one-task path a smoke test (or
    a later, laziness-minded caller) uses to exercise provisioning without paying for 192 MB."""
    dataset = dataset_dir(root)
    if name not in pinned_manifest():
        raise DatasetUnavailableError(f"{name!r} is not a task of {DATASET}@{DATASET_REVISION}")
    if _is_task_authentic(dataset, name):
        return dataset / name
    dataset.mkdir(parents=True, exist_ok=True)
    with _locked(dataset):
        if _is_task_authentic(dataset, name):
            return dataset / name
        _sweep_download_residue(dataset)
        for task in list_dataset_tasks():
            if task.name == name:
                _download_task(task, dataset)
                return dataset / name
    raise DatasetUnavailableError(f"{name!r} is not a task of {DATASET}@{DATASET_REVISION}")


def _is_task_authentic(dataset: Path, name: str) -> bool:
    """Whether a cached task is complete **and** is the bytes the pin names.

    Completeness is about filenames, and filenames are not the benchmark. The cold path
    authenticates what it publishes and then never looks at it again, so without this any later
    change to an instruction, a compose file, a verifier or an expected answer would be served as
    the pinned revision forever, and phase 2 executes several of those files straight out of this
    cache. Re-hashing the whole cache costs about 3 seconds serially and about 1.6 in parallel
    for the 755-task revision (176 MB), which is not a reason to invent a cheaper proxy."""
    if not _is_task_complete(dataset, name):
        return False
    expected = pinned_manifest().get(name)
    return expected is not None and content_hash(dataset / name) == expected


def modified_task_names(dataset: Path) -> List[str]:
    """Cached tasks whose files are all present but whose bytes are not the pinned ones.

    Public for the same reason as :func:`missing_task_files`: an operator looking at a cache that
    is being refused should be able to ask what is wrong with it."""
    if not dataset.is_dir():
        return []
    return sorted(
        name
        for name in pinned_manifest()
        if _is_task_complete(dataset, name) and not _is_task_authentic(dataset, name)
    )


def _authentic_task_names(dataset: Path) -> List[str]:
    """The pinned tasks that are present, complete and authentic, hashed in parallel."""
    pinned = sorted(pinned_manifest())
    with ThreadPoolExecutor(max_workers=_DOWNLOAD_WORKERS) as pool:
        verdicts = list(pool.map(lambda name: _is_task_authentic(dataset, name), pinned))
    return [name for name, authentic in zip(pinned, verdicts) if authentic]


def unpinned_task_dirs(dataset: Path) -> List[str]:
    """Task-shaped directories in a cache that the pin does not name.

    A cache is meant to hold the pinned revision and nothing else. Anything else that looks like
    a task gets indexed like one, which is how a leftover directory becomes an extra benchmark
    task: ``num_tasks`` changes, every numeric id after it shifts, and a slice or a published
    result quietly refers to something else."""
    if not dataset.is_dir():
        return []
    pinned = pinned_manifest()
    return sorted(
        entry.name
        for entry in dataset.iterdir()
        if entry.name not in pinned and (entry / "task.toml").is_file()
    )


def _refuse_residue(dataset: Path) -> None:
    """Refuse a cache holding task-shaped directories the pin does not name.

    Refused, not cleaned up. This code cannot tell an older provisioner's leftovers from a
    person's own work or from evidence of a bug worth keeping, and it is the one situation where
    deleting is least defensible: nothing this port writes lands here (its staging directories are
    hidden and prefixed), so residue means something happened that it did not do. The message
    names the directories and the remedy, and the remedy is one command."""
    residue = unpinned_task_dirs(dataset)
    if not residue:
        return
    listed = ", ".join(residue[:3]) + (", ..." if len(residue) > 3 else "")
    raise DatasetUnavailableError(
        f"the cache at {dataset} holds {len(residue)} task-shaped directories that "
        f"{DATASET}@{DATASET_REVISION} does not contain ({listed}). They would be indexed as extra "
        "tasks, changing how many there are and what every task id after them refers to. Nothing "
        "was removed: check whether they are yours, then remove them (for example "
        f"`rm -rf {dataset}/{residue[0]}`) and retry."
    )


def _cached_task_names(dataset: Path) -> List[str]:
    """The complete, **pinned** task names published in ``dataset``, or ``[]`` if any is missing.

    All-or-nothing on purpose: a partial cache (an interrupted first run, or a task whose files
    went missing) must not be mistaken for a warm one, or the index would silently describe a
    subset of the benchmark. Completeness is per task, not merely presence, so a damaged task
    drops the whole cache back to the provisioning path that repairs it.

    Identities, not a count: the question is whether *the pinned tasks* are here, not whether 755
    directories are. A cache holding 755 complete directories under names the pin never mentions
    is not this benchmark, and counting would call it warm.

    An **exact** set: a cache that also holds something task-shaped the pin does not name is not
    warm either, because that something would be indexed as a task. Answering ``[]`` there is not
    the refusal, only the routing: it sends the caller to the cold path, which refuses and says
    what to do (see :func:`_refuse_residue`).

    And the **bytes**, not just the names: a task whose files were changed after publication is
    not warm either, so it is re-fetched (see :func:`_is_task_authentic`)."""
    if not dataset.is_dir():
        return []
    names = _authentic_task_names(dataset)
    if len(names) != DATASET_TASK_COUNT or unpinned_task_dirs(dataset):
        return []
    return sorted(names)


__all__ = [
    "DATASET",
    "DATASET_CONTENT_HASH",
    "DATASET_LICENSE",
    "DATASET_REVISION",
    "DATASET_TASK_COUNT",
    "DATASET_MANIFEST_SHA256",
    "DatasetUnavailableError",
    "HubTask",
    "REQUIRED_TASK_FILES",
    "cache_dir",
    "content_hash",
    "dataset_dir",
    "ensure_dataset",
    "ensure_task",
    "list_dataset_tasks",
    "missing_task_files",
    "modified_task_names",
    "pinned_manifest",
    "unpinned_task_dirs",
    "resolve_dataset_version",
]
