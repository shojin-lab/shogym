"""Runtime provisioning of SHA-pinned upstream benchmark sources — the fetch-and-import pattern.

Several ports wrap an upstream benchmark that shogym must *not* depend on through ``pip``:

- **PyPI forbids direct (``@ git+https://``) references**, so any extra that pinned an upstream
  commit made the whole project unpublishable — and the two upstreams involved (``tau2-bench``,
  ``yc-bench``) publish no release that matches the pinned commit. ``tau2`` is worse than absent
  on PyPI: the name belongs to an *unrelated* magnetochemistry library, so a plain ``tau2``
  requirement would silently install the wrong package.
- ``automationbench`` additionally cannot be resolved at all under shogym's Python pin.

So those ports provision the pinned upstream **source** at runtime instead. This module is the one
implementation of that: download the SHA-pinned GitHub tarball, keep only the upstream's top-level
package, cache it under ``~/.cache/shogym/<package>/<sha>/``, and register it in ``sys.modules``.
Nothing from upstream is committed to shogym, and the SHA pin (hence the fidelity guarantee) is
unchanged — it just moves from a requirement string to a constant in the port's adapter.

Two rules the callers depend on:

- **Nothing lands on ``sys.path``.** These upstream checkouts carry sibling top-level dirs
  (``tests/``, ``visualizer/``, ``docs/``, ``scripts/``) that would shadow shogym's own packages.
  Only the upstream package itself is extracted, and it is bound by registering the top-level
  module with an explicit ``__path__`` — so every absolute ``from <package>.x import y`` resolves
  through that ``__path__`` and no directory is ever appended to ``sys.path``. A port that must
  hand the source to a *subprocess* (yc_bench runs the upstream CLI) can put the returned
  directory on that subprocess's ``PYTHONPATH`` — it contains the package and nothing else — but
  ``PYTHONPATH`` is not the front of ``sys.path``: the interpreter prepends the working directory
  for ``-m``, so such a subprocess must also run under ``-P``.
- **The pinned package wins, or nothing does.** :func:`ensure_package` refuses to hand back a name
  that is already bound to a *different* package rather than reporting success over the top of it
  — the whole point of the fetch-and-import move is that ``tau2`` on PyPI is an unrelated project
  and ``yc-bench``'s release is not this pin, so "some module called ``tau2`` is importable" is
  never good enough.

Each port keeps its own ``ensure_source()`` wrapper holding its ``UPSTREAM_SHA``; importing that
port's adapter triggers provisioning (a one-time network fetch on a cold cache), so it is only
ever paid when the env is *constructed* or *served* — never by ``import shogym``.

**When the fetch is allowed is a decision the caller makes**, through
:data:`PROVISIONING_ENV_VAR` and the three modes :func:`provisioning_mode` reads out of it. The
default fetches on demand, which is what a laptop wants. A run that has been prepared in advance
says ``offline`` and then this module never downloads: a source that was not prepared raises
:class:`UpstreamNotPrepared` naming it, rather than reaching for the network from inside a stage
that promised not to. Binding a prepared source is not the end of it, because *importing* an
upstream can fetch as well, so that mode also selects the bundled data of the libraries these
upstreams import (see :func:`use_bundled_import_data`).

An upstream's bulk **data** is provisioned separately by :func:`ensure_data`, under the same
rules. It is deliberately not part of a source: it is read, never imported, and for tau2-bench it
is ~700 MB, so a port that needs none of it should pay for none of it.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import importlib.util
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import urllib.request
import warnings
from pathlib import Path
from typing import Iterator, Optional, Sequence, Tuple

_DOWNLOAD_TIMEOUT_SECONDS = 120.0

_provision_lock = threading.Lock()

#: The env var that says which provisioning stage this process belongs to.
PROVISIONING_ENV_VAR = "SHOGYM_PROVISIONING"

#: Fetch whatever is missing. The stage that prepares a machine, and the one place a download is
#: the point rather than a surprise.
PREPARE = "prepare"

#: Use what is already prepared and download nothing. A source nobody prepared is an error that
#: names it, never a quiet fetch.
OFFLINE = "offline"

#: The default, for a developer's machine: fetch on demand, and let a caller read an unreachable
#: network as "this machine cannot run these tests" rather than as a defect.
LOCAL = "local"

_MODES = (PREPARE, OFFLINE, LOCAL)


class UpstreamNotPrepared(RuntimeError):
    """An upstream source is absent in a mode that may not fetch it. Says which one, and where."""


def provisioning_mode() -> str:
    """The provisioning mode this process runs under, read from :data:`PROVISIONING_ENV_VAR`.

    Three values, and the difference between them is only ever about the network:

    - ``prepare`` fetches and binds every pinned source, over the network, on purpose.
    - ``offline`` fetches nothing. A source that is already prepared (in the cache, or behind the
      port's ``_SRC`` override) is used; one that is not raises :class:`UpstreamNotPrepared`.
    - ``local``, the default, fetches on demand, which is what an unprepared laptop needs.

    An unset variable means ``local``. **An unrecognized value raises** rather than falling back
    to it: a typo that quietly restored on-demand downloading would give a stage that promised to
    fetch nothing the behavior it was configured out of, and say nothing about it.

    Which is why every entry point that provisions something calls this *first*, before the check
    that returns what is already there: :func:`ensure_package`, :func:`ensure_data`, the image and
    oracle-package entry points in :mod:`shogym.envs.frontier_bench.mcp_server`, and the test
    fixtures that prepare or require one of those. A read that happened only on the cold path
    would refuse a typo on the machines with nothing prepared and nowhere else, which is the same
    as not refusing it: the machines this mode is for are the prepared ones."""
    raw = os.environ.get(PROVISIONING_ENV_VAR, "").strip().lower()
    if not raw:
        return LOCAL
    if raw not in _MODES:
        raise RuntimeError(
            f"{PROVISIONING_ENV_VAR}={raw!r} is not a provisioning mode; it must be one of "
            f"{', '.join(_MODES)} (or unset, which means {LOCAL})."
        )
    return raw


#: What an ``offline`` process must put in its environment so that *importing* an upstream stays
#: offline too. LiteLLM fetches its model-cost map from a file server the moment ``import litellm``
#: runs, unless this says to read the copy that ships inside the wheel; it swallows the failure and
#: falls back to that same copy, so the fetch is invisible in the result and visible only on the
#: wire. Two of the three ports provisioned here reach LiteLLM as they import: tau2 from upstream's
#: own package ``__init__``, which runs while the source is being bound, and yc_bench from the
#: upstream module its adapter imports immediately afterwards. Either chain would otherwise reach
#: the network on a machine whose source is already prepared.
_OFFLINE_IMPORT_ENV = {"LITELLM_LOCAL_MODEL_COST_MAP": "true"}


def use_bundled_import_data() -> None:
    """Point the libraries an upstream imports at their bundled data instead of a download.

    Set unconditionally rather than as a default, because the mode is the stronger statement: a
    run that says it fetches nothing may not be talked into a fetch by an inherited variable. It
    is set in ``os.environ`` (not in a copy) so that the child processes a port launches, which
    import the same upstream from the same prepared source, inherit it."""
    os.environ.update(_OFFLINE_IMPORT_ENV)


def source_env_var(package: str) -> str:
    """The env var that overrides ``package``'s provisioned source (e.g. ``TAU2_SRC``).

    Set it to an existing checkout — a dir that *contains* the package — so a provisioned or
    offline environment needs no network."""
    return f"{package.upper()}_SRC"


def cache_root() -> Path:
    """The directory shogym provisions into: ``~/.cache/shogym``, or ``SHOGYM_CACHE``."""
    base = os.environ.get("SHOGYM_CACHE")
    return Path(base) if base else Path.home() / ".cache" / "shogym"


def _cache_root(package: str, sha: str) -> Path:
    return cache_root() / package / sha


def data_dir(package: str) -> Path:
    """Where a port's bulk upstream *data* is provisioned: beside its per-SHA source trees.

    Not inside one, because that is where upstream's own resolver looks. tau2 reads its domain
    data from three directories above its ``utils`` module plus ``data``, which for a source at
    ``<cache>/tau2/<sha>/tau2`` is exactly this path, so data provisioned here is found with no
    environment variable and no patch to the upstream. The cost of upstream's convention is that
    the location is not per-SHA; :func:`ensure_data` stamps the SHA it wrote and replaces a tree
    that was written for another one."""
    return cache_root() / package / "data"


def _source_dir(package: str, sha: str) -> Path:
    """The directory that *contains* the upstream package."""
    override = os.environ.get(source_env_var(package))
    if override:
        return Path(override).expanduser().resolve()
    return _cache_root(package, sha)


def _module_dir(module: object) -> Optional[Path]:
    """The directory an already-imported module was loaded from, or ``None`` if it can't be told.

    ``__path__`` for a package, the parent of ``__file__`` otherwise. A module with neither (a
    namespace package, a hand-built stub) is unplaceable, which is treated as "not ours"."""
    for entry in list(getattr(module, "__path__", ()) or ()):
        try:
            return Path(str(entry)).resolve()
        except OSError:  # pragma: no cover - unreadable path
            return None
    filename = getattr(module, "__file__", None)
    if filename:
        try:
            return Path(str(filename)).resolve().parent
        except OSError:  # pragma: no cover - unreadable path
            return None
    return None


def _purge_submodules(package: str) -> None:
    """Drop every ``<package>.x`` entry from ``sys.modules``.

    A submodule in ``sys.modules`` is returned by ``import`` *without consulting its parent's*
    ``__path__``, so a leftover ``tau2.config`` shadows the pinned one no matter how carefully the
    top-level module was bound. Entries get left behind by a registration that raised partway
    through the package's own imports; clearing them is what makes a retry a real retry."""
    for name in [n for n in sys.modules if n.startswith(package + ".")]:
        del sys.modules[name]


def _register_package(package: str, pkg_dir: Path) -> None:
    """Import ``package`` from ``pkg_dir`` into ``sys.modules`` directly. Idempotent.

    Deliberately does **not** put ``pkg_dir``'s parent on ``sys.path`` (see the module docstring):
    the top-level module is created with ``submodule_search_locations`` set to ``pkg_dir``, so the
    whole package tree resolves through its own ``__path__``.

    **A name already bound to something else is refused, not accepted.** Returning early on "the
    name is in ``sys.modules``" would be exactly the collision the fetch-and-import move exists to
    prevent: an application that imported PyPI's unrelated ``tau2`` (or ``yc-bench``'s non-pinned
    release) before constructing the env would be told provisioning succeeded while every later
    ``from tau2.x import y`` reached into the wrong project. The bound module is checked against
    ``pkg_dir`` and a mismatch raises. It is *not* silently replaced: whoever imported it holds
    references this module cannot rewrite, and one name cannot mean two packages in one
    interpreter — so the honest outcome is to refuse and say why."""
    existing = sys.modules.get(package)
    if existing is not None:
        if _module_dir(existing) == pkg_dir.resolve():
            return  # already bound to the pinned source
        raise RuntimeError(
            f"the name {package!r} is already imported in this process from "
            f"{_module_dir(existing) or '<unknown location>'}, which is not the pinned upstream "
            f"at {pkg_dir}. shogym cannot bind one name to two packages: import the shogym env "
            f"before anything else imports {package!r}, or point "
            f"{source_env_var(package)} at the checkout that is already loaded."
        )
    # Only reachable with the top level *absent*, so these are residue from a registration that
    # raised, or entries someone planted. Either way they would shadow the pinned package.
    _purge_submodules(package)

    init = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        package, init, submodule_search_locations=[str(pkg_dir)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the {package} package from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # Leaving a half-initialized module bound would make every later import see a broken
        # package instead of retrying, so unbind it — along with whatever submodules it managed
        # to import before it failed — and let the caller see the real error.
        sys.modules.pop(package, None)
        _purge_submodules(package)
        raise


# The errnos that mean "this filesystem cannot give you an flock here", as opposed to "something
# is wrong". ``ENOTSUP`` and ``EOPNOTSUPP`` are the same number on Linux and different on macOS, so
# both are listed. ``ENOLCK`` is how a mount whose lock manager is absent (a classic NFS setup)
# says it. Deliberately *not* ``EINVAL``: some filesystems use it for this too, but it is also what
# a wrong ``operation`` argument returns, and swallowing it would hide a future edit's mistake
# behind a silent loss of exclusion. If a real mount turns up returning ``EINVAL``, that should
# arrive as a bug report rather than as a cache that quietly stopped locking.
_LOCK_UNSUPPORTED = frozenset({errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOLCK})

_warned_unlocked = False


def _warn_unlocked(directory: Path, exc: OSError) -> None:
    """Say once, per process, that provisioning is running without inter-process exclusion.

    Silence would be the wrong call even though nothing becomes *incorrect*: on such a mount every
    concurrent cold start downloads the whole tarball again (93 MB apiece for tau2-bench) and
    :func:`_sweep_download_residue` can no longer reclaim a killed provisioner's partial download,
    so the only symptoms are a slow cache that grows without bound. Those are worth one line."""
    global _warned_unlocked
    if _warned_unlocked:
        return
    _warned_unlocked = True
    warnings.warn(
        f"shogym: {directory} is on a filesystem that cannot provide flock "
        f"({exc.strerror}); upstream sources will still be provisioned correctly, but "
        f"concurrent cold starts will each download the tarball, and abandoned '.dl-*' staging "
        f"directories will not be reclaimed. Point SHOGYM_CACHE at a local filesystem to avoid "
        f"both.",
        RuntimeWarning,
        # Points at `_locked` itself, the thing that degraded. Deliberately not walked further
        # out: the helper has two call sites at different depths, so any single number that
        # flattered one would misattribute the other, and `contextlib`'s own frame sits between
        # them either way. The message carries the directory and the remedy; the frame only needs
        # to say which part of shogym is speaking.
        stacklevel=2,
    )


@contextlib.contextmanager
def _locked(directory: Path) -> Iterator[None]:
    """Hold every other *process* out of ``directory`` for the length of the ``with`` body.

    ``_provision_lock`` only serializes threads inside one interpreter, and a cold cache is
    routinely hit by several processes at once (a parallel test run, several served episodes
    starting together). Without this they each download the same tarball, 93 MB apiece for
    tau2-bench, and then race check-then-``os.replace`` on the publish, where the loser's rename
    fails with ``ENOTEMPTY`` against the winner's finished tree.

    An ``flock`` rather than a lock *file*: a lock made of a file existing survives the process
    that made it, so a crash mid-download would wedge
    every later provisioner behind residue that only a liveness guess could clear. This lock is
    owned by the kernel and released when the descriptor closes, which happens however the process
    ends.

    Two directories are locked, for two jobs. ``<cache>/<package>/``, the parent of the per-SHA
    trees, is the provisioning critical section, so one upstream's provisioning never waits on
    another's. Each ``.dl-*`` staging directory is locked by its owner for as long as it is in use,
    which is what lets :func:`_sweep_download_residue` recognize an abandoned one.

    Blocking, not a try-lock: the only honest reading of "someone else holds it" is that they are
    fetching the very thing this call wants, and the wait ends with the work already done. (The
    sweep is the exception and passes ``LOCK_NB`` itself, because there "someone else holds it"
    means "leave it alone", not "wait".)

    **A filesystem that cannot lock at all is yielded to anyway, with a warning.** This module's
    exclusion is an efficiency and hygiene measure over a cache whose correctness is carried by
    other things entirely: the publish is a single atomic rename and a loser validates the winner
    (see :func:`_download_package`), and the sweep will not delete what it cannot lock (see
    :func:`_sweep_download_residue`). Refusing to provision on a mount where provisioning
    demonstrably works, because an optimization is unavailable, would make a nice-to-have
    load-bearing. So the unsupported-lock errnos degrade to redundant but correct work, and the
    remedy the warning gives is to point ``SHOGYM_CACHE`` at a local filesystem. Every other
    ``OSError`` still propagates."""
    # O_RDONLY: the lock is on the descriptor, not on what may be done through it, so this does
    # not ask for write access it never uses.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in _LOCK_UNSUPPORTED:
                raise
            _warn_unlocked(directory, exc)
        yield
    finally:
        os.close(descriptor)  # the close is the release


def _sweep_download_residue(directory: Path) -> None:
    """Delete the ``.dl-*`` staging dirs in ``directory`` whose owner is gone. Never a live one.

    A provisioner killed mid-download (SIGKILL, a cancelled CI job) never runs
    ``TemporaryDirectory``'s cleanup, and what it leaves behind holds the partial archive — up to
    93 MB per corpse for tau2-bench.

    "Is this directory abandoned?" is exactly the liveness question this codebase refuses to answer
    with a guess, so it is not guessed: every live provisioner holds an ``flock`` on its own
    staging directory for as long as it is using it (see :func:`_download_package`), and this only
    removes the ones it can take that lock on. A living owner still holds it and is skipped; a dead
    owner's lock was released by the kernel when its process ended, however it ended. Deliberately
    **not** relying on the caller holding :func:`_locked`: that lock is what makes a *sweep during
    someone else's download* impossible in the normal case, but a filesystem that fails to provide
    exclusion — quietly, or loudly enough that :func:`_locked` degraded — would otherwise turn
    this function into one that deletes its peers' work out from under them.

    Best-effort by construction. Any failure to take the lock leaves the directory alone, so on a
    mount with no locking at all nothing is ever reclaimed: a cache that grows is the price of
    never deleting live work, and it is the right way round. Reclaiming is housekeeping, and
    housekeeping may not raise into a provision that would otherwise have succeeded."""
    for residue in sorted(directory.glob(".dl-*")):
        try:
            descriptor = os.open(residue, os.O_RDONLY)
        except OSError:  # already gone, or not a directory this process may open
            continue
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # ``EWOULDBLOCK``: a live provisioner is inside it. An unsupported-lock errno: this
            # filesystem cannot tell a corpse from a peer, so neither may be touched. Anything
            # else: unexplained, and an unexplained directory is not one to delete. All three
            # answer "leave it alone".
            continue
        else:
            shutil.rmtree(residue, ignore_errors=True)
        finally:
            os.close(descriptor)


def _fetch_and_extract(
    tarball_url: str, staging: Path, keep: Sequence[str]
) -> Tuple[Path, Optional[str]]:
    """Download the pinned tarball into ``staging`` and extract the ``keep`` subtrees from it.

    ``keep`` holds paths relative to the archive's single root directory, whose name carries the
    SHA and is therefore not known until the first member is read. Everything outside them is
    streamed past rather than written: these archives carry top-level ``tests/`` / ``docs/`` /
    ``visualizer/`` dirs, and for tau2-bench ~700 MB of benchmark data and results, none of which
    a caller asked for. Returns the directory the members landed in and that root's name (``None``
    for an archive with no members at all, which the callers report as a layout error).

    tarfile's ``data`` filter rejects path traversal, and the archive is deleted as soon as it is
    extracted so the staging directory never holds both copies for longer than it must."""
    archive = staging / "archive.tar.gz"
    with urllib.request.urlopen(tarball_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
        with archive.open("wb") as fh:
            shutil.copyfileobj(resp, fh)

    extracted = staging / "x"
    root: Optional[str] = None
    wanted: list[str] = []
    with tarfile.open(archive, mode="r:gz") as tf:
        for member in tf:
            if root is None:
                root = member.name.split("/", 1)[0]
                wanted = [f"{root}/{path}" for path in keep]
            if any(member.name == w or member.name.startswith(w + "/") for w in wanted):
                tf.extract(member, extracted, filter="data")
    archive.unlink()
    return extracted, root


def _download_package(package: str, tarball_url: str, archive_subdir: str, dest: Path) -> None:
    """Fetch the pinned upstream tarball and extract it so ``dest/<package>`` exists.

    Only the upstream **package** is kept: these archives also carry top-level ``tests/`` /
    ``docs/`` / ``visualizer/`` dirs (and, for tau2-bench, ~700 MB of benchmark data and results),
    none of which may reach an import path. Members are streamed and filtered by prefix, so the
    siblings are never even written to disk. Extraction is atomic (temp dir + ``os.replace``) so a
    concurrent provisioner can't observe a half-written tree, and uses tarfile's ``data`` filter to
    reject path traversal.

    The staging directory is held under an ``flock`` for as long as this call is using it. That is
    what lets :func:`_sweep_download_residue` tell an abandoned staging directory from a live one
    without guessing at liveness — and it is held whether or not the caller took the outer
    provisioning lock, so the sweep stays safe even where that lock cannot be provided. Where
    *neither* can be taken (a mount with no locking, see :func:`_locked`), the sweep reclaims
    nothing and this still publishes correctly: the loser of a publish race validates the winner
    rather than raising."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(dir=str(dest.parent), prefix=".dl-")
    with staging as tmp, _locked(Path(tmp)):
        tmp_path = Path(tmp)
        # A GitHub archive extracts to a single `<repo>-<sha>/` root; the package sits at
        # `<root>/<archive_subdir>/<package>` (`archive_subdir` is "" for a flat layout, "src"
        # for a src-layout upstream).
        relative = f"{archive_subdir}/{package}" if archive_subdir else package
        extracted, root = _fetch_and_extract(tarball_url, tmp_path, [relative])
        wanted = f"{root}/{relative}" if root is not None else None

        staged_pkg = extracted / (wanted or "")
        if wanted is None or not (staged_pkg / "__init__.py").is_file():
            raise RuntimeError(
                f"unexpected {package} archive layout: no '{wanted}' package in {tarball_url}"
            )
        staged = tmp_path / ".staged"
        staged.mkdir()
        os.replace(staged_pkg, staged / package)
        try:
            os.replace(staged, dest)
        except OSError:
            # `dest` is published by a single atomic rename, so anything already there is a
            # complete tree a concurrent provisioner installed — this call raced it and lost.
            # (`_locked` makes that vanishingly rare; it stays handled because a lock the
            # filesystem cannot provide, on some network mounts, degrades to exactly this race
            # rather than to corruption.) Checking *after* the failed rename rather than before it
            # is the point: a check before is a TOCTOU pair with the rename, and the window
            # between them is where the loser's `ENOTEMPTY` comes from.
            if not (dest / package / "__init__.py").is_file():
                raise


def ensure_package(
    *,
    package: str,
    sha: str,
    tarball_url: str,
    archive_subdir: str = "",
) -> Path:
    """Ensure ``package``'s pinned upstream source is importable; return its containing directory.

    Idempotent, thread-safe, and safe against other processes sharing the cache. If the override
    env var (see :func:`source_env_var`) is set it is used as-is and no network is touched.
    Otherwise the pinned tarball is downloaded into ``~/.cache/shogym/<package>/<sha>/`` on the
    first call and reused thereafter — so only the very first construction on a cold cache pays
    for the fetch, and concurrent cold starts normally pay it *once between them* rather than once
    each (see :func:`_locked`; on a filesystem that cannot lock they each pay it, correctly, and
    say so once). The package is then registered directly in ``sys.modules``, never onto
    ``sys.path``, and never over the top of a different package already bound to that name (see
    :func:`_register_package`).

    Under ``SHOGYM_PROVISIONING=offline`` the download is refused rather than performed: a source
    that is present is bound as usual, and a source that is absent raises
    :class:`UpstreamNotPrepared` naming the package and the directory it was expected in (see
    :func:`provisioning_mode`).

    **The mode is read on every call, warm cache or cold**, for two reasons. An unrecognized
    value has to be refused wherever it is read, or a typo would be caught only on the machines
    that happened to need a download. And a prepared source still has an import to pay for, which
    is where :func:`use_bundled_import_data` comes in: binding is what runs upstream's own
    ``__init__``, and the port's own upstream imports follow it, so this is the last moment before
    any of them can start."""
    mode = provisioning_mode()
    if mode == OFFLINE:
        use_bundled_import_data()
    src = _source_dir(package, sha)
    with _provision_lock:
        if not (src / package).is_dir():
            override = os.environ.get(source_env_var(package))
            if override:
                raise RuntimeError(
                    f"{source_env_var(package)}={src} does not contain a '{package}' package"
                )
            if mode == OFFLINE:
                raise UpstreamNotPrepared(
                    f"{package} at {sha} was not prepared: no '{package}' package under {src}. "
                    f"{PROVISIONING_ENV_VAR}={OFFLINE} forbids the download that would fetch it, "
                    f"so fetch it first in a stage that may ({PROVISIONING_ENV_VAR}={PREPARE}), "
                    f"or point {source_env_var(package)} at a checkout that contains it."
                )
            src.parent.mkdir(parents=True, exist_ok=True)
            with _locked(src.parent):
                # Re-check inside the lock: waiting on it usually means waiting for exactly this
                # download, and the winner published before the wait returned.
                if not (src / package).is_dir():
                    _sweep_download_residue(src.parent)
                    _download_package(package, tarball_url, archive_subdir, src)
        _register_package(package, src / package)
    return src


#: Written into a provisioned data directory, holding the SHA it was extracted from. The data
#: location is upstream's to choose and is not per-SHA (see :func:`data_dir`), so this file is how
#: a later pin can tell its own data from an older pin's, and how :func:`ensure_data` knows a tree
#: is one it wrote rather than one a caller put there.
DATA_SHA_FILE = ".shogym-upstream-sha"


def ensure_data(
    *,
    package: str,
    sha: str,
    tarball_url: str,
    archive_subdir: str,
    keep: Sequence[str],
) -> Path:
    """Ensure the pinned archive's data subtree is on disk for ``package``; return its directory.

    The counterpart of :func:`ensure_package` for the part of an upstream that is *read* rather
    than imported. Some ports need it (tau2's domains are its tasks, policies and databases) and
    the source extraction deliberately drops it, because ~700 MB of benchmark data has no business
    on an import path.

    Idempotent: a directory already stamped with this SHA, holding every subtree asked for, is
    returned untouched, so preparing twice costs a few directory reads. A directory stamped with a
    *different* SHA, or missing a subtree a later caller added to ``keep``, is this function's own
    older work and is replaced. A directory with no stamp is somebody else's, and is left exactly
    as it is, because upstream's resolver would have used it and it is not shogym's to delete.

    Under ``offline`` an absent tree raises :class:`UpstreamNotPrepared` naming it, for the same
    reason a source does: a stage that promised to fetch nothing must say what it is missing
    rather than go and get it."""
    mode = provisioning_mode()
    dest = data_dir(package)
    with _provision_lock:
        if dest.is_dir() and not (dest / DATA_SHA_FILE).is_file():
            return dest  # a caller's own checkout, in the place upstream looks for one
        if _data_is_prepared(dest, sha, keep):
            return dest
        if mode == OFFLINE:
            raise UpstreamNotPrepared(
                f"{package} data at {sha} was not prepared: no {list(keep)} under {dest} carrying "
                f"that pin. {PROVISIONING_ENV_VAR}={OFFLINE} forbids the download that would "
                f"fetch it, so fetch it first in a stage that may "
                f"({PROVISIONING_ENV_VAR}={PREPARE})."
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        with _locked(dest.parent):
            # Re-check under the lock, as the source path does: waiting on it usually means
            # waiting for this very download.
            if not _data_is_prepared(dest, sha, keep):
                _sweep_download_residue(dest.parent)
                _download_data(package, tarball_url, archive_subdir, keep, sha, dest)
    return dest


def _data_is_prepared(dest: Path, sha: str, keep: Sequence[str]) -> bool:
    """Whether ``dest`` is this SHA's data *and* holds every subtree the caller asked for.

    The second half matters because the stamp records which pin was extracted, not which parts of
    it: a port that later reads one more subtree would otherwise be handed the tree that predates
    it and told it was prepared."""
    stamp = dest / DATA_SHA_FILE
    if not stamp.is_file() or stamp.read_text().strip() != sha:
        return False
    return all((dest / path).is_dir() for path in keep)


def _download_data(
    package: str,
    tarball_url: str,
    archive_subdir: str,
    keep: Sequence[str],
    sha: str,
    dest: Path,
) -> None:
    """Fetch the pinned tarball and publish ``<root>/<archive_subdir>`` at ``dest``, stamped.

    Only the ``keep`` subtrees within it are extracted, so the caller pays for the data it named
    and not for the rest of the archive. The publish is the same single rename the source path
    uses, with one difference: an existing tree that this function wrote before (it carries the
    stamp) is removed first, because a rename cannot replace a non-empty directory and an older
    pin's data in the place upstream reads from is worse than no data at all."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = tempfile.TemporaryDirectory(dir=str(dest.parent), prefix=".dl-")
    with staging as tmp, _locked(Path(tmp)):
        tmp_path = Path(tmp)
        relative = [f"{archive_subdir}/{path}" for path in keep]
        extracted, root = _fetch_and_extract(tarball_url, tmp_path, relative)
        staged = extracted / f"{root}/{archive_subdir}" if root is not None else extracted
        missing = [path for path in keep if not (staged / path).exists()]
        if root is None or missing:
            raise RuntimeError(
                f"unexpected {package} archive layout: no {missing or [archive_subdir]} under "
                f"'{archive_subdir}' in {tarball_url}"
            )
        (staged / DATA_SHA_FILE).write_text(sha + "\n")
        if (dest / DATA_SHA_FILE).is_file():
            shutil.rmtree(dest)
        os.replace(staged, dest)


__all__ = [
    "DATA_SHA_FILE",
    "LOCAL",
    "OFFLINE",
    "PREPARE",
    "PROVISIONING_ENV_VAR",
    "UpstreamNotPrepared",
    "cache_root",
    "data_dir",
    "ensure_data",
    "ensure_package",
    "provisioning_mode",
    "source_env_var",
    "use_bundled_import_data",
]
