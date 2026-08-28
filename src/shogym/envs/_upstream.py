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
from typing import Iterator, Optional

_DOWNLOAD_TIMEOUT_SECONDS = 120.0

_provision_lock = threading.Lock()


def source_env_var(package: str) -> str:
    """The env var that overrides ``package``'s provisioned source (e.g. ``TAU2_SRC``).

    Set it to an existing checkout — a dir that *contains* the package — so a provisioned or
    offline environment needs no network."""
    return f"{package.upper()}_SRC"


def _cache_root(package: str, sha: str) -> Path:
    base = os.environ.get("SHOGYM_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "shogym"
    return root / package / sha


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


class ExclusionUnavailable(RuntimeError):
    """A caller that needs real exclusion asked for a lock this filesystem cannot give.

    Its own type because it is neither a bug nor a failure of the work the caller wanted done: the
    directory is fine, the code is fine, and the mount underneath them cannot provide ``flock``.
    What a caller does about that depends entirely on what the lock was holding together, so the
    two answers live at the call sites (see :func:`_locked`) and this is what carries the refusal
    for the ones that cannot go on without it."""


@contextlib.contextmanager
def _locked(directory: Path, *, required: bool = False) -> Iterator[None]:
    """Hold every other *process* out of ``directory`` for the length of the ``with`` body.

    ``_provision_lock`` only serializes threads inside one interpreter, and a cold cache is
    routinely hit by several processes at once (a parallel test run, several served episodes
    starting together). Without this they each download the same tarball — 93 MB apiece for
    tau2-bench — and then race check-then-``os.replace`` on the publish, where the loser's rename
    fails with ``ENOTEMPTY`` against the winner's finished tree.

    An ``flock`` rather than a lock *file*, for the reason ``serve.stream._locked`` gives: a lock
    made of a file existing survives the process that made it, so a crash mid-download would wedge
    every later provisioner behind residue that only a liveness guess could clear. This lock is
    owned by the kernel and released when the descriptor closes, which happens however the process
    ends.

    Two directories are locked, for two jobs. ``<cache>/<package>/``, the parent of the per-SHA
    trees, is the provisioning critical section — so one upstream's provisioning never waits on
    another's. Each ``.dl-*`` staging directory is locked by its owner for as long as it is in use,
    which is what lets :func:`_sweep_download_residue` recognize an abandoned one.

    Blocking, not a try-lock: the only honest reading of "someone else holds it" is that they are
    fetching the very thing this call wants, and the wait ends with the work already done. (The
    sweep is the exception and passes ``LOCK_NB`` itself, because there "someone else holds it"
    means "leave it alone", not "wait".)

    **Whether a filesystem that cannot lock at all may be yielded to anyway is the caller's to
    say, and it is what ``required`` says.** The default is no, in the sense of "no, this does not
    need exclusion": the upstream-source path this module was written for is an efficiency and
    hygiene measure over a cache whose correctness is carried by other things entirely. Its publish
    is a single atomic rename and a loser validates the winner (see :func:`_download_package`), and
    the sweep will not delete what it cannot lock (see :func:`_sweep_download_residue`). Refusing
    to provision on a mount where provisioning demonstrably works, because an optimization is
    unavailable, would make a nice-to-have load-bearing. So there the unsupported-lock errnos
    degrade to redundant-but-correct work, with one warning.

    ``required=True`` is for the call sites where that reasoning does not hold, and it raises
    :class:`ExclusionUnavailable` rather than yielding. Two shapes need it. A builder that stages
    under a **fixed** name and deletes whatever is there first (the appworld runtime and corpus
    builders) has no atomic publish to fall back on: two cold builders remove and rename each
    other's staging tree, and the loser can publish half of the winner's. And a window that makes
    a shared directory writable for the length of a build (appworld's permission windows, which
    open a sealed cache directory and seal it again) is *only* correct while nothing else is
    inside it, because what a second builder observes is not a stale tree but an open one.

    Both are silent when they go wrong, and both are the material a run is scored against. A
    refusal that names the mount is the honest outcome there, and the remedy is the same one the
    warning already gives: point ``SHOGYM_CACHE`` at a local filesystem. Every other ``OSError``
    still propagates in both modes."""
    # O_RDONLY: the lock is on the descriptor, not on what may be done through it, so this does
    # not ask for write access it never uses.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except OSError as exc:
            if exc.errno not in _LOCK_UNSUPPORTED:
                raise
            if required:
                raise ExclusionUnavailable(
                    f"shogym: {directory} is on a filesystem that cannot provide flock "
                    f"({exc.strerror}), and this step needs real exclusion between processes "
                    f"rather than the redundant work an unlockable cache degrades to. Point "
                    f"SHOGYM_CACHE at a local filesystem"
                ) from exc
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
        archive = tmp_path / "archive.tar.gz"
        with urllib.request.urlopen(tarball_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
            with archive.open("wb") as fh:
                shutil.copyfileobj(resp, fh)

        # A GitHub archive extracts to a single `<repo>-<sha>/` root; the package sits at
        # `<root>/<archive_subdir>/<package>` (`archive_subdir` is "" for a flat layout, "src"
        # for a src-layout upstream).
        extracted = tmp_path / "x"
        wanted: str | None = None
        with tarfile.open(archive, mode="r:gz") as tf:
            for member in tf:
                if wanted is None:
                    root = member.name.split("/", 1)[0]
                    parts = [root, archive_subdir, package] if archive_subdir else [root, package]
                    wanted = "/".join(parts)
                if member.name == wanted or member.name.startswith(wanted + "/"):
                    tf.extract(member, extracted, filter="data")
        archive.unlink()

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
    :func:`_register_package`)."""
    src = _source_dir(package, sha)
    with _provision_lock:
        if not (src / package).is_dir():
            override = os.environ.get(source_env_var(package))
            if override:
                raise RuntimeError(
                    f"{source_env_var(package)}={src} does not contain a '{package}' package"
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


__all__ = ["ExclusionUnavailable", "ensure_package", "source_env_var"]
