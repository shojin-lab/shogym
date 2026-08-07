"""How shogym gets the pinned upstream ``tau2`` package — the fetch-and-import seam.

tau2-bench publishes **no distribution that matches the pinned commit** — and the name ``tau2``
on PyPI belongs to an *unrelated* magnetochemistry library, so a plain ``tau2`` requirement would
silently install the wrong package. The port used to pin the commit as a direct
(``@ git+https://``) requirement, which PyPI rejects outright and which therefore made all of
shogym unpublishable.

So this adapter **provisions the pinned upstream source at runtime** into a gitignored cache
(``~/.cache/shogym/tau2/<sha>/``, overridable via ``TAU2_SRC`` / ``SHOGYM_CACHE``) and registers it
directly in ``sys.modules`` — never onto ``sys.path``, because the tau2-bench archive root carries
sibling top-level dirs (``tests/``, ``docs/``, ``scripts/``, ``examples/``, ``web/``, ``data/``)
that would shadow shogym's own packages. Only ``src/tau2`` is extracted; the ~700 MB of benchmark
``data/`` in the archive is filtered out during extraction and never written to disk. The
mechanics live in :mod:`shogym.envs._upstream`, shared with the automationbench and yc_bench ports.
Nothing from upstream is committed to shogym, and the SHA pin — hence the fidelity guarantee —
is unchanged; it just moved from a requirement string to :data:`UPSTREAM_SHA` here.

tau2's *own* runtime dependencies used to be resolved transitively by pip through that direct
requirement. They are now declared explicitly by the ``tau2`` extra in ``pyproject.toml`` (the
upstream's ``[project] dependencies`` plus its ``[gym]`` and ``[knowledge]`` extras, verbatim at
the pinned SHA), so ``pip install shogym[tau2]`` still installs exactly the same set.

This module itself is side-effect-free: it holds the pin and the fetch, nothing more. Every ``tau2``
*import* in shogym lives in :mod:`shogym.envs.tau2.mcp_server` (the control-inversion bridge is the
only thing that touches upstream), and that module calls :func:`ensure_source` before them — so
provisioning, a one-time network fetch if the cache is cold, is paid only when a tau2 env is
*constructed* or *served*, never by ``import shogym``.

Note: tau2's **domain data** is a separate concern and is unchanged by the fetch-and-import move.
Upstream resolves it from ``TAU2_DATA_DIR`` (falling back to a path relative to the installed
package, which does not exist for either an installed wheel or this cache), so a tau2 env still
needs ``TAU2_DATA_DIR`` pointing at a tau2-bench ``data/`` checkout — exactly as before.
"""

from __future__ import annotations

from pathlib import Path

from shogym.envs._upstream import ensure_package

# Fidelity pin: the upstream commit this port reproduces.
UPSTREAM_SHA = "1d244f5dca42944b67a379b44bfeb9f5748f189d"
_TARBALL_URL = f"https://github.com/sierra-research/tau2-bench/archive/{UPSTREAM_SHA}.tar.gz"


def ensure_source() -> Path:
    """Ensure the upstream source is available and importable; return its containing directory.

    Idempotent and thread-safe. ``TAU2_SRC`` overrides the cache with an existing checkout (a dir
    that *contains* a ``tau2`` package), so a provisioned/offline environment needs no network;
    otherwise the pinned tarball is fetched on the first call and reused thereafter. tau2-bench is
    a src-layout project, so the package sits at ``<archive root>/src/tau2``."""
    return ensure_package(
        package="tau2", sha=UPSTREAM_SHA, tarball_url=_TARBALL_URL, archive_subdir="src"
    )


__all__ = ["UPSTREAM_SHA", "ensure_source"]
