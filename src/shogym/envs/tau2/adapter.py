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
``data/`` is filtered out during extraction and never lands on disk unpacked (the compressed
tarball is staged whole, then removed). The
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

tau2's **domain data** is the separate half, and :func:`ensure_data` is how it arrives. Upstream
resolves it from ``TAU2_DATA_DIR``, falling back to ``data`` three directories above its own
``utils`` module, which for a cached source is ``<cache>/tau2/data``; that is where
:func:`ensure_data` puts it, so a prepared machine needs no environment variable. It is a separate
call rather than part of :func:`ensure_source` because it is a separate size: the domains are 139
MB against the source's few, and a port that only needs to import tau2 should pay for neither the
domains nor the ~576 MB of upstream results beside them. The two caller-supplied routes still win
where they are set: ``TAU2_DATA_DIR``, or a ``TAU2_SRC`` pointing at a *full* clone's ``src/``,
whose sibling ``data/`` that fallback finds on its own. shogym version-checks neither.
"""

from __future__ import annotations

from pathlib import Path

from shogym.envs._upstream import ensure_data as _ensure_data
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


#: The part of upstream's ``data/`` a tau2 env reads: the five domains, each holding the policy,
#: the task set and the databases the env is made of, and the guidelines every non-solo domain's
#: user simulator is built from. Deliberately not ``data/`` whole, whose ``tau2/results/`` is four
#: times the size of everything here and which no env opens, nor its ``voice/``, which only the
#: voice modes this port does not serve would read.
DATA_SUBTREES = ("tau2/domains", "tau2/user_simulator")


def ensure_data() -> Path:
    """Ensure the pinned domain data is on disk where upstream looks for it; return that dir.

    Idempotent, and not called by :func:`ensure_source`: an env is *constructed* from this data,
    so a machine that has it prepared needs nothing else, and a machine that does not gets the
    same missing-data error from upstream it always did rather than a surprise 93 MB download in
    the middle of a construction. The preparation stage calls it; ``offline`` requires what it
    prepared and says so by name when it is absent."""
    return _ensure_data(
        package="tau2",
        sha=UPSTREAM_SHA,
        tarball_url=_TARBALL_URL,
        archive_subdir="data",
        keep=DATA_SUBTREES,
    )


__all__ = ["DATA_SUBTREES", "UPSTREAM_SHA", "ensure_data", "ensure_source"]
