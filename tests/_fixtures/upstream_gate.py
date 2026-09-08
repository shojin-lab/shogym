"""The provisioning gate for env tests whose upstream source or task image comes from elsewhere.

Three envs (``tau2``, ``yc_bench``, ``automationbench``) declare no pip dependency on their
upstream: each fetches a SHA-pinned source into ``~/.cache/shogym`` the first time its adapter is
imported. Their test modules therefore have to import a *production* module before they can
collect anything, and that import can fail for two very different reasons:

- **environmental** — the extra isn't installed, or the machine has no network and a cold cache.
  Skipping is right: the offline core suite has to stay green on a laptop with no extras.
- **a regression** — upstream drifted and a symbol the port imports is gone, a dependency is
  missing from a hand-maintained extra, the extraction is corrupt, or shogym's own module has a
  plain ``NameError``. Skipping is *wrong*, and worse than a failing test: every test for that env
  disappears and the run still reports success, so the pin whose entire job is to catch upstream
  drift stops catching it.

:func:`gate` separates the two. Anything it does not positively recognize as environmental is
re-raised, which pytest reports as a collection error.

**Which of those a run is even allowed to do is the provisioning mode**, one variable with three
values, read here and in :mod:`shogym.envs._upstream` from ``SHOGYM_PROVISIONING``:

- ``local`` (the default, and what an unset variable means) is a developer's machine. Sources are
  fetched on demand, and a failure this module recognizes as environmental skips with its reason.
- ``prepare`` is the stage that fetches: pinned sources, upstream data, task images, the Temporal
  test server binary and the packages a vendored oracle installs are all provisioned over the
  network, on purpose, and nothing may skip, so a preparation that did not prepare says so.
- ``offline`` is the stage that runs the suite. Nothing is fetched. A source nobody prepared is a
  hard failure naming it (:class:`~shogym.envs._upstream.UpstreamNotPrepared`), a task image
  nobody prepared is a hard failure naming it (:func:`require_prepared_image`), a prerequisite a
  test checks for itself is a hard failure naming it (:func:`environmental_skip`), and none of
  them skip. A missing extra is a failure here too, which is the point: the machine was prepared,
  so there is nothing left for it to legitimately skip.

The mode is a contract about who downloads, not a sandbox. Its worth is that every download this
project performs has one stage it is allowed in, and a stage that promised to fetch nothing fails
where it would have fetched rather than doing it quietly.

Two skips survive in every mode, because neither is about provisioning: a test that needs an API
key skips without one (the suite runs keyless on purpose), and a Docker-gated test skips where no
daemon is reachable, which is the same machine the preparation stage tells that it built no
images."""

from __future__ import annotations

import http.client
import importlib
import socket
import ssl
import urllib.error
from pathlib import Path
from types import ModuleType
from typing import Iterator, List, NoReturn, Optional

import pytest

from shogym.envs._upstream import LOCAL, OFFLINE, PREPARE, PROVISIONING_ENV_VAR, provisioning_mode

# Failures that mean "this machine cannot reach the tarball", not "this code is broken".
# `socket.gaierror`/`socket.timeout` are `OSError`/`TimeoutError` subclasses; both are listed for
# the reader rather than for the isinstance check.
_NETWORK_ERRORS = (
    urllib.error.URLError,
    ConnectionError,
    TimeoutError,
    socket.gaierror,
    socket.timeout,
    ssl.SSLError,
    http.client.HTTPException,
)


def _may_skip() -> bool:
    """Whether an environmental failure is allowed to skip, which only ``local`` mode allows."""
    return provisioning_mode() == LOCAL


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and everything it was raised from, so a wrapped cause is still recognized."""
    seen: List[int] = []
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.append(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _environmental_reason(exc: BaseException, *, package: str, extra: str) -> Optional[str]:
    """Why this failure is the machine's fault rather than the code's, or ``None`` if it isn't."""
    for cause in _chain(exc):
        if isinstance(cause, urllib.error.HTTPError):
            # A 4xx is the pin or the URL being wrong, which is a bug in this repo. Only a server
            # -side failure (5xx, rate limiting) is environmental.
            if cause.code < 500 and cause.code != 429:
                return None
            return f"{extra}: upstream tarball unavailable (HTTP {cause.code})"
        if isinstance(cause, _NETWORK_ERRORS):
            return f"{extra}: cannot reach the upstream tarball ({cause})"
        if isinstance(cause, ModuleNotFoundError):
            missing = cause.name or ""
            # The upstream package itself, or anything under it, means provisioning did not do its
            # job — that is this repo's bug. So is a missing `shogym` module. Anything else is a
            # third-party dependency the extra was supposed to bring, i.e. the extra is absent.
            if not missing or missing == package or missing.startswith(package + "."):
                return None
            if missing == "shogym" or missing.startswith("shogym."):
                return None
            return f"{extra} extra not installed (no module {missing!r})"
    return None


def environmental_skip(reason: str, *, module_level: bool = False) -> NoReturn:
    """Skip for a reason a laptop is allowed to have, unless this machine was prepared.

    A test that checks its own prerequisite (tau2's domain data, a Temporal test server binary, an
    extra it imports directly) has the same two readings the gate has to separate, and it should
    resolve them the same way. Under ``local`` this is the skip it always was. Under ``prepare``
    or ``offline`` the machine is one the preparation stage was run on, so a missing prerequisite
    is a preparation that did not prepare, and it fails naming what is missing rather than
    deleting the test from the run."""
    mode = provisioning_mode()
    if mode == LOCAL:
        pytest.skip(reason, allow_module_level=module_level)
    pytest.fail(
        f"{reason}. {PROVISIONING_ENV_VAR}={mode} says this machine was prepared, "
        f"so a missing prerequisite is a failure rather than a skip; prepare it with "
        f"{PROVISIONING_ENV_VAR}={PREPARE} (tests/prepare_offline_suite.py).",
        pytrace=False,
    )


def gate(module: str, *, package: str, extra: str) -> ModuleType:
    """Import ``module`` — the shogym module that provisions and imports ``package`` — and return it.

    Skips the calling test module only when the failure is recognizably environmental, and only
    under the ``local`` provisioning mode. Everything else propagates.

    The mode is read before the import, not only after a failure, so a value nobody recognizes is
    refused on the machines where the import succeeds too."""
    provisioning_mode()  # read for its refusal, before an import that may not need it
    try:
        return importlib.import_module(module)
    except BaseException as exc:
        reason = _environmental_reason(exc, package=package, extra=extra)
        if reason is None or not _may_skip():
            raise
        pytest.skip(reason, allow_module_level=True)


def require_prepared_image(tag: str, *, what: str) -> None:
    """Under ``offline``, fail unless the Docker image ``tag`` is already built. Otherwise pass.

    The image gate, and the counterpart of what :func:`gate` does for a source. A frontier_bench
    task image is built from a Dockerfile that installs packages from apt, PyPI and astral.sh, so
    a test that builds one on a cold machine reaches those services however the suite was
    described. The preparation stage builds these images; a stage that runs under ``offline``
    calls this first and fails naming the image it needed, rather than building it and calling
    that offline.

    The check is deliberately only made in ``offline`` mode. Under ``local`` and ``prepare`` a
    build is allowed, so requiring the image up front would refuse the very run that is supposed
    to produce it."""
    if provisioning_mode() != OFFLINE:
        return
    # Imported here so a tau2 or yc_bench test module does not drag the frontier_bench Docker
    # seam in behind its source gate.
    from shogym.envs.frontier_bench import docker_backend as dk

    if not dk.image_exists(tag):
        pytest.fail(
            f"{what} was not prepared: no Docker image {tag!r} on this machine, and "
            f"{PROVISIONING_ENV_VAR}={OFFLINE} forbids the build that would fetch it. Build it "
            f"first in a stage that may ({PROVISIONING_ENV_VAR}={PREPARE})."
        )


def require_prepared_path(path: Path, *, what: str) -> None:
    """Under ``offline``, fail unless ``path`` holds something. Otherwise pass.

    The same gate as :func:`require_prepared_image` for the prepared things that are files rather
    than images: a Temporal test server binary, the packages a vendored oracle installs while it
    runs. An empty directory counts as absent, because a half-published one is not a supply."""
    if provisioning_mode() != OFFLINE:
        return
    empty = path.is_dir() and not any(path.iterdir())
    if not path.exists() or empty:
        pytest.fail(
            f"{what} was not prepared: nothing at {path}, and {PROVISIONING_ENV_VAR}={OFFLINE} "
            f"forbids the download that would fetch it. Fetch it first in a stage that may "
            f"({PROVISIONING_ENV_VAR}={PREPARE}).",
            pytrace=False,
        )


__all__ = [
    "environmental_skip",
    "gate",
    "require_prepared_image",
    "require_prepared_path",
]
