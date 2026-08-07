"""The import gate for env tests whose upstream is provisioned at runtime.

Three envs (``tau2``, ``yc_bench``, ``automationbench``) declare no pip dependency on their
upstream: each fetches a SHA-pinned source into ``~/.cache/shogym`` the first time its adapter is
imported. Their test modules therefore have to import a *production* module before they can
collect anything, and that import can fail for two very different reasons:

- **environmental** — the extra isn't installed, or the machine is offline with a cold cache.
  Skipping is right: the offline core suite has to stay green on a laptop with no extras.
- **a regression** — upstream drifted and a symbol the port imports is gone, a dependency is
  missing from a hand-maintained extra, the extraction is corrupt, or shogym's own module has a
  plain ``NameError``. Skipping is *wrong*, and worse than a failing test: every test for that env
  disappears and the run still reports success, so the pin whose entire job is to catch upstream
  drift stops catching it.

:func:`gate` separates the two. Anything it does not positively recognize as environmental is
re-raised, which pytest reports as a collection error. ``SHOGYM_REQUIRE_UPSTREAM=1`` (set for CI,
where the extras are installed and the network is up) removes the environmental escape hatch too,
so nothing about these envs can go quiet there.
"""

from __future__ import annotations

import http.client
import importlib
import os
import socket
import ssl
import urllib.error
from types import ModuleType
from typing import Iterator, List, Optional

import pytest

REQUIRE_ENV_VAR = "SHOGYM_REQUIRE_UPSTREAM"

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


def _required() -> bool:
    return os.environ.get(REQUIRE_ENV_VAR, "").strip() not in ("", "0", "false", "False")


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


def gate(module: str, *, package: str, extra: str) -> ModuleType:
    """Import ``module`` — the shogym module that provisions and imports ``package`` — and return it.

    Skips the calling test module only when the failure is recognizably environmental, and only
    when ``SHOGYM_REQUIRE_UPSTREAM`` is unset. Everything else propagates."""
    try:
        return importlib.import_module(module)
    except BaseException as exc:
        reason = _environmental_reason(exc, package=package, extra=extra)
        if reason is None or _required():
            raise
        pytest.skip(reason, allow_module_level=True)


__all__ = ["REQUIRE_ENV_VAR", "gate"]
