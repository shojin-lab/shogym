"""Offline unit tests for the ``--cpus`` clamp in the frontier_bench Docker backend (no Docker).

The clamp must bound a task's ``--cpus`` request by the Docker *daemon's* CPU count, not the
client's ``os.cpu_count()``: on Docker Desktop / rootless / a remote daemon the client can report
more CPUs than the daemon enforces, so a client-side clamp still emits a request `docker run`
rejects. These stub the daemon-NCPU query (and ``os.cpu_count``) so the daemon count differs from
the host, and assert the emitted ``--cpus`` uses the daemon value — no real 2-CPU daemon needed.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional, Sequence

import pytest

from hgym.envs.frontier_bench import docker_backend as dk


def _capture_run_args(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    """Patch ``_run_docker`` to record its argv and return a benign success, so ``start`` never
    shells out. Returns the list into which each invocation's args are appended."""
    calls: List[List[str]] = []

    def fake_run_docker(
        args: Sequence[str], *, timeout: Optional[float] = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        calls.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, stdout="", stderr="")

    monkeypatch.setattr(dk, "_run_docker", fake_run_docker)
    return calls


def _emitted_cpus(run_args: List[str]) -> str:
    """Pull the value following ``--cpus`` out of a ``docker run …`` argv."""
    return run_args[run_args.index("--cpus") + 1]


def test_clamp_uses_daemon_count_when_request_exceeds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Daemon caps at 2 while the client host reports 8: a task asking for 4 must be clamped to
    # the daemon's 2, not left at 4 (which the daemon would reject) nor clamped to the host's 8.
    monkeypatch.setattr(dk, "_daemon_cpus", lambda: 2)
    monkeypatch.setattr(dk.os, "cpu_count", lambda: 8)
    calls = _capture_run_args(monkeypatch)

    dk.Container(image="img", cpus=4).start()

    assert _emitted_cpus(calls[0]) == "2"


def test_clamp_keeps_request_when_daemon_is_roomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A roomy daemon (8) still honors the task's full request (1): min(1, 8) == 1.
    monkeypatch.setattr(dk, "_daemon_cpus", lambda: 8)
    monkeypatch.setattr(dk.os, "cpu_count", lambda: 8)
    calls = _capture_run_args(monkeypatch)

    dk.Container(image="img", cpus=1).start()

    assert _emitted_cpus(calls[0]) == "1"


def test_clamp_falls_back_to_client_count_when_daemon_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failed/unparseable `docker info` (None) must not crash the start; fall back to the
    # client's os.cpu_count() so a healthy daemon still starts containers: min(4, 8) == 4.
    monkeypatch.setattr(dk, "_daemon_cpus", lambda: None)
    monkeypatch.setattr(dk.os, "cpu_count", lambda: 8)
    calls = _capture_run_args(monkeypatch)

    dk.Container(image="img", cpus=4).start()

    assert _emitted_cpus(calls[0]) == "4"


def _patch_info(
    monkeypatch: pytest.MonkeyPatch, *, returncode: int, stdout: str
) -> None:
    """Stub ``_run_docker`` so the ``docker info`` query returns a canned NCPU result."""

    def fake_run_docker(
        args: Sequence[str], *, timeout: Optional[float] = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(list(args), returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(dk, "_run_docker", fake_run_docker)
    dk._daemon_cpus.cache_clear()


def test_daemon_cpus_parses_ncpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_info(monkeypatch, returncode=0, stdout="2\n")
    assert dk._daemon_cpus() == 2
    dk._daemon_cpus.cache_clear()


def test_daemon_cpus_none_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_info(monkeypatch, returncode=1, stdout="")
    assert dk._daemon_cpus() is None
    dk._daemon_cpus.cache_clear()


def test_daemon_cpus_none_on_unparseable_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_info(monkeypatch, returncode=0, stdout="not-a-number")
    assert dk._daemon_cpus() is None
    dk._daemon_cpus.cache_clear()


def test_daemon_cpus_none_when_docker_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A missing docker CLI raises DockerError even with check=False; the query must swallow it.
    def raise_docker_error(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess:
        raise dk.DockerError("docker CLI not found on PATH")

    monkeypatch.setattr(dk, "_run_docker", raise_docker_error)
    dk._daemon_cpus.cache_clear()
    assert dk._daemon_cpus() is None
    dk._daemon_cpus.cache_clear()
