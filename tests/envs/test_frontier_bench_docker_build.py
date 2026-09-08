"""Offline unit tests for what a frontier_bench run is allowed to fetch (no Docker needed).

A task image is built from a Dockerfile that installs packages from apt, PyPI and astral.sh, so
"the offline suite" is only true of a run that builds none of them. Two things make that so, and
both are asserted here by capturing the ``docker`` argv rather than running it: a verifier image
that is already built is used as it is, which is what lets a prepared machine finalize an episode
without a build at all; and a build under ``SHOGYM_PROVISIONING=offline`` is given no network, so
one that happens anyway stops at its first networked ``RUN`` rather than reaching a third party.
That flag governs the ``RUN`` steps and not the builder's own resolution of the ``FROM`` image, so
a base image this machine lacks would still be pulled: requiring the built images is the
guarantee, and the flag is the backstop behind it.

The third fetch is not a build at all: one vendored oracle installs its own packages *inside* the
task container while it solves. No image can hold them without giving the agent a library upstream
does not, so what a prepared machine holds is the packages themselves, and the middle tests here
are that an offline oracle is pointed at that supply and told to consult no index.

The last group is about the mode rather than about any one resource: every entry point above reads
it before it returns what a prepared machine already holds, so a value nobody recognizes is
refused there too.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, NoReturn, Optional, Sequence

import pytest

from shogym.envs._upstream import LOCAL, OFFLINE, PREPARE, PROVISIONING_ENV_VAR
from shogym.envs.frontier_bench import docker_backend as dk
from shogym.envs.frontier_bench import manifest, mcp_server


def _capture_run_args(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    """Patch ``_run_docker`` to record its argv and return a benign success, so nothing shells
    out. Returns the list into which each invocation's args are appended."""
    calls: List[List[str]] = []

    def fake_run_docker(
        args: Sequence[str], *, timeout: Optional[float] = None, check: bool = True
    ) -> subprocess.CompletedProcess:
        calls.append(list(args))
        return subprocess.CompletedProcess(list(args), 0, stdout="", stderr="")

    monkeypatch.setattr(dk, "_run_docker", fake_run_docker)
    return calls


def _build(monkeypatch: pytest.MonkeyPatch) -> List[str]:
    """Build one image with the current mode and return the emitted ``docker build`` argv."""
    calls = _capture_run_args(monkeypatch)
    task = manifest.load_task("fin-saccr-rwa")
    dk.build_image(
        context_dir=task.environment_dir,
        dockerfile=task.environment_dockerfile,
        tag="shogym-frontier-test:build",
        timeout=task.build_timeout_sec,
    )
    assert len(calls) == 1
    return calls[0]


def test_an_offline_build_is_given_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nine of the ten vendored Dockerfiles install over the network, so a build in a stage that
    promised not to fetch has to be told, not trusted."""
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    args = _build(monkeypatch)
    assert args[args.index("--network") + 1] == "none"
    # The context is still the last argument, and the platform pin still stands.
    assert args[-1].endswith("environment")
    assert args[args.index("--platform") + 1] == "linux/amd64"


@pytest.mark.parametrize("mode", [PREPARE, LOCAL, None])
def test_every_other_mode_builds_with_the_network_it_always_had(
    monkeypatch: pytest.MonkeyPatch, mode: Optional[str]
) -> None:
    """Requiring the network to be absent outside ``offline`` would refuse the very builds that
    are supposed to produce the images, on a developer's machine and in preparation alike."""
    if mode is None:
        monkeypatch.delenv(PROVISIONING_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(PROVISIONING_ENV_VAR, mode)
    assert "--network" not in _build(monkeypatch)


def test_a_prepared_verifier_image_is_used_rather_than_rebuilt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier image is content-addressed by the task's own bytes, so an image already under
    that tag was built from them: the prepared one is the right one."""
    builds: List[Dict[str, Any]] = []
    monkeypatch.setattr(dk, "image_exists", lambda tag: True)
    monkeypatch.setattr(dk, "build_image", lambda **kwargs: builds.append(kwargs))

    tag = mcp_server.build_verifier_image("fin-saccr-rwa")

    assert tag.startswith("shogym-frontier-verifier-fin-saccr-rwa:")
    assert builds == [], "a prepared image must not be rebuilt"


def test_a_finalization_that_may_not_fetch_keeps_the_verifier_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Removing it would leave the next episode of the task needing a build this run may not do,
    so under ``offline`` the prepared image stays. Every other mode removes it as it always has."""
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    assert mcp_server.keeps_verifier_image(False) is True

    for mode in (PREPARE, LOCAL):
        monkeypatch.setenv(PROVISIONING_ENV_VAR, mode)
        assert mcp_server.keeps_verifier_image(False) is False
        # A caller that asked to keep its container still keeps the image, in every mode.
        assert mcp_server.keeps_verifier_image(True) is True


def test_an_unprepared_verifier_image_is_built_from_the_tasks_tests_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builds: List[Dict[str, Any]] = []
    monkeypatch.setattr(dk, "image_exists", lambda tag: False)
    monkeypatch.setattr(dk, "build_image", lambda **kwargs: builds.append(kwargs))

    tag = mcp_server.build_verifier_image("fin-saccr-rwa")

    task = manifest.load_task("fin-saccr-rwa")
    assert [b["tag"] for b in builds] == [tag]
    assert builds[0]["context_dir"] == task.tests_dir
    assert builds[0]["dockerfile"] == task.tests_dockerfile


# ----- the download that happens inside the container: an oracle installing its own packages -----


def test_an_oracles_own_requirements_are_read_out_of_its_script() -> None:
    """Read from the vendored ``solve.sh`` rather than listed here, so a re-pin that changes what
    an oracle installs changes what the preparation stage fetches, with no list to forget."""
    assert mcp_server.oracle_pip_requirements(manifest.load_task("fin-saccr-rwa")) == [
        "openpyxl==3.1.5"
    ]
    assert mcp_server.oracle_pip_requirements(
        manifest.load_task("protein-autointerp-disulfide")
    ) == ["requests==2.32.4", "biopython==1.85"]
    # The other three oracles install nothing, so there is nothing to prepare for them.
    assert mcp_server.oracle_pip_requirements(manifest.load_task("interleaved-vigenere")) == []


def test_an_offline_oracle_installs_from_the_prepared_supply_and_no_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PIP_NO_INDEX`` rather than a preference: the install cannot reach PyPI, it can only find
    what the preparation stage put beside it, and a supply nobody prepared fails the oracle."""
    task = manifest.load_task("fin-saccr-rwa")
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)

    environment = mcp_server.oracle_install_environment(task)

    assert environment == {
        "PIP_NO_INDEX": "1",
        "PIP_FIND_LINKS": mcp_server.ORACLE_PACKAGES_PATH,
    }


@pytest.mark.parametrize("mode", [PREPARE, LOCAL, None])
def test_every_other_mode_lets_the_oracle_install_the_way_upstream_wrote_it(
    monkeypatch: pytest.MonkeyPatch, mode: Optional[str]
) -> None:
    """The oracle is the task's own bytes. Where a download is allowed it runs untouched."""
    if mode is None:
        monkeypatch.delenv(PROVISIONING_ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(PROVISIONING_ENV_VAR, mode)
    assert mcp_server.oracle_install_environment(manifest.load_task("fin-saccr-rwa")) == {}


def test_an_oracle_that_installs_nothing_is_given_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    assert mcp_server.oracle_install_environment(manifest.load_task("interleaved-vigenere")) == {}


def test_prepared_packages_are_addressed_by_the_tasks_content_hash() -> None:
    """The same address the image tags use, so a re-pinned task looks for its own packages rather
    than installing the previous pin's."""
    task = manifest.load_task("fin-saccr-rwa")
    assert mcp_server.oracle_packages_dir(task.name).name == task.content_sha256[:12]
    assert task.name in str(mcp_server.oracle_packages_dir(task.name))


# ----- the mode itself, on the path a prepared machine takes -----


def _never(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("a warm entry point reached for a fetch instead of what was prepared")


def _everything_prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make every resource these entry points look for present, and every fetch fatal.

    What is left for them to raise is the mode, which is the whole point of the assertions below:
    the warm path is the one a prepared machine takes, so it is the one a typo has to be refused
    on. A read that only happened where a build or a download follows would be found by the
    machines with nothing prepared and by no others."""
    supply = tmp_path / "oracle-packages"
    supply.mkdir()
    (supply / "openpyxl-3.1.5-py2.py3-none-any.whl").write_bytes(b"")
    monkeypatch.setattr(dk, "image_exists", lambda tag: True)
    monkeypatch.setattr(dk, "build_image", _never)
    monkeypatch.setattr(dk, "Container", _never)
    monkeypatch.setattr(mcp_server, "oracle_packages_dir", lambda task_name=None: supply)


#: Every frontier_bench entry point whose answer a prepared machine already has on disk, so each
#: returns before it reaches the build or the download that would otherwise read the mode.
_WARM_ENTRY_POINTS: Dict[str, Callable[[], object]] = {
    "build_task_image": lambda: mcp_server.build_task_image("fin-saccr-rwa"),
    "build_verifier_image": lambda: mcp_server.build_verifier_image("fin-saccr-rwa"),
    "prepare_oracle_packages": lambda: mcp_server.prepare_oracle_packages("fin-saccr-rwa"),
    "oracle_install_environment": lambda: mcp_server.oracle_install_environment(
        manifest.load_task("fin-saccr-rwa")
    ),
    # `True` is what this answers without consulting the mode at all unless the read comes first,
    # so it is the case that catches a short circuit.
    "keeps_verifier_image": lambda: mcp_server.keeps_verifier_image(True),
}


@pytest.mark.parametrize("entry_point", sorted(_WARM_ENTRY_POINTS))
def test_a_warm_entry_point_refuses_an_unrecognized_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry_point: str
) -> None:
    """A typo that quietly restored on demand downloading has to be refused where it is read, and
    a prepared machine reads it here."""
    _everything_prepared(monkeypatch, tmp_path)
    monkeypatch.setenv(PROVISIONING_ENV_VAR, "ofline")

    with pytest.raises(RuntimeError, match=PROVISIONING_ENV_VAR):
        _WARM_ENTRY_POINTS[entry_point]()


@pytest.mark.parametrize("entry_point", sorted(_WARM_ENTRY_POINTS))
def test_a_recognized_mode_still_answers_from_what_was_prepared(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry_point: str
) -> None:
    """The other half of the read: it refuses a value nobody recognizes and changes nothing else.

    Every fetch is fatal here, so each of these answering at all is the assertion that a prepared
    machine still finalizes, solves and builds nothing."""
    _everything_prepared(monkeypatch, tmp_path)
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)

    assert _WARM_ENTRY_POINTS[entry_point]()
