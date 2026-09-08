"""The fetch-and-import machinery three envs depend on, and the gate that must not hide its breakage.

Two layers, both of which exist because a *silent* wrong answer is the failure mode here:

- **Offline unit tests of** :mod:`shogym.envs._upstream`. They provision from a ``file://`` tarball
  built in a temp dir, so they run in the core suite with no network and no extras. They pin the
  properties the ports are entitled to assume: only the upstream package is extracted, nothing
  reaches ``sys.path``, a name already bound to a *different* package is refused rather than
  reported as success, a half-torn-down registration leaves nothing behind to shadow the retry, and
  a process that loses the publish race reuses the winner instead of raising.
- **A live provisioning assertion for every port.** The per-env test modules each gate on their own
  upstream; this one asserts all three at once, so "the fetch-and-import path works" is a claim the
  suite makes in one identifiable place. It uses the same classifier, so an offline laptop still
  skips while a prepared run (``SHOGYM_PROVISIONING=offline``) cannot.

The unit tests provision a *fake* upstream out of a temp dir, so :func:`_ensure` pins the
provisioning mode to ``local`` for the length of the call: what they exercise is the machinery,
under whichever mode the suite around them happens to run. The mode's own rules are asserted
separately, below, including at the two entry points that read it outside this module: the gate,
and the preparation of the Temporal test server binary.
"""

from __future__ import annotations

import asyncio
import errno
import fcntl
import json
import os
import subprocess
import sys
import tarfile
import urllib.error
import warnings
from pathlib import Path
from typing import Iterator, List

import pytest

from shogym.envs import _upstream
from shogym.envs._upstream import LOCAL, OFFLINE, PREPARE, PROVISIONING_ENV_VAR
from tests._fixtures import temporal_server
from tests._fixtures.upstream_gate import _environmental_reason, gate

_PORTS = [
    ("tau2", "shogym.envs.tau2.mcp_server", "tau2"),
    ("yc_bench", "shogym.envs.yc_bench.adapter", "yc_bench"),
    ("automationbench", "shogym.envs.automationbench.adapter", "automationbench"),
]


@pytest.fixture(autouse=True)
def _no_module_leaks() -> Iterator[None]:
    """Unbind anything a test registered, so one test's fake package can't reach another."""
    before = set(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - before:
            if name.startswith("demo"):
                del sys.modules[name]


def _make_archive(
    tmp_path: Path,
    *,
    package: str,
    archive_subdir: str = "",
    body: str = "VALUE = 'pinned'\n",
    siblings: List[str] = ["tests", "docs"],
) -> str:
    """Build a GitHub-shaped ``<root>/[<subdir>/]<package>/`` tarball; return its ``file://`` URL."""
    root = tmp_path / "build" / f"{package}-repo-sha"
    pkg = root / archive_subdir / package if archive_subdir else root / package
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(body)
    (pkg / "sub.py").write_text("NAME = 'sub'\n")
    for sibling in siblings:  # the dirs that must never reach an import path
        (root / sibling).mkdir(parents=True, exist_ok=True)
        (root / sibling / "__init__.py").write_text("SHOULD_NOT_BE_EXTRACTED = True\n")
    archive = tmp_path / f"{package}.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(root, arcname=root.name)
    return archive.as_uri()


def _ensure(tmp_path: Path, package: str, **kwargs: object) -> Path:
    """Provision ``package`` into a cache under ``tmp_path``, in ``local`` mode.

    The mode is pinned because these tests are about the machinery, not about who may fetch: the
    archive is a ``file://`` URL in a temp dir, so "downloading" it costs nothing and reaches
    nobody, and a suite running under ``offline`` would otherwise refuse it for a reason that has
    no bearing on what is being asserted."""
    outer_mode = os.environ.get(PROVISIONING_ENV_VAR)
    os.environ["SHOGYM_CACHE"] = str(tmp_path / "cache")
    os.environ[PROVISIONING_ENV_VAR] = LOCAL
    try:
        return _upstream.ensure_package(package=package, sha="sha1", **kwargs)  # type: ignore[arg-type]
    finally:
        os.environ.pop("SHOGYM_CACHE", None)
        # Restored, not cleared: the suite around this may be running under a mode of its own,
        # and dropping it here would quietly hand every later test the default instead.
        if outer_mode is None:
            os.environ.pop(PROVISIONING_ENV_VAR, None)
        else:
            os.environ[PROVISIONING_ENV_VAR] = outer_mode


# ----- extraction + binding -----


def test_only_the_package_is_extracted_and_nothing_reaches_sys_path(tmp_path: Path) -> None:
    url = _make_archive(tmp_path, package="demo_flat")
    before = list(sys.path)
    src = _ensure(tmp_path, "demo_flat", tarball_url=url)

    assert [p.name for p in src.iterdir()] == ["demo_flat"], "siblings must not be kept"
    assert sys.path == before, "provisioning must never append to sys.path"
    assert sys.modules["demo_flat"].VALUE == "pinned"  # type: ignore[attr-defined]

    import demo_flat.sub  # noqa: F401 — resolves through the registered __path__

    assert sys.modules["demo_flat.sub"].NAME == "sub"  # type: ignore[attr-defined]


def test_src_layout_upstreams_are_found_under_the_subdir(tmp_path: Path) -> None:
    url = _make_archive(tmp_path, package="demo_src", archive_subdir="src")
    src = _ensure(tmp_path, "demo_src", tarball_url=url, archive_subdir="src")
    assert (src / "demo_src" / "__init__.py").is_file()


def test_an_archive_without_the_package_raises(tmp_path: Path) -> None:
    url = _make_archive(tmp_path, package="demo_elsewhere", archive_subdir="src")
    with pytest.raises(RuntimeError, match="archive layout"):
        _ensure(tmp_path, "demo_elsewhere", tarball_url=url)  # no archive_subdir


# ----- the name-collision refusal (the reason fetch-and-import exists at all) -----


def test_a_different_package_already_bound_to_the_name_is_refused(tmp_path: Path) -> None:
    """An app that imported PyPI's unrelated `tau2` first must not be told provisioning worked."""
    url = _make_archive(tmp_path, package="demo_taken")
    unpinned = tmp_path / "unpinned" / "demo_taken"
    unpinned.mkdir(parents=True)
    (unpinned / "__init__.py").write_text("VALUE = 'the wrong project'\n")
    sys.path.insert(0, str(tmp_path / "unpinned"))
    try:
        import demo_taken  # noqa: F401 — binds the name to the wrong package

        with pytest.raises(RuntimeError, match="already imported in this process"):
            _ensure(tmp_path, "demo_taken", tarball_url=url)
        # and the refusal did not quietly swap what the app is holding
        assert sys.modules["demo_taken"].VALUE == "the wrong project"  # type: ignore[attr-defined]
    finally:
        sys.path.remove(str(tmp_path / "unpinned"))


def test_rebinding_the_same_pinned_source_is_idempotent(tmp_path: Path) -> None:
    url = _make_archive(tmp_path, package="demo_twice")
    first = _ensure(tmp_path, "demo_twice", tarball_url=url)
    module = sys.modules["demo_twice"]
    assert _ensure(tmp_path, "demo_twice", tarball_url=url) == first
    assert sys.modules["demo_twice"] is module


def test_a_stale_submodule_cannot_shadow_the_pinned_package(tmp_path: Path) -> None:
    """A leftover `<pkg>.x` is returned by import without consulting the parent's __path__."""
    url = _make_archive(tmp_path, package="demo_stale")
    sys.modules["demo_stale.sub"] = type(sys)("demo_stale.sub")
    sys.modules["demo_stale.sub"].NAME = "hijacked"  # type: ignore[attr-defined]
    _ensure(tmp_path, "demo_stale", tarball_url=url)

    import demo_stale.sub  # noqa: F401

    assert sys.modules["demo_stale.sub"].NAME == "sub"  # type: ignore[attr-defined]


def test_a_registration_that_raises_leaves_nothing_bound(tmp_path: Path) -> None:
    url = _make_archive(
        tmp_path,
        package="demo_boom",
        body="import demo_boom.sub\nraise ValueError('upstream exploded')\n",
    )
    with pytest.raises(ValueError, match="upstream exploded"):
        _ensure(tmp_path, "demo_boom", tarball_url=url)
    assert "demo_boom" not in sys.modules
    assert not [n for n in sys.modules if n.startswith("demo_boom.")]


# ----- the cross-process publish race -----


def test_losing_the_publish_race_reuses_the_winner(tmp_path: Path, monkeypatch) -> None:
    """A peer publishes between this call staging its tree and renaming it into place.

    That is the interleaving an inter-process lock makes rare and a check-then-rename made fatal:
    the loser's ``os.replace`` hits a non-empty directory and fails with ``ENOTEMPTY``."""
    url = _make_archive(tmp_path, package="demo_race")
    dest = tmp_path / "cache" / "demo_race" / "sha1"

    real_replace = os.replace
    fired: List[int] = []

    def racing_replace(src, dst, **kwargs):  # type: ignore[no-untyped-def]
        if Path(dst) == dest and not fired:  # the publish rename, about to lose
            fired.append(1)
            (dest / "demo_race").mkdir(parents=True)
            (dest / "demo_race" / "__init__.py").write_text("VALUE = 'winner'\n")
        return real_replace(src, dst, **kwargs)

    monkeypatch.setattr(_upstream.os, "replace", racing_replace)
    _upstream._download_package("demo_race", url, "", dest)  # must not raise
    assert fired, "the race was never triggered"
    assert (dest / "demo_race" / "__init__.py").read_text() == "VALUE = 'winner'\n"


def test_a_genuinely_broken_publish_still_raises(tmp_path: Path, monkeypatch) -> None:
    """The race handler must not swallow a real rename failure into a half-published tree."""
    url = _make_archive(tmp_path, package="demo_broken")
    dest = tmp_path / "cache" / "demo_broken" / "sha1"
    dest.mkdir(parents=True)
    (dest / "not-a-package").write_text("")  # non-empty, but holds no package

    with pytest.raises(OSError):
        _upstream._download_package("demo_broken", url, "", dest)


def test_download_residue_from_a_killed_provisioner_is_swept(tmp_path: Path) -> None:
    url = _make_archive(tmp_path, package="demo_sweep")
    cache = tmp_path / "cache" / "demo_sweep"
    corpse = cache / ".dl-deadbeef"
    corpse.mkdir(parents=True)
    (corpse / "archive.tar.gz").write_bytes(b"partial download")
    _ensure(tmp_path, "demo_sweep", tarball_url=url)
    assert not list(cache.glob(".dl-*"))


def test_the_sweep_never_touches_a_live_provisioners_staging_dir(tmp_path: Path) -> None:
    """The sweep's liveness test is a lock another process holds, not a guess about age.

    Without this the sweep is a foot-gun: whoever enters the critical section first would delete
    the half-downloaded archives of everyone currently fetching, which is precisely what happens
    if the outer provisioning lock is ever unavailable."""
    cache = tmp_path / "cache" / "demo_live"
    live = cache / ".dl-inuse"
    live.mkdir(parents=True)
    dead = cache / ".dl-corpse"
    dead.mkdir(parents=True)

    # A separate process holding the staging lock, exactly as `_download_package` does.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import fcntl, os, sys;"
            "fd = os.open(sys.argv[1], os.O_RDONLY);"
            "fcntl.flock(fd, fcntl.LOCK_EX);"
            "print('locked', flush=True);"
            "sys.stdin.read()",
            str(live),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline().strip() == "locked"
        _upstream._sweep_download_residue(cache)
        assert live.is_dir(), "a live provisioner's staging dir must survive the sweep"
        assert not dead.exists(), "an abandoned staging dir must not"
    finally:
        holder.communicate("bye\n", timeout=30)


# ----- a filesystem that cannot lock at all -----


@pytest.fixture
def _unwarned() -> Iterator[None]:
    """The unlocked-filesystem warning fires once per process; let each test see its own."""
    _upstream._warned_unlocked = False
    try:
        yield
    finally:
        _upstream._warned_unlocked = False


def _flock_raising(code: int):
    def _raise(fd: int, operation: int) -> None:
        raise OSError(code, os.strerror(code))

    return _raise


@pytest.mark.parametrize(
    "code",
    sorted(_upstream._LOCK_UNSUPPORTED),
    ids=lambda c: errno.errorcode.get(c, str(c)),
)
def test_provisioning_completes_when_the_filesystem_cannot_flock(
    tmp_path: Path, monkeypatch, _unwarned: None, code: int
) -> None:
    """A cache on a mount with no locking still provisions, because the lock is an optimization.

    Both the outer cache lock and the per-staging lock go through the same helper, so this covers
    them together: an unsupported errno leaves the run without exclusion but with everything that
    actually carries correctness intact — the single atomic publish rename, and a sweep that
    refuses to delete what it cannot lock."""
    url = _make_archive(tmp_path, package="demo_nolockfs")
    monkeypatch.setattr(fcntl, "flock", _flock_raising(code))

    with pytest.warns(RuntimeWarning, match="cannot provide flock"):
        src = _ensure(tmp_path, "demo_nolockfs", tarball_url=url)

    assert (src / "demo_nolockfs" / "__init__.py").is_file()
    assert sys.modules["demo_nolockfs"].VALUE == "pinned"  # type: ignore[attr-defined]
    assert not list(src.parent.glob(".dl-*")), "the staging dir is still cleaned up normally"


def test_the_unlocked_warning_is_said_once_not_per_call(
    tmp_path: Path, monkeypatch, _unwarned: None
) -> None:
    monkeypatch.setattr(fcntl, "flock", _flock_raising(errno.EOPNOTSUPP))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name in ("demo_once_a", "demo_once_b"):
            _ensure(tmp_path, name, tarball_url=_make_archive(tmp_path, package=name))
    assert len([w for w in caught if issubclass(w.category, RuntimeWarning)]) == 1


@pytest.mark.parametrize("code", [errno.EPERM, errno.EBADF, errno.EINVAL])
def test_a_lock_error_that_is_not_about_support_still_propagates(
    tmp_path: Path, monkeypatch, _unwarned: None, code: int
) -> None:
    """Degrading on everything would hide a real bug behind a silent loss of exclusion.

    ``EINVAL`` is in here deliberately: it is what a wrong ``operation`` argument returns, so it
    stays loud even though some filesystems also use it to mean unsupported."""
    url = _make_archive(tmp_path, package="demo_badlock")
    monkeypatch.setattr(fcntl, "flock", _flock_raising(code))
    with pytest.raises(OSError) as raised:
        _ensure(tmp_path, "demo_badlock", tarball_url=url)
    assert raised.value.errno == code


def test_the_sweep_leaves_residue_alone_when_it_cannot_lock(
    tmp_path: Path, monkeypatch, _unwarned: None
) -> None:
    """No locking means no way to tell a corpse from a peer, so nothing may be deleted.

    A cache that grows is the right side of that trade: the alternative is deleting a live
    provisioner's in-flight download."""
    cache = tmp_path / "cache" / "demo_noreclaim"
    corpse = cache / ".dl-corpse"
    corpse.mkdir(parents=True)
    (corpse / "archive.tar.gz").write_bytes(b"partial download")
    monkeypatch.setattr(fcntl, "flock", _flock_raising(errno.EOPNOTSUPP))

    _upstream._sweep_download_residue(cache)

    assert corpse.is_dir(), "an unreclaimable corpse must be left, never guessed at"


# ----- the source override -----


def test_the_override_env_var_skips_the_fetch(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout" / "demo_override"
    checkout.mkdir(parents=True)
    (checkout / "__init__.py").write_text("VALUE = 'local'\n")
    os.environ["DEMO_OVERRIDE_SRC"] = str(tmp_path / "checkout")
    try:
        src = _upstream.ensure_package(
            package="demo_override", sha="sha1", tarball_url="http://unreachable.invalid/x.tar.gz"
        )
        assert src == (tmp_path / "checkout").resolve()
        assert sys.modules["demo_override"].VALUE == "local"  # type: ignore[attr-defined]
    finally:
        os.environ.pop("DEMO_OVERRIDE_SRC", None)


def test_a_wrong_override_says_so_instead_of_fetching(tmp_path: Path) -> None:
    os.environ["DEMO_WRONG_SRC"] = str(tmp_path / "empty")
    try:
        with pytest.raises(RuntimeError, match="DEMO_WRONG_SRC"):
            _upstream.ensure_package(
                package="demo_wrong", sha="sha1", tarball_url="http://unreachable.invalid/x.tar.gz"
            )
    finally:
        os.environ.pop("DEMO_WRONG_SRC", None)


# ----- the provisioning mode: who may fetch, and what a stage that may not does instead -----


def test_an_unset_mode_is_local_and_the_spelling_is_forgiving(monkeypatch) -> None:
    monkeypatch.delenv(PROVISIONING_ENV_VAR, raising=False)
    assert _upstream.provisioning_mode() == LOCAL
    monkeypatch.setenv(PROVISIONING_ENV_VAR, " Offline ")
    assert _upstream.provisioning_mode() == OFFLINE


@pytest.mark.parametrize("value", ["ofline", "1", "true", "no-download"])
def test_an_unrecognized_mode_is_refused_rather_than_defaulted(monkeypatch, value: str) -> None:
    """A typo must not read as the default.

    Silently falling back to ``local`` would give a stage that was configured to fetch nothing
    the fetching behavior it was configured out of, and the only evidence would be a download
    nobody was watching for."""
    monkeypatch.setenv(PROVISIONING_ENV_VAR, value)
    with pytest.raises(RuntimeError, match=PROVISIONING_ENV_VAR):
        _upstream.provisioning_mode()


def test_prepare_mode_provisions_like_the_default(tmp_path: Path, monkeypatch) -> None:
    """The stage that fetches, fetching. It is the default's behavior under another name, and it
    exists so the *test gate* can tell a prepared machine from an unprepared one."""
    url = _make_archive(tmp_path, package="demo_prepare_mode")
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, PREPARE)

    src = _upstream.ensure_package(package="demo_prepare_mode", sha="sha1", tarball_url=url)

    assert (src / "demo_prepare_mode" / "__init__.py").is_file()


def test_offline_mode_refuses_to_provision_what_was_not_prepared(
    tmp_path: Path, monkeypatch
) -> None:
    """The refusal is the mode's, not the network's: this tarball is right there and readable.

    That is the whole difference between this and a cache that downloads on a miss. The error
    names the package, the directory it was looked for in, and both ways out, because "not
    prepared" is only actionable if it says what was not prepared."""
    url = _make_archive(tmp_path, package="demo_unprepared")
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)

    with pytest.raises(_upstream.UpstreamNotPrepared) as raised:
        _upstream.ensure_package(package="demo_unprepared", sha="sha1", tarball_url=url)

    message = str(raised.value)
    assert "demo_unprepared" in message
    assert str(tmp_path / "cache") in message
    assert f"{PROVISIONING_ENV_VAR}={PREPARE}" in message
    assert "DEMO_UNPREPARED_SRC" in message
    assert not (tmp_path / "cache" / "demo_unprepared" / "sha1").exists(), "nothing was fetched"


def test_offline_mode_binds_a_source_that_was_prepared(tmp_path: Path, monkeypatch) -> None:
    """The other half: what preparation bought is used, with an unreachable URL to prove it."""
    url = _make_archive(tmp_path, package="demo_prepared")
    _ensure(tmp_path, "demo_prepared", tarball_url=url)  # the preparation stage
    sys.modules.pop("demo_prepared", None)  # a later process starts with nothing bound

    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    src = _upstream.ensure_package(
        package="demo_prepared", sha="sha1", tarball_url="http://unreachable.invalid/x.tar.gz"
    )

    assert (src / "demo_prepared" / "__init__.py").is_file()
    assert sys.modules["demo_prepared"].VALUE == "pinned"  # type: ignore[attr-defined]


def test_a_prepared_source_still_refuses_an_unrecognized_mode(tmp_path: Path, monkeypatch) -> None:
    """The warm path reads the mode too.

    A typo that only mattered on a cold cache would be found by whichever machine happened to
    need a download and by no other, which is the same as not refusing it: the machines that
    matter here are the prepared ones, and they take the branch that returns."""
    url = _make_archive(tmp_path, package="demo_warm_mode")
    _ensure(tmp_path, "demo_warm_mode", tarball_url=url)
    sys.modules.pop("demo_warm_mode", None)

    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, "ofline")
    with pytest.raises(RuntimeError, match=PROVISIONING_ENV_VAR):
        _upstream.ensure_package(
            package="demo_warm_mode", sha="sha1", tarball_url="http://unreachable.invalid/x.tar.gz"
        )


def test_a_prepared_temporal_test_server_still_refuses_an_unrecognized_mode(
    tmp_path: Path, monkeypatch
) -> None:
    """The binary the durable tests run on is prepared like a source, and reads the mode like one.

    Its preparation lives in a test fixture rather than in shogym, but what it obeys is this
    module's contract, so the assertion is here with the rest of it. The file being present is the
    warm path, and the first call is what shows that a recognized mode returns it rather than
    starting a download."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    prepared = temporal_server.test_server_path()
    prepared.parent.mkdir(parents=True, exist_ok=True)
    prepared.write_bytes(b"")

    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    assert asyncio.run(temporal_server.prepare_test_server()) == prepared

    monkeypatch.setenv(PROVISIONING_ENV_VAR, "ofline")
    with pytest.raises(RuntimeError, match=PROVISIONING_ENV_VAR):
        asyncio.run(temporal_server.prepare_test_server())


def test_binding_under_offline_selects_a_librarys_bundled_data_over_its_download(
    tmp_path: Path, monkeypatch
) -> None:
    """Binding a prepared source is where the *import* happens, and an import can fetch too.

    LiteLLM downloads its model-cost map the moment it is imported unless this variable says to
    read the copy inside its own wheel, and two of the three ports reach LiteLLM as they import:
    tau2 through upstream's own package ``__init__``, yc_bench through the upstream module its
    adapter imports once the source is bound. So the mode sets it here, before the source is
    registered and either chain starts, and it is set in the environment rather than in a copy so
    the subprocesses a port launches inherit it."""
    url = _make_archive(tmp_path, package="demo_import_data")
    _ensure(tmp_path, "demo_import_data", tarball_url=url)
    sys.modules.pop("demo_import_data", None)

    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

    _upstream.ensure_package(package="demo_import_data", sha="sha1", tarball_url=url)

    assert os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"


def test_binding_under_the_other_modes_leaves_that_environment_alone(
    tmp_path: Path, monkeypatch
) -> None:
    """A developer's machine keeps the library's own default, which is what it always had."""
    url = _make_archive(tmp_path, package="demo_import_data_local")
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, LOCAL)
    monkeypatch.delenv("LITELLM_LOCAL_MODEL_COST_MAP", raising=False)

    _upstream.ensure_package(package="demo_import_data_local", sha="sha1", tarball_url=url)

    assert "LITELLM_LOCAL_MODEL_COST_MAP" not in os.environ


# ----- upstream data: the part that is read rather than imported -----


def _make_data_archive(tmp_path: Path, *, present: List[str]) -> str:
    """Build a GitHub-shaped archive holding ``present`` directories; return its ``file://`` URL."""
    root = tmp_path / "build" / "demo-repo-sha"
    for relative in present:
        directory = root / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "tasks.json").write_text('{"pinned": true}\n')
    archive = tmp_path / "data.tar.gz"
    with tarfile.open(archive, "w:gz") as tf:
        tf.add(root, arcname=root.name)
    return archive.as_uri()


def _ensure_data(package: str, url: str, keep: List[str], sha: str = "sha1") -> Path:
    """Provision ``package``'s data from ``url`` under whatever mode the test has set."""
    return _upstream.ensure_data(
        package=package, sha=sha, tarball_url=url, archive_subdir="data", keep=keep
    )


def test_only_the_data_subtrees_a_port_asked_for_are_extracted(tmp_path: Path, monkeypatch) -> None:
    """tau2's ``data/`` is ~700 MB of which its envs read 139, so the data is a list of subtrees
    rather than a directory."""
    url = _make_data_archive(
        tmp_path, present=["data/domains/mock", "data/results/huge", "docs/manual"]
    )
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, PREPARE)

    dest = _ensure_data("demo_data", url, ["domains/mock"])

    assert dest == tmp_path / "cache" / "demo_data" / "data"
    assert (dest / "domains" / "mock" / "tasks.json").is_file()
    assert not (dest / "results").exists()
    assert (dest / _upstream.DATA_SHA_FILE).read_text().strip() == "sha1"


def test_offline_refuses_data_that_was_not_prepared(tmp_path: Path, monkeypatch) -> None:
    """Naming what is missing, as a source does, because a skipped env is a deleted env."""
    url = _make_data_archive(tmp_path, present=["data/domains/mock"])
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)

    with pytest.raises(_upstream.UpstreamNotPrepared) as raised:
        _ensure_data("demo_data_absent", url, ["domains/mock"])

    message = str(raised.value)
    assert "demo_data_absent" in message
    assert str(tmp_path / "cache" / "demo_data_absent" / "data") in message
    assert f"{PROVISIONING_ENV_VAR}={PREPARE}" in message
    assert not (tmp_path / "cache" / "demo_data_absent").exists(), "nothing was fetched"


def test_prepared_data_is_used_again_without_fetching(tmp_path: Path, monkeypatch) -> None:
    """The other half, with an unreachable URL as the proof, as the source path has."""
    url = _make_data_archive(tmp_path, present=["data/domains/mock"])
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, PREPARE)
    _ensure_data("demo_data_warm", url, ["domains/mock"])

    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    dest = _upstream.ensure_data(
        package="demo_data_warm",
        sha="sha1",
        tarball_url="http://unreachable.invalid/x.tar.gz",
        archive_subdir="data",
        keep=["domains/mock"],
    )

    assert (dest / "domains" / "mock" / "tasks.json").is_file()


def test_prepared_data_still_refuses_an_unrecognized_mode(tmp_path: Path, monkeypatch) -> None:
    """The data path has the warm return a source has, and reads the mode in front of it too."""
    url = _make_data_archive(tmp_path, present=["data/domains/mock"])
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, PREPARE)
    _ensure_data("demo_data_mode", url, ["domains/mock"])

    monkeypatch.setenv(PROVISIONING_ENV_VAR, "ofline")
    with pytest.raises(RuntimeError, match=PROVISIONING_ENV_VAR):
        _ensure_data("demo_data_mode", "http://unreachable.invalid/x.tar.gz", ["domains/mock"])


def test_a_subtree_a_port_added_later_is_fetched_rather_than_assumed(
    tmp_path: Path, monkeypatch
) -> None:
    """The stamp records which pin was extracted, not which parts of it, so the parts are checked
    as well: a port that starts reading one more directory must not be handed the tree from
    before it did."""
    url = _make_data_archive(tmp_path, present=["data/domains/mock", "data/guidelines"])
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, PREPARE)
    dest = _ensure_data("demo_data_grew", url, ["domains/mock"])
    assert not (dest / "guidelines").exists()

    dest = _ensure_data("demo_data_grew", url, ["domains/mock", "guidelines"])

    assert (dest / "guidelines" / "tasks.json").is_file()
    assert (dest / "domains" / "mock" / "tasks.json").is_file()


def test_data_provisioned_for_another_pin_is_replaced(tmp_path: Path, monkeypatch) -> None:
    """Upstream reads its data from a path that carries no SHA, so the stamp is what makes a
    re-pin an upgrade rather than a silent mixture of two pins."""
    url = _make_data_archive(tmp_path, present=["data/domains/mock"])
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, PREPARE)
    dest = _ensure_data("demo_data_repin", url, ["domains/mock"])
    (dest / "domains" / "stale").mkdir()

    dest = _ensure_data("demo_data_repin", url, ["domains/mock"], sha="sha2")

    assert (dest / _upstream.DATA_SHA_FILE).read_text().strip() == "sha2"
    assert not (dest / "domains" / "stale").exists(), "the previous pin's tree was replaced"


def test_a_data_directory_shogym_did_not_write_is_left_exactly_as_it_is(
    tmp_path: Path, monkeypatch
) -> None:
    """Upstream's resolver would have used it, and an unstamped tree is somebody's own checkout:
    not shogym's to replace, and not an absence to refuse either."""
    monkeypatch.setenv("SHOGYM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv(PROVISIONING_ENV_VAR, OFFLINE)
    theirs = tmp_path / "cache" / "demo_data_theirs" / "data" / "domains"
    theirs.mkdir(parents=True)
    (theirs / "tasks.json").write_text("{}\n")

    dest = _ensure_data("demo_data_theirs", "http://unreachable.invalid/x.tar.gz", ["domains"])

    assert (dest / "domains" / "tasks.json").is_file()
    assert not (dest / _upstream.DATA_SHA_FILE).exists()


# ----- the gate's classifier: what may skip, and what may not -----


@pytest.mark.parametrize(
    "exc",
    [
        urllib.error.URLError("offline"),
        urllib.error.HTTPError("u", 503, "busy", {}, None),  # type: ignore[arg-type]
        ConnectionResetError("reset"),
        ModuleNotFoundError("No module named 'litellm'", name="litellm"),
    ],
)
def test_environmental_failures_may_skip(exc: BaseException) -> None:
    assert _environmental_reason(exc, package="tau2", extra="tau2") is not None


@pytest.mark.parametrize(
    "exc",
    [
        # upstream drift: the pinned symbol is gone
        ImportError("cannot import name 'Renamed' from 'tau2.runner'"),
        # the provisioned package itself did not import: this repo's bug, not the machine's
        ModuleNotFoundError("No module named 'tau2'", name="tau2"),
        ModuleNotFoundError("No module named 'tau2.config'", name="tau2.config"),
        # shogym's own module is broken
        ModuleNotFoundError("No module named 'shogym.envs.tau2'", name="shogym.envs.tau2"),
        NameError("name 'undefined_name_typo' is not defined"),
        SyntaxError("bad adapter"),
        RuntimeError("unexpected tau2 archive layout"),
        # a wrong pin or URL is a 4xx, and that is a bug in this repo
        urllib.error.HTTPError("u", 404, "gone", {}, None),  # type: ignore[arg-type]
    ],
)
def test_regressions_may_not_skip(exc: BaseException) -> None:
    assert _environmental_reason(exc, package="tau2", extra="tau2") is None


@pytest.mark.parametrize("prepared_mode", [PREPARE, OFFLINE])
def test_gate_skips_an_environmental_failure_but_not_on_a_prepared_machine(
    tmp_path: Path, monkeypatch, prepared_mode: str
) -> None:
    """The one case the classifier cannot separate on its own, and why the mode exists.

    A dependency genuinely missing from a hand-copied extra list and a developer who simply never
    installed the extra raise the *same* ``ModuleNotFoundError``; nothing in the process can tell
    them apart. The provisioning mode is what resolves it, by asserting from outside that this
    machine was prepared, so on a prepared runner the first case fails instead of skipping."""
    module_dir = tmp_path / "gatee"
    module_dir.mkdir()
    (module_dir / "demo_gate_env.py").write_text(
        "raise ModuleNotFoundError(\"No module named 'litellm'\", name='litellm')\n"
    )
    monkeypatch.syspath_prepend(str(module_dir))

    monkeypatch.delenv(PROVISIONING_ENV_VAR, raising=False)
    # `pytest.skip` raises `Skipped`, a BaseException — catching it here is what proves the gate
    # skipped rather than propagating, without skipping this test too.
    with pytest.raises(pytest.skip.Exception, match="extra not installed"):
        gate("demo_gate_env", package="demo_gate", extra="demo_gate")

    sys.modules.pop("demo_gate_env", None)
    monkeypatch.setenv(PROVISIONING_ENV_VAR, prepared_mode)
    with pytest.raises(ModuleNotFoundError):
        gate("demo_gate_env", package="demo_gate", extra="demo_gate")


def test_the_gate_refuses_an_unrecognized_mode_before_an_import_that_would_have_worked(
    monkeypatch,
) -> None:
    """The gate reads the mode in front of the import, not only when one fails.

    ``json`` is importable on every machine, so a refusal here can only have come from the read
    ahead of it, which is what makes the typo a machine with everything prepared finds too."""
    monkeypatch.setenv(PROVISIONING_ENV_VAR, "ofline")
    with pytest.raises(RuntimeError, match=PROVISIONING_ENV_VAR):
        gate("json", package="json", extra="json")


def test_a_wrapped_cause_is_still_recognized() -> None:
    inner = urllib.error.URLError("offline")
    outer = RuntimeError("provisioning failed")
    outer.__cause__ = inner
    assert _environmental_reason(outer, package="tau2", extra="tau2") is not None


# ----- the live assertion: every port's fetch-and-import path actually works -----


@pytest.mark.parametrize("package,module,extra", _PORTS, ids=[p[0] for p in _PORTS])
def test_each_port_provisions_and_binds_its_pinned_upstream(
    package: str, module: str, extra: str
) -> None:
    """Skips only on a recognized environmental failure, and only in `local` mode; a prepared
    machine (`prepare` or `offline`) has nothing left to skip for, so there it fails instead."""
    gate(module, package=package, extra=extra)

    bound = sys.modules[package]
    source = _upstream._module_dir(bound)
    assert source is not None and (source / "__init__.py").is_file()
    # The provisioned tree is never importable via `sys.path` — it is bound through `sys.modules`.
    # (`sys.path` itself is not asserted unchanged here, unlike the offline test above: importing a
    # real upstream runs *its* imports, and one of them — `litellm.proxy.proxy_cli` — appends the
    # working directory. That is not this module's doing and not this module's to prevent; what
    # matters is that no provisioned directory is reachable that way.)
    for entry in sys.path:
        assert Path(entry).resolve() not in (source, source.parent)
    # the cache dir holds the package and nothing else, which is what makes it safe to hand to a
    # subprocess on PYTHONPATH
    assert [p.name for p in source.parent.iterdir()] == [package]


#: Imports one module with every outbound connection recorded *and* refused, then prints what was
#: attempted. Recording is the point: LiteLLM catches the failure and falls back to its bundled
#: copy, so a blocked request leaves no trace in the result and a run that only checked for an
#: exception would call a fetching import offline.
_IMPORT_PROBE = '''
import json
import socket
import sys

attempts = []


def refuse(address, *args, **kwargs):
    attempts.append(str(address))
    raise OSError("this probe permits no outbound connection")


def refuse_method(self, address, *args, **kwargs):
    return refuse(address)


def refuse_lookup(host, port=None, *args, **kwargs):
    attempts.append(f"{host}:{port}")
    raise socket.gaierror(8, "this probe permits no name resolution")


socket.create_connection = refuse
socket.socket.connect = refuse_method
socket.socket.connect_ex = refuse_method
socket.getaddrinfo = refuse_lookup

__import__(sys.argv[1])
print("PROBE " + json.dumps(attempts))
'''


@pytest.mark.parametrize("package,module,extra", _PORTS, ids=[p[0] for p in _PORTS])
def test_importing_a_prepared_upstream_under_offline_asks_for_no_connection(
    package: str, module: str, extra: str, tmp_path: Path
) -> None:
    """Binding a prepared source is only half of provisioning; the other half is the import.

    Two of these upstreams import LiteLLM, which fetches a model-cost map from a file server as
    its package initializes unless it is told to read the copy in its own wheel. Nothing about the
    result says so, because it falls back to that copy when the fetch fails, so the only way to
    assert "this import reached nobody" is to watch the socket. A fresh process, because the
    variable that settles it is read once, when LiteLLM is first imported."""
    gate(module, package=package, extra=extra)  # prepared here, or skipped as the mode allows
    script = tmp_path / "probe.py"
    script.write_text(_IMPORT_PROBE)
    environment = {**os.environ, PROVISIONING_ENV_VAR: OFFLINE}
    # Removed rather than left: the child has to be the one that selects the bundled copy, and an
    # inherited variable would let a broken production path pass on the parent's doing.
    environment.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)

    result = subprocess.run(
        [sys.executable, str(script), module],
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
    )

    assert result.returncode == 0, result.stderr[-3000:]
    reported = [line for line in result.stdout.splitlines() if line.startswith("PROBE ")]
    assert len(reported) == 1, result.stdout[-3000:]
    assert json.loads(reported[0][len("PROBE ") :]) == [], "the import reached for the network"
