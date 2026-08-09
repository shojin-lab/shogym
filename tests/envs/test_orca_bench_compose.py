"""``orca_bench`` live backend: everything about it that can be decided without Docker.

The stack itself needs ~133 GB and a daemon, so the live paths carry the ``docker`` mark and CI
skips them. What is here is the part that is a decision rather than an execution: how the clock
override is expressed and why its two constants are sized against each other, where the staged
cache lives, what order teardown happens in, and how a missing report is classified. Each of those
is a fact about the port, not about the machine, so each gets a test that runs anywhere.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import pytest

from shogym.envs.orca_bench import compose_backend, dataset, tasks
from shogym.envs.orca_bench.backend import BackendUnavailableError
from shogym.envs.orca_bench.judge import CapturedReport, JudgeConfig


class _FakeDocker:
    """Records every ``docker`` invocation and answers from a scripted table."""

    def __init__(self, answers: Optional[Dict[str, subprocess.CompletedProcess]] = None) -> None:
        self.calls: List[List[str]] = []
        self.environments: List[Mapping[str, str]] = []
        self._answers = answers or {}
        self.default = subprocess.CompletedProcess([], 0, "", "")

    def __call__(
        self,
        args: Sequence[str],
        *,
        timeout: Optional[float] = None,
        check: bool = True,
        capture: bool = True,
        env: Optional[Mapping[str, str]] = None,
    ) -> subprocess.CompletedProcess:
        self.calls.append(list(args))
        self.environments.append(env or {})
        for key, answer in self._answers.items():
            if key in " ".join(args):
                if check and answer.returncode != 0:
                    raise compose_backend.DockerError(f"scripted failure for {key}")
                return answer
        return self.default

    def commands(self) -> List[str]:
        return [" ".join(call) for call in self.calls]

    def environment_for(self, prefix: str) -> Mapping[str, str]:
        for command, environment in zip(self.commands(), self.environments):
            if command.startswith(prefix):
                return environment
        raise AssertionError(f"no docker command started with {prefix!r}")


def _ok(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], 0, stdout, "")


def _fail(stderr: str = "", code: int = 1) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess([], code, "", stderr)


# ----- the clock -----


def test_the_clock_override_turns_both_knobs_and_no_others() -> None:
    """The two knobs are one decision, and this is where it is legible.

    Jaeger's OpenSearch reader names one index per day of lookback and sends them all in the
    request line, so a longer lookback is a longer URL: widening the lookback alone dies with
    `too_long_http_line_exception`, and raising the limit alone changes nothing. The override has
    to name both services or it does not work at all."""
    override = compose_backend.clock_override_yaml()

    services = [
        line.strip().rstrip(":")
        for line in override.splitlines()
        if line.startswith("  ") and line.rstrip().endswith(":") and not line.startswith("    ")
    ]
    assert services == ["opensearch", "jaeger"], "both knobs, and nothing else reconfigured"
    assert f"http.max_initial_line_length={compose_backend.OPENSEARCH_MAX_LINE}" in override
    # Jaeger is pointed at the shadow config rather than having its own mount shadowed, because
    # compose replaces `command` but concatenates `volumes`, so two mounts on one target fight.
    assert f"--config=file:{compose_backend.CONTAINER_JAEGER_CONFIG}" in override
    assert f"{compose_backend.SHADOW_JAEGER_CONFIG}:{compose_backend.CONTAINER_JAEGER_CONFIG}:ro" in override
    # The staged cache is named the way the entrypoint names it, so it resolves on the daemon.
    assert f"${{{compose_backend.SNAPSHOT_CACHE_ENV}}}/" in override
    assert "issue #77" in override and "max_span_age" in override


def test_the_lookback_and_the_request_line_are_sized_against_each_other() -> None:
    """A lookback that outgrows the request line is the failure this whole override exists to fix,
    so the two constants are not free to drift apart.

    One daily index name is about 40 bytes with its separator, and Jaeger sends one per day of
    lookback. If someone widens the lookback without widening the limit, this fails here rather
    than three minutes into a live run."""
    hours = int(compose_backend.SNAPSHOT_LOOKBACK.removesuffix("h"))
    days = hours / 24
    budget = int(compose_backend.OPENSEARCH_MAX_LINE.removesuffix("kb")) * 1024
    spent = days * 40

    assert days > 365 * 4, "the snapshot has to stay reachable years from now, not months"
    assert spent < budget, f"{days:.0f} daily indices need ~{spent / 1024:.0f} kb of request line"


def test_preparing_the_stack_installs_the_clock_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """The override belongs to the staged cache every trial shares, so it is installed once with
    the staging rather than per episode."""
    docker = _FakeDocker(
        {"volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"), "test -f": _ok()}
    )
    monkeypatch.setattr(compose_backend, "_run", docker)

    compose_backend.prepare_stack()
    commands = docker.commands()
    assert any("docker-compose.snapshot.yml" in command for command in commands)
    assert any(compose_backend.SHADOW_JAEGER_CONFIG in command for command in commands)


def test_the_shadow_config_is_derived_from_the_published_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Derived by substituting the lookback, not written from scratch.

    The published config carries the storage backend, the index prefixes and the rollover layout,
    and a re-pin can change any of them. Copying it and editing one line tracks those; restating
    the file would quietly serve an old one."""
    mountpoint = "/var/lib/docker/volumes/cache/_data"
    docker = _FakeDocker()
    monkeypatch.setattr(compose_backend, "_run", docker)
    compose_backend.install_clock_override(mountpoint)

    script = next(c for c in docker.commands() if compose_backend.SHADOW_JAEGER_CONFIG in c)
    assert f"{mountpoint}/{compose_backend.PUBLISHED_JAEGER_CONFIG}" in script, (
        "the shadow has to be derived from the published file"
    )
    assert f"max_span_age: {compose_backend.PUBLISHED_LOOKBACK}" in script
    assert f"max_span_age: {compose_backend.SNAPSHOT_LOOKBACK}" in script
    # And the substitution is checked rather than assumed: a published file that stops saying
    # 2160h would otherwise yield a shadow identical to the original, silently.
    assert f"grep -q 'max_span_age: {compose_backend.SNAPSHOT_LOOKBACK}'" in script

    # The append is guarded by a marker, and the marker has to be text the override actually
    # carries. If the two ever drift apart the guard stops matching and every episode appends
    # another copy, which is a compose file with duplicate keys rather than a clean failure.
    marker = "shogym's orca_bench port"
    assert f'grep -q "{marker}"' in script, "the append is not guarded"
    assert marker in compose_backend.clock_override_yaml(), (
        "the guard greps for a marker the override does not contain, so it never matches"
    )


def test_a_run_records_what_the_stack_could_actually_see(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded run has to say so rather than look like a healthy one.

    An empty service list is what an agent's first move sees when the recorded telemetry has aged
    out of Jaeger's lookback, and nothing about it raises: the stack is up, the query answers, and
    the answer is that there are no services."""
    docker = _FakeDocker({"volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"), "test -f": _ok()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend.ComposeBackend, "_await_ready", lambda self: None)
    backend = _backend(tmp_path)
    backend.start()

    docker.default = _ok('{"data":["frontend","cart"],"total":2}')
    healthy = backend.telemetry_reach()
    assert healthy["count"] == 2 and healthy["services"] == ["frontend", "cart"]
    assert healthy["error"] == ""

    docker.default = _ok('{"data":[],"total":0,"errors":null}')
    degraded = backend.telemetry_reach()
    assert degraded["count"] == 0 and degraded["error"] == ""

    docker.default = _ok('{"data":null,"errors":[{"code":500,"msg":"An HTTP line is larger than 4096 bytes"}]}')
    refused = backend.telemetry_reach()
    assert refused["count"] == 0 and "4096" in refused["error"]


# ----- staging -----


def test_the_staged_cache_is_keyed_by_image_digest() -> None:
    # A re-pin stages beside the old cache rather than over it, so the two never mix.
    name = compose_backend.snapshot_volume_name("sha256:" + "ab" * 32)
    other = compose_backend.snapshot_volume_name("sha256:" + "cd" * 32)
    assert name != other and name.startswith("shogym-orca-snapshot-cache-")


def test_a_staged_cache_is_not_copied_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The copy is ~46 GB, so the marker check is what keeps a warm host cheap."""
    docker = _FakeDocker(
        {
            "volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"),
            "test -f": _ok(),
        }
    )
    monkeypatch.setattr(compose_backend, "_run", docker)
    mountpoint = compose_backend.ensure_snapshot_cache()

    assert mountpoint == "/var/lib/docker/volumes/x/_data"
    assert not any("cp -a /app/." in command for command in docker.commands())


def test_a_cold_cache_is_copied_inside_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """Into a named volume, not out to the host: the mountpoint is a path the daemon can
    bind-mount for the sibling services, which is what makes the copy stay internal."""
    docker = _FakeDocker(
        {
            "volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"),
            "test -f": _fail(),
            "image inspect": _ok(),
        }
    )
    monkeypatch.setattr(compose_backend, "_run", docker)
    compose_backend.ensure_snapshot_cache()

    staging = [command for command in docker.commands() if "cp -a /app/." in command]
    assert len(staging) == 1
    assert "/var/lib/docker/volumes/x/_data" in staging[0]
    assert "type=volume,source=shogym-orca-snapshot-cache-" in staging[0]


def test_a_pull_is_refused_when_the_daemon_cannot_hold_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The daemon's disk, not the host's: on Docker Desktop they are different numbers, and a
    pull that fills the daemon takes it down rather than failing politely."""
    docker = _FakeDocker({"image inspect": _fail()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend, "daemon_free_bytes", lambda: 10 * 1024**3)

    with pytest.raises(BackendUnavailableError, match="GB free"):
        compose_backend.ensure_image()
    assert not any(command.startswith("pull") for command in docker.commands())


def test_staging_is_refused_when_the_daemon_cannot_hold_the_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The copy is the other 46 GB, and it is reachable without a pull.

    A host that already holds the image skips the pull entirely, so guarding only the pull leaves
    the larger half of the same hazard open: `cp -a` of the snapshot cache onto a daemon with no
    room fills it, and a full daemon does not fail politely, it stops working."""
    docker = _FakeDocker(
        {
            "volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"),
            "test -f": _fail(),  # nothing staged yet, so the copy is next
            "image inspect": _ok(),  # and the image is already here, so no pull guards it
        }
    )
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend, "daemon_free_bytes", lambda: 10 * 1024**3)

    with pytest.raises(BackendUnavailableError, match="GB free"):
        compose_backend.ensure_snapshot_cache()
    assert not any("cp -a" in command for command in docker.commands())


def test_the_image_is_pulled_for_the_platform_it_is_published_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The image is amd64-only; without the flag an arm64 daemon refuses it outright.
    docker = _FakeDocker({"image inspect": _fail()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend, "daemon_free_bytes", lambda: 500 * 1024**3)

    compose_backend.ensure_image()
    pulls = [command for command in docker.commands() if command.startswith("pull")]
    assert len(pulls) == 1
    assert "--platform linux/amd64" in pulls[0]
    assert compose_backend.SNAPSHOT_IMAGE_PINNED in pulls[0]


# ----- the episode -----


def _backend(tmp_path: Path) -> compose_backend.ComposeBackend:
    return compose_backend.ComposeBackend(
        tmp_path,
        judge=JudgeConfig().resolve({"OPENAI_API_KEY": "sk-test"}),
        snapshot="20260423T050139Z-4f4aceafe624e619",
    )


def test_the_container_is_given_the_stack_s_own_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the image's entrypoint reads: the staged cache, a per-trial context, the snapshot to
    materialize, and the clock. Each one is what makes the run this task's rather than some
    other's."""
    docker = _FakeDocker({"volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"), "test -f": _ok()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend.ComposeBackend, "_await_ready", lambda self: None)

    backend = _backend(tmp_path)
    backend.start()

    run = next(command for command in docker.commands() if command.startswith("run --detach"))
    assert "--privileged" in run and "/var/run/docker.sock:/var/run/docker.sock" in run
    assert "SNAPSHOT_CACHE_HOST_DIR=/var/lib/docker/volumes/x/_data" in run
    assert "SNAPSHOT_NAME=20260423T050139Z-4f4aceafe624e619" in run
    # No clock is handed to the container: the telemetry window is restored by the lookback
    # override in the staged cache, not by pinning what anything thinks "now" is.
    assert "FAKETIME" not in run
    assert "-e OPENAI_API_KEY" in run, "the verifier needs the key the session resolved"
    assert "--platform linux/amd64" in run


def test_the_judge_key_never_reaches_a_command_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The container needs the key; `ps` does not.

    A benchmark host runs one episode at a time and is shared, and an argument list is readable by
    every local user for as long as the call runs. `docker run -e NAME` forwards the value from
    the CLI's own environment instead, so the secret is passed without being written down."""
    docker = _FakeDocker({"volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"), "test -f": _ok()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend.ComposeBackend, "_await_ready", lambda self: None)

    backend = _backend(tmp_path)
    backend.start()

    leaked = [command for command in docker.commands() if "sk-test" in command]
    assert not leaked, f"the key is in an argument list: {leaked}"
    assert docker.environment_for("run --detach")["OPENAI_API_KEY"] == "sk-test"


def test_teardown_stops_the_agent_then_sweeps_its_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Order matters: the entrypoint's own trap tears the inner project down when the container
    stops, and the sweep is for the case where that trap was killed. Sweeping first would race
    the trap; not sweeping at all leaks a 28-service project."""
    docker = _FakeDocker({"volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"), "test -f": _ok()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend.ComposeBackend, "_await_ready", lambda self: None)

    backend = _backend(tmp_path)
    handle = backend.start()
    docker.calls.clear()
    backend.teardown()

    commands = docker.commands()
    stop = next(index for index, c in enumerate(commands) if c.startswith("stop --timeout"))
    remove = next(index for index, c in enumerate(commands) if c.startswith("rm --force --volumes"))
    sweep = next(index for index, c in enumerate(commands) if c.startswith("ps --all"))
    volume = next(index for index, c in enumerate(commands) if c.startswith("volume rm"))
    assert stop < remove < sweep < volume
    assert handle.context_volume in commands[volume]


def test_teardown_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    docker = _FakeDocker({"volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"), "test -f": _ok()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend.ComposeBackend, "_await_ready", lambda self: None)

    backend = _backend(tmp_path)
    backend.start()
    backend.teardown()
    docker.calls.clear()
    backend.teardown()
    assert docker.calls == []


def test_the_sealed_bytes_reach_the_verifier_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract's whole point is that the bytes graded are the bytes captured at the seal.

    A report is agent-authored text and can hold anything UTF-8 can express, so the sealed bytes
    are moved as bytes. Handing them to a text pipe instead would re-encode them in whatever the
    subprocess's locale happens to be, and a `LC_ALL=C` host would fail on the first accent."""
    report = "Root cause: the café service ✓ emitted 500s\n".encode("utf-8")
    docker = _FakeDocker({"volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"), "test -f": _ok()})
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend.ComposeBackend, "_await_ready", lambda self: None)

    backend = _backend(tmp_path)
    backend.start()
    backend.run_verifier(CapturedReport(source=compose_backend.REPORT_PATH, data=report))

    sealed = [
        call for call in docker.calls
        if call[0] == "cp" and call[-1].endswith(":/tmp/shogym-sealed-report.md")
    ]
    assert sealed, f"the sealed report never reaches the container as bytes: {docker.commands()}"
    assert Path(sealed[0][1]).read_bytes() == report


def test_a_missing_report_is_captured_as_a_failed_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`docker cp` of a path that does not exist is the container-side shape of "the agent wrote
    no report", and it has to arrive as a graded zero rather than as a crash."""
    docker = _FakeDocker(
        {
            "volume inspect --format": _ok("/var/lib/docker/volumes/x/_data"),
            "test -f": _ok(),
            "cp shogym-orca-": _fail("Error: No such container:path: /app/report.md"),
        }
    )
    monkeypatch.setattr(compose_backend, "_run", docker)
    monkeypatch.setattr(compose_backend.ComposeBackend, "_await_ready", lambda self: None)

    backend = _backend(tmp_path)
    backend.start()
    captured = backend.capture_report()

    assert captured.problem and "no report" in captured.problem
    assert captured.source == "/app/report.md"
    assert captured.data is None


def test_serving_without_docker_says_so(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(compose_backend, "docker_available", lambda: False)
    with pytest.raises(BackendUnavailableError, match="Docker daemon"):
        compose_backend.create_backend(
            tmp_path,
            judge=JudgeConfig().resolve({"OPENAI_API_KEY": "sk-test"}),
                snapshot="snap",
        )


# ----- live -----


@pytest.mark.docker
def test_the_pinned_image_is_present_for_the_published_platform() -> None:
    """The one live check cheap enough to state plainly: the digest this port pins resolves on
    this daemon, for the platform the benchmark publishes."""
    if not compose_backend.docker_available():
        pytest.skip("no docker daemon")
    if not compose_backend.image_present():
        pytest.skip("the pinned image is not staged on this host")
    assert compose_backend.snapshot_volume_name().startswith("shogym-orca-snapshot-cache-")


def _staged_task() -> tasks.OrcaTaskRef:
    """The first real task of the pinned revision, or a skip if this host has no dataset.

    A staged host is one that already holds ~133 GB of image; the dataset is another 192 MB, and
    requiring it here keeps this test on real task bytes rather than a fixture pretending to be
    one. `SHOGYM_ORCA_BENCH_DATA_DIR` points at an existing copy."""
    cache = dataset.dataset_dir()
    if not cache.is_dir():
        pytest.skip(f"no provisioned orca_bench dataset at {cache}")
    refs = tasks.load_index(cache)
    if not refs:
        pytest.skip(f"the dataset at {cache} holds no tasks")
    return refs[0]


@pytest.mark.docker
def test_a_whole_episode_runs_on_a_staged_host() -> None:
    """One real episode, end to end, on a host that has the stack: the part no fake can check.

    Keyless on purpose, so it runs without a budget and without a secret: it stops at the seal,
    which is where the port's own responsibility ends. Grading past that point is the task's own
    verifier calling an LLM, and it is exercised by hand (see the PR's live evidence) rather than
    by charging every run of the suite for it.

    What it does establish is that the readiness marker arrives, that the stack answers as itself,
    that a report written by the agent comes back out **byte for byte** through the capture, and
    that teardown leaves nothing running."""
    if not compose_backend.docker_available():
        pytest.skip("no docker daemon")
    if not compose_backend.image_present():
        pytest.skip("the pinned image is not staged on this host")
    ref = _staged_task()

    backend = compose_backend.ComposeBackend(
        ref.task_dir,
        judge=JudgeConfig().resolve({}),
        snapshot=ref.snapshot,
    )
    # Non-ASCII on purpose: a report is prose, and the seal has to be byte-exact rather than
    # locale-exact. 4531 characters of oracle report came back as 4543 bytes on a live run.
    report = "# Root cause\n\nThe café service returned 500s, naïvely. ✓\n"
    try:
        handle = backend.start()
        health = backend.exec("curl -s -o /dev/null -w '%{http_code}' $GRAFANA_URL/api/health")
        assert health["stdout"].strip() == "200", f"the stack is up but Grafana is not: {health}"

        # The clock override, witnessed rather than assumed. This is the assertion that catches
        # the benchmark's silent expiry coming back: without both knobs the list is empty, the
        # stack still looks healthy, and every run is quietly incomparable to the paper's.
        reach = backend.telemetry_reach()
        assert reach["error"] == "", f"the telemetry query itself failed: {reach['error']}"
        assert reach["count"] > 10, (
            f"the recorded telemetry is not reachable: {reach['count']} services. "
            "The clock override is what keeps this populated."
        )

        backend.write_file(compose_backend.REPORT_PATH, report)
        captured = backend.capture_report()
        assert captured.problem == "", f"a written report failed to capture: {captured.problem}"
        assert captured.data == report.encode("utf-8")
    finally:
        backend.teardown()

    # The sibling services are the whole reason teardown is not just `docker rm`: they are named
    # for the compose project the entrypoint started, and they outlive the container that made
    # them. Scoped to this trial so an unrelated workload on the host is not mistaken for a leak.
    for scope in (handle.container, handle.project_hint):
        left = compose_backend._run(
            ["ps", "--all", "--quiet", "--filter", f"name={scope}"], timeout=120, check=False
        )
        assert not left.stdout.strip(), f"teardown left {scope} behind: {left.stdout}"
