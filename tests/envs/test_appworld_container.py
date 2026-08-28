"""The container seam: what a worker is given, and what it is refused, without starting one.

Offline and Docker-free. The end-to-end proof that an episode cannot reach the run tree or the
grader is in ``test_appworld_served.py``, where the probes run through a real ``execute`` in a
real container. What is here is the arithmetic behind it: the mount set is built from the task,
the flags say what they are supposed to say, and a machine with no daemon is refused when the env
is constructed rather than when an episode first runs code.

Both halves are worth having separately. The served suite needs Docker and a corpus and takes
minutes; this runs on a laptop with neither, and it is the test that fails first if someone adds
a convenient mount.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from shogym.envs.appworld import adapter, container
from shogym.envs.appworld.container import WORKER


def _corpus(root: Path) -> Path:
    """A derived root with the shape the real one has: shared parts and two tasks."""
    data = root / "data"
    for name in ("api_docs", "base_dbs", "datasets"):
        (data / name).mkdir(parents=True)
    (data / "version.txt").write_text("0.1.0\n")
    for task in ("abc_1", "def_2"):
        (data / "tasks" / task / "dbs").mkdir(parents=True)
        (data / "tasks" / task / "specs.json").write_text("{}")
    return root


# ----- the mount set -----


def test_one_episode_is_given_one_task_and_one_output_tree(tmp_path: Path) -> None:
    """The corpus holds 318 tasks and every episode has an output tree of its own, so mounting
    either wholesale would put another task's world or another episode's end state one ``listdir``
    away. The mount set names one of each, and the output tree is mounted outside the corpus at a
    fixed name, which is what the world is then told its experiment is."""
    root = _corpus(tmp_path / "seeded")
    outputs = tmp_path / "private" / "episode-abc"
    mounts = adapter.served_mounts(root=root, task_id="abc_1", outputs=outputs)
    targets = {mount.target: mount for mount in mounts}
    assert targets.keys() == {
        "/corpus/data/api_docs",
        "/corpus/data/base_dbs",
        "/corpus/data/datasets",
        "/corpus/data/version.txt",
        "/corpus/data/tasks/abc_1",
        "/outputs",
    }
    # The output tree is not under the corpus at all, so nothing an episode writes lands in the
    # tree the next episode is served.
    assert not str(outputs).startswith(str(root))
    # The other task is on the host, in the same tree, and is not in the mount set.
    assert (root / "data" / "tasks" / "def_2").is_dir()
    assert not [mount for mount in mounts if "def_2" in str(mount.source)]
    # And `data/tasks` itself is not mounted, so the directory inside the container holds exactly
    # what was named: mounting the parent would have brought the roster with it.
    assert "/corpus/data/tasks" not in targets


def test_the_only_writable_mount_is_the_episodes_own_output_tree(tmp_path: Path) -> None:
    """A world writes its end state and nothing else. Everything it reads is read-only, which is
    also what stops an episode editing the task it is about to be graded on."""
    root = _corpus(tmp_path / "seeded")
    mounts = adapter.served_mounts(
        root=root, task_id="abc_1", outputs=tmp_path / "private" / "episode-abc"
    )
    writable = [mount.target for mount in mounts if mount.writable]
    assert writable == ["/outputs"]
    for mount in mounts:
        assert mount.as_argument().endswith(":rw" if mount.writable else ":ro")


def test_the_graders_view_is_the_answers_and_the_end_state_and_they_are_two_trees(
    tmp_path: Path,
) -> None:
    """The answers live in a private tree and the episode's end state in another, and the
    evaluator wants a root and an experiment. That used to be a symlink from the private tree into
    the served one, published under a lock because two cold constructors raced on creating it. It
    is two mounts now, so there is no link and no race.

    The grading container is given this episode's output tree and no other's, which is the same
    property the world's container has and a different reason for it: nothing here runs a line an
    agent wrote, and one episode's evaluator has no business opening another's end state."""
    graded = _corpus(tmp_path / "graded")
    (graded / "data" / "tasks" / "abc_1" / "ground_truth").mkdir()
    outputs = tmp_path / "private" / "episode-abc"
    mounts = adapter.graded_mounts(graded=graded, task_id="abc_1", outputs=outputs)
    targets = {mount.target for mount in mounts}
    assert "/graded/data/tasks/abc_1" in targets
    assert "/outputs" in targets
    assert "/graded/data/tasks/def_2" not in targets
    # The two sources are in different trees on the host, which is the point of the second mount.
    sources = {mount.target: mount.source for mount in mounts}
    assert sources["/outputs"] == outputs
    assert not str(outputs).startswith(str(graded))


# ----- the flags -----


def _captured(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    """Run one container without a daemon, and hand back the command line it would have used."""
    # Warmed before the patch: the boot identity is read once through `subprocess` and memoized,
    # and a stub that answers every `Popen` would otherwise answer that one too. The birth stamp
    # and the resolved image id are read per call, so they are stubbed rather than warmed.
    container._boot_id()
    monkeypatch.setattr(container, "process_birth", lambda pid: "Thu Jan  1 00:00:00 2026")
    monkeypatch.setattr(container, "image_identity", lambda name: "sha256:stub linux/arm64")
    seen: List[List[str]] = []

    class _Fake:
        stdin = None
        stdout = None
        stderr = None

    def _popen(args: List[str], **_: Any) -> Any:
        seen.append(list(args))
        return _Fake()

    monkeypatch.setattr(container.subprocess, "Popen", _popen)
    monkeypatch.setattr(container, "image_name", lambda: "shogym-appworld-worker:test")
    return seen


def test_a_worker_container_has_no_network_and_no_published_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reason the transport had to stop being a port.

    A container-loopback listener cannot be forwarded, a published port is not loopback-only, and
    ``--network none`` and ``-p`` are mutually exclusive. Frames on the parent's own pipe pair are
    what let the first flag be there at all, so the flag and the absence of ``-p`` are one
    assertion in two halves."""
    seen = _captured(monkeypatch)
    container.run(role="serve", mounts=[container.Mount(tmp_path, "/corpus")])
    args = seen[0]
    assert "--network" in args and args[args.index("--network") + 1] == "none"
    assert "-p" not in args and "--publish" not in args
    assert "-i" in args


def test_a_worker_container_drops_what_a_world_does_not_need(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Agent-authored code runs in this container. A world driving nine simulated apps needs no
    capability, no writable root filesystem, and no ability to gain privileges, so it is given
    none of them; and a fork bomb is a plausible accident rather than an attack."""
    seen = _captured(monkeypatch)
    container.run(role="serve", mounts=[container.Mount(tmp_path, "/corpus")])
    args = seen[0]
    assert "--read-only" in args
    assert args[args.index("--cap-drop") + 1] == "ALL"
    assert args[args.index("--security-opt") + 1] == "no-new-privileges"
    assert "--pids-limit" in args
    assert "--rm" in args
    # Not root, and the host user's own uid, so what it writes into the mounted output tree is
    # owned by the run rather than by root.
    assert "--user" in args and args[args.index("--user") + 1].count(":") == 1


def test_a_worker_container_is_offered_nothing_from_the_hosts_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The old worker filtered ``os.environ`` through an allow-list, which is one forgotten name
    away from handing an agent a provider key. A container is given the image's own environment
    plus what ``-e`` names, so the host's is not filtered: it is never offered."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    seen = _captured(monkeypatch)
    container.run(
        role="serve",
        mounts=[container.Mount(tmp_path, "/corpus")],
        environment={"APPWORLD_ROOT": "/corpus"},
    )
    args = seen[0]
    passed = [args[index + 1] for index, item in enumerate(args) if item == "-e"]
    # What this port names, and the proxy profile it empties (see the test below). Nothing else.
    assert set(passed) == {"APPWORLD_ROOT=/corpus"} | {
        f"{name}=" for name in container._PROXY_VARIABLES
    }
    assert "sk-secret" not in " ".join(args)


# ----- the image -----


def test_the_image_is_named_for_what_it_was_built_from(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tag that did not move with the Dockerfile would serve a world built under the previous
    one, which is the class of bug the derived corpus's own digest exists to stop."""
    container._image_tag.cache_clear()
    first = container.image_name()
    edited = tmp_path / "worker.Dockerfile"
    edited.write_text(container.DOCKERFILE.read_text() + "\n# a comment\n")
    monkeypatch.setattr(container, "DOCKERFILE", edited)
    container._image_tag.cache_clear()
    try:
        assert container.image_name() != first
    finally:
        monkeypatch.undo()
        container._image_tag.cache_clear()
    # And it is the text and not the path: the original file gives the original tag back.
    assert container.image_name() == first


def test_the_dockerfile_pins_its_base_by_digest_and_upstream_by_version() -> None:
    """Neither pin is decorative. The base decides the interpreter every measured world runs
    under, and ``appworld`` decides what a world is."""
    text = container.DOCKERFILE.read_text()
    assert "python:3.12-slim-bookworm@sha256:" in text
    assert f'"appworld=={adapter.UPSTREAM_VERSION}"' in text
    # And the corpus is not in it. It carries every task's ground truth beside every task's
    # specs, so an image holding it would put the answers inside the container running the code.
    assert adapter.DATA_BUNDLE_URL not in text


# ----- the machine that cannot run this at all -----


def test_a_machine_without_docker_is_refused_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At construction, not at the first ``execute``. There is no host fallback to fall back to:
    a worker on the host runs agent-authored code as the user running the run, which is the
    arrangement the container exists to end. An hour into a run is the wrong time to learn that."""
    monkeypatch.setattr(container, "docker_available", lambda: False)
    with pytest.raises(container.DockerError) as refused:
        container.require_docker()
    said = str(refused.value)
    assert "docker" in said.lower()
    assert "no host fallback" in said
    # And the provisioning wrapper turns it into the type the test gate reads as "not provisioned
    # on this machine", so an offline laptop skips the served suite instead of erroring.
    with pytest.raises(adapter.ProvisioningError):
        adapter.ensure_image()


def test_the_env_says_so_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read off the constructor rather than by building one, which would want a real daemon."""
    import inspect

    from shogym.envs.appworld import env_v1

    source = inspect.getsource(env_v1.AppWorldEnv.__init__)
    before = source.split("ensure_corpus")[0]
    assert "container.require_docker()" in before


# ----- the frames -----


def test_a_frame_survives_a_payload_that_would_break_a_line_protocol() -> None:
    """Length-prefixed rather than newline-delimited. JSON escapes newlines inside strings, so a
    line protocol works right up until a frame carries something that is not JSON, and then it
    fails by reading half a message rather than by raising."""
    import io

    from shogym.envs.appworld import worker

    stream = io.BytesIO()
    payload: Dict[str, Any] = {"output": "one\ntwo\r\nthree", "code": "print('\\n')"}
    worker.send_frame(stream, payload)
    stream.seek(0)
    assert worker.read_frame(stream) == payload
    # And end of input is a value rather than an exception: it is how a worker learns its parent
    # is gone, which is what stops an orphaned container running forever.
    assert worker.read_frame(stream) is None


def test_two_frames_in_one_buffer_are_two_frames() -> None:
    """The parent reads with a deadline, so it reads whatever bytes have arrived and may hold the
    start of the next frame. A reader that re-read the descriptor for a frame it already had
    would wait forever on the last one."""
    import io

    from shogym.envs.appworld import worker

    stream = io.BytesIO()
    worker.send_frame(stream, {"ready": True})
    worker.send_frame(stream, {"output": {"calls": 1}})
    stream.seek(0)
    assert worker.read_frame(stream) == {"ready": True}
    assert worker.read_frame(stream) == {"output": {"calls": 1}}


# ----- the frame protocol, against a stub that misbehaves -----


def _stub_worker(script: str, monkeypatch: pytest.MonkeyPatch) -> adapter.Worker:
    """A `Worker` around a local interpreter speaking the protocol, and no daemon anywhere.

    The transport is the part under test, so the container is not: `close` is pointed at nothing,
    and what is left is exactly the pipe pair and the frames on it."""
    import subprocess

    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=0,
    )
    assert process.stdout is not None
    monkeypatch.setattr(container, "remove", lambda name, confirm=False: None)
    return adapter.Worker(
        root=Path("/nowhere"),
        process=process,
        container="stub",
        frames=adapter._Frames(process.stdout.fileno()),
    )


_ECHO = """
import json, os, sys, time
r = os.fdopen(os.dup(0), "rb", buffering=0)
w = os.fdopen(os.dup(1), "wb", buffering=0)


def send(payload):
    body = json.dumps(payload).encode()
    w.write(b"%d\\n" % len(body))
    w.write(body)
    w.flush()


while True:
    header = r.readline()
    if not header:
        break
    request = json.loads(r.read(int(header.strip())))
    if request["body"].get("slow"):
        time.sleep(float(request["body"]["slow"]))
    send({"id": request["id"], "output": {"saw": request["command"]}})
"""


def test_a_timed_out_call_makes_the_worker_unusable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A timeout on an ordered pipe is not a failure that ends when the caller stops waiting.

    HTTP gave each response its own connection, so abandoning one cost nothing. A pipe is one
    ordered stream: the command that timed out is still running, its answer is still coming, and
    the world it is running against is still changing. There is no state in which reusing that
    worker is right, so it is refused, and the refusal says why."""
    monkeypatch.setattr(adapter, "_CALL_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    worker = _stub_worker(_ECHO, monkeypatch)
    try:
        with pytest.raises(adapter.WorkerError) as timed_out:
            worker.call("execute", slow=3.0)
        assert "did not answer" in str(timed_out.value)
        with pytest.raises(adapter.WorkerError) as refused:
            worker.call("seal")
        # The second failure names the first, rather than being a fresh mystery.
        assert "not usable" in str(refused.value)
        assert "still running" in str(refused.value)
    finally:
        worker.close()


def test_an_answer_to_a_command_nobody_is_waiting_for_is_not_read_as_the_next_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure the identifiers exist for, forced by hand.

    A worker that answers late leaves its answer in the stream. Without an identifier the next
    call reads it as its own, which is an earlier block's output returned under a later step, or a
    finalizer handed the wrong shape entirely. Here the stub answers the first command twice, so
    the second call meets a stale frame before its own."""
    stub = _ECHO.replace(
        'send({"id": request["id"], "output": {"saw": request["command"]}})',
        'send({"id": request["id"], "output": {"saw": request["command"]}})\n'
        '    if request["command"] == "first":\n'
        '        send({"id": request["id"], "output": {"saw": "stale"}})',
    )
    worker = _stub_worker(stub, monkeypatch)
    try:
        assert worker.call("first")["saw"] == "first"
        # The stale duplicate is sitting in the pipe. The next call must step over it.
        assert worker.call("second")["saw"] == "second"
    finally:
        worker.close()


# ----- removal, confirmed -----


def test_a_close_that_cannot_confirm_stays_retryable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Finalization's close is the one that establishes the invariant, so it fails closed.

    It raises when the daemon will not say the container is gone, and it leaves the worker
    unclosed, because a worker marked closed is one teardown will not try again."""
    monkeypatch.setattr(container, "_run", lambda *a, **k: _Finished(0, ""))
    monkeypatch.setattr(container, "absent", lambda name: False)
    worker = _stub_worker(_ECHO, monkeypatch)
    monkeypatch.undo()
    monkeypatch.setattr(container, "absent", lambda name: False)
    monkeypatch.setattr(container, "_run", lambda *a, **k: _Finished(0, ""))
    with pytest.raises(container.DockerError) as refused:
        worker.close(confirm=True)
    assert "not confirmed" in str(refused.value)
    assert worker.closed is False
    # And the best-effort close teardown makes does not raise, so a failing removal never stops
    # an episode being torn down.
    monkeypatch.setattr(container, "absent", lambda name: True)
    worker.close()
    assert worker.closed is True


class _Finished:
    def __init__(self, returncode: int, stdout: str, stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_removal_confirms_by_asking_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docker rm -f` returning zero is not the same fact as the container being gone."""
    calls: List[List[str]] = []

    def _run(args, **_):
        calls.append(list(args))
        return _Finished(0, "")

    monkeypatch.setattr(container, "_run", _run)
    monkeypatch.setattr(container, "absent", lambda name: True)
    container.remove("c", confirm=True)
    # Stopped with no grace period and then removed. `rm -f` alone is a signal and the daemon's
    # own timeout; an explicit stop is the shortest path to "nothing in there is running".
    assert [call[0] for call in calls] == ["stop", "rm"]
    assert calls[0][:3] == ["stop", "--time", "0"]
    # Without confirmation it does not ask, because teardown must not pay for a question whose
    # answer it would ignore.
    calls.clear()
    container.remove("c")
    assert [call[0] for call in calls] == ["stop", "rm"]


# ----- the protocol descriptors -----


def test_the_protocol_descriptors_are_not_handed_to_a_child(tmp_path: Path) -> None:
    """What the redirection does and does not buy, checked rather than asserted in prose.

    Redirecting 0 and 1 stops an accidental `print` corrupting a frame. It does not take the
    duplicated endpoints away from code running in the same interpreter, and nothing can: the
    worker needs them between commands and a closed pipe end cannot be reopened. What it can do is
    keep them out of anything that agent code *starts*, and that is a real reduction, so it is a
    test: a child cannot forge a frame on a descriptor it did not receive."""
    import subprocess

    script = tmp_path / "channel.py"
    script.write_text(
        "import json, os, subprocess, sys\n"
        f"sys.path.insert(0, {str(WORKER.parent)!r})\n"
        "import worker as W\n"
        "reader, writer = W._take_channel()\n"
        "inherited = subprocess.run(\n"
        "    [sys.executable, '-c',\n"
        "     'import os,sys; print([f for f in (3,4,5,6) \\n"
        "      if os.path.exists(\"/dev/fd/%d\" % f)])'],\n"
        "    capture_output=True, text=True)\n"
        "W.send_frame(writer, {'inheritable': [os.get_inheritable(reader.fileno()),\n"
        "                                      os.get_inheritable(writer.fileno())],\n"
        "                      'stdin_is_devnull': sys.stdin.read() == '',\n"
        "                      'child': inherited.stdout.strip()})\n"
    )
    finished = subprocess.run(
        [sys.executable, str(script)], input=b"", capture_output=True
    )
    payload = json.loads(finished.stdout.split(b"\n", 1)[1])
    # Neither endpoint is inheritable, so a process agent code starts does not receive them.
    assert payload["inheritable"] == [False, False]
    # And the ordinary standard input is /dev/null, so agent code reading it cannot eat a command.
    assert payload["stdin_is_devnull"] is True


# ----- the reaper -----


def test_the_reaper_removes_a_container_whose_parent_is_gone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The case teardown cannot reach: a parent that dies while a world is wedged in a command.

    The worker learns its parent is gone only from end-of-file on its next read, and a command
    that never returns never reaches that read, so the container never exits and `--rm` never
    fires. A random name leaves a later run nothing to recognize it by, so every container carries
    the pid that started it and this boot, and construction sweeps the ones whose parent is not
    there any more.

    Removing containers is destructive, so the ambiguous cases must do nothing: an unreadable
    label, and a pid that belongs to somebody else's live process, are both left alone."""
    listed = ["dead1", "alive1", "unlabelled"]
    labels = {
        "dead1": '{"shogym.appworld.parent": "4242", "shogym.appworld.birth": "1700000000"}',
        "alive1": '{"shogym.appworld.parent": "4243", "shogym.appworld.birth": "1700000000"}',
        "unlabelled": "{}",
    }
    removed: List[str] = []

    def _run(args, **_):
        if args[0] == "ps":
            return _Finished(0, "\n".join(listed))
        if args[0] == "inspect":
            return _Finished(0, labels[args[-1]])
        if args[0] in ("rm", "stop"):
            removed.append(args[-1])
            return _Finished(0, "")
        raise AssertionError(args)

    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    monkeypatch.setattr(container, "_run", _run)
    swept = container.reap(alive=lambda pid, birth="": pid == 4243)
    assert swept == ["dead1"]
    assert removed == ["dead1"]


def test_every_container_carries_who_started_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen = _captured(monkeypatch)
    container.run(role="serve", mounts=[container.Mount(tmp_path, "/corpus")])
    args = seen[0]
    labels = [args[index + 1] for index, item in enumerate(args) if item == "--label"]
    assert f"{container.LABEL_OWNER}=1" in labels
    assert f"{container.LABEL_PARENT}={os.getpid()}" in labels
    assert any(item.startswith(f"{container.LABEL_BOOT}=") for item in labels)
    # And the hostname is a constant rather than the container's own short id, which Docker would
    # otherwise put in the environment.
    assert args[args.index("--hostname") + 1] == "worker"

# ----- absence, and the difference between "gone" and "I could not look" -----


def test_absence_is_the_daemons_word_and_not_merely_a_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one direction this may never fail in.

    Absence is what allows grading, so reading "I could not look" as "it is gone" is reading a
    control failure as the fact that lets a score be taken over a tree something may still be
    writing to. Every daemon outage, unreachable context, permission error and timeout exits
    nonzero, and only one of those is not-found."""
    monkeypatch.setattr(
        container, "_run", lambda *a, **k: _Finished(1, "", "Error: No such object: c")
    )
    assert container.absent("c") is True
    monkeypatch.setattr(container, "_run", lambda *a, **k: _Finished(0, "[{}]"))
    assert container.absent("c") is False
    # Anything else is unknown, and unknown is not a boolean.
    monkeypatch.setattr(
        container,
        "_run",
        lambda *a, **k: _Finished(1, "", "Cannot connect to the Docker daemon at unix:///..."),
    )
    with pytest.raises(container.DockerError) as unknown:
        container.absent("c")
    assert "cannot tell" in str(unknown.value)


def test_a_removal_that_cannot_be_confirmed_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """And the failure travels: `remove(confirm=True)` raises rather than returning quietly."""
    monkeypatch.setattr(
        container,
        "_run",
        lambda *a, **k: _Finished(1, "", "Cannot connect to the Docker daemon"),
    )
    with pytest.raises(container.DockerError):
        container.remove("c", confirm=True)
    # Teardown's own call is the other contract: it asks nothing and raises nothing.
    container.remove("c")


# ----- what a runaway costs its sibling -----


def test_a_serving_container_is_given_cpu_and_memory_quotas(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pid limit bounds a fork bomb and nothing else.

    Agent-authored code runs in here, and a block that spins or allocates without bound takes the
    machine away from whatever else is on it. The other arm of a pair is a sibling container on
    the same host, so an arm that ran slower because its twin was busy is a difference the
    treatment did not make."""
    seen = _captured(monkeypatch)
    container.run(role="serve", mounts=[container.Mount(tmp_path, "/corpus")])
    args = seen[0]
    assert args[args.index("--cpus") + 1]
    assert args[args.index("--memory") + 1]


def test_a_container_is_launched_by_resolved_id_rather_than_by_tag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A tag is mutable and the fingerprint is not.

    The run records the image the world ran in by resolving the tag once. Launching the tag again
    afterwards is launching whatever the tag names *now*, which a concurrent rebuild can change,
    so the bytes that ran and the bytes the record names come apart."""
    seen = _captured(monkeypatch)
    container.run(role="serve", mounts=[container.Mount(tmp_path, "/corpus")])
    args = seen[0]
    assert "sha256:stub" in args
    assert not [item for item in args if item.startswith("shogym-appworld-worker:")]


# ----- the orphan sweep, and a recycled pid -----


def test_a_recycled_pid_is_not_a_live_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    """`kill(pid, 0)` answers a different question from the one the sweep is asking.

    It says whether something is running under that number. The sweep is asking whether the
    process that started this container is still running, and within one boot a number comes back
    quickly. A container whose parent died and whose number was reused would otherwise be kept
    for ever, which is the leak the sweep exists to close."""
    monkeypatch.setattr(container, "process_birth", lambda pid: "now")
    # Same number, different process: the recorded birth no longer matches.
    assert container._process_is_alive(os.getpid(), "then") is False
    assert container._process_is_alive(os.getpid(), "now") is True
    # An unreadable birth says nothing, so it says nothing: the pid check stands alone.
    monkeypatch.setattr(container, "process_birth", lambda pid: "")
    assert container._process_is_alive(os.getpid(), "then") is True


# ----- the snapshot the grader is given -----


def test_a_link_in_the_output_tree_refuses_the_grade(tmp_path: Path) -> None:
    """The grader's namespace holds the answers, so what it is given may not be a link.

    A symlink under the output tree resolves inside the *grader's* filesystem, not the world's, so
    one planted there could make the digest and the evaluator read the private tree instead of
    what the episode submitted. Nothing returns those bytes to the agent, so this is score
    integrity rather than a leak, and it is refused rather than skipped: a grade computed over a
    tree with an entry quietly dropped is a grade over a tree nobody submitted."""
    outputs = tmp_path / "outputs"
    (outputs / "tasks" / "abc_1" / "dbs").mkdir(parents=True)
    (outputs / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").write_text("rows")
    # The ordinary case first, so the refusal below is not the only thing this can do.
    snapshot = adapter.snapshot_outputs(outputs, into=tmp_path / "graded")
    assert (snapshot / "tasks" / "abc_1" / "dbs" / "todoist.jsonl").read_text() == "rows"

    (outputs / "tasks" / "abc_1" / "dbs" / "answers.json").symlink_to("/graded/data")
    with pytest.raises(adapter.SnapshotError) as refused:
        adapter.snapshot_outputs(outputs, into=tmp_path / "graded2")
    assert "symbolic link" in str(refused.value)


# ----- the protocol has no lifecycle commands left -----


def test_the_worker_answers_no_lifecycle_command() -> None:
    """Nothing the host needs to know comes from the process that runs the agent's code.

    There was a `seal` and a `quiesce` and a `close`, and finalization treated the replies to them
    as proof that a flush had happened and that work had stopped. Request identifiers correlate a
    reply with its request; they do not say who wrote it, and the writer is reachable from inside
    the interpreter that runs agent-authored Python. So the commands are gone: the host stops the
    container and grades what upstream had already written to disk."""
    import inspect

    from shogym.envs.appworld import worker as worker_module

    source = inspect.getsource(worker_module.serve)
    assert '"open": episode.open' in source
    assert '"execute": episode.execute' in source
    for gone in ("seal", "quiesce", "close"):
        assert f'"{gone}"' not in source, gone
    assert not hasattr(worker_module.Episode, "seal")
    assert not hasattr(worker_module.Episode, "quiesce")

    from shogym.envs.appworld import env_v1

    finalize = inspect.getsource(env_v1.AppWorldEnv.finalize)
    assert "worker.call" not in finalize
    # And what is graded is what upstream persisted, which it does at the end of every block.
    assert "close, confirm=True" in finalize


def test_a_horizon_must_be_a_positive_whole_number_of_blocks() -> None:
    """Both ways of getting this wrong half-worked, which is worse than either failing.

    Zero disabled the guard (`if session.budget` is false) and let one block through before the
    serve layer ended the episode; a negative refused the first block and still spent the call
    that ends the horizon."""
    import inspect

    from shogym.envs.appworld import env_v1

    source = inspect.getsource(env_v1.AppWorldEnv.__init__)
    guard = source[: source.index("adapter.ensure_image")]
    assert "horizon < 1" in guard
    assert "positive whole number" in guard

# ----- what a timeout may and may not claim -----


def test_a_timeout_that_cannot_confirm_removal_leaves_the_worker_unclosed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Unusable and absent are two facts, and a timeout used to set both.

    The timeout path removed the container best effort and then marked the worker closed. Best
    effort ignores an ordinary nonzero stop or removal, so it can return while the daemon still
    owns a running container, and every later close, including finalization's own gate immediately
    before the snapshot, then returned early. The tree being graded could still have had the
    timed-out command writing into it."""
    monkeypatch.setattr(adapter, "_CALL_TIMEOUT_SECONDS", 0.4)
    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    worker = _stub_worker(_ECHO, monkeypatch)
    # A daemon that will not say the container is gone, patched after the stub's own no-op.
    monkeypatch.setattr(container, "remove", _refuses_to_confirm)
    with pytest.raises(adapter.WorkerError):
        worker.call("execute", slow=3.0)
    # Poisoned, so nothing uses it again; and not closed, so the gate before grading still asks.
    assert worker.poisoned
    assert worker.closed is False
    with pytest.raises(container.DockerError):
        worker.close(confirm=True)
    # And handed to the sweep, because this process has no other way to get rid of it.
    assert worker.container in _ledger_names()


def _refuses_to_confirm(name: str, *, confirm: bool = False) -> None:
    if confirm:
        raise container.DockerError("the daemon has not confirmed that it stopped")


def _ledger_names() -> List[str]:
    return container.outstanding()


def _empty_ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")


# ----- the decoder is a host allocation, so it is bounded -----


def test_a_frame_larger_than_the_reader_will_hold_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The container's memory limit does not reach the parent's buffer.

    A writer inside the container declares a length and this process allocates it. That writer is
    reachable from inside the interpreter that runs agent-authored code, so an unbounded declared
    length is an unbounded host allocation asked for by the episode. A frame past the limit, a
    header that is not a length, and a header with no newline in it are all refused before
    anything is read, and all of them are fatal: once a frame is not one this protocol writes, the
    stream's position is unknown and there is no next frame to look for."""
    oversized = """
import os, sys
r = os.fdopen(os.dup(0), "rb", buffering=0)
w = os.fdopen(os.dup(1), "wb", buffering=0)
r.readline()
w.write(b"%d\\n" % (64 * 1024 * 1024 * 1024))
w.flush()
import time; time.sleep(30)
"""
    monkeypatch.setattr(adapter, "_CALL_TIMEOUT_SECONDS", 5.0)
    stopped: List[str] = []
    worker = _stub_worker(oversized, monkeypatch)
    # After the stub, which points removal at nothing of its own.
    monkeypatch.setattr(container, "remove", lambda name, confirm=False: stopped.append(name))
    with pytest.raises(adapter.WorkerError) as refused:
        worker.call("execute")
    assert "declared a" in str(refused.value)
    # Fatal: the worker is refused from here on, and its container is gone rather than merely
    # unused.
    assert worker.poisoned
    assert stopped == [worker.container]
    with pytest.raises(adapter.WorkerError):
        worker.call("execute")


def test_a_header_that_is_not_a_length_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    junk = """
import os, time
r = os.fdopen(os.dup(0), "rb", buffering=0)
w = os.fdopen(os.dup(1), "wb", buffering=0)
r.readline()
w.write(b"not-a-length\\n{}")
w.flush()
time.sleep(30)
"""
    monkeypatch.setattr(adapter, "_CALL_TIMEOUT_SECONDS", 5.0)
    worker = _stub_worker(junk, monkeypatch)
    with pytest.raises(adapter.WorkerError) as refused:
        worker.call("execute")
    assert "where a byte count belongs" in str(refused.value)


# ----- the output tree is a host bind, so it is bounded -----


def test_a_container_nobody_could_remove_is_swept_even_while_its_parent_lives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The sweep skips containers whose parent is alive, which is right for the ordinary case and
    wrong for the one where removal failed.

    A long-lived serving process that could not remove a container has no later chance to try, and
    the container holds a writable mount until that process exits. Writing the name where the
    sweep also looks is what gives it a second owner."""
    ledger = tmp_path / "disowned.txt"
    monkeypatch.setattr(container, "_ledger", lambda: ledger)
    container.disowned("stuck-one")
    container.disowned("stuck-one")  # twice, because a retry writes again
    removed: List[str] = []

    def _run(args, **_):
        if args[0] == "ps":
            return _Finished(0, "")
        if args[0] in ("stop", "rm"):
            removed.append(args[-1])
            return _Finished(0, "")
        return _Finished(1, "", "Error: No such object: stuck-one")

    monkeypatch.setattr(container, "_run", _run)
    swept = container.reap(alive=lambda pid, birth="": True)
    assert swept == ["stuck-one"]
    assert "stuck-one" in removed
    # Tombstoned rather than rewritten, so the next sweep has nothing to do and an append that
    # landed during this one is still there to be read.
    assert container.outstanding() == []
    assert "-stuck-one" in ledger.read_text()


# ----- identities that do not move with a locale -----


def test_the_birth_and_boot_stamps_do_not_move_with_the_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both stamps were renderings rather than values.

    `ps` prints a start time for a human and `sysctl` prints a date beside its struct, so one live
    process and one boot produced two different identities under two `TZ` values. A sweep run in
    another zone would then either hide an orphan behind a different boot hash or read a live
    parent as replaced."""
    container._boot_id.cache_clear()
    monkeypatch.setenv("TZ", "UTC")
    utc = (container.process_birth(os.getpid()), container._boot_id())
    container._boot_id.cache_clear()
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    pacific = (container.process_birth(os.getpid()), container._boot_id())
    container._boot_id.cache_clear()
    assert utc == pacific, (utc, pacific)
    assert utc[0] and utc[1]


# ----- what the client adds that this port never passed -----


def test_the_proxy_profile_docker_injects_is_emptied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docker's client adds these from whatever proxy profile is configured, so leaving them out
    of the command line leaves them in the container.

    A proxy URL can carry credentials or an internal host name, by Docker's own documentation, and
    a world with no network has no use for one. An explicit assignment is what overrides an
    injected one."""
    seen = _captured(monkeypatch)
    container.run(role="serve", mounts=[container.Mount(tmp_path, "/corpus")])
    passed = {
        args.split("=", 1)[0]: args.split("=", 1)[1]
        for index, args in enumerate(seen[0])
        if index and seen[0][index - 1] == "-e" and "=" in args
    }
    for name in container._PROXY_VARIABLES:
        assert name in passed, name
        assert passed[name] == "", name


# ----- what a machine was, in the fingerprint -----


def test_the_resource_limits_are_captured_once_and_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """They decide latency, what a call timeout means, and whether a world is killed for
    allocating, which is exactly the kind of opportunity the deadline and the capacity are already
    identity-bearing for."""
    from shogym.envs.appworld.env_v1 import run_fingerprint

    base = run_fingerprint(pulse=0, report="graded", blocks=60, resources="2|2g")
    assert base != run_fingerprint(pulse=0, report="graded", blocks=60, resources="8|16g")
    # Captured once for the process, so an environment changed under a running run cannot move
    # what its later episodes were given.
    container.limits.cache_clear()
    monkeypatch.setenv("SHOGYM_APPWORLD_CPUS", "3")
    first = container.limits()
    monkeypatch.setenv("SHOGYM_APPWORLD_CPUS", "7")
    assert container.limits() == first
    container.limits.cache_clear()


def test_the_block_budget_counts_what_the_host_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply cannot earn a block, because a block is spent when the request goes out.

    The protocol's writer is reachable from inside the interpreter that runs agent code, so a
    forged completion can reach the parent before the real one. What it cannot do is add to the
    budget: the count is incremented under the session lock before the call is made, so it counts
    requests this process sent and not answers it received."""
    import inspect

    from shogym.envs.appworld import mcp_server

    source = inspect.getsource(mcp_server.execute)
    before, _, after = source.partition("session.calls += 1")
    assert before and after
    # The increment happens inside the lock, before the worker is called at all.
    assert "worker.call" not in before
    assert "worker.call" in after

def test_a_birth_stamp_survives_a_single_digit_day(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bug this had, and it could have removed a live episode.

    `ps` blank-pads a single-digit day, so a label written on the third of the month carried two
    spaces where a reader rebuilding it from words put one. The comparison then said the parent
    had been replaced, and the sweep removes what an absent parent left: on days one to nine it
    would have removed a live parent's own running world, or a sibling arm's.

    The stamp is epoch seconds now, which has no spacing to lose."""
    rendered = {"value": "Fri Aug  3 07:15:26 2026"}
    monkeypatch.setattr(
        container.subprocess,
        "run",
        lambda *a, **k: _Finished(0, rendered["value"]),
    )
    padded = container.process_birth(4242)
    assert padded.isdigit(), padded
    # The same instant written the way a reader might rebuild it is the same stamp.
    rendered["value"] = "Fri Aug 3 07:15:26 2026"
    assert container.process_birth(4242) == padded
    # And a different instant is a different stamp, so this is not a constant.
    rendered["value"] = "Fri Aug  4 07:15:26 2026"
    assert container.process_birth(4242) != padded


def test_a_live_parent_on_a_padded_day_keeps_its_container(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same failure at the level that would have caused it: the sweep, not the parse."""
    stamp = "1787908573"
    # This process is the parent, so it is genuinely alive: what is being tested is whether the
    # stamp comparison says so.
    labels = (
        f'{{"{container.LABEL_PARENT}": "{os.getpid()}", "{container.LABEL_BIRTH}": "{stamp}"}}'
    )
    removed: List[str] = []

    def _run(args, **_):
        if args[0] == "ps":
            return _Finished(0, "live1")
        if args[0] == "inspect":
            return _Finished(0, labels)
        removed.append(args[-1])
        return _Finished(0, "")

    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    monkeypatch.setattr(container, "_run", _run)
    monkeypatch.setattr(container, "process_birth", lambda pid: stamp)
    assert container.reap() == []
    assert removed == []


def test_a_removal_the_daemon_rejected_is_not_a_removal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Teardown may not raise, and it may not pretend either.

    Both Docker commands run without checking, so an ordinary nonzero stop or removal reached the
    success branch and the worker was marked closed. Nothing tried again after that, and the
    ordinary sweep skips a container whose parent is alive, so a wedged container outlived the
    process that made it. The name goes to the ledger instead."""
    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    monkeypatch.setattr(container, "_run", lambda *a, **k: _Finished(1, "", "device or resource busy"))
    assert container.remove("wedged") is False
    assert container.outstanding() == ["wedged"]
    # And a clean pair of commands is a removal, so the refusal above is about the failure.
    monkeypatch.setattr(container, "_run", lambda *a, **k: _Finished(0, ""))
    assert container.remove("fine") is True
    assert container.outstanding() == ["wedged"]


def test_a_worker_whose_removal_was_rejected_is_not_marked_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The same fact one layer up, where it decides whether anything tries again."""
    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    real_remove = container.remove
    worker = _stub_worker(_ECHO, monkeypatch)
    # The stub points removal at nothing; this puts the real one back so the daemon's answer is
    # what decides, and then makes the daemon refuse.
    monkeypatch.setattr(container, "remove", real_remove)
    monkeypatch.setattr(
        container, "_run", lambda *a, **k: _Finished(1, "", "device or resource busy")
    )
    worker.close()
    assert worker.closed is False
    assert worker.container in container.outstanding()


def test_a_name_appended_during_a_sweep_is_not_lost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lost update the ledger used to have.

    The sweep read the whole file and then published a new one, so a name appended in between was
    dropped, and the case where that matters is the only case the ledger is for: a container whose
    parent is alive, which the ordinary sweep skips. Every line is an event now, so a writer never
    has to know what a reader is holding."""
    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    container.disowned("first")

    def _run(args, **_):
        if args[0] == "stop":
            # Racing the sweep, in the window the old one lost: a second process disowns while
            # this one is between reading the ledger and writing it back.
            container.disowned("second")
            return _Finished(0, "")
        if args[0] == "inspect":
            return _Finished(1, "", "Error: No such object")
        return _Finished(0, "")

    monkeypatch.setattr(container, "_run", _run)
    assert container._sweep_disowned() == ["first"]
    # The one that arrived mid-sweep is still there to be tried.
    assert container.outstanding() == ["second"]


def test_a_worker_with_a_call_in_flight_does_not_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """What finalization asks before it stops a container.

    A terminal may overtake an ordinary call, which the serve layer does on purpose. What must not
    follow is removing the container while upstream is inside the save it ends every block with:
    the lock is held for the length of a call, so acquiring it is the fact that none is running,
    and failing to inside the bound is the fact that one is."""
    import threading

    worker = _stub_worker(_ECHO, monkeypatch)
    try:
        assert worker.settle(0.5) is True
        held = threading.Event()
        released = threading.Event()

        def _hold() -> None:
            with worker.lock:
                held.set()
                released.wait(5)

        thread = threading.Thread(target=_hold, daemon=True)
        thread.start()
        assert held.wait(5)
        assert worker.settle(0.2) is False
        released.set()
        thread.join(5)
        assert worker.settle(0.5) is True
    finally:
        worker.close()


def test_a_spawn_that_never_becomes_ready_releases_everything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Removal can raise, and everything after it used to be skipped.

    A control timeout raises even with the status unchecked, and the local client and its pipes
    were released after that call. Nothing was going to come back for them, because spawn returned
    no worker; and the ordinary sweep skips the labelled container because its parent is alive."""
    monkeypatch.setattr(container, "_ledger", lambda: tmp_path / "disowned.txt")
    import subprocess as _sub

    process = _sub.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdin=_sub.PIPE,
        stdout=_sub.PIPE,
        stderr=_sub.DEVNULL,
        bufsize=0,
    )

    def _refuses(name: str, *, confirm: bool = False) -> bool:
        raise container.DockerError("`docker rm` did not finish within 10s")

    monkeypatch.setattr(container, "remove", _refuses)
    adapter._release(process, "stuck-container")
    # The container went to the ledger, and this process's own handles went whatever the daemon
    # said.
    assert container.outstanding() == ["stuck-container"]
    assert process.poll() is not None
    assert process.stdout is None or process.stdout.closed


def test_a_snapshot_that_is_a_whole_prefix_is_not_a_whole_save(tmp_path: Path) -> None:
    """A newline is not the end of a save, and that was the hole.

    Upstream clears the database directory and writes the logs one after another, one JSON record
    per line, so an interruption between two complete records leaves every expected filename in
    place and a syntactically perfect tail with a suffix of state simply missing. Nothing about
    the bytes says a record should have followed. What says so is a length written after the save
    returned, and the block the host asked for: a save that never finished leaves the manifest of
    the block before it."""
    served = tmp_path / "served" / "dbs"
    served.mkdir(parents=True)
    for name in ("todoist.jsonl", "gmail.jsonl"):
        (served / name).write_text('{"row": 1}\n')
    snapshot = tmp_path / "snap"
    task = snapshot / "tasks" / "abc_1"
    dbs = task / "dbs"
    dbs.mkdir(parents=True)
    whole = '{"row": 1}\n{"row": 2}\n'
    (dbs / "todoist.jsonl").write_text(whole)
    (dbs / "gmail.jsonl").write_text(whole)

    def _record(block: int) -> None:
        (task / "save.manifest").write_text(
            json.dumps(
                {
                    "block": block,
                    "files": {p.name: p.stat().st_size for p in sorted(dbs.iterdir())
                              if p.suffix == ".jsonl"},
                }
            )
        )

    _record(3)
    # A whole save passes, so the refusals below are about the damage.
    adapter.verify_snapshot(snapshot, task_id="abc_1", expected=served, blocks=3)

    # Cut between two complete records: every name present, the tail a perfect JSON object, and a
    # record missing after it.
    (dbs / "gmail.jsonl").write_text('{"row": 1}\n')
    with pytest.raises(adapter.SnapshotError, match="bytes where the save"):
        adapter.verify_snapshot(snapshot, task_id="abc_1", expected=served, blocks=3)

    # And an empty file where the save had written records, which the earlier check permitted.
    (dbs / "gmail.jsonl").write_text("")
    with pytest.raises(adapter.SnapshotError, match="bytes where the save"):
        adapter.verify_snapshot(snapshot, task_id="abc_1", expected=served, blocks=3)

    # A save that never finished leaves the record of the block before it.
    (dbs / "gmail.jsonl").write_text(whole)
    _record(2)
    with pytest.raises(adapter.SnapshotError, match="did not finish"):
        adapter.verify_snapshot(snapshot, task_id="abc_1", expected=served, blocks=3)

    # And a snapshot with no record of its save at all.
    (task / "save.manifest").unlink()
    with pytest.raises(adapter.SnapshotError, match="no record of the save"):
        adapter.verify_snapshot(snapshot, task_id="abc_1", expected=served, blocks=3)


def test_verification_reads_in_chunks_and_stops_when_it_is_abandoned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A permitted tree may be most of a gibibyte, and this runs in the serving process.

    Reading one of its files whole would put that allocation where the container's memory limit
    does not reach, after a copy that was careful not to. And the copy takes an abandon flag while
    the check that follows it took none, so a cancelled finalization left this walking."""
    import threading

    served = tmp_path / "served" / "dbs"
    served.mkdir(parents=True)
    (served / "todoist.jsonl").write_text('{"row": 1}\n')
    snapshot = tmp_path / "snap"
    task = snapshot / "tasks" / "abc_1"
    dbs = task / "dbs"
    dbs.mkdir(parents=True)
    body = "".join('{"row": %d}\n' % n for n in range(20_000))
    (dbs / "todoist.jsonl").write_text(body)
    (task / "save.manifest").write_text(
        json.dumps({"block": 1, "files": {"todoist.jsonl": len(body)}})
    )
    # Nothing is read whole: the buffer is a chunk, whatever the file is.
    monkeypatch.setattr(adapter, "_VERIFY_CHUNK", 4096)
    adapter.verify_snapshot(snapshot, task_id="abc_1", expected=served, blocks=1)

    abandoned = threading.Event()
    abandoned.set()
    with pytest.raises(adapter.SnapshotError, match="abandoned"):
        adapter.verify_snapshot(
            snapshot, task_id="abc_1", expected=served, blocks=1, stop=abandoned
        )

    # A name the save did not record is not opened at all, so a file an episode planted cannot
    # cost the host a read.
    (dbs / "planted.jsonl").write_text("not json at all")
    adapter.verify_snapshot(snapshot, task_id="abc_1", expected=served, blocks=1)


def test_a_frame_of_the_wrong_shape_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid JSON is not a valid answer, and the difference used to escape the boundary.

    A scalar or a list reached the caller, which did `.get` on it; an object without `output`
    reached the caller, which indexed it. Both came out as an `AttributeError` or a `KeyError`
    from the transport, with the worker neither poisoned nor stopped. The writer is reachable from
    inside the interpreter that runs agent code, so the shape of an answer is part of framing."""
    for shape in ('"a string"', "[1, 2, 3]", '{"id": 1, "something": "else"}'):
        body = shape.encode()
        emit = (
            "body = {body!r}\n"
            "    w.write(str(len(body)).encode() + b'\\n')\n"
            "    w.write(body)\n"
            "    w.flush()"
        ).format(body=body)
        stub = _ECHO.replace(
            'send({"id": request["id"], "output": {"saw": request["command"]}})', emit
        )
        stopped: List[str] = []
        worker = _stub_worker(stub, monkeypatch)
        monkeypatch.setattr(container, "remove", lambda name, confirm=False: stopped.append(name))
        try:
            with pytest.raises(adapter.WorkerError) as refused:
                worker.call("execute")
            assert "protocol was broken" in str(refused.value), shape
            assert worker.poisoned
            assert stopped == [worker.container]
        finally:
            monkeypatch.undo()


def test_ownership_lives_beside_the_tree_and_not_inside_it(tmp_path: Path) -> None:
    """Cleanup authority was kept in the tree it governs, which the episode can write.

    The output root is bound into the container at `/outputs` writable, so a marker in there is a
    fact its own subject could rewrite: a forged one would have made a sibling constructor remove
    a live tree, and an unreadable one would have kept a dead tree for ever. The control plane is
    a directory of its own that no mount set names."""
    from shogym.envs.appworld import env_v1

    home = tmp_path / "episodes"
    root = home / "episode-abc"
    root.mkdir(parents=True)
    monkey = adapter.control_file(root, "owner")
    # Nothing of the control plane is under the tree it is about.
    assert not str(monkey).startswith(str(home))
    env_v1._claim_tree(root)
    assert monkey.exists()
    assert not list(root.iterdir())

    # A forged marker inside the tree changes nothing: it is not where anybody looks.
    (root / ".shogym-owner").write_text("1 forged\n")
    assert env_v1._reclaimable(root) is False
    monkey.write_text("999999 1700000000\n")
    assert env_v1._reclaimable(root) is True


def test_a_tree_teardown_declined_to_walk_is_reclaimed_while_the_server_lives(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The path that exists to reclaim what teardown could not delete could not reclaim it.

    Teardown leaves an oversized tree rather than walking it inside somebody else's deadline. The
    tree still named the process that made it, which is the serving one and is alive, so the sweep
    kept it until that process exited. An episode that ended says so, and after a short grace the
    tree is reclaimable however alive the server is."""
    from shogym.envs.appworld import env_v1

    home = tmp_path / "episodes"
    root = home / "episode-big"
    root.mkdir(parents=True)
    (root / "state").write_text("x")
    env_v1._claim_tree(root)
    # This process is alive, so nothing takes it while the episode is running.
    assert env_v1._reclaimable(root) is False

    monkeypatch.setattr(env_v1, "_DISCARD_MAX_NODES", 0)
    env_v1._discard(root)
    # Declined, and said to be over.
    assert root.exists()
    assert env_v1._reclaimable(root) is False  # inside the grace
    monkeypatch.setattr(env_v1, "_ENDED_GRACE_SECONDS", -1.0)
    assert env_v1._reclaimable(root) is True
    env_v1._sweep_leftovers(home)
    assert not root.exists()
    # And its control-plane records go with it.
    assert not adapter.control_file(root, "ended").exists()
