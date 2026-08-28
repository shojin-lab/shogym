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
    # and a stub that answers every `Popen` would otherwise answer that one too.
    container._boot_id()
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
    assert passed == ["APPWORLD_ROOT=/corpus"]
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


def test_a_timed_out_call_makes_the_worker_unusable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout on an ordered pipe is not a failure that ends when the caller stops waiting.

    HTTP gave each response its own connection, so abandoning one cost nothing. A pipe is one
    ordered stream: the command that timed out is still running, its answer is still coming, and
    the world it is running against is still changing. There is no state in which reusing that
    worker is right, so it is refused, and the refusal says why."""
    monkeypatch.setattr(adapter, "_CALL_TIMEOUT_SECONDS", 0.4)
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
    def __init__(self, returncode: int, stdout: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = ""


def test_removal_confirms_by_asking_the_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    """`docker rm -f` returning zero is not the same fact as the container being gone."""
    calls: List[List[str]] = []

    def _run(args, **_):
        calls.append(list(args))
        return _Finished(0, "")

    monkeypatch.setattr(container, "_run", _run)
    monkeypatch.setattr(container, "absent", lambda name: True)
    container.remove("c", confirm=True)
    assert calls and calls[0][:2] == ["rm", "-f"]
    # Without confirmation it does not ask, because teardown must not pay for a question whose
    # answer it would ignore.
    calls.clear()
    container.remove("c")
    assert len(calls) == 1


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
    monkeypatch: pytest.MonkeyPatch,
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
    labels = {"dead1": "4242", "alive1": "4243", "unlabelled": ""}
    removed: List[str] = []

    def _run(args, **_):
        if args[0] == "ps":
            return _Finished(0, "\n".join(listed))
        if args[0] == "inspect":
            return _Finished(0, labels[args[-1]])
        if args[0] == "rm":
            removed.append(args[-1])
            return _Finished(0, "")
        raise AssertionError(args)

    monkeypatch.setattr(container, "_run", _run)
    swept = container.reap(alive=lambda pid: pid == 4243)
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

