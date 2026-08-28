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

from pathlib import Path
from typing import Any, Dict, List

import pytest

from shogym.envs.appworld import adapter, container


def _corpus(root: Path) -> Path:
    """A derived root with the shape the real one has: shared parts, two tasks, an output tree."""
    data = root / "data"
    for name in ("api_docs", "base_dbs", "datasets"):
        (data / name).mkdir(parents=True)
    (data / "version.txt").write_text("0.1.0\n")
    for task in ("abc_1", "def_2"):
        (data / "tasks" / task / "dbs").mkdir(parents=True)
        (data / "tasks" / task / "specs.json").write_text("{}")
    (root / "experiments" / "outputs").mkdir(parents=True)
    return root


# ----- the mount set -----


def test_one_episode_is_given_one_task_and_one_output_tree(tmp_path: Path) -> None:
    """The corpus holds 318 tasks and a run's output tree holds one directory per episode, so
    mounting either wholesale would put every other task's world and every sibling episode's end
    state one ``listdir`` away. The mount set names one of each."""
    root = _corpus(tmp_path / "seeded")
    outputs = root / "experiments" / "outputs" / "shogym-one"
    mounts = adapter.served_mounts(
        root=root, task_id="abc_1", outputs=outputs, experiment="shogym-one"
    )
    targets = {mount.target: mount for mount in mounts}
    assert targets.keys() == {
        "/corpus/data/api_docs",
        "/corpus/data/base_dbs",
        "/corpus/data/datasets",
        "/corpus/data/version.txt",
        "/corpus/data/tasks/abc_1",
        "/corpus/experiments/outputs/shogym-one",
    }
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
        root=root,
        task_id="abc_1",
        outputs=root / "experiments" / "outputs" / "shogym-one",
        experiment="shogym-one",
    )
    writable = [mount.target for mount in mounts if mount.writable]
    assert writable == ["/corpus/experiments/outputs/shogym-one"]
    for mount in mounts:
        assert mount.as_argument().endswith(":rw" if mount.writable else ":ro")


def test_the_graders_view_is_the_answers_and_the_end_state_and_they_are_two_trees(
    tmp_path: Path,
) -> None:
    """The answers live in a private tree and the end state under the served root, and the
    evaluator wants both under one root. That used to be a symlink from the private tree into the
    served one, published under a lock because two cold constructors raced on creating it. It is
    two mounts now, so there is no link and no race."""
    graded = _corpus(tmp_path / "graded")
    (graded / "data" / "tasks" / "abc_1" / "ground_truth").mkdir()
    outputs = tmp_path / "seeded" / "experiments" / "outputs" / "shogym-one"
    mounts = adapter.graded_mounts(
        graded=graded, task_id="abc_1", outputs=outputs, experiment="shogym-one"
    )
    targets = {mount.target for mount in mounts}
    assert "/graded/data/tasks/abc_1" in targets
    assert "/graded/experiments/outputs/shogym-one" in targets
    # The two sources are in different trees on the host, which is the point of the second mount.
    sources = {mount.target: mount.source for mount in mounts}
    assert sources["/graded/experiments/outputs/shogym-one"] == outputs
    assert not str(outputs).startswith(str(graded))


# ----- the flags -----


def _captured(monkeypatch: pytest.MonkeyPatch) -> List[List[str]]:
    """Run one container without a daemon, and hand back the command line it would have used."""
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
