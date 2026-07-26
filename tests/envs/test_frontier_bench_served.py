"""End-to-end, Docker-gated: serve a frontier_bench episode through the seal contract.

These build+run the task's containers, so they need a working Docker daemon and are **skipped**
without one — the offline core suite (fidelity + pure verify + the offline exploit proof) stays
green regardless. The two sanity gates from issue #48 are the point, per vendored task: the
task's own oracle (``solution/solve.sh``) scores **1** through this port's verifier, and an empty
run (``nop``) scores **0** — now flowing through the ``score``-terminal ``finalize`` + ``_verify``
seal path (a ``done`` call seals the episode and finalize runs the verifier over the container
end-state).

These are heavy (image builds + a pip install inside the container + a pytest verifier run), so
they can take a few minutes per task on a cold cache.
"""

from __future__ import annotations

import json

import pytest

from hgym.envs.frontier_bench import docker_backend as dk
from hgym.envs.frontier_bench import manifest

pytestmark = pytest.mark.skipif(
    not dk.docker_available(),
    reason="Docker daemon not available; frontier_bench end-to-end path is Docker-gated",
)

from hgym.envs.frontier_bench import mcp_server  # noqa: E402
from hgym.serve import ServedEpisode  # noqa: E402

VENDORED = manifest.task_names()


async def test_served_shell_and_nop_done_seals_and_scores_zero() -> None:
    """Drive the served shell (exec / write_file / read_file), then a nop `done` seals + scores 0.

    `done` is the score terminal: the call terminates the episode (no separate `terminate`), the
    verifier runs over the empty container state, and `_verify` scores reward 0 off the evidence.
    """
    episode = await ServedEpisode.start("frontier_bench", task=0)
    try:
        spec = episode.describe()
        assert {"exec", "read_file", "write_file", "done", "terminate"} <= {
            t.name for t in spec.tools
        }
        # `done` is advertised as the score terminal; `terminate` as the abort terminal.
        by_kind = {t.name: t.terminal_kind for t in spec.tools}
        assert by_kind["done"] == "score"
        assert by_kind["terminate"] == "abort"

        # The task's seed inputs are present in the container.
        ls = json.loads((await episode.call("exec", {"command": "ls /app/inputs"})).content)
        assert ls["ok"], ls
        assert "portfolio.csv" in ls["stdout"]

        # write_file / read_file round-trip through docker cp.
        w = json.loads(
            (await episode.call("write_file", {"path": "/app/scratch.txt", "content": "hi"})).content
        )
        assert w["ok"], w
        r = json.loads((await episode.call("read_file", {"path": "/app/scratch.txt"})).content)
        assert r["ok"] and r["content"] == "hi", r

        # nop: `done` without producing the required outputs seals + runs the verifier → reward 0.
        res = await episode.call("done", {})
        assert res.terminated is True
        verdict = json.loads(res.content)
        assert verdict["reward"] == 0.0
        assert verdict["success"] is False
        assert verdict["finalize_error"] is False

        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["reward"] == 0.0
        assert feedback["success"] is False
        assert feedback["verified"] is True
    finally:
        await episode.close()


async def test_served_oracle_scores_one_through_verify() -> None:
    """The oracle sanity gate through the full seal path: run the task's own oracle inside the
    served container, then `done` seals + finalize runs the verifier → reward 1 via `_verify`."""
    episode = await ServedEpisode.start("frontier_bench", task=0)
    try:
        # Run the vendored oracle inside the live served container (a session helper).
        session = mcp_server._sessions[episode.session_id]
        oracle = session.run_oracle()
        assert oracle["ok"], oracle["stderr"][-2000:]

        res = await episode.call("done", {})
        assert res.terminated is True
        verdict = json.loads(res.content)
        assert verdict["reward"] == 1.0, verdict
        assert verdict["success"] is True
        # No grader internals leak to the agent even on a pass.
        assert "test_stdout_tail" not in verdict
        assert "test_exit_code" not in verdict

        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["reward"] == 1.0
        assert feedback["success"] is True
        assert feedback["verified"] is True
    finally:
        await episode.close()


async def test_served_done_is_sealed_no_reretry() -> None:
    """The read-and-retry exploit is closed by the seal, proven with a REAL container: after a
    failing nop `done`, a "fix"-then-re-`done` is tombstoned — the verifier never re-runs and the
    honest 0 stands."""
    episode = await ServedEpisode.start("frontier_bench", task=0)
    try:
        first = await episode.call("done", {})
        assert first.terminated is True
        assert json.loads(first.content)["reward"] == 0.0

        # "Fix" and re-grade: both tombstoned (no dispatch, no second verifier run).
        reread = await episode.call("exec", {"command": "echo fixed > /app/output/x"})
        assert "sealed" in reread.content
        again = await episode.call("done", {})
        assert "sealed" in again.content

        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["reward"] == 0.0  # the honest first verdict stands
        assert feedback["success"] is False
    finally:
        await episode.close()


@pytest.mark.parametrize("task_name", VENDORED)
def test_oracle_scores_one(task_name: str) -> None:
    """Each task's own oracle (solve.sh) scores 1 through this port's SEPARATE-mode verifier."""
    session_id = f"test-frontier-oracle-{task_name}"
    mcp_server.begin_session(session_id, task_name=task_name)
    try:
        session = mcp_server._sessions[session_id]
        oracle = session.run_oracle()
        assert oracle["ok"], oracle["stderr"][-2000:]
        outcome = session.finalize()
        assert outcome.reward == 1.0, outcome
        # All declared artifacts were collected off the container end-state.
        assert all(outcome.artifacts_collected.values()), outcome.artifacts_collected
    finally:
        mcp_server.end_session(session_id)


async def test_served_named_task_via_env_config() -> None:
    """Serve a task selected **by name** through ``env_config`` (the supported name path — the
    ``--task`` flag itself resolves to an integer index in the shared serve layer). The named
    default must build+serve that concrete task's container, not the index-0 default."""
    episode = await ServedEpisode.start(
        "frontier_bench", env_config={"task": "interleaved-vigenere"}
    )
    try:
        spec = episode.describe()
        # describe() resolves the configured default -> the named task's instruction.
        assert "interleaved-vigenere" in spec.instructions
        # The served container is the named task's, not fin-saccr's: its cracker input is present
        # and fin-saccr's portfolio.csv is absent.
        ls = json.loads((await episode.call("exec", {"command": "ls /app"})).content)
        assert ls["ok"], ls
        assert "portfolio.csv" not in ls["stdout"]
        res = await episode.call("done", {})  # nop seals + scores 0
        assert res.terminated is True
        assert json.loads(res.content)["reward"] == 0.0
    finally:
        await episode.close()


def test_build_task_image_builds_and_is_idempotent() -> None:
    """The preflight/serve build path: build_task_image builds the env image and is a no-op
    when it's already cached (a second call returns the same tag without rebuilding)."""
    tag = mcp_server.build_task_image("fin-saccr-rwa")
    assert dk.image_exists(tag)
    # Idempotent: cached tag, no rebuild.
    assert mcp_server.build_task_image("fin-saccr-rwa") == tag
    assert dk.image_exists(tag)


@pytest.mark.parametrize("task_name", VENDORED)
def test_nop_session_scores_zero_directly(task_name: str) -> None:
    """A fresh container with no outputs scores 0 (the anti-oracle sanity gate), per task."""
    session_id = f"test-frontier-nop-{task_name}"
    mcp_server.begin_session(session_id, task_name=task_name)
    try:
        session = mcp_server._sessions[session_id]
        outcome = session.finalize()
        assert outcome.reward == 0.0
    finally:
        mcp_server.end_session(session_id)
