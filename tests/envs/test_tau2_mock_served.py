"""End-to-end: drive a served ``tau2_mock`` episode through hgym's serve layer and check
that tau2's evaluator verdict flows back into episode feedback.

Requires the ``tau2`` extra (Python <3.14) and a loadable tau2 ``mock`` data set — skipped
otherwise, so the offline core suite stays green. Solo mode ⇒ no user-simulator LLM ⇒ the
whole path (engine + tools + evaluator) runs offline.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tau2", reason="tau2 extra not installed")

import hgym  # noqa: E402
from hgym.envs.tau2 import mcp_server  # noqa: E402
from hgym.serve import ServedEpisode  # noqa: E402


def _mock_task_index(task_id: str) -> int:
    """Resolve a tau2 mock task id to its index in the env's train split, or skip."""
    try:
        env = hgym.make("tau2_mock")
    except Exception as exc:  # missing data etc.
        pytest.skip(f"tau2 mock env not constructible offline: {exc}")
    if task_id not in env._task_ids:
        pytest.skip(f"task {task_id} not in mock train split")
    return env._task_ids.index(task_id)


async def _drive(episode: ServedEpisode, tool: str, args: dict):
    return await episode.call(tool, args)


async def test_served_mock_episode_scores_success() -> None:
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx)
    try:
        spec = episode.describe()
        tool_names = {t.name for t in spec.tools}
        assert {"create_task", "done", "terminate"} <= tool_names
        # The task contract surfaces the domain policy + this task's ticket.
        assert "Important Meeting" in spec.instructions

        # Perform the task's expected action, then finish + end the episode.
        r1 = await _drive(
            episode,
            "create_task",
            {"user_id": "user_1", "title": "Important Meeting"},
        )
        assert not r1.terminated
        assert "Important Meeting" in r1.content  # tool output echoed back

        r2 = await _drive(episode, "done", {})
        verdict = json.loads(r2.content)
        assert verdict[mcp_server.VERDICT_MARKER] is True
        assert verdict["reward"] == 1.0
        assert verdict["db_match"] is True

        r3 = await _drive(episode, "terminate", {})
        assert r3.terminated
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["reward"] == 1.0
        assert feedback["success"] is True
        assert feedback["db_match"] is True
        assert feedback["action_match_proportion"] == 1.0
    finally:
        await episode.close()


async def test_upstream_auto_termination_reaches_terminal_feedback() -> None:
    # When tau2 auto-terminates on a domain-tool step (here: max_steps exhausted during the
    # `create_task` action), the verdict must not be lost. The bridge stashes it and surfaces
    # it on the `done` step, so hgym's terminal feedback equals the evaluator's verdict — not
    # a silent premature zero produced independently by the verifier.
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx, env_config={"max_steps": 1})
    try:
        await _drive(episode, "create_task", {"user_id": "user_1", "title": "Important Meeting"})
        done = await _drive(episode, "done", {})
        verdict = json.loads(done.content)
        assert verdict[mcp_server.VERDICT_MARKER] is True  # real evaluator verdict, not an error
        await _drive(episode, "terminate", {})
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        # Terminal feedback reflects the evaluator's verdict (consistency), not an independent 0.
        assert feedback["reward"] == verdict["reward"]
    finally:
        await episode.close()


async def test_served_mock_premature_terminate_scores_zero() -> None:
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx)
    try:
        # End the episode without ever completing the task via `done`.
        result = await _drive(episode, "terminate", {})
        assert result.terminated
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["reward"] == 0.0
        assert feedback["success"] is False
    finally:
        await episode.close()
