"""Keyed fidelity check: the shogym bridge reproduces upstream tau2's simulation + scoring.

Issue #31 asks that, for a fixed agent policy + seed, shogym-driven scores match upstream
``tau2 run``. We assert this directly: replay a task's *gold agent actions* through both
(a) the shogym MCP bridge and (b) tau2's own upstream ``AgentGymEnv``, on the same task with
the **real user simulator**, and require the resulting tau2 evaluator scores (reward,
db_match, action_match_proportion) to be **equal**. The agent policy is a fixed tool-call
sequence, so the environment outcome is deterministic regardless of the (LLM) user
simulator's replies — making the equality assertion robust, while the real user simulator is
still exercised (its opening turn) and therefore requires ``OPENAI_API_KEY``.

Skipped when the key is absent, so offline CI stays green; run it with a key to confirm
real-user-sim fidelity.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("tau2", reason="tau2 extra not installed")

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set; keyed fidelity test skipped", allow_module_level=True)

from shogym.serve import ServedEpisode  # noqa: E402


def _gold_agent_actions(domain: str, task) -> list[tuple[str, dict]]:
    """The task's gold actions attributed to the assistant (agent), in order."""
    ec = getattr(task, "evaluation_criteria", None)
    actions = ec.actions if ec else []
    return [
        (a.name, dict(a.arguments or {}))
        for a in actions
        if getattr(a, "requestor", None) == "assistant"
    ]


def _pick_task(env_name: str, domain: str):
    """Pick a task from the env's train split that has replayable gold agent actions.

    Returns ``(index_in_split, task)`` — the shogym path indexes into the env's split, while the
    upstream path uses ``task.id``."""
    import shogym
    from shogym.envs.tau2 import mcp_server

    env = shogym.make(env_name)
    by_id = {t.id: t for t in mcp_server.load_tasks(domain)}
    for i, task_id in enumerate(env._task_ids):
        task = by_id.get(task_id)
        if task is not None and _gold_agent_actions(domain, task):
            return i, task
    # Fall back to the first split task with an empty (done-only) policy.
    return 0, by_id[env._task_ids[0]]


async def _score_via_shogym(env_name: str, task_idx: int, actions: list[tuple[str, dict]]) -> dict:
    episode = await ServedEpisode.start(env_name, task=task_idx)  # real user sim (default llm)
    try:
        for name, args in actions:
            await episode.call(name, args)
        done = await episode.call("done", {})
        return json.loads(done.content)
    finally:
        await episode.close()


def _score_via_upstream(domain: str, task_id: str, actions: list[tuple[str, dict]]) -> dict:
    from tau2.gym.gym_agent import AgentGymEnv

    # Same step budget as the shogym env's default (Tau2Env.DEFAULT_MAX_STEPS == upstream's 100),
    # so neither side hits MAX_STEPS differently.
    gym = AgentGymEnv(domain=domain, task_id=task_id, solo_mode=False, max_steps=100)
    gym.reset()
    info = {}
    for name, args in actions:
        _obs, _r, _term, _trunc, info = gym.step(json.dumps({"name": name, "arguments": args}))
    _obs, _r, _term, _trunc, info = gym.step(json.dumps({"name": "done", "arguments": {}}))
    ri = json.loads(info.get("reward_info") or "{}")
    db = ri.get("db_check") or {}
    db_match = db.get("db_match") if isinstance(db, dict) else None
    checks = ri.get("action_checks") or []
    amp = (sum(1 for a in checks if a.get("action_match")) / len(checks)) if checks else None
    return {
        "reward": ri.get("reward"),
        "db_match": db_match if isinstance(db_match, bool) else None,
        "action_match_proportion": amp,
    }


@pytest.mark.parametrize("env_name,domain", [("tau2_airline", "airline"), ("tau2_telecom", "telecom")])
async def test_shogym_scores_match_upstream(env_name: str, domain: str) -> None:
    task_idx, task = _pick_task(env_name, domain)
    actions = _gold_agent_actions(domain, task)

    hg = await _score_via_shogym(env_name, task_idx, actions)
    up = _score_via_upstream(domain, task.id, actions)

    assert hg["reward"] == up["reward"], f"{domain}: reward {hg['reward']} != upstream {up['reward']}"
    assert hg["db_match"] == up["db_match"], f"{domain}: db_match mismatch"
    assert hg["action_match_proportion"] == up["action_match_proportion"], (
        f"{domain}: action_match {hg['action_match_proportion']} != {up['action_match_proportion']}"
    )
