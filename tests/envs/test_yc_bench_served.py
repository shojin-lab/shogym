"""End-to-end: drive a served ``yc_bench`` episode through hgym's serve layer and check that
YC-Bench's sim state flows back into episode feedback.

Requires the ``yc_bench`` extra — skipped otherwise, so the offline core suite stays green.
YC-Bench generates its whole world deterministically from the seed and runs its sim in
process, so the entire path (seed → CLI commands → terminal verdict) runs offline, with no
model calls or API keys.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("yc_bench", reason="yc_bench extra not installed")

from hgym.envs.yc_bench import mcp_server  # noqa: E402
from hgym.serve import ServedEpisode  # noqa: E402


def _stdout(result_content: str) -> dict:
    """Parse the CLI JSON out of a `run_command` tool result."""
    payload = json.loads(result_content)
    assert payload["ok"], payload
    return json.loads(payload["stdout"])


async def _play_one_task(episode: ServedEpisode) -> dict:
    """Accept/assign/dispatch one market task and advance the sim once; return the sim-resume
    payload."""
    emps = [e["name"] for e in _stdout((await episode.call("run_command", {"command": "yc-bench employee list"})).content)["employees"]][:3]
    task_id = _stdout((await episode.call("run_command", {"command": "yc-bench market browse --limit 1"})).content)["tasks"][0]["task_id"]
    for cmd in (
        f"yc-bench task accept --task-id {task_id}",
        f"yc-bench task assign --task-id {task_id} --employees {','.join(emps)}",
        f"yc-bench task dispatch --task-id {task_id}",
    ):
        r = await episode.call("run_command", {"command": cmd})
        assert json.loads(r.content)["ok"], r.content
    return _stdout((await episode.call("run_command", {"command": "yc-bench sim resume"})).content)


async def test_served_episode_reads_state_and_premature_submit_scores_zero() -> None:
    episode = await ServedEpisode.start("yc_bench", task=0)
    try:
        spec = episode.describe()
        tool_names = {t.name for t in spec.tools}
        assert {"run_command", "submit", "terminate"} <= tool_names
        assert "CEO" in spec.instructions

        # A fresh company starts solvent at $200,000.
        status = _stdout((await episode.call("run_command", {"command": "yc-bench company status"})).content)
        assert status["funds_cents"] == 20_000_000

        resume = await _play_one_task(episode)
        assert "balance_delta" in resume  # the sim advanced and reported financials

        # `submit` reads the live sim state (still mid-run: not bankrupt, horizon not reached).
        verdict = json.loads((await episode.call("submit", {})).content)
        assert verdict[mcp_server.VERDICT_MARKER] is True
        assert verdict["seeded"] is True
        assert isinstance(verdict["final_funds_cents"], int)
        assert verdict["terminal_reason"] is None  # the run has not actually ended

        term = await episode.call("terminate", {})
        assert term.terminated
        # Anti-gaming: submitting before a terminal state scores a premature zero, not the
        # current (solvent, pre-horizon) funds — even though the sim read them fine.
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["reward"] == 0.0
        assert feedback["final_funds_cents"] == 0.0
        assert feedback["survived"] is False
        assert feedback["success"] is False
    finally:
        await episode.close()


async def test_bankrupt_state_is_terminal_and_credits_negative_funds() -> None:
    # Drive the sim DB to bankruptcy directly, then confirm `submit` reports a genuine terminal
    # verdict and the pure scorer credits the negative terminal funds (integration of
    # read_final_state → score_trajectory).
    from yc_bench.db.models.company import Company
    from yc_bench.db.session import session_scope

    from hgym.envs.yc_bench.env_v1 import score_trajectory
    from hgym.trajectory import Step

    session_id = "test-yc-bankrupt"
    mcp_server.begin_session(
        session_id,
        seed=5,
        config_name="default",
        start_date="2025-01-01",
        horizon_years=None,
        company_name="BenchCo",
    )
    try:
        session = mcp_server._sessions[session_id]
        with session_scope(session._factory) as db:
            db.query(Company).update({Company.funds_cents: -12_345})
        verdict = session.verdict()
        assert verdict["terminal_reason"] == "bankruptcy"
        assert verdict["survived"] is False
        assert verdict["final_funds_cents"] == -12_345

        traj = [Step(index=1, tool="submit", arguments={}, result=json.dumps(verdict))]
        ep = {f.name: f.value for f in score_trajectory(traj, terminated=True).episode}
        assert ep["reward"] == -12_345.0
        assert ep["survived"] is False
        assert ep["success"] is False
    finally:
        mcp_server.end_session(session_id)


async def test_served_premature_terminate_scores_zero() -> None:
    episode = await ServedEpisode.start("yc_bench", task=0)
    try:
        # End the episode without ever calling `submit`.
        result = await episode.call("terminate", {})
        assert result.terminated
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["reward"] == 0.0
        assert feedback["survived"] is False
        assert feedback["success"] is False
    finally:
        await episode.close()


async def test_sim_resume_requires_active_task() -> None:
    # Faithful yc-bench behavior: `sim resume` refuses to advance with no active task.
    episode = await ServedEpisode.start("yc_bench", task=0)
    try:
        r = await episode.call("run_command", {"command": "yc-bench sim resume"})
        payload = json.loads(r.content)
        assert payload["ok"] is False
        assert "active task" in payload["stdout"].lower() or "active task" in payload["stderr"].lower()
    finally:
        await episode.close()


async def test_run_command_rejects_agent_loop_and_interactive() -> None:
    # Security: `run_command` must not admit `yc-bench run` (YC-Bench's own credential-
    # inheriting LLM agent loop) or `yc-bench start` (interactive) — only the operational
    # groups that drive the already-seeded session. Rejection happens before any subprocess.
    episode = await ServedEpisode.start("yc_bench", task=0)
    try:
        for bad in (
            "yc-bench run --model openai/gpt-4o --seed 1",
            "yc-bench start",
            "yc-bench",
            "yc-bench nonsense subcommand",
        ):
            payload = json.loads((await episode.call("run_command", {"command": bad})).content)
            assert payload["ok"] is False, bad
            assert "not permitted" in payload["stderr"].lower(), bad
        # A legitimate operational command still works.
        ok = json.loads(
            (await episode.call("run_command", {"command": "yc-bench company status"})).content
        )
        assert ok["ok"] is True
    finally:
        await episode.close()


async def test_seed_is_deterministic_across_episodes() -> None:
    """Fidelity/reproducibility: the same task (seed) + the same commands yield the same final
    funds — the deterministic sim is preserved by the wrap."""
    funds = []
    for _ in range(2):
        episode = await ServedEpisode.start("yc_bench", task=3)
        try:
            await _play_one_task(episode)
            verdict = json.loads((await episode.call("submit", {})).content)
            funds.append(verdict["final_funds_cents"])
            await episode.call("terminate", {})
        finally:
            await episode.close()
    assert funds[0] == funds[1]


async def test_distinct_seeds_differ() -> None:
    """Different tasks (seeds) generate different market instances — so a fixed policy lands on
    different first-task rewards. Guards against the seed being ignored."""
    rewards = []
    for task in (0, 1):
        episode = await ServedEpisode.start("yc_bench", task=task)
        try:
            first = _stdout((await episode.call("run_command", {"command": "yc-bench market browse --limit 1"})).content)["tasks"][0]
            rewards.append((first["task_id"], first["reward_funds_cents"]))
            await episode.call("terminate", {})
        finally:
            await episode.close()
    assert rewards[0] != rewards[1]
