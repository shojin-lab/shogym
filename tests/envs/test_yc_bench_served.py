"""End-to-end: drive a served ``yc_bench`` episode through shogym's serve layer and check that
YC-Bench's sim state flows back into episode feedback through the seal-before-verdict path.

Requires the ``yc_bench`` extra and the provisioned upstream source — skipped otherwise (naming
the reason), so the offline core suite stays green. YC-Bench generates its whole world
deterministically from the seed and runs its sim in process, so once the source is cached the
entire path (seed → CLI commands → sealed finalize → verdict) runs offline, with no model calls
or API keys.
"""

from __future__ import annotations

import json

from tests._fixtures.upstream_gate import gate

# Provisions the pinned upstream source (network on a cold cache) and imports yc_bench, so this is
# also the check that the `yc_bench` extra is installed. A missing extra or an unreachable network
# skips; anything else — upstream drift, a broken adapter, a corrupt cache — fails, so a
# regression can never make this module's tests quietly disappear.
gate("shogym.envs.yc_bench.adapter", package="yc_bench", extra="yc_bench")

from shogym.envs.yc_bench import mcp_server  # noqa: E402
from shogym.serve import ServedEpisode  # noqa: E402


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


def _feedback(episode: ServedEpisode) -> dict:
    return {item["name"]: item["value"] for item in episode.terminal_feedback}


async def test_served_episode_reads_state_and_premature_submit_scores_zero() -> None:
    episode = await ServedEpisode.start("yc_bench", task=0)
    try:
        spec = episode.describe()
        kinds = {t.name: t.terminal_kind for t in spec.tools}
        assert {"run_command", "submit", "terminate"} <= set(kinds)
        # `submit` is the env's score terminal; `terminate` is the reserved abort.
        assert kinds["submit"] == "score"
        assert kinds["terminate"] == "abort"
        assert kinds["run_command"] == "none"
        assert "CEO" in spec.instructions

        # A fresh company starts solvent at $200,000.
        status = _stdout((await episode.call("run_command", {"command": "yc-bench company status"})).content)
        assert status["funds_cents"] == 20_000_000

        resume = await _play_one_task(episode)
        assert "balance_delta" in resume  # the sim advanced and reported financials

        # `submit` seals the episode and runs `finalize`, which reads the live sim state (still
        # mid-run: not bankrupt, horizon not reached). Its result is the core-owned, sanitized
        # verdict — no forgeable marker, plus the `finalize_error` flag.
        result = await episode.call("submit", {})
        assert result.terminated
        verdict = json.loads(result.content)
        assert verdict["seeded"] is True
        assert isinstance(verdict["final_funds_cents"], int)
        assert verdict["terminal_reason"] is None  # the run has not actually ended
        assert verdict["finalize_error"] is False

        # Anti-gaming (terminal-state gate): submitting before a terminal state scores a premature
        # zero, not the current (solvent, pre-horizon) funds — even though the sim read them fine.
        feedback = _feedback(episode)
        assert feedback["reward"] == 0.0
        assert feedback["final_funds_cents"] == 0.0
        assert feedback["survived"] is False
        assert feedback["success"] is False
    finally:
        await episode.close()


async def test_served_bankruptcy_finalize_credits_negative_funds() -> None:
    # Drive the served episode's live sim DB to bankruptcy, then `submit`: `finalize` reads the
    # (now bankrupt) live DB during the seal and the scorer credits the negative terminal funds.
    # Exercises the full seal → finalize → verify path AND the SQLite liveness ordering (finalize
    # reads before the serve layer disposes the engine).
    from yc_bench.db.models.company import Company
    from yc_bench.db.session import session_scope

    episode = await ServedEpisode.start("yc_bench", task=5)
    try:
        session = mcp_server._sessions[episode.session_id]
        with session_scope(session._factory) as db:
            db.query(Company).update({Company.funds_cents: -12_345})

        result = await episode.call("submit", {})
        assert result.terminated
        verdict = json.loads(result.content)
        assert verdict["terminal_reason"] == "bankruptcy"
        assert verdict["survived"] is False
        assert verdict["final_funds_cents"] == -12_345

        feedback = _feedback(episode)
        assert feedback["reward"] == -12_345.0
        assert feedback["survived"] is False
        assert feedback["success"] is False
    finally:
        await episode.close()


async def test_served_premature_terminate_scores_zero() -> None:
    episode = await ServedEpisode.start("yc_bench", task=0)
    try:
        # End the episode with the abort terminal (never calling `submit`).
        result = await episode.call("terminate", {})
        assert result.terminated
        feedback = _feedback(episode)
        assert feedback["reward"] == 0.0
        assert feedback["survived"] is False
        assert feedback["success"] is False
    finally:
        await episode.close()


async def test_command_budget_is_max_commands_then_submit() -> None:
    # The command budget is exactly `max_commands` non-terminal `run_command` steps; `submit`
    # (intercepted before the horizon check) is always available after them. A policy that keeps
    # issuing commands instead hits the horizon on its `max_commands + 1`-th call.
    episode = await ServedEpisode.start("yc_bench", task=0, env_config={"max_commands": 2})
    try:
        assert episode.describe().horizon == 3  # max_commands + 1
        r1 = await episode.call("run_command", {"command": "yc-bench company status"})
        r2 = await episode.call("run_command", {"command": "yc-bench company status"})
        assert not r1.terminated and not r2.terminated  # both non-terminal (2 == max_commands)
        result = await episode.call("submit", {})  # still available after the full budget
        assert result.terminated
    finally:
        await episode.close()

    # A policy that never submits: the (max_commands + 1)-th command is the horizon terminal.
    episode = await ServedEpisode.start("yc_bench", task=0, env_config={"max_commands": 2})
    try:
        assert not (await episode.call("run_command", {"command": "yc-bench company status"})).terminated
        assert not (await episode.call("run_command", {"command": "yc-bench company status"})).terminated
        assert (await episode.call("run_command", {"command": "yc-bench company status"})).terminated
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
    funds — the deterministic sim is preserved by the wrap. Read off the sealed `submit`
    verdict."""
    funds = []
    for _ in range(2):
        episode = await ServedEpisode.start("yc_bench", task=3)
        try:
            await _play_one_task(episode)
            verdict = json.loads((await episode.call("submit", {})).content)
            funds.append(verdict["final_funds_cents"])
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
