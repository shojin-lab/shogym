"""The seal-before-verdict terminal lifecycle, exercised end-to-end on the real (offline)
``tau2_mock`` env.

Covers the tau2-specific migration onto the terminal-seal substrate: the manifest marks
``done`` as the single ``score`` terminal (``terminate`` = abort); an explicit ``done`` seals
the episode and ``_verify`` scores from core-owned evidence (never marker JSON); a tau2
autonomous max-step stop is preserved and scored via the evaluator; reaching the hgym horizon
runs the evaluator over the completed run (**preserve_upstream_maxstep**), not a premature
zero; and ``close()`` waits for an in-flight finalize before tearing down (tau2's
``end_session``/``abort`` can't race the finalizer).

Requires the ``tau2`` extra and a loadable tau2 ``mock`` data set — skipped otherwise, so the
offline core suite stays green. Solo mode ⇒ no user-simulator LLM ⇒ fully offline.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("tau2", reason="tau2 extra not installed")

import hgym  # noqa: E402
from hgym.envs.tau2 import mcp_server  # noqa: E402
from hgym.serve import LifecycleState, ServedEpisode  # noqa: E402
from hgym.shared.terminate_mcp import TERMINATE_TOOL_NAME  # noqa: E402


def _mock_task_index(task_id: str) -> int:
    try:
        env = hgym.make("tau2_mock")
    except Exception as exc:  # missing data etc.
        pytest.skip(f"tau2 mock env not constructible offline: {exc}")
    if task_id not in env._task_ids:
        pytest.skip(f"task {task_id} not in mock train split")
    return env._task_ids.index(task_id)


def _feedback(ep: ServedEpisode) -> dict:
    return {i["name"]: i["value"] for i in ep.terminal_feedback}


# ----- manifest gating -----


async def test_manifest_marks_done_score_and_terminate_abort() -> None:
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx)
    try:
        spec = episode.describe()
        by_name = {t.name: t.terminal_kind for t in spec.tools}
        assert by_name["done"] == "score"
        assert by_name[TERMINATE_TOOL_NAME] == "abort"
        assert by_name["create_task"] == "none"
        assert sum(t.terminal_kind == "score" for t in spec.tools) == 1
        assert spec.contract_version == 2
        assert episode.seal_enabled is True
    finally:
        await episode.close()


# ----- explicit done: seal, then verify consumes core-owned evidence -----


async def test_explicit_done_seals_and_verify_consumes_core_evidence() -> None:
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx)
    try:
        await episode.call("create_task", {"user_id": "user_1", "title": "Important Meeting"})
        assert episode._state is LifecycleState.OPEN  # not sealed until the score terminal

        done = await episode.call("done", {})
        assert done.terminated is True
        # The episode sealed and finalized: state is terminal, evidence is core-owned.
        assert episode._state is LifecycleState.CLOSED
        assert episode._evidence is not None
        assert episode._evidence.verdict["reward"] == 1.0
        # Provenance is core-stamped (non-forgeable) — proof _verify scored trusted evidence.
        assert episode._evidence.provenance["core"] == "hgym-serve"
        assert episode._evidence.provenance["sealed_source"] == "explicit_tool"
        fb = _feedback(episode)
        assert fb["reward"] == 1.0 and fb["success"] is True and fb["db_match"] is True
        assert "eval_error" not in fb

        # Post-seal traffic is tombstoned (no inward dispatch, verdict not re-exposed).
        again = await episode.call("create_task", {"user_id": "user_1", "title": "x"})
        assert again.terminated is True
        assert "sealed" in again.content
    finally:
        await episode.close()


# ----- tau2 autonomous max-step stop: verdict preserved, scored via the evaluator -----


async def test_autonomous_maxstep_stop_verdict_is_preserved_through_the_seal() -> None:
    # max_steps=1: tau2 auto-terminates on its own step budget during the create_task action, and
    # the bridge stashes the evaluator verdict on the background thread. `done` then seals and
    # finalize returns the STORED outcome verbatim — the Orchestrator is not double-stopped, the
    # evaluator is not re-run, and the stashed verdict flows into the sealed evidence unchanged
    # (rather than being replaced by an independent premature zero).
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx, env_config={"max_steps": 1})
    try:
        await episode.call("create_task", {"user_id": "user_1", "title": "Important Meeting"})
        # The bridge already stashed a real evaluator verdict on the autonomous stop.
        with mcp_server._lock:
            session = mcp_server._sessions[episode.session_id]
        assert session.terminated is True
        stashed = dict(session.verdict)
        assert stashed[mcp_server.VERDICT_MARKER] is True

        done = await episode.call("done", {})
        assert done.terminated is True
        assert episode._evidence is not None
        assert episode._evidence.provenance["sealed_source"] == "explicit_tool"
        # The stashed autonomous-stop verdict is preserved verbatim into the sealed evidence.
        assert episode._evidence.verdict["reward"] == stashed["reward"]
        assert _feedback(episode)["reward"] == stashed["reward"]
    finally:
        await episode.close()


# ----- hgym horizon: preserve_upstream_maxstep (evaluator over the completed run) -----


async def test_horizon_runs_the_evaluator_finalize_not_a_premature_zero() -> None:
    # Drive past the hgym horizon: the budget-reaching step seals with source="horizon" and runs
    # this env's finalize (tau2's evaluator over the completed run) — NOT an independent premature
    # zero. Proof is deterministic and value-independent: finalize IS invoked with source=horizon
    # (an `abort`/`terminate` would instead have core synthesize no-score evidence WITHOUT calling
    # finalize). max_steps=1 -> horizon=3: step 1 auto-stops, steps 2-3 reach the cap.
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx, env_config={"max_steps": 1})
    seen: dict = {}
    real_finalize = episode._finalize

    async def spy(req):
        seen["source"] = req.source
        return await real_finalize(req)

    episode._finalize = spy  # type: ignore[assignment]
    try:
        r1 = await episode.call("create_task", {"user_id": "user_1", "title": "Important Meeting"})
        assert r1.terminated is False
        r2 = await episode.call("create_task", {"user_id": "user_1", "title": "again"})
        assert r2.terminated is False  # tau2 already stopped; still OPEN (not the score terminal)
        r3 = await episode.call("create_task", {"user_id": "user_1", "title": "horizon"})
        assert r3.terminated is True  # step 3 == horizon -> seal
        # finalize ran with source=horizon: the evaluator scored the completed run.
        assert seen["source"] == "horizon"
        assert episode._evidence is not None
        assert episode._evidence.provenance["sealed_source"] == "horizon"
        assert episode._evidence.verdict[mcp_server.VERDICT_MARKER] is True  # a tau2 verdict, not abort
    finally:
        await episode.close()


# ----- close() participates in the lifecycle: it waits for an in-flight finalize -----


async def test_close_waits_for_in_flight_finalize_and_tears_down_once() -> None:
    idx = _mock_task_index("create_task_1")
    episode = await ServedEpisode.start("tau2_mock", task=idx)
    try:
        await episode.call("create_task", {"user_id": "user_1", "title": "Important Meeting"})

        real_finalize = episode._finalize
        in_flight = asyncio.Event()
        release = asyncio.Event()

        async def blocking(req):
            in_flight.set()
            await release.wait()
            return await real_finalize(req)

        episode._finalize = blocking  # type: ignore[assignment]

        done = asyncio.create_task(episode.call("done", {}))
        await asyncio.wait_for(in_flight.wait(), timeout=2.0)
        assert episode._state is LifecycleState.FINALIZING

        close_task = asyncio.create_task(episode.close())
        await asyncio.sleep(0.05)
        # close() is WAITING for the finalizer, not tearing down (end_session/abort can't race it).
        assert not close_task.done()
        assert episode._teardown_runs == 0

        release.set()
        result = await done
        await close_task

        assert result.terminated is True
        assert episode._teardown_runs == 1  # teardown ran exactly once, after evidence
        assert episode._state is LifecycleState.CLOSED
        assert _feedback(episode)["reward"] == 1.0
        # The tau2 session was torn down (finalize marked it terminated -> end_session no-op abort).
        with mcp_server._lock:
            assert episode.session_id not in mcp_server._sessions
    finally:
        await episode.close()  # idempotent
