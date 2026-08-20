"""The terminal-tool seal lifecycle, exercised against a fixture ``score``-terminal env.

These drive ``_fixture_score`` (tests/_fixtures/score_env.py), NOT a real env: the base
substrate migrates no real env, so the lifecycle / durability / ingress-gate behaviour is
proven on a fixture. No-regression for the real envs is proven by the untouched existing
suite (test_serve_episode.py et al.).

Coverage: manifest gating (zero-or-one score terminal), validate-before-seal,
seal-before-evaluate, the request-level ingress-gate tombstone (tool call + a non-tool method
+ unknown tool + no recorded step), close-race, single-finalization cancellation, the
finalize-deadline fail-closed rule, the durable record + simulated-crash restart recovery to
fail-closed (including the directories the store had to create, which are only as durable as
the entries naming them), the rule that a replayed outcome comes from the record's core-owned
status and never from the env-authored verdict beside it, and the terminal TraceEvent schema +
the ``_verify``-consumes-evidence path.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from fastmcp import Client

from shogym.core import Env
from shogym.envs.registration import _ENV_REGISTRY, register
from shogym.serve import (
    FinalizationRecord,
    FinalizationStore,
    LifecycleState,
    ServedEpisode,
    TerminalEvidence,
)
from shogym.serve import lifecycle
from shogym.serve.lifecycle import FinalizeRequest, fail_closed_verdict
from shogym.serve.server import build_server
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.task import TaskSpec, ToolManifest
from shogym.trace import load_traces
from shogym.types import FeedbackCollection

import tests._fixtures.score_env as fixture  # registers `_fixture_score`
from tests._fixtures import score_mcp

_TASKS = [{"id": "q0", "question": "What is 2+2?", "answer": "4"}]


def _config(**extra):
    return {"tasks": _TASKS, **extra}


async def _start(**extra) -> ServedEpisode:
    return await ServedEpisode.start("_fixture_score", task=0, env_config=_config(**extra))


def _feedback(ep: ServedEpisode) -> dict:
    return {i["name"]: i["value"] for i in ep.terminal_feedback}


@pytest.fixture(autouse=True)
def _clean_sessions(tmp_path_factory, monkeypatch):
    # Redirect the no-trace durable fallback root off the real ~/.cache into a tmp dir, so
    # no-trace episodes don't pollute the home cache and the shared root is test-isolated.
    root = tmp_path_factory.mktemp("shogym-sessions")
    monkeypatch.setattr(lifecycle, "_sessions_cache_root", lambda: root)
    score_mcp.reset_state()
    yield
    score_mcp.reset_state()


# ----- manifest gating: exactly zero-or-one `score` terminal -----


def test_manifest_marks_exactly_one_score_terminal_and_abort() -> None:
    env = fixture._FixtureScoreEnv(tasks=_TASKS)
    spec = env.describe("0")
    by_name = {t.name: t.terminal_kind for t in spec.tools}
    assert by_name["submit"] == "score"
    assert by_name[TERMINATE_TOOL_NAME] == "abort"
    assert by_name["noop"] == "none"
    assert sum(t.terminal_kind == "score" for t in spec.tools) == 1
    assert spec.contract_version == 2


def test_taskspec_rejects_two_score_terminals_or_scoring_terminate() -> None:
    # The invariant is enforced in TaskSpec validation, so even a hand-rolled Env.describe()
    # (bypassing the base class's convenience check) cannot publish two score terminals or a
    # scoring `terminate`.
    def _tool(name: str, kind: str) -> ToolManifest:
        return ToolManifest(name=name, description="d", input_schema={}, terminal_kind=kind)

    with pytest.raises(ValueError, match="at most one `score` terminal"):
        TaskSpec(
            env_name="x", instructions="i",
            tools=[_tool("a", "score"), _tool("b", "score")],
        )
    with pytest.raises(ValueError, match="reserved `terminate`"):
        TaskSpec(
            env_name="x", instructions="i",
            tools=[_tool(TERMINATE_TOOL_NAME, "score")],
        )
    # `abort` iff `terminate`: a non-abort terminate, or an abort on any other tool, is rejected
    # (the serve layer only treats the literal `terminate` as the abort path).
    with pytest.raises(ValueError, match="terminate.*terminal_kind='abort'"):
        TaskSpec(
            env_name="x", instructions="i",
            tools=[_tool(TERMINATE_TOOL_NAME, "none")],
        )
    with pytest.raises(ValueError, match="reserved for the `terminate`"):
        TaskSpec(
            env_name="x", instructions="i",
            tools=[_tool("run_command", "abort")],
        )
    # One score + an abort terminate is fine.
    ok = TaskSpec(
        env_name="x", instructions="i",
        tools=[_tool("submit", "score"), _tool(TERMINATE_TOOL_NAME, "abort")],
    )
    assert sum(t.terminal_kind == "score" for t in ok.tools) == 1


def test_score_terminal_must_be_advertised_and_not_terminate() -> None:
    class _Bad(fixture._FixtureScoreEnv):
        score_terminal_tool = "does_not_exist"

    with pytest.raises(ValueError, match="not an advertised tool"):
        _Bad(tasks=_TASKS)

    class _BadTerminate(fixture._FixtureScoreEnv):
        score_terminal_tool = TERMINATE_TOOL_NAME

    with pytest.raises(ValueError, match="reserved"):
        _BadTerminate(tasks=_TASKS)

    # A score terminal without a callable finalize hook must fail fast (else the published v2
    # contract would advertise seal semantics that never engage). Both None and a non-callable
    # value are rejected — the check mirrors the serve layer's `callable(finalize)` gate.
    class _NoFinalize(fixture._FixtureScoreEnv):
        finalize = None  # drop the hook the fixture provides

    with pytest.raises(ValueError, match="finalize"):
        _NoFinalize(tasks=_TASKS)

    class _NonCallableFinalize(fixture._FixtureScoreEnv):
        finalize = False  # type: ignore[assignment]  # non-callable

    with pytest.raises(ValueError, match="finalize"):
        _NonCallableFinalize(tasks=_TASKS)


# ----- validate -> seal ordering -----


async def test_invalid_terminal_call_is_a_validation_error_while_open() -> None:
    ep = await _start()
    try:
        for bad in (
            {"confidence": 90},  # missing `answer`
            {"answer": "4", "confidence": "lots"},  # wrong type
            {"answer": "4", "surprise": True},  # additionalProperties: false
            {"answer": "   ", "confidence": 90},  # blank required string
        ):
            res = await ep.call("submit", bad)
            assert res.terminated is False
            assert json.loads(res.content)["validation_error"] is True
            assert ep._state is LifecycleState.OPEN  # NOT sealed
            assert ep._finalization is None
            assert ep.terminal_feedback == []
            assert ep._env.finalize_calls == 0

        # A now-valid submission seals + grades — the episode was never consumed.
        ok = await ep.call("submit", {"answer": "4", "confidence": 90})
        assert ok.terminated is True
        assert _feedback(ep)["correct"] is True
        assert ep._env.finalize_calls == 1
    finally:
        await ep.close()


async def test_validation_error_through_served_interface_does_not_seal() -> None:
    ep = await _start()
    try:
        server = build_server(ep)
        async with Client(server) as client:
            res = await client.call_tool("submit", {"answer": "4", "confidence": "lots"})
            assert json.loads(res.content[0].text)["validation_error"] is True
            assert ep._state is LifecycleState.OPEN
            assert ep._env.finalize_calls == 0
            ok = await client.call_tool("submit", {"answer": "4", "confidence": 80})
            assert json.loads(ok.content[0].text)["correct"] is True
    finally:
        await ep.close()


# ----- seal before evaluate -----


async def test_finalize_runs_only_after_the_episode_is_sealed() -> None:
    seen: dict = {}
    ep = await _start()
    real_finalize = ep._finalize

    async def probing(req):
        seen["state"] = ep._state
        return await real_finalize(req)

    ep._finalize = probing  # type: ignore[assignment]
    try:
        res = await ep.call("submit", {"answer": "4", "confidence": 60})
        assert res.terminated is True
        # The evaluator observed an already-sealed (FINALIZING) episode — never OPEN.
        assert seen["state"] is LifecycleState.FINALIZING
    finally:
        await ep.close()


# ----- request-level ingress gate (the tombstone) -----


async def test_ingress_gate_tombstones_every_post_seal_call_but_allows_readonly() -> None:
    ep = await _start()
    try:
        server = build_server(ep)
        async with Client(server) as client:
            first = await client.call_tool("submit", {"answer": "4", "confidence": 50})
            assert json.loads(first.content[0].text)["correct"] is True

            # Tombstoned tools/call: repeat terminal, the reserved terminate, an ordinary
            # tool, AND an unknown tool (which never reaches the Python dispatcher).
            for name, args in [
                ("submit", {"answer": "4"}),
                (TERMINATE_TOOL_NAME, {}),
                ("noop", {}),
                ("totally_unknown_tool", {}),
            ]:
                r = await client.call_tool(name, args)
                assert "sealed" in r.content[0].text

            # Read-only methods stay allowed: describe (a tool), tools/list, resource read.
            desc = await client.call_tool("describe", {})
            assert json.loads(desc.content[0].text)["contract_version"] == 2
            assert len(await client.list_tools()) == 4
            res = await client.read_resource("shogym://task")
            assert json.loads(res[0].text)["env_name"] == "_fixture_score"
    finally:
        await ep.close()


async def test_post_seal_tombstoned_call_records_no_trajectory_step() -> None:
    ep = await _start()
    try:
        await ep.call("submit", {"answer": "4"})  # seals + finalizes
        recorded = len(ep._trajectory)
        await ep.call("submit", {"answer": "5"})
        await ep.call(TERMINATE_TOOL_NAME, {})
        await ep.call("noop", {})
        assert len(ep._trajectory) == recorded  # nothing dispatched or recorded post-seal
    finally:
        await ep.close()


# ----- close() participates in the lifecycle -----


async def test_close_race_waits_for_finalizer_and_tears_down_once() -> None:
    ep = await _start()
    real_finalize = ep._finalize
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def blocking(req):
        in_flight.set()
        await release.wait()
        return await real_finalize(req)

    ep._finalize = blocking  # type: ignore[assignment]

    submit = asyncio.create_task(ep.call("submit", {"answer": "4", "confidence": 50}))
    await asyncio.wait_for(in_flight.wait(), timeout=1.0)
    assert ep._state is LifecycleState.FINALIZING

    close_task = asyncio.create_task(ep.close())
    await asyncio.sleep(0.02)  # let close() reach its await on the finalization
    assert not close_task.done()  # close is WAITING for the finalizer, not tearing down
    assert ep._teardown_runs == 0

    release.set()
    result = await submit
    await close_task

    assert result.terminated is True
    assert ep._env.finalize_calls == 1  # exactly one evaluation
    assert ep._teardown_runs == 1  # teardown ran once, after evidence
    assert ep._state is LifecycleState.CLOSED
    assert _feedback(ep)["correct"] is True


async def test_close_before_submit_claims_abort_and_scores_no_credit() -> None:
    ep = await _start()
    await ep.close()  # never submitted
    assert ep._state is LifecycleState.CLOSED
    assert ep._teardown_runs == 1
    # An abort is a no-score path: correct=False, and the evaluator was NOT invoked.
    assert _feedback(ep)["correct"] is False
    assert ep._env.finalize_calls == 0


# ----- post-seal cancellation (distinct from the close race) -----


async def test_post_seal_cancellation_awaits_the_single_finalization() -> None:
    ep = await _start()
    real_finalize = ep._finalize
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def blocking(req):
        in_flight.set()
        await release.wait()
        return await real_finalize(req)

    ep._finalize = blocking  # type: ignore[assignment]

    submit = asyncio.create_task(ep.call("submit", {"answer": "4", "confidence": 50}))
    await asyncio.wait_for(in_flight.wait(), timeout=1.0)
    finalization = ep._finalization
    assert finalization is not None and not finalization.done()

    submit.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submit

    assert not finalization.done()  # survived the cancellation (shielded); no re-dispatch
    assert ep._teardown_runs == 0

    release.set()
    result = await finalization  # the retained finalization runs to completion
    assert result.terminated is True
    assert ep._env.finalize_calls == 1  # exactly one evaluation, never a second
    assert _feedback(ep)["correct"] is True
    assert ep._state is LifecycleState.CLOSED
    await ep.close()  # idempotent


async def test_finalize_deadline_fails_closed_and_drains_evaluator_before_teardown() -> None:
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, env_config=_config(), finalize_deadline=0.05
    )
    real_finalize = ep._finalize
    release = asyncio.Event()  # only this test can let the evaluator finish
    evaluator_done = {"flag": False, "at_end_session": None}

    async def slow(req):
        await release.wait()  # outlives the 0.05s deadline by construction, not by clock
        ev = await real_finalize(req)
        evaluator_done["flag"] = True
        return ev

    real_end_session = ep._env.end_session

    def observing_end_session(session_id):
        # end_session drops per-session state — the evaluator must have drained by now.
        evaluator_done["at_end_session"] = evaluator_done["flag"]
        return real_end_session(session_id)

    ep._finalize = slow  # type: ignore[assignment]
    ep._env.end_session = observing_end_session  # type: ignore[method-assign]
    try:
        # The DEADLINE bounds caller latency: the fail-closed result returns while the evaluator
        # is STILL RUNNING, rather than teardown blocking on it. Asserted as ordering (the
        # evaluator provably cannot have finished, since only this test can release it) instead
        # of as a stopwatch: a wall-clock margin measures the runner, not the deadline, and a
        # loaded one broke a 0.15s bound on a test whose every behavioural assertion passed.
        # `wait_for` is only a hang detector: a deadline that never fires would wait on the
        # evaluator forever, and a bound this generous turns that into a failure here rather
        # than a job-level CI timeout.
        result = await asyncio.wait_for(
            ep.call("submit", {"answer": "4", "confidence": 50}), timeout=10.0
        )
        assert evaluator_done["flag"] is False  # returned BEFORE the evaluator finished
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False
        assert payload["finalize_error"] is True  # fail-closed, flagged
        assert _feedback(ep)["finalize_error"] is True
        # ...yet env state is not dropped until the evaluator has drained (no use-after-free):
        # the drain+teardown runs in the background; close() waits for it.
        assert ep._env.end_session is observing_end_session
        release.set()
        await ep.close()
        assert evaluator_done["at_end_session"] is True
        assert ep._env.finalize_calls == 1  # exactly one evaluation, never a second
    finally:
        release.set()  # never leave close() waiting on an evaluator this test still holds
        await ep.close()


async def test_evaluator_cancellederror_fails_closed_not_strands_the_episode() -> None:
    # An evaluator that raises CancelledError (its own downstream work was cancelled) must NOT
    # escape the finalization task: it fails closed, the caller gets a terminal result (not a
    # CancelledError), and the episode reaches CLOSED with teardown run — never stranded
    # FINALIZING with a PENDING record. (Distinct from an awaiter's cancellation, which the
    # shield handles at the call/close boundary.)
    ep = await _start()

    async def cancels(req):
        raise asyncio.CancelledError()

    ep._finalize = cancels  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False
        assert payload["finalize_error"] is True  # fail-closed
        assert ep._state is LifecycleState.CLOSED
        assert ep._torn_down is True
    finally:
        await ep.close()  # must reach teardown, not propagate cancellation


async def test_non_serializable_verdict_fails_closed_and_tears_down(tmp_path: Path) -> None:
    # A finalizer that returns a non-JSON verdict (NaN) must not strand a FINALIZED episode: the
    # commit/trace writes (allow_nan=False) would otherwise raise after the seal. Fail closed to
    # the canonical safe verdict, complete the transaction, and tear down.
    from shogym.serve.lifecycle import TerminalEvidence

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def nan_verdict(req):
        return TerminalEvidence(
            source=req.source, status="ok", verdict={"correct": True, "score": float("nan")}
        )

    ep._finalize = nan_verdict  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed, not the NaN verdict
        assert payload["finalize_error"] is True
        assert ep._state is LifecycleState.CLOSED  # teardown ran
        assert ep._torn_down is True
    finally:
        await ep.close()
    # The trace terminal event holds the safe fail-closed verdict (no NaN reached the store).
    event = [r for r in load_traces(trace) if r.get("kind") == "terminal"][0]
    assert event["verdict"]["correct"] is False


async def test_non_dict_verdict_fails_closed_and_records_finalized_not_pending(
    tmp_path: Path,
) -> None:
    # A finalizer that returns a NON-dict verdict (e.g. a list) is JSON-serializable yet would
    # raise in `_sanitize_terminal`'s `dict(evidence.verdict)` — AFTER the seal, with the durable
    # record still PENDING. That would (a) surface an exception to the client instead of the
    # documented fail-closed result, and (b) strand the durable record at PENDING. The pre-commit
    # guard must reject a non-dict verdict and fail closed so neither happens.
    from shogym.serve.lifecycle import TerminalEvidence

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def list_verdict(req):
        # A misbehaving env: a non-dict (list) verdict. json.dumps([...]) succeeds, so the old
        # `_json_safe`-only guard let it through to `dict(evidence.verdict)`, which raises.
        return TerminalEvidence(
            source=req.source, status="ok", verdict=["correct", True]  # type: ignore[arg-type]
        )

    ep._finalize = list_verdict  # type: ignore[assignment]
    try:
        # (a) The terminal call RETURNS the fail-closed result — it must NOT raise to the client.
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed, not the list verdict
        assert payload["finalize_error"] is True
        # The episode is scored fail-closed (verify consumed a finalize_error verdict).
        assert ep._evidence is not None
        assert ep._evidence.status == "finalize_error"
        assert ep._evidence.verdict == fail_closed_verdict(50)
        assert _feedback(ep)["finalize_error"] is True
        assert ep._state is LifecycleState.CLOSED  # teardown ran
    finally:
        await ep.close()

    # (b) The durable finalization record is RESOLVED fail-closed (FAILED) — never left PENDING.
    store = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace))
    recs = store.load_all()
    assert len(recs) == 1
    rec = recs[0]
    assert rec.status != "PENDING"  # not stranded mid-finalize
    assert rec.status == "FAILED"  # fail-closed terminal resolution
    assert rec.verdict["finalize_error"] is True
    assert rec.verdict["correct"] is False
    assert rec.to_evidence().finalize_error is True


async def test_durable_store_write_failure_still_yields_a_verdict() -> None:
    # A local-file persistence failure (ENOSPC / permissions) must never strand a sealed
    # episode: the seal, finalize, and in-memory verdict proceed; the episode still terminates.
    ep = await _start()
    assert ep._store is not None

    def boom(_record):
        raise OSError("disk full")

    ep._store.write = boom  # type: ignore[method-assign]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        assert json.loads(result.content)["correct"] is True  # verdict produced anyway
        assert ep._state is LifecycleState.CLOSED
        assert ep._persist_degraded is True  # flagged for audit, never raised
        assert _feedback(ep)["correct"] is True
    finally:
        await ep.close()


# ----- terminal TraceEvent schema + _verify-consumes-evidence -----


async def test_trace_records_step_then_versioned_terminal_event(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    try:
        await ep.call("noop", {})  # an ordinary mid-episode step
        await ep.call("submit", {"answer": "4", "confidence": 75})
    finally:
        await ep.close()

    rows = load_traces(trace)
    kinds = [r.get("kind", "step") for r in rows]
    # A step row for noop, a terminal step row, then the versioned terminal event last.
    assert kinds == ["step", "step", "terminal"]
    event = rows[-1]
    assert event["schema_version"] == 1
    assert event["source"] == "explicit_tool"
    assert event["status"] == "ok"
    assert event["verdict"] == {"correct": True, "confidence": 75}
    assert event["args_digest"].startswith("sha256:")
    assert isinstance(event["finalization_id"], str)
    # The public event never carries the private diagnostic / provenance.
    assert "diagnostic" not in event and "provenance" not in event


async def test_horizon_completion_is_the_budget_step_not_a_phantom_extra(
    tmp_path: Path,
) -> None:
    # A score env with horizon=3, reached by three ordinary calls, must produce EXACTLY three
    # trajectory/trace steps (the third IS the terminal step) — never a fabricated `<horizon>`
    # step or a tool invocation that didn't happen. Mirrors legacy horizon termination.
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    try:
        r1 = await ep.call("noop", {})
        assert r1.terminated is False
        r2 = await ep.call("noop", {})
        assert r2.terminated is False
        r3 = await ep.call("noop", {})  # step 3 == horizon
        assert r3.terminated is True
        # Exactly horizon-many trajectory steps; no phantom terminal step appended.
        assert len(ep._trajectory) == 3
        assert [s.tool for s in ep._trajectory] == ["noop", "noop", "noop"]
        # No submission reached the seal -> scored incorrect via the horizon evidence.
        assert _feedback(ep)["correct"] is False
    finally:
        await ep.close()

    rows = load_traces(trace)
    kinds = [r.get("kind", "step") for r in rows]
    assert kinds == ["step", "step", "step", "terminal"]  # 3 steps + the terminal event
    assert [r["tool"] for r in rows if r.get("kind", "step") == "step"] == [
        "noop", "noop", "noop",
    ]
    step_rows = [r for r in rows if r.get("kind", "step") == "step"]
    assert step_rows[-1]["terminated"] is True  # the budget step is the terminal step
    event = rows[-1]
    assert event["source"] == "horizon"
    assert event["step"] == 3
    assert event["args_digest"] is None  # no submission


async def test_verify_sees_the_terminal_step_in_the_trajectory() -> None:
    # A migrated verifier scores from evidence + the COMPLETE call history; the terminal
    # submit/terminate step must be present in the trajectory verify() receives.
    ep = await _start()
    seen: dict = {}
    real_verify = ep._env.verify

    def capturing(trajectory, task, *, terminated, evidence=None):
        seen["len"] = len(trajectory)
        seen["last_tool"] = trajectory[-1].tool if trajectory else None
        return real_verify(trajectory, task, terminated=terminated, evidence=evidence)

    ep._env.verify = capturing  # type: ignore[method-assign]
    try:
        await ep.call("submit", {"answer": "4", "confidence": 50})
        assert seen["last_tool"] == "submit"  # the terminal step is visible to verify
        assert seen["len"] == 1
    finally:
        await ep.close()


async def test_recovery_runs_on_the_in_process_start_path(tmp_path: Path) -> None:
    # Restart recovery is transport-independent: a dangling record next to a trace is resolved
    # by the next ServedEpisode.start (the evaluate()/in-process path), not only by `shogym serve`.
    trace = tmp_path / "run.jsonl"
    store = FinalizationStore(FinalizationStore.resolve_dir("prior", trace))
    store.write(
        FinalizationRecord(
            session_id="prior", finalization_id="f-crash", status="PENDING",
            source="explicit_tool",
        )
    )
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    try:
        rec = store.read("f-crash")
        assert rec.status == "FAILED"  # resolved fail-closed at start(), no `shogym serve` needed
        assert rec.verdict["finalize_error"] is True
    finally:
        await ep.close()


async def test_recovery_does_not_clobber_a_live_concurrent_episode(tmp_path: Path) -> None:
    # The shared store is scanned by every episode's startup recovery. A record owned by a LIVE
    # process (a concurrent worker mid-finalize) must never be resolved — only a dead owner's
    # (crashed) record is. Repro: A blocks in finalize (PENDING, owner=this live pid); B starts
    # on the same trace dir and must leave A's record untouched.
    trace = tmp_path / "run.jsonl"
    epA = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    real = epA._finalize
    in_flight = asyncio.Event()
    release = asyncio.Event()

    async def blocking(req):
        in_flight.set()
        await release.wait()
        return await real(req)

    epA._finalize = blocking  # type: ignore[assignment]
    submit = asyncio.create_task(epA.call("submit", {"answer": "4", "confidence": 50}))
    await asyncio.wait_for(in_flight.wait(), timeout=1.0)

    store = epA._store
    fid = epA._finalization_id
    assert store is not None and fid is not None
    assert store.read(fid).status == "PENDING"
    assert store.read(fid).owner_pid == os.getpid()  # live owner

    # B starts on the SAME trace dir; its constructor runs recovery — which must SKIP A's record.
    epB = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    assert store.read(fid).status == "PENDING"  # NOT clobbered by B's recovery

    release.set()
    result = await submit
    assert result.terminated is True
    assert store.read(fid).status == "FINALIZED"  # A finished normally
    await epA.close()
    await epB.close()


async def test_verify_scores_from_core_evidence_not_agent_payload(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    # A wrong answer: verify's `correct=False` must come from the core evidence verdict.
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    try:
        result = await ep.call("submit", {"answer": "5", "confidence": 100})
        assert result.terminated is True
        assert json.loads(result.content)["correct"] is False
        assert ep._evidence is not None and ep._evidence.verdict["correct"] is False
        # Provenance is core-stamped (non-forgeable); the diagnostic is private.
        assert ep._evidence.provenance["core"] == "shogym-serve"
        assert _feedback(ep)["correct"] is False
    finally:
        await ep.close()


# ----- durable record + simulated-crash restart recovery -----


async def test_durable_record_written_on_each_transition(tmp_path: Path) -> None:
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    try:
        await ep.call("submit", {"answer": "4", "confidence": 60})
    finally:
        await ep.close()
    store = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace))
    recs = store.load_all()
    assert len(recs) == 1
    rec = recs[0]
    assert rec.status == "FINALIZED"
    assert rec.source == "explicit_tool"
    assert rec.verdict == {"correct": True, "confidence": 60}
    assert rec.provenance["core"] == "shogym-serve"  # confidential, lives only here
    assert rec.diagnostic  # private diagnostic persisted off-trace


def test_restart_recovery_resolves_dangling_records_fail_closed(tmp_path: Path) -> None:
    # Simulate a crash mid-finalize: records left SEALED / PENDING on disk. The evaluator is
    # represented by a counter that recovery must NEVER touch.
    evaluator_calls = {"n": 0}
    store = FinalizationStore(tmp_path / "finalizations")
    store.write(
        FinalizationRecord(
            session_id="s1", finalization_id="f-sealed", status="SEALED",
            source="explicit_tool", args_digest="sha256:abc",
            verdict={"confidence": 42},
        )
    )
    store.write(
        FinalizationRecord(
            session_id="s1", finalization_id="f-pending", status="PENDING",
            source="horizon",
        )
    )
    # A clean prior FINALIZED record must be left alone (it replays its stored evidence).
    store.write(
        FinalizationRecord(
            session_id="s1", finalization_id="f-done", status="FINALIZED",
            source="explicit_tool", verdict={"correct": True},
        )
    )

    resolved = store.recover()
    assert evaluator_calls["n"] == 0  # evaluator NEVER re-invoked
    assert {r.finalization_id for r in resolved} == {"f-sealed", "f-pending"}
    for r in resolved:
        assert r.status == "FAILED"
        assert r.verdict["finalize_error"] is True
        assert r.verdict["correct"] is False

    # Confidence carried through the fail-closed resolution when known.
    sealed = store.read("f-sealed")
    assert sealed.status == "FAILED"
    assert sealed.verdict == fail_closed_verdict(42)
    assert sealed.to_evidence().finalize_error is True

    # The clean record is untouched and replays ok.
    done = store.read("f-done")
    assert done.status == "FINALIZED"
    assert done.to_evidence().status == "ok"

    # Recovery is idempotent: a second pass finds nothing new to resolve.
    assert store.recover() == []


def test_finalization_store_defaults_next_to_trace_and_falls_back(tmp_path: Path) -> None:
    trace = tmp_path / "sub" / "run.jsonl"
    d = FinalizationStore.resolve_dir("sid-1", trace)
    assert d == trace.parent / "finalizations"
    # No trace path -> a SHARED (not session-keyed) fallback root, so a later process can scan
    # it for a crashed prior session's records. Two different sessions resolve to the same root.
    d2 = FinalizationStore.resolve_dir("sid-2", None)
    d3 = FinalizationStore.resolve_dir("sid-3", None)
    assert d2 == d3


def test_no_trace_recovery_reaches_a_prior_sessions_dangling_record() -> None:
    # The zero-config recovery path: a crash leaves a dangling record under the shared
    # no-trace root; a *later* process (a fresh session id) recovers it, because the root is
    # shared rather than keyed by the crashed session's id.
    crashed_dir = FinalizationStore.resolve_dir("prior-crashed-session", None)
    store = FinalizationStore(crashed_dir)
    store.write(
        FinalizationRecord(
            session_id="prior-crashed-session", finalization_id="f-crash",
            status="PENDING", source="explicit_tool",
        )
    )
    # A later process resolves its own (shared) root and recovers the dangling record.
    later = FinalizationStore(FinalizationStore.resolve_dir("new-session", None))
    resolved = later.recover()
    assert [r.finalization_id for r in resolved] == ["f-crash"]
    assert later.read("f-crash").status == "FAILED"
    assert later.read("f-crash").verdict["finalize_error"] is True


# ----- a directory the store created is durable too -----


def _fsync_spy(monkeypatch: pytest.MonkeyPatch, watched: dict) -> list:
    """Record, by name, which of ``watched``'s paths each ``os.fsync`` was issued against.

    A crash cannot be staged in a test, so durability is shown structurally: the sync either
    happened or it did not. An fd is attributed to a path by ``(st_dev, st_ino)`` taken on BOTH
    sides at the moment of the sync. Inode numbers are recycled, so a key captured during the
    run and compared against a ``stat()`` taken after it can credit a deleted temp file's sync
    to a directory that later inherited its inode — and this store creates and renames away a
    temp file on every single write, so that is not hypothetical here. Read at the same instant,
    a match means the fd and the path name the same live object: two live objects on one device
    cannot share an inode number.
    """
    real_fsync = os.fsync
    synced = []

    def _watch(fd: int) -> None:
        info = os.fstat(fd)
        for name, path in watched.items():
            try:
                here = path.stat()
            except OSError:
                continue  # not created yet, or already gone — either way, not this fd
            if (here.st_dev, here.st_ino) == (info.st_dev, info.st_ino):
                synced.append(name)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _watch)
    return synced


def test_a_directory_the_store_created_is_synced_into_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Syncing the finalization directory persists the records *inside* it. The entry that names
    # the directory itself lives one level up, so the first episode against a fresh trace path
    # can lose the entire store — records, files and directory — with every write having
    # returned successfully. Every level the write had to create must be synced into the level
    # above it, which is the one case that matters most: the store's directory is routinely new.
    run_dir = tmp_path / "runs" / "run-1"  # two levels that do not exist yet...
    store_dir = FinalizationStore.resolve_dir("s1", run_dir / "run.jsonl")  # ...and a third
    watched = {
        "tmp_path": tmp_path,
        "runs": tmp_path / "runs",
        "run-1": run_dir,
        "finalizations": store_dir,
    }
    synced = _fsync_spy(monkeypatch, watched)

    FinalizationStore(store_dir).write(
        FinalizationRecord(
            session_id="s1", finalization_id="f1", status="FINALIZED",
            source="explicit_tool", verdict={"correct": True},
        )
    )

    assert (store_dir / "finalization-f1.json").exists()
    assert "finalizations" in synced, "the record's own directory was never synced"
    for created, holder in (("finalizations", "run-1"), ("run-1", "runs"), ("runs", "tmp_path")):
        assert holder in synced, (
            f"{created}/ was created by this write, so the entry naming it — which lives in "
            f"{holder}/, not in {created}/ — is what a crash can still take away"
        )


def test_the_shared_fallback_root_is_durable_the_first_time_it_is_used(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The no-trace root is created by `resolve_dir`, EAGERLY — so by the time a record is
    # written the directory is already there and `write` has nothing above it left to make
    # durable. First use is exactly the case the zero-config recovery contract rests on: the
    # root a later process scans for a crashed run's records is the one that crashed run created.
    monkeypatch.undo()  # the autouse fixture redirects this root; here the real one is the point
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    watched = {
        "tmp_path": tmp_path,
        "home": home,
        ".cache": home / ".cache",
        "shogym": home / ".cache" / "shogym",
        "sessions": home / ".cache" / "shogym" / "sessions",
    }
    synced = _fsync_spy(monkeypatch, watched)

    assert FinalizationStore.resolve_dir("sid-1", None) == watched["sessions"]

    for created, holder in (
        ("sessions", "shogym"), ("shogym", ".cache"), (".cache", "home"), ("home", "tmp_path")
    ):
        assert holder in synced, (
            f"{created}/ was created here, so the entry naming it — which lives in {holder}/ — "
            "is what a crash can take away, and with it every record recovery would have found"
        )


def test_a_directory_another_writer_created_is_published_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `mkdir` makes a level visible immediately and durable never, and this store is shared
    # between processes on purpose — recovery is written around a live concurrent worker. So one
    # writer can create the whole chain and then stall or die before it syncs anything, leaving
    # the next writer a path on which every level exists and none is on disk. A writer that
    # published only what it created itself would find nothing to do, return success, and lose
    # the entire store to the next crash. An existing level is not evidence that anyone synced
    # it, so what gets published is the path, not this call's share of it.
    store_dir = tmp_path / "runs" / "run-1" / "finalizations"
    store_dir.mkdir(parents=True, exist_ok=True)  # the other writer, this far and no further
    watched = {
        "tmp_path": tmp_path,
        "runs": tmp_path / "runs",
        "run-1": tmp_path / "runs" / "run-1",
        "finalizations": store_dir,
    }
    synced = _fsync_spy(monkeypatch, watched)

    FinalizationStore(store_dir).write(
        FinalizationRecord(
            session_id="s1", finalization_id="f1", status="FINALIZED",
            source="explicit_tool", verdict={"correct": True},
        )
    )

    assert (store_dir / "finalization-f1.json").exists()
    for existed, holder in (("finalizations", "run-1"), ("run-1", "runs"), ("runs", "tmp_path")):
        assert holder in synced, (
            f"{existed}/ was already there when this write ran, which says nothing about whether "
            f"the entry naming it in {holder}/ ever reached the disk"
        )


# ----- a replayed outcome comes from the record's status, never from the env's verdict -----

# The verdict names the core gives a meaning of its own along the terminal path — `finalize_error`
# is the flag the live payload carries, stamped in `_sanitize_terminal` from the core's own
# status. A *persisted* verdict is whatever the env's `finalize` returned, verbatim, so any of
# these can arrive holding anything an episode feedback value admits. None may steer a replay.
_CORE_OWNED_VERDICT_NAMES = ("finalize_error",)

# Everything an `EpisodeFeedbackValue` legally permits, on both sides of the intent. `True` is in
# here deliberately: it is the value a strict `is True` read would still have honoured, and
# honouring it is the same defect as coercing `"false"` — the env's word outranking the core's.
_ANY_ENV_VALUE = [True, False, "false", "true", "no", "0", 0, 1, 2.5, ""]


@pytest.mark.parametrize("name", _CORE_OWNED_VERDICT_NAMES)
@pytest.mark.parametrize("value", _ANY_ENV_VALUE)
def test_a_clean_record_replays_clean_whatever_its_verdict_says(
    tmp_path: Path, name: str, value: object
) -> None:
    # A FINALIZED record is the core's own statement that the finalization succeeded. Replay must
    # not go looking for a second opinion in the env-authored verdict: read by truthiness,
    # `"false"` becomes a failure (`bool("false")` is `True`); read strictly, a literal `True`
    # still does. Either way a clean episode comes back as an infrastructure failure with its
    # real outcome discarded — and contradicting the answer the agent was already given.
    store = FinalizationStore(tmp_path / "finalizations")
    store.write(
        FinalizationRecord(
            session_id="s1", finalization_id="f-clean", status="FINALIZED",
            source="explicit_tool", verdict={"correct": True, name: value},
        )
    )

    evidence = store.read("f-clean").to_evidence()
    assert evidence.status == "ok", f"the record says FINALIZED; {name}={value!r} is the env's word"
    assert evidence.finalize_error is False
    # The verdict is still evidence of what the env returned — only its authority is withheld.
    assert evidence.verdict == {"correct": True, name: value}


@pytest.mark.parametrize("name", _CORE_OWNED_VERDICT_NAMES)
@pytest.mark.parametrize("value", _ANY_ENV_VALUE)
def test_a_failed_record_replays_failed_whatever_its_verdict_says(
    tmp_path: Path, name: str, value: object
) -> None:
    # The same rule in the direction that costs something: a fail-closed record must stay
    # fail-closed. An env that publishes `finalize_error: False` next to a finalization the core
    # already failed cannot talk it back open, which is what makes the fail-closed contract worth
    # anything — the reason to read only the core's status rather than to read the verdict more
    # carefully.
    store = FinalizationStore(tmp_path / "finalizations")
    store.write(
        FinalizationRecord(
            session_id="s1", finalization_id="f-failed", status="FAILED",
            source="explicit_tool", verdict={"correct": False, name: value},
        )
    )

    evidence = store.read("f-failed").to_evidence()
    assert evidence.status == "finalize_error", f"the record says FAILED; {name}={value!r} is not"
    assert evidence.finalize_error is True


def test_an_empty_verdict_is_a_verdict_and_a_missing_one_is_not(tmp_path: Path) -> None:
    # `{}` is a verdict an env may legally return: the pre-commit guard asks whether a verdict is
    # a JSON-safe dict, never whether it says anything. A record that never reached a verdict is
    # a different thing, and only *that* gets the synthetic fail-closed stand-in. Told apart by
    # truthiness the two collapse together, and a clean replay comes back carrying a
    # `correct=False` the env never returned — for a verifier scoring off the reconstructed
    # evidence, an invented zero.
    store = FinalizationStore(tmp_path / "finalizations")
    store.write(
        FinalizationRecord(
            session_id="s1", finalization_id="f-empty", status="FINALIZED",
            source="explicit_tool", verdict={},
        )
    )
    store.write(
        FinalizationRecord(
            session_id="s1", finalization_id="f-none", status="SEALED", source="explicit_tool",
        )
    )

    empty = store.read("f-empty").to_evidence()
    assert empty.status == "ok"
    assert empty.verdict == {}, "an empty verdict is what the env returned, reconstructed"

    # A record that never reached a verdict still resolves fail-closed, stand-in and all.
    missing = store.read("f-none").to_evidence()
    assert missing.status == "finalize_error"
    assert missing.verdict == fail_closed_verdict()


async def test_a_replay_reports_the_outcome_the_live_path_published(tmp_path: Path) -> None:
    # The whole point of the durable record: what a crashed run replays is what the run itself
    # answered. An env is free to publish a key the core also owns — nothing about that is
    # malformed — and the live path already ignores it, stamping the public `finalize_error`
    # from the core's own status. So an episode answered as clean, whose verdict happens to carry
    # `finalize_error: True`, must not become a failure the moment it is read back off disk.
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def clean_but_flagged(req):
        return TerminalEvidence(
            source=req.source, status="ok", verdict={"correct": True, "finalize_error": True}
        )

    ep._finalize = clean_but_flagged  # type: ignore[assignment]
    try:
        payload = json.loads((await ep.call("submit", {"answer": "4"})).content)
    finally:
        await ep.close()

    assert payload["finalize_error"] is False  # live: the core stamped its own flag over the env's
    assert ep._evidence is not None and ep._evidence.status == "ok"

    rec = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace)).load_all()[0]
    assert rec.status == "FINALIZED"
    assert rec.verdict["finalize_error"] is True  # the env's value IS what got persisted
    assert rec.to_evidence().status == ep._evidence.status, (
        "the replayed outcome contradicts the one the episode published"
    )
    assert rec.to_evidence().finalize_error is False


# ----- serve-boundary enforcement for an env that hand-builds its manifest -----
#
# `Env` rejects a declared `score_terminal_tool` that lacks a callable finalize at
# CONSTRUCTION, but an env that overrides `describe()` and hand-builds its TaskSpec/manifest
# declares no `score_terminal_tool`, so it never runs that check. The authoritative enforcement
# therefore lives at the serve boundary every env passes through (`ServedEpisode.__init__`),
# and these tests prove it there — with an env whose `score` terminal exists only in the
# manifest it publishes by hand.


def _raw_score_tools() -> list:
    """A hand-built manifest with exactly one `score` terminal + the reserved abort — the
    contract an env that writes its own ``describe()`` publishes directly."""
    return [
        ToolManifest(
            name="submit",
            description="submit the final answer",
            input_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            terminal_kind="score",
        ),
        ToolManifest(
            name=TERMINATE_TOOL_NAME, description="abort", input_schema={},
            terminal_kind="abort",
        ),
    ]


class _RawScoreEnvNoFinalize(Env):
    """An env that publishes a `score`-terminal manifest from a hand-written ``describe()`` but
    provides NO callable finalize — the exact configuration the construction check would catch
    if the env declared ``score_terminal_tool``, which it never does. ``essential_specs`` is
    empty so ``start()`` reaches the constructor guard without opening any MCP session."""

    def __init__(self) -> None:
        super().__init__(horizon=3)

    def describe(self, task_id=None) -> TaskSpec:
        return TaskSpec(env_name="_raw_score", instructions="i", tools=_raw_score_tools())

    def essential_specs(self):
        return []

    def _load_task(self, task_idx):
        return {"task_idx": 0}

    def _verify(self, trajectory, task, *, terminated) -> FeedbackCollection:
        return FeedbackCollection()


class _RawScoreEnvWithFinalize(_RawScoreEnvNoFinalize):
    """The valid counterpart: same hand-built score manifest, WITH a callable finalize."""

    async def finalize(self, req: FinalizeRequest) -> TerminalEvidence:
        return TerminalEvidence(source=req.source, status="ok", verdict={"correct": True})


if "_raw_score_no_finalize" not in _ENV_REGISTRY:
    register("_raw_score_no_finalize")(_RawScoreEnvNoFinalize)
if "_raw_score_with_finalize" not in _ENV_REGISTRY:
    register("_raw_score_with_finalize")(_RawScoreEnvWithFinalize)


async def test_score_terminal_without_callable_finalize_rejected_at_serve_boundary() -> None:
    # An env whose hand-built manifest advertises a `score` terminal but has no callable
    # finalize must be REJECTED loudly at the serve boundary — never silently downgraded to _seal_enabled=False
    # and routed through the legacy marker path (which would reopen the grade->read->fix->grade
    # exploit for an env that expected the seal to protect it).
    with pytest.raises(TypeError, match=r"score.*finalize"):
        await ServedEpisode.start("_raw_score_no_finalize", task=0)


async def test_valid_hand_built_score_env_still_seals() -> None:
    # The positive: the SAME hand-built score manifest WITH a callable finalize is accepted and
    # seals normally. The guard rejects only the misconfigured env; no valid env is broken.
    ep = await ServedEpisode.start("_raw_score_with_finalize", task=0)
    try:
        assert ep.seal_enabled is True
    finally:
        await ep.close()


# ----- what a fail-closed terminal transaction is allowed to say about itself -----


def test_failure_summary_names_the_type() -> None:
    assert lifecycle.failure_summary(RuntimeError("evaluator exploded")) == {
        "error": "RuntimeError"
    }


def test_failure_summary_keeps_locations_and_drops_the_message() -> None:
    # A structured failure contributes the field paths it objected to, and nothing else: the
    # values behind those paths are the env's state, which is the thing being graded.
    from pydantic import BaseModel, ValidationError

    class _Verdict(BaseModel):
        depth: int
        label: str

    with pytest.raises(ValidationError) as caught:
        _Verdict(depth="not-an-int", label=None)  # type: ignore[arg-type]
    summary = lifecycle.failure_summary(caught.value)
    assert summary["error"] == "ValidationError"
    assert summary["locations"] == ["depth", "label"]
    assert summary["location_count"] == 2
    assert "not-an-int" not in json.dumps(summary)


def test_failure_summary_truncates_but_still_counts() -> None:
    # One structured failure can carry a location per record in a collection. The list is capped
    # and the true count travels beside it, so a truncated summary is short without being wrong.
    from pydantic import BaseModel, ValidationError

    class _Item(BaseModel):
        depth: int

    class _Batch(BaseModel):
        items: list[_Item]

    with pytest.raises(ValidationError) as caught:
        _Batch(items=[{"depth": "x"} for _ in range(20)])
    summary = lifecycle.failure_summary(caught.value)
    assert summary["location_count"] == 20
    assert len(summary["locations"]) == lifecycle._MAX_FAILURE_LOCATIONS


def test_failure_summary_contains_a_failure_that_cannot_describe_itself() -> None:
    # Describing a caught failure runs the raiser's code a second time, outside the `except` that
    # contained it. One that raises again must not escape carrying the fail-closed commit with it.
    class _Hostile(Exception):
        def errors(self):
            raise ValueError("asking costs you the row")

    assert lifecycle.failure_summary(_Hostile()) == {"error": "_Hostile"}


async def test_terminal_failure_is_not_in_the_payload_the_agent_gets() -> None:
    # The summary is safe for a row and not for a reply: it describes the env's own state, so it
    # travels on the harness-side channel and never widens the sanitized terminal payload.
    def explode(_req: FinalizeRequest, _correct: bool) -> None:
        raise RuntimeError("evaluator exploded")

    ep = await _start(finalize_hook=explode)
    try:
        await ep.call("submit", {"answer": "4"})
        assert ep.terminal_payload is not None
        assert ep.terminal_failure == {"error": "RuntimeError"}
        assert "failure" not in ep.terminal_payload
        assert not any("RuntimeError" in str(value) for value in ep.terminal_payload.values())
    finally:
        await ep.close()
