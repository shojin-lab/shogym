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
from typing import Any

import pytest

from fastmcp import Client

from shogym.core import Env
from shogym.envs.registration import _ENV_REGISTRY, register
from shogym.feedback.wire import FEEDBACK_META_KEY
from shogym.serve import (
    FinalizationRecord,
    FinalizationStore,
    LifecycleState,
    ServedEpisode,
    TerminalEvidence,
)
from shogym.serve import lifecycle
from shogym.serve.lifecycle import FinalizeRequest, args_digest, fail_closed_verdict
from shogym.serve.server import build_server
from shogym.shared.terminate_mcp import TERMINATE_TOOL_NAME
from shogym.task import TaskSpec, ToolManifest
from shogym.trace import load_traces
from shogym.types import EpisodeFeedback, FeedbackCollection

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
            # The fixture's own tools plus the reserved ones, named rather than counted: a bare
            # count says nothing about which tool a change added or took away.
            assert {tool.name for tool in await client.list_tools()} == {
                "submit",
                "noop",
                "block",
                "describe",
                "terminate",
            }
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

    # The hook, not the public wrapper: the serve layer claims a session and then runs the hook
    # it was handed, so `_end_session` is where the release actually happens (see
    # `Env.claim_session`) and the only place an observer can stand.
    real_end_session = ep._env._end_session

    def observing_end_session(session_id):
        # _end_session drops per-session state, so the evaluator must have drained by now.
        evaluator_done["at_end_session"] = evaluator_done["flag"]
        return real_end_session(session_id)

    ep._finalize = slow  # type: ignore[assignment]
    ep._env._end_session = observing_end_session  # type: ignore[method-assign]
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
        assert ep._env._end_session is observing_end_session
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
        # A misbehaving env: a non-dict (list) verdict. json.dumps([...]) succeeds, so a guard
        # that asked only whether the verdict serializes let it through to
        # `dict(evidence.verdict)`, which raises.
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


async def test_a_verdict_that_raises_while_being_read_fails_closed(tmp_path: Path) -> None:
    # Deciding whether a verdict is serializable runs the env's own object: `json.dumps` reads a
    # dict subclass through its `items()`, which is the env's code. A failure there is a reason to
    # refuse the verdict, so it has to end in the same fail-closed commit a NaN or a list does —
    # not escape the guard, raise at the client, and leave the durable record at PENDING.
    from shogym.serve.lifecycle import TerminalEvidence

    class _UnreadableVerdict(dict):
        def items(self):
            raise RuntimeError("this verdict will not be read")

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def unreadable_verdict(req):
        return TerminalEvidence(
            source=req.source, status="ok", verdict=_UnreadableVerdict(correct=True)
        )

    ep._finalize = unreadable_verdict  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed, not the unreadable verdict
        assert payload["finalize_error"] is True
        assert ep._evidence is not None
        assert ep._evidence.verdict == fail_closed_verdict(50)
        assert ep._state is LifecycleState.CLOSED  # teardown ran
    finally:
        await ep.close()

    store = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace))
    recs = store.load_all()
    assert len(recs) == 1
    assert recs[0].status == "FAILED"  # resolved fail-closed, never stranded at PENDING


async def test_a_failure_that_will_not_render_still_gets_its_fail_closed_row(
    tmp_path: Path,
) -> None:
    # The private diagnostic renders the evaluator's exception, which runs that exception's own
    # `__str__`. One that raises used to replace the evaluator failure with the formatter failure
    # inside the handler that was writing the fail-closed record, so no record was written at all.
    # The row is what the failure is for: it is still written, and it still names the type.
    class _UnrenderableFailure(RuntimeError):
        def __str__(self) -> str:
            raise RuntimeError("this failure will not describe itself")

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def unrenderable_failure(req):
        raise _UnrenderableFailure()

    ep._finalize = unrenderable_failure  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed
        assert payload["finalize_error"] is True
        assert ep._evidence is not None
        assert ep._evidence.status == "finalize_error"
        # The diagnostic keeps the one thing the failure could still be asked for: its type.
        assert ep._evidence.diagnostic is not None
        assert "_UnrenderableFailure" in ep._evidence.diagnostic
        assert ep._evidence.failure == {"error": "_UnrenderableFailure"}
        assert ep._state is LifecycleState.CLOSED  # teardown ran
    finally:
        await ep.close()

    store = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace))
    recs = store.load_all()
    assert len(recs) == 1
    assert recs[0].status == "FAILED"
    assert recs[0].diagnostic is not None
    assert "_UnrenderableFailure" in recs[0].diagnostic


async def test_a_verdict_that_cancels_while_being_read_fails_closed(tmp_path: Path) -> None:
    # An env's `items()` can raise cancellation as readily as anything else, and this boundary
    # already holds that an env's own cancellation is contained: the evaluator handler above
    # catches it and fails closed rather than stranding the episode FINALIZING. Reading the
    # verdict to see whether it can be committed is the same kind of read, so it ends the same
    # way, in the fail-closed record, not carrying the transaction out with it.
    class _CancellingVerdict(dict):
        def items(self):
            raise asyncio.CancelledError()

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def cancelling_verdict(req):
        return TerminalEvidence(
            source=req.source, status="ok", verdict=_CancellingVerdict(correct=True)
        )

    ep._finalize = cancelling_verdict  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed, not the unreadable verdict
        assert payload["finalize_error"] is True
        assert ep._evidence is not None
        assert ep._evidence.verdict == fail_closed_verdict(50)
        assert ep._state is LifecycleState.CLOSED  # teardown ran
    finally:
        await ep.close()

    store = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace))
    recs = store.load_all()
    assert len(recs) == 1
    assert recs[0].status == "FAILED"  # resolved fail-closed, never stranded at PENDING


async def test_a_failure_whose_type_will_not_name_itself_still_gets_its_row(
    tmp_path: Path,
) -> None:
    # Falling back to the type name is only a fallback if the name is a string. A metaclass
    # decides what `__name__` answers, so it can answer with an object of the env's own, and the
    # diagnostic that object lands in is the one being written because the evaluator already
    # failed. The row survives with a fixed name in place of the one that would not render.
    class _UnprintableName:
        def __str__(self) -> str:
            raise RuntimeError("this type will not name itself")

    class _NamelessType(type):
        @property
        def __name__(cls) -> Any:  # type: ignore[override]
            return _UnprintableName()

    class _FailureWithoutAName(RuntimeError, metaclass=_NamelessType):
        pass

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def nameless_failure(req):
        raise _FailureWithoutAName()

    ep._finalize = nameless_failure  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed
        assert payload["finalize_error"] is True
        assert ep._evidence is not None
        assert ep._evidence.diagnostic == "finalize failed: <unreadable>"
        assert ep._evidence.failure == {"error": "<unreadable>"}
        assert ep._state is LifecycleState.CLOSED  # teardown ran
    finally:
        await ep.close()

    store = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace))
    recs = store.load_all()
    assert len(recs) == 1
    assert recs[0].status == "FAILED"


async def test_the_diagnostic_reads_a_failure_once_and_before_the_summary() -> None:
    # A structured failure describes itself through `errors()`, which is the env's method and may
    # do anything, including change what the exception says about itself. The diagnostic renders
    # the failure as it was caught, and the summary is the only thing that asks it for structure,
    # so the row carries the message the evaluator failed with rather than one produced by being
    # asked about it.
    class _Stateful(RuntimeError):
        calls = 0

        def __init__(self) -> None:
            super().__init__("original")

        def errors(self):
            type(self).calls += 1
            self.args = (f"mutated on errors call {type(self).calls}",)
            return [{"type": "value_error", "loc": ("answer",)}]

    ep = await _start()

    async def stateful_failure(req):
        raise _Stateful()

    ep._finalize = stateful_failure  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        assert json.loads(result.content)["finalize_error"] is True
        assert ep._evidence is not None
        assert ep._evidence.diagnostic == "finalize failed: _Stateful: original"
        assert _Stateful.calls == 1  # the summary asked; the diagnostic did not
        assert ep._evidence.failure == {"error": "_Stateful", "error_count": 1}
    finally:
        await ep.close()


async def test_feedback_that_cannot_be_recorded_still_gets_the_fail_closed_row(
    tmp_path: Path,
) -> None:
    # Between the caught evaluator failure and the durable record that says so, the verifier
    # runs and its feedback is put into wire form, which validates it. The models do not validate
    # on assignment, so a mutated item is an ordinary malformed env output; recording it used to
    # raise after the evidence had been failed closed in memory and before the record was
    # replaced, leaving the record at PENDING and the failure invisible to recovery.
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def failing_finalize(req):
        raise RuntimeError("the evaluator is down")

    def unrecordable_feedback(trajectory, task, *, terminated, evidence=None):
        collection = FeedbackCollection(episode=[EpisodeFeedback(name="score", value=1.0)])
        collection.episode[0].value = object()  # type: ignore[assignment]
        return collection

    ep._finalize = failing_finalize  # type: ignore[assignment]
    ep._env.verify = unrecordable_feedback  # type: ignore[method-assign]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed
        assert payload["finalize_error"] is True
        assert ep.terminal_feedback == []  # nothing recordable survived the verifier
        assert ep._evidence is not None
        assert ep._evidence.status == "finalize_error"
        assert ep._evidence.failure == {"error": "ValueError"}  # what actually happened
        assert ep._state is LifecycleState.CLOSED  # teardown ran
    finally:
        await ep.close()

    store = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace))
    recs = store.load_all()
    assert len(recs) == 1
    assert recs[0].status == "FAILED"  # resolved fail-closed, never stranded at PENDING


async def _fail_closed_record(tmp_path: Path, finalize) -> FinalizationRecord:
    """Run one terminal call against `finalize`, require the fail-closed result at the caller,
    and hand back the single durable record the transaction left behind."""
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    ep._finalize = finalize  # type: ignore[assignment]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed, never the env's own word
        assert payload["finalize_error"] is True
        assert ep._state is LifecycleState.CLOSED  # teardown ran
    finally:
        await ep.close()
    recs = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace)).load_all()
    assert len(recs) == 1
    return recs[0]


async def test_a_verdict_the_record_cannot_rebuild_still_resolves_the_row(
    tmp_path: Path,
) -> None:
    # Serializing a verdict proves it can be serialized and nothing else. The durable record is
    # built with `asdict`, which rebuilds a mapping by calling its own type, and the type is the
    # env's: one that refuses the rebuild used to raise inside the write, after the transaction
    # had committed in memory and while the row still said PENDING. The verdict the record is
    # written from is the plain one this core read out of the env's, so the rebuild is its own.
    class _UnrebuildableVerdict(dict):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            if args:  # `asdict` rebuilds by passing the pairs back in
                raise asyncio.CancelledError()
            super().__init__(**kwargs)

    async def unrebuildable_verdict(req):
        return TerminalEvidence(
            source=req.source,
            status="finalize_error",
            verdict=_UnrebuildableVerdict(correct=False, finalize_error=True),
        )

    rec = await _fail_closed_record(tmp_path, unrebuildable_verdict)
    assert rec.status == "FAILED"  # resolved fail-closed, never stranded at PENDING
    assert rec.verdict == {"correct": False, "finalize_error": True}
    assert type(rec.verdict) is dict  # the plain value, not the env's mapping


async def test_a_verdict_whose_type_takes_no_arguments_still_resolves_the_row(
    tmp_path: Path,
) -> None:
    # The same rebuild, refused by an ordinary mapping rather than a hostile one: a dict subclass
    # that only constructs empty. The write failed, the flag it set was in memory, and the row on
    # disk stayed PENDING, which is the state a recovery reads as an unfinished finalization.
    class _ZeroArgVerdict(dict):
        def __init__(self) -> None:
            super().__init__(correct=True, score=1)

    async def zero_arg_verdict(req):
        return TerminalEvidence(source=req.source, status="ok", verdict=_ZeroArgVerdict())

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    ep._finalize = zero_arg_verdict  # type: ignore[assignment]
    try:
        # This verdict is honest, only oddly typed: it is scored, not failed closed.
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        payload = json.loads(result.content)
        assert payload["correct"] is True
        assert payload["finalize_error"] is False
        assert ep._persist_degraded is False  # the record was written, not degraded
    finally:
        await ep.close()

    recs = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace)).load_all()
    assert len(recs) == 1
    assert recs[0].status == "FINALIZED"  # resolved, never stranded at PENDING
    assert recs[0].verdict == {"correct": True, "score": 1}


async def test_a_status_that_will_not_compare_fails_closed(tmp_path: Path) -> None:
    # The declared status is env data too, and it is read by comparison: the public payload asks
    # whether it is `finalize_error` before the durable write. A string subclass whose equality
    # raises used to take the whole transaction out through that question. The status this core
    # keeps is its own constant, matched once inside the guard, so an answer that will not be
    # compared is an outcome rather than an escape.
    class _HostileStatus(str):
        __hash__ = str.__hash__

        def __eq__(self, other: object) -> bool:
            raise RuntimeError("status equality failed")

    async def hostile_status(req):
        return TerminalEvidence(
            source=req.source,
            status=_HostileStatus("ok"),  # type: ignore[arg-type]
            verdict={"correct": True},
        )

    rec = await _fail_closed_record(tmp_path, hostile_status)
    assert rec.status == "FAILED"  # resolved fail-closed, never stranded at PENDING
    # The verdict was fine; the row says which field the refusal came from.
    assert rec.diagnostic == "finalize returned an undeclared or unreadable status"


async def test_an_undeclared_status_says_so_in_the_row(tmp_path: Path) -> None:
    # The returned status is dropped, so the diagnostic is the only thing left to send whoever
    # repairs the env to the field that refused. A row that blames the verdict for a status this
    # core does not declare sends them to a value that was serializable and correct.
    async def undeclared_status(req):
        return TerminalEvidence(
            source=req.source,
            status="future_status",  # type: ignore[arg-type]
            verdict={"correct": True, "score": 1.0},
        )

    rec = await _fail_closed_record(tmp_path, undeclared_status)
    assert rec.status == "FAILED"
    assert rec.diagnostic == "finalize returned an undeclared or unreadable status"
    assert "future_status" not in rec.diagnostic  # the env's value is named nowhere


async def test_an_integer_grade_keeps_its_type_everywhere(tmp_path: Path) -> None:
    # A feedback value is read once and the items are rebuilt from that read, so the rebuild is
    # what the score becomes. The wire contract admits a whole number and the models' value type
    # does not name one, so rebuilding through validation would turn the grade into a float: the
    # retained score, the trace row and the sidecar would carry a different number from the one
    # the verifier reported, and a large one would carry a different number from each other.
    big = 2**60 + 1  # exact as an int, not as a float

    def integer_verify(trajectory, task, *, terminated, evidence=None):
        item = EpisodeFeedback(name="score", value=1.0)
        item.value = big  # a post-construction assignment, which the models allow
        return FeedbackCollection(episode=[item])

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    ep._env.verify = integer_verify  # type: ignore[method-assign]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        retained = ep.terminal_feedback[0]["value"]
        assert retained == big and type(retained) is int
        surfaced = result.meta[FEEDBACK_META_KEY][0]["value"]
        assert surfaced == big and type(surfaced) is int
    finally:
        await ep.close()

    step = [r for r in load_traces(trace) if r.get("kind") != "terminal"][-1]
    recorded = step["feedback"][0]["value"]
    assert recorded == big and type(recorded) is int


async def test_a_verifier_cannot_rewrite_a_caught_failure_into_a_success(
    tmp_path: Path,
) -> None:
    # The verifier is handed the evidence to score, and the evidence is a plain dataclass, so
    # what it is handed it can also edit. One that declares the failed evaluation `ok` and swaps
    # the verdict used to have the handler persist FINALIZED with a success verdict and the
    # fail-closed diagnostic still beside it. The verifier scores a copy; the row is written from
    # the instance it never saw.
    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )

    async def failing_finalize(req):
        raise RuntimeError("the evaluator is down")

    def rewriting_verify(trajectory, task, *, terminated, evidence=None):
        assert evidence is not None
        evidence.status = "ok"
        evidence.verdict.clear()
        evidence.verdict.update({"correct": True, "mutated_by_verify": True})
        evidence.diagnostic = None
        evidence.failure = None
        evidence.provenance = {"core": "not-shogym"}
        evidence.finalization_id = "forged"
        return FeedbackCollection()

    ep._finalize = failing_finalize  # type: ignore[assignment]
    ep._env.verify = rewriting_verify  # type: ignore[method-assign]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        payload = json.loads(result.content)
        assert payload["correct"] is False  # fail-closed, not the verifier's rewrite
        assert payload["finalize_error"] is True
        assert "mutated_by_verify" not in payload
        assert ep._evidence is not None
        assert ep._evidence.status == "finalize_error"
        assert ep._evidence.verdict == fail_closed_verdict(50)
        assert ep._evidence.provenance is not None
        assert ep._evidence.provenance["core"] == "shogym-serve"
    finally:
        await ep.close()

    recs = FinalizationStore(FinalizationStore.resolve_dir(ep.session_id, trace)).load_all()
    assert len(recs) == 1
    assert recs[0].status == "FAILED"  # grading never completed; the row says so
    assert recs[0].verdict is not None and recs[0].verdict["correct"] is False
    assert recs[0].diagnostic is not None and "finalize failed" in recs[0].diagnostic


async def test_a_finalizer_that_mutates_the_request_args_still_gets_its_row(
    tmp_path: Path,
) -> None:
    # The finalizer is handed the caller's own args dict, and the durable write used to digest
    # that dict again on its way in, which is an env-influenced read outside the write's guard.
    # A cycle added by the finalizer made the digest raise and left the row PENDING. The digest
    # is taken once at the seal, before any env code runs, and every row uses that one string.
    async def cyclic_args(req):
        req.args["itself"] = req.args  # the caller's dict, by reference
        raise RuntimeError("the evaluator is down")

    rec = await _fail_closed_record(tmp_path, cyclic_args)
    assert rec.status == "FAILED"  # resolved fail-closed, never stranded at PENDING
    # The digest is of the args the transaction was entered with, not of what the env made them.
    assert rec.args_digest == args_digest({"answer": "4", "confidence": 50})


async def test_feedback_is_read_once_and_the_trace_reads_the_core_copy(
    tmp_path: Path,
) -> None:
    # A feedback item is the verifier's object and is serialized more than once on the way out:
    # for the retained score, for the trace row, and for the result's sidecar. An item that
    # answered the first read and refused the later ones degraded the trace and then raised at
    # the caller, after the episode had already been scored. The item is read once and rebuilt
    # from its own wire form, so everything after that serializes a core object.
    reads = {"value": 0}

    class _ReadOnceItem(EpisodeFeedback):
        def __getattribute__(self, name: str) -> Any:
            if name == "value":
                reads["value"] += 1
                if reads["value"] > 1:
                    raise RuntimeError(f"value reread {reads['value']}")
            return super().__getattribute__(name)

    def read_once_verify(trajectory, task, *, terminated, evidence=None):
        return FeedbackCollection(episode=[_ReadOnceItem(name="score", value=1.0)])

    trace = tmp_path / "run.jsonl"
    ep = await ServedEpisode.start(
        "_fixture_score", task=0, trace_path=trace, env_config=_config()
    )
    ep._env.verify = read_once_verify  # type: ignore[method-assign]
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": 50})
        assert result.terminated is True
        assert json.loads(result.content)["finalize_error"] is False
        assert reads["value"] == 1  # the verifier's object, read exactly once
        assert _feedback(ep) == {"score": 1.0}
        assert ep._persist_degraded is False
    finally:
        await ep.close()

    step = [r for r in load_traces(trace) if r.get("kind") != "terminal"][-1]
    assert step["feedback"] == [{"name": "score", "value": 1.0, "level": "episode"}]


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


def test_failure_summary_keeps_the_kind_and_count_and_drops_the_message() -> None:
    # A structured failure contributes how many errors there were and which kinds, and nothing
    # else: everything else it reports is drawn from the data being validated, which for an env
    # is the thing being graded.
    from pydantic import BaseModel, ValidationError

    class _Verdict(BaseModel):
        depth: int
        label: str

    with pytest.raises(ValidationError) as caught:
        _Verdict(depth="not-an-int", label=None)  # type: ignore[arg-type]
    summary = lifecycle.failure_summary(caught.value)
    assert summary["error"] == "ValidationError"
    assert summary["error_count"] == 2
    assert summary["error_kinds"] == ["int_parsing", "string_type"]
    assert "not-an-int" not in json.dumps(summary)


def test_failure_summary_never_publishes_a_mapping_key() -> None:
    # A reported location descends into the INPUT, not the schema: a failure inside a mapping
    # contributes that mapping's own keys. Those keys are the env's data, so a summary that
    # published locations would put answer-bearing state into a record that outlives the episode.
    from pydantic import BaseModel, ValidationError
    from typing import Dict

    class _Verdict(BaseModel):
        answers: Dict[str, int]

    with pytest.raises(ValidationError) as caught:
        _Verdict(answers={"gold-answer-42": "not-an-int", "the capital is paris": "nope"})
    # The location really does carry the keys, which is what this guards against.
    assert ("answers", "gold-answer-42") in [err["loc"] for err in caught.value.errors()]
    blob = json.dumps(lifecycle.failure_summary(caught.value))
    assert "gold-answer-42" not in blob
    assert "paris" not in blob
    assert "not-an-int" not in blob


def test_failure_summary_refuses_an_error_kind_the_validator_invented() -> None:
    # A kind is a bare string, and only the library's own documentation link says where it came
    # from. One supplied by whoever wrote the validator has no link and is dropped, so a kind
    # built out of the data cannot ride out on this channel either.
    from pydantic import BaseModel, ValidationError, field_validator
    from pydantic_core import PydanticCustomError

    class _Verdict(BaseModel):
        label: str

        @field_validator("label")
        @classmethod
        def _reject(cls, value: str) -> str:
            raise PydanticCustomError("the_answer_is_paris", "boom")

    with pytest.raises(ValidationError) as caught:
        _Verdict(label="anything")
    summary = lifecycle.failure_summary(caught.value)
    assert summary == {"error": "ValidationError", "error_count": 1}
    assert "paris" not in json.dumps(summary)


def test_failure_summary_caps_the_kinds_it_lists() -> None:
    # A pathological failure reporting many distinct kinds is bounded rather than unbounded; the
    # true count travels beside the list, so a capped summary is short without being wrong.
    cap = lifecycle._MAX_FAILURE_ERROR_KINDS

    class _Many(Exception):
        def errors(self):
            return [
                {"type": f"kind_{i}", "url": f"https://errors.pydantic.dev/2.13/v/kind_{i}"}
                for i in range(cap + 5)
            ]

    summary = lifecycle.failure_summary(_Many())
    assert summary["error_count"] == cap + 5
    assert len(summary["error_kinds"]) == cap


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
