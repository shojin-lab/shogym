"""What a served episode does when the *env* is the thing that fails.

Every case here is an env fault reached on a path the episode runs before, during or just after
the terminal transaction — a budget the env answers differently than it published, an evaluator
or verifier that raises the one exception ``except Exception`` does not catch, a value whose own
code runs while the episode is writing the record about it. The property under test is the same
one each time and it is not "the episode survives": it is that the fault is **recorded as a
fault**, distinguishable from the agent having answered wrong, and never converted into an
outcome that reads as legitimate.

The two shapes that are failures here are the two faces of one defect. A run that reports
``score.success=False`` over a task the agent never got to finish is the first. A run that drops
an *earned* verdict — or strands the episode mid-finalize and closes clean — is the second. Both
are a wrong number that looks like a right one, which is the one thing this layer may not
produce.

These drive the ``_fixture_score`` env (tests/_fixtures/score_env.py) and Wordle, and reach the
faults by replacing one env method or property per test, because the fault has to belong to the
env: containing it here is only meaningful if the code that fails is code the serve layer did
not write.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import pytest

from shogym.serve import QueueInfo, ServedEpisode, TaskContractError, TaskRef, TaskStream
from shogym.serve import lifecycle
from shogym.serve.lifecycle import (
    LifecycleState,
    TerminalEvidence,
    args_digest,
    fail_closed_verdict,
)
from shogym.task import TaskSpec, ToolManifest
from shogym.feedback import parse_meta
from shogym.trace import load_traces
from shogym.types import EpisodeFeedback, FeedbackCollection

import tests._fixtures.score_env as fixture  # registers `_fixture_score`  # noqa: F401
from tests._fixtures import score_mcp

_TASKS = [{"id": "q0", "question": "What is 2+2?", "answer": "4"}]


@pytest.fixture(autouse=True)
def _clean_sessions(tmp_path_factory, monkeypatch):
    # Redirect the no-trace durable fallback root off the real ~/.cache into a tmp dir, so
    # these episodes don't pollute the home cache and each test reads only its own records.
    root = tmp_path_factory.mktemp("shogym-sessions")
    monkeypatch.setattr(lifecycle, "_sessions_cache_root", lambda: root)
    score_mcp.reset_state()
    yield
    score_mcp.reset_state()


def _env(**kwargs: Any) -> Any:
    return fixture._FixtureScoreEnv(tasks=_TASKS, **kwargs)


async def _open(env: Any) -> ServedEpisode:
    return await ServedEpisode.open_env(env, env_name="_fixture_score", task=0)


# ----- the published budget is the enforced budget -----


async def test_a_budget_the_env_answers_differently_than_it_published_is_refused() -> None:
    """The snapshot says the episode has three calls; the env's live ``horizon`` says one.

    Enforcing the live answer ends the task on the agent's *first* ordinary call and files the
    finalization the budget owes — a sealed episode, a ``correct=False`` verdict, a clean close.
    Nothing in that record says the agent was cut off two calls before the budget it was framed
    with: it reads exactly like an agent that used its turns and got the answer wrong.

    So the published budget is the only one enforced, and an env that answers a different one is
    not quietly overruled either — it is refused out loud, on the call that found the
    disagreement, before that call is dispatched. A refusal costs this task its score (the layer
    above files an unscored row naming the failure); a silent correction would leave an env free
    to publish one contract and run on another for as long as nobody compared them.
    """
    env = _env()
    ep = await _open(env)
    try:
        assert ep.describe().horizon == 3
        # The env changes its mind *after* the contract is published and the agent is framed.
        env._horizon = 1

        with pytest.raises(RuntimeError, match="budget"):
            await ep.call("noop", {})

        # The refusal is not an ending. Nothing was sealed, nothing was scored, and no
        # finalization ran: the task is still open, and the record above it is free to say the
        # call was lost rather than that the agent answered wrong.
        assert not ep.terminated
        assert ep._state is LifecycleState.OPEN
        assert ep.terminal_source is None
        assert ep.terminal_payload is None
        assert ep.terminal_feedback == []
        assert ep._store is not None and ep._store.load_all() == []
    finally:
        env._horizon = 3
        await ep.close()


async def test_the_published_budget_still_ends_the_episode_when_the_env_agrees() -> None:
    """The other side of that line: an env that publishes what it enforces is untouched. Three
    calls, the third is the terminal step, and the verdict is the finalizer's."""
    env = _env()
    ep = await _open(env)
    try:
        assert (await ep.call("noop", {})).terminated is False
        assert (await ep.call("noop", {})).terminated is False
        end = await ep.call("noop", {})
        assert end.terminated is True
        assert ep.terminal_source == "horizon"
        assert ep.terminal_payload == {"correct": False, "finalize_error": False}
    finally:
        await ep.close()


async def test_a_budget_that_cannot_be_read_is_refused_on_the_non_seal_path() -> None:
    """The same rule on the legacy (non-seal) step, which reads the budget on its own path.

    Wordle publishes six; an env object that answers something else there would end the episode
    early with the terminal feedback of a completed task — a scored row for a task that was cut
    short."""
    import shogym

    env = shogym.make("wordle_v1")
    ep = await ServedEpisode.open_env(env, env_name="wordle_v1", task=0)
    try:
        assert ep.describe().horizon == 6
        env._horizon = 1
        with pytest.raises(RuntimeError, match="budget"):
            await ep.call("guess", {"word": "crane"})
        assert not ep.terminated
        assert ep.terminal_feedback == []
        # And the refused call was never run against the env: no step was spent on it.
        assert ep._step == 0 and ep._trajectory == []
    finally:
        env._horizon = 6
        await ep.close()


async def test_the_verifier_scores_from_a_copy_and_cannot_rewrite_the_outcome() -> None:
    """``verify`` takes the terminal evidence so a migrated env can score from it, which hands the
    env a reference to it for the length of that call. The commit then reads ``status``,
    ``verdict``, ``provenance`` and ``diagnostic`` off the same object again, and asks none of the
    questions the evaluator boundary asked.

    So a verifier could return ordinary feedback and rewrite the outcome on its way out. Measured
    on a finalizer that failed honestly and a verifier that rewrote ``status`` to an undeclared
    string and cleared the diagnostic: the record was written ``FINALIZED``, the agent was
    answered ``finalize_error=false``, and the undeclared-status path this transaction closes was
    open again through a different door. The same handle rewrote ``provenance``, which the core
    stamps precisely so a harness cannot supply it, and the forgery reached the durable record.

    "The value checked is the value committed" needs its corollary: nothing foreign holds a
    reference to that value between the check and the commit. The verifier gets a copy, and what
    it does to the copy is its own business."""
    env = _env()
    seen: List[Any] = []

    async def finalize(req: Any) -> Any:
        return TerminalEvidence(
            source=req.source,
            status="finalize_error",
            verdict=fail_closed_verdict(),
            diagnostic="the judge fell over",
        )

    def verify(trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None) -> Any:
        from shogym.types import FeedbackCollection

        if evidence is not None:
            seen.append(evidence)
            evidence.status = "not-a-terminal-status"
            evidence.diagnostic = None
            evidence.provenance = {"core": "not-shogym"}
            evidence.verdict["correct"] = True
        return FeedbackCollection()

    env.finalize = finalize  # type: ignore[method-assign]
    env.verify = verify  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        result = await ep.call("submit", {"answer": "4"})

        # What the agent is answered with is the outcome the evaluator actually reached.
        assert json.loads(result.content) == {"correct": False, "finalize_error": True}
        # And so is the record, down to the private diagnostic and the stamped provenance.
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FAILED"
        assert record.verdict == {"correct": False, "finalize_error": True}
        assert record.diagnostic == "the judge fell over"
        assert (record.provenance or {}).get("core") == "shogym-serve"
        # The episode's own evidence is untouched, and it is not the object the verifier held.
        assert ep._evidence is not None
        assert ep._evidence.status == "finalize_error"
        assert seen and seen[0] is not ep._evidence
    finally:
        await ep.close()


def _failing_at_step(mode: str, at: int) -> Any:
    """An env whose ordinary (non-terminal) verification fails on the call that makes the
    trajectory ``at`` steps long: by raising, or by publishing feedback the wire refuses."""

    class _FailsMidEpisode(fixture._FixtureScoreEnv):  # type: ignore[name-defined]
        def _verify(self, trajectory, task, *, terminated, evidence=None) -> FeedbackCollection:
            if not terminated and len(trajectory) >= at:
                if mode == "raises":
                    raise RuntimeError("this env cannot verify its own step")
                collection = FeedbackCollection()
                item = EpisodeFeedback(name="note", value=True)
                item.value = object()  # the models do not validate on assignment
                collection.episode.append(item)
                return collection
            return super()._verify(trajectory, task, terminated=terminated, evidence=evidence)

    return _FailsMidEpisode


@pytest.mark.parametrize("mode", ["raises", "publishes what the wire refuses"])
async def test_an_env_that_fails_at_the_budget_cannot_buy_its_agent_another_turn(
    mode: str,
) -> None:
    """The budget-reaching call commits its result and then used to run the env again.

    ``_dispatch_step`` commits the step count and the trajectory, and only then verifies and
    renders what the env returned. The caller claims the horizon finalization after that returns,
    so a raise in between left the published budget spent, the trajectory extended, the episode
    still ``OPEN`` at ``step == horizon``, and no finalization in existence. The *next* call then
    sealed a task that had already run out of turns: measured on the public path, an extra turn
    and a clean ``sealed`` row reading ``success=true`` with no diagnostic anywhere. An env fault
    at the exact boundary, wearing the shape of a result the agent earned.

    The preliminary pass is not only dangerous at the budget, it is dead weight there: its trace
    row is deferred so the terminal row can be that same step, the ``CallResult`` it builds is
    discarded by the caller, and it touches nothing retained. The terminal ``verify`` sees the
    same trajectory including this step and is the authoritative sink for it. So the work is not
    reordered at the budget, it is skipped, and the window it opened is gone."""
    env = _failing_at_step(mode, fixture.HORIZON)(tasks=_TASKS)
    ep = await _open(env)
    try:
        for _ in range(fixture.HORIZON - 1):
            assert (await ep.call("noop", {})).terminated is False
        # The call that spends the budget ends the task, whatever the env does after the result.
        budget = await ep.call("noop", {})
        assert budget.terminated is True, "the env's fault at the budget left the task open"
        assert ep._step == fixture.HORIZON, "the budget bought more than it should have"
        assert ep.terminated is True
        assert ep.terminal_source == "horizon"
        # And nothing follows it: the next call is answered by the tombstone, not dispatched.
        after = await ep.call("submit", {"answer": "4"})
        assert after.tombstoned is True, "a task out of turns took another one"
        assert ep._step == fixture.HORIZON
        # The verdict is the horizon's own, which is what a spent budget earns.
        assert ep.terminal_payload == {"correct": False, "finalize_error": False}
    finally:
        await ep.close()


@pytest.mark.parametrize("mode", ["raises", "publishes what the wire refuses"])
async def test_the_same_failure_away_from_the_budget_is_still_a_lost_call(mode: str) -> None:
    """The scope of that fix, stated as its own test.

    Away from the budget the same window is unchanged and is meant to be: an env that fails after
    its result is committed costs the agent that call, the episode stays ``OPEN`` with the turn
    spent, and the layer above records a lost call. What makes the budget different is not the
    failure, it is that a lifecycle transition is owed at exactly that step, and nothing fallible
    may stand between the commit and the claim of it."""
    env = _failing_at_step(mode, 1)(tasks=_TASKS)
    ep = await _open(env)
    try:
        with pytest.raises((RuntimeError, ValueError)):
            await ep.call("noop", {})
        # The call is lost, the turn is spent, and the episode is still the agent's to finish.
        assert ep._state is LifecycleState.OPEN
        assert ep._step == 1 and ep.terminated is False
        # Which it can still do: the terminal path is untouched by the failure above.
        result = await ep.call("submit", {"answer": "4"})
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
    finally:
        await ep.close()


# ----- nothing escapes between the seal and the finalization that answers for it -----


def _unconstrained_confidence(spec: Any) -> None:
    for manifest in spec.tools:
        if manifest.name == "submit":
            manifest.input_schema["properties"]["confidence"] = {}


def _unusable_terminal_schema(spec: Any) -> None:
    for manifest in spec.tools:
        if manifest.name == "submit":
            # Plain JSON, so it renders and copies like any other schema, and resolving it while
            # validating a call raises out of the validator.
            #
            # The dangling reference sits under an **optional** property deliberately. A schema
            # whose machinery cannot run at all is refused at construction now, because a document
            # this layer cannot execute is one it cannot promise to enforce (a bare
            # ``{"$ref": "#/definitions/answer"}`` root is exactly that, and is covered by
            # ``test_a_schema_this_layer_cannot_execute_is_refused_before_it_is_advertised``).
            # What stays call-time-only is a reference nothing reaches until a call carries the
            # property: the empty object satisfies this schema, so construction establishes that a
            # call is possible, and the resolution failure lands on the call that supplies
            # ``answer``. That is the shape this arm is about.
            manifest.input_schema = {
                "type": "object",
                "properties": {"answer": {"$ref": "#/definitions/answer"}},
            }


def _horizon_that_cannot_be_read(env: Any) -> None:
    type(env).horizon = property(  # type: ignore[assignment]
        lambda self: (_ for _ in ()).throw(RuntimeError("this env cannot say what its budget is"))
    )


def _verify_that_raises(env: Any) -> None:
    def verify(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("this env cannot verify a step")

    env.verify = verify  # type: ignore[method-assign]


class _ConfidenceThatCannotBeDigested:
    """A submitted value that passes an unconstrained schema and refuses to be rendered."""

    def __str__(self) -> str:
        raise RuntimeError("this argument cannot be digested")


@pytest.mark.parametrize(
    ("describe_as", "arm", "tool", "arguments", "failure"),
    [
        pytest.param(
            _unusable_terminal_schema,
            None,
            "submit",
            {"answer": "4"},
            "definitions/answer",
            id="the terminal call cannot be validated",
        ),
        pytest.param(
            _unconstrained_confidence,
            None,
            "submit",
            {"answer": "4", "confidence": _ConfidenceThatCannotBeDigested()},
            "cannot be digested",
            id="the terminal arguments cannot be digested",
        ),
        pytest.param(
            None,
            _horizon_that_cannot_be_read,
            "noop",
            {},
            "cannot say what its budget is",
            id="the budget cannot be read",
        ),
        pytest.param(
            None,
            _verify_that_raises,
            "noop",
            {},
            "cannot verify a step",
            id="an ordinary call cannot be verified",
        ),
    ],
)
async def test_a_call_that_fails_before_the_seal_leaves_an_episode_that_can_still_end(
    describe_as: Any, arm: Any, tool: str, arguments: Dict[str, Any], failure: str
) -> None:
    """The terminal transaction is a state machine, and this is the one invariant that makes it
    a transaction: **an episode that has left OPEN owes exactly one verdict, and the finalization
    future is the only thing that can pay it.**

    So everything that runs code this module did not write has to happen on one side or the other
    of the seal, never between it and the finalization claim. Above it, a failure is an ordinary
    lost call: the episode is still open, the agent can call again, and if it never does, the
    layer above drives the terminal itself and files an unscored row naming the failure. Inside
    the finalization, a failure becomes the canonical fail-closed verdict. In between there is no
    third answer, and one was reachable: the argument digest ran after the transition, so a
    submitted value that refused to render left an episode that had sealed, would never
    terminate, held no future for ``wait_finalized()`` or ``close()`` to join, and had not even
    written the ``SEALED`` record recovery would have resolved.

    Every env-reachable step before the seal is held to it here, not just the one that was
    broken: validating the call against the env's schema, digesting the arguments, reading the
    published budget, and dispatching an ordinary call through the env. Each is armed in turn and
    each must leave the same thing behind: an episode still OPEN, unsealed, with no finalization
    and nothing durable, which ``close()`` can still end into a verdict."""
    env = _env()
    if describe_as is not None:
        _describing(env, describe_as)
    ep = await _open(env)
    if arm is not None:
        arm(env)
    try:
        with pytest.raises(Exception, match=failure):  # noqa: B017 — the type is the env's
            await ep.call(tool, arguments)

        # Nothing left OPEN, so nothing is owed a verdict yet.
        assert ep._state is LifecycleState.OPEN
        assert ep.sealed is False
        assert ep.terminated is False
        assert ep._finalization is None, "a sealed episode with nothing to answer for it"
        assert ep._store is not None
        assert [r for r in ep._store.load_all() if r.session_id == ep.session_id] == []
        # And the episode is still the agent's to finish, which is what "still open" has to mean.
        assert ep.terminal_source is None and ep.terminal_payload is None
    finally:
        if arm is _horizon_that_cannot_be_read:
            del type(env).horizon
        if arm is _verify_that_raises:
            del env.verify
        await ep.close()

    # Closing an episode nobody ended claims the abort itself, so the task ends with a verdict
    # rather than with nothing. That is the half the broken ordering removed: there was no
    # finalization for close to join and none was claimed, so the episode simply stopped.
    assert ep.terminated is True
    assert ep.terminal_source == "abort"
    assert ep.terminal_payload == {"correct": False, "aborted": True, "finalize_error": False}
    (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
    assert record.status == "FINALIZED"


# ----- an env that raises `CancelledError` is an env that failed -----


async def test_a_verifier_that_raises_cancellation_fails_the_verdict_closed() -> None:
    """``verify`` scoring the terminal evidence raises ``asyncio.CancelledError``.

    ``CancelledError`` is a ``BaseException``, so the fail-closed guard around the verifier —
    written as ``except Exception`` — does not catch it, and it is the one exception an env can
    raise that walks straight out of the finalization. Everything the commit still owed is
    skipped: the durable record stays ``PENDING``, the trace never gets its terminal event, the
    lifecycle never leaves ``FINALIZING``. The caller's ``call()`` raises cancellation, so does
    ``wait_finalized()``, and so does every ``close()`` — the episode cannot even be shut down,
    and the layer above files the earned submission as a broker abort.

    An env raising cancellation is that env failing; nothing here asked for it (the finalization
    is shielded from every caller, so an awaiter's cancellation never reaches this code). It is
    contained exactly like any other verifier crash: the verdict fails closed to
    ``finalize_error`` and the caller is answered with it."""
    env = _env()
    real_verify = env.verify

    def verify(*a: Any, **kw: Any) -> Any:
        if kw.get("terminated"):
            raise asyncio.CancelledError("this env cannot verify the terminal evidence")
        return real_verify(*a, **kw)

    env.verify = verify  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        # The correct answer, so nothing about the *submission* explains the zero.
        result = await ep.call("submit", {"answer": "4"})

        assert result.terminated is True
        assert json.loads(result.content) == {"correct": False, "finalize_error": True}
        # Flagged, so the zero is filterable from an honest wrong answer.
        assert ep.terminal_payload == {"correct": False, "finalize_error": True}
        assert ep._evidence is not None and ep._evidence.finalize_error is True

        # The transaction completed rather than stranding: FINALIZED in memory, FAILED on disk,
        # with the failure named in the private diagnostic.
        assert ep._state is LifecycleState.CLOSED
        assert ep._torn_down is True
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FAILED"
        assert record.verdict == {"correct": False, "finalize_error": True}
        assert "verify" in (record.diagnostic or "")

        await ep.wait_finalized()  # must not raise
    finally:
        await ep.close()  # must not raise


async def test_a_verifier_that_raises_cancellation_does_not_wedge_a_horizon_episode() -> None:
    """The same fault on the other way in. A horizon terminal reuses the call that hit the
    budget, so the failure lands on a step already committed to the trajectory — the episode has
    to finish the transaction from there rather than leave the budget-reaching call unrecorded."""
    env = _env()
    real_verify = env.verify

    def verify(*a: Any, **kw: Any) -> Any:
        if kw.get("terminated"):
            raise asyncio.CancelledError("this env cannot verify the terminal evidence")
        return real_verify(*a, **kw)

    ep = await _open(env)
    try:
        await ep.call("noop", {})
        await ep.call("noop", {})
        env.verify = verify  # type: ignore[method-assign]
        end = await ep.call("noop", {})  # the budget-reaching call
        assert end.terminated is True
        assert json.loads(end.content) == {"correct": False, "finalize_error": True}
        assert ep.terminal_source == "horizon"
        assert ep._state is LifecycleState.CLOSED
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FAILED"
    finally:
        await ep.close()


# ----- a failed setup reports the failure that was the setup's -----


async def test_a_cleanup_that_raises_cancellation_does_not_replace_the_setup_failure() -> None:
    """Setup fails, and the env's ``close`` raises ``CancelledError`` while releasing it.

    The cleanup catches ``Exception``, so the cancellation walks out in place of the setup
    failure, which survives only as ``__context__``. What the layer above is then handed is a
    cancellation with no attribution: the env whose ``load_task`` failed is not named anywhere in
    it, and a cancellation is the shape that layer reads as *its own* request rather than as an
    env that could not open a task at all — an env that will fail the same way on every task in
    the queue, reported as a request nobody made.

    Releasing what setup half-built is best-effort by design; the failure being released *from*
    is not. So the cleanup contains every raise it can and the original propagates."""
    env = _env()

    def load_task(idx: Optional[int]) -> Dict[str, Any]:
        raise RuntimeError("this env cannot load a task")

    async def close() -> None:
        raise asyncio.CancelledError("this env cannot be closed")

    env.load_task = load_task  # type: ignore[method-assign]
    env.close = close  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="cannot load a task"):
        await _open(env)


async def test_a_session_close_that_raises_cancellation_does_not_replace_it_either() -> None:
    """The same rule one line up: the MCP sessions setup already opened are released first, and
    a cancellation from one of those must not become the answer either."""
    env = _env()

    def describe(task_id: Any = None) -> Any:
        raise RuntimeError("this env cannot describe its task")

    env.describe = describe  # type: ignore[method-assign]

    opened: List[Any] = []

    class _CancellingSession:
        def __init__(self, inner: Any) -> None:
            self._inner = inner
            opened.append(self)

        async def close(self) -> None:
            await self._inner.close()
            raise asyncio.CancelledError("this session cannot be closed")

        def __getattr__(self, name: str) -> Any:
            return getattr(self._inner, name)

    from shogym.mcp import toolset

    real_factory = toolset._open_session_for_spec

    async def factory(spec: Any, *, session_id: str) -> Any:
        return _CancellingSession(await real_factory(spec, session_id=session_id))

    import shogym.serve.episode as episode_module

    toolset._open_session_for_spec = factory  # type: ignore[assignment]
    episode_module._open_session_for_spec = factory  # type: ignore[assignment]
    try:
        with pytest.raises(RuntimeError, match="cannot describe its task"):
            await _open(env)
        assert opened, "the test never got as far as opening a session"
    finally:
        toolset._open_session_for_spec = real_factory  # type: ignore[assignment]
        episode_module._open_session_for_spec = real_factory  # type: ignore[assignment]


# ----- the evaluator's own output cannot break the record about it -----


async def _submit_and_assert_failed_closed(ep: ServedEpisode, diagnostic: str) -> None:
    """The whole fail-closed commit, asserted the same way for every evaluator that produced no
    usable verdict: a terminal result the caller actually receives, a flagged zero, a durable
    ``FAILED`` record naming the failure, and a torn-down episode that closes clean."""
    result = await ep.call("submit", {"answer": "4", "confidence": 90})

    assert result.terminated is True
    assert json.loads(result.content) == {
        "correct": False,
        "finalize_error": True,
        "confidence": 90,
    }
    assert ep._state is LifecycleState.CLOSED
    assert ep._torn_down is True
    (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
    assert record.status == "FAILED"
    assert record.verdict == {"correct": False, "finalize_error": True, "confidence": 90}
    assert diagnostic in (record.diagnostic or ""), record.diagnostic
    await ep.wait_finalized()


class _UnrenderableFailure(Exception):
    """An evaluator failure that cannot be written down: asking for its message raises."""

    def __str__(self) -> str:
        raise RuntimeError("this failure cannot be rendered")


async def test_an_evaluator_failure_that_cannot_be_rendered_is_still_a_failed_verdict() -> None:
    """``finalize`` raises, and formatting that failure into the private diagnostic raises again.

    The second exception is not the one the handler caught, so it does not stay caught: it walks
    out of the ``except`` carrying the handler's job with it. Everything the fail-closed verdict
    was about to do is skipped — no evidence, no ``FAILED`` record, no teardown — and the
    episode is left sealed-but-unterminated at ``FINALIZING`` while ``close()`` returns clean and
    the durable record sits at ``PENDING``. A run that reports itself intact over an evaluator
    that produced no verdict at all.

    A failure this module has already decided to contain may not be un-contained by the act of
    writing it down. The message is attempted, then the type alone, then a constant."""
    env = _env()

    async def finalize(req: Any) -> Any:
        raise _UnrenderableFailure()

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        await _submit_and_assert_failed_closed(ep, "_UnrenderableFailure")
        # The type survived even though the message did not, so the record still names what
        # failed — and the env's exception text never reaches the agent either way.
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert "unrenderable" in (record.diagnostic or "")
    finally:
        await ep.close()


class _UnreadableVerdict(dict):
    """A verdict that serializes only if you never ask it for its contents."""

    def items(self) -> Any:
        raise RuntimeError("this verdict cannot be read")


async def test_a_verdict_that_cannot_be_serialized_is_a_failed_verdict_not_a_raise() -> None:
    """``finalize`` returns a verdict whose ``items()`` raises.

    A verdict this episode cannot render is a verdict it cannot commit, whatever the reason, and
    the refusal is made once, at the evaluator boundary, rather than by handing the value on and
    checking it again. See the test below for what checking it again bought."""
    env = _env()

    async def finalize(req: Any) -> Any:
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict=_UnreadableVerdict({"correct": True}),
        )

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        await _submit_and_assert_failed_closed(ep, "finalize failed")
    finally:
        await ep.close()


async def test_a_verdict_that_failed_to_render_is_never_handed_on_and_checked_again(
    tmp_path: Path,
) -> None:
    """The rendering used to hand the env's object back when it failed, for the check downstream
    to catch. That check is a *second walk*, so it is a second question, and a stateful container
    can answer the two differently.

    One that refuses the first walk and permits the second was therefore accepted as a verdict,
    on the strength of a walk that succeeded, and rode into the commit as the original object.
    Every later consumer walks it again: the payload the agent is answered with, the durable
    write, the trace event. The one measured here is the durable write, which is best-effort and
    degrades: the run answered ``{"correct": true}`` while the record on disk was left at
    ``PENDING`` with no verdict at all, which is the durable-versus-public disagreement the rest
    of this file exists to prevent. The same ride-through reaches the trace event a step later,
    where the guard catches only ``Exception``, and a cancellation raised there takes the whole
    finalization out with it.

    A value this episode failed to render cannot be rescued by asking it a second time. The
    failure is the answer, and it is the answer at the first asking."""

    class _AnswersTheSecondWalk(dict):
        """Refuses the first walk of itself and permits every one after it."""

        walks = 0

        def items(self) -> Any:
            type(self).walks += 1
            if type(self).walks == 1:
                raise RuntimeError("this verdict refuses the first walk")
            return super().items()

    env = _env()

    async def finalize(req: Any) -> Any:
        return TerminalEvidence(
            source=req.source, status="ok", verdict=_AnswersTheSecondWalk({"correct": True})
        )

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await ServedEpisode.open_env(
        env, env_name="_fixture_score", task=0, trace_path=tmp_path / "run.jsonl"
    )
    try:
        result = await ep.call("submit", {"answer": "4"})

        # The rendering failed, so this is an evaluator that produced no usable verdict.
        assert result.terminated is True
        assert json.loads(result.content) == {"correct": False, "finalize_error": True}
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FAILED"
        assert record.verdict == {"correct": False, "finalize_error": True}
        # What the run reports and what the record holds are the same thing, which is the
        # property the ride-through broke.
        assert ep.terminal_payload == record.verdict
        # And the env's object never reached anything downstream of the boundary.
        assert ep._evidence is not None
        assert type(ep._evidence.verdict) is dict
        await ep.wait_finalized()
    finally:
        await ep.close()


async def test_a_confidence_the_core_cannot_serialize_never_enters_its_own_verdict() -> None:
    """The sweep beside it. The canonical fail-closed verdict echoes the caller's ``confidence``
    for calibration, so the core builds its *own* verdict out of a value it was handed.

    A schema that does not constrain that argument lets a non-JSON value through validation and
    into the replacement, and then the replacement is unserializable for the same reason the
    thing it was replacing was: the commit that has to write it raises, after the seal, with no
    verdict left to fall back to. A value the core cannot serialize is not echoed at all."""

    class _Unserializable:
        pass

    def unconstrain_confidence(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "submit":
                manifest.input_schema["properties"]["confidence"] = {}

    env = _env()
    _describing(env, unconstrain_confidence)

    async def finalize(req: Any) -> Any:
        raise RuntimeError("this evaluator crashed")

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        result = await ep.call(
            "submit", {"answer": "4", "confidence": _Unserializable()}
        )

        assert result.terminated is True
        assert json.loads(result.content) == {"correct": False, "finalize_error": True}
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FAILED"
        assert record.verdict == {"correct": False, "finalize_error": True}
        await ep.wait_finalized()
    finally:
        await ep.close()


# ----- the evidence a finalizer hands back is read once, and then it is the core's -----


class _EvidenceThatAnswersOnce(TerminalEvidence):
    """Evidence shaped like the contract and made of the env's code.

    ``isinstance`` admits a subclass, so ``verdict`` here is a property: it answers the first
    read with the graded verdict and raises on every read after it. Nothing about the object says
    so, and an env with a lazily-built verdict reaches this by accident."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        object.__setattr__(self, "reads", 0)

    @property  # type: ignore[misc]
    def verdict(self) -> Any:
        reads = object.__getattribute__(self, "reads")
        object.__setattr__(self, "reads", reads + 1)
        if reads >= 1:
            raise RuntimeError("this verdict answers once")
        return {"correct": True}

    @verdict.setter
    def verdict(self, value: Any) -> None:
        object.__setattr__(self, "_verdict_slot", value)


async def test_the_evidence_a_finalizer_returns_is_read_once_and_then_it_is_the_core_s() -> None:
    """The commit used to read ``verdict`` twice before trusting it, and ``status`` and
    ``diagnostic`` after that, all outside the guard that turns an evaluator failure into a
    fail-closed verdict. Every one of those reads is the env's code on a subclass the
    ``isinstance`` check admits, and nothing obliges two of them to agree.

    So the check was made against one value and the commit against another. An evidence object
    that answered the first read and raised on the second walked out of the finalization with the
    commit half-made: the durable record left ``PENDING`` with no verdict, the lifecycle at
    ``FINALIZING``, ``terminated`` still false, and ``close()`` clean over it. That is precisely
    the sealed-but-uncommitted shape the rest of this file exists to eliminate, reached through
    the checking code itself.

    Read once, here, and what the transaction commits is what it checked: the answer the env gave
    the one time it was asked."""
    env = _env()
    handed: List[Any] = []

    async def finalize(req: Any) -> Any:
        evidence = _EvidenceThatAnswersOnce(
            source=req.source, status="ok", verdict={"correct": True}
        )
        handed.append(evidence)
        return evidence

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        result = await ep.call("submit", {"answer": "4"})

        # The one answer it gave is the outcome, committed and durable.
        assert result.terminated is True
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
        assert ep._state is LifecycleState.CLOSED
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FINALIZED"
        assert record.verdict == {"correct": True}
        # And it was asked exactly once, which is what makes the two the same value.
        assert object.__getattribute__(handed[0], "reads") == 1
        # What the transaction went on to use is the core's own object, not the env's.
        assert ep._evidence is not None and type(ep._evidence) is TerminalEvidence
        await ep.wait_finalized()
    finally:
        await ep.close()


async def test_evidence_this_episode_cannot_read_is_a_failed_verdict_not_a_raise() -> None:
    """The same reads, failing on the first one instead of the second.

    Reading a field off the env's evidence is the evaluator's code finishing its job, so a field
    that raises is an evaluator that produced no verdict, and it fails closed exactly like one
    that crashed. Both channels are covered: the verdict itself, and ``status`` — the one field
    that *declares* an outcome, which the commit compares against ``"finalize_error"`` and which
    a ``str`` subclass can answer, or raise from, on its own terms."""

    class _UnreadableVerdictField(TerminalEvidence):
        @property  # type: ignore[misc]
        def verdict(self) -> Any:
            raise RuntimeError("this verdict cannot be read at all")

        @verdict.setter
        def verdict(self, value: Any) -> None:
            pass

    class _UnreadableStatus(str):
        def __eq__(self, other: Any) -> bool:
            raise RuntimeError("this status cannot be compared")

        def __hash__(self) -> int:
            return 0

    for build in (
        lambda req: _UnreadableVerdictField(source=req.source, status="ok", verdict={}),
        lambda req: TerminalEvidence(
            source=req.source,
            status=_UnreadableStatus("ok"),  # type: ignore[arg-type]
            verdict={"correct": True},
        ),
    ):
        score_mcp.reset_state()
        env = _env()

        async def finalize(req: Any, _build: Any = build) -> Any:
            return _build(req)

        env.finalize = finalize  # type: ignore[method-assign]
        ep = await _open(env)
        try:
            result = await ep.call("submit", {"answer": "4"})
            assert result.terminated is True
            assert json.loads(result.content) == {"correct": False, "finalize_error": True}
            (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
            assert record.status == "FAILED"
            assert "finalize failed" in (record.diagnostic or ""), record.diagnostic
        finally:
            await ep.close()


async def test_a_terminal_status_this_core_does_not_declare_is_not_a_success() -> None:
    """``status`` is the one channel that declares an outcome, and ``TerminalEvidence`` is a
    plain dataclass, so its ``Literal`` annotation does not check anything at runtime.

    Reading that channel as "failure if it is exactly ``finalize_error``, success otherwise" made
    every value an env could put there a *success*: a finalizer returning
    ``status="not-a-terminal-status"`` with ``correct=True`` was accepted, published as
    ``{"correct": true, "finalize_error": false}`` and recorded ``FINALIZED`` with no diagnostic.
    An evaluator that cannot say which of the two outcomes it reached has not reached one, and
    reading its silence as the good one is the definition of failing open.

    Exactly the two strings this core declares are accepted. Anything else is the evaluator
    breaking its own contract, and takes the same fail-closed route as one that crashed."""
    for status in ("not-a-terminal-status", "", "OK", "error", None, 0, True):
        score_mcp.reset_state()
        env = _env()

        async def finalize(req: Any, _status: Any = status) -> Any:
            return TerminalEvidence(
                source=req.source,
                status=_status,  # type: ignore[arg-type]
                verdict={"correct": True},
            )

        env.finalize = finalize  # type: ignore[method-assign]
        ep = await _open(env)
        try:
            result = await ep.call("submit", {"answer": "4"})
            assert json.loads(result.content) == {
                "correct": False,
                "finalize_error": True,
            }, f"status={status!r} was read as an outcome"
            (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
            assert record.status == "FAILED", f"status={status!r} was recorded as a success"
            assert "terminal status" in (record.diagnostic or ""), record.diagnostic
        finally:
            await ep.close()


async def test_the_two_statuses_this_core_declares_are_still_honoured() -> None:
    """The other side of that line, so the check is a gate and not a wall: both legal values
    still mean what the env said they mean."""
    for status, expected in (("ok", False), ("finalize_error", True)):
        score_mcp.reset_state()
        env = _env()

        async def finalize(req: Any, _status: Any = status) -> Any:
            return TerminalEvidence(
                source=req.source, status=_status, verdict={"correct": not _status == "ok"}
            )

        env.finalize = finalize  # type: ignore[method-assign]
        ep = await _open(env)
        try:
            result = await ep.call("submit", {"answer": "4"})
            assert json.loads(result.content)["finalize_error"] is expected
            (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
            assert record.status == ("FAILED" if expected else "FINALIZED")
        finally:
            await ep.close()


async def test_a_status_that_impersonates_a_declared_one_is_not_one() -> None:
    """The gate that rejects an undeclared status compared the env's object with ``==``, and the
    env's object is on the dispatching side of that comparison whichever way it is written: for a
    ``str`` subclass Python offers the reflected operation first, and for a non-``str`` there is
    nothing else to offer. An object whose ``__eq__`` returns ``other == "ok"`` therefore
    impersonated the literal and was rewritten into this core's success, so an undeclared status
    still became a clean ``FINALIZED`` outcome, just through equality instead of a plain string.

    The value has to *be* one of the declared strings, and what the transaction keeps is this
    core's copy of it and never the object that claimed to match."""

    class _PretendsToBeOK:
        def __eq__(self, other: Any) -> bool:
            return other == "ok"

        def __hash__(self) -> int:
            return hash("ok")

    class _SubclassSayingOK(str):
        def __eq__(self, other: Any) -> bool:
            return True

        def __hash__(self) -> int:
            return hash("ok")

    for impostor in (_PretendsToBeOK(), _SubclassSayingOK("nonsense")):
        score_mcp.reset_state()
        env = _env()

        async def finalize(req: Any, _status: Any = impostor) -> Any:
            return TerminalEvidence(
                source=req.source,
                status=_status,  # type: ignore[arg-type]
                verdict={"correct": True},
            )

        env.finalize = finalize  # type: ignore[method-assign]
        ep = await _open(env)
        try:
            result = await ep.call("submit", {"answer": "4"})
            assert json.loads(result.content) == {
                "correct": False,
                "finalize_error": True,
            }, f"{type(impostor).__name__} was accepted as an outcome"
            (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
            assert record.status == "FAILED"
            assert "terminal status" in (record.diagnostic or ""), record.diagnostic
        finally:
            await ep.close()


async def test_a_confidence_is_carried_as_its_rendering_and_not_as_the_object() -> None:
    """The confidence guard repeated the check-then-use the verdict rendering had just given up:
    it asked the object whether it serialized and then put the *object* into the core's own
    fail-closed verdict.

    The question is a walk and the use is another walk, so a container that permits the first and
    refuses the second is admitted by one and committed by the other. Measured with a dict that
    tolerates the two argument-digest walks and the check and refuses the next: the call raised at
    terminal-step construction, the durable record stayed ``PENDING`` with no verdict, and the
    payload in memory carried the failure verdict. That is the durable-versus-public split-brain
    this transaction exists to make impossible, reached through the guard meant to prevent it.

    The rendering is what is carried, so what the verdict holds is plain data and no later walk
    can disagree with the one that admitted it."""

    # One walk digests the arguments at the seal, one renders this value. A submission is not
    # asked anything after that, so a value that refuses from there on must cost nothing.
    walks = [0]

    class _RefusesEveryLaterWalk(dict):
        def items(self) -> Any:
            walks[0] += 1
            if walks[0] >= 3:
                raise RuntimeError("confidence changed after validation")
            return super().items()

    def unconstrain_confidence(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "submit":
                manifest.input_schema["properties"]["confidence"] = {}

    env = _env()
    _describing(env, unconstrain_confidence)

    async def finalize(req: Any) -> Any:
        raise RuntimeError("this evaluator crashed")

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        result = await ep.call(
            "submit", {"answer": "4", "confidence": _RefusesEveryLaterWalk({"v": 1})}
        )

        assert result.terminated is True
        assert json.loads(result.content) == {
            "correct": False,
            "finalize_error": True,
            "confidence": {"v": 1},
        }
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FAILED"
        # The record and the run agree, and what both hold is plain data.
        assert record.verdict == json.loads(result.content)
        assert ep._evidence is not None
        assert type(ep._evidence.verdict["confidence"]) is dict
        await ep.wait_finalized()
    finally:
        await ep.close()


async def test_a_schema_key_that_answers_every_comparison_cannot_excuse_a_blank_answer() -> None:
    """The sweep beside the three. A schema that would not render is kept as the env published
    it, so the keys inside it are the env's objects, and the terminal-argument check compares
    them: once against the reserved argument the transport injects, once against the JSON type
    the non-blank rule is about.

    A key that answers True to the first excuses itself from the only rule standing between a
    blank submission and a seal. Both comparisons are matches against the strings this core
    declares now, so a key that merely *claims* to be one is not one."""

    class _AnswersEveryComparison(str):
        def __eq__(self, other: Any) -> bool:
            return True

        def __hash__(self) -> int:
            return str.__hash__(self)

    def rig(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "submit":
                manifest.input_schema = {
                    **manifest.input_schema,
                    "required": [_AnswersEveryComparison("answer")],
                }

    env = _env()
    _describing(env, rig)
    ep = await _open(env)
    try:
        # Closed twice over. What this episode enforces is the *rendering* of what the env
        # published, and rendering a key flattens the subclass away, so the object never reaches
        # the check at all. The match inside the check is what would answer one that did.
        (key,) = ep._score_schemas["submit"]["required"]
        assert type(key) is str, "the enforced schema kept an object the env supplied"

        result = await ep.call("submit", {"answer": "   "})
        assert result.terminated is False, "a blank submission sealed the episode"
        assert json.loads(result.content)["validation_error"] is True
        assert ep._state is LifecycleState.OPEN
    finally:
        await ep.close()


async def test_an_env_raised_cancellation_from_the_serializer_is_an_ordinary_failure() -> None:
    """The feedback serializer is deliberately left to raise, because the layer above owns that
    failure: it redacts the agent's answer, files the row ``finalize_error`` with the failure
    named on it, and stops the run. That is right for an ordinary ``Exception``, and it was the
    whole of the deliberate part.

    ``CancelledError`` is not an ordinary exception. Letting it out marks the finalization *task*
    cancelled, and a cancelled task is control flow rather than a failure, so every owner that
    was going to answer for it steps aside instead: ``call()``, ``wait_finalized()`` and every
    ``close()`` raised cancellation, the stream saw the env fail and then lost the row to the same
    cancelled future, ``env.close()`` was never reached, and the durable record stayed ``PENDING``.

    Nothing is awaited in that serializer read, so a cancellation observed there was raised by the
    env's own value and is not one requested against this task, which is the distinction this
    module already draws everywhere else. It is translated into an ordinary failure carrying the
    same information, contained once at the boundary where the foreign code runs rather than at
    every awaiter forever, and the ordinary-``Exception`` semantics above it are untouched
    (``test_a_forced_abort_the_env_fails_is_not_an_earned_give_up`` still pins those)."""
    from shogym.types import EpisodeFeedback, FeedbackCollection

    class _NameRaisesCancellation(EpisodeFeedback):
        def __getattribute__(self, attr: str) -> Any:
            if attr == "name":
                raise asyncio.CancelledError("this feedback name cannot be read")
            return super().__getattribute__(attr)

    env = _env()
    closed: List[bool] = []
    real_close = env.close

    async def close() -> None:
        closed.append(True)
        await real_close()

    def verify(trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None) -> Any:
        collection = FeedbackCollection()
        if terminated:
            collection.episode.append(_NameRaisesCancellation(name="correct", value=True))
        return collection

    env.verify = verify  # type: ignore[method-assign]
    env.close = close  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        # A failure the layer above owns, and an ordinary one, so nothing downstream reads it as
        # a request to stop waiting.
        with pytest.raises(RuntimeError, match="cannot record"):
            await ep.call("submit", {"answer": "4"})
        assert "CancelledError" in str(ep._finalization.exception())  # type: ignore[union-attr]
        await ep.wait_finalized()  # must not raise
    finally:
        await ep.close()  # must not raise
    assert closed == [True], "the run could not even close itself"


async def test_terminal_feedback_is_rendered_once_and_every_sink_reads_the_rendering(
    tmp_path: Path,
) -> None:
    """The guarded render was one of three serializations of the same env objects.

    A feedback item is the env's object, and this module asks it for its contents again in every
    sink: the retained terminal feedback, the trace row, and the in-band sidecar the caller is
    answered with. Three serializations are three questions, and nothing obliges an env to answer
    them alike. A name that answered the first read and raised cancellation on the second let the
    guarded render succeed, the evidence commit, the durable record land and the teardown run,
    and then took the finalization out from under a verdict that was already committed and
    already public: `finalization.cancelled()` true, `wait_finalized()` and `close()` both
    raising, with `terminal_payload` still reading `correct=True`.

    So the feedback is rendered once and rebuilt as this module's own items, and every sink reads
    the rebuild. That **reverses** the decision documented earlier in this file to leave feedback
    items as the env's objects: that reasoning held that a scribbler deceives only its own later
    reads, and it did not account for this module doing the later reads. Who owns a feedback
    *failure* is unchanged, and the tests that pin it are untouched."""
    reads = [0]

    class _NameAnswersOnce(EpisodeFeedback):
        def __getattribute__(self, attr: str) -> Any:
            if attr == "name":
                reads[0] += 1
                if reads[0] >= 2:
                    raise asyncio.CancelledError("this feedback name answers once")
            return super().__getattribute__(attr)

    env = _env()

    def verify(trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None) -> Any:
        collection = FeedbackCollection()
        if terminated:
            collection.episode.append(_NameAnswersOnce(name="correct", value=True))
        return collection

    env.verify = verify  # type: ignore[method-assign]
    ep = await ServedEpisode.open_env(
        env, env_name="_fixture_score", task=0, trace_path=tmp_path / "run.jsonl"
    )
    try:
        result = await ep.call("submit", {"answer": "4"})

        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
        assert ep._finalization is not None and not ep._finalization.cancelled()
        assert ep.terminal_feedback == [
            {"name": "correct", "value": True, "level": "episode"}
        ]
        # The env was asked exactly once, which is what makes every sink the same value.
        assert reads[0] == 1, reads
        # And the trace carries that one value too.
        steps = [r for r in load_traces(tmp_path / "run.jsonl") if "feedback" in r]
        assert steps[-1]["feedback"] == ep.terminal_feedback
        await ep.wait_finalized()
    finally:
        await ep.close()


async def test_the_legacy_terminal_step_reads_its_feedback_once_too(tmp_path: Path) -> None:
    """The same rule on the path a non-seal env takes, where the terminal step is the one that
    reads its feedback three times: the retained terminal feedback, the trace row, and the
    in-band sidecar that carries episode feedback out at the end.

    Held to a value that answers differently on each read, so a second reading is visible rather
    than merely possible: what the record says and what the agent is told have to be the same
    value, and they are the same value only because there is one reading of it."""
    import shogym

    reads = [0]

    class _ValueDrifts(EpisodeFeedback):
        def __getattribute__(self, attr: str) -> Any:
            if attr == "value":
                reads[0] += 1
                return reads[0]
            return super().__getattribute__(attr)

    env = shogym.make("wordle_v1")

    def verify(trajectory: Any, task: Any, *, terminated: bool) -> Any:
        collection = FeedbackCollection()
        if terminated:
            collection.episode.append(_ValueDrifts(name="check_answer", value=0))
        return collection

    env.verify = verify  # type: ignore[method-assign]
    ep = await ServedEpisode.open_env(
        env, env_name="wordle_v1", task=0, trace_path=tmp_path / "run.jsonl"
    )
    try:
        end = await ep.call("terminate")
        assert end.terminated is True

        items, _ = parse_meta(end.meta)
        sidecar = [{"name": i.name, "value": i.value, "level": "episode"} for i in items]
        steps = [r for r in load_traces(tmp_path / "run.jsonl") if "feedback" in r]
        # One value, three sinks. The env was asked once, so they cannot disagree.
        assert reads[0] == 1, reads
        assert ep.terminal_feedback == steps[-1]["feedback"] == sidecar
    finally:
        await ep.close()


@pytest.mark.parametrize("field", ["name", "value"])
async def test_the_terminal_feedback_this_episode_keeps_is_a_rendering_not_an_object(
    tmp_path: Path, field: str
) -> None:
    """Reading each field once is not the same as rendering it.

    The render read ``name``, ``value`` and ``step`` exactly once each and then put *those
    objects* into a new dict: the container was the core's and everything in it was still the
    env's. The models are mutable and do not validate on assignment, so an env can hang a ``str``
    subclass on a validated item, and that subclass was what the retained terminal feedback held
    while the trace row and the sidecar read the rebuilt items. Measured through a stream: a
    terminal ``correct`` whose name answered every comparison false was the object the row's
    headline is picked with, so a task the agent solved was filed sealed, with the evidence
    intact and ``score.success`` null. A valid answer, recorded without its headline, over a
    scalar nobody had to be able to read.

    So both halves come out of one rendering, and what this episode keeps is plain data: the
    subclass is gone by the time anything asks the value a question."""

    class _Deaf(str):
        """A JSON string on the wire that answers every comparison false."""

        def __eq__(self, other: object) -> bool:
            return False

        def __ne__(self, other: object) -> bool:
            return True

        __hash__ = str.__hash__

    env = _env()

    def verify(trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None) -> Any:
        collection = FeedbackCollection()
        if terminated:
            item = EpisodeFeedback(name="correct", value=True)
            # Assigned rather than constructed: pydantic coerces the subclass away when the
            # model is built, and the models do not validate on assignment.
            setattr(item, field, _Deaf("correct") if field == "name" else _Deaf("yes"))
            collection.episode.append(item)
        return collection

    env.verify = verify  # type: ignore[method-assign]
    ep = await ServedEpisode.open_env(
        env, env_name="_fixture_score", task=0, trace_path=tmp_path / "run.jsonl"
    )
    try:
        await ep.call("submit", {"answer": "4"})

        (kept,) = ep.terminal_feedback
        assert kept == {
            "name": "correct",
            "value": True if field == "name" else "yes",
            "level": "episode",
        }
        # The equalities above are the point, and they only mean anything because the value
        # answering them is this module's: the env's object answers every one of them false.
        assert type(kept[field]) is str, "the env's object was retained, not its rendering"
        # And the retained half and the rebuilt half are the same data, from one reading.
        steps = [r for r in load_traces(tmp_path / "run.jsonl") if "feedback" in r]
        assert steps[-1]["feedback"] == ep.terminal_feedback
        await ep.wait_finalized()
    finally:
        await ep.close()


class _NameRaisesCancellationAtTheRender(EpisodeFeedback):
    """A feedback item whose own name raises the one exception a task cannot own."""

    def __getattribute__(self, attr: str) -> Any:
        if attr == "name":
            raise asyncio.CancelledError("this feedback name cannot be read")
        return super().__getattribute__(attr)


def _unwireable_item() -> EpisodeFeedback:
    item = EpisodeFeedback(name="correct", value=True)
    item.value = object()  # the models do not validate on assignment
    return item


@pytest.mark.parametrize(
    ("build", "raised"),
    [
        pytest.param(_unwireable_item, ValueError, id="a value the wire refuses"),
        pytest.param(
            lambda: _NameRaisesCancellationAtTheRender(name="correct", value=True),
            RuntimeError,
            id="a read that raises cancellation",
        ),
    ],
)
async def test_feedback_the_wire_refuses_still_commits_the_finalization(
    build: Any, raised: type
) -> None:
    """Who owns a feedback failure is unchanged. What it may not do is leave the record half-made.

    The serializer failure is deliberately the layer above's: a stream catches it out of the
    terminating call, files the row ``finalize_error`` with ``score=None`` and stops the run,
    because an env whose feedback cannot be serialized will fail the same way on every task behind
    this one. That policy is about the *row*. The record underneath it is this module's, and on
    the transport-independent path there is no layer above at all: ``evaluate()`` and
    ``run_stdio()`` drive this class directly.

    Raising before the commit therefore reproduced this PR's own defect one boundary later. The
    episode ended CLOSED and terminated, holding the evaluator's ``ok`` verdict in memory, while
    the durable record still said ``PENDING`` with no verdict at all: the answer the run gave and
    the record it left disagreed, and recovery would have resolved that disagreement the other
    way round.

    So the finalization commits first, fail-closed and attributed, and the failure is handed on
    after. Both directions the boundary can take are held here, because the translated one is a
    second path to the same commit."""
    env = _env()

    def verify(trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None) -> Any:
        collection = FeedbackCollection()
        if terminated:
            collection.episode.append(build())
        return collection

    env.verify = verify  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        with pytest.raises(raised):
            await ep.call("submit", {"answer": "4"})

        # The commit the seal owed: durable, fail-closed, and saying whose fault it was.
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]  # type: ignore[union-attr]
        assert record.status == "FAILED", "a sealed episode was left with nothing committed"
        assert record.verdict == fail_closed_verdict(None)
        assert "cannot record" in (record.diagnostic or "")
        # And the evidence this episode holds says the same thing the record does, so a caller
        # with no stream to compose a row for it still has an outcome it can read.
        assert ep._evidence is not None and ep._evidence.finalize_error is True
        assert ep.terminal_payload == {"correct": False, "finalize_error": True}
        assert ep.terminal_feedback == []
        # The lifecycle ended rather than stranding, which it did before this too: the state was
        # never the half that was missing.
        assert ep._state is LifecycleState.CLOSED
        await ep.wait_finalized()  # must not raise: the verdict is committed
    finally:
        await ep.close()
    assert ep.terminated is True


async def test_a_refusal_interrupted_by_its_own_cleanup_still_reaches_the_run(
    tmp_path: Path,
) -> None:
    """The other window on the refusal path, and the one the stream cannot see into.

    A contract refusal raised while the episode is still being built is caught by ``open_env``,
    which releases the sessions and the env it took ownership of before re-raising. Those
    releases are awaits. A pull cancelled during one of them gets its cancellation, which is
    right and stays right, and the refusal that cancellation *replaced* was gone before anything
    could classify it: the stop was never latched, the position stayed owed, and the run closed
    clean over an env that cannot publish a contract at all.

    The rule this file already holds is that a classification of something already discovered
    runs before anything cancellable. A release that has to happen cannot be moved above the
    discovery, so the discovery is carried across it instead, and the layer above reads it with
    ``contract_refusal`` rather than by type."""
    from shogym.serve.episode import contract_refusal

    blocked = asyncio.Event()
    armed: List[bool] = []

    def factory(_name: str) -> Any:
        built = _env()
        if armed:

            def describe(task_id: Any = None) -> Any:
                raise RuntimeError("this env cannot describe a task")

            async def close() -> None:
                await blocked.wait()

            built.describe = describe  # type: ignore[method-assign]
            built.close = close  # type: ignore[method-assign]
        return built

    stream = TaskStream(factory, [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov")
    armed.append(True)
    pull = asyncio.ensure_future(stream.get_task())
    # The refusal is found inside `open_env`, which then blocks releasing the env, so the stop
    # cannot have been latched yet: this window is the one the stream never sees.
    for _ in range(200):
        await asyncio.sleep(0.005)
        if blocked._waiters:  # type: ignore[attr-defined]
            break
    assert not stream.stopped, "the test never reached the window it is about"
    pull.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await pull

    # The caller's cancellation is still the caller's cancellation, and it carries the finding.
    carried = contract_refusal(cancelled.value)
    assert carried is not None, "the refusal was replaced by the cancellation that interrupted it"
    assert "cannot describe a task" in str(carried)
    assert stream.stopped, "a fault the run had already found was lost to its own cleanup"
    assert stream.queue_info() == QueueInfo(remaining=1, consumed=0, in_flight=0)
    assert stream.results == ()
    blocked.set()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()


async def test_a_cleanup_its_caller_cancelled_still_releases_what_it_took(
    tmp_path: Path,
) -> None:
    """The other half of that window, and the one the finding travelling across it does not fix.

    Ownership transfers at the call: ``open_env`` closes the sessions it opened and the env it was
    handed when setup fails, and that promise is the only reason a caller may hand over a fresh
    env per episode. Run inline in the caller's task, the promise was worth exactly as much as the
    caller's patience. A cancellation delivered while ``env.close()`` was in flight re-raised out
    of the middle of the release, and nothing was left holding the env: it was never closed, its
    per-episode state was never dropped, and no episode was ever returned or registered, so
    ``aclose()`` had nothing to recover it from.

    A cancellation of the *waiter* cannot end a task, so the release is one and the join is
    shielded. The caller is still cancelled, the stop the refusal owes is still latched, and the
    release finishes on its own."""
    blocked = asyncio.Event()
    closed: List[bool] = []
    armed: List[bool] = []

    def factory(_name: str) -> Any:
        built = _env()
        if armed:
            real_close = built.close

            def describe(task_id: Any = None) -> Any:
                raise RuntimeError("this env cannot describe a task")

            async def close() -> None:
                await blocked.wait()
                await real_close()
                closed.append(True)

            built.describe = describe  # type: ignore[method-assign]
            built.close = close  # type: ignore[method-assign]
        return built

    stream = TaskStream(factory, [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov")
    armed.append(True)
    pull = asyncio.ensure_future(stream.get_task())
    for _ in range(200):
        await asyncio.sleep(0.005)
        if blocked._waiters:  # type: ignore[attr-defined]
            break
    assert blocked._waiters, "the test never reached the release it is about"  # type: ignore[attr-defined]
    # `begin_session` ran before `describe` refused, so the env holds per-episode state that only
    # its own `close` drops.
    assert len(score_mcp._sessions) == 1, "the test never reached the state it is about"

    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull
    assert stream.stopped, "the refusal is still latched before anything cancellable"
    assert not closed, "the release cannot have finished while its env is still blocked"

    # The caller is gone and the release is not: it is the only owner of these resources, and it
    # runs to completion behind the call that started it.
    blocked.set()
    for _ in range(200):
        await asyncio.sleep(0.005)
        if closed:
            break
    assert closed == [True], "the env this call took ownership of was never released"
    assert score_mcp._sessions == {}, "the env's per-episode state outlived the episode"
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()


async def test_a_caller_cancelled_mid_finalization_is_still_a_cancelled_caller() -> None:
    """The other direction, so the translation above is scoped to the serializer and not to
    cancellation in general: a caller cancelled while its terminal call is finalizing is still
    cancelled, the single finalization keeps running because every awaiter shields it, and the
    verdict it reached is there for ``close()`` to collect."""
    release = asyncio.Event()
    runs: List[bool] = []
    env = _env()

    async def finalize(req: Any) -> Any:
        runs.append(True)
        await release.wait()
        return TerminalEvidence(source=req.source, status="ok", verdict={"correct": True})

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        call = asyncio.ensure_future(ep.call("submit", {"answer": "4"}))
        for _ in range(200):
            await asyncio.sleep(0.005)
            if ep.sealed:
                break
        assert ep.sealed, "the call never reached the seal"
        call.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call
        release.set()
        await ep.wait_finalized()
        # The evaluation the cancelled caller started still finished, exactly once.
        assert ep.terminal_payload == {"correct": True, "finalize_error": False}
        assert runs == [True], "the evaluation ran twice, or not at all"
    finally:
        await ep.close()
    (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
    assert record.status == "FINALIZED"


# ----- the snapshot is one contract, and taking it is not the env's decision -----


_COPIES: Dict[str, int] = {}


class _DriftingText(str):
    """Advisory contract text whose *copy* is a different answer every time it is asked for."""

    def __deepcopy__(self, memo: Any) -> Any:
        _COPIES["drift"] = _COPIES.get("drift", 0) + 1
        return _DriftingText(f"read-{_COPIES['drift']}")


def _describing(env: Any, mutate: Any) -> None:
    """Have ``env`` publish the contract it would have, with one field replaced."""
    real = env.describe

    def describe(task_id: Any = None) -> Any:
        spec = real(task_id)
        mutate(spec)
        return spec

    env.describe = describe  # type: ignore[method-assign]


async def test_every_reader_of_the_contract_is_shown_the_same_contract() -> None:
    """``describe()`` promises the same snapshot to every reader for as long as the episode
    lives, and then takes a fresh deep copy per read to keep readers from reaching the original.

    Copying is the one step that runs code the env wrote. The tool manifests are round-tripped
    through JSON before the copy, so the env's objects are gone from *those* — but the contract's
    advisory fields are not, and a value left there decides what each reader is shown. Whoever
    frames the agent gets one contract, the check that compares the manifest gets another, and
    the episode enforces a third. That is the exact failure the snapshot was introduced to close,
    reintroduced by the defence itself: the agent sends what it was shown and the record says it
    answered wrong.

    So the whole contract is normalized before it is copied, not just the tools, and the copy a
    reader gets is taken from data rather than from anything the env still owns."""
    _COPIES.clear()
    env = _env()
    _describing(env, lambda spec: object.__setattr__(spec, "instructions", _DriftingText("go")))
    ep = await _open(env)
    try:
        first = ep.describe()
        second = ep.describe()
        assert first.instructions == second.instructions, "two readers, two contracts"
        # And what they are shown is what the episode holds — not a third value.
        assert first.instructions == ep._spec.instructions
        # The env's copy code is not what answers a read of the published contract.
        before = _COPIES.get("drift", 0)
        ep.describe()
        assert _COPIES.get("drift", 0) == before, "reading the contract ran the env's code"
    finally:
        await ep.close()


class _CopiesIntoADifferentContract(TaskSpec):
    """A contract that renders perfectly and answers every copy of itself differently."""

    def model_copy(self, **kwargs: Any) -> Any:
        _COPIES["contract"] = _COPIES.get("contract", 0) + 1
        out = super().model_copy(**kwargs)
        object.__setattr__(out, "instructions", f"copy-{_COPIES['contract']}")
        return out


async def test_a_contract_that_copies_into_a_different_contract_cannot() -> None:
    """Normalizing every field did not make the contract core-owned, because the object holding
    them was still the env's.

    ``describe`` hands out a copy per reader and the copy was taken with ``spec.model_copy``,
    which is a *method*: a subclass can override it, return a perfectly serializable contract, and
    return a different one every time without ever raising. Containment answers a copy that fails.
    Nothing answers a copy that succeeds and lies. Measured on the shape below: the episode stored
    ``copy-2``, the first reader was shown ``copy-3``, the second ``copy-4`` — one enforced
    contract and a different framing per read, which is the publish-one-enforce-another failure
    the whole snapshot exists to close, arrived at through the copy that closes it.

    So the class is not kept either. The contract is rebuilt here out of one reading of each
    field, and the copies readers get are pydantic's own method on this module's own model."""
    _COPIES.clear()
    env = _env()
    real = env.describe

    def describe(task_id: Any = None) -> Any:
        return _CopiesIntoADifferentContract.model_construct(**dict(real(task_id)))

    env.describe = describe  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        assert type(ep._spec) is TaskSpec, "the snapshot kept the class the env published"
        first = ep.describe()
        second = ep.describe()
        assert type(first) is TaskSpec
        assert first.instructions == second.instructions == ep._spec.instructions
        assert _COPIES.get("contract", 0) == 0, "a copy of the contract ran the env's code"
    finally:
        await ep.close()


class _UncarriableInstructions:
    """Contract text the wire cannot carry, and which therefore has no place in a contract."""


async def test_the_published_contract_holds_no_object_the_env_supplied() -> None:
    """The invariant the rebuild exists to give, stated positively and checked exhaustively.

    Every value the episode publishes and enforces is a builtin this module produced from a
    rendering, or one of the literals it declares. Not "detached because a copy of it returned",
    which proves only that the copy returned; not "normalized for the fields we listed", which is
    an inventory rather than a property. A contract whose every field is a hostile subclass opens
    an ordinary episode, and the subclasses are simply not in it."""

    class _Text(str):
        pass

    class _Number(int):
        pass

    def mutate(spec: Any) -> None:
        object.__setattr__(spec, "instructions", _Text("go"))
        object.__setattr__(spec, "env_name", _Text("_fixture_score"))
        object.__setattr__(spec, "task_id", _Text("0"))
        object.__setattr__(spec, "horizon", _Number(3))
        object.__setattr__(spec, "contract_version", _Number(2))
        for manifest in spec.tools:
            manifest.name = _Text(str(manifest.name))
            manifest.description = _Text(str(manifest.description))
            manifest.input_schema = dict(manifest.input_schema)
            manifest.input_schema["title"] = _Text("t")

    env = _env()
    _describing(env, mutate)
    ep = await _open(env)
    try:

        def plain(value: Any) -> None:
            assert type(value) in (str, int, float, bool, dict, list, type(None)), value
            if isinstance(value, dict):
                for key, item in value.items():
                    plain(key)
                    plain(item)
            elif isinstance(value, list):
                for item in value:
                    plain(item)

        spec = ep._spec
        assert type(spec) is TaskSpec
        for field in ("env_name", "task_id", "instructions", "horizon", "contract_version"):
            plain(getattr(spec, field))
        for manifest in spec.tools:
            assert type(manifest) is ToolManifest
            for field in ("name", "description", "input_schema", "provenance", "terminal_kind"):
                plain(getattr(manifest, field))
        for template in spec.reference_templates:
            for field in ("role", "template", "variables_schema"):
                plain(getattr(template, field))

        # And it is an ordinary episode: the contract was normalized, not refused.
        assert ep.seal_enabled is True
        result = await ep.call("submit", {"answer": "4"})
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
    finally:
        await ep.close()


async def test_a_contract_field_the_wire_cannot_carry_is_refused_in_the_env_s_name() -> None:
    """A field of the published contract that will not go on the wire at all.

    Refusing to open the episode is right — no task is dispensed and no row is written. What is
    not right is refusing in the env's own words: the exception walks out of ``open_env``
    unattributed, and the layer above is handed a failure with nothing in it naming the env that
    could not publish a contract, for an env that will fail identically on every task queued
    behind this one.

    So the refusal is the serve layer's, it names the env, it says which kind of failure it is,
    and the env's failure rides along as the cause rather than as the whole answer."""
    env = _env()
    _describing(
        env,
        lambda spec: object.__setattr__(spec, "instructions", _UncarriableInstructions()),
    )
    with pytest.raises(TaskContractError, match="_fixture_score.*cannot take as a contract") as refused:
        await _open(env)
    # The env's own failure is named rather than swallowed, so an operator is pointed at it.
    assert "could not be put on the wire" in str(refused.value)
    assert "_UncarriableInstructions" in str(refused.value)


class _SchemaThatCannotBeWalked(dict):
    """An advertised schema that serializes only if you never ask it for its contents."""

    def items(self) -> Any:
        raise RuntimeError("schema walk exploded")


def _publishing_an_unwalkable_schema(env: Any) -> None:
    def mutate(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "submit":
                manifest.input_schema = _SchemaThatCannotBeWalked(manifest.input_schema)

    _describing(env, mutate)


async def test_a_schema_the_round_trip_cannot_walk_is_the_same_attributed_refusal() -> None:
    """The normalization round trip serializes what the env published, and serializing walks the
    env's containers by calling their methods, so it can raise anything the env likes.

    Catching only the two raises ``json.dumps`` makes on its own behalf let every other one out
    of normalization raw, above the snapshot that attributes a contract failure: the caller got
    the env's bare exception with ``__cause__`` unset and nothing naming the env. A value that
    declines to serialize and one that explodes while being asked are the same fact about the
    contract, so they get the same answer, and it is the answer the layer above already knows how
    to classify."""
    env = _env()
    _publishing_an_unwalkable_schema(env)
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert "schema walk exploded" in str(refused.value), "the refusal must name what refused it"
    assert isinstance(refused.value.__cause__, ValueError)


class _UnwalkableButCopyable(dict):
    """A schema that will not serialize, however you ask, and that copies as itself.

    The copy is the point: it is what "detached" used to be established by. A value can refuse
    the wire, hand back *itself* from ``__deepcopy__`` without raising, and be treated as a
    snapshot for having done so."""

    def items(self) -> Any:
        raise RuntimeError("schema walk exploded")

    def __deepcopy__(self, memo: Any) -> Any:
        return self


async def test_a_schema_that_will_not_serialize_is_refused_here_and_not_deferred_upward() -> None:
    """The policy this reverses, and why.

    A schema that would not serialize used to be kept exactly as the env published it, with the
    decision to refuse deferred to the layer that could compare it against something. That reads
    well and holds only where a layer above exists. This class is also the transport-independent
    engine that ``evaluate()`` and ``run_stdio()`` drive directly, and there the kept object was
    the whole story: it stayed the env's, its ``__deepcopy__`` was taken as proof of detachment
    for returning without raising, and the episode went on to advertise one contract through
    ``describe()`` while its seal enforced another. The agent sends exactly what it was shown and
    is told it is invalid.

    So the rule is one rule now, and it is the rule every other boundary here already follows: a
    value this layer cannot put on the wire is a contract it cannot publish, refused where it is
    found and attributed to the env that published it. A copy that returns is not proof it
    returned something detached; only rebuilding from the data a successful walk produced is."""
    env = _env()

    def mutate(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "submit":
                manifest.input_schema = _UnwalkableButCopyable(manifest.input_schema)

    _describing(env, mutate)
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert "schema walk exploded" in str(refused.value)


class _SpecWhoseToolsCannotBeRead(TaskSpec):
    """The published contract as an object that will not say what it advertises."""

    def __getattribute__(self, name: str) -> Any:
        if name == "tools":
            raise RuntimeError("tools cannot be read")
        return super().__getattribute__(name)


def _publishing_unreadable_tools(env: Any) -> None:
    real = env.describe

    def describe(task_id: Any = None) -> Any:
        return _SpecWhoseToolsCannotBeRead.model_construct(**dict(real(task_id)))

    env.describe = describe  # type: ignore[method-assign]


async def test_a_tool_collection_that_cannot_be_read_is_the_same_attributed_refusal() -> None:
    """Normalizing each advertised tool is guarded; getting the list of them to normalize was not.

    The published contract is the env's object and ``tools`` is an attribute of it, so reading
    and iterating that collection is the env's code exactly like reading one manifest's schema
    is. Unguarded, it left normalization as the env's bare exception with ``__cause__`` unset,
    above the point that attributes a contract failure, so the layer above got a failure with
    nothing in it naming the env and nothing it could classify.

    It cannot be answered the way an unreadable *advisory* field is either. Leaving that one
    alone is honest, because the layer above still reads it and refuses. There is nothing to
    leave alone here: the collection is the contract this episode enforces, and an empty stand-in
    would publish a task with no tools and no scoring terminal, run it down the legacy path, and
    record whatever came out."""
    env = _env()
    _publishing_unreadable_tools(env)
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert "tools cannot be read" in str(refused.value)
    assert isinstance(refused.value.__cause__, RuntimeError)


async def test_the_env_is_asked_for_each_part_of_its_contract_exactly_once() -> None:
    """The same read, counted. The snapshot used to keep the class the env published, so the
    collection stayed the env's to answer for after normalization had finished with it, and this
    episode read it again to find the scoring terminal it enforces. An env that answered the
    first read and refused the second raised from the middle of construction.

    Containing that second read was the smaller half of the answer. The larger one is that there
    is no second read: the contract is rebuilt out of one reading of each part, so the number of
    questions this layer asks the env about its contract is fixed by this code and not by how
    many times something downstream happens to look."""
    counts: Dict[str, int] = {}

    class _Counting(TaskSpec):
        def __getattribute__(self, name: str) -> Any:
            if not name.startswith("_"):
                counts[name] = counts.get(name, 0) + 1
            return super().__getattribute__(name)

    class _CountingManifest(ToolManifest):
        def __getattribute__(self, name: str) -> Any:
            if not name.startswith("_"):
                counts[f"tool.{name}"] = counts.get(f"tool.{name}", 0) + 1
            return super().__getattribute__(name)

    env = _env()
    real = env.describe

    def describe(task_id: Any = None) -> Any:
        spec = real(task_id)
        fields = dict(spec)
        fields["tools"] = [
            _CountingManifest.model_construct(**dict(m)) for m in fields["tools"]
        ]
        return _Counting.model_construct(**fields)

    env.describe = describe  # type: ignore[method-assign]
    counts.clear()
    ep = await _open(env)
    try:
        assert counts["tools"] == 1, counts
        assert counts["instructions"] == 1, counts
        assert counts["horizon"] == 1, counts
        assert counts["reference_templates"] == 1, counts
        assert counts["tool.terminal_kind"] == len(ep.describe().tools), counts
        # And reading the published contract, over and over, asks the env nothing more.
        before = dict(counts)
        ep.describe()
        ep.describe()
        assert counts == before, "reading the contract went back to the env"
    finally:
        await ep.close()


async def test_a_contract_that_survives_the_round_trip_never_reaches_the_env_s_copy_code() -> None:
    """The same value, JSON-clean this time: the round trip replaces it with the text the wire
    carries, so opening the episode never asks the env to copy anything and a later read cannot
    fail — or differ — on code the env supplied."""
    _COPIES.clear()

    class _LateBoom(str):
        def __deepcopy__(self, memo: Any) -> Any:
            _COPIES["boom"] = _COPIES.get("boom", 0) + 1
            raise RuntimeError("this value cannot be copied twice")

    env = _env()
    _describing(env, lambda spec: object.__setattr__(spec, "instructions", _LateBoom("go")))
    ep = await _open(env)
    try:
        assert ep.describe().instructions == "go"
        assert ep.describe().instructions == "go"
        assert _COPIES.get("boom", 0) == 0
    finally:
        await ep.close()


async def test_a_marker_this_core_does_not_declare_is_not_normalized_into_one() -> None:
    """A manifest's markers are how the serve layer reads the contract: ``provenance`` is how a
    harness finds the reserved stop tool without hard-coding its name, and ``terminal_kind`` is
    how this episode decides which tool seals and scores.

    Rendering them through JSON was not enough. A ``str`` subclass renders to the text it
    subclasses, so an object that answers a comparison its own way was normalized into the very
    literal it was impersonating; and a value that renders to a string nobody declares was
    written into the contract as though it meant something. So they are **matched** against the
    strings this core declares rather than rendered into them, the constant that matched is what
    the contract keeps, and a marker outside that set is a contract this layer cannot read.

    That the alternative is a refusal rather than a guess is the point: guessing ``none`` for an
    unreadable ``terminal_kind`` is an env's advertised scoring terminal silently downgraded to
    an ordinary tool, which is the exact hole the constructor's fail-loud check exists to close."""
    _COPIES.clear()

    def mutate(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "terminate":
                manifest.provenance = _DriftingText("reserved")

    env = _env()
    _describing(env, mutate)
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert "marker this core does not declare" in str(refused.value)
    # The env's object never got to answer for itself, so it was never copied either.
    assert _COPIES.get("drift", 0) == 0


async def test_the_markers_this_core_declares_are_still_carried() -> None:
    """The other side of that line: an ordinary contract publishes ordinary markers, and every
    reader is shown the same ones."""
    env = _env()
    ep = await _open(env)
    try:
        first = {t.name: (t.provenance, t.terminal_kind) for t in ep.describe().tools}
        second = {t.name: (t.provenance, t.terminal_kind) for t in ep.describe().tools}
        assert first == second
        assert first["terminate"] == ("reserved", "abort")
        assert first["submit"] == ("env-mandatory", "score")
        assert ep.seal_enabled is True
    finally:
        await ep.close()


async def test_a_field_this_layer_cannot_read_is_refused_rather_than_guessed_at() -> None:
    """Reading a field off the contract is the env's code as much as serializing it is, and the
    two failures are not answered the same way.

    A value that will not *serialize* is kept exactly as it was read and the episode opens: there
    is a value, and whoever can tell whether the task is servable gets to judge it. A field that
    will not be *read* leaves nothing to hand on. Rebuilding the contract as this module's own
    object is what makes that difference concrete, because a rebuilt contract has to hold
    something for every field, and inventing one is the guess this layer must never make.

    So the refusal is here, attributed, and the layer above stops the run on it rather than being
    handed a contract with a field it cannot ask about."""

    class _Unreadable(TaskSpec):
        def __getattribute__(self, name: str) -> Any:
            if name in ("instructions", "reference_templates"):
                raise RuntimeError("this field cannot be read")
            return super().__getattribute__(name)

    env = _env()
    real = env.describe

    def describe(task_id: Any = None) -> Any:
        return _Unreadable.model_construct(**dict(real(task_id)))

    env.describe = describe  # type: ignore[method-assign]
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert "this field cannot be read" in str(refused.value)
    assert isinstance(refused.value.__cause__, RuntimeError)


def _publishing_a_schema(schema: Any) -> Any:
    """Publish this env's own contract with ``submit``'s argument schema replaced.

    Replaced on **every** instance, catalog and episode alike, so no drift check applies and what
    is under test is the schema itself rather than an env answering two readers differently."""

    def arm(env: Any) -> None:
        def mutate(spec: Any) -> None:
            for manifest in spec.tools:
                if manifest.name == "submit":
                    manifest.input_schema = schema

        _describing(env, mutate)

    return arm


@pytest.mark.parametrize(
    ("schema", "cause"),
    [
        pytest.param(
            {"type": "definitely-not-a-json-schema-type"},
            "not a valid JSON Schema",
            id="a type keyword no draft declares",
        ),
        pytest.param(
            {"properties": {"answer": {"type": "not-a-type"}}},
            "not a valid JSON Schema",
            id="the same thing one level down",
        ),
        pytest.param([], "not a JSON object", id="a list where a schema belongs"),
        pytest.param(
            {"type": "array"},
            "rather than an object",
            id="a root no call could satisfy",
        ),
        # Four valid schemas with nothing wrong at the `type` member and no object anywhere in
        # what they accept. Reading the root's declared type cannot see any of them.
        pytest.param(
            {"not": {"type": "object"}},
            "no request this transport can make would satisfy",
            id="a root that excludes every object",
        ),
        pytest.param(
            {"const": "not-an-object"},
            "no request this transport can make would satisfy",
            id="a root const that is not an object",
        ),
        pytest.param(
            {"enum": ["a", 1]},
            "no request this transport can make would satisfy",
            id="a root enum with no object among its choices",
        ),
        pytest.param(
            {"allOf": [{"type": "object"}, {"type": "array"}]},
            "no request this transport can make would satisfy",
            id="a composition whose branches contradict each other",
        ),
        pytest.param(
            {"oneOf": [{"type": "array"}, {"const": "x"}]},
            "no request this transport can make would satisfy",
            id="a composition every branch of which excludes objects",
        ),
        pytest.param(
            {"not": {}},
            "no request this transport can make would satisfy",
            id="a root that excludes everything there is",
        ),
    ],
)
async def test_a_schema_that_is_not_one_is_refused_before_a_task_is_dispensed(
    schema: Any, cause: str
) -> None:
    """Rendering a schema proves it is JSON. It does not prove it is a schema.

    This document is two things at once: what the transport advertises as a tool's arguments, and
    what the seal validates a terminal call against. A document that is not a schema fails both,
    and it failed them *as the caller's mistake*: the exactly correct submission came back
    ``{"error": "tool schema is invalid", "validation_error": true}``, the run was never stopped,
    and an orderly shutdown filed ``closure="drained"`` with ``score.success=False`` and no
    diagnostic. A right answer recorded as a clean scored loss, with nothing anywhere saying the
    env was at fault, and a retry is no better because the schema is the same on every call.

    The root that is not an object is the same failure without a broken schema: an object instance
    cannot satisfy it, so every call any transport can make is refused, and refused as a
    validation error the agent is invited to correct.

    Checked on the validator the seal itself will use, so what passes here is what is enforced
    there, and checked at construction, where a contract this layer cannot enforce is refused in
    the env's name and the run is stopped rather than scored."""
    env = _env()
    _publishing_a_schema(schema)(env)
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert cause in str(refused.value)
    assert "submit" in str(refused.value), "the refusal must name the tool an operator has to fix"


async def test_a_schema_this_layer_can_enforce_is_carried_exactly_as_published() -> None:
    """The gate is not a wall. A schema that constrains nothing at the root is a schema an object
    satisfies, so it opens, and it is advertised exactly as the env wrote it: a root that declares
    no type is left alone rather than given one this layer invented."""
    published = {"properties": {"answer": {"type": "string"}}, "required": ["answer"]}
    env = _env()
    _publishing_a_schema(published)(env)
    ep = await _open(env)
    try:
        advertised = {m.name: m.input_schema for m in ep.describe().tools}
        assert advertised["submit"] == published
        result = await ep.call("submit", {"answer": "4"})
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
    finally:
        await ep.close()


async def test_a_schema_error_reaching_the_seal_is_the_env_s_fault_not_the_caller_s() -> None:
    """And the classification holds wherever the discovery happens.

    Answering a ``SchemaError`` with a validation error told the agent that the request it got
    exactly right was malformed, and told the run nothing at all. Nothing a caller sends can
    satisfy a document that is not a schema, so it is not offered to the caller to fix: it
    propagates, and the layer that owns the task's record classifies it, like every other failure
    here that the caller could not have caused.

    Injected rather than grown from an env, because :func:`_core_schema` now refuses such a
    contract before an episode exists and there is no env-shaped way left to reach this line. The
    classification is what is under test, and it has to hold for whichever input reaches it
    next."""
    ep = await _open(_env())
    try:
        ep._score_schemas["submit"] = {"type": "definitely-not-a-json-schema-type"}
        with pytest.raises(jsonschema.SchemaError):
            await ep.call("submit", {"answer": "4"})
        # Not sealed, not scored, and not answered as though the agent could do better.
        assert ep._state is LifecycleState.OPEN
        assert ep.sealed is False and ep.terminated is False
    finally:
        await ep.close()


def _moving_a_marker(tool: str, kind: str) -> Any:
    """Publish the contract this env would have published, with one terminal marker moved after
    the models have validated it."""

    def arm(env: Any) -> None:
        def mutate(spec: Any) -> None:
            for manifest in spec.tools:
                if manifest.name == tool:
                    manifest.terminal_kind = kind

        _describing(env, mutate)

    return arm


@pytest.mark.parametrize(
    ("tool", "kind", "cause"),
    [
        ("noop", "score", "at most one `score` terminal"),
        ("terminate", "score", "at most one `score` terminal"),
        ("terminate", "none", "must be advertised with terminal_kind='abort'"),
        ("noop", "abort", "reserved for the 'terminate' tool"),
    ],
)
async def test_a_terminal_marker_moved_after_validation_is_still_refused(
    tool: str, kind: str, cause: str
) -> None:
    """Two of this contract's rules are about the manifest collection rather than any one tool,
    and the rebuild was skipping both.

    :class:`TaskSpec` states them as a model validator, so a contract built the ordinary way
    passes through them. This module rebuilds the contract with ``model_construct``, deliberately,
    so that a field the models would have rejected still reaches the layer that can say whether
    the task is servable. That skips the validators too, and pydantic models are mutable and do
    not validate on assignment, so an env can build a valid spec, let it be validated, and move a
    marker on the way out.

    Both rules decide how a call is *dispatched*, which is why nothing above can answer for them.
    A second ``score`` terminal made an ordinary tool a sealing one while the framing still
    described it as ordinary: calling ``noop`` sealed the episode and filed a clean scored row
    with ``success=False``, so an agent's ordinary action became its terminal wrong answer. And
    ``abort`` is honoured by name at runtime rather than by marker, so a marker that disagrees
    advertises a stop tool that does not stop, or hides the one that does.

    Re-stated rather than revalidated: ``model_validate`` on the rebuilt data would also refuse
    the wrong-*typed* fields this layer carries through on purpose, and those belong to the layer
    that knows whether the task is servable. These two belong to nobody else."""
    env = _env()
    _moving_a_marker(tool, kind)(env)
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert cause in str(refused.value)
    assert isinstance(refused.value.__cause__, ValueError)


async def test_the_terminals_this_contract_declares_are_still_carried() -> None:
    """The gate is not a wall: the ordinary contract, with one ``score`` terminal and the reserved
    abort where they belong, opens an ordinary episode and still seals on the tool it advertised.
    """
    env = _env()
    ep = await _open(env)
    try:
        kinds = {m.name: m.terminal_kind for m in ep.describe().tools}
        assert kinds == {"submit": "score", "terminate": "abort", "noop": "none"}
        assert ep.seal_enabled is True
        result = await ep.call("submit", {"answer": "4"})
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
    finally:
        await ep.close()


def _unreadable_finalize(env: Any) -> None:
    """Make *this instance's* ``finalize`` unreadable.

    Through the class, because a special method is looked up on the type, and on a subclass built
    per instance, because the reachable shape is an env that answered while the catalog instance
    was validated and refuses once it is serving: a cached fetch, a flag, the wall clock."""

    class _FinalizeCannotBeRead(type(env)):  # type: ignore[misc]
        def __getattribute__(self, attr: str) -> Any:
            if attr == "finalize":
                raise RuntimeError("finalize cannot be read")
            return super().__getattribute__(attr)

    env.__class__ = _FinalizeCannotBeRead


async def test_a_promised_finalizer_this_layer_cannot_read_is_the_same_refusal() -> None:
    """Checking the hook was inside the refusal; *getting* it was not.

    A published ``score`` terminal is a promise that this call seals and finalizes, and the check
    that the promise can be kept has been an attributed contract refusal since the round that
    reclassified it. The ``getattr`` that reaches that check is the env's own code too: an
    attribute that raises is a different event from one that is absent, and ``getattr``'s default
    answers only for the second. It came out of the constructor as the env's bare error with no
    cause and nothing the layer above could classify, so the run refused the dispense, kept the
    position owed, met the identical failure on the next pull against a freshly built env, and
    closed clean over a queue nothing served."""
    env = _env()
    _unreadable_finalize(env)
    with pytest.raises(
        TaskContractError, match="_fixture_score.*cannot take as a contract"
    ) as refused:
        await _open(env)
    assert "finalize cannot be read" in str(refused.value)
    assert isinstance(refused.value.__cause__, RuntimeError)


async def test_an_env_that_promised_no_terminal_is_never_asked_for_a_finalizer() -> None:
    """And it is asked only because it promised. A contract advertising no ``score`` terminal
    undertakes nothing about a finalizer, so this layer does not go looking for one: there is no
    promise to keep, nothing downstream can call it, and a hook nobody undertook to provide is not
    a fact about the contract this episode publishes."""
    reads: List[str] = []

    class _CountsFinalizeReads(fixture._FixtureScoreEnv):  # type: ignore[name-defined]
        def __getattribute__(self, attr: str) -> Any:
            if attr == "finalize":
                reads.append(attr)
            return super().__getattribute__(attr)

        def describe(self, task_id: Any = None) -> Any:
            spec = super().describe(task_id)
            for manifest in spec.tools:
                if manifest.terminal_kind == "score":
                    manifest.terminal_kind = "none"
            return spec

    env = _CountsFinalizeReads(tasks=_TASKS)
    # The env's own construction check reads the hook, which is its business and not this
    # episode's; what is under test is what the *serve layer* asks for.
    reads.clear()
    ep = await _open(env)
    try:
        assert ep.seal_enabled is False
        assert reads == [], "an env was asked for a hook its contract never promised"
    finally:
        await ep.close()


# ----- a decided verdict survives whatever the env does after it -----
#
# The commit's tail still runs the env's values after the verdict is settled, and a raise there
# discards it. Not every one of them belongs here: the feedback serializer in the middle of that
# tail is deliberately left to raise, because a stream catches it out of the terminating call,
# redacts the agent's answer, files the row `finalize_error` with `score=None` and the failure
# named on it, and stops the run — an env whose feedback cannot be serialized will fail the same
# way on every task behind this one, and containing it here would answer with a *scored* row and
# let the queue carry on (see `test_a_forced_abort_the_env_fails_is_not_an_earned_give_up`).
# The two below have no such owner: nothing above can recover a verdict this layer threw away.


async def test_a_teardown_that_raises_cancellation_does_not_discard_the_verdict() -> None:
    """The agent submits the right answer, the finalizer grades it, the evidence is committed and
    the durable record says ``FINALIZED``. Then dropping the env's per-session state raises
    ``asyncio.CancelledError``.

    Teardown runs in the finalization's ``finally``, so a raise there does not just fail the
    cleanup — it replaces the finalization's *return value*. The caller's ``call()`` raises
    cancellation instead of receiving the verdict it earned, ``wait_finalized()`` raises,
    ``close()`` raises, and the run reconciles a correct, durably-recorded answer as a broker
    abort. The record on disk and the result the run reports disagree, and the one that reaches
    the score sheet is the wrong one.

    The teardown was already best-effort against ``Exception`` for exactly this reason; the
    verdict is not less earned because a plugin's cleanup raised the one exception that catch did
    not cover. (An env's own ``close`` raising later is a different boundary and is contained by
    the layer that owns the row — see ``stream._release_episode`` — so the fault here is the one
    call the finalization makes, and the env behaves after it.)"""
    env = _env()
    ep = await _open(env)
    real_end_session = env.end_session
    failed = [False]

    def end_session(session_id: str) -> None:
        if not failed[0]:
            failed[0] = True
            raise asyncio.CancelledError("this env cannot end a session")
        real_end_session(session_id)

    env.end_session = end_session  # type: ignore[method-assign]
    try:
        result = await ep.call("submit", {"answer": "4"})
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
        assert result.terminated is True
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FINALIZED"
        assert record.verdict == {"correct": True}
        # The record and the returned result agree, which is the whole point.
        assert ep.terminal_payload == {"correct": True, "finalize_error": False}
        assert failed[0], "the test never reached the teardown it was arming"
        await ep.wait_finalized()  # must not raise
    finally:
        await ep.close()  # must not raise


class _Uncopyable(str):
    """Text that refuses to be copied, with the one exception a `except Exception` misses."""

    def __deepcopy__(self, memo: Any) -> Any:
        raise asyncio.CancelledError("this value cannot be copied")


async def test_an_evidence_value_that_breaks_its_own_durable_write_only_degrades_it() -> None:
    """The other place the commit's tail touches the env's values: the durable record.

    Writing one copies what it is given on the way in, and a copy is the value's own code. The
    write is best-effort *by design* — a persistence failure degrades crash-recovery for one
    record and is flagged for audit, never raised, because a sealed episode must always yield an
    outcome — and catching ``Exception`` made that promise for every failure but one. A value
    whose copy raised cancellation left the record at ``PENDING`` and took the verdict with it:
    the caller's ``call()`` raised, ``close()`` raised, and the graded answer the env had already
    returned never reached anyone.

    Held to the ``diagnostic``, which is the env's object all the way to this write: it is
    private, it decides nothing, and it is carried as read rather than normalized. The verdict
    beside it is *not* a way in any more, and this asserts that too — it is round-tripped to
    plain data at the evaluator boundary, so the value this write copies is the core's, not the
    env's, however the env built it.

    Degraded, then, and only degraded: the flag is set, the verdict stands, and the episode
    finishes."""
    env = _env()

    async def finalize(req: Any) -> Any:
        return TerminalEvidence(
            source=req.source,
            status="ok",
            # Both channels at once: the verdict is normalized out of harm's way, the private
            # diagnostic is not, so this write meets exactly one value that can break it.
            verdict={"correct": True, "grade": _Uncopyable("A")},
            diagnostic=_Uncopyable("graded"),
        )

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        result = await ep.call("submit", {"answer": "4"})

        assert result.terminated is True
        assert json.loads(result.content) == {
            "correct": True,
            "grade": "A",
            "finalize_error": False,
        }
        assert ep._persist_degraded is True, "a failed write must be flagged for audit"
        assert ep._state is LifecycleState.CLOSED
        # The verdict this episode holds is plain data, so nothing downstream of the evaluator
        # ever asks the env's object anything again.
        assert ep._evidence is not None
        assert type(ep._evidence.verdict["grade"]) is str
        await ep.wait_finalized()
    finally:
        await ep.close()


async def test_a_setup_that_fails_before_the_first_await_still_releases_the_env() -> None:
    """The sweep beside the three: ``open_env`` took ownership of the env at the call, and read
    ``env.name`` one line above the ``try`` that releases it.

    ``name`` is the env's own code like everything else here, so an env that raised from it was
    one this method had promised to close on a failed setup and then never closed. It is a leak
    rather than a wrong number, but it is the same class of line: env code on the setup path,
    outside the guard that exists for it."""
    env = _env()
    closed: List[bool] = []
    real_close = env.close

    async def close() -> None:
        closed.append(True)
        await real_close()

    env.close = close  # type: ignore[method-assign]
    type(env).name = property(  # type: ignore[assignment]
        lambda self: (_ for _ in ()).throw(RuntimeError("this env cannot name itself"))
    )
    try:
        with pytest.raises(RuntimeError, match="cannot name itself"):
            await ServedEpisode.open_env(env)
        assert closed == [True], "the env this call took ownership of was never released"
    finally:
        del type(env).name


# ----- what a contract refusal costs the run above the episode -----


@pytest.mark.parametrize(
    ("arm", "cause"),
    [
        pytest.param(
            lambda env: _describing(
                env,
                lambda spec: object.__setattr__(
                    spec, "instructions", _UncarriableInstructions()
                ),
            ),
            "could not be put on the wire",
            id="the contract cannot be carried",
        ),
        pytest.param(
            _publishing_an_unwalkable_schema,
            "schema walk exploded",
            id="the contract cannot be normalized",
        ),
        pytest.param(
            _publishing_unreadable_tools,
            "tools cannot be read",
            id="the contract cannot be read",
        ),
        pytest.param(
            lambda env: setattr(env, "finalize", None),
            "no callable finalize",
            id="the advertised terminal cannot seal",
        ),
        pytest.param(
            _moving_a_marker("noop", "score"),
            "at most one `score` terminal",
            id="the advertised terminals contradict each other",
        ),
        pytest.param(
            _publishing_a_schema({"type": "definitely-not-a-json-schema-type"}),
            "not a valid JSON Schema",
            id="the advertised arguments are not a schema",
        ),
        pytest.param(
            lambda env: _describing(env, lambda spec: setattr(spec, "horizon", True)),
            "expected int, got True",
            id="the advertised budget is not a number of steps",
        ),
        pytest.param(
            _unreadable_finalize,
            "finalize cannot be read",
            id="the promised terminal hook cannot be read",
        ),
        pytest.param(
            lambda env: setattr(
                env,
                "describe",
                lambda task_id=None: (_ for _ in ()).throw(
                    RuntimeError("this env cannot describe a task")
                ),
            ),
            "cannot describe a task",
            id="the contract cannot be obtained at all",
        ),
    ],
)
async def test_a_contract_refusal_stops_the_run_instead_of_being_retried_forever(
    tmp_path: Path, arm: Any, cause: str
) -> None:
    """A refusal with a good message is still only half the answer.

    ``TaskStream.get_task()`` opened the episode above its own refusal boundary, so a contract
    the serve layer could not take as a contract arrived as an ordinary raise: the stream was not
    stopped, the queue position was still owed, and ``aclose()`` reported a clean run over a task
    nothing had served. Nothing durable said the task was unservable, and the caller's obvious
    response — pull the next task — walked into the identical failure, forever, because the
    contract is described by the same code every time.

    That is the difference this refusal has its own class for. A task that could not be loaded is
    about that task and costs one dispense. A contract the layer cannot read is about the **env**,
    so the stop is latched here on the same grounds ``_require_published_manifest`` latches one
    for a manifest it cannot compare, and the run says at both boundaries what it stopped for."""

    # Armed after construction, because construction validates a *catalog* instance and every
    # episode runs on a different one the factory built. That is the reachable shape for all of
    # these: a contract that varies with load order, a cached fetch, a flag, the wall clock.
    armed: List[bool] = []

    def factory(_name: str) -> Any:
        env = _env()
        if armed:
            arm(env)
        return env

    stream = TaskStream(factory, [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov")
    armed.append(True)
    with pytest.raises(TaskContractError, match="_fixture_score") as refused:
        await stream.get_task()
    assert cause in str(refused.value)

    assert stream.stopped, "the stop a contract refusal owes was never latched"
    # No task was handed out, so the position is still owed and no row is due.
    assert stream.queue_info() == QueueInfo(remaining=1, consumed=0, in_flight=0)
    assert stream.results == ()
    assert not (tmp_path / "prov" / "dispenses.jsonl").exists()
    assert not (tmp_path / "prov" / "results.jsonl").exists()
    # And the run no longer closes clean over the queue it never served.
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()
    # A second pull is answered by the stop rather than by another trip through the same env.
    with pytest.raises(RuntimeError, match="no further task can be scored"):
        await stream.get_task()


async def test_a_refusal_found_before_a_cancelled_cleanup_is_still_latched(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """An ordering rule, pinned directly rather than through an env.

    The classification is bookkeeping about something the run has *already discovered*, and it
    used to run after the cleanup that releases the refused episode. That made the record of a
    fault depend on an ``await`` a caller can cancel: a pull cancelled while the episode was
    closing propagated the cancellation, which is right, and took the finding with it, which is
    not. The stream came back unstopped, position owed, closing clean over a queue nothing had
    served, having already been told the env cannot publish a contract.

    Every contract refusal this module can currently raise happens while the episode is still
    being built, so there is no env-shaped way left to reach that window: it is refused before
    there is an episode to close. That is why the refusal is injected here instead of grown from
    an env. The rule is what is under test, and it has to hold for whichever site raises there
    next, not only for the ones that raise there today."""
    from shogym.serve import ServedEpisode as _Episode

    blocked = asyncio.Event()
    armed: List[bool] = []

    def refusing_describe(self: Any) -> Any:
        raise TaskContractError("env 'x' published a task contract this episode cannot take")

    def factory(_name: str) -> Any:
        built = _env()
        if armed:

            async def close() -> None:
                await blocked.wait()

            built.close = close  # type: ignore[method-assign]
        return built

    stream = TaskStream(factory, [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov")
    armed.append(True)
    monkeypatch.setattr(_Episode, "describe", refusing_describe)

    pull = asyncio.ensure_future(stream.get_task())
    # The refusal is latched on the way in, so `stopped` is what says the cleanup was entered.
    for _ in range(200):
        await asyncio.sleep(0.005)
        if stream.stopped:
            break
    assert stream.stopped, "the pull never reached the refusal"
    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull

    # The caller was cancelled and the finding survived it.
    assert stream.stopped, "a fault the run had already found was lost to a cancelled cleanup"
    assert stream.queue_info() == QueueInfo(remaining=1, consumed=0, in_flight=0)
    assert stream.results == ()
    blocked.set()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()


def _refusing_after_the_episode_exists(monkeypatch: Any) -> None:
    """Refuse the dispense at the first use of a contract, with the episode already built."""
    from shogym.serve import ServedEpisode as _Episode

    def refusing_describe(self: Any) -> Any:
        raise TaskContractError("env 'x' published a task contract this episode cannot take")

    monkeypatch.setattr(_Episode, "describe", refusing_describe)


def _refusing_to_record_the_dispense(monkeypatch: Any) -> None:
    """Refuse it at the other pre-dispense window: the record that has to land before the task
    is exposed. The episode is released by the sibling path at the end of ``get_task``."""

    def refusing_write(self: Any, live: Any) -> None:
        raise RuntimeError("this dispense could not be recorded")

    monkeypatch.setattr(TaskStream, "_write_dispense", refusing_write)


@pytest.mark.parametrize(
    ("arm", "closing"),
    [
        pytest.param(
            _refusing_after_the_episode_exists,
            "stopped before its queue was served",
            id="the contract is refused",
        ),
        pytest.param(
            _refusing_to_record_the_dispense,
            "could not record a dispense",
            id="the dispense cannot be recorded",
        ),
    ],
)
async def test_a_cleanup_a_cancelled_pull_started_still_releases_its_episode(
    tmp_path: Path, monkeypatch: Any, arm: Any, closing: str
) -> None:
    """The ownership half of the same window, on the stream's side of it.

    The test above pins that a *finding* survives a cancelled cleanup. This pins that what the
    cleanup was cleaning up survives it too. A pull that opens an episode and then refuses to
    dispense it owns that episode: it is in no registry, no caller was handed it, and the close
    at the end of the refusal is the only thing that will ever let go of its env and its MCP
    sessions. Awaited inline, that close was as durable as the caller's patience: a cancellation
    delivered while it was suspended ended the only coroutine performing it, and the episode was
    left ``OPEN`` holding a live session, before **and** after ``aclose()``, which cannot recover
    an episode it was never told about.

    Both pre-dispense windows are held, because they are two call sites of one rule: the contract
    refusal near the top of the dispense, and the episode nobody could be handed at the end of it.
    """
    from shogym.serve import ServedEpisode as _Episode

    blocked = asyncio.Event()
    entered = asyncio.Event()
    episodes: List[Any] = []
    real_close = _Episode.close

    async def blocking_close(self: Any) -> None:
        # Blocked *before* delegating, so a cancellation delivered here leaves the whole release
        # undone rather than most of it.
        episodes.append(self)
        entered.set()
        await blocked.wait()
        await real_close(self)

    stream = TaskStream(
        lambda _name: _env(), [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov"
    )
    arm(monkeypatch)
    monkeypatch.setattr(_Episode, "close", blocking_close)

    pull = asyncio.ensure_future(stream.get_task())
    for _ in range(400):
        await asyncio.sleep(0.005)
        if entered.is_set():
            break
    assert entered.is_set(), "the pull never reached the cleanup this test is about"
    (episode,) = episodes
    assert score_mcp._sessions, "the test never reached the state it is about"

    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull
    assert stream.stopped, "the stop is still latched before anything cancellable"
    assert episode._state is LifecycleState.OPEN, "the release cannot have finished yet"

    # The caller is gone; the release is not.
    blocked.set()
    for _ in range(400):
        await asyncio.sleep(0.005)
        if episode._state is LifecycleState.CLOSED:
            break
    assert episode._state is LifecycleState.CLOSED, "the episode this pull created was abandoned"
    assert score_mcp._sessions == {}, "the episode's session state outlived the pull"

    with pytest.raises(RuntimeError, match=closing):
        await stream.aclose()
    # And it stays released. This ordering proves the release makes progress on its own; that
    # shutdown *waits* for it is the sibling below, which never lets the close finish first.
    assert episode._state is LifecycleState.CLOSED
    assert score_mcp._sessions == {}


async def _reached(predicate: Any, *, tries: int = 400) -> bool:
    """Let the loop run until ``predicate`` holds, or give up. Used to establish that something
    has *not* happened, so the give-up is the interesting outcome."""
    for _ in range(tries):
        await asyncio.sleep(0.005)
        if predicate():
            return True
    return False


def _closing_blocked_on(entered: asyncio.Event, blocked: asyncio.Event, monkeypatch: Any) -> None:
    """Hold every episode close open until ``blocked`` is set, announcing on ``entered``."""
    from shogym.serve import ServedEpisode as _Episode

    real_close = _Episode.close

    async def blocking_close(self: Any) -> None:
        entered.set()
        await blocked.wait()
        await real_close(self)

    monkeypatch.setattr(_Episode, "close", blocking_close)


async def test_shutdown_waits_for_the_episodes_it_opened_and_never_dispensed(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Retaining the close kept it alive. It did not give it an owner.

    ``aclose()`` is the statement that this stream has let go of what it held, and the episodes a
    pull built and never handed out are held by nothing else: no registry has them, the seal drain
    never sees them, and the pull that started the close may be long cancelled. The set they were
    parked in was only a strong reference, so an orderly shutdown returned over an episode still
    ``OPEN`` with its env session live, and the run was told it had released something it had not.

    The ordering here is the whole test. The close is held open for the entire shutdown, so
    nothing can pass by finishing first: ``aclose()`` has to still be waiting, and it may only
    return once the episode is closed and its session gone."""
    entered, blocked = asyncio.Event(), asyncio.Event()
    episodes: List[Any] = []

    def refusing_describe(self: Any) -> Any:
        episodes.append(self)
        raise TaskContractError("env 'x' published a task contract this episode cannot take")

    from shogym.serve import ServedEpisode as _Episode

    monkeypatch.setattr(_Episode, "describe", refusing_describe)
    _closing_blocked_on(entered, blocked, monkeypatch)

    stream = TaskStream(
        lambda _name: _env(), [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov"
    )
    pull = asyncio.ensure_future(stream.get_task())
    assert await _reached(entered.is_set), "the pull never reached the cleanup"
    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull
    (episode,) = episodes
    assert stream._cleanups, "the close the cancelled pull left behind was not retained"

    # Shutdown starts while that close is still held open, and may not get past it.
    closing = asyncio.ensure_future(stream.aclose())
    assert not await _reached(closing.done, tries=60), (
        "aclose() reported an orderly shutdown over an episode it had not seen released"
    )
    assert episode._state is LifecycleState.OPEN
    assert len(score_mcp._sessions) == 1

    blocked.set()
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await asyncio.wait_for(closing, timeout=10)
    # Only now, and this is the claim `aclose()` returning is supposed to make.
    assert episode._state is LifecycleState.CLOSED
    assert score_mcp._sessions == {}
    assert stream._cleanups == set()


async def test_a_close_claimed_while_shutdown_is_joining_is_joined_too(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A snapshot is not synchronization.

    Joining one of these takes as long as an env's ``close`` takes, and a pull that was already
    inside ``get_task`` when shutdown began can claim its own close in that time, one statement
    behind the look that found nothing. Taken once, shutdown joins what it happened to see and
    returns over the rest.

    So the drain takes what is there and comes back, ending only on a pass that finds the set
    empty. Two closes here, and the second is claimed while shutdown is already waiting on the
    first, which is the interleaving a single snapshot cannot answer. It terminates because
    ``_closed`` is latched before the drain exists: the pulls that can still claim a close are
    the ones already in flight, and every later one is refused at the door.

    What shutdown does *not* wait for is a dispense still in flight, which is this module's own
    line (see ``test_shutdown_is_not_bypassed_by_a_dispense_in_flight``): such a pull abandons
    the episode it built and releases it itself. Both pulls here have let go of theirs before
    shutdown is asked to finish."""
    entered, first, second, paused = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    episodes: List[Any] = []
    releases: List[asyncio.Event] = [first, second]

    from shogym.serve import ServedEpisode as _Episode

    real_close = _Episode.close
    real_open_env = _Episode.open_env

    async def paused_open_env(*args: Any, **kwargs: Any) -> Any:
        # Only the second pull waits here, and it waits holding the dispense lock, which is what
        # makes it the pull that is still in flight when shutdown starts.
        if episodes:
            await paused.wait()
        return await real_open_env(*args, **kwargs)

    async def blocking_close(self: Any) -> None:
        release = releases[episodes.index(self)]
        entered.set()
        await release.wait()
        await real_close(self)

    def refusing_framing(self: Any, ref: Any, instructions: Any, budget: Any, lease: Any) -> Any:
        raise RuntimeError("this task cannot be handed over")

    real_describe = _Episode.describe

    def watched_describe(self: Any) -> Any:
        if self not in episodes:
            episodes.append(self)
        return real_describe(self)

    monkeypatch.setattr(_Episode, "open_env", staticmethod(paused_open_env))
    monkeypatch.setattr(_Episode, "describe", watched_describe)
    monkeypatch.setattr(_Episode, "close", blocking_close)
    monkeypatch.setattr(TaskStream, "_deliverable_framing", refusing_framing)

    stream = TaskStream(
        lambda _name: _env(),
        [TaskRef("_fixture_score", 0), TaskRef("_fixture_score", 1)],
        prov_dir=tmp_path / "prov",
    )

    async def cancel_when_it_reaches_its_close(pull: "asyncio.Task[Any]") -> None:
        assert await _reached(entered.is_set), "the pull never reached its close"
        entered.clear()
        pull.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pull

    # One close claimed and left behind by a cancelled pull.
    await cancel_when_it_reaches_its_close(asyncio.ensure_future(stream.get_task()))
    # And a second pull already in flight, holding the dispense lock with no episode yet.
    late = asyncio.ensure_future(stream.get_task())
    await asyncio.sleep(0.05)
    assert len(episodes) == 1, "the second pull was supposed to still be opening"

    # Shutdown starts, and is waiting inside the join of the first close.
    closing = asyncio.ensure_future(stream.aclose())
    assert not await _reached(closing.done, tries=60)

    # Now the pull in flight builds its episode, refuses it, and claims a close of its own.
    paused.set()
    await cancel_when_it_reaches_its_close(late)
    assert len(episodes) == 2, "the second pull never claimed a close"

    first.set()
    assert not await _reached(closing.done, tries=60), (
        "shutdown returned over a close claimed after it had looked"
    )
    assert episodes[0]._state is LifecycleState.CLOSED
    assert episodes[1]._state is LifecycleState.OPEN

    second.set()
    # Refused at the framing, which stops nothing, so this is an orderly close: what it waited
    # for is the whole of what it owed.
    await asyncio.wait_for(closing, timeout=10)
    assert [ep._state for ep in episodes] == [LifecycleState.CLOSED] * 2
    assert score_mcp._sessions == {}


@pytest.mark.parametrize(
    "finishes_first",
    [
        pytest.param(False, id="the close is still running when shutdown joins it"),
        pytest.param(True, id="the close already failed before shutdown looked"),
    ],
)
async def test_a_pre_dispense_close_that_fails_is_reported_rather_than_discarded(
    tmp_path: Path, monkeypatch: Any, finishes_first: bool
) -> None:
    """And what the join reads is not thrown away either.

    The set discarded its members on completion, so a background close that raised had nobody left
    to read it: the failure went to the same place the episode went, which is nowhere. A teardown
    failure this layer cannot hand to a caller is one it records on the stream, which is what it
    already does with the deadline watchdog and with a catalog env, and a recorded stop reaches
    both boundaries in the same words: ``aclose()`` raises it, and any later pull is answered by
    it.

    Driven through the framing refusal rather than the contract one, because that path stops
    nothing by itself: what this asserts is the stop the *close* caused, and the first stop is the
    one a stream keeps."""
    entered, blocked = asyncio.Event(), asyncio.Event()
    episodes: List[Any] = []

    from shogym.serve import ServedEpisode as _Episode

    real_close = _Episode.close

    async def failing_close(self: Any) -> None:
        episodes.append(self)
        entered.set()
        await blocked.wait()
        await real_close(self)
        raise RuntimeError("this episode could not let go of its env")

    def refusing_framing(self: Any, ref: Any, instructions: Any, budget: Any, lease: Any) -> Any:
        raise RuntimeError("this task cannot be handed over")

    monkeypatch.setattr(_Episode, "close", failing_close)
    monkeypatch.setattr(TaskStream, "_deliverable_framing", refusing_framing)

    stream = TaskStream(
        lambda _name: _env(), [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov"
    )
    pull = asyncio.ensure_future(stream.get_task())
    assert await _reached(entered.is_set), "the pull never reached the cleanup"
    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull
    assert not stream.stopped, "this path stops nothing by itself; the close is what does"

    if finishes_first:
        # The failure happens with nobody waiting at all, and well before shutdown looks. Dropped
        # from the set on completion it was gone by then, so the same fault stopped the run when
        # it was slow and vanished when it was quick.
        blocked.set()
        (episode,) = episodes
        assert await _reached(lambda: episode._state is LifecycleState.CLOSED)
        assert await _reached(lambda: any(task.done() for task in stream._cleanups)), (
            "the failed close was let go of before any owner had read it"
        )
        closing = asyncio.ensure_future(stream.aclose())
    else:
        closing = asyncio.ensure_future(stream.aclose())
        assert not await _reached(closing.done, tries=60)
        blocked.set()

    with pytest.raises(RuntimeError, match="could not close an episode it opened") as reported:
        await asyncio.wait_for(closing, timeout=10)
    assert isinstance(reported.value.__cause__, RuntimeError)
    assert "could not let go of its env" in str(reported.value.__cause__)
    assert stream.stopped, "a teardown failure with no caller to hand it to went unrecorded"
    # Read, so it is not asyncio's unretrieved-exception warning either, and let go of only then.
    assert stream._cleanups == set()
    # And the same finding answers the next pull, in the words that boundary uses.
    with pytest.raises(RuntimeError, match="could not be closed"):
        await stream.get_task()


def _opening_paused_on(paused: asyncio.Event, opening: asyncio.Event, monkeypatch: Any) -> None:
    """Hold every episode open inside ``open_env``, announcing on ``opening``.

    A pull suspended there is one this stream deliberately lets a whole shutdown run past: it is
    past the door, holding the dispense lock, with no episode yet."""
    from shogym.serve import ServedEpisode as _Episode

    real_open_env = _Episode.open_env

    async def paused_open_env(*args: Any, **kwargs: Any) -> Any:
        opening.set()
        await paused.wait()
        return await real_open_env(*args, **kwargs)

    monkeypatch.setattr(_Episode, "open_env", staticmethod(paused_open_env))


def _closing_recorded(
    entered: asyncio.Event,
    blocked: asyncio.Event,
    episodes: List[Any],
    monkeypatch: Any,
    *,
    fails: bool,
) -> None:
    """Hold every episode close open, naming the episode, and fail after the release if asked."""
    from shogym.serve import ServedEpisode as _Episode

    real_close = _Episode.close

    async def blocking_close(self: Any) -> None:
        episodes.append(self)
        entered.set()
        await blocked.wait()
        await real_close(self)
        if fails:
            raise RuntimeError("this episode could not let go of its env")

    monkeypatch.setattr(_Episode, "close", blocking_close)


async def _abandoned_close(
    stream: TaskStream, paused: asyncio.Event, opening: asyncio.Event, entered: asyncio.Event
) -> None:
    """Reach the state these two tests are about: a close claimed after a completed shutdown, by a
    pull that is then cancelled.

    The window is this module's own and is not an accident: a dispense in flight neither blocks
    shutdown nor is blocked by it, so a pull already inside ``get_task`` finds ``_closed`` when it
    comes back, abandons the episode it built, and claims the only release that episode will ever
    get, one statement after the release that was supposed to be the last."""
    pull = asyncio.ensure_future(stream.get_task())
    assert await _reached(opening.is_set), "the pull never reached its episode"
    # The shutdown this module allows to finish inside that window.
    await stream.aclose()
    assert stream._releasing is not None and stream._releasing.done()
    paused.set()
    assert await _reached(entered.is_set), "the pull never claimed its close"
    pull.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pull


@pytest.mark.parametrize(
    "close_fails",
    [
        pytest.param(False, id="the close outlives the shutdown that ran before it"),
        pytest.param(True, id="the close fails after the shutdown that ran before it"),
    ],
)
async def test_a_close_claimed_after_a_completed_shutdown_is_still_drained(
    tmp_path: Path, monkeypatch: Any, close_fails: bool
) -> None:
    """A release claimed once is not a release that covers everything owed once.

    The stream's own teardown is memoized deliberately: the watchdog is stopped and the catalog
    envs are let go exactly once, and every later ``aclose`` joins that same task. What that
    memoization also did was answer for the *closes*, which are not the release's to answer for:
    a pull still opening an episode when shutdown completes goes on to find ``_closed``, abandon
    that episode and claim its close after the drain inside the release has already run. Cancel
    that pull and the close sits in the set with nobody waiting, while every later ``aclose``
    awaits a task that finished before the close existed and returns immediately. Measured: the
    second ``aclose()`` returned clean with the episode still ``OPEN``, one pending close in the
    set, its env session live, and ``stopped`` false; and a failure from that close was read by
    nobody at all.

    So the closes are joined on every arrival, not only on the first. Both halves of the failed-
    close policy hold across that window too: the join waits for the close, and what it reads is
    recorded on the stream rather than discarded."""
    paused, opening, entered, blocked = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    episodes: List[Any] = []

    _opening_paused_on(paused, opening, monkeypatch)
    _closing_recorded(entered, blocked, episodes, monkeypatch, fails=close_fails)

    stream = TaskStream(
        lambda _name: _env(), [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov"
    )
    await _abandoned_close(stream, paused, opening, entered)
    assert stream._cleanups, "the close the cancelled pull left behind was not retained"
    (episode,) = episodes
    assert episode._state is LifecycleState.OPEN
    assert len(score_mcp._sessions) == 1, "the test never reached the state it is about"

    # The second shutdown is the only owner left, and it may not pass this close either.
    closing = asyncio.ensure_future(stream.aclose())
    assert not await _reached(closing.done, tries=60), (
        "a completed release made a nonempty cleanup set invisible to shutdown"
    )
    assert episode._state is LifecycleState.OPEN
    assert len(score_mcp._sessions) == 1

    blocked.set()
    if close_fails:
        with pytest.raises(RuntimeError, match="could not close an episode it opened") as reported:
            await asyncio.wait_for(closing, timeout=10)
        assert "could not let go of its env" in str(reported.value.__cause__)
        assert stream.stopped, "a teardown failure the second shutdown read went unrecorded"
        with pytest.raises(RuntimeError, match="could not be closed"):
            await stream.get_task()
    else:
        await asyncio.wait_for(closing, timeout=10)
    # What `aclose()` returning is supposed to mean, on the second one as much as on the first.
    assert episode._state is LifecycleState.CLOSED
    assert score_mcp._sessions == {}, "shutdown returned over a session it had not seen released"
    assert stream._cleanups == set()


async def test_a_drain_a_cancelled_shutdown_started_is_still_the_one_that_answers(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """And the late join is claimed, for the reason every release on this path is claimed.

    The drain takes what it is about to read *out* of the set, so a shutdown cancelled inside the
    join would be the second way an entry leaves with nobody having read it: the set comes back
    empty, the close is still running, and the next ``aclose`` finds nothing to wait for and
    reports a clean run over it. So the join runs as its own task and the waiter is shielded, and
    a later arrival joins that same task rather than starting a second one over a set the first
    has already emptied."""
    paused, opening, entered, blocked = (
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
        asyncio.Event(),
    )
    episodes: List[Any] = []

    _opening_paused_on(paused, opening, monkeypatch)
    _closing_recorded(entered, blocked, episodes, monkeypatch, fails=True)

    stream = TaskStream(
        lambda _name: _env(), [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov"
    )
    await _abandoned_close(stream, paused, opening, entered)

    # A shutdown that reaches the join and is then cancelled: the set is empty by then, because
    # the join is the remover.
    interrupted = asyncio.ensure_future(stream.aclose())
    assert await _reached(lambda: not stream._cleanups), "the shutdown never reached the join"
    interrupted.cancel()
    with pytest.raises(asyncio.CancelledError):
        await interrupted

    blocked.set()
    # The close still had an owner, so what it raised is still the run's to report.
    with pytest.raises(RuntimeError, match="could not close an episode it opened") as reported:
        await asyncio.wait_for(stream.aclose(), timeout=10)
    assert "could not let go of its env" in str(reported.value.__cause__)
    (episode,) = episodes
    assert episode._state is LifecycleState.CLOSED
    assert score_mcp._sessions == {}


def _unconstrained_context(spec: Any) -> None:
    """Let the submission carry a nested value the schema does not constrain."""
    for manifest in spec.tools:
        if manifest.name == "submit":
            manifest.input_schema["properties"]["context"] = {}


def _rewriting_the_answer(req: Any, _correct: bool) -> None:
    req.args["answer"] = "rewritten by the finalizer"


def _rewriting_inside_the_submission(req: Any, _correct: bool) -> None:
    req.args["context"]["note"] = "rewritten by the finalizer"


@pytest.mark.parametrize(
    ("describe_as", "submitted", "rewrite"),
    [
        pytest.param(
            None,
            {"answer": "4"},
            _rewriting_the_answer,
            id="the finalizer replaces the answer",
        ),
        pytest.param(
            _unconstrained_context,
            {"answer": "4", "context": {"note": "as submitted"}},
            _rewriting_inside_the_submission,
            id="the finalizer rewrites a value inside it",
        ),
    ],
)
async def test_a_finalizer_cannot_rewrite_the_submission_the_seal_witnessed(
    describe_as: Any, submitted: Dict[str, Any], rewrite: Any
) -> None:
    """The digest is taken at the seal and the evaluator ran on the same dictionary.

    ``FinalizeRequest`` carries the arguments so an evaluator can grade them, and the terminal
    ``Step`` the verifier scores and ``evidence.args`` were both built from that same dictionary
    afterwards. So a finalizer that rewrote ``req.args`` changed what the trajectory said the
    agent had submitted, while the durable digest went on witnessing the call that actually
    arrived: measured, the record matched ``{"answer": "4"}``, ``verify`` scored
    ``{"answer": "rewritten by the finalizer"}``, and the run reported a clean success over the
    disagreement. A seal that means anything cannot have two answers to what was sealed.

    So the submission is rendered once at the seal and everything after it reads that rendering,
    and what the evaluator gets is a copy it may do as it likes with, which is the corollary
    ``_detached_evidence`` states for the evidence, one hook earlier. The copy is deep, because a
    submission is a document and rewriting a value *inside* it says the same wrong thing as
    replacing it."""
    seen: List[Any] = []
    scored: List[Any] = []

    env = _env(finalize_hook=rewrite)
    if describe_as is not None:
        _describing(env, describe_as)

    real_finalize = env.finalize

    async def watched_finalize(req: Any) -> Any:
        seen.append(json.loads(json.dumps(req.args)))
        return await real_finalize(req)

    env.finalize = watched_finalize  # type: ignore[method-assign]

    real_verify = env._verify

    def watched_verify(trajectory: Any, task: Any, **kwargs: Any) -> Any:
        scored.append([dict(step.arguments) for step in trajectory])
        return real_verify(trajectory, task, **kwargs)

    env._verify = watched_verify  # type: ignore[method-assign]

    ep = await _open(env)
    try:
        result = await ep.call("submit", dict(submitted))

        # The evaluator was handed the submission, graded it, and the rewrite cost it nothing.
        assert seen == [submitted]
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
        # One submission, and every witness of it says the same thing.
        assert ep._args_digest == args_digest(submitted)
        assert scored == [[submitted]], "the verifier scored a submission that never arrived"
        assert dict(ep._trajectory[-1].arguments) == submitted
        assert ep._evidence is not None and ep._evidence.args == submitted
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FINALIZED"
        assert record.args_digest == args_digest(submitted)
        await ep.wait_finalized()
    finally:
        await ep.close()


async def test_the_submission_this_seal_carries_holds_no_object_the_caller_supplied() -> None:
    """And the snapshot is a *rendering*, which is what makes it the core's.

    Copying the submission would otherwise be the caller's own code (a value in it decides what
    a copy of it is, or whether there is one at all), and that code would run twice inside the
    terminal transaction: once for the evaluator's copy and once for the terminal step, the second
    of them after the verdict is decided and inside the commit. So the walk that the durable digest
    was already taking is taken as a *value*: the same normalization, admitting exactly what the
    digest admits, and what it produces is plain data that copies without asking anybody anything.

    The digest is unchanged by this, which is the point of taking the digest's own walk: what the
    record witnesses and what the transaction carries are two shapes of one rendering."""

    class _NoCopyOfItsOwn:
        """A submitted value that renders through ``str`` and refuses to be copied."""

        def __deepcopy__(self, memo: Any) -> Any:
            raise RuntimeError("this value cannot be copied")

        def __str__(self) -> str:
            return "a value with no copy of its own"

    env = _env()
    _describing(env, _unconstrained_confidence)
    ep = await _open(env)
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": _NoCopyOfItsOwn()})

        rendered = {"answer": "4", "confidence": "a value with no copy of its own"}
        assert json.loads(result.content) == {
            "correct": True,
            "finalize_error": False,
            "confidence": "a value with no copy of its own",
        }
        # The record's own walk, and the transaction's, are the same walk.
        assert ep._args_digest == args_digest(rendered)
        assert dict(ep._trajectory[-1].arguments) == rendered
        assert ep._evidence is not None and ep._evidence.args == rendered
        assert all(
            type(value) in (str, int, float, bool, type(None), dict, list)
            for value in ep._trajectory[-1].arguments.values()
        ), "the caller's object reached the trajectory the verifier scores"
        await ep.wait_finalized()
    finally:
        await ep.close()


@pytest.mark.parametrize(
    "reward",
    [
        pytest.param(7, id="an ordinary integer"),
        pytest.param(9007199254740993, id="an integer past float precision"),
        pytest.param(10**400, id="an integer past float range"),
    ],
)
async def test_an_integer_reward_is_the_same_number_at_every_sink(
    tmp_path: Path, reward: int
) -> None:
    """Rendering once is worth nothing if the rebuild changes the value.

    The terminal feedback is rendered once and both halves come out of that rendering: the wire
    dicts this episode retains for a stream to headline, and the items the trace row and the
    in-band sidecar are written from. Rebuilding those items through the feedback *model* put a
    ``float | bool | str`` annotation in charge of a wire contract that admits any JSON scalar, so
    the two halves stopped being two shapes of one value: an env publishing
    ``9007199254740993`` had that retained and ``9007199254740992.0`` published to the trace and
    to the agent. Two rewards for one rendering, neither of them announced.

    An integer past the float *range* was worse: the serializer validates through the same
    rebuild, so it was refused there, and feedback this wire can carry perfectly well became a
    terminal ``finalize_error`` with the answer redacted.

    Value **and** type at all three sinks, because equality cannot see this on its own."""
    env = _env()

    def verify(trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None) -> Any:
        collection = FeedbackCollection()
        if terminated:
            item = EpisodeFeedback(name="reward", value=1.0)
            item.value = reward  # off-wire, the way a mutable model lets an env publish one
            collection.episode.append(item)
        return collection

    env.verify = verify  # type: ignore[method-assign]
    ep = await ServedEpisode.open_env(
        env, env_name="_fixture_score", task=0, trace_path=tmp_path / "run.jsonl"
    )
    try:
        result = await ep.call("submit", {"answer": "4"})

        # Recordable, on every value this wire declares legal.
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
        expected = [{"name": "reward", "value": reward, "level": "episode"}]
        assert ep.terminal_feedback == expected
        assert type(ep.terminal_feedback[0]["value"]) is int

        inband, _ = parse_meta(result.meta)
        assert [item.value for item in inband] == [reward]
        assert type(inband[0].value) is int

        steps = [r for r in load_traces(tmp_path / "run.jsonl") if "feedback" in r]
        assert steps[-1]["feedback"] == expected
        assert type(steps[-1]["feedback"][0]["value"]) is int
        await ep.wait_finalized()
    finally:
        await ep.close()


@pytest.mark.parametrize(
    ("reward", "headline", "stops"),
    [
        pytest.param(
            9007199254740993,
            9007199254740992.0,
            False,
            id="the headline holds it, to the precision the row declares",
        ),
        pytest.param(10**400, None, True, id="the headline cannot hold it at all"),
    ],
)
async def test_an_integer_reward_reaches_the_row_with_its_value_intact(
    tmp_path: Path, reward: int, headline: Optional[float], stops: bool
) -> None:
    """Where the integer goes once the episode stops rewriting it, stated rather than assumed.

    A row keeps two different things and this is the test that says so. ``observed`` and
    ``Score.feedback`` are the record: the env's items, verbatim, and the exact integer survives
    into both. ``Score.reward`` is a *headline*, declared ``float`` by the row's own schema, so it
    is the one place a conversion is meant to happen, in the open, beside the value it was taken
    from.

    And a number that headline cannot hold at all does what every unreadable summary here does:
    the row lands with its evidence intact and no score, the file says why, and the run stops. It
    is contained by the funnel that already exists for exactly this, so a value this layer now
    lets through cannot leave by a new door."""
    def factory(_name: str) -> Any:
        env = _env()

        def verify(trajectory: Any, task: Any, *, terminated: bool, evidence: Any = None) -> Any:
            collection = FeedbackCollection()
            if terminated:
                item = EpisodeFeedback(name="reward", value=1.0)
                item.value = reward
                collection.episode.append(item)
                collection.episode.append(EpisodeFeedback(name="success", value=True))
            return collection

        env.verify = verify  # type: ignore[method-assign]
        return env

    stream = TaskStream(factory, [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov")
    await stream.get_task()
    await stream.dispatch("submit", {"answer": "4"})
    if stops:
        with pytest.raises(RuntimeError, match="cannot headline"):
            await stream.aclose()
    else:
        await stream.aclose()

    (row,) = stream.results
    assert row.closure == "sealed"
    # The record keeps what the env published, exactly, either way.
    assert row.observed[0] == {"name": "reward", "value": reward, "level": "episode"}
    assert type(row.observed[0]["value"]) is int
    if headline is None:
        assert row.score is None
        assert "cannot headline" in (row.diagnostic or "")
        assert stream.stopped
    else:
        assert row.score is not None and row.score.reward == headline
        assert row.score.feedback[0]["value"] == reward
        assert not stream.stopped


def _naming_the_submit_tool(value: Any) -> Any:
    def mutate(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "submit":
                manifest.name = value

    return mutate


def _describing_the_submit_tool(value: Any) -> Any:
    def mutate(spec: Any) -> None:
        for manifest in spec.tools:
            if manifest.name == "submit":
                manifest.description = value

    return mutate


@pytest.mark.parametrize(
    ("break_spec", "defect"),
    [
        pytest.param(
            lambda spec: setattr(spec, "horizon", True),
            "expected int, got True",
            id="the budget is a bool",
        ),
        pytest.param(
            lambda spec: setattr(spec, "horizon", 2.5),
            "expected int, got 2.5",
            id="the budget is not a whole number of steps",
        ),
        pytest.param(
            lambda spec: setattr(spec, "instructions", 5),
            "expected str, got 5",
            id="the framing an agent is handed is not text",
        ),
        pytest.param(
            _naming_the_submit_tool(5),
            "expected str, got 5",
            id="an advertised tool is not named by anything a call could carry",
        ),
        pytest.param(
            _describing_the_submit_tool(5),
            "expected str, got 5",
            id="an advertised tool has no description an agent could read",
        ),
    ],
)
async def test_a_contract_this_layer_runs_on_is_refused_by_every_surface_alike(
    tmp_path: Path, break_spec: Any, defect: str
) -> None:
    """Carrying a wrong-typed field on to the layer above is only right for a field this layer
    passes on, and these are fields it runs on.

    The rebuild is a ``model_construct`` deliberately: a value the models would have rejected is
    still the env's published contract, and whether the *task* is servable belongs to whoever can
    tell. That reasoning was applied to every field alike, and two of them are not passed on at
    all. ``horizon`` is the budget this episode enforces and ``instructions`` are the framing an
    agent is handed, and a tool's ``name`` is the key a call is dispatched against.

    So the two supported surfaces disagreed about the same contract. ``TaskStream`` refused a
    boolean budget at the framing; the transport-independent engine that ``evaluate()`` and
    ``run_stdio()`` drive had no such gate and simply enforced it, so ``horizon=True`` was a
    budget of one step: one ordinary call ended the task, the record landed ``FINALIZED`` with
    ``{"correct": false}`` and ``finalize_error`` false, and an invalid env contract came out as
    an ordinary scored loss. One contract, two answers about whether it can be served, and the
    quieter answer was the wrong one.

    Held to the declared type at the contract boundary now, so every surface refuses the same
    contracts in the same words and in the env's name."""
    # Armed after construction, for the reason the class-A test gives: a stream validates the
    # *catalog* instance up front, and every episode runs on a different one the factory built.
    armed: List[bool] = []

    def factory(_name: str) -> Any:
        env = _env()
        if armed:
            _describing(env, break_spec)
        return env

    # 1. The transport-independent engine, which is what `evaluate()` and `run_stdio()` drive.
    armed.append(True)
    with pytest.raises(TaskContractError, match="_fixture_score.*cannot take") as refused:
        await _open(factory("_fixture_score"))
    assert defect in str(refused.value)

    # 2. And the stream, which reaches the same boundary through the same constructor and answers
    # for it the way it answers for every env-wide contract failure.
    armed.clear()
    stream = TaskStream(factory, [TaskRef("_fixture_score", 0)], prov_dir=tmp_path / "prov")
    armed.append(True)
    with pytest.raises(TaskContractError, match="_fixture_score"):
        await stream.get_task()
    assert stream.stopped, "a contract refusal that stops one surface must stop the other"
    assert stream.queue_info() == QueueInfo(remaining=1, consumed=0, in_flight=0)
    with pytest.raises(RuntimeError, match="stopped before its queue was served"):
        await stream.aclose()


async def test_the_budget_this_contract_declares_is_still_the_budget_it_enforces() -> None:
    """The gate is not a wall. A published integer budget is still enforced exactly as published,
    and an env that publishes none still runs unbounded."""
    ep = await _open(_env())
    try:
        assert ep.describe().horizon == 3
        await ep.call("noop", {})
        await ep.call("noop", {})
        assert ep._state is LifecycleState.OPEN and ep._step == 2
        result = await ep.call("noop", {})  # the third spends the published budget
        assert result.terminated is True
        assert ep._finalization_source == "horizon"
    finally:
        await ep.close()

    unbounded = _env()
    unbounded._horizon = None
    _describing(unbounded, lambda spec: setattr(spec, "horizon", None))
    ep = await _open(unbounded)
    try:
        assert ep.describe().horizon is None
        for _ in range(4):
            await ep.call("noop", {})
        assert ep._state is LifecycleState.OPEN and ep._step == 4
    finally:
        await ep.close()


async def test_a_submitted_value_is_walked_once_and_every_witness_agrees() -> None:
    """The submission is rendered once at the seal, and the confidence was read again after it.

    The echoed confidence is the one piece of the caller's data the core puts into a verdict it
    builds itself, so it is rendered strictly rather than through the digest's ``default=str``.
    Rendering it *after* the submission made that two walks of one value, and a walk is a
    question: a container free to answer differently the second time was digested as one
    submission and echoed as another. The witnesses then disagreed about what was submitted, with
    the durable record on one side and the answer the run gave on the other, which is the split
    this whole transaction exists to make impossible.

    So the strict rendering is taken first and put back into the arguments, and the submission's
    own walk reads plain data for that key. One walk of the caller's object, and every witness of
    it says the same thing."""
    walks = [0]

    class _TwoAnswers(list):
        """A submitted container that renders differently every time it is asked."""

        def __iter__(self) -> Any:
            walks[0] += 1
            return iter([walks[0]])

    env = _env()
    _describing(env, _unconstrained_confidence)
    seen: List[Any] = []

    async def finalize(req: Any) -> Any:
        seen.append(json.loads(json.dumps(req.args)))
        raise RuntimeError("this evaluator crashed")

    env.finalize = finalize  # type: ignore[method-assign]
    ep = await _open(env)
    try:
        result = await ep.call("submit", {"answer": "4", "confidence": _TwoAnswers([0])})

        submitted = {"answer": "4", "confidence": [1]}
        assert walks[0] == 1, "the caller's value was asked twice"
        # Every witness of the submission, and they are all the one rendering.
        assert seen == [submitted], "the evaluator was handed a different submission"
        assert dict(ep._trajectory[-1].arguments) == submitted
        assert ep._args_digest == args_digest(submitted)
        assert json.loads(result.content) == {
            "correct": False,
            "finalize_error": True,
            "confidence": [1],
        }
        (record,) = [r for r in ep._store.load_all() if r.session_id == ep.session_id]
        assert record.status == "FAILED"
        assert record.args_digest == args_digest(submitted)
        assert record.verdict == json.loads(result.content)
        await ep.wait_finalized()
    finally:
        await ep.close()


@pytest.mark.parametrize(
    "published",
    [
        pytest.param({}, id="a schema that constrains nothing"),
        pytest.param({"type": "object"}, id="a root that declares the object it takes"),
        pytest.param(
            {"properties": {"answer": {"type": "string"}}, "required": ["answer"]},
            id="required arguments the ladder has to fill",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {"answer": {"enum": ["4", "5"]}},
                "required": ["answer"],
            },
            id="a required argument constrained to a choice",
        ),
        pytest.param(
            {"anyOf": [{"type": "object"}, {"type": "string"}]},
            id="a composition that admits objects among other shapes",
        ),
        pytest.param(
            {
                "allOf": [
                    {"type": "object"},
                    {"properties": {"answer": {"type": "string"}}, "required": ["answer"]},
                ]
            },
            id="a composition whose branches agree",
        ),
    ],
)
async def test_a_schema_some_call_can_satisfy_is_still_carried_exactly_as_published(
    published: Dict[str, Any],
) -> None:
    """The gate is not a wall, and this is the half that says how wide it is.

    Admissibility is proved by exhibiting a call, so what the proof can build decides what this
    layer accepts. These are the shapes a contract actually uses: nothing at all, a plain object
    root, required arguments, a constrained choice, and compositions that admit an object among
    their branches. Each opens, and each is advertised exactly as the env wrote it, because the
    proof is a question asked of the schema and never a correction made to it.

    Every env registered in this repo that constructs offline was run through the same check and
    every advertised schema is admitted, which is the other half of this claim."""
    env = _env()
    _publishing_a_schema(published)(env)
    ep = await _open(env)
    try:
        advertised = {m.name: m.input_schema for m in ep.describe().tools}
        assert advertised["submit"] == published, "the proof rewrote the contract"
    finally:
        await ep.close()


@pytest.mark.parametrize(
    ("published", "requested", "defect"),
    [
        pytest.param(
            ["not-the-trace-id"], 0, "published ['not-the-trace-id']", id="a task id that is a list"
        ),
        pytest.param("7", 0, "published '7'", id="a task id naming a different task"),
        pytest.param(0, 0, "published 0", id="a task id that is the index unrendered"),
    ],
)
async def test_a_contract_that_names_another_task_cannot_answer_for_this_one(
    tmp_path: Path, published: Any, requested: Any, defect: str
) -> None:
    """``task_id`` looked like a publication, and one public consumer uses it as a key.

    This episode is handed the identity, writes it on every trace row it appends, and
    ``evaluate()`` reads the published field back off ``describe()`` and *filters* those rows with
    it before the session filter narrows them further. So a contract naming a different task is a
    key that matches nothing the run wrote: measured through public ``evaluate()``, the terminal
    row was on disk with ``task_id="0"`` and ``correct=true`` while the call returned
    ``terminated=False`` with an empty feedback list. An earned verdict thrown away, and a run
    that finished reported as one that never ended.

    So the published identity is reconciled with the one this episode was asked to serve, at the
    boundary that already refuses every other contract this layer cannot take."""
    env = _env()
    _describing(env, lambda spec: setattr(spec, "task_id", published))
    with pytest.raises(TaskContractError, match="_fixture_score.*cannot take") as refused:
        await ServedEpisode.open_env(env, env_name="_fixture_score", task=requested)
    assert "names a different task" in str(refused.value)
    assert defect in str(refused.value)


async def test_a_contract_that_names_no_task_is_still_served() -> None:
    """``None`` is not a wrong answer, it is no answer.

    The consumer reads an absent id as no filter at all rather than as a key that selects nothing,
    so a contract that names no task claims nothing and cannot mis-select. An env that hand-builds
    its contract and never fills the field in is unaffected, which is what keeps this a
    reconciliation rather than a new requirement on every env."""
    env = _env()
    _describing(env, lambda spec: setattr(spec, "task_id", None))
    ep = await ServedEpisode.open_env(env, env_name="_fixture_score", task=0)
    try:
        assert ep.describe().task_id is None
        result = await ep.call("submit", {"answer": "4"})
        assert json.loads(result.content) == {"correct": True, "finalize_error": False}
    finally:
        await ep.close()


async def test_a_contract_that_names_a_task_this_episode_was_not_asked_for_is_refused_too() -> None:
    """The other direction, on the env shape that reaches it.

    An env that does not index its tasks leaves this episode with no identity to write on its
    rows, so a contract that names one anyway hands ``evaluate()`` a key every row will miss. It
    is the same empty answer as a mismatched name, arrived at from the opposite side, and it is
    refused on the same rule."""

    class _KeepsNoIndex(fixture._FixtureScoreEnv):
        """An env whose task data carries no index, so the core resolves no identity."""

        def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
            data = super()._load_task(task_idx)
            data.pop("task_idx", None)
            return data

    env = _KeepsNoIndex(tasks=_TASKS)
    _describing(env, lambda spec: setattr(spec, "task_id", "0"))
    with pytest.raises(TaskContractError, match="_fixture_score.*cannot take") as refused:
        await ServedEpisode.open_env(env, env_name="_fixture_score", task=None)
    assert "asked for None" in str(refused.value)
    assert "published '0'" in str(refused.value)


@pytest.mark.parametrize(
    ("published", "call"),
    [
        pytest.param(
            {
                "type": "object",
                "required": ["n"],
                "properties": {"n": {"type": "integer", "minimum": 2}},
                "additionalProperties": False,
            },
            {"n": 2},
            id="a bound the ladder does not reach",
        ),
        pytest.param(
            {
                "type": "object",
                "required": ["s"],
                "properties": {"s": {"type": "string", "minLength": 3}},
            },
            {"s": "abc"},
            id="a length the ladder does not reach",
        ),
        pytest.param(
            {
                "type": "object",
                "required": ["s"],
                "properties": {"s": {"type": "string", "pattern": "^[0-9]{4}$"}},
            },
            {"s": "2026"},
            id="a pattern the ladder cannot guess",
        ),
        pytest.param(
            {
                "type": "object",
                "required": ["n"],
                "properties": {"n": {"type": "integer", "multipleOf": 7, "minimum": 7}},
            },
            {"n": 7},
            id="a multiple the ladder does not reach",
        ),
        pytest.param(
            {
                "type": "object",
                "required": ["v"],
                "properties": {"v": {"anyOf": [{"type": "integer", "minimum": 10}]}},
            },
            {"v": 10},
            id="a nested composition the ladder cannot satisfy",
        ),
        pytest.param(
            {
                "type": "object",
                "required": ["n"],
                "properties": {"n": {"type": "integer", "minimum": 2, "default": 0}},
            },
            {"n": 2},
            id="an annotation default that does not satisfy its own schema",
        ),
        pytest.param(
            # A `not` that excludes only *some* objects, with no witness the ladder can build:
            # the exclusion proof has to stay narrow enough to say nothing about this one.
            {
                "not": {"type": "object", "required": ["a"]},
                "required": ["b"],
                "properties": {"b": {"type": "integer", "minimum": 5}},
            },
            {"b": 5},
            id="a negation that excludes only some objects",
        ),
    ],
)
async def test_a_schema_this_layer_cannot_witness_is_still_served(
    published: Dict[str, Any], call: Dict[str, Any]
) -> None:
    """Absence of a witness is not a witness of absence, and this is where that was claimed.

    The candidate ladder is a handful of examples built from what a schema says it wants, and
    treating an exhausted ladder as proof of unsatisfiability made every constraint it cannot
    guess a contract refusal: an ordinary ``minimum: 2`` argument, a ``minLength``, a ``pattern``,
    a multiple, a nested ``anyOf``, even an annotation ``default`` that does not satisfy its own
    schema. Each of these is satisfied by a real call, and each was refused on every advertised
    tool of every episode, so envs that had always worked stopped opening at all.

    So the ladder proves and never refuses. A schema it cannot witness and cannot prove exclusive
    is accepted exactly as it was before this gate existed, which is the honest answer: general
    satisfiability is not decidable here, and the residual risk of a contract that admits nothing
    and is not provably exclusive is the risk this layer had all along."""
    env = _env()
    _publishing_a_schema(published)(env)
    ep = await _open(env)
    try:
        assert {m.name: m.input_schema for m in ep.describe().tools}["submit"] == published
        # And the call the schema really does admit is served, rather than refused by a guess.
        assert ep._validate_terminal_args("submit", dict(call)) is None
    finally:
        await ep.close()


async def test_a_schema_this_layer_cannot_execute_is_refused_before_it_is_advertised() -> None:
    """The other direction of the same gate, and the one that must not stay inconclusive.

    An exception out of the validator is not an instance being rejected: the document failed to
    resolve or compile, and it will fail that way on every call any agent can make. Left
    inconclusive it was admitted, and then the failure landed where a *task-local* fault lands:
    measured through ``TaskStream``, the terminal call raised, the row was filed
    ``finalize_error``, the run was never stopped, and the same unusable env went on to serve the
    next task in the queue.

    This layer's claim is that it can enforce what it advertises, and it cannot enforce a document
    it cannot execute, so this is a contract refusal with the validator's own failure as the
    cause. The distinction it turns on: a *witness search* that ends without an answer is
    inconclusive and stays that way; *machinery* that raises has proved something."""
    published = {
        "type": "object",
        "required": ["x"],
        "properties": {"x": {"$ref": "#/$defs/Missing"}},
    }
    env = _env()
    _publishing_a_schema(published)(env)
    with pytest.raises(TaskContractError, match="_fixture_score.*cannot take") as refused:
        await _open(env)
    assert "cannot enforce" in str(refused.value)
    assert "submit" in str(refused.value), "the refusal must name the tool an operator has to fix"
    assert "Missing" in str(refused.value), "the validator's own failure is the cause"
