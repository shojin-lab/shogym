"""Offline served-episode tests for all five tau2 domains through shogym's serve layer.

Every non-solo domain runs with a **deterministic, offline** user simulator via litellm's
``mock_response`` (no OpenAI key), so these live in the suite (gated on the tau2 extra). They
prove the control-inversion bridge round-trips per domain — domain tool calls, the
``send_message`` user turn, and the ``done`` verdict — and that tau2's evaluator scores the
recorded state. Domains whose ``reward_basis`` includes NL assertions (retail, banking) are
scored with the offline ``evaluation_type="env"`` (DB component); their NL/judge component is
covered only on the keyed path (see ``test_tau2_fidelity.py``).
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("tau2", reason="tau2 extra not installed")

import shogym  # noqa: E402
from shogym.envs.tau2 import mcp_server  # noqa: E402
from shogym.serve import ServedEpisode  # noqa: E402

_MOCK_USER = "This is my request; please help."

# domain -> env_config for a fully-offline served run
_NON_SOLO = {
    "tau2_airline": {"user_llm_args": {"mock_response": _MOCK_USER}},
    "tau2_retail": {"user_llm_args": {"mock_response": _MOCK_USER}, "evaluation_type": "env"},
    "tau2_telecom": {"user_llm_args": {"mock_response": _MOCK_USER}},
    "tau2_banking_knowledge": {
        "user_llm_args": {"mock_response": _MOCK_USER},
        "evaluation_type": "env",
    },
}


def _skip_if_unconstructible(env_name: str) -> None:
    try:
        shogym.make(env_name)
    except Exception as exc:  # missing data / offline construction blocker
        pytest.skip(f"{env_name} not constructible offline: {exc}")


@pytest.mark.parametrize(
    "env_name,domain", [("tau2_airline", "airline"), ("tau2_retail", "retail"), ("tau2_telecom", "telecom")]
)
def test_env_task_ids_match_tau2_declared_split(env_name: str, domain: str) -> None:
    # The env must expose tau2's *declared* train/test splits verbatim — not a positional
    # slice — so held-out test tasks never leak into the train env. (Airline/retail/telecom
    # all declare train+test; mock/banking declare no holdout and are excluded here.)
    _skip_if_unconstructible(env_name)
    import shogym

    loader = mcp_server._reg.get_tasks_loader(domain)
    for split in ("train", "test"):
        env = shogym.make(env_name, {"task_split": split})
        expected = [t.id for t in loader(task_split_name=split)]
        assert env._task_ids == expected, f"{domain} {split} split diverges from tau2"
    # And the two splits are disjoint (no leakage).
    train = set(shogym.make(env_name, {"task_split": "train"})._task_ids)
    test = set(shogym.make(env_name, {"task_split": "test"})._task_ids)
    assert train and test and train.isdisjoint(test)
    # An unsupported split for a canonical-split domain is rejected — never silently widened
    # to the full (test-leaking) set.
    with pytest.raises(ValueError):
        shogym.make(env_name, {"task_split": "traim"})


def test_non_solo_default_user_args_match_upstream() -> None:
    # The default user-simulator kwargs must copy tau2's DEFAULT_LLM_ARGS_USER (temperature
    # 0.0) — not an empty dict — so the default non-solo config matches upstream `tau2 run`.
    _skip_if_unconstructible("tau2_airline")
    from tau2.config import DEFAULT_LLM_ARGS_USER

    task = mcp_server.load_tasks("airline")[0]
    # Construct the session without starting it (no LLM call): inspect the user's kwargs.
    session = mcp_server._Tau2Session(
        domain="airline",
        task=task,
        solo_mode=False,
        max_steps=100,
        user_llm="gpt-4.1",
        user_llm_args=None,
        evaluation_type="all",
        env_kwargs={},
    )
    assert session._orch.user.llm_args == DEFAULT_LLM_ARGS_USER
    assert session._orch.user.llm_args is not DEFAULT_LLM_ARGS_USER  # a copy, not the global


@pytest.mark.parametrize("env_name,cfg", list(_NON_SOLO.items()))
async def test_non_solo_domain_round_trips_and_scores(env_name: str, cfg: dict) -> None:
    _skip_if_unconstructible(env_name)
    episode = await ServedEpisode.start(env_name, task=0, env_config=cfg)
    try:
        spec = episode.describe()
        names = {t.name for t in spec.tools}
        # Non-solo tool surface: send_message + done + terminate + domain tools.
        assert {"send_message", "done", "terminate"} <= names
        assert len(names) > 4
        # Domain policy is surfaced in the task contract.
        assert "# Domain policy" in spec.instructions

        # A user turn round-trips through the (offline) user simulator.
        reply = await episode.call("send_message", {"content": "Hello, how can I help?"})
        assert _MOCK_USER in reply.content

        # `done` runs tau2's evaluator; the verdict is well-formed and self-marked.
        done = await episode.call("done", {})
        verdict = json.loads(done.content)
        assert verdict[mcp_server.VERDICT_MARKER] is True
        assert isinstance(verdict["db_match"], bool)  # DB scored offline

        # `terminate` ends the shogym episode; verify parses the verdict into feedback.
        term = await episode.call("terminate", {})
        assert term.terminated
        fb = {i["name"]: i["value"] for i in episode.terminal_feedback}
        assert "reward" in fb and "success" in fb
        assert isinstance(fb["db_match"], bool)
    finally:
        await episode.close()


async def test_airline_gold_actions_satisfy_db() -> None:
    # Positive, fully-offline non-solo check: replaying a task's gold agent actions makes
    # tau2's DB check pass (db_match True). Robust to the split — finds a train task that has
    # replayable assistant actions.
    _skip_if_unconstructible("tau2_airline")
    import shogym

    env = shogym.make("tau2_airline")
    by_id = {t.id: t for t in mcp_server.load_tasks("airline")}
    pick = None
    for idx, task_id in enumerate(env._task_ids):
        task = by_id[task_id]
        ec = getattr(task, "evaluation_criteria", None)
        actions = [
            (a.name, dict(a.arguments or {}))
            for a in (ec.actions if ec else [])
            if getattr(a, "requestor", None) == "assistant"
        ]
        if actions:
            pick = (idx, actions)
            break
    if pick is None:
        pytest.skip("no airline train task with gold agent actions")

    idx, actions = pick
    cfg = {"user_llm_args": {"mock_response": _MOCK_USER}}
    episode = await ServedEpisode.start("tau2_airline", task=idx, env_config=cfg)
    try:
        for name, args in actions:
            await episode.call(name, args)
        done = await episode.call("done", {})
        assert json.loads(done.content)["db_match"] is True
    finally:
        await episode.close()


async def test_user_stop_verdict_retrieved_via_done() -> None:
    # If tau2 auto-terminates (here the user simulator says ###STOP###), the verdict is
    # stashed and surfaced when the harness calls `done` — so it lands on a `done` step and
    # the verifier (which trusts only `done`) still scores the episode.
    _skip_if_unconstructible("tau2_airline")
    cfg = {"user_llm_args": {"mock_response": "###STOP###"}}
    episode = await ServedEpisode.start("tau2_airline", task=0, env_config=cfg)
    try:
        # This message drives the user to stop → tau2 terminates on its own.
        await episode.call("send_message", {"content": "How can I help?"})
        # The stashed verdict is returned on the `done` step (not lost on the message step).
        done = await episode.call("done", {})
        verdict = json.loads(done.content)
        assert verdict[mcp_server.VERDICT_MARKER] is True
        assert isinstance(verdict["db_match"], bool)
        term = await episode.call("terminate", {})
        assert term.terminated
        fb = {i["name"]: i["value"] for i in episode.terminal_feedback}
        assert "reward" in fb and isinstance(fb["db_match"], bool)
    finally:
        await episode.close()


async def test_premature_terminate_scores_zero_non_solo() -> None:
    # Ending a non-solo episode without `done` scores premature zero (no verdict recorded).
    _skip_if_unconstructible("tau2_telecom")
    cfg = {"user_llm_args": {"mock_response": _MOCK_USER}}
    episode = await ServedEpisode.start("tau2_telecom", task=0, env_config=cfg)
    try:
        result = await episode.call("terminate", {})
        assert result.terminated
        fb = {i["name"]: i["value"] for i in episode.terminal_feedback}
        assert fb["reward"] == 0.0 and fb["success"] is False
    finally:
        await episode.close()
