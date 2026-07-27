"""End-to-end: drive a served ``automationbench`` episode through hgym's seal-before-verdict serve.

The whole path — build a per-session ``WorldState``, discover an endpoint with ``api_search``,
mutate state with ``api_fetch``, then call ``done`` (the ``score`` terminal, which seals + scores
in one step) — runs in-process and offline (no model, no key). Tasks are injected directly so these
served tests need neither the ``datasets`` extra nor the domain loaders.

These also exercise the seal lifecycle the migration adopts: ``done`` seals the episode and the
verdict is core-owned + public-safe (score numbers only, no oracle), a post-seal tool call is
tombstoned, an explicit ``terminate`` is a no-score abort, and the **horizon** scores the current
partial state (``finalize_current_state``).

The upstream source is provisioned lazily on first use; the module skips if it can't be fetched
(offline + cold cache), like the tau2 tests skip without their extra.
"""

from __future__ import annotations

import json

import pytest

try:
    from hgym.envs.automationbench import adapter  # noqa: F401 — triggers provisioning
except Exception as exc:  # pragma: no cover - network/provisioning failure
    pytest.skip(f"AutomationBench upstream source unavailable: {exc}", allow_module_level=True)

from hgym.serve import ServedEpisode  # noqa: E402

# A minimal, self-contained task: a seeded Salesforce contact whose phone must be updated. The
# assertion is initially failing (phone is +1-555-0000), so scoring is a clean 0 -> 1 signal.
_CONTACT_ID = "003001"
_SF_UPDATE_URL = (
    f"https://yourinstance.salesforce.com/services/data/v61.0/sobjects/Contact/{_CONTACT_ID}"
)
_TASK = {
    "example_id": 999001,
    "task": "test.sf_contact_phone_update",
    "prompt": [
        {"role": "system", "content": "You are a workflow automation agent."},
        {
            "role": "user",
            "content": "Update Jordan Lee's phone number in Salesforce to +1-555-0101.",
        },
    ],
    "answer": "",
    "info": {
        "zapier_tools": ["salesforce_contact_update"],
        "initial_state": {
            "salesforce": {
                "contacts": [
                    {
                        "id": _CONTACT_ID,
                        "first_name": "Jordan",
                        "last_name": "Lee",
                        "email": "jordan.lee@acme.example.com",
                        "phone": "+1-555-0000",
                    }
                ]
            }
        },
        "assertions": [
            {
                "type": "salesforce_field_equals",
                "collection": "contacts",
                "record_id": _CONTACT_ID,
                "field": "phone",
                "value": "+1-555-0101",
            }
        ],
    },
}


def _config(**over):
    cfg = {"tasks": [_TASK], "max_steps": 50}
    cfg.update(over)
    return cfg


async def _update_phone(episode) -> str:
    return (
        await episode.call(
            "api_fetch",
            {
                "method": "PATCH",
                "url": _SF_UPDATE_URL,
                "body": json.dumps({"Phone": "+1-555-0101"}),
            },
        )
    ).content


def _fb(episode) -> dict:
    return {i["name"]: i["value"] for i in episode.terminal_feedback}


async def test_describe_publishes_tools_and_task() -> None:
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        spec = episode.describe()
        names = {t.name for t in spec.tools}
        assert {"api_search", "api_fetch", "base64_encode", "done", "terminate"} <= names
        # `done` is advertised as the score terminal; `terminate` as the reserved abort.
        kinds = {t.name: t.terminal_kind for t in spec.tools}
        assert kinds["done"] == "score"
        assert kinds["terminate"] == "abort"
        # The task's request rides in the published instructions.
        assert "Jordan Lee" in spec.instructions
        assert spec.horizon == 52  # max_steps (50) + room for an explicit done
    finally:
        await episode.close()


async def test_done_seals_and_scores_one() -> None:
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        # Discover the endpoint (BM25 top-5), then mutate state through it.
        search = json.loads(
            (await episode.call("api_search", {"query": "update salesforce contact"})).content
        )
        assert 1 <= search["count"] <= 5

        resp = await _update_phone(episode)
        assert "error" not in resp.lower()

        # `done` is the score terminal: it seals + scores in one step.
        result = await episode.call("done", {})
        assert result.terminated
        verdict = json.loads(result.content)
        assert verdict["partial_credit"] == 1.0
        assert verdict["success"] is True
        assert verdict["finalize_error"] is False
        # The public verdict must not leak the rubric (assertions / target values / world).
        assert "assertions" not in verdict
        assert "world" not in verdict
        assert "salesforce" not in result.content.lower()

        fb = _fb(episode)
        assert fb["reward"] == 1.0
        assert fb["partial_credit"] == 1.0
        assert fb["success"] is True
    finally:
        await episode.close()


async def test_post_seal_tool_call_is_tombstoned() -> None:
    # After `done` seals, any further tool call is tombstoned — no inward dispatch, no re-score.
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        await _update_phone(episode)
        await episode.call("done", {})
        after = await episode.call("api_fetch", {"method": "GET", "url": _SF_UPDATE_URL})
        assert after.terminated
        assert "sealed" in after.content.lower()
        # The verdict stands — the tombstoned call changed nothing.
        assert _fb(episode)["partial_credit"] == 1.0
    finally:
        await episode.close()


async def test_done_seals_before_scoring_no_read_and_retry() -> None:
    # The reward-hack the seal closes structurally: call `done` early (locking a zero), then try to
    # satisfy the task and re-score. `done` seals the episode, so the post-seal `api_fetch` is
    # tombstoned (never mutates the world) and a second `done` is tombstoned too — the first (zero)
    # score stands.
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        first = await episode.call("done", {})
        assert first.terminated
        assert json.loads(first.content)["partial_credit"] == 0.0  # nothing done yet

        # These are all post-seal: tombstoned, no effect.
        await _update_phone(episode)
        await episode.call("done", {})

        assert _fb(episode)["partial_credit"] == 0.0
        assert _fb(episode)["success"] is False
    finally:
        await episode.close()


async def test_premature_terminate_scores_zero() -> None:
    # An explicit `terminate` is a no-score abort even with partial progress on the world.
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        await _update_phone(episode)  # would score 1.0 under `done`
        result = await episode.call("terminate", {})
        assert result.terminated
        fb = _fb(episode)
        assert fb["reward"] == 0.0
        assert fb["success"] is False
    finally:
        await episode.close()


async def test_done_without_action_scores_zero() -> None:
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        result = await episode.call("done", {})
        assert result.terminated
        fb = _fb(episode)
        assert fb["partial_credit"] == 0.0
        assert fb["success"] is False
    finally:
        await episode.close()


async def test_horizon_scores_current_partial_state() -> None:
    # `on_horizon = finalize_current_state`: a run that acts then runs out of steps without `done`
    # is scored on its current world (partial credit), not a flat zero. max_steps=1 -> horizon=3.
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config=_config(max_steps=1)
    )
    try:
        # Step 1 mutates the world to a passing state; then burn steps until the horizon hits.
        await _update_phone(episode)
        for _ in range(5):
            if episode.terminated:
                break
            await episode.call("api_search", {"query": "contact"})
        assert episode.terminated
        fb = _fb(episode)
        # The single assertion is satisfied, so the horizon-scored partial state is a full 1.0.
        assert fb["partial_credit"] == 1.0
        assert fb["success"] is True
    finally:
        await episode.close()


async def test_horizon_without_action_scores_zero() -> None:
    # Hitting the horizon with nothing done scores a clean zero (no partial credit to earn).
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config=_config(max_steps=1)
    )
    try:
        for _ in range(5):
            if episode.terminated:
                break
            await episode.call("api_search", {"query": "contact"})
        assert episode.terminated
        fb = _fb(episode)
        assert fb["partial_credit"] == 0.0
        assert fb["success"] is False
    finally:
        await episode.close()


async def test_service_gating_rejects_out_of_scope_calls() -> None:
    # The task seeds/asserts only Salesforce, so a Slack call fails like an unconnected account —
    # closing the silent-diversion hole. This is upstream's `allowed_services` gate, preserved.
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        resp = (
            await episode.call(
                "api_fetch",
                {
                    "method": "POST",
                    "url": "https://slack.com/api/chat.postMessage",
                    "body": json.dumps({"channel": "C1", "text": "hi"}),
                },
            )
        ).content
        payload = json.loads(resp)
        assert payload["error"]["code"] == 401
        assert "slack" in payload["error"]["message"].lower()
    finally:
        await episode.close()


async def test_scoring_is_deterministic_across_episodes() -> None:
    scores = []
    for _ in range(2):
        episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
        try:
            await _update_phone(episode)
            await episode.call("done", {})
            scores.append(_fb(episode)["partial_credit"])
        finally:
            await episode.close()
    assert scores[0] == scores[1] == 1.0


async def test_sessions_are_isolated() -> None:
    # Two concurrent episodes must not share WorldState: mutating one leaves the other's contact
    # unchanged (state is keyed by the injected _session_id).
    a = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    b = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        await _update_phone(a)
        await b.call("done", {})
        assert _fb(b)["partial_credit"] == 0.0  # b never acted
    finally:
        await a.close()
        await b.close()
