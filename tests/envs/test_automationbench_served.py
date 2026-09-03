"""End-to-end: drive a served ``automationbench`` episode through shogym's seal-before-verdict serve.

The whole path — build a per-session ``WorldState``, discover an endpoint with ``api_search``,
mutate state with ``api_fetch``, then call ``done`` (the ``score`` terminal, which seals + scores
in one step) — runs in-process and offline (no model, no key). Tasks are injected directly so these
served tests need neither the ``datasets`` extra nor the domain loaders.

These also exercise the seal lifecycle the migration adopts: ``done`` seals the episode and the
verdict is core-owned + public-safe (score numbers only, no oracle), a post-seal tool call is
tombstoned, an explicit ``terminate`` is a no-score abort, and the **horizon** scores the current
partial state (``finalize_current_state``).

The upstream source is provisioned lazily on first use; the module skips if it can't be fetched
(offline + cold cache), like the tau2 and yc_bench tests.
"""

from __future__ import annotations

import json
from typing import Any, Dict

import pytest

from tests._fixtures.upstream_gate import gate

adapter = gate(
    "shogym.envs.automationbench.adapter",
    package="automationbench",
    extra="automationbench",
)

from shogym.serve import ServedEpisode  # noqa: E402

# Upstream before its 1.0.6 release advertised every Jira endpoint with `/rest/api/3` doubled: the
# schema's base URL already ended there and the endpoint paths repeat it, so the URL `api_search`
# handed back reached no handler and the documented search-then-call handoff could not be walked at
# all. The release corrects the base URL, so the handoff test below runs once the pin moves to it.
_PIN_ADVERTISING_DOUBLED_JIRA_PATHS = "a321764ace3cfbe42289e6a13abef2f0f4f56fad"

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


# A task with a Jira project in the world and its key named nowhere the agent can read. The project
# search is the only Jira lookup on the served surface, so it is the only thing that can hand the
# key over.
_JIRA_SEARCH_URL = "https://company.atlassian.net/rest/api/3/project/search"
_JIRA_TASK = {
    "example_id": 999003,
    "task": "test.jira_project_from_the_world",
    "prompt": [{"role": "user", "content": "File a ticket for the billing page errors."}],
    "answer": "",
    "info": {
        "zapier_tools": ["jira_create_issue"],
        "initial_state": {"jira": {"projects": [{"key": "SUP", "name": "Support Issues"}]}},
        "assertions": [
            {
                "type": "jira_issue_exists_with_summary",
                "project": "SUP",
                "summary": "Billing page errors",
            }
        ],
    },
}


# The same world with a spreadsheet beside the Jira project, and a read of a range that decodes into
# the Jira lookup's own path. Sheets takes an A1 range as its last path segment and a decoded slash
# stays part of that range, so this is a successful Sheets answer whose path ends in the Jira route.
_SHEETS_RANGE_URL = (
    "https://sheets.googleapis.com/v4/spreadsheets/ss_patterns/values/"
    "Pattern%20Definitions!A1%2Frest%2Fapi%2F3%2Fproject%2Fsearch"
)
_SHEETS_AND_JIRA_INFO: Dict[str, Any] = {
    "zapier_tools": ["jira_create_issue", "google_sheets_lookup_row"],
    "initial_state": {
        "jira": {"projects": [{"key": "SUP", "name": "Support Issues"}]},
        "google_sheets": {
            "spreadsheets": [{"id": "ss_patterns", "title": "Escalation Patterns"}],
            "worksheets": [
                {
                    "id": "ws_patterns",
                    "spreadsheet_id": "ss_patterns",
                    "title": "Pattern Definitions",
                    "headers": ["pattern", "owner"],
                }
            ],
            "rows": [
                {
                    "id": "row_1",
                    "spreadsheet_id": "ss_patterns",
                    "worksheet_id": "ws_patterns",
                    "row_id": 2,
                    "cells": {"pattern": "auth-failure", "owner": "platform"},
                }
            ],
        },
    },
    "assertions": _JIRA_TASK["info"]["assertions"],
}
_SHEETS_AND_JIRA_TASK = {
    **_JIRA_TASK,
    "example_id": 999005,
    "task": "test.jira_project_beside_a_spreadsheet",
    "info": _SHEETS_AND_JIRA_INFO,
}


# The same world, plus the project already in the action log under a lowercase key. The route finds
# that one itself, so the completion must recognize it as the project it holds and not report it a
# second time in the seeded spelling.
_JIRA_ACTION_LOG_TASK = {
    **_JIRA_TASK,
    "example_id": 999004,
    "task": "test.jira_project_in_both_places",
    "info": {
        **_JIRA_TASK["info"],
        "initial_state": {
            "jira": {
                "actions": {
                    "project": [
                        {
                            "id": "jira_proj_1",
                            "action_key": "project",
                            "params": {"project": "sup", "project_id": "10001"},
                        }
                    ]
                },
                "projects": [{"key": "SUP", "name": "Support Issues"}],
            }
        },
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


async def test_a_seeded_jira_project_is_discoverable_through_lookup() -> None:
    # The project search is the only Jira lookup on the served surface, and the key is named nowhere
    # else, so a project the task seeded is discoverable only if the search reports it.
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config={"tasks": [_JIRA_TASK], "max_steps": 50}
    )
    try:
        assert "SUP" not in episode.describe().instructions

        async def search(**params) -> dict:
            call = {"method": "GET", "url": _JIRA_SEARCH_URL}
            if params:
                call["params"] = json.dumps(params)
            return json.loads((await episode.call("api_fetch", call)).content)

        found = await search()
        assert [p["key"] for p in found["values"]] == ["SUP"]
        assert found["values"][0]["name"] == "Support Issues"
        assert found["total"] == 1

        # Jira's own search matches a literal against the key or the name, case insensitively.
        assert [p["key"] for p in (await search(query="sup"))["values"]] == ["SUP"]
        assert [p["key"] for p in (await search(query="Support Iss"))["values"]] == ["SUP"]
        assert (await search(query="Marketing"))["values"] == []

        # The route filters on `query` and reads no other field, so an empty one is a search for
        # everything even when the request carries the name of the filter the route builds.
        with_alias = await search(query="", searchByParameter="missing")
        assert [p["key"] for p in with_alias["values"]] == ["SUP"]

        # The key the search handed back is usable: it files the issue the assertion names.
        await episode.call(
            "api_fetch",
            {
                "method": "POST",
                "url": "https://company.atlassian.net/rest/api/3/issue",
                "body": json.dumps(
                    {"fields": {"project": {"key": "SUP"}, "summary": "Billing page errors"}}
                ),
            },
        )
        await episode.call("done", {})
        assert _fb(episode)["partial_credit"] == 1.0
    finally:
        await episode.close()


async def test_an_encoded_spelling_of_the_search_route_is_completed_too() -> None:
    # The router percent-decodes each path segment before it matches, so `project%2Fsearch` reaches
    # the same handler. The completion has to follow it there, or the answer would depend on how the
    # agent spelled the URL.
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config={"tasks": [_JIRA_TASK], "max_steps": 50}
    )
    try:
        found = json.loads(
            (
                await episode.call(
                    "api_fetch",
                    {
                        "method": "GET",
                        "url": "https://company.atlassian.net/rest/api/3/project%2Fsearch",
                    },
                )
            ).content
        )
        assert [p["key"] for p in found["values"]] == ["SUP"]
        assert found["total"] == 1
    finally:
        await episode.close()


async def test_a_path_parameter_on_the_search_route_is_completed_too() -> None:
    # The router parses the URL before it matches, which drops a trailing path parameter, so
    # `project/search;v=1` reaches the same handler and the completion has to reach it too.
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config={"tasks": [_JIRA_TASK], "max_steps": 50}
    )
    try:
        found = json.loads(
            (
                await episode.call(
                    "api_fetch", {"method": "GET", "url": f"{_JIRA_SEARCH_URL};v=1"}
                )
            ).content
        )
        assert [p["key"] for p in found["values"]] == ["SUP"]
        assert found["total"] == 1
    finally:
        await episode.close()


async def test_the_routers_own_bare_path_is_completed_too() -> None:
    # A URL with no host at all is routed by its leading segment, which is how the router answers
    # its own internal spelling of the route, so the completion follows it there too.
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config={"tasks": [_JIRA_TASK], "max_steps": 50}
    )
    try:
        found = json.loads(
            (
                await episode.call(
                    "api_fetch", {"method": "GET", "url": "jira/rest/api/3/project/search"}
                )
            ).content
        )
        assert [p["key"] for p in found["values"]] == ["SUP"]
    finally:
        await episode.close()


async def test_another_service_answering_on_that_path_is_left_alone() -> None:
    # Sheets takes an A1 range as its own last path segment and a decoded slash stays part of that
    # range, so a successful spreadsheet read can end in the Jira lookup's path while being someone
    # else's route. The served bytes have to be the router's own, project and `total` key included.
    upstream_world, _, _ = adapter.build_world(_SHEETS_AND_JIRA_INFO)
    upstream = adapter._api_fetch(upstream_world, "GET", _SHEETS_RANGE_URL, None, None)

    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config={"tasks": [_SHEETS_AND_JIRA_TASK], "max_steps": 50}
    )
    try:
        served = (
            await episode.call("api_fetch", {"method": "GET", "url": _SHEETS_RANGE_URL})
        ).content
        assert json.loads(upstream)["values"], "the spreadsheet read has to succeed to mean anything"
        assert served == upstream
    finally:
        await episode.close()


@pytest.mark.skipif(
    adapter.UPSTREAM_SHA == _PIN_ADVERTISING_DOUBLED_JIRA_PATHS,
    reason="the pinned upstream advertises the Jira routes with /rest/api/3 doubled",
)
async def test_the_url_the_search_tool_advertises_reaches_the_completed_answer() -> None:
    # The served instructions tell the agent to `api_search` for an endpoint and then `api_fetch`
    # its url, so the completion is only reachable if the advertised url is the one that routes.
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config={"tasks": [_JIRA_TASK], "max_steps": 50}
    )
    try:
        results = json.loads(
            (
                await episode.call(
                    "api_search", {"query": "jira project search key", "top_k": 10}
                )
            ).content
        )["results"]
        advertised = [r for r in results if r["id"] == "jira.projects.search"]
        assert advertised, "the project search endpoint is not among the search results"

        found = json.loads(
            (
                await episode.call(
                    "api_fetch",
                    {"method": advertised[0]["method"], "url": advertised[0]["url"]},
                )
            ).content
        )
        assert [p["key"] for p in found["values"]] == ["SUP"]
    finally:
        await episode.close()


async def test_a_project_the_route_found_is_not_reported_again_in_another_case() -> None:
    # The route answers out of the action log, where the key is lowercase, while the world seeds it
    # uppercase. That is one project, so the search must report it once.
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config={"tasks": [_JIRA_ACTION_LOG_TASK], "max_steps": 50}
    )
    try:
        found = json.loads(
            (
                await episode.call("api_fetch", {"method": "GET", "url": _JIRA_SEARCH_URL})
            ).content
        )
        assert [v["project"] for v in found["values"]] == ["sup"]
        assert found["total"] == 1
    finally:
        await episode.close()


async def test_jira_project_search_is_untouched_when_the_service_is_not_connected() -> None:
    # The task subscribes to Salesforce only, so the search must still fail like an unconnected
    # account rather than being answered out of the world.
    episode = await ServedEpisode.start("automationbench", task=0, env_config=_config())
    try:
        payload = json.loads(
            (
                await episode.call(
                    "api_fetch", {"method": "GET", "url": _JIRA_SEARCH_URL}
                )
            ).content
        )
        assert payload["error"]["code"] == 401
        assert "jira" in payload["error"]["message"].lower()
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
