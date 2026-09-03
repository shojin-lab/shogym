"""Fidelity checks for the ``automationbench`` port: real domain loading + local world seeding.

Nothing here compares a seeded world against upstream's own ``setup_state``; the checks are that
the real domain loader works, that two local loads agree, and that the re-hosted helpers behave
on synthetic state.

Unlike the verify/served tests (which inject synthetic tasks), these exercise the real upstream
domain loader — ``datasets``-backed, with deterministic per-``example_id`` noise injection — so
they require the ``datasets`` extra and the provisioned upstream source; the module skips without
either, keeping the core offline suite green.
"""

from __future__ import annotations

import pytest

pytest.importorskip("datasets", reason="automationbench extra (datasets) not installed")

from tests._fixtures.upstream_gate import gate

adapter = gate(
    "shogym.envs.automationbench.adapter",
    package="automationbench",
    extra="automationbench",
)

import shogym  # noqa: E402,F401 — registers the env
from shogym.envs.registration import make, registered_envs  # noqa: E402


def test_env_is_registered() -> None:
    assert "automationbench" in registered_envs()


def test_upstream_sha_is_pinned() -> None:
    # The fidelity pin the port reproduces (guards against an accidental bump).
    assert adapter.UPSTREAM_SHA == "6d210543b7a046f0f451c828cd2dadef774276eb"


def test_simple_domain_loads_and_seeds() -> None:
    env = make("automationbench", config={"domain": "simple"})
    assert env.num_tasks and env.num_tasks > 0
    spec = env.describe("0")
    # The task's request text is surfaced in the published instructions.
    assert "# Request" in spec.instructions
    assert spec.horizon == 52  # default max_steps (50) + done + terminate


def test_domain_loading_is_deterministic() -> None:
    # Noise is seeded by example_id upstream, so two loads of the same domain are identical.
    a = adapter.load_domain_tasks("simple")
    b = adapter.load_domain_tasks("simple")
    assert len(a) == len(b) and len(a) > 0
    assert [row["info"] for row in a] == [row["info"] for row in b]


def test_public_alias_expands_to_six_domains() -> None:
    # The default `public` alias is the 6 distributed domains combined (>= 600 tasks).
    tasks = adapter.load_domain_tasks("public")
    assert len(tasks) >= 600


def test_compute_allowed_services_matches_seed_and_assertions() -> None:
    allowed = adapter.compute_allowed_services(
        initial_state={"gmail": {}},
        assertions=[{"type": "salesforce_field_equals", "record_id": "x"}],
        zapier_tools=["slack_send_channel_message"],
    )
    assert allowed == ["gmail", "salesforce", "slack"]


def test_build_world_sets_allowed_services() -> None:
    info = {
        "initial_state": {"salesforce": {"contacts": []}},
        "assertions": [{"type": "gmail_message_sent", "to": "x@y.example.com"}],
        "zapier_tools": [],
    }
    world, initial, assertions = adapter.build_world(info)
    assert set(world.meta.allowed_services) == {"salesforce", "gmail"}
    assert "salesforce" in initial


def test_every_shipped_row_carries_its_name_inside_info() -> None:
    # The pinned upstream carries the name as `info["task_name"]` and ships no top-level `task`
    # column at all, the combined alias included. Both halves are pinned: a row that regained the
    # column would make the fallback live again, and a row that lost the new key would leave the
    # name falling back to a default that names nothing.
    import json

    from shogym.envs.automationbench.env_v1 import task_name

    for domain in ("public", "simple"):
        rows = adapter.load_domain_tasks(domain)
        assert rows, domain
        assert all("task" not in row for row in rows), f"{domain} regained the task column"
        for row in rows:
            info = json.loads(row["info"]) if isinstance(row["info"], str) else row["info"]
            assert info.get("task_name"), f"{domain} row {row.get('example_id')} has no task_name"
        assert task_name(rows[0]) == json.loads(rows[0]["info"])["task_name"]


def test_task_name_answers_the_same_for_a_raw_row_and_a_normalized_one() -> None:
    # `load_domain_tasks` hands back `info` as JSON text and the env normalizes it to a mapping.
    # The name is the same either way, so a caller reading it off a raw row is not silently
    # answered with an empty string.
    from shogym.envs.automationbench.env_v1 import _normalize_row, task_name

    raw = adapter.load_domain_tasks("public")[0]
    assert isinstance(raw["info"], str)
    assert task_name(raw) == task_name(_normalize_row(raw)) == "sales.multi_hop_lookup"
    # The fallbacks, in order: a legacy top-level column, then the caller's default.
    assert task_name({"task": "legacy.name", "info": {}}) == "legacy.name"
    assert task_name({"info": {}}, 7) == "7"
    assert task_name({"info": "not json"}, "fallback") == "fallback"
