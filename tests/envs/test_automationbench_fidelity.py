"""Fidelity checks for the ``automationbench`` port: real domain loading + a seed matching upstream.

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
    assert adapter.UPSTREAM_SHA == "a321764ace3cfbe42289e6a13abef2f0f4f56fad"


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
