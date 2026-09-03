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


def test_drive_is_granted_whenever_sheets_is() -> None:
    seed = {"initial_state": {"google_sheets": {"spreadsheets": []}}, "assertions": []}
    # Upstream's own rule leaves Drive out, so the served set is where the grant comes from.
    assert adapter.compute_allowed_services(seed["initial_state"], [], []) == ["google_sheets"]
    assert adapter.allowed_services_for_task(seed["initial_state"], [], []) == [
        "google_drive",
        "google_sheets",
    ]
    world, _, _ = adapter.build_world({**seed, "zapier_tools": []})
    assert "google_drive" in world.meta.allowed_services


def test_drive_is_not_granted_to_a_task_that_never_touches_sheets() -> None:
    # The grant is Sheets-shaped, not a blanket subscription: nothing else pulls Drive in.
    assert adapter.allowed_services_for_task({"gmail": {}}, [], []) == ["gmail"]


def test_no_pool_task_needs_a_spreadsheet_id_no_endpoint_returns() -> None:
    # Sheets' four read routes all take the spreadsheet id as a path segment, so Drive's file
    # listing is the world's only enumeration of spreadsheets. Every seeded spreadsheet in the
    # shipped pool has to come back from that listing, or the task names a resource the agent can
    # only reach by guessing an opaque author string.
    import json

    tasks = adapter.load_domain_tasks("public")
    unlistable: list[tuple[int, str]] = []
    seen = 0
    for index, row in enumerate(tasks):
        info = row["info"]
        if isinstance(info, str):
            info = json.loads(info)
        world, _, _ = adapter.build_world(info)
        spreadsheets = list(world.google_sheets.spreadsheets)
        if not spreadsheets:
            continue
        seen += 1
        listing = json.loads(
            adapter.api_fetch(world, "GET", "https://www.googleapis.com/drive/v3/files")
        )
        returned = {f.get("id") for f in listing.get("files", [])}
        unlistable.extend(
            (index, sheet.id) for sheet in spreadsheets if sheet.id not in returned
        )
    assert seen > 100, f"expected the pool to seed spreadsheets on many tasks, saw {seen}"
    assert unlistable == []


def test_recorded_undiscoverable_indices_name_the_pool_they_describe() -> None:
    # The recorded list is a fact about the shipped pool, so it has to keep matching it: the tasks
    # upstream's own rule leaves reading Sheets with no Drive to enumerate them, plus the one whose
    # unreachable id is a Jira project key rather than a spreadsheet.
    import json

    from shogym.envs.automationbench.undiscoverable import PREVIOUSLY_UNDISCOVERABLE

    tasks = adapter.load_domain_tasks("public")
    sheets_without_drive = set()
    for index, row in enumerate(tasks):
        info = row["info"]
        if isinstance(info, str):
            info = json.loads(info)
        upstream = adapter.compute_allowed_services(
            adapter.strip_none_values(info.get("initial_state", {})),
            [adapter.strip_none_values(a) for a in info.get("assertions", [])],
            info.get("zapier_tools", []),
        )
        if "google_sheets" in upstream and "google_drive" not in upstream:
            sheets_without_drive.add(index)

    recorded = set(PREVIOUSLY_UNDISCOVERABLE)
    assert len(PREVIOUSLY_UNDISCOVERABLE) == len(recorded) == 106
    assert max(recorded) < len(tasks)
    assert len(sheets_without_drive) == 105
    assert recorded - sheets_without_drive == {375}
    assert sheets_without_drive - recorded == set()
