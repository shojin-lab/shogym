"""The frozen pass-count table, checked against the generator that produced it.

What this protects is the placebo arm. A drawn receipt's pass count is read against this table, so
a table that does not match the generator makes the placebo a different quantity from the real
receipt while both keep reporting a number.

**Opt in with ``SHOGYM_CHECK_PASS_TABLE=1``.** Building 318 backlogs is minutes of computation and
its wall time varies enough with what else is running that it does not belong in a default run.
The table only changes when the roster, the seed rule or the convention set changes, which is when
to run this:

    SHOGYM_CHECK_PASS_TABLE=1 pytest tests/envs/test_appworld_table.py

The cheap half of the check is unconditional, in the shape test below.
"""

from __future__ import annotations

import datetime as dt
import os
import zlib
from pathlib import Path

import pytest

from tests._fixtures.upstream_gate import provisioned

from shogym.envs.appworld import adapter  # noqa: E402

provisioned(adapter.ensure_corpus, package="appworld", extra="appworld")

from shogym.envs.appworld import ledger, payload  # noqa: E402


def _backlog_at(task_id: str, root: Path) -> ledger.Backlog:
    """The backlog production would build for this task: its own seed, its own reference date."""
    specs = adapter.task_specs(root, task_id)
    built = ledger.build_backlog(
        zlib.crc32(task_id.encode()),
        dt.datetime.fromisoformat(specs["datetime"]).date(),
    )
    assert built is not None, task_id
    return built


def test_the_committed_task_dates_are_the_ones_this_corpus_carries() -> None:
    """The offline roster tests build backlogs from a committed table of dates; this is the corpus
    saying that table is true.

    ``tests/envs/test_appworld_ledger.py`` runs without a corpus, so it cannot read the date
    production draws each task's backlog against. A committed table is the only way it can use
    production's inputs, and a committed table with nothing checking it is a fixture that drifts
    silently the first time the manifest or the pinned bundle moves. This is the check, and it
    runs wherever the corpus is provisioned. Regenerate the table from this corpus when it fails
    and the move was intended."""
    root = adapter.ensure_corpus()
    committed = [
        line.split("\t")
        for line in Path(__file__).with_name("appworld_task_dates.tsv").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert [task_id for task_id, _ in committed] == list(adapter.task_ids())
    live = [[task_id, adapter.task_specs(root, task_id)["datetime"]] for task_id, _ in committed]
    assert [list(row) for row in committed] == live


def test_the_table_covers_the_roster_it_claims_to() -> None:
    """The shape, unconditionally: one entry per possible pass count, and every task-by-convention
    pair filed exactly once. Cheap, and it catches a table built for a different roster. It does
    not catch a permuted one, which is what the full check above is for."""
    assert len(payload.pass_counts()) == ledger.DATED + 1
    assert sum(payload.pass_counts()) == len(adapter.task_ids()) * len(ledger.CONVENTIONS)


@pytest.mark.skipif(
    os.environ.get("SHOGYM_CHECK_PASS_TABLE") != "1",
    reason="minutes of computation; set SHOGYM_CHECK_PASS_TABLE=1 to run it",
)
def test_the_frozen_table_is_the_one_this_roster_produces() -> None:
    """Every served task, at the date its own specification carries, compared exactly.

    The cheap versions of this check pass while being wrong. A length survives a permuted table
    and so does a sum, and a fixture built at one hard-coded reference date is not what production
    builds, because production reads the date out of each task.
    """
    root = adapter.ensure_corpus()
    served = adapter.task_ids()
    built = ledger.pass_count_marginal(_backlog_at(task_id, root) for task_id in served)
    assert built == payload.pass_counts()
    assert sum(built) == len(served) * len(ledger.CONVENTIONS)
