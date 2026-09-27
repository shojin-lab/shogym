"""A plugin that leaves one test to run in a shard's whole selection: the guard's equality.

tests/test_ci_scheduling.py loads it with ``-p`` to put a real shard's selection through the
scheduler its job runs, at the cost of a collection. Every test is collected exactly as the job
collects it, grouped tests renamed and all, so the guard reads the job's whole collection. Every
test but the guard's equality is then skipped before any of its fixtures are set up, which is what
keeps the bundles and banks those tests build out of the check.
"""

from __future__ import annotations

import pytest

#: The one test left to run, by file and by name.
GUARD_FILE = "test_ci_shards.py"
GUARD_TEST = "test_this_job_collected_exactly_what_its_shard_names"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip every test except the guard's equality."""
    skip = pytest.mark.skip(reason="this run checks the guard and runs nothing else")
    for item in items:
        if item.path.name != GUARD_FILE or item.name != GUARD_TEST:
            item.add_marker(skip)
