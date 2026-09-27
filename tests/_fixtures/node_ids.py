"""The id each test was collected under, kept where a scheduler cannot rename it.

Under ``--dist loadgroup`` a pytest-xdist worker renames every test that carries an ``xdist_group``
mark: it appends ``@`` and the test's group names to the node id, which is how its scheduler knows
to send a group to one worker. A serial collection renames nothing. The guard in
tests/test_ci_shards.py holds what a job collected against a serial collection of the job's shard,
so it has to read the ids pytest collected rather than the names a worker gave them, or every
grouped test would read as missing under one name and as a stray under the other.

So each id is kept as the test is collected, before anything later in collection renames it, and
:func:`collected_as` reads it back. Nothing is parsed off a renamed id, which is why a parameter id
that holds an ``@`` of its own is compared exactly as it was collected.

tests/conftest.py imports the hook below, which is how pytest finds it.
"""

from __future__ import annotations

import pytest

_COLLECTED_AS = pytest.StashKey[str]()


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep each test's id as pytest collected it.

    First among the hooks that see the collection, because pytest-xdist renames grouped tests in
    this same hook, and a copy taken after it would be a copy of the new name.
    """
    for item in items:
        item.stash[_COLLECTED_AS] = item.nodeid


def collected_as(item: pytest.Item) -> str:
    """The id ``item`` was collected under, whatever a scheduler has renamed it since."""
    return item.stash[_COLLECTED_AS]


__all__ = ["collected_as", "pytest_collection_modifyitems"]
