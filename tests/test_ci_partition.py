"""Every job collects what one job collected, test by test.

:mod:`tests.test_ci_shards` reads the partition in files: every test file on disk is named by
exactly one shard. That is the cheap claim, and it runs in every job. It is not the claim CI rests
on, because a job runs identities rather than files, and a file can be named by a shard while the
tests inside it are collected by nobody: a marker expression that deselects more than the suite's
own, a path spelled for a file that moved, a parametrization that only exists when something else
is imported first.

So this holds the selections against the suite itself, in identities. The whole tests tree is
collected once, each selection is collected once, and they are held against the one: nothing the
suite collects is missing from all of them, nothing appears in two of them, and nothing appears
in a shard that the suite does not collect at all. Collection only, so the audit costs seconds
rather than the run it describes.

:data:`~tests.ci_shards.GUARD` is the one deliberate repetition and is checked as such: every
selection carries it, because a check made in one job is a check the other jobs never made, so
its identities are expected in every selection and in exactly every one of them.

This file runs in one job rather than in every job, and the reason is the tree it collects. Ten
modules bind a pinned upstream source while they are being collected, so the whole tree collects
only where those sources are prepared, which is the job that prepares them. A job that prepares
nothing cannot collect the suite at all, and a check that quietly skipped there would be worth
less than none.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import pytest

from tests.ci_shards import (
    GUARD,
    SHARDS,
    TESTS_DIR,
    CollectionFailed,
    collect_ids,
    listing,
)

#: What a collection that did not finish usually means here, said where it will be read.
_PREPARED = (
    "A tree that cannot be collected is not an empty tree. The whole suite collects only where "
    "the pinned upstream sources are prepared, so this file belongs to a shard whose assets "
    "include them; see tests/ci_shards.py."
)


@pytest.fixture(scope="module")
def suite() -> Tuple[str, ...]:
    """Every identity the suite collects, which is the one thing the selections are held against."""
    try:
        return collect_ids((TESTS_DIR,))
    except CollectionFailed as failure:
        pytest.fail(f"{failure}\n\n{_PREPARED}")


@pytest.fixture(scope="module")
def by_shard() -> Dict[str, Tuple[str, ...]]:
    """Every identity each shard's selection collects, by shard name."""
    collected: Dict[str, Tuple[str, ...]] = {}
    for candidate in SHARDS:
        try:
            collected[candidate.name] = collect_ids(candidate.selection())
        except CollectionFailed as failure:
            pytest.fail(f"the {candidate.name} selection did not collect: {failure}\n\n{_PREPARED}")
    return collected


def test_the_jobs_together_collect_every_test_the_suite_collects(
    suite: Tuple[str, ...], by_shard: Dict[str, Tuple[str, ...]]
) -> None:
    """A test the suite collects and no shard collects is a test CI stopped running."""
    sharded = {node for ids in by_shard.values() for node in ids}
    missing = set(suite) - sharded
    assert not missing, (
        f"collected by the suite and by no shard, so CI would run them nowhere: "
        f"{listing(missing)}. That is {len(missing)} of the suite's {len(suite)} tests. Name "
        "each one's file in a shard in tests/ci_shards.py."
    )


def test_the_jobs_collect_nothing_the_suite_does_not(
    suite: Tuple[str, ...], by_shard: Dict[str, Tuple[str, ...]]
) -> None:
    """A test a shard collects and the suite does not is a job running something else."""
    for name, ids in by_shard.items():
        strayed = set(ids) - set(suite)
        assert not strayed, (
            f"collected by the {name} selection and not by the suite: {listing(strayed)}. A shard "
            "names files the suite collects, so a path here reaches outside the tests tree or "
            "under a marker the suite does not run."
        )


def test_no_test_is_collected_by_two_jobs(by_shard: Dict[str, Tuple[str, ...]]) -> None:
    """A test in two selections is a test two jobs pay for, and the guard is the one exception."""
    times = Counter(node for ids in by_shard.values() for node in ids)
    guard = {node for node in times if node.split("::")[0] == GUARD}

    twice = {node for node, count in times.items() if count > 1 and node not in guard}
    assert not twice, (
        f"collected by more than one shard, so CI would run them twice: {listing(twice)}. A file "
        "belongs to one shard's paths in tests/ci_shards.py."
    )

    everywhere = sorted(node for node in guard if times[node] != len(SHARDS))
    assert not everywhere, (
        f"the guard is the one file every job runs, and these were not collected by all "
        f"{len(SHARDS)} selections: {listing(everywhere)}. Every shard's selection carries "
        f"{GUARD}."
    )
