"""The four selections CI runs the suite in are still the whole suite.

CI runs the suite as four jobs over the four selections in :mod:`tests.ci_shards`. What that buys
is wall time. What it risks is a test that quietly stops running: a file no selection names is a
file no job runs, and every job still reports success. The check against that is here.

It runs in every job rather than in one, because a check the other three jobs never made is a
check made three times too seldom, and because the job whose selection went wrong is the one that
has to say so. That is why :data:`~tests.ci_shards.GUARD` names this file and every shard's
selection carries it.

The claim being checked here is about the files: every test file on disk is named by exactly one
shard, so the union of the four selections is the whole tree with nothing missed, and no file is
named twice, so no test is collected in two jobs and paid for twice. Beside that, what this job
collected is held against what its shard's selection collects, identity for identity, so a run
given something other than its shard is a failure in the job that was given it.

That second check is an equality rather than a membership, because the failure worth catching is
the quiet one. A run handed fewer paths, a narrower ``-k`` or another marker expression collects
less than its shard names, and what it dropped no other job collects either: those tests stopped
running while four green jobs said the suite passed. Reading only whether what it collected
belongs to its shard calls that run correct.

The claim in identities across all four jobs at once is :mod:`tests.test_ci_partition`, which
collects the whole tests tree and the four selections and holds the four against the one. That
needs a tree that collects, which is a job with the pinned upstream sources prepared, so it runs
in one job while this file runs in every one.

A new test file that no shard claims fails the first test below, in all four jobs, naming the
file. Adding it to a shard in ``tests/ci_shards.py`` is the fix, and which shard to add it to is
a question about where its seconds should go.
"""

from __future__ import annotations

import os
from collections import Counter

import pytest

from tests.ci_shards import (
    GUARD,
    SHARDS,
    collect_ids,
    listing,
    named_paths,
    owners,
    shard,
    suite_test_files,
)


def test_every_test_file_is_named_by_exactly_one_shard() -> None:
    """The union of the four selections is the whole tests tree, and it is a partition of it."""
    named = named_paths()
    twice = sorted({path for path in named if named.count(path) > 1})
    assert not twice, (
        f"named by more than one shard, so CI would run them twice: {twice}. "
        "A file belongs to one shard's paths."
    )

    on_disk = set(suite_test_files())
    unclaimed = sorted(on_disk - set(named))
    assert not unclaimed, (
        f"collected by the suite and named by no shard, so CI would run them nowhere: "
        f"{unclaimed}. Add each to a shard in tests/ci_shards.py."
    )

    gone = sorted(set(named) - on_disk)
    assert not gone, (
        f"named by a shard and absent from the tests tree, so a rename left a selection behind: "
        f"{gone}. Update tests/ci_shards.py."
    )


def test_this_job_collected_exactly_what_its_shard_names(request: pytest.FixtureRequest) -> None:
    """What this run collected is what its shard's selection collects, identity for identity.

    The job announces its shard in ``SHOGYM_CI_SHARD``, and that shard's selection is collected
    here, in a subprocess, so the comparison is against a fresh reading of the partition rather
    than against the run's own idea of itself. It costs a collection and no run: fractions of a
    second for three of the four selections, seconds for the one holding most of the files.

    Without that variable there is no shard to be, which is every run a developer makes. What is
    left to read off the collection is then the weaker claim it can support: every file in it is
    named by exactly one selection, the guard excepted, which every selection carries.
    """
    collected = [item.nodeid for item in request.session.items]
    assert collected, "this run collected nothing, so it proves nothing about the partition"

    files = sorted({node.split("::")[0] for node in collected})
    unclaimed = sorted(path for path in files if not owners(path))
    assert not unclaimed, (
        f"collected here and named by no shard, so CI would run them nowhere: {unclaimed}. "
        "Add each to a shard in tests/ci_shards.py."
    )
    twice = sorted(path for path in files if path != GUARD and len(owners(path)) > 1)
    assert not twice, f"named by more than one shard: {twice}"

    name = os.environ.get("SHOGYM_CI_SHARD")
    if not name:
        return

    repeated = [node for node, times in Counter(collected).items() if times > 1]
    assert not repeated, (
        f"the {name} job collected the same test more than once, and paid for it twice: "
        f"{listing(repeated)}. Its pytest run was given a path more than once."
    )
    expected = set(collect_ids(shard(name).selection()))
    missing = expected - set(collected)
    strayed = set(collected) - expected
    assert not missing and not strayed, (
        f"the {name} job did not collect its shard's selection. Named by the shard and not "
        f"collected here, so run in no job at all: {listing(missing)}. Collected here and not "
        f"named by the shard, so run twice or in the wrong one: {listing(strayed)}. Its pytest "
        "run was given something other than the shard's selection."
    )


def test_exactly_one_shard_carries_the_lint_and_type_checks() -> None:
    """Two shards carrying them pays twice for one answer, and none carrying them asks nobody."""
    carrying = [candidate.name for candidate in SHARDS if candidate.quality]
    assert len(carrying) == 1, f"ruff and pyright run in {carrying}, and they run in one job"

    names = [candidate.name for candidate in SHARDS]
    assert len(set(names)) == len(names), f"two shards share a name: {names}"
    empty = [candidate.name for candidate in SHARDS if not candidate.paths]
    assert not empty, f"a shard with no paths is a job with nothing to run: {empty}"
