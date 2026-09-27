"""The selections CI runs the suite in are still the whole suite.

CI runs the suite as one job per selection in :mod:`tests.ci_shards`. What that buys is wall
time. What it risks is a test that quietly stops running: a file no selection names is a file
no job runs, and every job still reports success. The check against that is here.

It runs in every job rather than in one, because a check the other jobs never made is a check
made too seldom, and because the job whose selection went wrong is the one that has to say so.
That is why :data:`~tests.ci_shards.GUARD` names this file and every shard's selection carries
it.

The claim being checked here is about the files: every test file on disk is named by exactly one
shard, so the union of the selections is the whole tree with nothing missed, and no file is
named twice, so no test is collected in two jobs and paid for twice. Beside that, what this job
collected is held against what its shard's selection collects, identity for identity, so a run
given something other than its shard is a failure in the job that was given it.

That second check is an equality rather than a membership, because the failure worth catching is
the quiet one. A run handed fewer paths, a narrower ``-k`` or another marker expression collects
less than its shard names, and what it dropped no other job collects either: those tests stopped
running while every green job said the suite passed. Reading only whether what it collected
belongs to its shard calls that run correct.

A job that runs its tests in several processes collects the same identities under other names: a
pytest-xdist worker under ``--dist loadgroup`` appends ``@`` and the group to the id of every test
it groups. So the job's collection is read as the ids pytest collected, which
tests/_fixtures/node_ids.py keeps before any renaming, and the equality is the same equality.
:mod:`tests.test_ci_scheduling` holds this to a grouped run that collected everything and to one
that collected less.

The claim in identities across every job at once is :mod:`tests.test_ci_partition`, which
collects the whole tests tree and each selection and holds them against the one. That
needs a tree that collects, which is a job with the pinned upstream sources prepared, so it runs
in one job while this file runs in every one.

Last, it holds which jobs run their tests on several workers and that each job is told how many,
because a module that drives the test server, forks the interpreter or edits the package's source
may not share a job with workers.

A new test file that no shard claims fails the first test below, in every job, naming the
file. Adding it to a shard in ``tests/ci_shards.py`` is the fix, and which shard to add it to is
a question about where its seconds should go.
"""

from __future__ import annotations

import os

import pytest

from tests._fixtures.node_ids import collected_as
from tests.ci_shards import (
    DURABLE_SERVICE,
    GUARD,
    SHARDS,
    collect_ids,
    main,
    named_paths,
    owners,
    selection_problems,
    shard,
    suite_test_files,
)

#: Test files that may only run in a shard without workers, and why. The modules that drive the
#: Temporal test server are not listed, because the asset they all need holds them: see
#: :func:`test_only_the_receipt_selections_run_with_workers`.
_ONE_PROCESS_ONLY = {
    "tests/envs/test_receipts_pin_drift.py": "it rewrites pinned source files every worker reads",
    "tests/test_serve_session_lifecycle.py": "it forks the interpreter",
}


def test_every_test_file_is_named_by_exactly_one_shard() -> None:
    """The union of the selections is the whole tests tree, and it is a partition of it."""
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
    second for most of the selections, seconds for the one holding most of the files.

    Without that variable there is no shard to be, which is every run a developer makes. What is
    left to read off the collection is then the weaker claim it can support: every file in it is
    named by exactly one selection, the guard excepted, which every selection carries.

    Each test is read by the id it was collected under rather than by ``nodeid``, because a
    parallel worker renames a grouped test and a serial collection does not. A parallel worker
    also holds the whole collection rather than the share it was sent, so this reads the same
    job whichever worker runs it.
    """
    collected = [collected_as(item) for item in request.session.items]
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

    problems = selection_problems(name, collected, collect_ids(shard(name).selection()))
    assert not problems, "\n".join(problems)


def test_exactly_one_shard_carries_the_lint_and_type_checks() -> None:
    """Two shards carrying them pays twice for one answer, and none carrying them asks nobody."""
    carrying = [candidate.name for candidate in SHARDS if candidate.quality]
    assert len(carrying) == 1, f"ruff and pyright run in {carrying}, and they run in one job"

    names = [candidate.name for candidate in SHARDS]
    assert len(set(names)) == len(names), f"two shards share a name: {names}"
    empty = [candidate.name for candidate in SHARDS if not candidate.paths]
    assert not empty, f"a shard with no paths is a job with nothing to run: {empty}"


def test_only_the_receipt_selections_run_with_workers() -> None:
    """The three receipt selections run four workers each, and every other shard one process.

    A durable test holds real time timeouts, a fork copies a process whatever its other threads
    are doing, and a test that edits the package's source moves the code pin under every test that
    computes it, so none of them may share a job with workers. The durable tests are held by the
    one asset they all need: a shard that prepares the test server runs in one process. This fails
    if a worker count is not a whole number, if a receipt selection does not run four, if any
    other shard runs workers, or if a file listed above is in a shard that does.
    """
    for candidate in SHARDS:
        assert isinstance(candidate.workers, int) and candidate.workers >= 0, candidate
        if DURABLE_SERVICE in candidate.assets:
            assert candidate.workers == 0, f"{candidate.name} starts the test server on workers"
    running = {candidate.name: candidate.workers for candidate in SHARDS if candidate.workers}
    assert running == {"receipt-attacks": 4, "receipt-serving": 4, "receipt-families": 4}
    for path, reason in _ONE_PROCESS_ONLY.items():
        held = [shard(name) for name in owners(path)]
        assert held, f"{path} is named by no shard"
        crowded = [candidate.name for candidate in held if candidate.workers]
        assert not crowded, f"{path} runs with workers in {crowded}, and {reason}"


def test_each_job_is_told_its_worker_count(capsys: pytest.CaptureFixture[str]) -> None:
    """The command the workflow reads a shard with writes that shard's worker count, zero too.

    The step that runs pytest passes it to ``-n`` as it is written, so this fails if a shard's
    assignments leave the count out or write another one.
    """
    for candidate in SHARDS:
        assert main(["github-env", candidate.name]) == 0
        written = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
        assert written["SHOGYM_CI_SHARD"] == candidate.name
        assert written["SHOGYM_CI_WORKERS"] == str(candidate.workers)
