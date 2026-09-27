"""The CI guard reads a parallel job by the ids its tests were collected under.

A job that runs its tests in several processes under ``--dist loadgroup`` still collects every
test its shard names, and each pytest-xdist worker renames the tests that carry an ``xdist_group``
mark: ``path::test`` becomes ``path::test@group``. A serial collection of the shard renames
nothing. The guard in tests/test_ci_shards.py holds the two against each other, so a guard that did
not read through the renaming would call every grouped test missing under one name and a stray
under the other.

These checks hold the guard to that in both directions, in real grouped runs. In a small tree
built here, a grouped run that collected everything passes, and its workers did rename, so the
names they gave would have failed; a parameter id holding an ``@`` of its own is compared as it
was collected; and a run that collected less is refused, naming exactly what it dropped. On the
receipt selections themselves, with ``SHOGYM_CI_SHARD`` set the way a job sets it, the guard passes
a grouped run of the whole selection and refuses a narrowed one.

Each run is its own pytest, in a subprocess, with two workers. The collections the guard makes
inside those runs stay serial. They run once, in one job, because what they check is the guard's
code rather than any job's selection.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tests.ci_shards import (
    GUARD,
    REPO_ROOT,
    SUITE_MARKER,
    collect_ids,
    selection_problems,
    shard,
)

#: The selections checked here under the grouped scheduler, which are the receipt selections: the
#: ones that scheduler is for.
SCHEDULED = ("receipt-attacks", "receipt-serving")

#: The guard's equality, by node id.
_GUARD_TEST = f"{GUARD}::test_this_job_collected_exactly_what_its_shard_names"

#: How long one of these runs is given. Each is a collection and one short test, so this is minutes
#: for seconds of work, and a bound rather than a runner waiting on a stuck child until the job's
#: own timeout.
_RUN_SECONDS = 600

#: The small tree's tests: one group, a test in two groups, a test in a class, and a parameter id
#: holding an ``@`` both inside a group and outside one.
_GROUPED = """
import pytest

ONE = pytest.mark.xdist_group("one")


@ONE
def test_grouped():
    pass


@ONE
@pytest.mark.parametrize("value", ["a@b", "plain"])
def test_grouped_parameter(value):
    pass


@pytest.mark.xdist_group("two")
@ONE
def test_in_two_groups():
    pass


@pytest.mark.parametrize("value", ["c@d"])
def test_ungrouped_parameter(value):
    pass


class TestGrouped:
    @ONE
    def test_method(self):
        pass
"""

#: The small tree's guard: the real guard's reading and comparison, held against the tree's own
#: selection. It also writes down the ids its run holds after any renaming, so that a check can
#: see the renaming happened.
_TREE_GUARD = """
import json
from pathlib import Path

from tests._fixtures.node_ids import collected_as
from tests.ci_shards import collect_ids, selection_problems

ROOT = Path(__file__).resolve().parent
SELECTION = ("test_grouped.py", "test_guard.py")


def test_this_run_collected_its_selection(request):
    items = request.session.items
    (ROOT / "renamed.json").write_text(json.dumps([item.nodeid for item in items]))
    collected = [collected_as(item) for item in items]
    problems = selection_problems("tree", collected, collect_ids(SELECTION, root=ROOT))
    assert not problems, "\\n".join(problems)
"""

#: The small tree's selection.
_TREE_SELECTION = ("test_grouped.py", "test_guard.py")


def _tree(room: Path) -> Path:
    """Write the small tree under ``room`` and return its root, which is its own rootdir.

    The repository is on the tree's path, which is how its conftest and its guard reach the
    guard's code, and its markers are strict, as they are here.
    """
    root = room / "tree"
    root.mkdir()
    (root / "pytest.ini").write_text(
        f"[pytest]\npythonpath = {shlex.quote(str(REPO_ROOT))}\naddopts = --strict-markers\n",
        encoding="utf-8",
    )
    (root / "conftest.py").write_text(
        "from tests._fixtures.node_ids import pytest_collection_modifyitems  # noqa: F401\n",
        encoding="utf-8",
    )
    (root / "test_grouped.py").write_text(_GROUPED, encoding="utf-8")
    (root / "test_guard.py").write_text(_TREE_GUARD, encoding="utf-8")
    return root


def _pytest(
    cwd: Path, room: Path, *args: str, shard_name: str | None = None
) -> subprocess.CompletedProcess[str]:
    """One pytest run in a subprocess, carrying nothing of this run's own pytest.

    Its temporary directories are under ``room``, so it never touches this run's.
    """
    env = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_")}
    env.pop("SHOGYM_CI_SHARD", None)
    if shard_name is not None:
        env["SHOGYM_CI_SHARD"] = shard_name
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--basetemp={room / 'basetemp'}",
        "-m",
        SUITE_MARKER,
        *args,
    ]
    return subprocess.run(
        command, cwd=cwd, capture_output=True, text=True, env=env, timeout=_RUN_SECONDS
    )


def _said(finished: subprocess.CompletedProcess[str]) -> str:
    """The end of what a run printed, for a failure message."""
    return (finished.stdout + finished.stderr)[-6000:]


def test_a_grouped_run_that_collected_everything_passes_the_guard(tmp_path: Path) -> None:
    """The workers rename every grouped test, and the guard still finds the tree it collected.

    It fails if the grouped run is refused, if its workers did not rename the grouped tests or
    renamed another, if the names they gave would have passed the guard (then reading through them
    is not what made the run pass), or if a serial run of the same tree is refused or renamed.
    """
    root = _tree(tmp_path)
    expected = collect_ids(_TREE_SELECTION, root=root)

    grouped = _pytest(root, tmp_path, "-n", "2", "--dist", "loadgroup", *_TREE_SELECTION)
    assert grouped.returncode == 0, _said(grouped)
    renamed = json.loads((root / "renamed.json").read_text(encoding="utf-8"))
    assert set(renamed) - set(expected) == {
        "test_grouped.py::test_grouped@one",
        "test_grouped.py::test_grouped_parameter[a@b]@one",
        "test_grouped.py::test_grouped_parameter[plain]@one",
        "test_grouped.py::test_in_two_groups@one_two",
        "test_grouped.py::TestGrouped::test_method@one",
    }
    assert "test_grouped.py::test_ungrouped_parameter[c@d]" in renamed
    assert selection_problems("tree", renamed, expected)

    serial = _pytest(root, tmp_path, "-n", "0", *_TREE_SELECTION)
    assert serial.returncode == 0, _said(serial)
    assert json.loads((root / "renamed.json").read_text(encoding="utf-8")) == list(expected)


def test_a_grouped_run_that_collected_less_is_refused_by_what_it_dropped(tmp_path: Path) -> None:
    """A narrowed grouped run fails the guard, which names the dropped test as it was collected.

    The one test dropped is grouped and has an ``@`` in its parameter id. So this fails if the run
    passes, if the guard names the dropped test by a worker's name for it or cuts its parameter id
    at the ``@``, or if it calls a test the run did collect a stray.
    """
    root = _tree(tmp_path)
    narrowed = _pytest(
        root,
        tmp_path,
        "-n",
        "2",
        "--dist",
        "loadgroup",
        "--deselect",
        "test_grouped.py::test_grouped_parameter[a@b]",
        *_TREE_SELECTION,
    )
    assert narrowed.returncode == 1, _said(narrowed)
    assert (
        "Named by the shard and not collected here, so run in no job at all: "
        "test_grouped.py::test_grouped_parameter[a@b]. Collected here and not named by the shard, "
        "so run twice or in the wrong one: none."
    ) in narrowed.stdout, _said(narrowed)


def test_the_comparison_refuses_a_repeat_a_gap_and_a_stray() -> None:
    """Equality and nothing weaker: one id collected twice, one missing and one extra each fail.

    It fails if any of the three passes, or if the selection itself does not.
    """
    expected = ("a.py::one", "a.py::two[x@y]")
    assert selection_problems("job", list(expected), expected) == []
    repeated = selection_problems("job", [*expected, "a.py::one"], expected)
    assert len(repeated) == 1 and "more than once" in repeated[0]
    assert selection_problems("job", ["a.py::one"], expected)
    assert selection_problems("job", [*expected, "b.py::three"], expected)
    assert selection_problems("job", ["a.py::one", "a.py::two[x@y]@group"], expected)


@pytest.mark.parametrize("name", SCHEDULED)
def test_the_guard_passes_a_grouped_run_of_a_whole_receipt_selection(
    name: str, tmp_path: Path
) -> None:
    """A receipt job's whole selection, with its shard named, passes under the grouped scheduler.

    Every test is collected as the job collects it, and all but the guard's equality are skipped
    by tests/_fixtures/only_the_guard.py before their fixtures are set up, so the guard reads the
    job's whole collection while the run costs a collection. It fails if the guard refuses a job
    that collected its whole selection.
    """
    finished = _pytest(
        REPO_ROOT,
        tmp_path,
        "-p",
        "tests._fixtures.only_the_guard",
        "-n",
        "2",
        "--dist",
        "loadgroup",
        *shard(name).selection(),
        shard_name=name,
    )
    assert finished.returncode == 0, _said(finished)
    assert re.search(r"\b1 passed\b", finished.stdout), _said(finished)


@pytest.mark.parametrize("name", SCHEDULED)
def test_the_guard_refuses_a_grouped_run_of_a_narrowed_receipt_selection(
    name: str, tmp_path: Path
) -> None:
    """A receipt job given only the guard, with its shard named, is refused under the scheduler.

    It fails if the run passes, if the refusal is not the equality naming this shard, or if the
    guard calls the one test the run did collect a stray.
    """
    finished = _pytest(
        REPO_ROOT, tmp_path, "-n", "2", "--dist", "loadgroup", _GUARD_TEST, shard_name=name
    )
    assert finished.returncode == 1, _said(finished)
    assert f"the {name} job did not collect its shard's selection" in finished.stdout, _said(
        finished
    )
    assert "so run twice or in the wrong one: none." in finished.stdout, _said(finished)
