"""The four selections CI runs the suite in, in the one place the workflow and a test both read.

CI is the suite: one job running every test takes about forty five minutes, and ninety nine
percent of that is pytest. The seconds are not spread evenly over the tree. One offline run
measured 1402s in total, of which two receipt modules and one frontier_bench module hold more
than half. Splitting by count would therefore split nothing, so the suite is split by module into
four selections, and each is a job that checks out, installs, prepares only the assets its own
modules need and runs only its own paths. The run then costs what its slowest shard costs
instead of what all four cost together.

The partition below is that measurement, in the seconds it printed for phases of a second or
more (1316s of the 1402s; the remaining 86s is thousands of shorter phases and collection, which
falls mostly on the shard holding the most files, so the numbers below understate ``rest``):

    receipt-attacks   350s   the attack surface of a bundle, and the cells a seal publishes
    frontier          311s   the vendored task oracles, their containers and their verifiers
    receipt-serving   363s   the receipt environment, its bank, and the operator's CLI
    rest              292s   every other module, which is most of the files and few of the seconds

The selections live here rather than in the workflow because two readers need the same answer.
The workflow asks which paths a shard runs, which assets it needs prepared and whether it carries
the lint and type checks. :mod:`tests.test_ci_shards` asks for the same partition and holds it
against the test files on disk and against what its own job collected, so a file that no shard
names fails a test in every job rather than quietly running in none of them.

Two rules keep the partition honest, and both are checked there rather than trusted here: no file
is named by two shards, so nothing is paid for twice, and every test file on disk is named by
exactly one, so nothing is lost. :data:`GUARD` is the single deliberate exception, and it is the
file holding those checks: a check that ran in one job is a check the other three never made.

Files are what a shard names, and identities are what a job runs, so the deeper claim is made in
identities: :mod:`tests.test_ci_partition` collects the whole tests tree and each of the four
selections through :func:`collect_ids` and holds the four against the one. It runs where the
whole tree can be collected, which is the job that prepares the pinned upstream sources.

This module imports the standard library only. The workflow reads it with the runner's own
interpreter, before the project is installed, and a shard that is renamed or added here reaches
CI without a second edit to the workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

#: The repository root, taken from this file rather than from a working directory, because the
#: workflow runs this module by path and the suite imports it.
REPO_ROOT = Path(__file__).resolve().parents[1]

#: The directory the suite is collected from, which is pytest's ``testpaths``.
TESTS_DIR = "tests"

#: The file patterns pytest collects a test module from, which is its ``python_files`` default and
#: what this project leaves it as. The roster on disk is read with these, so "every test file" here
#: means the same set pytest means.
TEST_FILE_PATTERNS = ("test_*.py", "*_test.py")

#: The one test file every job runs: it is the check that these four selections are still the whole
#: suite, and it is in no shard's ``paths`` so that the partition stays a partition.
GUARD = "tests/test_ci_shards.py"

#: The marker expression the offline suite is selected with, which the workflow's pytest step
#: spells for itself. A collection made here is made with this one and compared against what the
#: job collected, so the two spellings drifting apart is a failed check in every job rather than a
#: difference nobody sees.
SUITE_MARKER = "not network"

# The asset groups tests/prepare_offline_suite.py provisions, named here because a shard declares
# which ones its modules need and its job prepares those and nothing else. The receipt shards need
# no upstream at all; only the frontier shard needs a task image built.
SOURCES = "sources"
TAU2_DATA = "tau2-data"
DURABLE_SERVICE = "durable-service"
TASK_IMAGES = "task-images"


@dataclass(frozen=True)
class Shard:
    """One job's share of the suite: what it runs, what it needs prepared, what else it carries."""

    #: The job's name, which is also how the workflow's matrix names it.
    name: str
    #: The test files this shard, and only this shard, is responsible for.
    paths: Tuple[str, ...]
    #: The asset groups this shard's own modules need provisioned before they run.
    assets: Tuple[str, ...] = ()
    #: Whether this job also runs ruff and pyright. Exactly one shard does.
    quality: bool = False

    def selection(self) -> Tuple[str, ...]:
        """The paths this shard's pytest run is given: its own files, plus the guard."""
        return self.paths + (GUARD,)


SHARDS: Tuple[Shard, ...] = (
    # The heaviest single module in the suite, and the artifact module beside it: both build
    # receipt bundles and neither needs an upstream source, a task image or a test server, so this
    # job installs and starts testing. That is why it carries ruff and pyright: it is the first
    # job to be free, while the frontier shard is still building images.
    Shard(
        name="receipt-attacks",
        paths=(
            "tests/envs/test_receipts_artifact.py",
            "tests/envs/test_receipts_attacks.py",
        ),
        quality=True,
    ),
    # The one module that runs real task containers. It is alone because its preparation is the
    # expensive one: five environment images, five verifier images and one oracle's packages.
    Shard(
        name="frontier",
        paths=("tests/envs/test_frontier_bench_served.py",),
        assets=(TASK_IMAGES,),
    ),
    # The other side of the receipts port: the served environment, the bank underneath it and the
    # CLI an operator reaches both through. These pay for bundle construction the way the attacks
    # module does, so they are balanced against it rather than added to it.
    Shard(
        name="receipt-serving",
        paths=(
            "tests/envs/test_receipts_bank.py",
            "tests/envs/test_receipts_cli.py",
            "tests/envs/test_receipts_protocol_v2.py",
            "tests/envs/test_receipts_served.py",
        ),
    ),
    # Everything else, which is most of the files and few of the seconds. It is the only shard
    # whose modules bind a pinned upstream, read tau2's domains or start a test server, so it is
    # the only one that prepares them, and for the same reason it is the one job in which the
    # whole tests tree can be collected: the identity audit is therefore one of its files.
    Shard(
        name="rest",
        paths=(
            "tests/envs/test_automationbench_fidelity.py",
            "tests/envs/test_automationbench_protocol_v2.py",
            "tests/envs/test_automationbench_served.py",
            "tests/envs/test_automationbench_verify.py",
            "tests/envs/test_browsecomp_plus_data.py",
            "tests/envs/test_browsecomp_plus_fidelity.py",
            "tests/envs/test_browsecomp_plus_judge.py",
            "tests/envs/test_browsecomp_plus_metrics.py",
            "tests/envs/test_browsecomp_plus_served.py",
            "tests/envs/test_browsecomp_plus_verify.py",
            "tests/envs/test_frontier_bench_docker_build.py",
            "tests/envs/test_frontier_bench_docker_cpus.py",
            "tests/envs/test_frontier_bench_exploit.py",
            "tests/envs/test_frontier_bench_fidelity.py",
            "tests/envs/test_frontier_bench_verify.py",
            "tests/envs/test_hle_data.py",
            "tests/envs/test_hle_fidelity.py",
            "tests/envs/test_hle_judge.py",
            "tests/envs/test_hle_served.py",
            "tests/envs/test_hle_verify.py",
            "tests/envs/test_receipts_artifact_carry.py",
            "tests/envs/test_receipts_artifact_route.py",
            "tests/envs/test_receipts_checks.py",
            "tests/envs/test_receipts_read_back.py",
            "tests/envs/test_receipts_recovery.py",
            "tests/envs/test_receipts_render.py",
            "tests/envs/test_tau2_domains_served.py",
            "tests/envs/test_tau2_fidelity.py",
            "tests/envs/test_tau2_mock_served.py",
            "tests/envs/test_tau2_seal.py",
            "tests/envs/test_tau2_verify.py",
            "tests/envs/test_upstream_provisioning.py",
            "tests/envs/test_wordle_verify.py",
            "tests/envs/test_yc_bench_protocol_v2.py",
            "tests/envs/test_yc_bench_served.py",
            "tests/envs/test_yc_bench_verify.py",
            "tests/mcp/test_in_process.py",
            "tests/mcp/test_stdio.py",
            "tests/mcp/test_types.py",
            "tests/test_automationbench_cell.py",
            # The identity audit, here because this is the job that prepares the pinned upstream
            # sources, and the whole tests tree only collects where they are.
            "tests/test_ci_partition.py",
            "tests/test_cli.py",
            "tests/test_durable_service.py",
            "tests/test_env_grading.py",
            "tests/test_episode_horizon.py",
            "tests/test_evaluate.py",
            "tests/test_feedback_wire.py",
            "tests/test_protocol_v2_artifact.py",
            "tests/test_protocol_v2_artifact_carry.py",
            "tests/test_protocol_v2_blobs.py",
            "tests/test_protocol_v2_finalize.py",
            "tests/test_protocol_v2_gateway.py",
            "tests/test_protocol_v2_gateway_schedules.py",
            "tests/test_protocol_v2_kernel.py",
            "tests/test_protocol_v2_operation_failures.py",
            "tests/test_protocol_v2_policy.py",
            "tests/test_protocol_v2_reader.py",
            "tests/test_protocol_v2_receipt_policies.py",
            "tests/test_protocol_v2_receipt_read_back.py",
            "tests/test_protocol_v2_recovery.py",
            "tests/test_protocol_v2_resume.py",
            "tests/test_protocol_v2_schedule_model.py",
            "tests/test_protocol_v2_scheduled_stream.py",
            "tests/test_protocol_v2_turnover.py",
            "tests/test_protocol_v2_wire.py",
            "tests/test_quickstart_claude_code.py",
            "tests/test_quickstart_codex.py",
            "tests/test_quickstart_hermes.py",
            "tests/test_quickstart_pi.py",
            "tests/test_quickstart_prime_agent.py",
            "tests/test_receipts_gates.py",
            "tests/test_receipts_screen.py",
            "tests/test_seal_versions.py",
            "tests/test_serve_episode.py",
            "tests/test_serve_lifecycle_matrix.py",
            "tests/test_serve_server.py",
            "tests/test_serve_session_lifecycle.py",
            "tests/test_served_name.py",
            "tests/test_task_describe.py",
            "tests/test_terminal_lifecycle.py",
            "tests/test_trace_store.py",
            "tests/test_v1_runs.py",
        ),
        assets=(SOURCES, TAU2_DATA, DURABLE_SERVICE),
    ),
)


def names() -> List[str]:
    """Every shard's name, in the order the jobs are declared."""
    return [candidate.name for candidate in SHARDS]


def shard(name: str) -> Shard:
    """The shard called ``name``, or a ``KeyError`` naming the ones there are.

    A workflow asking for a shard that no longer exists is a job that would otherwise run
    nothing, so the refusal is the point.
    """
    for candidate in SHARDS:
        if candidate.name == name:
            return candidate
    raise KeyError(f"no shard named {name!r}; the shards are {', '.join(names())}")


def owners(path: str) -> List[str]:
    """The names of the shards whose selection includes ``path``, which should be exactly one.

    The guard is the exception and belongs to every selection, because every job runs it.
    """
    return [candidate.name for candidate in SHARDS if path in candidate.selection()]


def named_paths() -> List[str]:
    """Every test file the shards name, the guard included, with any repeat kept.

    Repeats are kept rather than collapsed so that a file named by two shards can be reported as
    the double payment it would be.
    """
    named = [path for candidate in SHARDS for path in candidate.paths]
    named.append(GUARD)
    return named


def suite_test_files() -> List[str]:
    """Every test file pytest would collect from the tests tree, as repository relative paths."""
    root = REPO_ROOT / TESTS_DIR
    found = {
        path.relative_to(REPO_ROOT).as_posix()
        for pattern in TEST_FILE_PATTERNS
        for path in root.rglob(pattern)
        if "__pycache__" not in path.parts
    }
    return sorted(found)


#: What pytest prints per collected test under ``-q --collect-only``: a test module's path, then
#: the identity inside it. A parametrized identity can hold spaces, so this anchors on the module
#: path rather than on the line having none, and a warning's ``path.py:12: SomeWarning`` misses it
#: because one colon is not two.
_COLLECTED = re.compile(r"^[\w./+-]+\.py::")

#: The line a collection ends with, either ``N tests collected`` or ``N/M tests collected``. Its
#: count is held against the identities parsed out of the lines above it, so a format this parser
#: stopped understanding is loud rather than an empty answer.
_COUNTED = re.compile(r"^(\d+)(?:/\d+)? tests? collected")


#: How long a collection is given, which is minutes for work that takes seconds. The alternative
#: to a bound is a runner sitting on a collection that will never finish until the job's own
#: timeout hours later, with nothing printed to say what it was waiting for.
_COLLECTION_SECONDS = 600


class CollectionFailed(RuntimeError):
    """A pytest collection that did not finish, carrying what it was asked and what it printed.

    A tree that cannot be collected is not a tree with no tests in it, and the difference is the
    whole point of the checks that call :func:`collect_ids`: an empty answer taken for an honest
    one would say every test in that tree is missing, or that none of them ever existed.
    """


@lru_cache(maxsize=None)
def collect_ids(paths: Tuple[str, ...]) -> Tuple[str, ...]:
    """The node ids ``paths`` collect under :data:`SUITE_MARKER`, in the order pytest collects them.

    A selection is the identities it collects rather than the files it names, because a job runs
    identities: a run given a file it does not name, or given a narrower expression than the one
    that reaches everything the file holds, differs from its shard in identities while its files
    still look right.

    Collection only, in a subprocess of this interpreter and from the repository root, so the
    answer costs a collection rather than the run it describes, and cached by paths because more
    than one check asks the same selection the same question. It writes nothing: no cache plugin
    and no bytecode, so a check that reads the tests tree does not leave anything in it.
    """
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--collect-only",
        "-m",
        SUITE_MARKER,
        *paths,
    ]
    try:
        finished = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            timeout=_COLLECTION_SECONDS,
        )
    except subprocess.TimeoutExpired as expired:
        raise CollectionFailed(
            f"collecting {' '.join(paths)} did not finish inside {_COLLECTION_SECONDS}s. A "
            "collection imports the test modules, so something one of them imports is waiting."
        ) from expired
    printed = finished.stdout.splitlines()
    ids = tuple(line for line in printed if _COLLECTED.match(line))
    counted: Optional[int] = None
    for line in printed:
        found = _COUNTED.match(line)
        if found:
            counted = int(found.group(1))
    if finished.returncode != 0 or counted != len(ids):
        tail = "\n".join((finished.stdout + finished.stderr).splitlines()[-20:])
        raise CollectionFailed(
            f"collecting {' '.join(paths)} did not finish: pytest exited {finished.returncode} "
            f"after printing {len(ids)} identities and counting {counted}. Its last lines were:\n"
            f"{tail}"
        )
    return ids


def listing(ids: Iterable[str], limit: int = 8) -> str:
    """A few identities and how many more there are, because a message nobody reads says nothing."""
    shown = sorted(ids)
    if not shown:
        return "none"
    if len(shown) <= limit:
        return ", ".join(shown)
    return ", ".join(shown[:limit]) + f", and {len(shown) - limit} more"


def _github_env(chosen: Shard) -> str:
    """The shard as environment assignments, for a step that appends them to ``GITHUB_ENV``.

    The paths are space separated because the step that runs pytest wants them split into
    arguments, and the assets are comma separated because the preparation script takes one value.
    An empty asset list is a shard that needs nothing prepared, and it has to stay distinguishable
    from a shard that said nothing at all.
    """
    return "\n".join(
        (
            f"SHOGYM_CI_SHARD={chosen.name}",
            f"SHOGYM_CI_PATHS={' '.join(chosen.selection())}",
            f"SHOGYM_CI_ASSETS={','.join(chosen.assets)}",
            f"SHOGYM_CI_QUALITY={'true' if chosen.quality else 'false'}",
        )
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="The selections CI runs the suite in.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("matrix", help="the shard names as a JSON matrix for the workflow")
    selection = commands.add_parser("selection", help="the paths one shard's pytest run is given")
    selection.add_argument("shard", choices=names())
    environment = commands.add_parser("github-env", help="one shard as GITHUB_ENV assignments")
    environment.add_argument("shard", choices=names())

    args = parser.parse_args(argv)
    if args.command == "matrix":
        print(json.dumps({"shard": names()}))
        return 0
    chosen = shard(args.shard)
    if args.command == "selection":
        print(" ".join(chosen.selection()))
        return 0
    print(_github_env(chosen))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
