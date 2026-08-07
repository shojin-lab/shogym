"""Read a run's scores back out, independently of the harness that produced them.

Nothing here talks to Hermes, and nothing here scores anything. The scoring already happened,
server-side, inside the stream: the agent called a tool, the stream sealed the episode, ran the
env's verifier over it and appended a row. This file only reads those rows. That is the whole
point of the arrangement. **The harness cannot grade itself**, because the harness was never
handed a verdict to report.

Run it after a run::

    uv run python results.py               # the newest run under runs/
    uv run python results.py runs/<dir>    # a specific one
    uv run python results.py --verbose     # plus the env's verbatim feedback per task
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

from shogym.serve.stream import ResultRow, read_results, reconcile

RUNS = Path(__file__).resolve().parent / "runs"


def latest_run(runs: Path = RUNS) -> Path:
    """The most recent run directory, the one ``serve.py`` last created.

    A directory only counts once a task has actually been dispensed into it, which is what makes
    `hermes mcp test` harmless here: it connects, lists tools and disconnects without pulling a
    task, so it never lands between you and your last real run."""
    candidates = [p for p in runs.glob("*") if (p / "dispenses.jsonl").is_file()]
    if not candidates:
        raise SystemExit(f"no runs under {runs}; play some tasks first (see README.md)")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def rows(prov_dir: Path) -> List[ResultRow]:
    """Every dispensed task's outcome, in the order the stream handed them out.

    Two sources, and both are needed for the record to be complete:

    * ``read_results``: the rows the stream sealed and scored.
    * ``reconcile``: dispenses with no result, i.e. the stream was killed holding a task. Each
      becomes a ``broker_abort`` row with ``score=None``. A clean run yields none.
    """
    return sorted([*read_results(prov_dir), *reconcile(prov_dir)], key=lambda row: row.seq)


def report(prov_dir: Path, *, verbose: bool = False) -> Sequence[ResultRow]:
    recorded = rows(prov_dir)
    print(f"{prov_dir}  ({len(recorded)} tasks)\n")
    for row in recorded:
        # `score` is None whenever the outcome was not earned by the agent: a timeout, a killed
        # broker, a verifier that published nothing readable. `closure` says which. Keeping those
        # unscored rather than zero is why an infrastructure failure can never be averaged in.
        reward = row.score.reward if row.score is not None else None
        success = row.score.success if row.score is not None else None
        print(
            f"  #{row.seq:<3} {row.env}[{row.task_idx}]  {row.closure:<14} "
            f"reward={reward}  success={success}"
        )
        if row.diagnostic:
            print(f"        {row.diagnostic}")
        if verbose:
            # Everything the env published, verbatim, at every level. The two headline numbers
            # above are only a summary of this.
            for item in row.observed:
                print(f"        {item['level']:<9} {item['name']} = {item['value']!r}")

    scored = [row.score.reward for row in recorded if row.score and row.score.reward is not None]
    solved = [row.score.success for row in recorded if row.score and row.score.success is not None]
    print(f"\n  scored   {len(scored)}/{len(recorded)}")
    if scored:
        print(f"  reward   mean {sum(scored) / len(scored):.3f}")
    if solved:
        print(f"  success  {sum(solved)}/{len(solved)}")
    return recorded


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    verbose = "--verbose" in args
    positional = [a for a in args if not a.startswith("-")]
    report(Path(positional[0]) if positional else latest_run(), verbose=verbose)


if __name__ == "__main__":
    main()
