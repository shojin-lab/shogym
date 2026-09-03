"""The ``shogym`` command-line entrypoint.

``shogym serve`` runs an env as a stdio MCP server any harness can spawn::

    shogym serve wordle_v1 --task 17 --run-dir runs/one --trace ./shogym_logs/run.jsonl

The model asks for its work with ``pull`` and its terminal call is intercepted into one seal
transaction, so the stream decides when an attempt ends and the harness never grades itself.

``shogym results`` reads a run directory back afterwards::

    shogym results runs/one

That prints one row per attempt and leaves the same rows in the directory as a derived file.
Both subcommands run on Temporal, which ``pip install shogym`` installs; the imports live inside
:func:`main` rather than here, so reading the help costs nothing that running costs.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import NoReturn, Optional, Sequence


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shogym")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="serve an env as a stdio MCP server")
    serve.add_argument("env", help="registered env name, e.g. wordle_v1")
    serve.add_argument("--task", default=None, help="task instance id/index (default: random)")
    serve.add_argument("--trace", default=None, help="JSONL trace path (default: no trace)")
    serve.add_argument(
        "--run-dir",
        default=None,
        help=(
            "directory for the stream's blobs, resume manifest, and finalization records "
            "(default: none)"
        ),
    )

    results = sub.add_parser("results", help="read a run directory's attempt records")
    results.add_argument("run_dir", help="a directory a stream was served with --run-dir")
    return parser


def _install_temporal(parser: argparse.ArgumentParser, missing: ModuleNotFoundError) -> NoReturn:
    """Answer an install that is missing Temporal with the command that finishes it.

    Anything else that failed to import is a fault and is raised: only the one dependency this
    package installs for itself has an instruction that would help.
    """
    if missing.name != "temporalio":
        raise missing
    parser.error(
        "this needs temporalio, which this install does not have: `pip install shogym` "
        "installs it, and installs everything else this needs, so run that and run this again"
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        # Imported here, not above: serving runs on Temporal, and importing this module must
        # not pull that in for a caller who only wanted the parser.
        try:
            from shogym.serve.protocol_v2.gateway import run_stdio_v2
        except ModuleNotFoundError as missing:
            _install_temporal(parser, missing)

        asyncio.run(
            run_stdio_v2(
                args.env, task=args.task, trace_path=args.trace, run_directory=args.run_dir
            )
        )
    elif args.command == "results":
        try:
            from shogym.serve.protocol_v2.reader import (
                NothingToRead,
                ReadRefused,
                format_records,
                read_records,
                write_records,
            )
            from shogym.serve.protocol_v2.rundir import ResumeRefused
        except ModuleNotFoundError as missing:
            _install_temporal(parser, missing)

        try:
            run = asyncio.run(read_records(args.run_dir))
        except NothingToRead as empty:
            # A run that kept no history is an answer, not a failure: the command says which
            # half is missing and leaves the directory exactly as it found it.
            print(f"nothing to read: {empty}")
            return
        except (ResumeRefused, ReadRefused) as refused:
            # A refusal is a different fact. It covers a directory holding no manifest, one
            # holding a manifest this code cannot read, one holding the retired protocol's
            # logs, and a run holding work reading it would be what applied. A caller that
            # mistyped a path is the common way to reach the first three. Answering any of them
            # with a success status would let a script collect no results and carry on as
            # though it had.
            parser.exit(1, f"cannot read {args.run_dir}: {refused}\n")
        print(format_records(run.records))
        print(f"wrote {write_records(run)}")


if __name__ == "__main__":
    main()
