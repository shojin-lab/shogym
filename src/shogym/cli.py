"""The ``shogym`` command-line entrypoint.

One subcommand, ``shogym serve``, runs an env as a stdio MCP server any harness can spawn::

    shogym serve wordle_v1 --task 17 --trace ./shogym_logs/run.jsonl

The model asks for its work with ``pull`` and its terminal call is intercepted into one seal
transaction, so the stream decides when an attempt ends and the harness never grades itself.
Serving runs on Temporal, which ``pip install shogym`` installs; the import lives inside
:func:`main` rather than here, so reading the help costs nothing that serving costs.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional, Sequence


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
        help="directory for the stream's blobs and resume manifest (default: none)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        # Imported here, not above: serving runs on Temporal, and importing this module must
        # not pull that in for a caller who only wanted the parser.
        try:
            from shogym.serve.protocol_v2.gateway import run_stdio_v2
        except ModuleNotFoundError as missing:
            if missing.name != "temporalio":
                raise
            # An install that is missing it is an install that did not finish, answered as the
            # command that finishes it rather than as a traceback.
            parser.error(
                "serving needs temporalio, which this install does not have: "
                "`pip install shogym` installs it, and installs everything else serving "
                "needs, so run that and run this again"
            )

        asyncio.run(
            run_stdio_v2(
                args.env, task=args.task, trace_path=args.trace, run_directory=args.run_dir
            )
        )


if __name__ == "__main__":
    main()
