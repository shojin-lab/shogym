"""The ``shogym`` command-line entrypoint.

One subcommand, ``shogym serve``, runs an env as a stdio MCP server any harness can spawn::

    shogym serve wordle_v1 --task 17 --trace ./shogym_logs/run.jsonl

``--protocol v2`` serves the same env through the durable stream instead, where the model asks
for its work with ``pull`` and its terminal call is intercepted into one seal transaction. That
path needs the ``durable`` extra and is imported only when it is asked for, so the default
install and the default protocol still start an episode with nothing else installed.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional, Sequence

from shogym.serve.server import run_stdio


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shogym")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="serve an env as a stdio MCP server")
    serve.add_argument("env", help="registered env name, e.g. wordle_v1")
    serve.add_argument("--task", default=None, help="task instance id/index (default: random)")
    serve.add_argument("--trace", default=None, help="JSONL trace path (default: no trace)")
    serve.add_argument(
        "--protocol",
        choices=("v1", "v2"),
        default="v1",
        help="serving protocol (default: v1; v2 needs the `durable` extra)",
    )
    serve.add_argument(
        "--run-dir",
        default=None,
        help="v2 only: directory for the stream's blobs and resume manifest (default: none)",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        if args.run_dir and args.protocol != "v2":
            parser.error("--run-dir belongs to protocol v2, which keeps blobs and a manifest")
        if args.protocol == "v2":
            # Imported here, not above: protocol v2 runs on Temporal, and the quickstart path
            # must not import it to serve an episode.
            from shogym.serve.protocol_v2.gateway import run_stdio_v2

            asyncio.run(
                run_stdio_v2(
                    args.env, task=args.task, trace_path=args.trace, run_directory=args.run_dir
                )
            )
        else:
            asyncio.run(run_stdio(args.env, task=args.task, trace_path=args.trace))


if __name__ == "__main__":
    main()
