"""The ``hgym`` command-line entrypoint.

One subcommand — ``hgym serve`` — runs an env as a stdio MCP server any harness can spawn::

    hgym serve wordle_v1 --task 17 --trace ./hgym_logs/run.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
from typing import Optional, Sequence

from hgym.serve.server import run_stdio


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hgym")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="serve an env as a stdio MCP server")
    serve.add_argument("env", help="registered env name, e.g. wordle_v1")
    serve.add_argument("--task", default=None, help="task instance id/index (default: random)")
    serve.add_argument("--trace", default=None, help="JSONL trace path (default: no trace)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        asyncio.run(run_stdio(args.env, task=args.task, trace_path=args.trace))


if __name__ == "__main__":
    main()
