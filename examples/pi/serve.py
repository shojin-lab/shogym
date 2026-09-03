"""Serve one shogym task to Pi over stdio MCP.

Pi spawns this file as an MCP server (see ``.pi/mcp.json``); it never spawns Pi. What it publishes
is one generation of the protocol v2 stream: ``pull`` hands out the work, every env tool is
wrapped so each call names the attempt it belongs to, and the terminal tool is intercepted and
sealed by the stream instead of reaching the env. The agent is not told which task it played: a
task record carries an attempt id and a body, and has no field an index or a target could be
written into.

Pi reaches this endpoint through the MCP bridge extension, which namespaces every served tool as
``mcp_shogym_<tool>``. The prefix is the harness's; the wire is plain MCP, and this file is
unaware of either.

One launch is one episode, over one env at one task, so three tasks are three launches. Serving
needs the durable extra (``uv sync --extra durable``, or ``pip install "shogym[durable]"``).
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from shogym.serve.protocol_v2.gateway import run_stdio_v2

# --------------------------------------------------------------------------------------------
# The one variable. Swapping envs is this line and nothing else: the endpoint, the wrapper shape
# and the sealing are all env-agnostic, and the tools the agent is handed come from whatever env
# this names.
#
#     automationbench   wordle_v1   hle   yc_bench   browsecomp_plus   frontier_bench
#     tau2_mock   tau2_airline   tau2_retail   tau2_telecom   tau2_banking_knowledge
#
# (`python -c "import shogym; print(shogym.registered_envs())"` prints the live catalogue. Some
# envs need an extra installed and a key; see their READMEs under src/shogym/envs/.)
# `SHOGYM_ENV` wins when it is set, so a run can swap envs without editing this file:
#     SHOGYM_ENV=wordle_v1 <your harness command>
ENV = os.environ.get("SHOGYM_ENV") or "automationbench"

# Which task this launch serves. One index, because one generation is one episode over one env.
#     SHOGYM_TASK=7 <your harness command>
TASK = int(os.environ.get("SHOGYM_TASK") or 0)
# --------------------------------------------------------------------------------------------

RUNS = Path(__file__).resolve().parent / "runs"


def new_run_dir(env: str = ENV, runs: Path = RUNS, task: int = TASK) -> Path:
    """A fresh directory for one generation, holding its blobs and its resume manifest.

    Fresh per launch on purpose. The manifest is written once and never rewritten, because a
    resume compares against it, so a directory that already holds one is refused rather than
    added to."""
    return runs / f"{env}-{task}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


async def main() -> None:
    run_dir = new_run_dir()
    # stdout is the MCP wire, so everything this process says goes to stderr. Pi keeps the last
    # lines of it per server; `/mcp` in a session prints them.
    print(f"[shogym] serving {ENV} task {TASK} -> {run_dir}", file=sys.stderr)
    await run_stdio_v2(ENV, task=TASK, run_directory=run_dir)


if __name__ == "__main__":
    asyncio.run(main())
