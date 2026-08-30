"""Serve one shogym task to Claude Code over stdio MCP.

Claude Code spawns this file as an MCP server (see ``.mcp.json``); it never spawns Claude Code.
What it publishes is one generation of the protocol v2 stream: ``pull`` hands out the work, every
env tool is wrapped so each call names the attempt it belongs to, and the terminal tool is
intercepted and sealed by the stream instead of reaching the env. The agent is not told which
task it played: a task record carries an attempt id and a body, and has no field an index or a
target could be written into.

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

# Where the generation keeps its blobs and the manifest a later owner would resume it from.
# Beside this file by default, which is what a quickstart wants; somewhere the agent is not
# working when the run is one whose record you intend to defend.
#     SHOGYM_RUNS=~/somewhere-else/runs <your harness command>
RUNS = (
    Path(os.environ["SHOGYM_RUNS"]).expanduser()
    if os.environ.get("SHOGYM_RUNS")
    else Path(__file__).resolve().parent / "runs"
)
# --------------------------------------------------------------------------------------------


def new_run_dir(env: str = ENV, runs: Path = RUNS, task: int = TASK) -> Path:
    """A fresh directory for one generation.

    Fresh per launch on purpose. The manifest is written once and never rewritten, because a
    resume compares against it, so a directory that already holds one is refused rather than
    added to."""
    return runs / f"{env}-{task}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


async def main() -> None:
    run_dir = new_run_dir()
    # stdout is the MCP wire, so everything this process says goes to stderr.
    print(f"[shogym] serving {ENV} task {TASK} -> {run_dir}", file=sys.stderr)
    await run_stdio_v2(ENV, task=TASK, run_directory=run_dir)


if __name__ == "__main__":
    asyncio.run(main())
