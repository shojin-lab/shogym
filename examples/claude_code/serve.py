"""Serve a queue of shogym tasks to Claude Code over stdio MCP.

Claude Code spawns this file as an MCP server (see ``.mcp.json``); it never spawns Claude Code.
What it publishes is a :class:`~shogym.serve.stream.TaskStream`: one endpoint that hands out tasks
one at a time (``get_task``), routes the env's own tools to whichever task is live, and, the part
that matters, **seals and scores each task itself**. The agent is never told its score. It is not
even told which task it played: ``get_task`` answers with ``{env, instructions, budget, tools}``
and has no field a task index or a target could be written into.

Every dispensed task lands exactly one durable row under ``runs/<env>-<stamp>/``. Read them back
with ``results.py`` once the run is over.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import shogym
from shogym.serve.stream import Immediate, TaskRef, TaskStream, build_stream_server

# --------------------------------------------------------------------------------------------
# The one variable. Swapping envs is this line and nothing else: the queue, the endpoint, the
# scoring and the readout are all env-agnostic, and the tools the agent is handed come from
# whatever env this names.
#
#     automationbench   wordle_v1   hle   yc_bench   browsecomp_plus   frontier_bench
#     tau2_mock   tau2_airline   tau2_retail   tau2_telecom   tau2_banking_knowledge
#
# (`python -c "import shogym; print(shogym.registered_envs())"` prints the live catalogue. Some envs
# need an extra installed and a key; see their READMEs under src/shogym/envs/.)
# `SHOGYM_ENV` wins when it is set, so a run can swap envs without editing this file:
#     SHOGYM_ENV=wordle_v1 <your harness command>
ENV = os.environ.get("SHOGYM_ENV") or "automationbench"

# Which tasks to serve, in order. A repeat is legal: a task's identity within a run is its
# position in this queue, not its index, so `[0, 0, 1]` plays task 0 twice and records both.
# `SHOGYM_TASKS` overrides it the same way: SHOGYM_TASKS=0,0,1
TASKS = [int(t) for t in os.environ["SHOGYM_TASKS"].split(",")] if os.environ.get("SHOGYM_TASKS") else [0, 1, 2]
# --------------------------------------------------------------------------------------------

RUNS = Path(__file__).resolve().parent / "runs"


def new_run_dir(env: str = ENV, runs: Path = RUNS) -> Path:
    """A fresh provenance directory for one run.

    Fresh per run on purpose. A stream numbers rows from the start of its own queue, so pointing
    two runs at one directory would file both under the same positions. ``TaskStream`` refuses
    that outright (pass ``resume=True`` to continue an interrupted run instead)."""
    return runs / f"{env}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def build_stream(
    env: str = ENV,
    tasks: Sequence[int] = TASKS,
    prov_dir: Optional[Path] = None,
) -> TaskStream:
    """The queue, ready to serve. ``shogym.make`` is passed as a **factory**: the stream builds a
    fresh env per task and closes it, so no two tasks share state."""
    return TaskStream(
        shogym.make,
        [TaskRef(env, i) for i in tasks],
        prov_dir=prov_dir if prov_dir is not None else new_run_dir(env),
        # deadline=600.0,  # optional: seconds per task; an expired task is recorded unscored
        # Feedback on submission: the terminal response carries the env's published
        # verdict (the practice default). For evaluation-grade scores use EvalStream.
        feedback=Immediate(),
    )


async def main() -> None:
    stream = build_stream()
    # stdout is the MCP wire, so everything this process says goes to stderr.
    print(f"[shogym] serving {ENV} tasks {list(TASKS)} -> {stream.prov_dir}", file=sys.stderr)
    async with stream:
        # `async with` is what makes the record complete: on disconnect it forces the terminal on
        # a task still in flight and records it, rather than leaving a dispense with no outcome.
        await build_stream_server(stream, name="shogym").run_async(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
