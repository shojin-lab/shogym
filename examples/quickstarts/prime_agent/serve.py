"""Serve a queue of hgym tasks to Prime Agent over streamable HTTP MCP.

Unlike the other quickstarts, Prime Agent never spawns this process: its host drops every
non-HTTP entry in ``mcpServers`` ("stdio servers self-manage in Python", ``mcp-manager.ts``),
so there is nobody to spawn a stdio server. You start this yourself, in your own shell, and
the agent's kernel connects to the URL. See README.md.

What it publishes is a :class:`~hgym.serve.stream.TaskStream`: one endpoint that hands out
tasks one at a time (``get_task``), routes the env's own tools to whichever task is live, and,
the part that matters, **seals and scores each task itself**. The agent is not told which task
it played: ``get_task`` answers with ``{env, instructions, budget, tools}`` and has no field a
task index or a target could be written into.

    python serve.py [port]         # streamable HTTP on 127.0.0.1:<port>/mcp (default 8973)

Every dispensed task lands exactly one durable row under ``runs/<env>-<stamp>/``. Read them
back with ``results.py`` once the run is over.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

import hgym
from hgym.serve.stream import Immediate, TaskRef, TaskStream, build_stream_server

# --------------------------------------------------------------------------------------------
# The one variable. Swapping envs is this line and nothing else: the queue, the endpoint, the
# scoring and the readout are all env-agnostic, and the tools the agent is handed come from
# whatever env this names.
#
#     automationbench   wordle_v1   hle   yc_bench   browsecomp_plus   frontier_bench
#     tau2_mock   tau2_airline   tau2_retail   tau2_telecom   tau2_banking_knowledge
#
# (`python -c "import hgym; print(hgym.registered_envs())"` prints the live catalogue. Some envs
# need an extra installed and a key; see their READMEs under src/hgym/envs/.)
# `HGYM_ENV` wins when it is set, so a run can swap envs without editing this file:
#     HGYM_ENV=wordle_v1 <your harness command>
ENV = os.environ.get("HGYM_ENV") or "automationbench"

# Which tasks to serve, in order. A repeat is legal: a task's identity within a run is its
# position in this queue, not its index, so `[0, 0, 1]` plays task 0 twice and records both.
# `HGYM_TASKS` overrides it the same way: HGYM_TASKS=0,0,1
TASKS = [int(t) for t in os.environ["HGYM_TASKS"].split(",")] if os.environ.get("HGYM_TASKS") else [0, 1, 2]

# Where the skill connects. The port is the only thing the skill and this file have to agree
# on; change it here and in `.prime/agent/settings.json` (or pass a port on the command line).
PORT = int(os.environ.get("HGYM_PORT") or 8973)
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
    """The queue, ready to serve. ``hgym.make`` is passed as a **factory**: the stream builds a
    fresh env per task and closes it, so no two tasks share state."""
    return TaskStream(
        hgym.make,
        [TaskRef(env, i) for i in tasks],
        prov_dir=prov_dir if prov_dir is not None else new_run_dir(env),
        # deadline=600.0,  # optional: seconds per task; an expired task is recorded unscored
        # Feedback on submission: the terminal response carries the env's published
        # verdict (the practice default). For evaluation-grade scores use EvalStream.
        feedback=Immediate(),
    )


async def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    port = int(args[0]) if args else PORT

    stream = build_stream()
    # Nothing here reaches the agent: this process is yours, and its stdout is your terminal
    # rather than an MCP wire. Kept on stderr anyway, so redirecting the log is one `2>`.
    print(
        f"[hgym] serving {ENV} tasks {list(TASKS)} on http://127.0.0.1:{port}/mcp"
        f" -> {stream.prov_dir}",
        file=sys.stderr,
    )
    async with stream:
        # `async with` is what makes the record complete: on disconnect it forces the terminal on
        # a task still in flight and records it, rather than leaving a dispense with no outcome.
        server = build_stream_server(stream, name="hgym")
        await server.run_async(transport="http", host="127.0.0.1", port=port)


if __name__ == "__main__":
    asyncio.run(main())
