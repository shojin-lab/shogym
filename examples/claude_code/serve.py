"""Serve a queue of shogym tasks to Claude Code over stdio MCP.

Claude Code spawns this file as an MCP server (see ``.mcp.json``); it never spawns Claude Code.
What it publishes is a :class:`~shogym.serve.stream.TaskStream`: one endpoint that hands out tasks
one at a time (``get_task``), routes the env's own tools to whichever task is live, and, the part
that matters, **seals and scores each task itself**. The agent is never told its score. It is not
even told which task it played: ``get_task`` answers with ``{env, instructions, budget, tools}``
and has no field a task index or a target could be written into.

Every dispensed task lands exactly one durable row under ``runs/<env>-<regime>-<stamp>/``. Read
them back with ``results.py`` once the run is over.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Type

import shogym
from shogym.serve.stream import (
    FeedbackPolicy,
    Immediate,
    Information,
    Never,
    Placebo,
    TaskRef,
    TaskStream,
    build_stream_server,
)

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

# What a terminating call tells the agent. `immediate` is the **practice** default: it hands back
# every episode-level item the env published, its numbers included, which is what you want while
# iterating on an agent and is not an experimental arm of anything.
#
# An env that publishes a matched pair of payloads is run as a pair instead, one arm per launch:
# `information` reveals the item the env filed under `report`, `placebo` reveals the one it filed
# under `notice`, one item each and the same shape on the wire. `never` opens no channel at all.
#     SHOGYM_FEEDBACK=information <your harness command>
FEEDBACK = os.environ.get("SHOGYM_FEEDBACK") or "immediate"

# What this run is called, in the record. The stream folds the deadline, the capacity and what
# each env said about itself into the same identity, so this is the human-readable half; a
# directory that already names one refuses a resume naming another, which is what keeps two arms
# of a pair from being appended into one record.
#     SHOGYM_IDENTITY=appworld-pulse-0-pilot <your harness command>
IDENTITY = os.environ.get("SHOGYM_IDENTITY") or ""

# Seconds per task, and how many tasks may be in flight at once. Both are members of the identity
# above, because a deadline decides whether a slow episode is scored or timed out and a capacity
# decides what an agent may work on next, so two arms compared against each other have to agree
# on them. Unset is no deadline and one task at a time.
DEADLINE = float(os.environ["SHOGYM_DEADLINE"]) if os.environ.get("SHOGYM_DEADLINE") else None
IN_FLIGHT = int(os.environ.get("SHOGYM_IN_FLIGHT") or 1)
# --------------------------------------------------------------------------------------------

# Where the records go. Beside this file by default, which is what a quickstart wants. A run whose
# scores you intend to defend puts them somewhere the agent is not working: the directory holds
# `claim.json`, which names the regime and the identity before the first task is dispensed, and
# `results.jsonl`, which keeps every payload the env published on every row whatever the arm was
# told. An agent that can read either can read its own assignment and the receipt a control arm
# withholds, so a paired run names a directory of its own here and takes the file-reading built-ins
# away (see src/shogym/envs/appworld/README.md).
#     SHOGYM_RUNS=~/appworld-pair/runs <your harness command>
RUNS = (
    Path(os.environ["SHOGYM_RUNS"]).expanduser()
    if os.environ.get("SHOGYM_RUNS")
    else Path(__file__).resolve().parent / "runs"
)

# The policies a launch may name. Closed, and looked up rather than evaluated: the regime is
# stamped onto every row of the run, so what may produce one is a list in the open.
POLICIES: Dict[str, Type[FeedbackPolicy]] = {
    "never": Never,
    "immediate": Immediate,
    "information": Information,
    "placebo": Placebo,
}


def policy(regime: str = FEEDBACK) -> FeedbackPolicy:
    """The feedback policy a run named, or a refusal that names the ones there are.

    Refused rather than quietly defaulted. The regime decides what the agent is told and what
    every row of the run is stamped with, so a typo that fell back to the practice default would
    file an experimental arm's rows under a treatment nobody served."""
    try:
        return POLICIES[regime]()
    except KeyError:
        raise SystemExit(
            f"[shogym] SHOGYM_FEEDBACK={regime!r} is not a feedback policy; use one of "
            + ", ".join(sorted(POLICIES))
        ) from None


def new_run_dir(env: str = ENV, runs: Path = RUNS, regime: str = FEEDBACK) -> Path:
    """A fresh provenance directory for one run.

    Fresh per run on purpose. A stream numbers rows from the start of its own queue, so pointing
    two runs at one directory would file both under the same positions. ``TaskStream`` refuses
    that outright (pass ``resume=True`` to continue an interrupted run instead).

    The regime is in the name because the two arms of a paired run are two records and never one:
    a directory holding rows served under one regime refuses rows served under another, and a
    reader that has to open the directory to find out which arm it holds is a reader that can get
    it wrong."""
    return runs / f"{env}-{regime}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"


def build_stream(
    env: str = ENV,
    tasks: Sequence[int] = TASKS,
    prov_dir: Optional[Path] = None,
    regime: str = FEEDBACK,
    identity: str = IDENTITY,
    deadline: Optional[float] = DEADLINE,
    max_in_flight: int = IN_FLIGHT,
) -> TaskStream:
    """The queue, ready to serve. ``shogym.make`` is passed as a **factory**: the stream builds a
    fresh env per task and closes it, so no two tasks share state.

    Synchronous, and called off the loop by :func:`main` below. Building an env is blocking work
    and for some envs it is real work (AppWorld provisions a corpus and copies two views of it),
    and this constructor builds one per env name for the manifest it publishes. Called from
    inside the serving loop it would hold that loop for the whole of it."""
    return TaskStream(
        shogym.make,
        [TaskRef(env, i) for i in tasks],
        prov_dir=prov_dir if prov_dir is not None else new_run_dir(env, regime=regime),
        # Seconds per task; an expired task is recorded unscored.
        deadline=deadline,
        max_in_flight=max_in_flight,
        # What a terminating call reveals (see `FEEDBACK`). For evaluation-grade scores use
        # EvalStream, which is `Never` made structural and refuses any policy at construction.
        feedback=policy(regime),
        # The caller's half of what this record is filed under. The env's own answer and the two
        # numbers above are folded in by the stream itself.
        identity=identity,
        # `shogym.make` binds no event loop, so each task's env is built in a thread rather than
        # on the loop that is serving the others.
        off_loop_factory=True,
    )


async def main() -> None:
    stream = await asyncio.to_thread(build_stream)
    # stdout is the MCP wire, so everything this process says goes to stderr.
    print(
        f"[shogym] serving {ENV} tasks {list(TASKS)} under {FEEDBACK} -> {stream.prov_dir}",
        file=sys.stderr,
    )
    async with stream:
        # `async with` is what makes the record complete: on disconnect it forces the terminal on
        # a task still in flight and records it, rather than leaving a dispense with no outcome.
        await build_stream_server(stream, name="shogym").run_async(transport="stdio")


if __name__ == "__main__":
    asyncio.run(main())
