"""Serve one generation over a roster of AutomationBench tasks to Claude Code.

This is the quickstart's endpoint with one thing changed: the queue is a roster rather than a
single task. ``pull`` hands out one task at a time, every env tool is wrapped so each call names
the attempt it belongs to, and the terminal is intercepted and sealed by the stream. What the
agent sees is what the quickstart's agent sees, and it is not told which task it played or how
many are left.

It speaks either transport. Told a port it publishes over streamable HTTP, which is how the cell
runs it: this process, the benchmark source and every grade the run commits are then in a
container the agent has no mount of, and the endpoint is the only way across. Told none it speaks
stdio as a child of whatever spawned it, which is what a run on one host wants.

One launch is one generation over the whole roster, because that is what a cell is: one agent
session working a list of tasks in order. Each task is worked in a world of its own, opened when
the task is reserved and let go of when it ends, so a task never inherits the workspace its
predecessor filed.

Two knobs decide what the agent is told. ``SHOGYM_CELL_SCHEDULE=immediate`` releases the honest
payload at each seal, and ``never`` creates no payload obligation at all, so the agent works the
roster and is told nothing about any of it. Both are ordinary runs under the platform's own
policy: the difference is the release plan, and the run directory records which one was served.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import re
import secrets
import sys
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import shogym
from shogym.serve.episode import ServedEpisode
from shogym.serve.protocol_v2 import GRADED_HORIZON
from shogym.serve.protocol_v2.gateway import (
    RefusalSink,
    build_gateway_server,
    durable_client,
    environment_terminal,
    open_gateway,
    stream_start,
    stream_worker,
    terminal_manifest,
)
from shogym.serve.protocol_v2.kernel.messages import StreamStart
from shogym.serve.protocol_v2.policy import GradeIdentity, ORDINARY
from shogym.serve.protocol_v2.schedule import IMMEDIATE, NEVER, ReleasePlan
from shogym.task import TaskSpec, ToolManifest

#: The environment this cell is over. The cell is a rerun of one recorded on AutomationBench, so
#: unlike the quickstart's this is not a variable.
ENV = "automationbench"

#: The key a harness namespaces this server's tools under, so they reach the model as
#: ``mcp__curriculum__*``. It is the name the earlier cell's agent saw. A tool name is part of the
#: prompt prefix, so it is pinned to that rather than renamed to this repo's own default.
SERVER = "curriculum"

#: The file the run directory holds saying which task each position was. Nothing on the wire
#: names a task, and the records the history answers with are keyed by attempt, so this is what
#: a read joins a score back to a benchmark task through.
ROSTER_FILE = "cell.json"

#: The file the run directory holds saying how many calls this transport refused. A refusal
#: advances no protocol state, so the generation counts none of them and the agent's transcript
#: is their whole record; this is the count the party that issued them kept, which is what a read
#: cross-checks that record against. It is written inside the call that issues each refusal,
#: because a server is taken away rather than asked to stop: a run that is finished with its
#: server removes the container by force, and a process that is killed runs nothing on its way
#: out. Anything that samples the count instead has an interval to be killed inside, and what it
#: leaves behind is a stale number that reads as a good one.
REFUSAL_FILE = "refusals.json"

#: What a schedule may be named on the command line, and the plan each name means. ``immediate``
#: is the honest receipt at every seal; ``never`` is a roster the agent is told nothing about.
SCHEDULES: Dict[str, ReleasePlan] = {"immediate": IMMEDIATE, "never": NEVER}

#: How the cell this one reruns drew its roster out of AutomationBench's 600 public tasks. The
#: first seed splits the benchmark into a held-out fifth and a training pool, both re-sorted; the
#: second shuffles the training pool into the order tasks were dispensed in. Both are written
#: here because the order is part of the measurement: a rerun that drew its own would differ from
#: the cell it is being compared with in the tasks as well as in the serving.
PUBLIC_TASKS = 600
HELDOUT_SHARE = 0.2
SPLIT_SEED = 20260726
STREAM_SEED = SPLIT_SEED + 1

#: What names that roster on the command line. ``cell-one`` is all of it and ``cell-one:20`` is
#: the first twenty, which is the prefix a smaller rerun works through in the same order. The
#: spelling is exact: a roster is the measurement, so a name that is nearly this one names no
#: roster rather than the whole of it.
CELL_ONE = "cell-one"
_CELL_ONE = re.compile(rf"{re.escape(CELL_ONE)}(?::(\d+))?$")


def cell_one_stream(size: Optional[int] = None) -> List[int]:
    """Return the task positions the cell this one reruns was served, in the order it saw them.

    Recomputed from the seeds rather than read from that run's records, so the roster is a fact
    about the split and not about one directory somebody still has. ``size`` takes a prefix,
    which is what a shorter rerun works: the first twenty of the stream are the same twenty
    tasks in the same order whether or not the rest follows.
    """
    order = list(range(PUBLIC_TASKS))
    random.Random(SPLIT_SEED).shuffle(order)
    train = sorted(order[round(PUBLIC_TASKS * HELDOUT_SHARE) :])
    random.Random(STREAM_SEED).shuffle(train)
    return train if size is None else train[:size]


def roster(text: str) -> List[int]:
    """Return the task positions ``text`` names, in the order it names them.

    ``0,1,2`` is three tasks and ``0-19`` is twenty, because a cell's roster is usually a range
    and writing one out is how a typo becomes a different measurement. Order is the roster's
    own: this does not sort, because the order tasks arrive in is part of what a rerun matches.
    A repeat is refused rather than served, since two positions over one task would be scored
    twice against one world's worth of work and read back as one row per position.

    ``cell-one`` is the whole of the roster this cell reruns and ``cell-one:20`` is its first
    twenty, which is the spelling a rerun uses: naming the stream is what keeps the tasks and
    their order out of the list of things that could differ between the two runs. That name is
    matched whole and its prefix has to be a length the stream has, because every near miss is a
    different measurement rather than a smaller one: a typo would be the entire cell, a zero or a
    negative prefix would be no roster or all but one of it, and a prefix longer than the stream
    would be the whole of it under a name that says otherwise.
    """
    named = text.strip()
    if named.startswith(CELL_ONE):
        matched = _CELL_ONE.fullmatch(named)
        if matched is None:
            raise ValueError(
                f"{text!r} is not this cell's roster, which is named {CELL_ONE!r} for all of it "
                f"or {CELL_ONE + ':<n>'!r} for its first n"
            )
        whole = len(cell_one_stream())
        if matched.group(1) is None:
            return cell_one_stream()
        size = int(matched.group(1))
        if not 1 <= size <= whole:
            raise ValueError(
                f"{text!r} asks for a prefix of {size} of a stream {whole} tasks long, and a "
                f"shorter rerun works a prefix of it: 1 to {whole}"
            )
        return cell_one_stream(size)
    positions: List[int] = []
    for piece in (part.strip() for part in text.split(",")):
        if not piece:
            continue
        if "-" in piece.lstrip("-"):
            low, _, high = piece.partition("-")
            span = range(int(low), int(high) + 1)
            if not span:
                raise ValueError(f"{piece!r} names no task, so this roster would be shorter")
            positions.extend(span)
        else:
            positions.append(int(piece))
    if not positions:
        raise ValueError("a cell serves a roster, and this one names no task")
    if len(set(positions)) != len(positions):
        raise ValueError(f"this roster names a task twice: {text!r}")
    return positions


def release_for(name: str) -> ReleasePlan:
    """Return the release plan ``name`` means, or refuse a name that means nothing here.

    The regime a cell runs under is the thing a rerun has to match, so an unknown name is a
    refusal rather than a fall back to whichever plan happens to be the default.
    """
    try:
        return SCHEDULES[name]
    except KeyError:
        raise ValueError(
            f"{name!r} is not a schedule this cell serves, and what the agent is told is what a "
            f"rerun matches, so it is named rather than defaulted: {', '.join(sorted(SCHEDULES))}"
        ) from None


def bodies_for(positions: List[int], *, domain: str) -> List[str]:
    """Return the instructions each roster position carries, read off one env.

    The queue is committed before anything is served, so every task's body has to exist before
    the first world does. One env answers for all of them: ``describe`` is a pure read of a task
    row and opens no session, and the worlds are opened later, one per attempt.
    """
    env = shogym.make(ENV, config={"domain": domain})
    return [env.describe(str(position)).instructions for position in positions]


def compose(
    spec: TaskSpec,
    terminal: ToolManifest,
    *,
    bodies: List[str],
    release: ReleasePlan,
    grade: GradeIdentity,
    claim_hash: str,
) -> StreamStart:
    """Return the generation this cell serves: one ordinary run over the whole roster.

    Ordinary is the profile on purpose. What the agent is told is decided by the release plan
    and by nothing else here, so the run cannot be blinded by an omission: under ``immediate``
    every position is stamped with the honest policy, and under ``never`` every position is
    stamped with the reason it delivers nothing. Concealing a score that was released is an
    experiment arm, and this cell registers none.
    """
    return stream_start(
        spec,
        terminal,
        claim_hash=claim_hash,
        bodies=bodies,
        release=release,
        profile=ORDINARY,
        grade=grade,
    )


def record_roster(
    run_dir: Path,
    *,
    domain: str,
    schedule: str,
    positions: List[int],
    attempts: List[str],
) -> Path:
    """Write down which benchmark task each attempt was, and return the path.

    The history is the record of what an attempt scored, and it is keyed by attempt because
    nothing on the wire names a task. So the join is written here, once, by the only party that
    knows both halves. It is a harness-side note and never an authority: no score is in it, and
    a read that could not find it would still read every score the run kept.
    """
    path = run_dir / ROSTER_FILE
    path.write_text(
        json.dumps(
            {
                "env": ENV,
                "domain": domain,
                "schedule": schedule,
                "release_plan_id": release_for(schedule).release_plan_id,
                "profile": ORDINARY,
                "tasks": [
                    {"task_position": position, "task": str(task), "attempt_id": attempt}
                    for position, (task, attempt) in enumerate(zip(positions, attempts))
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def record_refusals(run_dir: Path, refusals: int) -> Path:
    """Write down how many calls this transport refused, and return the path.

    A refusal is model-visible text and nothing else: it advances no protocol state, so the
    generation has nothing to count and the agent's transcript is where the refusal exists. This
    is the count kept by the party that issued them, and a read compares the two. A refusal sent
    and never delivered is then a difference somebody can find rather than a turn nobody holds.

    It is written whole or not at all. A read that found half a number would call it no count and
    a run finished with one would fail its own cross-check, so the bytes go to a file beside it
    and are moved onto the name in one step.
    """
    path = run_dir / REFUSAL_FILE
    written = path.with_suffix(".writing")
    written.write_text(json.dumps({"refusals": refusals}) + "\n", encoding="utf-8")
    os.replace(written, path)
    return path


def refusal_sink(run_dir: Path) -> RefusalSink:
    """Return what the transport tells its refusal count to, which puts it straight on disk.

    Writing the count when the server stops writes it at the one moment the server may not
    reach. A run that has finished with its server takes the container away rather than asking it
    to stop, and nothing runs on the way out of that. Sampling the count on a timer is no better:
    it only shortens the window, and a refusal issued inside the last interval leaves a stale
    number that a read cannot tell from a good one.

    So the transport hands the count over as it makes it, in the call that makes it and before
    the model sees the refusal. There is no interval left to be killed inside.
    """

    def write(refusals: int) -> None:
        record_refusals(run_dir, refusals)

    return write


async def serve(
    positions: List[int],
    *,
    domain: str,
    schedule: str,
    run_dir: Path,
    claim_hash: str,
    bind: Optional[Tuple[str, int]] = None,
) -> None:
    """Serve the roster until the stream says it is done.

    The service, the Worker and the stream all belong to this process, exactly as they do for
    the one-task quickstart. What is added is the opener: every attempt after the first gets a
    world of its own, and the position it belongs to is the one the roster assigned that attempt
    before anything was served.

    ``bind`` is what decides which side of the boundary this process is on. Without it the server
    speaks stdio and is a child of whatever spawned it, which is the quickstart's shape. With it
    the same generation and the same tools are published over streamable HTTP, which is what lets
    the process, the benchmark source and every grade this run commits sit in a container the
    agent has no mount of and reach it only through the endpoint.

    Each of those worlds is opened without an ending of its own, because the stream is what ends
    an attempt here. The step budget is still the task's, and what running it out comes to is
    checked before anything is served: this cell is a rerun of one whose harness graded a spent
    budget on the partial state, so a generation served under the floor instead would score every
    unfinished task at nothing and be compared against a cell that did not.
    """
    plan = release_for(schedule)
    config = {"domain": domain}
    bodies = bodies_for(positions, domain=domain)
    first = await ServedEpisode.start(
        ENV, task=positions[0], env_config=config, ends_on_horizon=False
    )
    stopped = False
    try:
        spec = first.describe()
        environment = environment_terminal(first)
        if environment.horizon_ending != GRADED_HORIZON:
            raise ValueError(
                f"this cell reruns a harness that graded a spent step budget, and {ENV} is "
                f"serving its horizon as {environment.horizon_ending!r}"
            )
        composed = compose(
            spec,
            terminal_manifest(spec),
            bodies=bodies,
            release=plan,
            grade=environment.grade,
            claim_hash=claim_hash,
        )
        at = {item.attempt_id: item.task_position for item in composed.tasks}

        async def open_world(attempt_id: str) -> ServedEpisode:
            return await ServedEpisode.start(
                ENV, task=positions[at[attempt_id]], env_config=config, ends_on_horizon=False
            )

        async with durable_client(run_directory=run_dir) as client:
            async with stream_worker(client, activities=environment.activities):
                counted = refusal_sink(run_dir)
                gateway = await open_gateway(
                    client,
                    first,
                    start=composed,
                    open_episode=open_world,
                    run_directory=run_dir,
                    environment=environment,
                    on_refusal=counted,
                )
                record_roster(
                    run_dir,
                    domain=domain,
                    schedule=schedule,
                    positions=positions,
                    attempts=[item.attempt_id for item in composed.tasks],
                )
                # The manifest is complete the moment it is built: this roster and no more. So
                # the queue is closed before the agent can pull, which is what makes Done
                # reachable once the last task has been sealed and paid out.
                await gateway.close_queue()
                # Nought, before a call can be made. A server that started and refused nothing is
                # a different fact from a server that never started, and only the first of those
                # has a count to show for itself.
                counted(gateway.refusals)
                server = build_gateway_server(gateway)
                try:
                    if bind is None:
                        await server.run_async(transport="stdio")
                    else:
                        host, port = bind
                        await server.run_async(transport="http", host=host, port=port)
                finally:
                    # A server that reaches here says so once more. Everything the count needed
                    # to survive was already written by the calls that made it.
                    counted(gateway.refusals)
                    await gateway.aclose()
                    stopped = True
    finally:
        if not stopped:
            await first.close()


def binding(environment: Mapping[str, str]) -> Optional[Tuple[str, int]]:
    """Where this server publishes, or nothing when it is a stdio child of whoever spawned it."""
    port = environment.get("SHOGYM_CELL_PORT")
    return (environment.get("SHOGYM_CELL_HOST") or "127.0.0.1", int(port)) if port else None


async def main() -> None:
    positions = roster(os.environ.get("SHOGYM_CELL_TASKS") or "0")
    domain = os.environ.get("SHOGYM_CELL_DOMAIN") or "public"
    schedule = os.environ.get("SHOGYM_CELL_SCHEDULE") or "immediate"
    run_dir = Path(os.environ["SHOGYM_CELL_RUN_DIR"]).expanduser()
    bind = binding(os.environ)
    # Under stdio this process's stdout is the MCP wire, so everything it says goes to stderr.
    where = "stdio" if bind is None else f"http://{bind[0]}:{bind[1]}/mcp/"
    print(
        f"[shogym] serving {ENV}/{domain} tasks {positions} under {schedule} on {where} "
        f"-> {run_dir}",
        file=sys.stderr,
    )
    await serve(
        positions,
        domain=domain,
        schedule=schedule,
        run_dir=run_dir,
        bind=bind,
        # The claim secret is minted here and never leaves this process, exactly as the one the
        # quickstart's gateway mints for itself: it is what binds this transport as the
        # generation's one consumer, and a composition made here has to carry its own.
        claim_hash=sha256(secrets.token_bytes(32)).hexdigest(),
    )


if __name__ == "__main__":
    asyncio.run(main())
