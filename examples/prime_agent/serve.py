"""Serve one shogym task to Prime Agent over streamable HTTP MCP.

Unlike the other quickstarts, Prime Agent never spawns this process: its host drops every
non-HTTP entry in ``mcpServers`` ("stdio servers self-manage in Python", ``mcp-manager.ts``), so
there is nobody to spawn a stdio server. You start this yourself, in your own shell, and the
agent's kernel connects to the URL. See README.md.

    python serve.py [port]         # streamable HTTP on 127.0.0.1:<port>/mcp (default 8973)

What it publishes is one generation of the protocol v2 stream: ``pull`` hands out the work, every
env tool is wrapped so each call names the attempt it belongs to, and the terminal tool is
intercepted and sealed by the stream instead of reaching the env. The agent is not told which task
it played: a task record carries an attempt id and a body, and has no field an index or a target
could be written into.

Protocol v2 ships one serving entrypoint, ``run_stdio_v2``, and it speaks stdio. This file is that
function with the transport swapped, because a stdio server is unreachable from this harness. It
is the only quickstart carrying that duplication, so a change to the generation's lifecycle has to
be made here too.

One launch is one episode, over one env at one task, so three tasks are three launches. Serving
runs on Temporal, which ``pip install shogym`` installs, so there is no extra to ask for.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from secrets import token_hex
from typing import Optional, Sequence

from shogym.serve.episode import ServedEpisode
from shogym.serve.protocol_v2.gateway import (
    build_gateway_server,
    durable_client,
    environment_terminal,
    open_gateway,
    stream_worker,
)

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
ENV = os.environ.get("SHOGYM_ENV") or "wordle_v1"

# Which task this launch serves. One index, because one generation is one episode over one env.
#     SHOGYM_TASK=7 <your harness command>
TASK = int(os.environ.get("SHOGYM_TASK") or 0)

# Where the skill connects. The port is the only thing the skill and this file have to agree
# on; change it here and in `.prime/agent/settings.json` (or pass a port on the command line).
PORT = int(os.environ.get("SHOGYM_PORT") or 8973)
# --------------------------------------------------------------------------------------------

RUNS = Path(__file__).resolve().parent / "runs"


def new_run_dir(env: str = ENV, runs: Path = RUNS, task: int = TASK) -> Path:
    """A fresh directory for one generation, holding its blobs and its resume manifest.

    Fresh per launch on purpose. The manifest is written once and never rewritten, because a
    resume compares against it, so a directory that already holds one is refused rather than
    added to. The stamp on its own does not make it fresh: it names whole seconds, so two
    launches of one task inside one second would take one directory, and the generation's
    durable history is a file in that directory. The token is what keeps them apart."""
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    return runs / f"{env}-{task}-{stamp}-{token_hex(3)}"


async def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    port = int(args[0]) if args else PORT
    run_dir = new_run_dir()
    # Nothing here reaches the agent: this process is yours, and its stdout is your terminal
    # rather than an MCP wire. Kept on stderr anyway, so redirecting the log is one `2>`.
    print(
        f"[shogym] serving {ENV} task {TASK} on http://127.0.0.1:{port}/mcp -> {run_dir}",
        file=sys.stderr,
    )
    # The episode is built before anything durable starts, and closed however this ends, so a
    # Ctrl-C leaves no env holding a session open behind a stream nobody is serving.
    episode = await ServedEpisode.start(ENV, task=TASK, ends_on_horizon=False)
    try:
        version, activities = environment_terminal(episode)
        # The run directory holds this generation's history as well as its blobs and its
        # manifest, so a second server started beside this one has a database of its own.
        async with durable_client(run_directory=run_dir) as client:
            async with stream_worker(client, activities=activities):
                gateway = await open_gateway(
                    client, episode, run_directory=run_dir, canonicalization_version=version
                )
                # This process is the controller as well as the transport, and its manifest is
                # complete the moment it is built: one episode, one task. So it closes the queue
                # before the model can pull, which is what makes `done` reachable once that task
                # has been sealed and acknowledged.
                await gateway.close_queue()
                server = build_gateway_server(gateway)
                await server.run_async(transport="http", host="127.0.0.1", port=port)
    finally:
        await episode.close()


if __name__ == "__main__":
    asyncio.run(main())
