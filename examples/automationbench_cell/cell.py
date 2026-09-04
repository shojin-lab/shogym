"""Run one Claude Code cell over AutomationBench under protocol v2, and read what it scored.

The run this reruns queued the same task ids in the same order, to the same model at the same
effort, and told the agent its score the moment each task ended. It queued them at an earlier
revision of the benchmark, and a task id is an index into a revision's own list, so the same id
can carry other instructions and other scoring assertions here; the launch record names both
revisions. What differs on purpose is the serving contract: the agent asks for work with ``pull``
rather than being handed it, every call names the attempt it belongs to, the terminal is
intercepted and sealed by the stream rather than by the harness, and the score reaches the agent
as a payload the stream released against the attempt instead of as a field on the tool result that
ended it. That difference is the measurement, so everything else here is pinned to what that run
ran. The launch record names what it is pinned to; the differences that could not be pinned, the
timers, the continuation, the topology, the evaluation bookend and the rest of that image, are
set out in the pull request that brought this cell to that run rather than in this directory.

The run is two containers on a private network. The generation, the benchmark source and every
grade the run commits are in one of them, and the agent is in the other with the directory it
works in and the Claude Code home it writes to and nothing else. The recorded run kept the same
property by another arrangement, and the property is what makes this a rerun rather than a rerun
on the honour system: an agent under ``bypassPermissions`` reads whatever filesystem it is on, so
the only useful question is whether the answers were ever on that filesystem.

Three commands. ``run`` composes the generation, serves it to the agent's container over the
network, and keeps the transcript and the durable history side by side in one directory. ``probe``
stands the same two domains up and asks, from inside a container started as the agent's is, save
that it carries a credential only where the environment has one, whether any of it can be reached.
``table`` reads a run directory back: the rows ``shogym
results`` prints, and beside them the join that says which benchmark task each attempt was and how
much work went into it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from secrets import token_hex
from types import FrameType
from typing import Any, Dict, Iterator, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from examples.automationbench_cell import pinned
from examples.automationbench_cell import sandbox
from examples.automationbench_cell import table as read_back
from examples.automationbench_cell.serve import (
    CAPACITY,
    CELL_ONE,
    POOL_CEILING,
    ROSTER_FILE,
    SERVED_TOOLS,
    SERVER,
    STREAM_SEED,
    release_for,
    roster,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

#: The model and effort the recorded run used. Both are overridable, because a smoke run proving
#: the pipe should not cost what a cell costs, and the recorded harness took the same two as
#: overrides for the same reason.
MODEL = "claude-opus-5"
EFFORT = "xhigh"

#: The user turn, byte for byte as the recorded run sent it, trailing newline included. Everything
#: else the agent is told is the system prompt, which is where that run put it so that it survives
#: compaction. The bytes matter beyond tidiness: the turn is the end of the cached prompt prefix,
#: so a run that trimmed it is a run the model met under a different prefix.
KICKOFF = "Begin.\n"

#: The regime the recorded run served, read off its broker rather than its documentation: the score
#: and the environment's numbers reached the agent inside the result of the call that ended each
#: task, on every task, with no path that could withhold one. Under this protocol the equivalent
#: is the honest payload released at each seal, which is what ``immediate`` is.
CELL_ONE_SCHEDULE = "immediate"

#: What the run directory holds. The durable history and the sealed grades live under ``grades``,
#: which the agent's container has no mount of: it is bound into the server's container and into
#: nothing else, so the two directories the agent keeps are siblings of a directory it cannot see.
GRADES = "grades"
SELF = "self"
HOME = "home"
CONFIG = "cfg"
TRANSCRIPT = "stream.jsonl"
STDERR = "stream.err.txt"
SERVER_LOG = "server.log"
RUN_FILE = "run.json"

#: What the run's record says of itself when it is read back. A launch is complete when work
#: crossed the boundary in both directions and the two containers were taken down afterwards, and
#: it says why it is not when it is not: an exit code alone cannot tell a cell that served two
#: hundred tasks from one whose agent never found the endpoint. A read holds a run this calls
#: complete to checks it lets an incomplete one report as unavailable.
COMPLETE = "complete"
INCOMPLETE = "incomplete"

#: The signals a launcher owning two containers has to survive long enough to take them down.
STOPPING = (signal.SIGINT, signal.SIGTERM)


class Stopped(Exception):
    """The launcher was signalled, so what it started is taken down before it goes."""

    def __init__(self, number: int) -> None:
        self.number = number
        super().__init__(f"the launcher was stopped by {signal.Signals(number).name}")


@contextmanager
def _handling(handler: Any) -> Iterator[None]:
    """Install one handler for the stopping signals, and put back whatever was there before."""
    previous = [(number, signal.signal(number, handler)) for number in STOPPING]
    try:
        yield
    finally:
        for number, restore in previous:
            signal.signal(number, restore)


@contextmanager
def stopping() -> Iterator[None]:
    """Turn termination into an exception for as long as this cell owns containers.

    A container outlives the process that started it, and a signal ends a Python process without
    unwinding it. So an ordinary ``kill`` or a scheduler timeout used to leave the agent calling
    the server and writing into the two directories this run is measuring, with the launcher that
    was watching them gone and the run's own record never saying how it ended. Raising instead
    means the teardown runs and the record is finished.
    """

    def stop(number: int, frame: Optional[FrameType]) -> None:
        raise Stopped(number)

    with _handling(stop):
        yield


@contextmanager
def undisturbed() -> Iterator[None]:
    """Ignore termination while a teardown runs, so a second signal cannot leave half of one."""
    with _handling(signal.SIG_IGN):
        yield


def new_run_dir(runs: Path, *, schedule: str, prefix: str = "cell") -> Path:
    """A fresh directory for one cell.

    Fresh per launch for the same reason the quickstart's is: the generation writes its manifest
    once and refuses a directory that already holds one. The stamp names whole seconds, so the
    token is what keeps two launches inside one second apart. The token is also what names the
    run's own network and containers, so two cells on one host reach each other's endpoint by no
    name they could resolve.
    """
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    return runs / f"{prefix}-{schedule}-{stamp}-{token_hex(3)}"


def mcp_config(run_dir: Path, *, url: str) -> Path:
    """Write the config naming the endpoint the agent connects to, and return its path.

    The agent spawns nothing. What it is given is a URL on the private network, which is what
    keeps the server a process on the other side of a container rather than a child of the agent's
    own. The file goes in a directory of its own, mounted read only and outside the directory the
    agent works in, so it never becomes part of a self a later run would start from. That is where
    the recorded run kept its own, under the name that run gave it.
    """
    directory = run_dir / CONFIG
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / sandbox.MCP_CONFIG
    path.write_text(
        json.dumps({"mcpServers": {SERVER: {"type": "http", "url": url}}}, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def claude_argv(
    config: Path, *, model: str, effort: str, system_prompt: str, session_id: str
) -> List[str]:
    """The command the recorded run launched, with this protocol's prompt in it.

    Every flag is the one that run passed, in the order that run's own launcher wrote them. The
    order is unlikely to change what the CLI does, and it is what a command line is compared by:
    an argument list in another order is not the recorded command, and a rerun claiming to be that
    command should be readable beside it word for word.

    ``--strict-mcp-config`` keeps the operator's own MCP servers out of the run, and nothing is
    denied: that run's rollout arm ran with the agent's own tools left in place, web included, so
    a rerun that took them away would be comparing two things at once. ``--setting-sources`` is
    passed with nothing after it for the same reason the home is fresh: a settings file the image
    or the home happened to carry would configure the CLI from outside the launch, and the empty
    value is what says no source at all rather than the default set.
    """
    return [
        "claude",
        "-p",
        KICKOFF,
        "--model",
        model,
        "--mcp-config",
        str(config),
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--permission-mode",
        "bypassPermissions",
        "--append-system-prompt",
        system_prompt,
        "--forward-subagent-text",
        "--output-format",
        "stream-json",
        "--verbose",
        "--include-partial-messages",
        "--effort",
        effort,
        "--session-id",
        session_id,
    ]


def write_run_file(run_dir: Path, record: Dict[str, object]) -> Path:
    """Write the run's own record of itself, and return the path."""
    path = run_dir / RUN_FILE
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def pinned_to(system_prompt: str) -> Dict[str, object]:
    """What this cell is a rerun of, named strongly enough to tell it from another cell's.

    A run directory outlives the checkout that made it, and the names in it move: a roster called
    by the same name can mean other tasks a year later, and a benchmark revision can change the
    text of every task under ids that did not move. So the launch record carries the identifiers
    rather than the words: the run this cell is pinned to, the digest of the standing instruction
    that run served, the digest of the split its roster came from, the seed and the cap that turn
    that split into this roster, the capacity the generation serves at, and both benchmark
    revisions, the one this checkout pins and the one that run was over.

    The prompt is named twice on purpose. The recorded digest is the run's, and the second is of
    the bytes this launch actually passed, so a prompt edited between them is a difference a
    reader can see rather than one they have to reconstruct.
    """
    from shogym.envs.automationbench.adapter import UPSTREAM_SHA

    return {
        "recorded_run": pinned.RECORDED_RUN,
        "recorded_prompt_sha256": pinned.RECORDED_PROMPT_SHA256,
        "prompt_sha256": sha256(system_prompt.encode("utf-8")).hexdigest(),
        "split_id_digest": pinned.RECORDED_SPLIT_DIGEST,
        "roster_seed": STREAM_SEED,
        "roster_ceiling": POOL_CEILING,
        "capacity": CAPACITY,
        "benchmark_revision": UPSTREAM_SHA,
        "recorded_benchmark_revision": pinned.RECORDED_BENCHMARK_REVISION,
        "recorded_shogym_revision": pinned.RECORDED_SHOGYM_REVISION,
    }


class Domains(NamedTuple):
    """The two sides of one run, and what each was given."""

    network: str
    server: str
    agent: str
    agent_image: str
    server_image: str
    mounts: List[Tuple[Path, str, str]]
    environment: Dict[str, str]
    url: str


class AgentLaunch(NamedTuple):
    """The agent's container as one command, which the probe is started from too."""

    argv: List[str]
    mounts: List[Tuple[Path, str, str]]


def open_domains(
    run_dir: Path, *, tasks: str, domain: str, schedule: str, cache: Path
) -> Domains:
    """Make the private network and start the measurement's domain on it.

    Both images are the caller's to have built, because a launch checks what they were built from
    before it starts anything. Everything the run has to be told is decided here and handed to the
    server as arguments, so the process that loads the benchmark and commits the scores inherits
    nothing from the shell the launch was typed into. It returns once the endpoint is answering,
    which is later than the port opening: a first MCP call that races an unready server ends the
    agent with nothing.
    """
    if not sandbox.docker_available():
        raise ValueError(
            "this cell runs the agent in a container and the measurement in another, and there is "
            "no docker daemon here to run either"
        )
    agent_image = f"{sandbox.AGENT_IMAGE}:{pinned.CLI_VERSION}"
    server_image = f"{sandbox.SERVER_IMAGE}:latest"
    network, server, agent = sandbox.names(run_dir.name.rsplit("-", 1)[-1])
    (run_dir / GRADES).mkdir(parents=True, exist_ok=True)
    for source, _, _ in sandbox.cache_mounts(cache):
        source.mkdir(parents=True, exist_ok=True)
    # The benchmark source is fetched before the server starts, because the server has it mounted
    # read only and a cache nobody has filled cannot be filled from behind that mount. A clean
    # host used to fail here, with the loader raising on a read-only filesystem and the endpoint
    # never opening at all.
    print("[cell] provisioning the benchmark source", flush=True)
    fetch = sandbox.provisioner_name(server)
    try:
        sandbox.provision_source(server_image, cache=cache, name=fetch)
    finally:
        # The fetch is the one container a launch owns before it owns anything else, and a signal
        # ends the client rather than the container: an interrupted launch used to leave it filling
        # the cache with nothing left holding it. A fetch that finished is already gone, which is
        # what removing a name nothing answers to comes to.
        with undisturbed():
            unfinished = sandbox.remove_container(fetch)
        if unfinished is not None:
            print(f"[cell] {unfinished}", flush=True)
    mounts = sandbox.server_mounts(run_dir, grades_dir=GRADES, cache=cache)
    environment = sandbox.server_environment(tasks=tasks, domain=domain, schedule=schedule)
    print(f"[cell] starting the server on {network}", flush=True)
    try:
        # The network is made inside the guarded region because it is one of the things that has
        # to come down again: a run interrupted between making it and starting the server used to
        # leave it behind.
        sandbox.create_network(network)
        subprocess.run(
            sandbox.server_argv(
                image=server_image,
                name=server,
                network=network,
                mounts=mounts,
                environment=environment,
            ),
            check=True,
            capture_output=True,
        )
        sandbox.wait_for_gateway(server)
    except BaseException:
        # A domain that never came up is torn down here, because the caller has nothing to hold
        # it by and a container left running is a container serving this run's grades to whoever
        # starts next on a network of that name.
        with undisturbed():
            sandbox.remove_container(server)
            sandbox.remove_network(network)
        raise
    return Domains(
        network=network,
        server=server,
        agent=agent,
        agent_image=agent_image,
        server_image=server_image,
        mounts=mounts,
        environment=environment,
        url=sandbox.gateway_url(server),
    )


def close_domains(domains: Domains, *, log: Optional[Path] = None) -> List[str]:
    """Stop both containers, take the network down, and return what would not go.

    Every target is attempted before anything is reported, and termination is ignored while this
    runs. A teardown that stopped at its first failure, or at a second signal, would leave
    whatever it had not reached yet running against the directories this run is measuring, and
    the caller relying on it would never hear about it.
    """
    failures: List[str] = []
    with undisturbed():
        if log is not None:
            try:
                sandbox.save_logs(domains.server, log)
            except OSError as failure:
                failures.append(f"the server's log could not be saved: {failure}")
        for reason in (
            sandbox.remove_container(domains.agent),
            sandbox.remove_container(domains.server),
            sandbox.remove_network(domains.network),
        ):
            if reason is not None:
                failures.append(reason)
    return failures


def agent_command(
    domains: Domains,
    run_dir: Path,
    *,
    command: Sequence[str],
    environment: Mapping[str, str],
    credential: Optional[str],
) -> AgentLaunch:
    """The command that starts the agent's container, and the three directories it is given.

    The probe is started through this too. What the probe measures is only what the agent would
    have met if the container it asks from is the container a launch builds, so the image, the
    mounts, the network, the working directory and the environment are decided here once rather
    than described twice.
    """
    mounts = sandbox.agent_mounts(run_dir, self_dir=SELF, home_dir=HOME, config_dir=CONFIG)
    return AgentLaunch(
        argv=sandbox.agent_argv(
            image=domains.agent_image,
            name=domains.agent,
            network=domains.network,
            mounts=mounts,
            environment={
                name: value for name, value in environment.items() if name != credential
            },
            credential=credential,
            command=command,
        ),
        mounts=mounts,
    )


def unserved(run_dir: Path) -> List[str]:
    """Why this launch is not a run anybody can compare, or nothing when it is one.

    A launch that reached no server still exits nought. The CLI prints its opening line, fails to
    negotiate with an endpoint that is not there, and stops with the code a finished run stops
    with, which is how a slash in a URL once turned into a cell of no tasks reported as a success.
    So whether the run happened is not what the process returned but whether work crossed the
    boundary, and both sides are asked. Either one empty is a cell with nothing in it.

    What the agent's side has to show is a task. A ``pull`` in the transcript is a call the model
    wrote, and a call is not a delivery: the request can be refused, redirected or answered with a
    protocol error, and the model can go on to exit nought having received no work at all. So what
    is counted is the result the call came back with, and it counts only if it decodes as a Task.

    The gateway's log is the other side and never stands alone. It says that something reached the
    endpoint, which the handshake and the tool listing do as well, so an initialization that
    arrived before the first pull failed would answer this half of a run that served nothing.

    The init line is not enough on its own to answer either. It reports the server it was
    configured with, which the recorded run's own first line reports as connected, and a
    connection is not a delivery: it is written before any pull has been made.
    """
    reasons: List[str] = []
    transcript = run_dir / TRANSCRIPT
    tasks = read_back.read_transcript(transcript).tasks if transcript.is_file() else 0
    if tasks == 0:
        reasons.append("no pull came back with a task, so this run served nothing")
    log = run_dir / SERVER_LOG
    answered = (
        sandbox.served_requests(log.read_text(encoding="utf-8", errors="replace"))
        if log.is_file()
        else 0
    )
    if answered == 0:
        reasons.append("the gateway answered no request, so nothing reached the measurement")
    return reasons


def certifies(run_dir: Path) -> bool:
    """Whether the launch presents this run as one it finished.

    It decides what a check the read could not make comes to. A launch that recorded an
    incomplete run is saying so already, and a missing cross-check there is reported as missing.
    A launch that says nothing to the contrary is offering the run as a measurement, and a
    measurement whose checks could not be made is not one a read may pass.

    A directory with no launch record at all claims nothing, so nothing is certified out of it.
    """
    try:
        written = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(written, dict) and written.get("status") != INCOMPLETE


def launch(
    run_dir: Path,
    *,
    tasks: str,
    domain: str,
    schedule: str,
    model: str,
    effort: str,
    cache: Optional[Path] = None,
    allow_cli_drift: bool = False,
    allow_image_drift: bool = False,
    build: bool = False,
) -> int:
    """Serve the roster and run the agent against it, and return nought only if it ran.

    The agent's container holds the directory it works in, the Claude Code home it writes to, and
    the one file naming the endpoint. It holds no run directory, no repository and no benchmark
    cache, so the roster, the history, the answers and the sealed grades are not files it could
    read whatever it was permitted to do. That is the boundary, and the run's own record names
    every mount it is made of.

    What the two containers were handed is built rather than inherited, and what it resolved to is
    written down: the argv, the environment, the topology, a digest of each directory the agent
    started from, what each image was built from, and the CLI build, so that a launch nobody could
    pin is still a launch somebody can read back.

    The record ends by saying whether this is a run at all. A cell whose agent never reached the
    endpoint exits nought with an empty transcript, and a pilot reading exit codes would file it
    beside a cell that served two hundred.
    """
    positions = roster(tasks)
    # These are read here for their refusals. A misspelled roster, an unknown schedule, a CLI that
    # is not the recorded build and an image built from inputs nobody recorded are all mistakes to
    # hear about now rather than from a server the agent has already been launched at, or from a
    # comparison months later.
    release_for(schedule)
    # The credential is checked here too. The agent is given a fresh Claude Code home, so nothing
    # the operator's own home holds reaches it, and a launch the environment does not authenticate
    # would build two images and serve a roster to an agent that exits at once.
    carried = pinned.check_credential(os.environ)
    cache = sandbox.default_cache() if cache is None else cache
    agent_image = f"{sandbox.AGENT_IMAGE}:{pinned.CLI_VERSION}"
    server_image = f"{sandbox.SERVER_IMAGE}:latest"
    images = sandbox.build_images(agent=agent_image, server=server_image, rebuild=build)
    pinned.check_image_build(images[agent_image], allow_drift=allow_image_drift)
    # Taken whether or not it is empty, and kept in the record either way. An operator who allowed
    # the drift was told it would be recorded, and the resolved inputs alone cannot say which of
    # them this cell did not expect.
    image_drift = pinned.image_drift(images[agent_image])
    cli_version = pinned.resolve_cli_version(sandbox.cli_version_command(agent_image))
    pinned.check_cli_version(cli_version, allow_drift=allow_cli_drift)

    work = run_dir / SELF
    pinned.empty_workdir(work)
    home = run_dir / HOME
    home.mkdir(parents=True, exist_ok=True)
    record: Dict[str, object] = {}
    domains: Optional[Domains] = None
    stopped: Optional[Stopped] = None
    returncode: Optional[int] = None
    cleanup: List[str] = []
    with stopping():
        try:
            domains = open_domains(
                run_dir, tasks=tasks, domain=domain, schedule=schedule, cache=cache
            )
            config = mcp_config(run_dir, url=domains.url)
            session_id = str(uuid.uuid4())
            # Read whole and passed whole. The file is the recorded standing instruction with the
            # two substitutions this protocol forces, and its trailing newline is part of it: the
            # prompt is the front of the cached prefix, so trimming it served a different one.
            system_prompt = (HERE / "PROMPT.txt").read_text(encoding="utf-8")
            environment = pinned.agent_environment(os.environ)
            started = agent_command(
                domains,
                run_dir,
                command=claude_argv(
                    Path(sandbox.CONFIG_MOUNT) / config.name,
                    model=model,
                    effort=effort,
                    system_prompt=system_prompt,
                    session_id=session_id,
                ),
                environment=environment,
                credential=carried,
            )
            record = {
                "model": model,
                "effort": effort,
                "schedule": schedule,
                "domain": domain,
                "tasks": tasks,
                "task_count": len(positions),
                "session_id": session_id,
                "server": SERVER,
                "started": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}",
                "argv": started.argv,
                "cwd": sandbox.WORK,
                # The environment as the container was handed it, never as the operator's shell
                # had it, and with the credential named rather than copied: which name
                # authenticated the run is part of the launch, and what it was worth is not. It is
                # passed to docker by name for the same reason, so no argument list this record
                # keeps has ever held it.
                "environment": pinned.redacted(environment),
                "credential": carried,
                "cache": str(cache),
                "topology": sandbox.topology(
                    network=domains.network,
                    server=domains.server,
                    agent=domains.agent,
                    agent_image=domains.agent_image,
                    server_image=domains.server_image,
                    agent_mount_list=started.mounts,
                    server_mount_list=domains.mounts,
                    agent_environment=environment,
                    server_env=domains.environment,
                    credential=carried,
                    images=images,
                ),
                "server_listening": sandbox.listening_sockets(domains.server),
                "digests": {
                    "config": pinned.digest_tree(run_dir / CONFIG),
                    "work": pinned.digest_tree(work),
                    "home": pinned.digest_tree(home),
                },
                "cli_version": cli_version,
                "cli_version_recorded": pinned.CLI_VERSION,
                "image_drift": image_drift,
                "pinned": pinned_to(system_prompt),
                # Written before the launch and rewritten after it, so a run killed hard enough to
                # skip the rest of this reads back as the unfinished thing it is.
                "status": INCOMPLETE,
                "reason": ["this launch has not finished"],
            }
            # Written before the launch, so a run that dies mid-flight still says what it started
            # as. The session is what the transcript is found under afterwards.
            write_run_file(run_dir, record)
            print(f"[cell] {len(positions)} tasks, {schedule}, {model}/{effort} -> {run_dir}")
            with (run_dir / TRANSCRIPT).open("wb") as out, (run_dir / STDERR).open("wb") as err:
                finished = subprocess.run(started.argv, stdout=out, stderr=err, check=False)
            returncode = finished.returncode
        except Stopped as signalled:
            stopped = signalled
        finally:
            if domains is not None:
                cleanup = close_domains(domains, log=run_dir / SERVER_LOG)
    # The tool surface is the half of the launch that only exists once the agent has started, so
    # it is read out of the transcript's first line and compared to the recorded one here.
    init = pinned.init_event(run_dir / TRANSCRIPT)
    drift = pinned.surface_drift(init, served=SERVED_TOOLS)
    if stopped is not None:
        reasons = [str(stopped)]
    elif returncode:
        reasons = [f"the agent exited {returncode}"]
    else:
        reasons = unserved(run_dir)
    reasons += cleanup
    record["exit_code"] = returncode
    record["init"] = init
    record["drift"] = drift
    record["status"] = INCOMPLETE if reasons else COMPLETE
    record["reason"] = reasons
    write_run_file(run_dir, record)
    outcome = (
        f"agent exited {returncode}"
        if returncode is not None
        else "the agent's container came down with the launcher"
    )
    print(f"[cell] {outcome}; transcript {run_dir / TRANSCRIPT}")
    for line in pinned.drift_report(drift):
        print(line)
    for reason in reasons:
        print(f"[cell] {INCOMPLETE}: {reason}")
    if stopped is not None:
        return 128 + stopped.number
    return returncode or (1 if reasons else 0)


def probe(
    run_dir: Path,
    *,
    tasks: str,
    domain: str,
    schedule: str,
    cache: Optional[Path] = None,
    build: bool = False,
) -> int:
    """Stand the two domains up, ask the boundary the questions an agent would, and report.

    The container the questions are asked from is the agent's: the same image, the same name, the
    same three mounts, the same network, the same working directory, and the environment a launch
    builds rather than one written out again here. So what it can reach is what the agent could
    have reached, and a claim about the boundary is a measurement of it rather than a description
    of the code that built it.

    A real generation is served while this runs, with a real roster written and a real history
    open, because the interesting question is not whether an empty directory is unreachable.

    The network is measured rather than described. Which namespace the agent's container is in is
    compared with the server's, and every address the server keeps on its own loopback is asked
    for an answer on the agent's: a container started to share the server's network stack passes
    every other check here, and the history it could then reach is on exactly those addresses.

    What it establishes is that this run's roster, history and grades are on the far side of the
    boundary. It is not a claim of isolation: the agent keeps general egress, as the cell this one
    reruns did, so what it can reach out there is reported as the retained egress it is.
    """
    cache = sandbox.default_cache() if cache is None else cache
    pinned.empty_workdir(run_dir / SELF)
    (run_dir / HOME).mkdir(parents=True, exist_ok=True)
    sandbox.build_images(
        agent=f"{sandbox.AGENT_IMAGE}:{pinned.CLI_VERSION}",
        server=f"{sandbox.SERVER_IMAGE}:latest",
        rebuild=build,
    )
    domains: Optional[Domains] = None
    stopped: Optional[Stopped] = None
    failed = 0
    with stopping():
        try:
            domains = open_domains(
                run_dir, tasks=tasks, domain=domain, schedule=schedule, cache=cache
            )
            mcp_config(run_dir, url=domains.url)
            environment = pinned.agent_environment(os.environ)
            # The server's half of the network claim, read from the server's own side because
            # that is where it is a fact rather than a guess, and read before the agent starts
            # because the agent is asked about it: the endpoint is the only listener bound to an
            # address another container could route to, the durable service holding the history is
            # on this container's own loopback, and which namespace that loopback belongs to is
            # what decides whether it is out of reach. A read that fails is a check that failed,
            # because a listener nobody could enumerate is not one anybody has shown to be absent.
            listening: Optional[List[str]] = None
            namespace = ""
            try:
                listening = sandbox.listening_sockets(domains.server)
                namespace = sandbox.network_namespace(domains.server)
            except ValueError as unreadable:
                print(f"FAIL  {unreadable}")
                failed += 1
            started = agent_command(
                domains,
                run_dir,
                command=sandbox.probe_command(
                    run_dir=run_dir,
                    cache=cache,
                    server=domains.server,
                    environment=sorted(environment),
                    server_namespace=namespace,
                    server_loopback=sandbox.loopback_listeners(listening or []),
                ),
                environment=environment,
                credential=pinned.credential_name(os.environ),
            )
            done = subprocess.run(started.argv, capture_output=True, text=True, check=False)
            print(done.stdout, end="")
            if done.stderr:
                print(done.stderr, end="", file=sys.stderr)
            failed += sandbox.read_probe(done.stdout)
            if listening is not None:
                print(
                    f"\nthe server's container is listening on "
                    f"{', '.join(listening) or 'nothing'}"
                )
                for socket in sandbox.unexpected_listeners(listening):
                    print(f"FAIL  {socket} is reachable and is not the gateway")
                    failed += 1
        except Stopped as signalled:
            stopped = signalled
        finally:
            if domains is not None:
                for reason in close_domains(domains, log=run_dir / SERVER_LOG):
                    print(f"FAIL  {reason}")
                    failed += 1
    if stopped is not None:
        print(f"\n{stopped}")
        return 128 + stopped.number
    print(
        f"{failed} checks failed"
        if failed
        else "this run's roster, history and grades are not reachable from the agent's container"
    )
    return 1 if failed else 0


async def table(run_dir: Path) -> int:
    """Print what the run scored: the history's own rows, then the join to the roster.

    The first table is exactly what ``shogym results`` prints over the grades directory, and it
    writes the same derived file there. The second is this cell's join, and it exists because
    neither half can answer alone: the history is keyed by attempt because nothing on the wire
    names a task, and the transcript is the only place a tool call exists.

    The read ends by reconciling the two records of what the model was shown, and it returns
    nonzero where they differ. What a cell is for is saying what the agent did with what it was
    told, so an episode whose own transcript does not hold the bytes the generation delivered is
    one whose analysis would be about feedback that may never have arrived: that is a read that
    failed rather than a read with a note under it.
    """
    from shogym.serve.protocol_v2.reader import (
        NothingToRead,
        ReadRefused,
        format_records,
        read_records,
        write_records,
    )
    from shogym.serve.protocol_v2.rundir import ResumeRefused

    grades = run_dir / GRADES
    try:
        run = await read_records(grades)
    except NothingToRead as empty:
        print(f"nothing to read: {empty}")
        return 0
    except (ResumeRefused, ReadRefused) as refused:
        print(f"cannot read {grades}: {refused}", file=sys.stderr)
        return 1
    print(format_records(run.records))
    print(f"wrote {write_records(run)}\n")

    roster_file = grades / ROSTER_FILE
    if not roster_file.is_file():
        print(f"no {ROSTER_FILE} in {grades}, so there is nothing to join these rows to")
        return 0
    path = run_dir / TRANSCRIPT
    transcript = read_back.read_transcript(path) if path.is_file() else None
    if transcript is None:
        print(read_back.format_table(read_back.rows(read_back.read_roster(grades), run.records)))
        print(f"\nno {TRANSCRIPT} here, so nothing says what the model was shown")
        return 0
    checked = read_back.reconcile(run.presentations, transcript)
    rows = read_back.rows(read_back.read_roster(grades), run.records, transcript, checked)
    print(read_back.format_table(rows))
    print(
        f"\n{transcript.pulls} pulls answered with {transcript.tasks} tasks, "
        f"{transcript.unserved} calls to the agent's own tools"
    )
    if transcript.refusals:
        # A refusal is model-visible text that advances no protocol state, so this is the whole
        # record of one and it is reported rather than counted into anything.
        print(f"{len(transcript.refusals)} refusals: {', '.join(sorted(set(transcript.refusals)))}")
    # The history and the transcript are two records of one delivery, and a run where they
    # disagree is a run whose analysis cannot say the agent read what was sent to it. Whether the
    # launch calls this run finished decides what a check nobody could make comes to.
    refused = read_back.read_refusals(grades)
    certified = certifies(run_dir)
    differences = read_back.disagreements(checked, transcript, refused, certified=certified)
    if refused.absent is not None and not certified:
        # Said out loud rather than passed over: this run was not reconciled against the count,
        # and a read that stayed quiet about it would read as one that had been.
        print(f"the refusal cross-check is unavailable: {refused.absent}")
    if not differences:
        print(f"{len(checked)} presented messages, all of them in this transcript")
        return 0
    print(
        f"\n{len(differences)} differences between this transcript and the {len(checked)} "
        f"messages the generation delivered, over {read_back.unconfirmed(rows)} rows:",
        file=sys.stderr,
    )
    for difference in differences:
        print(f"  {difference}", file=sys.stderr)
    return 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cell")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="serve a roster and run the agent over it")
    run.add_argument(
        "--tasks",
        default=f"{CELL_ONE}:{POOL_CEILING}",
        help=(
            f"the roster: `{CELL_ONE}` is the {POOL_CEILING} tasks the recorded run queued, in "
            f"the order it queued them, `{CELL_ONE}:20` is the first twenty of them, and a "
            f"plain list or range (`0-19`, `4,0,2`) is a roster of your own (default: "
            f"{CELL_ONE}:{POOL_CEILING})"
        ),
    )
    run.add_argument("--domain", default="public", help="automationbench domain (default: public)")
    run.add_argument(
        "--schedule",
        default=CELL_ONE_SCHEDULE,
        help=(
            "what the agent is told: `immediate` is the honest score at every seal, which is "
            f"the regime the recorded run served; `never` tells it nothing (default: "
            f"{CELL_ONE_SCHEDULE})"
        ),
    )
    run.add_argument("--model", default=MODEL, help=f"(default: {MODEL})")
    run.add_argument("--effort", default=EFFORT, help=f"(default: {EFFORT})")
    run.add_argument(
        "--runs", default=str(HERE / "runs"), help="where run directories go (default: ./runs)"
    )
    run.add_argument(
        "--allow-cli-drift",
        action="store_true",
        help=(
            f"run on a Claude Code other than the recorded {pinned.CLI_VERSION}, and have the "
            "difference recorded in run.json rather than refused"
        ),
    )
    run.add_argument(
        "--allow-image-drift",
        action="store_true",
        help=(
            "run on an agent image built from other inputs than the recorded base, package, "
            "registry and recipe, and have the difference recorded rather than refused"
        ),
    )
    check = sub.add_parser("probe", help="ask, from inside the agent's container, what it can reach")
    check.add_argument("--tasks", default="cell-one:2", help="(default: cell-one:2)")
    check.add_argument("--domain", default="public", help="(default: public)")
    check.add_argument(
        "--schedule", default=CELL_ONE_SCHEDULE, help=f"(default: {CELL_ONE_SCHEDULE})"
    )
    check.add_argument(
        "--runs", default=str(HERE / "runs"), help="where the probe's run directory goes"
    )
    for command in (run, check):
        command.add_argument(
            "--cache",
            default=str(sandbox.default_cache()),
            help=(
                "where the benchmark source and the durable service's binary are kept, mounted "
                f"into the server's container and into no other (default: "
                f"{sandbox.default_cache()})"
            ),
        )
        command.add_argument(
            "--build", action="store_true", help="rebuild both images before starting"
        )

    read = sub.add_parser("table", help="read a run directory back")
    read.add_argument("run_dir", help="a directory `run` created")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        run_dir = new_run_dir(Path(args.runs).expanduser(), schedule=args.schedule)
        return launch(
            run_dir,
            tasks=args.tasks,
            domain=args.domain,
            schedule=args.schedule,
            model=args.model,
            effort=args.effort,
            cache=Path(args.cache).expanduser(),
            allow_cli_drift=args.allow_cli_drift,
            allow_image_drift=args.allow_image_drift,
            build=args.build,
        )
    if args.command == "probe":
        run_dir = new_run_dir(Path(args.runs).expanduser(), schedule=args.schedule, prefix="probe")
        return probe(
            run_dir,
            tasks=args.tasks,
            domain=args.domain,
            schedule=args.schedule,
            cache=Path(args.cache).expanduser(),
            build=args.build,
        )
    return asyncio.run(table(Path(args.run_dir).expanduser()))


if __name__ == "__main__":
    sys.exit(main())
